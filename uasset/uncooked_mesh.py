"""Parser for uncooked UE4.27 StaticMesh assets.

Uncooked StaticMesh assets store their geometry inside a compressed
``FMeshDescription`` bulk-data blob referenced from the export serial data.
This module implements the full pipeline:

    export serial data -> FByteBulkData header -> UE4 chunked ZLIB decompress
        -> FMeshDescription parse -> geometry (vertices/UVs/normals/triangles)

All byte layouts are derived directly from the UE4.27 source code in
``./UnrealEngine-4.27.2-release``:

* ``FMeshDescription::Serialize``  (MeshDescription.cpp:34)
* ``TMeshElementArrayBase``        (MeshElementArray.h:25)
* element serializers             (MeshDescription.h:28..261)
* ``FAttributesSetBase``           (MeshAttributeArray.cpp:44)
* ``TMeshAttributeArraySet``       (MeshAttributeArray.h:388)
* ``FUntypedBulkData::Serialize``  (BulkData.cpp:911)
* ``FArchive::SerializeCompressed``(Archive.cpp:684)
"""

import struct
import zlib
from typing import Dict, List, Optional, Tuple

from .reader import BinaryReader
from .properties import skip_properties


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PACKAGE_FILE_TAG = 0x9E2A83C1
LOADING_COMPRESSION_CHUNK_SIZE = 131072  # 128 KiB

# EBulkDataFlags (BulkDataCommon.h)
BULKDATA_PayloadAtEndOfFile = 1 << 0        # 0x01
BULKDATA_SerializeCompressedZLIB = 1 << 1   # 0x02
BULKDATA_Size64Bit = 1 << 13                # 0x2000

# AttributeTypes tuple (MeshAttributeArray.h:19):
#   FVector4, FVector, FVector2D, float, int, bool, FName
# Type index -> (name, default-value byte size).  None size == FString.
ATTR_TYPE_INFO = {
    0: ('FVector4', 16),
    1: ('FVector', 12),
    2: ('FVector2D', 8),
    3: ('float', 4),
    4: ('int32', 4),
    5: ('bool', 4),     # DefaultValue serializes as UBOOL (4-byte int32); NOTE the
                        # bulk-array *elements* are 1 byte (sizeof bool) -- those are
                        # read straight from the SerializedElementSize field instead.
    6: ('FName', None), # serialized as FString (not bulk-serializable)
}


# ---------------------------------------------------------------------------
# Low-level binary readers
# ---------------------------------------------------------------------------

def _u32(data, pos):
    return struct.unpack_from('<I', data, pos)[0], pos + 4


def _i32(data, pos):
    return struct.unpack_from('<i', data, pos)[0], pos + 4


def _i64(data, pos):
    return struct.unpack_from('<q', data, pos)[0], pos + 8


def _fstring(data, pos):
    """Read an FString (int32 length + chars + null terminator).

    UE serializes the FName attribute keys (inside the FMeshDescription bulk
    blob) as FStrings.  Note: the serialized names sometimes carry a trailing
    space (e.g. ``"Position "``); callers should ``.strip()`` them.
    """
    length, pos = _i32(data, pos)
    if length > 0:
        raw = data[pos:pos + length]
        pos += length
        # drop the trailing null terminator
        return raw[:-1].decode('utf-8', errors='replace'), pos
    elif length < 0:
        cc = -length
        raw = data[pos:pos + cc * 2]
        pos += cc * 2
        return raw.decode('utf-16-le', errors='replace'), pos
    return "", pos


def _bit_array(data, pos):
    """TBitArray<> serialization: NumBits(int32) + ceil(NumBits/32)*uint32 words."""
    num_bits, pos = _i32(data, pos)
    if num_bits < 0 or num_bits > 100_000_000:
        raise ValueError(f"TBitArray NumBits out of range: {num_bits}")
    num_words = (num_bits + 31) // 32
    words = list(struct.unpack_from('<%dI' % num_words, data, pos))
    pos += num_words * 4
    return num_bits, words, pos


def _allocated_indices(num_bits, words):
    """Return the list of allocated (set) bit positions, in order."""
    result = []
    for b in range(num_bits):
        wi, bi = b >> 5, b & 31
        if wi < len(words) and (words[wi] >> bi) & 1:
            result.append(b)
    return result


# ---------------------------------------------------------------------------
# UE4 chunked ZLIB decompression (Archive.cpp SerializeCompressed)
# ---------------------------------------------------------------------------

def decompress_ue4_chunked_zlib(data: bytes) -> Optional[bytes]:
    """Decompress UE4's custom chunked ZLIB bulk-data format.

    Layout (each field is a signed 64-bit ``FCompressedChunkInfo``):

    * PackageFileTag:   CompressedSize = PACKAGE_FILE_TAG, UncompressedSize = chunk_size
    * Summary:          total compressed size, total uncompressed size
    * Chunk headers:    N x (CompressedSize, UncompressedSize)
    * Compressed blocks: raw-deflate data per chunk

    Returns the decompressed bytes, or ``None`` on failure.
    """
    if len(data) < 32:
        return None

    pos = 0
    pkg_compressed, pos = _i64(data, pos)
    pkg_uncompressed, pos = _i64(data, pos)

    if pkg_compressed != PACKAGE_FILE_TAG:
        return None

    chunk_size = (LOADING_COMPRESSION_CHUNK_SIZE
                  if pkg_uncompressed == PACKAGE_FILE_TAG else pkg_uncompressed)

    # Summary
    summary_compressed, pos = _i64(data, pos)
    summary_uncompressed, pos = _i64(data, pos)

    # Determine chunk count and read chunk headers
    total_chunks = (summary_uncompressed + chunk_size - 1) // chunk_size
    chunks = []
    for _ in range(total_chunks):
        c_size, pos = _i64(data, pos)
        u_size, pos = _i64(data, pos)
        chunks.append((c_size, u_size))

    # Decompress each chunk with raw deflate (wbits=-15)
    out = bytearray()
    for c_size, u_size in chunks:
        block = data[pos:pos + c_size]
        pos += c_size
        if u_size > 0 and c_size >= u_size:
            # Stored uncompressed
            out.extend(block[:u_size])
            continue
        try:
            out.extend(zlib.decompress(block, -15))
        except zlib.error:
            # Fall back to zlib-with-header
            try:
                out.extend(zlib.decompress(block))
            except zlib.error:
                return None

    return bytes(out)


# ---------------------------------------------------------------------------
# Export serial data -> FByteBulkData extraction
# ---------------------------------------------------------------------------

def _read_fstring_sized(data, pos):
    """Read an FString used by the StaticMesh serial stream."""
    length, pos = _i32(data, pos)
    if length > 0:
        pos += length
    elif length < 0:
        pos += (-length) * 2
    return pos


def extract_mesh_description_bulk(pkg) -> Optional[bytes]:
    """Locate and decompress the LOD0 FMeshDescription from a StaticMesh export.

    Follows ``UStaticMesh::Serialize`` (StaticMesh.cpp:4682) to walk the export
    serial data, then ``FMeshDescriptionBulkData::Serialize``
    (MeshDescription.cpp:1277) to read the ``FByteBulkData`` header, and finally
    decompresses the UE4 chunked-ZLIB payload.

    Returns the decompressed ``FMeshDescription`` bytes, or ``None``.
    """
    # Find the StaticMesh export
    sm_export_idx = None
    for i in range(pkg.export_count):
        if pkg.get_export_class_name(i) == 'StaticMesh':
            sm_export_idx = i
            break
    if sm_export_idx is None:
        return None

    reader = pkg.get_export_data(sm_export_idx)
    if reader is None:
        return None
    data = reader.data

    # 1. Skip UObject properties (Super::Serialize) -- UE4.27 uses file_version_ue5=0
    reader.seek(0)
    try:
        skip_properties(reader, pkg.name_map, 0)
    except Exception:
        return None
    pos = reader.position()

    # 2. FStripDataFlags (GlobalStripFlags + ClassStripFlags), 2 bytes
    if pos + 2 > len(data):
        return None
    pos += 2

    # 3. bCooked (int32 bool)
    b_cooked, pos = _i32(data, pos)
    # Uncooked asset expected; cooked assets take a different path.

    # 4. BodySetup (FPackageIndex int32) + 5. NavCollision (FPackageIndex int32)
    pos += 8

    # 6. Deprecated_HighResSourceMeshName (FString)
    pos = _read_fstring_sized(data, pos)
    # 7. Deprecated_HighResSourceMeshCRC (uint32)
    pos += 4

    # 8. LightingGuid (FGuid, 16 bytes)
    pos += 16

    # 9. Sockets (TArray<FPackageIndex>: int32 count + count*int32)
    num_sockets, pos = _i32(data, pos)
    if num_sockets < 0 or num_sockets > 4096:
        return None
    pos += num_sockets * 4

    # 10. SourceModels: for each LOD, [bIsValid(int32)] + if valid
    #     [FByteBulkData(20) + Guid(16) + bGuidIsHash].  Walk all LODs and
    #     return the first one whose bulk data is valid.
    for _lod in range(8):  # safety cap
        if pos + 4 > len(data):
            break
        b_is_valid, pos = _i32(data, pos)
        if not b_is_valid:
            continue

        # FByteBulkData header (BulkDataCommon.h): when Size64Bit is clear the
        # layout is BulkDataFlags(u32) + ElementCount(i32) + SizeOnDisk(i32) +
        # OffsetInFile(i64) == 20 bytes.
        if pos + 4 > len(data):
            return None
        bulk_flags, pos2 = _u32(data, pos)
        size_64bit = bool(bulk_flags & BULKDATA_Size64Bit)

        if size_64bit:
            element_count, pos2 = _i32(data, pos2)      # treat as i32 then skip
            size_on_disk, pos2 = _i64(data, pos2)       # 64-bit
            offset_in_file, pos2 = _i64(data, pos2)
            header_size = 4 + 8 + 8
        else:
            element_count, pos2 = _i32(data, pos2)
            size_on_disk, pos2 = _i32(data, pos2)
            offset_in_file, pos2 = _i64(data, pos2)
            header_size = 4 + 4 + 4 + 8

        if size_on_disk <= 0 or size_on_disk > len(pkg.reader.data):
            # Not a usable LOD payload; skip past bulk header + Guid + bGuidIsHash
            pos = pos + header_size + 16 + 1
            continue

        # Compute the absolute file offset of the payload
        if bulk_flags & BULKDATA_PayloadAtEndOfFile:
            actual_offset = offset_in_file + pkg.bulk_data_start_offset
        else:
            actual_offset = offset_in_file

        file_data = pkg.reader.data
        if actual_offset < 0 or actual_offset + size_on_disk > len(file_data):
            return None
        payload = file_data[actual_offset:actual_offset + size_on_disk]

        if bulk_flags & BULKDATA_SerializeCompressedZLIB:
            decompressed = decompress_ue4_chunked_zlib(payload)
            if decompressed is None:
                return None
            return decompressed
        return payload

    return None


# ---------------------------------------------------------------------------
# FMeshDescription parser
# ---------------------------------------------------------------------------

def _parse_element_array(data, pos, elem_reader):
    """Parse a TMeshElementArray: TBitArray(allocated mask) + element data
    written once per allocated element.

    Returns ``(num_bits, allocated_ids, elements, pos)`` where *elements* is a
    dense list parallel to *allocated_ids* (the sparse element IDs).
    """
    num_bits, words, pos = _bit_array(data, pos)
    allocated = _allocated_indices(num_bits, words)
    elements = []
    for _ in allocated:
        elem, pos = elem_reader(data, pos)
        elements.append(elem)
    return num_bits, allocated, elements, pos


def _read_vertex(_d, _p):
    # FMeshVertex serializes nothing in the new format (MeshDescription.h:48)
    return None, _p


def _read_vertex_instance(data, pos):
    # FMeshVertexInstance: VertexID (int32)
    vid, pos = _i32(data, pos)
    return vid, pos


def _read_edge(data, pos):
    # FMeshEdge: VertexID[0], VertexID[1]
    v0, pos = _i32(data, pos)
    v1, pos = _i32(data, pos)
    return (v0, v1), pos


def _read_polygon(data, pos):
    # FMeshPolygon: TArray<FVertexInstanceID> + FPolygonGroupID
    # (new format: triangle lists are rebuilt at load, not serialized)
    cnt, pos = _i32(data, pos)
    if cnt < 0 or cnt > 1024:
        raise ValueError(f"polygon vertex-instance count out of range: {cnt}")
    vis = list(struct.unpack_from('<%di' % cnt, data, pos)) if cnt else []
    pos += cnt * 4
    pgid, pos = _i32(data, pos)
    return {'vis': vis, 'polygon_group_id': pgid}, pos


def _read_polygon_group(_d, _p):
    # FMeshPolygonGroup serializes nothing in the new format (MeshDescription.h:254)
    return None, _p


def _read_triangle(data, pos):
    # FMeshTriangle: VertexInstanceID[0..2] + PolygonID (new format)
    vi0, pos = _i32(data, pos)
    vi1, pos = _i32(data, pos)
    vi2, pos = _i32(data, pos)
    polygon_id, pos = _i32(data, pos)
    return {'vi': (vi0, vi1, vi2), 'polygon_id': polygon_id}, pos


def _read_default_value(data, pos, attr_type):
    """Read a default value of the given attribute type (for the DefaultValue
    field of TMeshAttributeArraySet)."""
    info = ATTR_TYPE_INFO.get(attr_type)
    if info is None:
        raise ValueError(f"Unknown attribute type: {attr_type}")
    if attr_type == 6:  # FName -> FString
        return _fstring(data, pos)
    size = info[1]
    val = data[pos:pos + size]
    pos += size
    return val, pos


def _parse_attribute_array(data, pos, attr_type):
    """Parse one TMeshAttributeArrayBase element (a single attribute index).

    Bulk-serializable types use ``TArray::BulkSerialize``:
        SerializedElementSize(int32) + ArrayNum(int32) + raw data.
    FName uses regular TArray serialization:
        count(int32) + count * FString.
    """
    if attr_type == 6:  # FName, element-by-element
        count, pos = _i32(data, pos)
        if count < 0 or count > 100_000_000:
            raise ValueError(f"FName attribute count out of range: {count}")
        strings = []
        for _ in range(count):
            s, pos = _fstring(data, pos)
            strings.append(s)
        return {'kind': 'fname', 'count': count, 'data': strings}, pos

    # bulk serializable
    elem_size, pos = _i32(data, pos)
    arr_num, pos = _i32(data, pos)
    if arr_num < 0 or arr_num > 100_000_000:
        raise ValueError(f"attribute array count out of range: {arr_num}")
    raw = data[pos:pos + arr_num * elem_size]
    pos += arr_num * elem_size
    return {'kind': 'bulk', 'elem_size': elem_size,
            'count': arr_num, 'data': raw}, pos


def _parse_attribute_set_entry(data, pos):
    """Parse a single FAttributesSetEntry -> TMeshAttributeArraySet<T>.

    Layout:
        AttributeType (uint32)
        NumElements (int32)               # TMeshAttributeArraySet::NumElements
        ArrayForIndices: TArray count (int32) + count * TMeshAttributeArrayBase
        DefaultValue (T)
        Flags (uint32)
    """
    attr_type, pos = _u32(data, pos)
    num_elements, pos = _i32(data, pos)
    num_indices, pos = _i32(data, pos)
    if num_indices < 0 or num_indices > 1024:
        raise ValueError(f"num_indices out of range: {num_indices}")
    arrays = []
    for _ in range(num_indices):
        arr, pos = _parse_attribute_array(data, pos, attr_type)
        arrays.append(arr)
    default, pos = _read_default_value(data, pos, attr_type)
    flags, pos = _u32(data, pos)
    return {
        'type': attr_type,
        'type_name': ATTR_TYPE_INFO.get(attr_type, ('?',))[0],
        'num_elements': num_elements,
        'num_indices': num_indices,
        'arrays': arrays,
        'default': default,
        'flags': flags,
    }, pos


def _parse_attributes_set(data, pos):
    """Parse FAttributesSetBase: NumElements(int32) + Map<FName, Entry>.

    The TMap serializes as: count(int32) + count * (key FString, entry).
    """
    num_elements, pos = _i32(data, pos)
    map_count, pos = _i32(data, pos)
    if map_count < 0 or map_count > 4096:
        raise ValueError(f"attribute map count out of range: {map_count}")
    attributes: Dict[str, dict] = {}
    order: List[str] = []
    for _ in range(map_count):
        key, pos = _fstring(data, pos)
        entry, pos = _parse_attribute_set_entry(data, pos)
        name = key.strip()
        attributes[name] = entry
        order.append(name)
    return {'num_elements': num_elements, 'attributes': attributes,
            'order': order}, pos


def parse_mesh_description(data: bytes) -> Optional[dict]:
    """Parse a decompressed FMeshDescription blob.

    Follows ``FMeshDescription::Serialize`` (MeshDescription.cpp:34):

        VertexArray, VertexInstanceArray, EdgeArray, PolygonArray,
        PolygonGroupArray, VertexAttributesSet, VertexInstanceAttributesSet,
        EdgeAttributesSet, PolygonAttributesSet, PolygonGroupAttributesSet,
        TriangleArray, TriangleAttributesSet
    """
    pos = 0
    try:
        # 5 element arrays
        v_num_bits, v_ids, _vertices, pos = _parse_element_array(data, pos, _read_vertex)
        vi_num_bits, vi_ids, vi_elems, pos = _parse_element_array(data, pos, _read_vertex_instance)
        e_num_bits, e_ids, _edges, pos = _parse_element_array(data, pos, _read_edge)
        p_num_bits, p_ids, p_elems, pos = _parse_element_array(data, pos, _read_polygon)
        pg_num_bits, pg_ids, _pgs, pos = _parse_element_array(data, pos, _read_polygon_group)

        # 5 attribute sets
        vertex_attrs, pos = _parse_attributes_set(data, pos)
        vertex_instance_attrs, pos = _parse_attributes_set(data, pos)
        edge_attrs, pos = _parse_attributes_set(data, pos)
        polygon_attrs, pos = _parse_attributes_set(data, pos)
        polygon_group_attrs, pos = _parse_attributes_set(data, pos)

        # TriangleArray (present since MeshDescriptionTriangles version; UE4.27)
        triangles = []
        if pos + 4 <= len(data):
            t_num_bits, t_ids, tri_elems, pos = _parse_element_array(data, pos, _read_triangle)
            triangles = tri_elems

        return {
            'vertex_ids': v_ids,
            'vertex_instance_ids': vi_ids,
            'vertex_instance_elements': vi_elems,   # vertex-instance-id -> VertexID
            'polygon_ids': p_ids,
            'polygon_elements': p_elems,            # polygon-id -> {vis, polygon_group_id}
            'polygon_group_ids': pg_ids,
            'vertex_attributes': vertex_attrs,
            'vertex_instance_attributes': vertex_instance_attrs,
            'polygon_group_attributes': polygon_group_attrs,
            'triangles': triangles,
            'end_pos': pos,
        }
    except (struct.error, ValueError, IndexError) as exc:
        import traceback
        traceback.print_exc()
        print(f"[uncooked] FMeshDescription parse failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Geometry extraction
# ---------------------------------------------------------------------------

def _bulk_floats(entry: dict, per_elem: int) -> List[Tuple]:
    """Unpack a bulk attribute entry's data into a list of float tuples."""
    arr = entry['arrays'][0]
    raw = arr['data']
    count = arr['count']
    fmt = '<%df' % (count * per_elem)
    flat = struct.unpack(fmt, raw[:count * per_elem * 4])
    return [tuple(flat[i * per_elem:(i + 1) * per_elem]) for i in range(count)]


def _build_sparse_lookup(num_bits, allocated_ids, elements, default=None):
    """Map sparse element ID -> element value (expanding dense->sparse)."""
    lookup = {}
    for sparse_id, elem in zip(allocated_ids, elements):
        lookup[sparse_id] = elem
    return lookup


def extract_geometry(desc: dict) -> Optional[dict]:
    """Extract renderable geometry from a parsed FMeshDescription.

    Returns a dict with: ``vertices``, ``vi_to_vertex``, ``normals``, ``uvs``
    (list of UV channels), ``triangles`` (vi0, vi1, vi2, material_index) and
    ``material_slot_names``.
    """
    if desc is None:
        return None

    v_attrs = desc['vertex_attributes']['attributes']
    vi_attrs = desc['vertex_instance_attributes']['attributes']
    pg_attrs = desc['polygon_group_attributes'].get('attributes', {})

    # --- Vertex positions (indexed by vertex ID) ---
    positions = []
    pos_entry = v_attrs.get('Position')
    if pos_entry is None or pos_entry['type'] != 1 or not pos_entry['arrays']:
        return None
    positions = _bulk_floats(pos_entry, 3)

    # --- Vertex-instance -> vertex-ID mapping (a list indexed by VI ID) ---
    vi_ids = desc['vertex_instance_ids']
    vi_elems = desc['vertex_instance_elements']
    max_vi = (max(vi_ids) + 1) if vi_ids else 0
    vi_to_vertex: List[int] = [0] * max_vi
    for sparse_id, v in zip(vi_ids, vi_elems):
        vi_to_vertex[sparse_id] = v

    # --- Normals (per vertex instance, FVector) ---
    normals = []
    normal_entry = vi_attrs.get('Normal')
    if normal_entry and normal_entry['type'] == 1 and normal_entry['arrays']:
        normals = _bulk_floats(normal_entry, 3)

    # --- UV channels (per vertex instance, FVector2D) ---
    uv_channels = []
    uv_entry = vi_attrs.get('TextureCoordinate')
    if uv_entry and uv_entry['type'] == 2 and uv_entry['arrays']:
        # TextureCoordinate may have multiple indices (UV0, UV1, ...).
        for idx_arr in uv_entry['arrays']:
            raw = idx_arr['data']
            count = idx_arr['count']
            flat = struct.unpack_from('<%df' % (count * 2), raw[:count * 8])
            uv_channels.append([(flat[i * 2], flat[i * 2 + 1]) for i in range(count)])

    # --- Polygons: sparse polygon-ID -> polygon group ID ---
    p_ids = desc['polygon_ids']
    p_elems = desc['polygon_elements']
    poly_group = _build_sparse_lookup(max(p_ids) + 1 if p_ids else 0, p_ids,
                                      [e['polygon_group_id'] for e in p_elems],
                                      default=0)
    poly_group_lookup = {}
    for pid, elem in zip(p_ids, p_elems):
        poly_group_lookup[pid] = elem['polygon_group_id']

    # --- Triangles: (vi0, vi1, vi2, material_index) ---
    # Two sources depending on the FEditorObjectVersion::MeshDescriptionTriangles
    # custom version.  Recent assets serialize a dedicated TriangleArray; older
    # (and these uncooked) assets embed the corner vertex-instances inside each
    # FMeshPolygon and must be triangulated here.
    triangles = []
    if desc['triangles']:
        for tri in desc['triangles']:
            vi0, vi1, vi2 = tri['vi']
            material_index = poly_group_lookup.get(tri['polygon_id'], 0)
            triangles.append((vi0, vi1, vi2, material_index))
    else:
        for _pid, poly in zip(desc['polygon_ids'], desc['polygon_elements']):
            vis = poly['vis']
            if len(vis) < 3:
                continue
            mat = poly['polygon_group_id']
            # Fan triangulation; a triangle polygon (3 vis) yields one triangle.
            for k in range(1, len(vis) - 1):
                triangles.append((vis[0], vis[k], vis[k + 1], mat))

    # --- Material slot names (ImportedMaterialSlotName per polygon group) ---
    material_slot_names: List[Optional[str]] = []
    pg_ids = desc['polygon_group_ids']
    slot_entry = pg_attrs.get('ImportedMaterialSlotName')
    if slot_entry and slot_entry['type'] == 6 and slot_entry['arrays']:
        slot_names_data = slot_entry['arrays'][0]['data']
        # Indexed by polygon-group ID; build a lookup covering all group IDs.
        slot_lookup = {}
        for i, name in enumerate(slot_names_data):
            slot_lookup[i] = name.strip() if name else None
        material_slot_names = [slot_lookup.get(gid) for gid in range(len(pg_ids))]
        # If names are sparse, also try direct index alignment
        if not any(material_slot_names):
            material_slot_names = [slot_names_data[i].strip() if i < len(slot_names_data) else None
                                   for i in range(len(pg_ids))]

    return {
        'vertices': positions,
        'vi_to_vertex': vi_to_vertex,
        'normals': normals,
        'uvs': uv_channels,
        'triangles': triangles,
        'material_slot_names': material_slot_names,
    }


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def parse_uncooked_static_mesh(pkg) -> Optional[dict]:
    """Extract geometry from an uncooked StaticMesh package.

    Returns the geometry dict from :func:`extract_geometry`, or ``None``.
    """
    raw = extract_mesh_description_bulk(pkg)
    if raw is None:
        print("[uncooked] Failed to extract FMeshDescription bulk data")
        return None
    print(f"[uncooked] Decompressed FMeshDescription: {len(raw):,} bytes")
    desc = parse_mesh_description(raw)
    if desc is None:
        return None
    print(f"[uncooked] Parsed: {len(desc['vertex_ids'])} vertices, "
          f"{len(desc['vertex_instance_ids'])} vertex instances, "
          f"{len(desc['triangles'])} triangles")
    return extract_geometry(desc)
