"""Static mesh parser and OBJ/GLB exporter for UE 5.5 .uasset files.

Parses FMeshDescription from FCompressedBuffer payload in the package trailer.
Pipeline: .uasset → Package → Trailer → FCompressedBuffer → Oodle decompress → FMeshDescription → OBJ/GLB
"""
import os
import struct
from typing import List, Tuple, Optional

import numpy as np
import ooz

from .reader import BinaryReader
from .package import Package

try:
    from pygltflib import (
        GLTF2,
        Scene as GLTFScene,
        Node as GLTFNode,
        Mesh as GLTFMesh,
        Primitive,
        Material,
        PbrMetallicRoughness,
        TextureInfo,
        Texture as GLTFTexture,
        Sampler,
        Image as GLTFImage,
        BufferView,
        Accessor,
        Buffer,
    )
    _HAS_PYGLTFLIB = True
except ImportError:
    _HAS_PYGLTFLIB = False


# ---------------------------------------------------------------------------
# FCompressedBuffer decompression
# ---------------------------------------------------------------------------

def _be_uint32(data, offset):
    return struct.unpack_from('>I', data, offset)[0]


def _be_uint64(data, offset):
    return struct.unpack_from('>Q', data, offset)[0]


def decompress_compressed_buffer(data: bytes) -> Optional[bytes]:
    """Decompress an FCompressedBuffer (UE5 big-endian header + Oodle/LZ4 blocks).

    Returns the raw decompressed bytes, or None on failure.
    """
    if len(data) < 64:
        return None

    magic = _be_uint32(data, 0)
    if magic != 0xB7756362:
        return None

    method = data[8]
    block_size_exp = data[11]
    block_count = _be_uint32(data, 12)
    total_raw_size = _be_uint64(data, 16)
    block_size = 1 << block_size_exp if block_size_exp > 0 else 0

    # Parse block sizes (big-endian uint32 array after 64-byte header)
    block_sizes_offset = 64
    block_sizes = []
    for i in range(block_count):
        bs = _be_uint32(data, block_sizes_offset + i * 4)
        block_sizes.append(bs)

    # Calculate raw block sizes
    raw_block_sizes = []
    for i in range(block_count):
        if i < block_count - 1:
            raw_block_sizes.append(block_size)
        else:
            raw_block_sizes.append(total_raw_size - block_size * (block_count - 1))

    # Decompress blocks
    blocks_data_offset = block_sizes_offset + block_count * 4
    decompressed = bytearray()
    current_offset = blocks_data_offset

    for i in range(block_count):
        compressed_block_size = block_sizes[i]
        raw_block_size = raw_block_sizes[i]
        compressed_block = data[current_offset:current_offset + compressed_block_size]

        if compressed_block_size >= raw_block_size:
            # Block stored uncompressed
            decompressed.extend(compressed_block[:raw_block_size])
        elif method == 3:  # Oodle
            decompressed_block = ooz.decompress(compressed_block, raw_block_size)
            decompressed.extend(decompressed_block)
        else:
            return None

        current_offset += compressed_block_size

    if len(decompressed) != total_raw_size:
        return None

    return bytes(decompressed)


# ---------------------------------------------------------------------------
# Package trailer helpers
# ---------------------------------------------------------------------------

def extract_trailer_payload(file_data: bytes) -> Optional[bytes]:
    """Extract the FCompressedBuffer payload from the package trailer.

    Returns the compressed buffer bytes, or None on failure.
    """
    file_size = len(file_data)

    # Trailer footer is the last 20 bytes:
    #   FooterTag: uint64 (8 bytes)
    #   TrailerLength: uint64 (8 bytes)
    #   PackageTag: uint32 (4 bytes)
    if file_size < 20:
        return None

    trailer_length = struct.unpack_from('<Q', file_data, file_size - 12)[0]
    if trailer_length <= 0 or trailer_length > file_size:
        return None

    trailer_start = file_size - trailer_length

    # Trailer header (28 bytes):
    #   HeaderTag: uint64 (8 bytes)
    #   Version: int32 (4 bytes)
    #   HeaderLength: uint32 (4 bytes)
    #   PayloadsDataLength: uint64 (8 bytes)
    #   NumPayloads: int32 (4 bytes)
    if trailer_start + 28 > file_size:
        return None

    header_length = struct.unpack_from('<I', file_data, trailer_start + 12)[0]
    payloads_data_length = struct.unpack_from('<Q', file_data, trailer_start + 16)[0]

    payload_section_start = trailer_start + header_length
    if payload_section_start + payloads_data_length > file_size:
        return None

    return file_data[payload_section_start:payload_section_start + payloads_data_length]


# ---------------------------------------------------------------------------
# FMeshDescription binary parser
# ---------------------------------------------------------------------------

# Attribute type sizes for bulk-serializable types
_ATTR_TYPE_SIZES = {
    0: 16,  # FVector4f
    1: 12,  # FVector3f
    2: 8,   # FVector2f
    3: 4,   # float
    4: 4,   # int32
    5: 4,   # bool (serialized as int32)
}

_ATTR_TYPE_NAMES = {
    0: 'FVector4f',
    1: 'FVector3f',
    2: 'FVector2f',
    3: 'float',
    4: 'int32',
    5: 'bool',
    6: 'FName',
}


class _MeshDescReader:
    """Low-level reader for FMeshDescription binary data."""

    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def read_int32(self):
        val = struct.unpack_from('<i', self.data, self.pos)[0]
        self.pos += 4
        return val

    def read_uint32(self):
        val = struct.unpack_from('<I', self.data, self.pos)[0]
        self.pos += 4
        return val

    def read_float(self):
        val = struct.unpack_from('<f', self.data, self.pos)[0]
        self.pos += 4
        return val

    def read_bytes(self, n):
        val = self.data[self.pos:self.pos + n]
        self.pos += n
        return val

    def read_fstring(self):
        length = self.read_int32()
        if length > 0:
            s = self.data[self.pos:self.pos + length - 1].decode('latin-1', errors='replace')
            self.pos += length
            return s
        elif length < 0:
            char_count = -length
            s = self.data[self.pos:self.pos + char_count * 2].decode('utf-16-le', errors='replace')
            self.pos += char_count * 2
            return s
        return ""

    def read_tbit_array(self):
        num_bits = self.read_int32()
        num_words = (num_bits + 31) // 32
        words = []
        for _ in range(num_words):
            words.append(self.read_uint32())
        return num_bits, words

    @staticmethod
    def count_valid_elements(num_bits, words):
        count = 0
        for word in words:
            count += bin(word).count('1')
        return count

    def read_attribute_default_value(self, attr_type):
        if attr_type == 0:  # FVector4f
            vals = struct.unpack_from('<ffff', self.data, self.pos)
            self.pos += 16
            return vals
        elif attr_type == 1:  # FVector3f
            x, y, z = struct.unpack_from('<fff', self.data, self.pos)
            self.pos += 12
            return (x, y, z)
        elif attr_type == 2:  # FVector2f
            x, y = struct.unpack_from('<ff', self.data, self.pos)
            self.pos += 8
            return (x, y)
        elif attr_type == 3:  # float
            return self.read_float()
        elif attr_type == 4:  # int32
            return self.read_int32()
        elif attr_type == 5:  # bool
            return self.read_int32()
        elif attr_type == 6:  # FName -> FString
            return self.read_fstring()
        else:
            raise ValueError(f"Unknown attribute type: {attr_type}")

    def parse_attribute_array_base(self, attr_type, is_bulk):
        """Parse TMeshAttributeArrayBase: Extent(u32) + Container data."""
        extent = self.read_uint32()

        if is_bulk and attr_type in _ATTR_TYPE_SIZES:
            # BulkSerialize: ElementSize(i32) + Count(i32) + raw data
            serialized_elem_size = self.read_int32()
            count = self.read_int32()
            raw = self.read_bytes(count * serialized_elem_size)
            return {'extent': extent, 'count': count, 'data': raw}
        else:
            # Element-by-element: TArray serialization (Count + elements)
            count = self.read_int32()
            if attr_type == 6:  # FName -> FString
                strings = [self.read_fstring() for _ in range(count)]
                return {'extent': extent, 'count': count, 'data': strings}
            elif attr_type == 5:  # bool as int32
                bools = [self.read_int32() for _ in range(count)]
                return {'extent': extent, 'count': count, 'data': bools}
            else:
                raise ValueError(f"Unexpected non-bulk type {attr_type}")

    def parse_chunk(self, attr_type):
        """Parse FChunk from TAttributeArrayContainer."""
        if attr_type in _ATTR_TYPE_SIZES:
            serialized_elem_size = self.read_int32()
            data_count = self.read_int32()
            raw = self.read_bytes(data_count * serialized_elem_size)
        elif attr_type == 6:
            data_count = self.read_int32()
            raw = [self.read_fstring() for _ in range(data_count)]
        else:
            data_count = self.read_int32()
            raw = self.read_bytes(data_count * 4)

        chunk_num_elements = self.read_int32()
        start_indices = [self.read_int32() for _ in range(chunk_num_elements)]
        counts = [self.read_int32() for _ in range(chunk_num_elements)]
        max_counts = [self.read_int32() for _ in range(chunk_num_elements)]

        return {'data': raw, 'num_elements': chunk_num_elements,
                'start_indices': start_indices, 'counts': counts, 'max_counts': max_counts}

    def parse_unbounded_channel(self, attr_type):
        """Parse TAttributeArrayContainer."""
        num_chunks = self.read_int32()
        chunks = [self.parse_chunk(attr_type) for _ in range(num_chunks)]
        num_elements = self.read_int32()
        default = self.read_attribute_default_value(attr_type)
        return {'chunks': chunks, 'num_elements': num_elements, 'default': default}

    def parse_attribute_set_entry(self):
        """Parse FAttributesSetEntry."""
        attr_type = self.read_uint32()
        extent = self.read_uint32()
        is_bulk = attr_type in _ATTR_TYPE_SIZES

        num_elements = self.read_int32()
        num_channels = self.read_int32()

        if extent > 0:
            # Bounded: TMeshAttributeArraySet
            channels = [self.parse_attribute_array_base(attr_type, is_bulk) for _ in range(num_channels)]
        else:
            # Unbounded: TMeshUnboundedAttributeArraySet
            channels = [self.parse_unbounded_channel(attr_type) for _ in range(num_channels)]

        default = self.read_attribute_default_value(attr_type)
        flags = self.read_uint32()

        return {'type': attr_type, 'type_name': _ATTR_TYPE_NAMES.get(attr_type, f'?{attr_type}'),
                'extent': extent, 'num_elements': num_elements,
                'num_channels': num_channels, 'channels': channels,
                'default': default, 'flags': flags, 'bounded': extent > 0}

    def parse_attributes_set_base(self):
        """Parse FAttributesSetBase."""
        num_elements = self.read_int32()
        map_count = self.read_int32()
        attributes = {}
        for _ in range(map_count):
            key = self.read_fstring()
            entry = self.parse_attribute_set_entry()
            attributes[key] = entry
        return {'num_elements': num_elements, 'attributes': attributes}

    def parse_element_container(self):
        """Parse FMeshElementContainer."""
        num_bits, words = self.read_tbit_array()
        num_holes = self.read_int32()
        valid_count = self.count_valid_elements(num_bits, words)
        attributes = self.parse_attributes_set_base()
        return {'num_bits': num_bits, 'num_holes': num_holes,
                'valid_count': valid_count, 'words': words,
                'attributes': attributes}

    def parse_mesh_description(self):
        """Parse the full FMeshDescription."""
        num_entries = self.read_int32()
        elements = {}
        for i in range(num_entries):
            key = self.read_fstring()
            # FMeshElementChannels: TArray<FMeshElementContainer>
            num_channels = self.read_int32()
            channels = [self.parse_element_container() for _ in range(num_channels)]
            elements[key] = {'channels': channels}
        return elements


# ---------------------------------------------------------------------------
# Mesh data extraction
# ---------------------------------------------------------------------------

def _extract_attr_data(channel, attr_name, expected_type=None):
    """Extract raw attribute data from a channel's attributes."""
    attrs = channel['attributes']
    for name, attr in attrs['attributes'].items():
        if name.strip() == attr_name or name == attr_name:
            if expected_type is not None and attr['type'] != expected_type:
                continue
            if attr['bounded'] and isinstance(attr['channels'][0].get('data'), bytes):
                return attr['channels'][0]
    return None


def _extract_fname_attr(channel, attr_name):
    """Extract FName (string list) attribute data from a channel.

    Handles both bounded (TMeshAttributeArraySet) and unbounded
    (TMeshUnboundedAttributeArraySet) FName attributes.
    """
    attrs = channel['attributes']
    for name, attr in attrs['attributes'].items():
        if name.strip() == attr_name or name == attr_name:
            if attr['type'] != 6:  # FName
                continue
            if not attr['channels']:
                continue
            ch = attr['channels'][0]
            if attr['bounded']:
                data = ch.get('data')
                if isinstance(data, list):
                    return data
            else:
                # Unbounded: collect from chunks
                chunks = ch.get('chunks', [])
                result = []
                for chunk in chunks:
                    chunk_data = chunk.get('data')
                    if isinstance(chunk_data, list):
                        result.extend(chunk_data)
                if result:
                    return result
    return None


def _build_sparse_mapping(num_bits, words):
    """Build sparse-to-dense mapping from TBitArray validity mask.

    Returns dict mapping valid sparse element IDs to sequential dense indices.
    """
    sparse_to_dense = {}
    dense_idx = 0
    for bit_pos in range(num_bits):
        word_idx = bit_pos // 32
        bit_idx = bit_pos % 32
        if word_idx < len(words) and (words[word_idx] & (1 << bit_idx)):
            sparse_to_dense[bit_pos] = dense_idx
            dense_idx += 1
    return sparse_to_dense


def _maybe_expand_sparse(data_list, container_info, default=None):
    """Expand dense data to sparse array if the element container has holes.

    When an element container has holes (num_holes > 0), element IDs are
    not contiguous.  If the attribute data is stored densely (only valid
    entries), this function expands it to a sparse array indexed by the
    actual element ID, so that lookups by sparse ID work correctly.
    """
    num_bits = container_info['num_bits']
    num_holes = container_info['num_holes']
    words = container_info.get('words', [])

    # No holes or no words — data is already contiguous
    if num_holes <= 0 or not words:
        return data_list

    # Data already covers all sparse IDs — no expansion needed
    if len(data_list) >= num_bits:
        return data_list

    # Data is dense — expand to sparse array
    sparse_mapping = _build_sparse_mapping(num_bits, words)
    result = [default] * num_bits
    for sparse_id, dense_idx in sparse_mapping.items():
        if dense_idx < len(data_list):
            result[sparse_id] = data_list[dense_idx]
    return result


def extract_mesh_data(elements: dict) -> Optional[dict]:
    """Extract vertices, normals, UVs, and triangles from parsed FMeshDescription."""
    result = {
        'vertices': [],
        'vi_to_vertex': [],
        'normals': [],
        'uvs': [],       # List of UV channel arrays
        'triangles': [],
    }

    # --- Vertex positions ---
    if 'Vertices' not in elements:
        return None
    vert_ch = elements['Vertices']['channels'][0]
    pos_data = _extract_attr_data(vert_ch, 'Position', expected_type=1)
    if pos_data is None:
        return None
    raw = pos_data['data']
    count = pos_data['count']
    vertices = []
    for i in range(count):
        x, y, z = struct.unpack_from('<fff', raw, i * 12)
        vertices.append((x, y, z))
    result['vertices'] = _maybe_expand_sparse(vertices, vert_ch, default=(0.0, 0.0, 0.0))

    # --- Vertex instance → vertex mapping ---
    if 'VertexInstances' not in elements:
        return None
    vi_ch = elements['VertexInstances']['channels'][0]
    vi_data = _extract_attr_data(vi_ch, 'VertexIndex', expected_type=4)
    if vi_data is None:
        return None
    raw = vi_data['data']
    count = vi_data['count']
    vi_to_vertex = []
    for i in range(count):
        idx = struct.unpack_from('<i', raw, i * 4)[0]
        vi_to_vertex.append(idx)
    result['vi_to_vertex'] = _maybe_expand_sparse(vi_to_vertex, vi_ch, default=0)

    # --- Normals (per vertex instance) ---
    normal_data = _extract_attr_data(vi_ch, 'Normal', expected_type=1)
    if normal_data:
        raw = normal_data['data']
        count = normal_data['count']
        normals = []
        for i in range(count):
            x, y, z = struct.unpack_from('<fff', raw, i * 12)
            normals.append((x, y, z))
        result['normals'] = _maybe_expand_sparse(normals, vi_ch, default=(0.0, 0.0, 1.0))

    # --- UV coordinates (per vertex instance, channel 0 = texture UV) ---
    uv_channels = []
    vi_attrs = vi_ch['attributes']

    for name, attr in vi_attrs['attributes'].items():
        if (name.strip() == 'TextureCoordinate' or name == 'TextureCoordinate') and attr['type'] == 2:
            # Use only channel 0 (primary texture UV).
            # Channel 1 is lightmap UV and is skipped.
            if attr['channels']:
                channel = attr['channels'][0]
                if isinstance(channel.get('data'), bytes):
                    ch_raw = channel['data']
                    ch_count = channel['count']
                    dense_uvs = []
                    for i in range(ch_count):
                        u, v = struct.unpack_from('<ff', ch_raw, i * 8)
                        dense_uvs.append((u, v))
                    uv_channels.append(_maybe_expand_sparse(dense_uvs, vi_ch, default=(0.0, 0.0)))
            break
    result['uvs'] = uv_channels

    # --- Triangle data ---
    if 'Triangles' not in elements:
        return None
    tri_ch = elements['Triangles']['channels'][0]
    tri_attrs = tri_ch['attributes']

    # VertexInstanceIndex (extent=3, int32)
    raw_tri = None
    tri_count = 0
    for name, attr in tri_attrs['attributes'].items():
        if name == 'VertexInstanceIndex' and attr['type'] == 4 and attr['extent'] == 3:
            raw_tri = attr['channels'][0]['data']
            tri_count = attr['channels'][0]['count']
            break
    if raw_tri is None:
        return result

    triangles_raw = []
    for i in range(tri_count // 3):
        v0, v1, v2 = struct.unpack_from('<iii', raw_tri, i * 12)
        triangles_raw.append((v0, v1, v2))

    # Material assignment — UE5 uses PolygonGroupIndex (one per triangle)
    # which maps directly to the material slot index.  Fall back to the
    # older MaterialIndex attribute if it exists.
    material_indices: List[int] = []
    mat_attr_name = None
    for name, attr in tri_attrs['attributes'].items():
        n = name.strip()
        if n == 'PolygonGroupIndex' and attr['type'] == 4:
            mat_attr_name = n
            break
        if n == 'MaterialIndex' and attr['type'] == 4:
            mat_attr_name = n
            # keep looking — prefer PolygonGroupIndex if it exists
    if mat_attr_name:
        for name, attr in tri_attrs['attributes'].items():
            n = name.strip()
            if n == mat_attr_name and attr['type'] == 4:
                ch_list = attr.get('channels', [])
                if ch_list:
                    ch = ch_list[0]
                    if isinstance(ch.get('data'), bytes):
                        raw_mi = ch['data']
                        count_mi = ch['count']
                        for i in range(count_mi):
                            mi = struct.unpack_from('<i', raw_mi, i * 4)[0]
                            material_indices.append(mi)
                break

    # Build triangles with material indices: (vi0, vi1, vi2, material_index)
    result['triangles'] = [
        (t[0], t[1], t[2], material_indices[i] if i < len(material_indices) else 0)
        for i, t in enumerate(triangles_raw)
    ]

    # --- Material slot names from PolygonGroups ---
    # ImportedMaterialSlotName maps each polygon group to a material slot
    # name.  This is needed because polygon group N does NOT necessarily
    # correspond to material import slot N — the slot names must be matched
    # against the ordered material import names to build the correct mapping.
    material_slot_names = None
    if 'PolygonGroups' in elements:
        pg_ch = elements['PolygonGroups']['channels'][0]
        slot_names = _extract_fname_attr(pg_ch, 'ImportedMaterialSlotName')
        if slot_names:
            slot_names = _maybe_expand_sparse(slot_names, pg_ch, default=None)
            material_slot_names = slot_names
    result['material_slot_names'] = material_slot_names

    return result


# ---------------------------------------------------------------------------
# StaticMesh class
# ---------------------------------------------------------------------------

class StaticMesh:
    def __init__(self):
        self.vertices: List[Tuple[float, float, float]] = []
        self.normals: List[Tuple[float, float, float]] = []
        self.uvs: List[List[Tuple[float, float]]] = []  # list of UV channels
        self.triangles: List[Tuple[int, int, int, int]] = []  # (vi0, vi1, vi2, material_index)
        self.vi_to_vertex: List[int] = []
        self.material_slot_names: Optional[List[Optional[str]]] = None

    @classmethod
    def from_package(cls, pkg: Package) -> Optional['StaticMesh']:
        """Parse static mesh from a Package using the FMeshDescription pipeline."""
        mesh = cls()

        # Read raw file data
        file_data = pkg.reader.data

        # Extract FCompressedBuffer from package trailer
        compressed_buffer = extract_trailer_payload(file_data)
        if compressed_buffer is None:
            return None

        # Decompress
        raw_data = decompress_compressed_buffer(compressed_buffer)
        if raw_data is None:
            return None

        # Parse FMeshDescription
        reader = _MeshDescReader(raw_data)
        try:
            elements = reader.parse_mesh_description()
        except Exception:
            return None

        # Extract mesh data
        mesh_data = extract_mesh_data(elements)
        if mesh_data is None:
            return None

        mesh.vertices = mesh_data['vertices']
        mesh.vi_to_vertex = mesh_data['vi_to_vertex']
        mesh.normals = mesh_data['normals']
        mesh.uvs = mesh_data['uvs']
        mesh.triangles = mesh_data['triangles']
        mesh.material_slot_names = mesh_data.get('material_slot_names')

        return mesh


# ---------------------------------------------------------------------------
# GLB export
# ---------------------------------------------------------------------------

def export_glb(mesh: StaticMesh, filepath: str,
               textures: Optional[List[Tuple[int, object]]] = None):
    """Export a StaticMesh as GLB (binary glTF 2.0) with embedded textures.

    Args:
        mesh: StaticMesh object with geometry data.
        filepath: Output ``.glb`` file path.
        textures: Optional list of ``(material_index, PIL.Image or numpy.ndarray)``
            tuples.  Each texture is embedded as PNG inside the GLB and assigned
            to the corresponding material slot.  If *None* or empty, a default
            grey material is used for every primitive.
    """
    if not _HAS_PYGLTFLIB:
        raise ImportError(
            "pygltflib is required for GLB export.  "
            "Install with: pip install pygltflib"
        )

    from io import BytesIO
    from PIL import Image as PILImage

    dirpath = os.path.dirname(filepath)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    # Primary UV channel only
    uvs = mesh.uvs[0] if mesh.uvs else []
    has_normals = bool(mesh.normals)
    has_uvs = bool(uvs)

    # ------------------------------------------------------------------
    # Group triangles by material index
    # ------------------------------------------------------------------
    material_groups: dict = {}  # material_index -> [(vi0, vi1, vi2), ...]
    for tri in mesh.triangles:
        vi0, vi1, vi2 = tri[0], tri[1], tri[2]
        mat_idx = tri[3] if len(tri) >= 4 else 0
        material_groups.setdefault(mat_idx, []).append((vi0, vi1, vi2))

    if not material_groups:
        return

    # ------------------------------------------------------------------
    # Build texture lookup  material_index -> PIL.Image
    # ------------------------------------------------------------------
    texture_lookup: dict = {}
    if textures:
        for item in textures:
            mat_idx, tex_data = item[0], item[1]
            if isinstance(tex_data, np.ndarray):
                if tex_data.ndim == 3 and tex_data.shape[2] == 4:
                    texture_lookup[mat_idx] = PILImage.fromarray(tex_data, 'RGBA')
                else:
                    texture_lookup[mat_idx] = PILImage.fromarray(tex_data)
            elif hasattr(tex_data, 'save'):  # PIL.Image already
                texture_lookup[mat_idx] = tex_data

    # ------------------------------------------------------------------
    # Binary buffer assembly
    # ------------------------------------------------------------------
    binary = bytearray()
    buffer_views: list = []
    accessors: list = []
    gltf_images: list = []
    gltf_textures: list = []
    gltf_samplers: list = []
    gltf_materials: list = []
    gltf_primitives: list = []

    def _pad4():
        """Pad *binary* to 4-byte alignment."""
        rem = len(binary) % 4
        if rem:
            binary.extend(b'\x00' * (4 - rem))

    def _add_buffer_view(data: bytes, target=None) -> int:
        _pad4()
        offset = len(binary)
        binary.extend(data)
        bv = BufferView()
        bv.buffer = 0
        bv.byteOffset = offset
        bv.byteLength = len(data)
        if target is not None:
            bv.target = target
        buffer_views.append(bv)
        return len(buffer_views) - 1

    def _add_accessor(bv_idx: int, component_type: int, count: int,
                      acc_type: str, min_vals=None, max_vals=None) -> int:
        acc = Accessor()
        acc.bufferView = bv_idx
        acc.byteOffset = 0
        acc.componentType = component_type
        acc.count = count
        acc.type = acc_type
        if min_vals is not None:
            acc.min = list(min_vals)
        if max_vals is not None:
            acc.max = list(max_vals)
        accessors.append(acc)
        return len(accessors) - 1

    # glTF constants
    ARRAY_BUFFER = 34962
    ELEMENT_ARRAY_BUFFER = 34963
    COMP_FLOAT = 5126
    COMP_UNSIGNED_SHORT = 5123
    COMP_UNSIGNED_INT = 5125

    # ------------------------------------------------------------------
    # Materials & textures
    # ------------------------------------------------------------------
    sorted_mat_indices = sorted(material_groups.keys())
    mat_idx_to_gltf_mat: dict = {}

    has_any_texture = any(mi in texture_lookup for mi in sorted_mat_indices)
    if has_any_texture:
        sampler = Sampler()
        sampler.magFilter = 9729   # LINEAR
        sampler.minFilter = 9987   # LINEAR_MIPMAP_LINEAR
        sampler.wrapS = 10497      # REPEAT
        sampler.wrapT = 10497      # REPEAT
        gltf_samplers.append(sampler)

    for gltf_mat_idx, mat_idx in enumerate(sorted_mat_indices):
        mat = Material()
        mat.pbrMetallicRoughness = PbrMetallicRoughness()
        mat.pbrMetallicRoughness.baseColorFactor = [1.0, 1.0, 1.0, 1.0]
        mat.pbrMetallicRoughness.metallicFactor = 0.0
        mat.pbrMetallicRoughness.roughnessFactor = 1.0

        if mat_idx in texture_lookup:
            pil_img = texture_lookup[mat_idx]
            buf = BytesIO()
            pil_img.save(buf, format='PNG')
            png_bytes = buf.getvalue()

            bv_idx = _add_buffer_view(png_bytes)

            img = GLTFImage()
            img.bufferView = bv_idx
            img.mimeType = 'image/png'
            gltf_images.append(img)

            tex = GLTFTexture()
            tex.source = len(gltf_images) - 1
            tex.sampler = 0
            gltf_textures.append(tex)

            tex_info = TextureInfo()
            tex_info.index = len(gltf_textures) - 1
            tex_info.texCoord = 0
            mat.pbrMetallicRoughness.baseColorTexture = tex_info

        gltf_materials.append(mat)
        mat_idx_to_gltf_mat[mat_idx] = gltf_mat_idx

    # ------------------------------------------------------------------
    # Geometry — one primitive per material group
    # ------------------------------------------------------------------
    # UE left-handed (X fwd, Y right, Z up) → glTF right-handed
    # (X right, Y up, -Z fwd).  The mapping  glTF = (-UE_X, UE_Z, -UE_Y)
    # has determinant -1 so it flips handedness (and therefore face
    # winding CW→CCW) without needing a negative node scale.
    # ------------------------------------------------------------------
    for mat_idx in sorted_mat_indices:
        tris = material_groups[mat_idx]

        # Collect unique vertex instances for this primitive
        vi_to_local: dict = {}
        local_verts: list = []  # (px, py, pz, nx, ny, nz, u, v)
        indices: list = []

        for vi0, vi1, vi2 in tris:
            for vi in (vi0, vi1, vi2):
                if vi not in vi_to_local:
                    # Position — resolve through vi_to_vertex
                    v_idx = mesh.vi_to_vertex[vi] if vi < len(mesh.vi_to_vertex) else vi
                    pos = mesh.vertices[v_idx] if v_idx < len(mesh.vertices) else (0.0, 0.0, 0.0)
                    # UE → glTF:  x=-ue_x, y=ue_z, z=-ue_y
                    px, py, pz = -pos[0], pos[2], -pos[1]

                    # Normal — same coordinate conversion
                    if has_normals and vi < len(mesh.normals):
                        n = mesh.normals[vi]
                        nx, ny, nz = -n[0], n[2], -n[1]
                    else:
                        nx, ny, nz = 0.0, 1.0, 0.0

                    # UV — no V-flip needed: UE5 stores textures top-to-bottom
                    # (same as PNG/glTF), and the UV coordinate (0,0) already
                    # maps to the first pixel row in both engines.
                    if has_uvs and vi < len(uvs):
                        u, v = uvs[vi]
                    else:
                        u, v = 0.0, 0.0

                    local_verts.append((px, py, pz, nx, ny, nz, u, v))
                    vi_to_local[vi] = len(local_verts) - 1

                indices.append(vi_to_local[vi])

        if not local_verts:
            continue

        num_verts = len(local_verts)

        # Numpy arrays
        pos_arr = np.array([(v[0], v[1], v[2]) for v in local_verts], dtype=np.float32)
        norm_arr = np.array([(v[3], v[4], v[5]) for v in local_verts], dtype=np.float32)
        uv_arr = np.array([(v[6], v[7]) for v in local_verts], dtype=np.float32)

        if num_verts <= 65535:
            idx_arr = np.array(indices, dtype=np.uint16)
            idx_comp = COMP_UNSIGNED_SHORT
        else:
            idx_arr = np.array(indices, dtype=np.uint32)
            idx_comp = COMP_UNSIGNED_INT

        # Position accessor
        pos_bv = _add_buffer_view(pos_arr.tobytes(), target=ARRAY_BUFFER)
        pos_acc = _add_accessor(
            pos_bv, COMP_FLOAT, num_verts, "VEC3",
            pos_arr.min(axis=0).tolist(), pos_arr.max(axis=0).tolist())

        # Normal accessor
        norm_acc = None
        if has_normals:
            norm_bv = _add_buffer_view(norm_arr.tobytes(), target=ARRAY_BUFFER)
            norm_acc = _add_accessor(norm_bv, COMP_FLOAT, num_verts, "VEC3")

        # UV accessor
        uv_acc = None
        if has_uvs:
            uv_bv = _add_buffer_view(uv_arr.tobytes(), target=ARRAY_BUFFER)
            uv_acc = _add_accessor(uv_bv, COMP_FLOAT, num_verts, "VEC2")

        # Index accessor
        idx_bv = _add_buffer_view(idx_arr.tobytes(), target=ELEMENT_ARRAY_BUFFER)
        idx_acc = _add_accessor(idx_bv, idx_comp, len(indices), "SCALAR")

        # Primitive
        prim = Primitive()
        prim.attributes.POSITION = pos_acc
        if norm_acc is not None:
            prim.attributes.NORMAL = norm_acc
        if uv_acc is not None:
            prim.attributes.TEXCOORD_0 = uv_acc
        prim.indices = idx_acc
        prim.material = mat_idx_to_gltf_mat[mat_idx]

        gltf_primitives.append(prim)

    # ------------------------------------------------------------------
    # Assemble glTF
    # ------------------------------------------------------------------
    gltf = GLTF2()
    gltf.scene = 0
    gltf.scenes = [GLTFScene(nodes=[0])]
    # No negative scale needed — the axis remap (UE_Y, UE_Z, -UE_X)
    # already flips handedness and is baked into the geometry.
    node = GLTFNode(mesh=0)
    gltf.nodes = [node]
    gltf.meshes = [GLTFMesh(primitives=gltf_primitives)]
    gltf.materials = gltf_materials
    if gltf_textures:
        gltf.textures = gltf_textures
    if gltf_samplers:
        gltf.samplers = gltf_samplers
    if gltf_images:
        gltf.images = gltf_images
    gltf.accessors = accessors
    gltf.bufferViews = buffer_views
    gltf.buffers = [Buffer(byteLength=len(binary))]

    gltf.set_binary_blob(bytes(binary))
    gltf.save(filepath)
