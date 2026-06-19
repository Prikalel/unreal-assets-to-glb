"""Static mesh parser and OBJ/GLB exporter for UE 4.27 .uasset files.

Supports both cooked (FStaticMeshRenderData) and uncooked (FMeshDescription) formats.
Pipeline: .uasset → Package → Export Bulk Data → Render Data → OBJ/GLB

Cooked UE4.27 Format (FStaticMeshRenderData):
- VertexBuffers: PositionVertexBuffer (positions), StaticMeshVertexBuffer (UVs, tangents)
- IndexBuffer: Triangle indices
- Sections: Material slot information per section

Uncooked Format (FMeshDescription):
- Triangles have structural data: VertexInstanceIDs[3] + PolygonID
- Material index comes from PolygonGroupID
"""
import os
import sys
import struct
import zlib
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import ooz

from .reader import BinaryReader
from .package import Package
from .bulk_data import find_mesh_description_bulk_data, BULKDATA_SerializeCompressedZLIB, BulkDataEntry, extract_bulk_data, find_bulk_data_in_export
from .uncooked_mesh import parse_uncooked_static_mesh

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


def decompress_zlib(data: bytes) -> Optional[bytes]:
    """Decompress ZLIB-compressed data.
    
    Returns the decompressed bytes, or None on failure.
    """
    try:
        return zlib.decompress(data)
    except Exception as e:
        #print(f"[DEBUG] ZLIB decompression failed: {e}")
        return None


def decompress_compressed_buffer(data: bytes) -> Optional[bytes]:
    """Decompress an FCompressedBuffer (UE4.27 big-endian header + Oodle/LZ4 blocks).

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
        # Safety check for corrupt string lengths
        if abs(length) > 1000000:  # 1MB string limit
            raise ValueError(f"String length too large: {length}")
        if length > 0:
            s = self.data[self.pos:self.pos + length - 1].decode('latin-1', errors='replace')
            self.pos += length
            return s
        elif length < 0:
            char_count = -length
            if char_count > 500000:  # 500k chars for UTF-16
                raise ValueError(f"UTF-16 string char count too large: {char_count}")
            s = self.data[self.pos:self.pos + char_count * 2].decode('utf-16-le', errors='replace')
            self.pos += char_count * 2
            return s
        return ""

    def read_tbit_array(self):
        num_bits = self.read_int32()
        # Safety check for corrupt num_bits
        if num_bits < 0 or num_bits > 10000000:  # 10M elements max
            raise ValueError(f"num_bits out of range: {num_bits}")
        num_words = (num_bits + 31) // 32
        # Safety check for too many words
        if num_words > 1000000:  # 4MB of words max
            raise ValueError(f"num_words too large: {num_words}")
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
            if count < 0 or count > 10000000:
                raise ValueError(f"count out of range in parse_attribute_array_base: {count}")
            raw = self.read_bytes(count * serialized_elem_size)
            return {'extent': extent, 'count': count, 'data': raw}
        else:
            # Element-by-element: TArray serialization (Count + elements)
            count = self.read_int32()
            if count < 0 or count > 100000:
                raise ValueError(f"count out of range in element-by-element parsing: {count}")
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
            if data_count < 0 or data_count > 10000000:
                raise ValueError(f"data_count out of range: {data_count}")
            raw = self.read_bytes(data_count * serialized_elem_size)
        elif attr_type == 6:
            data_count = self.read_int32()
            if data_count < 0 or data_count > 100000:
                raise ValueError(f"data_count out of range for FName: {data_count}")
            raw = [self.read_fstring() for _ in range(data_count)]
        else:
            data_count = self.read_int32()
            if data_count < 0 or data_count > 10000000:
                raise ValueError(f"data_count out of range: {data_count}")
            raw = self.read_bytes(data_count * 4)

        chunk_num_elements = self.read_int32()
        if chunk_num_elements < 0 or chunk_num_elements > 10000000:
            raise ValueError(f"chunk_num_elements out of range: {chunk_num_elements}")
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
        
        # Safety checks
        if num_elements < 0 or num_elements > 10000000:
            raise ValueError(f"num_elements out of range: {num_elements}")
        if num_channels < 0 or num_channels > 1000:
            raise ValueError(f"num_channels out of range: {num_channels}")

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

    def parse_element_container(self, element_type=None):
        """Parse FMeshElementContainer for UE4.27.
        
        UE4.27 format:
        - TBitArray (allocated indices)
        - Element data (for valid elements only)
        - Attributes
        
        Args:
            element_type: Hint about element type for parsing structural data
                          ('Triangles', 'Polygons', etc.)
        """
        num_bits, words = self.read_tbit_array()
        num_holes = self.read_int32()
        valid_count = self.count_valid_elements(num_bits, words)
        
        # Read element data for each valid element (UE4.27)
        element_data = []
        
        if element_type == 'Triangles':
            # FMeshTriangle in UE4.27:
            # - VertexInstanceID[0] (int32)
            # - VertexInstanceID[1] (int32)
            # - VertexInstanceID[2] (int32)
            # - PolygonID (int32) - always present in UE4.27 MeshDescriptionTriangles
            for _ in range(valid_count):
                vi0 = self.read_int32()
                vi1 = self.read_int32()
                vi2 = self.read_int32()
                polygon_id = self.read_int32()
                element_data.append({
                    'vi0': vi0, 'vi1': vi1, 'vi2': vi2, 'polygon_id': polygon_id
                })
        elif element_type == 'Polygons':
            # FMeshPolygon in UE4.27:
            # - VertexInstanceIDs (TArray<int32>)
            # - TriangleIDs (TArray<int32>)
            # - PolygonGroupID (int32)
            for _ in range(valid_count):
                num_vis = self.read_int32()
                vis = [self.read_int32() for __ in range(num_vis)]
                num_tris = self.read_int32()
                tris = [self.read_int32() for __ in range(num_tris)]
                poly_group_id = self.read_int32()
                element_data.append({
                    'vis': vis, 'tris': tris, 'poly_group_id': poly_group_id
                })
        elif element_type == 'Vertices':
            # FMeshVertex: no data (just empty struct after new serialization)
            pass
        elif element_type == 'VertexInstances':
            # FMeshVertexInstance:
            # - VertexID (int32)
            for _ in range(valid_count):
                vertex_id = self.read_int32()
                element_data.append({'vertex_id': vertex_id})
        elif element_type == 'Edges':
            # FMeshEdge:
            # - VertexID[0] (int32)
            # - VertexID[1] (int32)
            for _ in range(valid_count):
                vid0 = self.read_int32()
                vid1 = self.read_int32()
                element_data.append({'vid0': vid0, 'vid1': vid1})
        else:
            # Unknown element type - try to skip data
            # We can't skip properly without knowing the structure
            # Assume no element data for safety
            pass
        
        attributes = self.parse_attributes_set_base()
        
        return {'num_bits': num_bits, 'num_holes': num_holes,
                'valid_count': valid_count, 'words': words,
                'element_data': element_data, 'attributes': attributes}

    def parse_mesh_description(self):
        """Parse the full FMeshDescription."""
        num_entries = self.read_int32()
        #print(f"[DEBUG] Parsing mesh description with {num_entries} entries")
        elements = {}
        for i in range(num_entries):
            key = self.read_fstring()
            #print(f"[DEBUG] Entry {i}/{num_entries}: {key}")
            # FMeshElementChannels: TArray<FMeshElementContainer>
            num_channels = self.read_int32()
            #print(f"[DEBUG]   Parsing {num_channels} channels...")
            # Pass element type hint for UE4.27 structural data parsing
            channels = [self.parse_element_container(element_type=key) for _ in range(num_channels)]
            elements[key] = {'channels': channels}
        #print(f"[DEBUG] Finished parsing mesh description")
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
    vi_element_data = vi_ch.get('element_data', [])
    
    # UE4.27: Read vertex ID from element data
    vi_to_vertex_dense = []
    for elem in vi_element_data:
        if 'vertex_id' in elem:
            vi_to_vertex_dense.append(elem['vertex_id'])
    
    if not vi_to_vertex_dense:
        return result
    
    result['vi_to_vertex'] = _maybe_expand_sparse(vi_to_vertex_dense, vi_ch, default=0)

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
                dense_uvs = []
                if isinstance(channel.get('data'), bytes):
                    # Bounded attribute — direct data array
                    ch_raw = channel['data']
                    ch_count = channel['count']
                    for i in range(ch_count):
                        u, v = struct.unpack_from('<ff', ch_raw, i * 8)
                        dense_uvs.append((u, v))
                elif 'chunks' in channel:
                    # Unbounded attribute — collect data from chunks
                    for chunk in channel.get('chunks', []):
                        chunk_data = chunk.get('data')
                        if isinstance(chunk_data, bytes):
                            num_uvs = len(chunk_data) // 8
                            for i in range(num_uvs):
                                u, v = struct.unpack_from('<ff', chunk_data, i * 8)
                                dense_uvs.append((u, v))
                if dense_uvs:
                    uv_channels.append(_maybe_expand_sparse(dense_uvs, vi_ch, default=(0.0, 0.0)))
            break
    result['uvs'] = uv_channels

    # --- Triangle data ---
    if 'Triangles' not in elements:
        return None
    tri_ch = elements['Triangles']['channels'][0]
    tri_element_data = tri_ch.get('element_data', [])

    # UE4.27: Read triangle vertex instances from element data
    triangles_raw = []
    
    if not tri_element_data:
        return result
    
    # Parse triangles from element data
    # Each triangle has: vi0, vi1, vi2, polygon_id
    tri_to_polygon = []  # Map triangle index to polygon ID
    tri_idx = 0
    for elem in tri_element_data:
        if 'vi0' in elem and 'vi1' in elem and 'vi2' in elem and 'polygon_id' in elem:
            triangles_raw.append((elem['vi0'], elem['vi1'], elem['vi2']))
            tri_to_polygon.append(elem['polygon_id'])
            tri_idx += 1

    # --- Material indices from Triangle → Polygon → PolygonGroup ---
    material_indices: List[int] = []
    
    if 'Polygons' in elements:
        poly_ch = elements['Polygons']['channels'][0]
        poly_element_data = poly_ch.get('element_data', [])
        
        # Build mapping from polygon ID (sparse index) to polygon group ID
        # We need to account for sparse indices (holes)
        poly_sparse_to_dense = _build_sparse_mapping(
            poly_ch['num_bits'], poly_ch['words']
        )
        
        # Build dense array of polygon group IDs indexed by dense polygon ID
        poly_group_dense = []
        for elem in poly_element_data:
            if 'poly_group_id' in elem:
                poly_group_dense.append(elem['poly_group_id'])
        
        # Map triangle polygon IDs to polygon group IDs
        for poly_id in tri_to_polygon:
            # poly_id is the sparse polygon ID
            if poly_id in poly_sparse_to_dense:
                dense_poly_idx = poly_sparse_to_dense[poly_id]
                if dense_poly_idx < len(poly_group_dense):
                    material_indices.append(poly_group_dense[dense_poly_idx])
                else:
                    material_indices.append(0)
            else:
                material_indices.append(0)
    else:
        # No polygons data, default to material 0
        material_indices = [0] * len(triangles_raw)

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
# StaticMaterials parser
# ---------------------------------------------------------------------------

def _parse_static_materials(pkg: Package) -> Optional[List[Tuple[str, str]]]:
    """Parse the StaticMaterials property from the StaticMesh export data.

    Reads the real material-slot-to-import mapping from the UStaticMesh
    export's serialized ``StaticMaterials`` array.  Each ``FStaticMaterial``
    element contains an ``ImportedMaterialSlotName`` and a
    ``MaterialInterface`` (FPackageIndex pointing to a material import).

    Returns:
        Ordered list of ``(ImportedMaterialSlotName, material_import_name)``
        tuples, or ``None`` if parsing fails.
    """
    from .properties import read_properties

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

    # Find the FName index for "StaticMaterials"
    sm_name_idx = None
    for ni, n in enumerate(pkg.name_map):
        if n == 'StaticMaterials':
            sm_name_idx = ni
            break
    if sm_name_idx is None:
        return None

    # Search for the FName (index + number=0) in the binary data
    target = struct.pack('<ii', sm_name_idx, 0)
    offset = 0
    while offset < len(data) - len(target):
        idx = data.find(target, offset)
        if idx == -1:
            return None
        try:
            result = _parse_static_materials_at(data, idx, pkg)
            if result is not None:
                return result
        except Exception:
            pass
        offset = idx + 1
    return None


def _parse_static_materials_at(data: bytes, offset: int,
                               pkg: Package) -> Optional[List[Tuple[str, str]]]:
    """Try to parse the StaticMaterials array property at *offset*."""
    from .properties import read_properties

    r = BinaryReader(data)
    r.seek(offset)

    # Skip Name FName (8 bytes)
    r.skip(8)

    # Skip type tree (FPropertyTypeName)
    total_nodes = 1
    i = 0
    while i < total_nodes:
        r.skip(8)  # FName (idx + number)
        inner_count = r.read_int32()
        total_nodes += inner_count
        i += 1

    # Size
    size = r.read_int32()
    if size <= 0 or size > len(data):
        return None

    # Flags
    flags = r.read_uint8()
    if flags & 0x01:  # HasArrayIndex
        r.skip(4)
    if flags & 0x02:  # HasPropertyGuid
        r.skip(16)
    if flags & 0x04:  # HasPropertyExtensions
        ext = r.read_uint8()
        if ext & 0x01:
            r.skip(1 + 4)

    # Array element count
    arr_count = r.read_int32()
    if arr_count < 0 or arr_count > 256:
        return None

    # Parse each FStaticMaterial struct (properties until "None")
    result: List[Tuple[str, str]] = []
    for _ in range(arr_count):
        elem_props = read_properties(r, pkg.name_map, pkg.file_version_ue5)
        imported_slot_name = elem_props.get('ImportedMaterialSlotName')
        material_interface = elem_props.get('MaterialInterface')
        if imported_slot_name is None or material_interface is None:
            continue
        # Resolve FPackageIndex to import name
        if isinstance(material_interface, int) and material_interface < 0:
            imp_idx = -material_interface - 1
            if 0 <= imp_idx < len(pkg.imports):
                result.append(
                    (imported_slot_name, pkg.imports[imp_idx].object_name))

    return result if result else None


def _parse_section_info_map(data: bytes, name_map: list,
                            num_sections: int) -> Optional[List[int]]:
    """Parse the SectionInfoMap to extract MaterialIndex overrides for LOD 0.

    UE5's ``FMeshSectionInfoMap`` stores a ``Map<UInt32, FMeshSectionInfo>``
    keyed by ``GetMeshMaterialKey(LOD, Section) = (LOD << 16) | Section``.
    For LOD 0 the keys are simply 0, 1, 2, … and the map is populated in
    insertion order (section 0 first).  Each ``FMeshSectionInfo`` contains a
    ``MaterialIndex`` *IntProperty* that overrides which entry in the
    ``StaticMaterials`` array the section should use.

    Returns a list of *MaterialIndex* values for the first *num_sections*
    sections of LOD 0, or ``None`` on failure.
    """
    if num_sections <= 0:
        return None

    # Locate required FName indices
    sim_idx = sm_idx = mi_idx = None
    for i, n in enumerate(name_map):
        if n == 'SectionInfoMap':
            sim_idx = i
        elif n == 'StaticMaterials':
            sm_idx = i
        elif n == 'MaterialIndex':
            mi_idx = i
    if sim_idx is None or mi_idx is None:
        return None

    # Find SectionInfoMap property start
    target = struct.pack('<ii', sim_idx, 0)
    sim_offset = data.find(target)
    if sim_offset < 0:
        return None

    # Determine end of SectionInfoMap range (StaticMaterials comes after)
    search_end = len(data)
    if sm_idx is not None:
        target = struct.pack('<ii', sm_idx, 0)
        sm_offset = data.find(target, sim_offset + 8)
        if sm_offset >= 0:
            search_end = sm_offset

    # Scan for MaterialIndex FName occurrences and extract IntProperty values
    mi_pattern = struct.pack('<ii', mi_idx, 0)
    material_indices: List[int] = []
    offset = sim_offset
    while offset < search_end and len(material_indices) < num_sections:
        idx = data.find(mi_pattern, offset, search_end)
        if idx == -1:
            break

        # After the MaterialIndex FName (8 bytes) comes the type tree
        pos = idx + 8
        if pos + 12 > len(data):
            break

        type_idx = struct.unpack_from('<i', data, pos)[0]
        type_name = name_map[type_idx] if 0 <= type_idx < len(name_map) else ''
        pos += 8  # type FName
        inner_count = struct.unpack_from('<i', data, pos)[0]
        pos += 4  # inner_count

        if type_name == 'IntProperty' and inner_count == 0:
            if pos + 5 > len(data):
                break
            size = struct.unpack_from('<i', data, pos)[0]
            pos += 4  # size
            flags = data[pos]
            pos += 1  # flags

            if flags & 0x01:  # HasArrayIndex
                pos += 4
            if flags & 0x02:  # HasPropertyGuid
                pos += 16
            if flags & 0x04:  # HasPropertyExtensions
                if pos >= len(data):
                    break
                ext = data[pos]
                pos += 1
                if ext & 0x01:
                    pos += 1 + 4

            if size >= 4 and pos + 4 <= len(data):
                value = struct.unpack_from('<i', data, pos)[0]
                material_indices.append(value)

        offset = idx + 1

    if len(material_indices) >= num_sections:
        return material_indices[:num_sections]
    return None


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
        self.indices: Optional[List[int]] = None  # Raw index buffer (for cooked meshes)
        self.material_slot_names: Optional[List[Optional[str]]] = None
        # Ordered list of (ImportedMaterialSlotName, material_import_name)
        # tuples parsed from the StaticMaterials export property, indexed by
        # material slot index.
        self.material_slots: Optional[List[Tuple[str, str]]] = None
        # SectionInfoMap: maps polygon group index → material slot index.
        self.section_info_map: Optional[List[int]] = None

    @classmethod
    def from_package(cls, pkg: Package) -> Optional['StaticMesh']:
        """Parse static mesh from a Package.
        
        Tries cooked FStaticMeshRenderData format first, then falls back to
        uncooked FMeshDescription format.
        """
        import sys
        mesh = cls()

        # Find StaticMesh export
        sm_export_idx = None
        for i in range(pkg.export_count):
            if pkg.get_export_class_name(i) == 'StaticMesh':
                sm_export_idx = i
                break
        
        if sm_export_idx is None:
            print("[DEBUG] No StaticMesh export found")
            return None
        
        print(f"[DEBUG] StaticMesh export index: {sm_export_idx}")

        # ------------------------------------------------------------------
        # PRIMARY PATH (uncooked UE4.27 assets): the geometry lives inside a
        # compressed FMeshDescription bulk-data blob referenced from the
        # StaticMesh export serial data.  This is the correct, fully-decoded
        # path implemented in uasset/uncooked_mesh.py.  Try it first because
        # these are uncooked (bCooked == 0) assets.
        # ------------------------------------------------------------------
        print("[DEBUG] Trying uncooked export-serial FMeshDescription path...")
        sys.stdout.flush()
        try:
            geo = parse_uncooked_static_mesh(pkg)
            if geo is not None and geo.get('vertices'):
                mesh.vertices = geo['vertices']
                mesh.vi_to_vertex = geo['vi_to_vertex']
                mesh.normals = geo['normals']
                mesh.uvs = geo['uvs']
                mesh.triangles = geo['triangles']
                mesh.material_slot_names = geo.get('material_slot_names')
                _apply_material_mapping(mesh, pkg)
                print(f"[DEBUG] Uncooked path OK: {len(mesh.vertices)} verts, "
                      f"{len(mesh.triangles)} tris")
                return mesh
            print("[DEBUG] Uncooked path returned no geometry; trying fallbacks")
        except Exception as e:
            print(f"[DEBUG] Uncooked path failed: {e}")
            import traceback
            traceback.print_exc()

        # FALLBACK 1: cooked FStaticMeshRenderData format
        print("[DEBUG] Trying cooked FStaticMeshRenderData format...")
        sys.stdout.flush()
        try:
            mesh = parse_cooked_static_mesh(pkg)
            if mesh is not None:
                print("[DEBUG] Successfully parsed cooked mesh")
                return mesh
        except Exception as e:
            print(f"[DEBUG] Cooked mesh parsing failed: {e}")
            import traceback
            traceback.print_exc()

        # Fall back to uncooked FMeshDescription format
        print("[DEBUG] Falling back to FMeshDescription format...")
        sys.stdout.flush()

        # Extract mesh description from file trailer (uncooked packages)
        print("[DEBUG] Extracting trailer payload...")
        compressed_buffer = extract_trailer_payload(pkg.reader.data)
        if compressed_buffer is None:
            print("[DEBUG] Failed to extract trailer payload")
            return None
        
        print(f"[DEBUG] Compressed buffer size: {len(compressed_buffer)} bytes")

        # Decompress FCompressedBuffer (Oodle/LZ4)
        raw_data = decompress_compressed_buffer(compressed_buffer)
        if raw_data is None:
            print("[DEBUG] Failed to decompress FCompressedBuffer")
            return None
        
        print(f"[DEBUG] Decompressed FCompressedBuffer: {len(raw_data)} bytes")

        # Parse FMeshDescription
        print("[DEBUG] Parsing mesh description...")
        sys.stdout.flush()
        reader = _MeshDescReader(raw_data)
        try:
            elements = reader.parse_mesh_description()
        except Exception as e:
            print(f"[DEBUG] Exception during mesh description parsing: {e}")
            import traceback
            traceback.print_exc()
            return None
        print(f"[DEBUG] Mesh description parsed, elements: {list(elements.keys())}")

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

        _apply_material_mapping(mesh, pkg)

        return mesh


def _apply_material_mapping(mesh: 'StaticMesh', pkg: Package) -> None:
    """Populate the material slot mapping on *mesh* from the export data.

    Reads the ordered ``StaticMaterials`` array (ImportedMaterialSlotName +
    material import name) and the ``SectionInfoMap`` (polygon-group index ->
    material slot index) and applies the latter as an override so that each
    polygon group resolves to the correct ``ImportedMaterialSlotName``.
    """
    mesh.material_slots = _parse_static_materials(pkg)

    if mesh.material_slots and mesh.material_slot_names:
        sm_export_idx = None
        for i in range(pkg.export_count):
            if pkg.get_export_class_name(i) == 'StaticMesh':
                sm_export_idx = i
                break
        if sm_export_idx is not None:
            exp_reader = pkg.get_export_data(sm_export_idx)
            if exp_reader is not None:
                section_map = _parse_section_info_map(
                    exp_reader.data, pkg.name_map,
                    len(mesh.material_slot_names))
                mesh.section_info_map = section_map
                if section_map is not None:
                    # section_map[pg_idx] = material slot index into
                    # StaticMaterials; mesh.material_slots[slot_idx] =
                    # (slot_name, material_name).  Remap material_slot_names
                    # so each PG gets the ImportedMaterialSlotName from the
                    # correct StaticMaterials entry.
                    remapped: List[Optional[str]] = []
                    for pg_idx in range(len(mesh.material_slot_names)):
                        if pg_idx < len(section_map):
                            mat_idx = section_map[pg_idx]
                            if mat_idx < len(mesh.material_slots):
                                remapped.append(
                                    mesh.material_slots[mat_idx][0])
                                continue
                        remapped.append(
                            mesh.material_slot_names[pg_idx])
                    mesh.material_slot_names = remapped


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
    # (X right, Y up, -Z fwd).  The mapping  glTF = (UE_Y, UE_Z, -UE_X)
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
                    # UE → glTF:  x=ue_y, y=ue_z, z=-ue_x
                    px, py, pz = pos[1], pos[2], -pos[0]

                    # Normal — same coordinate conversion
                    if has_normals and vi < len(mesh.normals):
                        n = mesh.normals[vi]
                        nx, ny, nz = n[1], n[2], -n[0]
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
    return filepath


# ---------------------------------------------------------------------------
# FStaticMeshRenderData parsing (Cooked UE4.27)
# ---------------------------------------------------------------------------

def parse_cooked_static_mesh(pkg: Package) -> Optional[StaticMesh]:
    """Parse a cooked StaticMesh using FStaticMeshRenderData format.
    
    Args:
        pkg: Package object containing the StaticMesh
        
    Returns:
        StaticMesh object or None if parsing fails
    """
    from .mesh_render_data import (
        find_render_data_bulk_data,
        decompress_bulk_data,
        parse_position_vertex_buffer,
        parse_static_mesh_vertex_buffer,
        parse_index_buffer,
        parse_static_mesh_sections
    )
    
    # Find StaticMesh export
    sm_export_idx = None
    for i in range(pkg.export_count):
        if pkg.get_export_class_name(i) == 'StaticMesh':
            sm_export_idx = i
            break
    
    if sm_export_idx is None:
        return None
    
    # Find bulk data entries
    bulk_data_list = find_render_data_bulk_data(pkg, sm_export_idx)
    if bulk_data_list is None:
        return None
    
    mesh = StaticMesh()
    
    # Try to identify and parse each bulk data entry
    for entry, raw_data in bulk_data_list:
        # Decompress if needed
        decompressed = decompress_bulk_data(raw_data, entry.flags)
        if decompressed is None:
            continue
        
        data_len = len(decompressed)
        
        # Try to parse as different buffer types based on size and structure
        
        # PositionVertexBuffer: starts with num_vertices (uint32), ~12 bytes per vertex
        if data_len >= 4 and data_len <= 100000000 and not mesh.vertices:
            try:
                num_vertices = struct.unpack_from('<I', decompressed, 0)[0]
                if 0 < num_vertices <= 10000000 and data_len == 4 + num_vertices * 12:
                    positions = parse_position_vertex_buffer(decompressed)
                    if positions and len(positions) == num_vertices:
                        mesh.vertices = positions
                        continue
            except:
                pass
        
        # IndexBuffer: starts with stride (uint32) + num_indices (uint32)
        if data_len >= 8 and not mesh.indices:
            try:
                stride = struct.unpack_from('<I', decompressed, 0)[0]
                if stride in (2, 4):
                    indices = parse_index_buffer(decompressed)
                    if indices and len(indices) > 0:
                        mesh.indices = indices
                        continue
            except:
                pass
        
        # StaticMeshVertexBuffer: has stride, num_tex_coords, num_vertices
        if data_len >= 12 and mesh.vertices and not mesh.uvs:
            try:
                stride = struct.unpack_from('<I', decompressed, 0)[0]
                num_vertices = len(mesh.vertices)
                if stride >= 32 and stride <= 64:
                    uv_data = parse_static_mesh_vertex_buffer(decompressed, num_vertices)
                    if uv_data:
                        mesh.uvs = uv_data['uvs']
                        continue
            except:
                pass
        
        # Sections array
        if data_len >= 8 and not hasattr(mesh, 'sections'):
            sections = parse_static_mesh_sections(decompressed)
            if sections:
                mesh.sections = sections
                continue
    
    # Validate we have essential data
    if not mesh.vertices or not mesh.indices:
        return None
    
    # Build triangles from indices
    if len(mesh.indices) % 3 != 0:
        return None
    
    mesh.triangles = []
    num_tris = len(mesh.indices) // 3
    
    for i in range(num_tris):
        i0 = mesh.indices[i * 3 + 0]
        i1 = mesh.indices[i * 3 + 1]
        i2 = mesh.indices[i * 3 + 2]
        
        # Determine material index from sections
        mat_idx = 0
        if hasattr(mesh, 'sections') and mesh.sections:
            for section in mesh.sections:
                first_tri = section['first_index'] // 3
                tri_count = section['num_triangles']
                if first_tri <= i < first_tri + tri_count:
                    mat_idx = section['material_index']
                    break
        
        mesh.triangles.append((i0, i1, i2, mat_idx))
    
    # Parse material slots from export
    mesh.material_slots = _parse_static_materials(pkg)
    
    # Build material slot names
    if mesh.material_slots:
        mesh.material_slot_names = [slot[0] if slot else f"Material_{i}"
                                    for i, slot in enumerate(mesh.material_slots)]
    else:
        mesh.material_slot_names = ["Material_0"]
    
    # Build vi_to_vertex (identity mapping for cooked meshes)
    mesh.vi_to_vertex = list(range(len(mesh.vertices)))
    
    return mesh
    gltf.save(filepath)
