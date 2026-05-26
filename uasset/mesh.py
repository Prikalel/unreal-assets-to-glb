"""Static mesh parser and OBJ exporter for UE 5.5 .uasset files.

Parses FMeshDescription from FCompressedBuffer payload in the package trailer.
Pipeline: .uasset → Package → Trailer → FCompressedBuffer → Oodle decompress → FMeshDescription → OBJ
"""
import struct
from typing import List, Tuple, Optional

import ooz

from .reader import BinaryReader
from .package import Package


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
                'valid_count': valid_count, 'attributes': attributes}

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


def extract_mesh_data(elements: dict) -> Optional[dict]:
    """Extract vertices, normals, UVs, and triangles from parsed FMeshDescription."""
    result = {
        'vertices': [],
        'vi_to_vertex': [],
        'normals': [],
        'uvs': [],
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
    result['vertices'] = vertices

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
    result['vi_to_vertex'] = vi_to_vertex

    # --- Normals (per vertex instance) ---
    normal_data = _extract_attr_data(vi_ch, 'Normal', expected_type=1)
    if normal_data:
        raw = normal_data['data']
        count = normal_data['count']
        normals = []
        for i in range(count):
            x, y, z = struct.unpack_from('<fff', raw, i * 12)
            normals.append((x, y, z))
        result['normals'] = normals

    # --- UV coordinates (per vertex instance, TextureCoordinate) ---
    uv_data = _extract_attr_data(vi_ch, 'TextureCoordinate', expected_type=2)
    if uv_data:
        raw = uv_data['data']
        count = uv_data['count']
        uvs = []
        for i in range(count):
            u, v = struct.unpack_from('<ff', raw, i * 8)
            uvs.append((u, v))
        result['uvs'] = uvs

    # --- Triangle data ---
    if 'Triangles' not in elements:
        return None
    tri_ch = elements['Triangles']['channels'][0]
    tri_attrs = tri_ch['attributes']

    # VertexInstanceIndex (extent=3, int32)
    for name, attr in tri_attrs['attributes'].items():
        if name == 'VertexInstanceIndex' and attr['type'] == 4 and attr['extent'] == 3:
            raw = attr['channels'][0]['data']
            count = attr['channels'][0]['count']
            triangles = []
            for i in range(count // 3):
                v0, v1, v2 = struct.unpack_from('<iii', raw, i * 12)
                triangles.append((v0, v1, v2))
            result['triangles'] = triangles
            break

    return result


# ---------------------------------------------------------------------------
# StaticMesh class
# ---------------------------------------------------------------------------

class StaticMesh:
    def __init__(self):
        self.vertices: List[Tuple[float, float, float]] = []
        self.normals: List[Tuple[float, float, float]] = []
        self.uvs: List[Tuple[float, float]] = []
        self.triangles: List[Tuple[int, int, int]] = []  # vertex instance indices
        self.vi_to_vertex: List[int] = []

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

        return mesh


# ---------------------------------------------------------------------------
# OBJ export
# ---------------------------------------------------------------------------

def export_obj(mesh: StaticMesh, filepath: str):
    """Export a StaticMesh as Wavefront OBJ."""
    with open(filepath, 'w') as f:
        f.write("# UE5 Static Mesh Export\n")
        num_tris = len(mesh.triangles)
        f.write(f"# {len(mesh.vertices)} vertices, {num_tris} triangles\n\n")

        # Vertex positions
        for x, y, z in mesh.vertices:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        f.write("\n")

        # Normals (per vertex instance)
        if mesh.normals:
            for nx, ny, nz in mesh.normals:
                f.write(f"vn {nx:.6f} {ny:.6f} {nz:.6f}\n")
            f.write("\n")

        # UVs (per vertex instance)
        if mesh.uvs:
            for u, v in mesh.uvs:
                f.write(f"vt {u:.6f} {1.0 - v:.6f}\n")  # Flip V for OBJ
            f.write("\n")

        # Faces
        has_normals = bool(mesh.normals)
        has_uvs = bool(mesh.uvs)

        for vi0, vi1, vi2 in mesh.triangles:
            # Resolve vertex positions from vertex instance → vertex mapping
            v0 = mesh.vi_to_vertex[vi0] if vi0 < len(mesh.vi_to_vertex) else vi0
            v1 = mesh.vi_to_vertex[vi1] if vi1 < len(mesh.vi_to_vertex) else vi1
            v2 = mesh.vi_to_vertex[vi2] if vi2 < len(mesh.vi_to_vertex) else vi2

            # OBJ uses 1-based indices
            p0, p1, p2 = v0 + 1, v1 + 1, v2 + 1

            if has_uvs and has_normals:
                f.write(f"f {p0}/{vi0+1}/{vi0+1} {p1}/{vi1+1}/{vi1+1} {p2}/{vi2+1}/{vi2+1}\n")
            elif has_uvs:
                f.write(f"f {p0}/{vi0+1} {p1}/{vi1+1} {p2}/{vi2+1}\n")
            elif has_normals:
                f.write(f"f {p0}//{vi0+1} {p1}//{vi1+1} {p2}//{vi2+1}\n")
            else:
                f.write(f"f {p0} {p1} {p2}\n")
