"""FStaticMeshRenderData binary parser for cooked UE4.27 packages.

This module provides functions to parse cooked StaticMesh data using the
FStaticMeshRenderData format instead of FMeshDescription.
"""
import struct
from typing import Dict, List, Tuple, Optional

from .reader import BinaryReader
from .package import Package
from .bulk_data import BulkDataEntry, extract_bulk_data, BULKDATA_PayloadAtEndOfFile, BULKDATA_Size64Bit, BULKDATA_SerializeCompressedZLIB


class _RenderDataReader:
    """Low-level reader for FStaticMeshRenderData bulk data in cooked packages."""
    
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0
        
    def read_int8(self):
        val = struct.unpack_from('<b', self.data, self.pos)[0]
        self.pos += 1
        return val
    
    def read_uint8(self):
        val = struct.unpack_from('<B', self.data, self.pos)[0]
        self.pos += 1
        return val
    
    def read_int32(self):
        val = struct.unpack_from('<i', self.data, self.pos)[0]
        self.pos += 4
        return val
    
    def read_uint32(self):
        val = struct.unpack_from('<I', self.data, self.pos)[0]
        self.pos += 4
        return val
    
    def read_int64(self):
        val = struct.unpack_from('<q', self.data, self.pos)[0]
        self.pos += 8
        return val
    
    def read_uint64(self):
        val = struct.unpack_from('<Q', self.data, self.pos)[0]
        self.pos += 8
        return val
    
    def read_float(self):
        val = struct.unpack_from('<f', self.data, self.pos)[0]
        self.pos += 4
        return val
    
    def read_uint16(self):
        val = struct.unpack_from('<H', self.data, self.pos)[0]
        self.pos += 2
        return val
    
    def read_bytes(self, n):
        val = self.data[self.pos:self.pos + n]
        self.pos += n
        return val
    
    def skip(self, n):
        self.pos += n
    
    def tell(self):
        return self.pos
    
    def seek(self, pos):
        self.pos = pos
    
    def remaining(self):
        return len(self.data) - self.pos


def parse_bulk_data_entry(data: bytes, offset: int) -> Tuple[Optional[BulkDataEntry], int]:
    """Parse a bulk data entry header at the given offset.
    
    Returns:
        Tuple of (BulkDataEntry, new_offset) or (None, offset) if parsing fails
    """
    if offset + 20 > len(data):
        return None, offset
    
    try:
        flags = struct.unpack_from('<I', data, offset)[0]
        
        # Check if this looks like a valid flags value
        if flags >= 0x10000000:
            return None, offset
        
        # Determine 64-bit size flag
        size_64bit = (flags & BULKDATA_Size64Bit) != 0
        
        # Read element count
        if size_64bit:
            if offset + 12 > len(data):
                return None, offset
            element_count = struct.unpack_from('<Q', data, offset + 4)[0]
            size_on_disk_offset = offset + 12
        else:
            if offset + 8 > len(data):
                return None, offset
            element_count = struct.unpack_from('<I', data, offset + 4)[0]
            size_on_disk_offset = offset + 8
        
        # Validate element count
        if element_count < 0 or element_count >= 1e9:
            return None, offset
        
        # Read size on disk
        if size_on_disk_offset + 8 > len(data):
            return None, offset
        
        if size_64bit:
            size_on_disk = struct.unpack_from('<Q', data, size_on_disk_offset)[0]
            offset_offset = size_on_disk_offset + 8
        else:
            size_on_disk = struct.unpack_from('<I', data, size_on_disk_offset)[0]
            offset_offset = size_on_disk_offset + 4
        
        # Validate size
        if size_on_disk < 0 or size_on_disk >= 1e10:
            return None, offset
        
        # Read offset in file
        if offset_offset + 8 > len(data):
            return None, offset
        
        offset_in_file = struct.unpack_from('<q', data, offset_offset)[0]
        
        entry = BulkDataEntry()
        entry.flags = flags
        entry.element_count = element_count
        entry.size_on_disk = size_on_disk
        entry.offset_in_file = offset_in_file
        entry.header_size = offset_offset + 8
        
        # Check if payload is inline
        if (flags & BULKDATA_PayloadAtEndOfFile) == 0:
            entry.data_location = 'inline'
            entry.actual_offset = entry.header_size
        else:
            entry.data_location = 'end_of_file'
        
        return entry, entry.header_size
        
    except Exception:
        return None, offset


def decompress_bulk_data(raw_data: bytes, flags: int) -> Optional[bytes]:
    """Decompress bulk data based on flags.
    
    Args:
        raw_data: Raw bulk data bytes (possibly compressed)
        flags: Bulk data flags
        
    Returns:
        Decompressed data or None if decompression fails
    """
    if flags & BULKDATA_SerializeCompressedZLIB:
        try:
            import zlib
            return zlib.decompress(raw_data)
        except Exception:
            return None
    
    # Try Oodle decompression
    try:
        from .mesh import decompress_compressed_buffer
        decompressed = decompress_compressed_buffer(raw_data)
        if decompressed is not None:
            return decompressed
    except Exception:
        pass
    
    # Assume uncompressed
    return raw_data


def parse_position_vertex_buffer(data: bytes) -> Optional[List[Tuple[float, float, float]]]:
    """Parse FPositionVertexBuffer data.
    
    Expected format: NumVertices (uint32) + NumVertices * 12 bytes (FVector3f per vertex)
    
    Returns:
        List of (x, y, z) position tuples or None if parsing fails
    """
    if len(data) < 4:
        return None
    
    reader = _RenderDataReader(data)
    try:
        num_vertices = reader.read_uint32()
        
        if num_vertices <= 0 or num_vertices > 10000000:
            return None
        
        expected_size = 4 + num_vertices * 12
        if len(data) < expected_size:
            return None
        
        positions = []
        for _ in range(num_vertices):
            x = reader.read_float()
            y = reader.read_float()
            z = reader.read_float()
            positions.append((x, y, z))
        
        return positions
        
    except Exception:
        return None


def parse_static_mesh_vertex_buffer(data: bytes, num_vertices: int, num_tex_coords: int = 1) -> Optional[Dict]:
    """Parse FStaticMeshVertexBuffer data.
    
    Expected format:
    - Stride (uint32)
    - NumTexCoords (uint32)
    - NumVertices (uint32)
    - Vertex data:
        - TangentX (FVector4f, 16 bytes) per vertex
        - TangentZ (FVector4f, 16 bytes) per vertex  
        - TexCoords (NumTexCoords * FVector2f, 8 bytes each) per vertex
    
    Returns:
        Dict with 'uvs' (list of UV channel lists) and 'tangents' (optional) or None
    """
    if len(data) < 12:
        return None
    
    reader = _RenderDataReader(data)
    try:
        stride = reader.read_uint32()
        read_num_tex_coords = reader.read_uint32()
        read_num_vertices = reader.read_uint32()
        
        if read_num_vertices != num_vertices or read_num_vertices <= 0:
            return None
        
        # Calculate expected stride
        # TangentX (16) + TangentZ (16) + NumTexCoords * 8
        expected_stride = 32 + read_num_tex_coords * 8
        
        # Sometimes stride differs (e.g., for packed normals)
        if stride == 0:
            stride = expected_stride
        
        uvs = [[] for _ in range(read_num_tex_coords)]
        
        for vi in range(read_num_vertices):
            # Skip TangentX (FVector4f)
            reader.skip(16)
            # Skip TangentZ (FVector4f)  
            reader.skip(16)
            
            # Read UVs
            for ti in range(read_num_tex_coords):
                u = reader.read_float()
                v = reader.read_float()
                uvs[ti].append((u, v))
            
            # Skip any remaining stride padding
            remaining = stride - (32 + read_num_tex_coords * 8)
            if remaining > 0:
                reader.skip(remaining)
        
        return {
            'uvs': uvs,
            'num_tex_coords': read_num_tex_coords
        }
        
    except Exception:
        return None


def parse_index_buffer(data: bytes) -> Optional[List[int]]:
    """Parse FRawStaticIndexBuffer data.
    
    Expected format:
    - Index stride (uint32, 2 or 4)
    - NumIndices (uint32)
    - Index data (2 or 4 bytes per index)
    
    Returns:
        List of index values or None if parsing fails
    """
    if len(data) < 8:
        return None
    
    reader = _RenderDataReader(data)
    try:
        index_stride = reader.read_uint32()
        num_indices = reader.read_uint32()
        
        if num_indices <= 0 or num_indices > 100000000:
            return None
        
        if index_stride not in (2, 4):
            return None
        
        expected_data_size = num_indices * index_stride
        if reader.remaining() < expected_data_size:
            return None
        
        indices = []
        for _ in range(num_indices):
            if index_stride == 2:
                idx = reader.read_uint16()
            else:
                idx = reader.read_uint32()
            indices.append(idx)
        
        return indices
        
    except Exception:
        return None


def parse_static_mesh_sections(data: bytes) -> Optional[List[Dict]]:
    """Parse FStaticMeshSection array from LOD resources.
    
    Expected format per section:
    - MaterialIndex (int32)
    - FirstIndex (uint32)
    - NumTriangles (uint32)
    - MinVertexIndex (uint32)
    - MaxVertexIndex (uint32)
    - bEnableCollision (bool as int32)
    - bCastShadow (bool as int32)
    - bVisibleInRayTracing (bool as int32)
    - bForceOpaque (bool as int32)
    
    Returns:
        List of section dicts or None if parsing fails
    """
    if len(data) < 8:
        return None
    
    reader = _RenderDataReader(data)
    try:
        # Read array header
        num_elements = reader.read_uint32()
        array_flags = reader.read_uint32() if reader.remaining() >= 4 else 0
        
        if num_elements < 0 or num_elements > 1000:
            return None
        
        sections = []
        for _ in range(num_elements):
            if reader.remaining() < 28:
                return None
            
            section = {
                'material_index': reader.read_int32(),
                'first_index': reader.read_uint32(),
                'num_triangles': reader.read_uint32(),
                'min_vertex_index': reader.read_uint32(),
                'max_vertex_index': reader.read_uint32(),
                'enable_collision': reader.read_int32() != 0,
                'cast_shadow': reader.read_int32() != 0,
                'visible_in_ray_tracing': reader.read_int32() != 0,
                'force_opaque': reader.read_int32() != 0,
            }
            sections.append(section)
        
        return sections
        
    except Exception:
        return None


def find_render_data_bulk_data(pkg: Package, export_idx: int) -> Optional[List[Tuple[BulkDataEntry, bytes]]]:
    """Find and extract all relevant bulk data entries for FStaticMeshRenderData.
    
    Returns:
        List of (BulkDataEntry, data) tuples or None if not found
    """
    reader = pkg.get_export_data(export_idx)
    if reader is None:
        return None
    
    data = reader.data
    
    # Scan for bulk data entries
    bulk_entries = []
    pos = 0
    
    while pos < len(data) - 20:
        entry, next_pos = parse_bulk_data_entry(data, pos)
        if entry is None:
            pos += 4
            continue
        
        # Extract the bulk data
        bulk_data = extract_bulk_data(pkg, export_idx, entry)
        if bulk_data is not None:
            bulk_entries.append((entry, bulk_data))
        
        pos = next_pos
    
    if not bulk_entries:
        return None
    
    return bulk_entries