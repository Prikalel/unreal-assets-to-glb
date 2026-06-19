"""Bulk data extraction utilities for UE4 packages."""

import struct
from typing import Optional, Tuple, List
from .package import Package


BULKDATA_PayloadAtEndOfFile = 1 << 0
BULKDATA_SerializeCompressedZLIB = 1 << 1
BULKDATA_ForceSingleElementSerialization = 1 << 2
BULKDATA_SingleUse = 1 << 3
BULKDATA_Unused = 1 << 5
BULKDATA_ForceInlinePayload = 1 << 6
BULKDATA_ForceStreamPayload = 1 << 7
BULKDATA_PayloadInSeperateFile = 1 << 8
BULKDATA_Force_NOT_InlinePayload = 1 << 10
BULKDATA_OptionalPayload = 1 << 11
BULKDATA_MemoryMappedPayload = 1 << 12
BULKDATA_Size64Bit = 1 << 13


class BulkDataEntry:
    """Represents a bulk data structure in export serial data."""
    
    def __init__(self):
        self.flags = 0
        self.element_count = 0
        self.size_on_disk = 0
        self.offset_in_file = 0
        self.header_size = 0
        self.data_location = None  # 'inline' or 'end_of_file' or None
        self.actual_offset = 0


def find_bulk_data_in_export(data: bytes) -> List[BulkDataEntry]:
    """Find all bulk data structures in export serial data.
    
    Returns a list of BulkDataEntry objects.
    """
    entries = []
    pos = 0
    
    while pos < len(data) - 20:
        try:
            # Try to parse as bulk data header
            flags = struct.unpack_from('<I', data, pos)[0]
            
            # Valid bulk data flags are typically small values (bit flags)
            # Check if it looks like a reasonable flags value
            if flags >= 0x10000000:  # Too large to be flags
                pos += 4
                continue
            
            # Determine 64-bit flag
            size_64bit = (flags & BULKDATA_Size64Bit) != 0
            
            if pos + 8 > len(data):
                pos += 4
                continue
            
            # Read element count
            if size_64bit:
                element_count = struct.unpack_from('<Q', data, pos + 4)[0]
                size_on_disk_offset = pos + 12
            else:
                element_count = struct.unpack_from('<I', data, pos + 4)[0]
                size_on_disk_offset = pos + 8
            
            # Check if element count is reasonable
            if element_count < 0 or element_count >= 1e9:  # Too large or negative
                pos += 4
                continue
            
            if size_on_disk_offset + 8 > len(data):
                pos += 4
                continue
            
            # Read size on disk
            if size_64bit:
                size_on_disk = struct.unpack_from('<Q', data, size_on_disk_offset)[0]
                offset_offset = size_on_disk_offset + 8
            else:
                size_on_disk = struct.unpack_from('<I', data, size_on_disk_offset)[0]
                offset_offset = size_on_disk_offset + 4
            
            # Check if size is reasonable
            if size_on_disk < 0 or size_on_disk >= 1e10:  # Too large or negative
                pos += 4
                continue
            
            if offset_offset + 8 > len(data):
                pos += 4
                continue
            
            # Read offset in file
            offset_in_file = struct.unpack_from('<q', data, offset_offset)[0]
            
            # Check if payload is inline
            payload_inline = (flags & BULKDATA_PayloadAtEndOfFile) == 0
            
            entry = BulkDataEntry()
            entry.flags = flags
            entry.element_count = element_count
            entry.size_on_disk = size_on_disk
            entry.offset_in_file = offset_in_file
            entry.header_size = offset_offset + 8
            
            if payload_inline:
                entry.data_location = 'inline'
                entry.actual_offset = entry.header_size
            else:
                entry.data_location = 'end_of_file'
            
            entries.append(entry)
            
            # Skip ahead to avoid re-finding
            pos = entry.header_size
            
        except Exception as e:
            # Skip this position if parsing fails
            pos += 4
    
    return entries


def extract_bulk_data(pkg: Package, export_idx: int, bulk_entry: BulkDataEntry) -> Optional[bytes]:
    """Extract bulk data from a package.
    
    Args:
        pkg: Package object
        export_idx: Index of the export containing this bulk data
        bulk_entry: BulkDataEntry describing the bulk data
        
    Returns:
        The bulk data bytes, or None if extraction fails
    """
    export = pkg.exports[export_idx]
    
    if bulk_entry.data_location == 'inline':
        # Data is inline in the export serial data
        reader = pkg.get_export_data(export_idx)
        if reader is None:
            return None
        
        data = reader.data
        offset = bulk_entry.actual_offset
        size = bulk_entry.size_on_disk if bulk_entry.size_on_disk > 0 else bulk_entry.element_count
        
        if offset + size > len(data):
            return None
        
        return data[offset:offset + size]
    
    elif bulk_entry.data_location == 'end_of_file':
        # Data is at the end of the file, apply offset fixup
        fixup = pkg.bulk_data_start_offset
        actual_offset = bulk_entry.offset_in_file + fixup
        
        # Check if offset is valid
        if actual_offset < 0 or actual_offset >= len(pkg.reader.data):
            return None
        
        size = bulk_entry.size_on_disk
        
        if actual_offset + size > len(pkg.reader.data):
            return None
        
        return pkg.reader.data[actual_offset:actual_offset + size]
    
    return None


def find_mesh_description_bulk_data(pkg: Package, export_idx: int) -> Optional[Tuple[BulkDataEntry, bytes]]:
    """Find and extract the mesh description bulk data from a StaticMesh export.
    
    This function looks for bulk data that:
    - Has compression enabled (ZLIB)
    - Has a reasonable size (likely to contain mesh data)
    - Is not marked as optional or unused
    
    Returns:
        Tuple of (BulkDataEntry, data) or None if not found
    """
    reader = pkg.get_export_data(export_idx)
    if reader is None:
        return None
    
    data = reader.data
    bulk_entries = find_bulk_data_in_export(data)
    
    if not bulk_entries:
        return None
    
    # Look for the most likely candidate: compressed, not unused, reasonable size
    candidates = []
    
    for entry in bulk_entries:
        # Skip unused or optional data
        if entry.flags & BULKDATA_Unused:
            continue
        if entry.flags & BULKDATA_OptionalPayload:
            continue
        
        # Prefer compressed data
        has_compression = (entry.flags & BULKDATA_SerializeCompressedZLIB) != 0
        
        # Calculate actual data size (use size_on_disk if available, otherwise element_count)
        data_size = entry.size_on_disk if entry.size_on_disk > 0 else entry.element_count
        
        # Filter for reasonable sizes (mesh data is typically > 1KB and < 100MB for single mesh)
        if data_size < 1024 or data_size > 100 * 1024 * 1024:
            continue
        
        score = 0
        if has_compression:
            score += 100  # Compression is a strong indicator
        if entry.data_location == 'inline':
            score += 50   # Inline is more common for cooked packages
        if 10 * 1024 <= data_size <= 10 * 1024 * 1024:
            score += 30   # Reasonable mesh size range
        
        candidates.append((score, entry))
    
    if not candidates:
        return None
    
    # Sort by score (descending)
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Try to extract the best candidate
    for score, entry in candidates:
        extracted = extract_bulk_data(pkg, export_idx, entry)
        if extracted is not None:
            return (entry, extracted)
    
    return None