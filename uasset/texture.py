"""Texture parser and PNG exporter for UE4.27 uncooked .uasset files.

Handles uncooked editor assets where texture source art is stored in
FEditorBulkData (Source.BulkData) in the package trailer, compressed
with Oodle inside an FCompressedBuffer, and optionally delta-encoded
with TSCF_UEDELTA.

Only exports base color (diffuse / _BC / _B) textures as the user requested.
"""
import os
import struct
from typing import Optional, List

import numpy as np
from PIL import Image

from .package import Package
from .reader import BinaryReader
from .properties import (
    _read_property_type_name,
    UE5_PROPERTY_TAG_COMPLETE_TYPE_NAME,
    UE5_PROPERTY_TAG_EXTENSION,
    TAG_HasArrayIndex, TAG_HasPropertyGuid, TAG_HasPropertyExtensions,
)
from .mesh import extract_trailer_payload, decompress_compressed_buffer

# ---------------------------------------------------------------------------
# ETextureSourceFormat enum
# ---------------------------------------------------------------------------
TSF_FORMAT_MAP = {
    'TSF_Invalid': -1,
    'TSF_G8': 0,
    'TSF_BGRA8': 1,
    'TSF_BGRE8': 2,
    'TSF_RGBA16': 3,
    'TSF_RGBA16F': 4,
    'TSF_G16': 5,
}

TSF_BPP = {-1: 0, 0: 1, 1: 4, 2: 8, 3: 8, 4: 8, 5: 2}

# ETextureSourceCompressionFormat enum
TSCF_MAP = {
    'TSCF_None': 0,
    'TSCF_PNG': 1,
    'TSCF_JPEG': 2,
    'TSCF_UEJPEG': 3,
    'TSCF_UEDELTA': 4,
}


# ---------------------------------------------------------------------------
# UEDELTA reverse transform
# ---------------------------------------------------------------------------
# Constants must match ImageCoreDelta.cpp exactly
_CUT_STRIDE_BYTES = 4096
_MIN_PIXELS_PER_CUT = 32768
_MAX_NUM_CUTS = 512


def _compute_num_cuts(num_pixels: int) -> int:
    if num_pixels <= _MIN_PIXELS_PER_CUT:
        return 1
    nc = num_pixels // _MIN_PIXELS_PER_CUT
    while nc > _MAX_NUM_CUTS:
        nc >>= 1
    return nc


def _compute_rows_per_cut(size_y: int, num_cuts: int) -> int:
    return (size_y + num_cuts - 1) // num_cuts


def _reverse_delta_tile_u8(arr: np.ndarray, start_y: int, height: int,
                           col_start: int, col_end: int) -> None:
    """Reverse uint8 delta for one tile in-place.

    Matches DeltaImage8 in ImageCoreDelta.cpp:
      Reverse: Out[Y] = In[Y] + Out[Y-1]  (uint8 wrapping)
    """
    for y in range(1, height):
        arr[start_y + y, col_start:col_end] = (
            arr[start_y + y, col_start:col_end].astype(np.int16)
            + arr[start_y + y - 1, col_start:col_end].astype(np.int16)
        ) % 256
    # Ensure uint8
    arr[start_y + 1:start_y + height, col_start:col_end] = \
        arr[start_y + 1:start_y + height, col_start:col_end].astype(np.uint8)


def _reverse_delta_tile_u16(arr: np.ndarray, start_y: int, height: int,
                            col_start: int, col_end: int) -> None:
    """Reverse uint16 delta for one tile in-place (with 0x8080 bias).

    Matches DeltaImage16 in ImageCoreDelta.cpp with UEDELTA_DO_BIAS=1:
      Reverse: Out[Y] = In[Y] + Out[Y-1] - 0x8080  (uint16 wrapping)
    """
    for y in range(1, height):
        row = arr[start_y + y, col_start:col_end].astype(np.int32)
        prev = arr[start_y + y - 1, col_start:col_end].astype(np.int32)
        arr[start_y + y, col_start:col_end] = \
            ((row + prev - 0x8080) & 0xFFFF).astype(np.uint16)


def undo_ue_delta(data: bytes, size_x: int, size_y: int, bpp: int,
                  element_size: int = 1) -> bytes:
    """Reverse the UEDELTA transform on texture source data.

    The image is split into tiles matching AddSplitStridedViewsForDelta,
    then each tile has row 0 stored as-is and subsequent rows as deltas.

    element_size=1 → uint8 delta (no bias), for G8, BGRA8
    element_size=2 → uint16 delta (0x8080 bias), for RGBA16, RGBA16F, G16
    """
    stride_bytes = size_x * bpp

    if element_size == 2:
        # Work with uint16 elements; cut calculations stay byte-based
        arr = np.frombuffer(data, dtype=np.uint16).copy().reshape(
            size_y, stride_bytes // 2)
        stride = stride_bytes // 2  # stride in uint16 elements
    else:
        arr = np.frombuffer(data, dtype=np.uint8).copy().reshape(
            size_y, stride_bytes)
        stride = stride_bytes

    if stride_bytes <= _CUT_STRIDE_BYTES:
        # No horizontal cuts, just vertical
        num_pixels = size_x * size_y
        num_cuts = _compute_num_cuts(num_pixels)
        rows_per_cut = _compute_rows_per_cut(size_y, num_cuts)
        num_cuts = (size_y + rows_per_cut - 1) // rows_per_cut

        for cut in range(num_cuts):
            start_y = cut * rows_per_cut
            height = min(rows_per_cut, size_y - start_y)
            if element_size == 2:
                _reverse_delta_tile_u16(arr, start_y, height, 0, stride)
            else:
                _reverse_delta_tile_u8(arr, start_y, height, 0, stride)
    else:
        # Horizontal cuts (stride > 4096)
        num_h_parts = (stride_bytes + _CUT_STRIDE_BYTES - 1) // _CUT_STRIDE_BYTES
        h_part_bytes = (stride_bytes + (num_h_parts // 2)) // num_h_parts
        h_part_bytes = (h_part_bytes + 63) & ~63  # align to 64
        h_part_pixels = h_part_bytes // bpp
        num_h_parts = (size_x + h_part_pixels - 1) // h_part_pixels

        for h_idx in range(num_h_parts):
            start_x = h_idx * h_part_pixels
            strip_width = min(h_part_pixels, size_x - start_x)
            strip_bytes = strip_width * bpp

            num_pixels = strip_width * size_y
            num_cuts = _compute_num_cuts(num_pixels)
            rows_per_cut = _compute_rows_per_cut(size_y, num_cuts)
            num_cuts = (size_y + rows_per_cut - 1) // rows_per_cut

            for cut in range(num_cuts):
                start_y = cut * rows_per_cut
                height = min(rows_per_cut, size_y - start_y)
                col_start_bytes = start_x * bpp
                if element_size == 2:
                    _reverse_delta_tile_u16(
                        arr, start_y, height,
                        col_start_bytes // 2,
                        (col_start_bytes + strip_bytes) // 2)
                else:
                    _reverse_delta_tile_u8(
                        arr, start_y, height,
                        col_start_bytes, col_start_bytes + strip_bytes)

    return arr.tobytes()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_fname(r: BinaryReader, name_map: List[str]) -> str:
    idx = r.read_int32()
    r.read_int32()  # instance number
    return name_map[idx] if 0 <= idx < len(name_map) else f"#{idx}"


def _extract_source_struct(r: BinaryReader, name_map: List[str],
                           file_version_ue5: int, struct_size: int) -> dict:
    """Parse FTextureSource struct to extract key fields."""
    use_ext = file_version_ue5 >= UE5_PROPERTY_TAG_EXTENSION
    end_pos = r.position() + struct_size
    result = {}

    while r.position() < end_pos - 8:
        name_idx = r.read_int32()
        _name_num = r.read_int32()
        name = name_map[name_idx] if 0 <= name_idx < len(name_map) else f"#{name_idx}"
        if name == "None":
            break

        type_name = _read_property_type_name(r, name_map)
        size = r.read_int32()
        flags = r.read_uint8()

        if flags & TAG_HasArrayIndex:
            r.skip(4)

        if name in ("SizeX", "SizeY", "NumSlices", "NumMips", "NumLayers") \
                and type_name == "IntProperty" and size == 4:
            result[name] = r.read_int32()
        elif name == "Format" and type_name == "ByteProperty":
            if size == 8:
                fmt_str = _parse_fname(r, name_map)
                result['Format'] = TSF_FORMAT_MAP.get(fmt_str, -1)
                result['FormatStr'] = fmt_str
            elif size == 1:
                result['Format'] = r.read_uint8()
            else:
                r.skip(size)
        elif name == "CompressionFormat" and type_name == "ByteProperty":
            if size == 8:
                cs = _parse_fname(r, name_map)
                result['CompressionFormat'] = TSCF_MAP.get(cs, -1)
                result['CompressionFormatStr'] = cs
            elif size == 1:
                result['CompressionFormat'] = r.read_uint8()
            else:
                r.skip(size)
        elif name == "bPNGCompressed" and type_name == "BoolProperty":
            result['bPNGCompressed'] = bool(flags & 0x10)
        else:
            r.skip(size)

        if flags & TAG_HasPropertyGuid:
            r.skip(16)
        if use_ext and (flags & TAG_HasPropertyExtensions):
            ext = r.read_uint8()
            if ext & 0x01:
                r.skip(1 + 4)

    if r.position() < end_pos:
        r.seek(end_pos)
    return result


# ---------------------------------------------------------------------------
# Texture2D class
# ---------------------------------------------------------------------------

class Texture2D:
    """Parsed uncooked Texture2D from a UE4.27 .uasset package."""

    __slots__ = ('width', 'height', 'format', 'format_str',
                 'compression_format', 'pixels')

    def __init__(self):
        self.width = 0
        self.height = 0
        self.format = -1
        self.format_str = '?'
        self.compression_format = -1
        self.pixels: Optional[np.ndarray] = None  # HxWxC uint8

    @classmethod
    def from_package(cls, pkg: Package) -> Optional['Texture2D']:
        """Parse texture from an uncooked Package."""
        tex_indices = pkg.find_exports_by_class("Texture2D")
        if not tex_indices:
            return None

        data_reader = pkg.get_export_data(tex_indices[0])
        if data_reader is None:
            return None

        tex = cls()

        # SerializationControl (1 byte)
        data_reader.read_uint8()

        # Parse properties to find Source struct
        use_ext = pkg.file_version_ue5 >= UE5_PROPERTY_TAG_EXTENSION
        source_data = {}

        while True:
            if not data_reader.can_read(8):
                break
            name_idx = data_reader.read_int32()
            _name_num = data_reader.read_int32()
            name = pkg.name_map[name_idx] if 0 <= name_idx < len(pkg.name_map) else f"#{name_idx}"
            if name == "None":
                break

            type_name = _read_property_type_name(data_reader, pkg.name_map)
            size = data_reader.read_int32()
            flags = data_reader.read_uint8()

            if flags & TAG_HasArrayIndex:
                data_reader.skip(4)

            if name == "Source" and "StructProperty" in type_name:
                source_data = _extract_source_struct(
                    data_reader, pkg.name_map, pkg.file_version_ue5, size)
            else:
                data_reader.skip(size)

            if flags & TAG_HasPropertyGuid:
                data_reader.skip(16)
            if use_ext and (flags & TAG_HasPropertyExtensions):
                ext = data_reader.read_uint8()
                if ext & 0x01:
                    data_reader.skip(1 + 4)

        # HasObjectGuid
        has_guid = data_reader.read_int32()
        if has_guid:
            data_reader.skip(16)

        # Skip native data: FStripDataFlags + FEditorBulkData + FStripDataFlags + bCooked
        data_reader.skip(2)          # FStripDataFlags (UTexture)
        data_reader.skip(4 + 16 + 20 + 8)  # FEditorBulkData header
        data_reader.skip(2)          # FStripDataFlags (UTexture2D)
        data_reader.skip(4)          # bCooked

        # Extract source info
        tex.width = source_data.get('SizeX', 0)
        tex.height = source_data.get('SizeY', 0)
        tex.format = source_data.get('Format', -1)
        tex.format_str = source_data.get('FormatStr', '?')
        tex.compression_format = source_data.get('CompressionFormat', -1)
        bpp = TSF_BPP.get(tex.format, 0)

        if tex.width <= 0 or tex.height <= 0 or tex.format < 0 or bpp == 0:
            return None

        # Extract and decompress trailer payload
        compressed_buffer = extract_trailer_payload(pkg.reader.data)
        if compressed_buffer is None:
            return None

        raw_data = decompress_compressed_buffer(compressed_buffer)
        if raw_data is None:
            return None

        needed = tex.width * tex.height * bpp
        if len(raw_data) < needed:
            return None

        # Check for PNG data
        bpng = source_data.get('bPNGCompressed', False)
        if bpng and raw_data[:4] == b'\x89PNG':
            from io import BytesIO
            try:
                img = Image.open(BytesIO(raw_data))
                tex.pixels = np.array(img.convert('RGBA'))
                tex.width = img.width
                tex.height = img.height
                tex.format = 1  # BGRA8 equivalent
                tex.format_str = 'TSF_BGRA8'
                return tex
            except Exception:
                return None

        # Apply UEDELTA reverse transform if needed
        if tex.compression_format == 4:  # TSCF_UEDELTA
            # RGBA16, RGBA16F, G16 use uint16 delta with 0x8080 bias
            elem_size = 2 if tex.format in (3, 4, 5) else 1
            raw_data = undo_ue_delta(raw_data, tex.width, tex.height, bpp,
                                     element_size=elem_size)

        # Decode pixel data based on format
        if tex.format == 1:  # TSF_BGRA8
            img_data = np.frombuffer(raw_data[:needed], dtype=np.uint8)
            img = img_data.reshape(tex.height, tex.width, 4).copy()
            img[:, :, [0, 2]] = img[:, :, [2, 0]]  # BGRA -> RGBA
            tex.pixels = img

        elif tex.format == 3:  # TSF_RGBA16
            img_data = np.frombuffer(raw_data[:needed], dtype=np.uint16)
            img = img_data.reshape(tex.height, tex.width, 4).copy()
            # Convert 16-bit RGBA to 8-bit RGBA
            tex.pixels = (img >> 8).astype(np.uint8)

        elif tex.format == 0:  # TSF_G8
            img_data = np.frombuffer(raw_data[:needed], dtype=np.uint8)
            gray = img_data.reshape(tex.height, tex.width)
            tex.pixels = np.stack([gray, gray, gray,
                                   np.full_like(gray, 255)], axis=2)

        elif tex.format == 5:  # TSF_G16
            img_data = np.frombuffer(raw_data[:needed], dtype=np.uint16)
            gray = (img_data.reshape(tex.height, tex.width) / 256).astype(np.uint8)
            tex.pixels = np.stack([gray, gray, gray,
                                   np.full_like(gray, 255)], axis=2)

        else:
            return None

        return tex


# ---------------------------------------------------------------------------
# PNG export
# ---------------------------------------------------------------------------

def export_png(texture: Texture2D, filepath: str) -> None:
    """Export a Texture2D as a PNG file."""
    if texture.pixels is None:
        raise ValueError("Texture has no pixel data")

    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

    if texture.pixels.shape[2] == 4:
        pil_img = Image.fromarray(texture.pixels, 'RGBA')
    else:
        pil_img = Image.fromarray(texture.pixels)

    pil_img.save(filepath)

