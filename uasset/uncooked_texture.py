"""Uncooked UE4.27 Texture2D (``.uasset``) parser.

Reads the *uncooked* editor texture format:

    UTexture2D::Serialize
      Super::Serialize                  -> object properties (old-format UProperty blob)
      FStripDataFlags                   -> 2 bytes (GlobalStrip + EditorStrip flags)
      Source.BulkData.Serialize         -> FByteBulkData header (20 bytes)
        + (editor-only) FTextureSource properties

The actual pixel payload is **not** inline.  For these uncooked packages the
``FByteBulkData`` payload is a UE4 *chunked-ZLIB* blob written to the package
trailer.  Empirically (confirmed across many textures) the blob begins exactly
at ``Package.bulk_data_start_offset`` and starts with the package-file tag
``0x9E2A83C1``.

The blob decompresses to the *source art*.  When ``bPNGCompressed`` is set
(always observed so far) the source art is a PNG file -- we decode it with
Pillow.  Otherwise the source art is raw pixel data laid out according to
``ETextureSourceFormat``.

This module mirrors the approach used by :mod:`uasset.uncooked_mesh` for bulk
data reading / decompression, but uses a *corrected* chunked-ZLIB decompressor
(see :func:`decompress_texture_bulk`) that does not wrongly treat
incompressible zlib-compressed chunks as "stored".

The entry point :func:`parse_uncooked_texture` returns a
:class:`uasset.texture.Texture2D` instance whose ``pixels`` are populated, so it
can be used as a drop-in primary path inside ``Texture2D.from_package``.
"""
from __future__ import annotations

import struct
import zlib
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PACKAGE_FILE_TAG = 0x9E2A83C1            # UE4 chunked-ZLIB signature (LE i64)
LOADING_COMPRESSION_CHUNK_SIZE = 1 << 17  # 128 KiB per chunk

# FByteBulkData flag bits (from BulkDataFlags.h) -- read for diagnostics only.
BULKDATA_PayloadAtEndOfFile = 1 << 0
BULKDATA_SerializeCompressedZLIB = 1 << 1
BULKDATA_SerializeCompressed = 1 << 4
BULKDATA_Size64Bit = 1 << 13
BULKDATA_Unused = 1 << 6


# ETextureSourceFormat enum values (Texture.h).  Used only when the source art
# is *raw* pixel data (bPNGCompressed == False).
ETextureSourceFormat_BYTES_PER_PIXEL: Dict[str, int] = {
    "TSF_Invalid": 0,
    "TSF_G8": 1,      # 8-bit grayscale
    "TSF_BGRA8": 4,   # 32-bit BGRA
    "TSF_BGRE8": 4,   # 32-bit BGRA (HDR exposure)
    "TSF_RGBA16": 8,  # 64-bit RGBA (16bpc)
    "TSF_RGBA16F": 8, # 64-bit RGBA float
    "TSF_RGBA8": 4,   # 32-bit RGBA
    "TSF_RGBE8": 4,   # 32-bit RGBE
    "TSF_G16": 2,     # 16-bit grayscale
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _u32(b: bytes, o: int) -> int:
    return struct.unpack_from("<I", b, o)[0]


def _i32(b: bytes, o: int) -> int:
    return struct.unpack_from("<i", b, o)[0]


def _i64(b: bytes, o: int) -> int:
    return struct.unpack_from("<q", b, o)[0]


def _read_fname(data: bytes, pos: int, name_map: List[str]) -> Tuple[str, int]:
    idx = struct.unpack_from("<i", data, pos)[0]
    num = struct.unpack_from("<i", data, pos + 4)[0]
    pos += 8
    if 0 <= idx < len(name_map):
        name = name_map[idx]
    else:
        name = "#%d" % idx
    if num:
        name = "%s_%d" % (name, num)
    return name, pos


def _skip_has_guid(data: bytes, pos: int) -> int:
    hg = data[pos]
    pos += 1
    if hg:
        pos += 16
    return pos


# ---------------------------------------------------------------------------
# Corrected UE4 chunked-ZLIB decompression
# ---------------------------------------------------------------------------

def _decompress_chunk(block: bytes, uncompressed_size: int) -> bytes:
    """Decompress a single ZLIB chunk.

    Texture source chunks use the *zlib container* (start ``78 9c``) while some
    other bulk data uses raw DEFLATE.  Try the zlib container first because it
    validates the header/checksum and avoids spurious raw-deflate success.

    A chunk whose compressed size happens to be >= its uncompressed size is NOT
    necessarily "stored" -- it can simply be incompressible data that zlib made
    slightly larger.  Only fall back to raw bytes when *both* decompressors
    genuinely fail.
    """
    # zlib container (validates adler32)
    try:
        out = zlib.decompress(block)
        if uncompressed_size <= 0 or len(out) == uncompressed_size:
            return out
        # Length mismatch -- keep trying, it may be a coincidental partial.
    except zlib.error:
        pass

    # raw DEFLATE (wbits = -15)
    try:
        out = zlib.decompress(block, -15)
        if uncompressed_size <= 0 or len(out) == uncompressed_size:
            return out
    except zlib.error:
        pass

    # stored / uncompressed
    if uncompressed_size > 0:
        return block[:uncompressed_size]
    return block


def decompress_texture_bulk(blob: bytes) -> bytes:
    """Decompress a UE4 chunked-ZLIB blob (the ``FByteBulkData`` payload).

    Layout (all little-endian int64):

        +0  0x9E2A83C1 (package-file tag)        [pkg_compressed]
        +8  chunk size (or tag for default)       [pkg_uncompressed]
        +16 total compressed size                 [summary_compressed]
        +24 total uncompressed size               [summary_uncompressed]
        +32 N * { chunk_compressed(8), chunk_uncompressed(8) }
        ...then the compressed chunk bytes.

    Returns the concatenated uncompressed bytes, or ``None`` if the blob does
    not look like a chunked-ZLIB stream.
    """
    if len(blob) < 32:
        return None
    pkg_compressed = _i64(blob, 0)
    if pkg_compressed != PACKAGE_FILE_TAG:
        return None

    pkg_uncompressed = _i64(blob, 8)
    chunk_size = LOADING_COMPRESSION_CHUNK_SIZE if pkg_uncompressed == PACKAGE_FILE_TAG else pkg_uncompressed
    if chunk_size <= 0:
        chunk_size = LOADING_COMPRESSION_CHUNK_SIZE

    summary_uncompressed = _i64(blob, 24)
    num_chunks = max(1, (summary_uncompressed + chunk_size - 1) // chunk_size)

    pos = 32
    headers: List[Tuple[int, int]] = []
    for _ in range(num_chunks):
        if pos + 16 > len(blob):
            return None
        c = _i64(blob, pos)
        u = _i64(blob, pos + 8)
        pos += 16
        headers.append((c, u))

    out = bytearray()
    for c_size, u_size in headers:
        if c_size <= 0 or pos + c_size > len(blob):
            # truncated stream -- bail
            break
        block = blob[pos:pos + c_size]
        pos += c_size
        out.extend(_decompress_chunk(block, u_size))
    return bytes(out)


# ---------------------------------------------------------------------------
# Old-format UProperty walker (object level + nested Source struct)
# ---------------------------------------------------------------------------

def _walk_object_properties(data: bytes, pos: int, name_map: List[str]
                            ) -> Tuple[Dict[str, Any], int]:
    """Walk the top-level object properties until the ``None`` terminator.

    Only the ``Source`` StructProperty is descended into; every other property
    is skipped.  Returns ``(source_dict, position_after_None)``.
    """
    source: Dict[str, Any] = {}
    n = len(data)
    while pos + 8 <= n:
        name, pos = _read_fname(data, pos, name_map)
        if name == "None":
            break
        if pos + 8 > n:
            break
        type_name, pos = _read_fname(data, pos, name_map)
        if pos + 8 > n:
            break
        size = _i32(data, pos); pos += 4
        _array_index = _i32(data, pos); pos += 4  # array index

        if type_name == "StructProperty":
            # struct name + GUID + has-guid flag precede the value bytes
            _struct_name, pos = _read_fname(data, pos, name_map)
            pos += 16  # struct GUID
            pos = _skip_has_guid(data, pos)
            value_start = pos
            if name == "Source":
                source, pos = _walk_struct_body(
                    data, value_start, name_map, value_start + size)
            # Always advance to the declared end of the struct value.
            pos = value_start + size
        elif type_name == "BoolProperty":
            pos += 1            # bool value
            pos = _skip_has_guid(data, pos)
        else:
            pos = _skip_has_guid(data, pos)
            pos += size

    return source, pos


def _walk_struct_body(data: bytes, start: int, name_map: List[str],
                      end: int) -> Tuple[Dict[str, Any], int]:
    """Walk a struct's inner property list (e.g. FTextureSource::Source)."""
    out: Dict[str, Any] = {}
    pos = start
    while pos + 8 <= end:
        name, pos = _read_fname(data, pos, name_map)
        if name == "None":
            break
        if pos + 8 > end:
            break
        type_name, pos = _read_fname(data, pos, name_map)
        if pos + 8 > end:
            break
        size = _i32(data, pos); pos += 4
        _array_index = _i32(data, pos); pos += 4
        value_start = pos

        if type_name == "BoolProperty":
            out[name] = data[pos] != 0
            pos = value_start + 1
            pos = _skip_has_guid(data, pos)
            continue
        if type_name == "StructProperty":
            _struct_name, pos = _read_fname(data, pos, name_map)
            pos += 16
            pos = _skip_has_guid(data, pos)
            value_start = pos
        elif type_name in ("ByteProperty", "EnumProperty"):
            _enum_name, pos = _read_fname(data, pos, name_map)
            pos = _skip_has_guid(data, pos)
            value_start = pos
        else:
            pos = _skip_has_guid(data, pos)
            value_start = pos

        raw = data[value_start:value_start + size]
        if type_name in ("IntProperty",) and size == 4:
            out[name] = _i32(raw, 0)
        elif type_name == "IntProperty" and size == 8:
            out[name] = _i64(raw, 0)
        elif type_name in ("ByteProperty", "EnumProperty"):
            if size == 8:
                enum_value_idx = _i32(raw, 0)
                out[name] = (name_map[enum_value_idx]
                             if 0 <= enum_value_idx < len(name_map)
                             else "?")
            elif size == 1:
                out[name] = raw[0]
            else:
                out[name] = raw
        else:
            out[name] = raw

        pos = value_start + size

    return out, pos


# ---------------------------------------------------------------------------
# Bulk-data locating & decoding
# ---------------------------------------------------------------------------

def _locate_payload_blob(pkg) -> bytes:
    """Return the chunked-ZLIB blob bytes for the texture source payload.

    For these uncooked packages the blob lives at the start of the package
    trailer (``bulk_data_start_offset``).  We verify the tag and, if the very
    first bytes do not match, search a short window forward for it (robustness
    against packages whose first trailer entry belongs to another export).
    """
    data = pkg.reader.data
    bsd = getattr(pkg, "bulk_data_start_offset", 0) or 0

    def _is_blob(at: int) -> bool:
        return (at + 4 <= len(data)
                and _u32(data, at) == (PACKAGE_FILE_TAG & 0xFFFFFFFF))

    if _is_blob(bsd):
        return data[bsd:]

    # Search forward for the tag.
    needle = struct.pack("<I", PACKAGE_FILE_TAG & 0xFFFFFFFF)
    idx = data.find(needle, bsd, bsd + 4096)
    if idx >= 0:
        return data[idx:]
    return data[bsd:]


def _decode_png(png_bytes: bytes) -> Tuple[int, int, np.ndarray]:
    """Decode a PNG blob into (width, height, HxWxC uint8 RGBA) pixels."""
    from PIL import Image as PILImage
    img = PILImage.open(BytesIO(png_bytes))
    # Normalise to a packed contiguous array.  Keep the source mode but expose
    # as RGBA so downstream consumers always get 4 channels.
    mode = img.mode
    if mode not in ("RGBA", "RGB", "L", "LA", "P"):
        img = img.convert("RGBA")
        mode = "RGBA"
    if mode == "P":
        img = img.convert("RGBA")
        mode = "RGBA"
    w, h = img.size
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    if arr.ndim == 2:                       # grayscale 'L'
        arr = np.stack([arr] * 3 + [np.full_like(arr, 255)], axis=-1)
    elif arr.shape[-1] == 1:                # 'LA'/'I;16' single-channel safety
        arr = np.repeat(arr, 3, axis=-1)
    return w, h, arr


def _decode_raw(raw: bytes, width: int, height: int,
                fmt_name: str) -> Optional[np.ndarray]:
    """Decode raw (non-PNG) source art into HxWx4 uint8 RGBA."""
    bpp = ETextureSourceFormat_BYTES_PER_PIXEL.get(fmt_name, 0)
    if bpp <= 0 or width <= 0 or height <= 0:
        return None
    expected = width * height * bpp
    if len(raw) < expected:
        return None
    buf = np.frombuffer(raw[:expected], dtype=np.uint8)

    if fmt_name in ("TSF_G8", "TSF_G16"):
        chan = 1 if fmt_name == "TSF_G8" else 2
        if fmt_name == "TSF_G8":
            g = buf.reshape(height, width)
            rgba = np.stack([g, g, g, np.full_like(g, 255)], axis=-1)
            return rgba
        # TSF_G16 -- take high byte as a rough 8-bit preview
        g16 = buf.reshape(height, width, 2)
        g8 = (g16[..., 1].astype(np.uint16) | (g16[..., 0].astype(np.uint16) << 0))
        g8 = (g8 >> 8).astype(np.uint8)
        return np.stack([g8, g8, g8, np.full_like(g8, 255)], axis=-1)

    if fmt_name == "TSF_BGRA8":
        pix = buf.reshape(height, width, 4)              # B,G,R,A
        rgba = pix[..., [2, 1, 0, 3]].copy()
        return rgba
    if fmt_name == "TSF_RGBA8":
        return buf.reshape(height, width, 4).copy()
    if fmt_name == "TSF_RGBE8":
        return buf.reshape(height, width, 4).copy()
    if fmt_name == "TSF_BGRE8":
        pix = buf.reshape(height, width, 4)
        rgba = pix[..., [2, 1, 0, 3]].copy()
        return rgba
    # 16-bit / float formats: crude 8-bit RGBA preview from low bytes.
    pix = buf.reshape(height, width, bpp)
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[..., :3] = pix[..., :3]
    rgba[..., 3] = 255
    return rgba


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def parse_uncooked_texture(pkg) -> Optional[Any]:
    """Parse an uncooked UE4.27 Texture2D package.

    Returns a :class:`uasset.texture.Texture2D` instance with ``width``,
    ``height``, ``format``/``format_str`` and ``pixels`` populated, or ``None``
    if this package does not contain a parseable Texture2D export.
    """
    from .texture import Texture2D

    tex_idx = None
    for i in range(pkg.export_count):
        try:
            if pkg.get_export_class_name(i) == "Texture2D":
                tex_idx = i
                break
        except Exception:
            continue
    if tex_idx is None:
        return None

    edata = pkg.get_export_data(tex_idx).data
    name_map = pkg.name_map

    # 1) Object properties -> FTextureSource (Source struct)
    source, pos = _walk_object_properties(edata, 0, name_map)
    width = source.get("SizeX")
    height = source.get("SizeY")
    fmt_name = source.get("Format")
    b_png = bool(source.get("bPNGCompressed", False))

    if not width or not height:
        return None

    # 2/3) FStripDataFlags (2 bytes) + FByteBulkData header (20 bytes) follow
    #      the property blob.  These are *informational only*: the editor-strip
    #      flags cannot be set (otherwise the Source payload would not exist)
    #      and the actual payload is located through the package trailer below
    #      (bulk_data_start_offset), NOT through OffsetInFile.  The property
    #      walker's end-position is therefore not trusted for the critical path
    #      (some textures expose extra properties that make it overshoot).
    if 0 <= pos + 22 <= len(edata):
        size_on_disk = _i32(edata, pos + 10)   # after 2 strip + 4 flags + 4 elem
    else:
        size_on_disk = -1

    # 4) Locate the source payload.  It begins at bulk_data_start_offset and is
    #    either a UE4 *chunked-ZLIB* stream (tag 0x9E2A83C1) wrapping the source
    #    art, or the source art stored *raw/uncompressed* (a PNG file or raw
    #    pixels).  Dispatch on the first 4 bytes.
    blob = _locate_payload_blob(pkg)
    payload: Optional[bytes] = None
    if len(blob) >= 4:
        first4 = _u32(blob, 0)
        if first4 == PACKAGE_FILE_TAG:
            payload = decompress_texture_bulk(blob)
        else:
            payload = blob          # raw/uncompressed bulk data
    if not payload:
        return None

    # 5) Decode the source art.
    pixels: Optional[np.ndarray] = None
    pw = ph = 0
    decoded_as_png = False

    if b_png and payload[:8] == b"\x89PNG\r\n\x1a\n":
        try:
            pw, ph, pixels = _decode_png(payload)
            decoded_as_png = True
        except Exception:
            pixels = None

    if pixels is None and not decoded_as_png:
        # Raw source art (bPNGCompressed == False) or a PNG that failed to load.
        pixels = _decode_raw(payload, width, height, fmt_name or "")
        if pixels is not None:
            pw, ph = width, height

    if pixels is None:
        return None

    # 6) Build a Texture2D result object.
    tex = Texture2D()
    tex.width = pw if pw else width
    tex.height = ph if ph else height
    tex.format = fmt_name or "TSF_Invalid"
    tex.format_str = fmt_name or "TSF_Invalid"
    tex.compression_format = "TC_Default"
    tex.pixels = pixels
    return tex


__all__ = [
    "PACKAGE_FILE_TAG",
    "ETextureSourceFormat_BYTES_PER_PIXEL",
    "decompress_texture_bulk",
    "parse_uncooked_texture",
]
