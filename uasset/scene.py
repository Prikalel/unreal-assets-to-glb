"""Asset-based texture resolution for UE5 mesh export.

Provides utilities to walk the UE asset import chain
(mesh → material → texture) and resolve base-color textures
for GLB export.  Used by main.py during the export pipeline.

Texture resolution works by parsing ``TextureParameterValues`` from
material instance export data and walking the parent chain
(MI → parent MI → … → master Material) until a base-color
parameter override is found.

Legacy preview code (MatplotlibPreviewer, PygletPreviewer, build_preview_scene,
show_preview) has been removed in favour of the browser-based preview
(preview_server.py + preview.html).
"""
import os
import struct
import logging
from typing import Optional, Dict, List, Tuple

from .package import Package
from .reader import BinaryReader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base-color parameter names (matched against *parameter* name, not texture
# filename).  These are the names UE material authors use for the diffuse /
# albedo slot inside TextureParameterValues.
# ---------------------------------------------------------------------------
_BASE_COLOR_PARAM_NAMES = frozenset({
    'BaseTexture', 'BaseColor', 'Diffuse', 'Albedo',
    'Base Color', 'DiffuseTexture',
})

# ---------------------------------------------------------------------------
# Low-level helpers for reading UE property headers from export data
# ---------------------------------------------------------------------------

def _read_fname(r: BinaryReader, name_map: List[str]) -> str:
    """Read an FName (index + number) and resolve to string."""
    idx = r.read_int32()
    _num = r.read_int32()
    return name_map[idx] if 0 <= idx < len(name_map) else f"#{idx}"


def _read_type_tree(r: BinaryReader, name_map: List[str]) -> List[str]:
    """Read the UE5 property type tree (used in property tags)."""
    total_nodes = 1
    names: List[str] = []
    i = 0
    while i < total_nodes:
        idx = r.read_int32()
        _num = r.read_int32()
        inner_count = r.read_int32()
        name = name_map[idx] if 0 <= idx < len(name_map) else f"#{idx}"
        names.append(name)
        total_nodes += inner_count
        i += 1
    return names


def _skip_property_extensions(r: BinaryReader) -> None:
    """Skip UE5 property extension data."""
    flags = r.read_uint8()
    if flags & 0x01:
        r.skip(5)


def _read_property_header(r: BinaryReader, name_map: List[str],
                          use_extensions: bool
                          ) -> Optional[Tuple[str, List[str], int, int]]:
    """Read a UE5 property tag header.

    Returns ``(name, type_names, size, flags)`` or ``None`` when the
    sentinel ``None`` FName is reached (end of property list).
    """
    name = _read_fname(r, name_map)
    if name == 'None':
        return None
    type_names = _read_type_tree(r, name_map)
    size = r.read_int32()
    flags = r.read_uint8()
    if flags & 0x01:   # HasArrayIndex
        r.skip(4)
    if flags & 0x02:   # HasPropertyGuid
        r.skip(16)
    if use_extensions and (flags & 0x04):
        _skip_property_extensions(r)
    return name, type_names, size, flags


# ---------------------------------------------------------------------------
# TextureParameterValues / ScalarParameterValues parser
# ---------------------------------------------------------------------------

def _find_property_offset(data: bytes, name_map: List[str],
                          prop_name: str) -> int:
    """Scan *data* for the FName of *prop_name* and return its byte offset.

    The FName is encoded as ``<i index><i 0>``.  Returns ``-1`` if not
    found.
    """
    try:
        name_idx = name_map.index(prop_name)
    except ValueError:
        return -1
    target = struct.pack('<ii', name_idx, 0)
    return data.find(target)


def _parse_texture_parameter_values(pkg: Package) -> Dict[str, str]:
    """Parse ``TextureParameterValues`` from the first export of *pkg*.

    Returns ``{parameter_name: texture_asset_name}`` for every texture
    parameter override present in the material instance export data.
    """
    reader = pkg.get_export_data(0)
    if reader is None:
        return {}

    data = reader.data
    offset = _find_property_offset(data, pkg.name_map,
                                   'TextureParameterValues')
    if offset < 0:
        return {}

    use_ext = pkg.file_version_ue5 >= 1011  # UE5_PROPERTY_TAG_EXTENSION
    reader.seek(offset)

    hdr = _read_property_header(reader, pkg.name_map, use_ext)
    if hdr is None:
        return {}

    _prop_name, type_names, _size, _flags = hdr
    if 'ArrayProperty' not in type_names:
        return {}

    arr_count = reader.read_int32()
    result: Dict[str, str] = {}

    for _ in range(arr_count):
        param_name: Optional[str] = None
        param_value: Optional[str] = None

        # Read inner properties until 'None' sentinel
        while True:
            prop_hdr = _read_property_header(reader, pkg.name_map, use_ext)
            if prop_hdr is None:
                break

            p_name, p_types, p_size, p_flags = prop_hdr
            value_start = reader.position()

            if p_name == 'ParameterInfo' and p_types[0] == 'StructProperty':
                # FMaterialParameterInfo — read inner props
                while True:
                    inner_hdr = _read_property_header(reader, pkg.name_map,
                                                      use_ext)
                    if inner_hdr is None:
                        break
                    i_name, i_types, i_size, _i_flags = inner_hdr
                    if i_types[0] == 'NameProperty' and i_size >= 8:
                        val = _read_fname(reader, pkg.name_map)
                        if i_name == 'Name':
                            param_name = val
                    else:
                        reader.skip(i_size)

            elif p_name == 'ParameterValue' and p_types[0] == 'ObjectProperty':
                pkg_idx = reader.read_int32()
                if pkg_idx > 0:
                    exp_idx = pkg_idx - 1
                    if 0 <= exp_idx < len(pkg.exports):
                        param_value = pkg.exports[exp_idx].object_name
                elif pkg_idx < 0:
                    imp_idx = -pkg_idx - 1
                    if 0 <= imp_idx < len(pkg.imports):
                        param_value = pkg.imports[imp_idx].object_name

            elif p_name == 'ExpressionGUID' and p_types[0] == 'StructProperty':
                reader.skip(16)  # FGuid raw bytes

            else:
                reader.skip(p_size)

            # Ensure we consumed exactly *p_size* bytes of value data
            remaining = p_size - (reader.position() - value_start)
            if remaining > 0:
                reader.skip(remaining)

        if param_name is not None and param_value is not None:
            result[param_name] = param_value

    return result


def _parse_scalar_parameter_values(pkg: Package) -> Dict[str, float]:
    """Parse ``ScalarParameterValues`` from the first export of *pkg*.

    Returns ``{parameter_name: float_value}`` for every scalar parameter
    override present in the material instance export data.
    """
    reader = pkg.get_export_data(0)
    if reader is None:
        return {}

    data = reader.data
    offset = _find_property_offset(data, pkg.name_map,
                                   'ScalarParameterValues')
    if offset < 0:
        return {}

    use_ext = pkg.file_version_ue5 >= 1011
    reader.seek(offset)

    hdr = _read_property_header(reader, pkg.name_map, use_ext)
    if hdr is None:
        return {}

    _prop_name, type_names, _size, _flags = hdr
    if 'ArrayProperty' not in type_names:
        return {}

    arr_count = reader.read_int32()
    result: Dict[str, float] = {}

    for _ in range(arr_count):
        param_name: Optional[str] = None
        param_value: Optional[float] = None

        while True:
            prop_hdr = _read_property_header(reader, pkg.name_map, use_ext)
            if prop_hdr is None:
                break

            p_name, p_types, p_size, _p_flags = prop_hdr
            value_start = reader.position()

            if p_name == 'ParameterInfo' and p_types[0] == 'StructProperty':
                while True:
                    inner_hdr = _read_property_header(reader, pkg.name_map,
                                                      use_ext)
                    if inner_hdr is None:
                        break
                    i_name, i_types, i_size, _i_flags = inner_hdr
                    if i_types[0] == 'NameProperty' and i_size >= 8:
                        val = _read_fname(reader, pkg.name_map)
                        if i_name == 'Name':
                            param_name = val
                    else:
                        reader.skip(i_size)

            elif p_name == 'ParameterValue' and p_types[0] == 'FloatProperty':
                param_value = reader.read_float()

            elif p_name == 'ExpressionGUID' and p_types[0] == 'StructProperty':
                reader.skip(16)

            else:
                reader.skip(p_size)

            remaining = p_size - (reader.position() - value_start)
            if remaining > 0:
                reader.skip(remaining)

        if param_name is not None and param_value is not None:
            result[param_name] = param_value

    return result


# ---------------------------------------------------------------------------
# Asset index & material resolution
# ---------------------------------------------------------------------------

def _build_uasset_index(input_dir: str) -> Dict[str, str]:
    """Scan *input_dir* for .uasset files and return ``{name: filepath}``.

    Recursively walks ``Content/`` (or *input_dir* itself if no Content/
    sub-directory exists) and builds a lookup table used by the other
    resolution functions.

    Args:
        input_dir: Project root containing ``Content/`` or the Content
            directory itself.

    Returns:
        Dictionary mapping asset name (without extension) to its full path.
    """
    index: Dict[str, str] = {}
    content_dir = os.path.join(input_dir, 'Content')
    if not os.path.isdir(content_dir):
        content_dir = input_dir
    for root, _dirs, files in os.walk(content_dir):
        for f in files:
            if f.endswith('.uasset'):
                name = os.path.splitext(f)[0]
                index[name] = os.path.join(root, f)
    return index


def _get_material_names_from_mesh(mesh_name: str,
                                  uasset_index: Dict[str, str]) -> List[str]:
    """Return the ordered list of material asset names referenced by a mesh.

    Opens the mesh's .uasset package and collects all imports whose class
    is ``MaterialInstanceConstant`` or ``Material``.

    Args:
        mesh_name:      Static mesh asset name (no extension).
        uasset_index:   Index from :func:`_build_uasset_index`.

    Returns:
        List of material object names, possibly empty.
    """
    filepath = uasset_index.get(mesh_name)
    if filepath is None:
        return []
    try:
        pkg = Package(filepath)
        return [imp.object_name for imp in pkg.imports
                if imp.class_name in ('MaterialInstanceConstant', 'Material')]
    except Exception as e:
        logger.debug(f"Failed to read mesh package for '{mesh_name}': {e}")
        return []


def _find_parent_material_name(pkg: Package,
                               material_name: str) -> Optional[str]:
    """Find the parent material name from a package's import table.

    Looks for a ``MaterialInstanceConstant`` or ``Material`` import whose
    ``object_name`` differs from *material_name*.
    """
    for imp in pkg.imports:
        if (imp.class_name in ('MaterialInstanceConstant', 'Material')
                and imp.object_name != material_name):
            return imp.object_name
    return None


def _get_base_color_texture_from_material(material_name: str,
                                          uasset_index: Dict[str, str],
                                          tex_map: Dict[str, str],
                                          _depth: int = 0) -> Optional[str]:
    """Resolve a base-colour texture by walking the material parent chain.

    At each level the function parses ``TextureParameterValues`` from the
    material instance's export data and checks whether any parameter name
    matches a known base-color slot (``BaseTexture``, ``BaseColor``,
    ``Diffuse``, ``Albedo``, etc.).  If found, the corresponding texture
    asset name is looked up in *tex_map* and returned.

    If no override exists at the current level the function follows the
    parent ``MaterialInstanceConstant`` import chain recursively (up to
    8 levels deep).

    Args:
        material_name:  Material asset name.
        uasset_index:   Index from :func:`_build_uasset_index`.
        tex_map:        Mapping of texture asset name → exported name / identifier.
        _depth:         Recursion guard (max 8).

    Returns:
        The resolved texture name from *tex_map*, or ``None``.
    """
    if _depth > 8:
        return None

    filepath = uasset_index.get(material_name)
    if filepath is None:
        return None

    try:
        pkg = Package(filepath)
    except Exception as e:
        logger.debug(f"Failed to open material package for '{material_name}': {e}")
        return None

    # Parse TextureParameterValues at this level
    tex_params = _parse_texture_parameter_values(pkg)
    logger.debug(f"Material '{material_name}' texture params: {tex_params}")

    # Look for a base-color parameter name
    for param_name, tex_asset_name in tex_params.items():
        if param_name in _BASE_COLOR_PARAM_NAMES:
            # Resolve via tex_map (case-insensitive fallback)
            mapped = tex_map.get(tex_asset_name)
            if mapped is None:
                for en, ep in tex_map.items():
                    if en.lower() == tex_asset_name.lower():
                        mapped = ep
                        break
            if mapped is not None:
                logger.debug(
                    f"  '{material_name}': base color param '{param_name}' "
                    f"→ texture '{tex_asset_name}'")
                return mapped

    # Not found — walk to parent
    parent_name = _find_parent_material_name(pkg, material_name)
    if parent_name is not None:
        # Check if parent is a master Material (not MaterialInstanceConstant)
        parent_is_master = False
        for imp in pkg.imports:
            if imp.object_name == parent_name and imp.class_name == 'Material':
                parent_is_master = True
                break

        if not parent_is_master:
            result = _get_base_color_texture_from_material(
                parent_name, uasset_index, tex_map, _depth + 1)
            if result is not None:
                return result

    # Last resort: if we found ANY texture parameter that maps into
    # tex_map, use the first one as a fallback
    for _param_name, tex_asset_name in tex_params.items():
        mapped = tex_map.get(tex_asset_name)
        if mapped is None:
            for en, ep in tex_map.items():
                if en.lower() == tex_asset_name.lower():
                    mapped = ep
                    break
        if mapped is not None:
            logger.debug(
                f"  '{material_name}': fallback to first available "
                f"texture '{tex_asset_name}'")
            return mapped

    return None

