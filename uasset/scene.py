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
from .properties import read_properties

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

# ---------------------------------------------------------------------------
# Robust material-parameter parsing via the proven tagged-property reader
# ---------------------------------------------------------------------------
#
# The previous implementation located property data with a raw
# ``data.find(b'\xii\x00\x00\x00\x00\x00\x00\x00')`` scan and then walked the
# bytes with a hand-rolled UE5-format reader.  That scan produced false
# positives inside property *values*, and the hand-rolled reader used the
# UE5 ``FPropertyTypeName`` layout even though these are UE4.27 assets
# (``file_version_ue5 == 0`` → old/UE4 tag format).  Both problems caused
# ``EOFError`` when reading garbage past the end of the export.
#
# The functions below replace that with :func:`uasset.properties.read_properties`
# — the bounds-validated, exception-guarded reader already proven in
# mesh.py's ``_parse_static_materials_at``.  See MaterialInstance.cpp:3406 and
# MaterialInstance.h:155 in the UE4.27 source for the on-disk layout:
# ``TextureParameterValues`` is an ``ArrayProperty<StructProperty>`` whose
# elements are ``FTextureParameterValue`` (ParameterInfo/ParameterValue/
# ExpressionGUID), each serialised as a tagged property block.

def _resolve_package_index_name(pkg: Package, fpi) -> Optional[str]:
    """Resolve an FPackageIndex (``int``) to an import/export object name.

    Positive indices reference exports (1-based), negative indices reference
    imports (also 1-based, negated).  Returns ``None`` for null (0) or
    out-of-range indices.
    """
    if not isinstance(fpi, int) or fpi == 0:
        return None
    if fpi > 0:
        idx = fpi - 1
        if 0 <= idx < len(pkg.exports):
            return pkg.exports[idx].object_name
    else:
        idx = -fpi - 1
        if 0 <= idx < len(pkg.imports):
            return pkg.imports[idx].object_name
    return None


def _find_material_instance_export(pkg: Package) -> Optional[int]:
    """Return the export index of the MaterialInstance[Constant] export."""
    for i in range(pkg.export_count):
        cn = pkg.get_export_class_name(i)
        if cn in ('MaterialInstanceConstant', 'MaterialInstance'):
            return i
    return None


def _read_material_instance_properties(pkg: Package) -> Dict:
    """Read tagged properties of the MaterialInstance export.

    Uses the robust :func:`read_properties` reader.  Returns an empty dict if
    there is no MaterialInstance export, no serial data, or a parse error.
    """
    mi_idx = _find_material_instance_export(pkg)
    if mi_idx is None:
        return {}
    reader = pkg.get_export_data(mi_idx)
    if reader is None:
        return {}
    try:
        return read_properties(reader, pkg.name_map, pkg.file_version_ue5) or {}
    except Exception as e:  # pragma: no cover - defensive, reader is guarded
        logger.debug("read_properties failed for material export %d: %s",
                     mi_idx, e)
        return {}


def _parse_struct_array(pkg: Package, props: Dict, array_name: str) -> List[Dict]:
    """Parse a UE ``ArrayProperty<StructProperty>`` value into a list of dicts.

    *props* is the top-level property dict produced by
    :func:`read_properties`.  For an ``ArrayProperty`` that reader returns the
    raw array payload as ``bytes``.  That payload is::

        int32 Count
        <one inner-type property wrapper: a StructProperty describing the
            element type, whose *value* is the contiguous element region>
        Count × element (each a tagged-property block terminated by ``None``)

    The wrapper is consumed with :func:`read_properties`, which yields the
    element region as a ``bytes`` value; each element inside is then parsed
    with :func:`read_properties` as well.
    """
    arr_bytes = props.get(array_name)
    if not isinstance(arr_bytes, (bytes, bytearray)):
        return []
    try:
        sub = BinaryReader(bytes(arr_bytes))
        count = sub.read_int32()
    except Exception:
        return []
    if count < 0 or count > 4096:
        return []

    # Consume the single inner-type wrapper.  Its value (the element region)
    # comes back as bytes for an unrecognised struct type.
    try:
        wrapper = read_properties(sub, pkg.name_map, pkg.file_version_ue5) or {}
    except Exception:
        return []
    region = None
    for v in wrapper.values():
        if isinstance(v, (bytes, bytearray)):
            region = bytes(v)
            break
    if region is None:
        return []

    es = BinaryReader(region)
    elements: List[Dict] = []
    for _ in range(count):
        if not es.can_read(8):
            break
        try:
            elem = read_properties(es, pkg.name_map, pkg.file_version_ue5) or {}
        except Exception:
            break
        if elem:
            elements.append(elem)
    return elements


def _extract_parameter_name(sprops: Dict, pkg: Package) -> Optional[str]:
    """Extract the parameter name from a parsed parameter-value struct.

    Handles both serialisations seen in UE4.27 uncooked assets:

    * Older assets (FileVersion 514) store ``ParameterName`` (FName) — the
      ``UPROPERTY()`` deprecated field of ``FTextureParameterValue`` /
      ``FScalarParameterValue``.
    * Newer assets (FileVersion 517) store ``ParameterInfo``
      (``FMaterialParameterInfo``); its nested ``Name`` member carries the
      parameter name.
    """
    pname = sprops.get('ParameterName')
    if isinstance(pname, str) and pname:
        return pname
    pinfo = sprops.get('ParameterInfo')
    if isinstance(pinfo, (bytes, bytearray)):
        try:
            sub = read_properties(BinaryReader(bytes(pinfo)),
                                  pkg.name_map, pkg.file_version_ue5) or {}
            nm = sub.get('Name')
            if isinstance(nm, str) and nm:
                return nm
        except Exception:
            pass
    return None


def _parse_texture_parameter_values(pkg: Package) -> Dict[str, str]:
    """Parse ``TextureParameterValues`` from the MaterialInstance export.

    Uses the robust :func:`read_properties` reader (the same one proven in
    mesh.py's ``_parse_static_materials_at``).  Returns
    ``{parameter_name: texture_object_name}`` for every texture-parameter
    override present in the material instance export data.
    """
    props = _read_material_instance_properties(pkg)
    if not props:
        return {}
    elements = _parse_struct_array(pkg, props, 'TextureParameterValues')
    result: Dict[str, str] = {}
    for sprops in elements:
        pname = _extract_parameter_name(sprops, pkg)
        tex_name = _resolve_package_index_name(pkg, sprops.get('ParameterValue'))
        if pname and tex_name:
            result[pname] = tex_name
    return result


def _parse_scalar_parameter_values(pkg: Package) -> Dict[str, float]:
    """Parse ``ScalarParameterValues`` from the MaterialInstance export.

    Returns ``{parameter_name: float_value}`` for every scalar parameter
    override present.  Kept for completeness; not used by base-color binding.
    """
    props = _read_material_instance_properties(pkg)
    if not props:
        return {}
    elements = _parse_struct_array(pkg, props, 'ScalarParameterValues')
    result: Dict[str, float] = {}
    for sprops in elements:
        pname = _extract_parameter_name(sprops, pkg)
        pval = sprops.get('ParameterValue')
        if pname and isinstance(pval, (int, float)):
            result[pname] = float(pval)
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


def _lookup_texmap(texture_name: str, tex_map: Dict[str, str]) -> Optional[str]:
    """Look up *texture_name* in *tex_map* (exact, then case-insensitive)."""
    if not texture_name:
        return None
    mapped = tex_map.get(texture_name)
    if mapped is not None:
        return mapped
    low = texture_name.lower()
    for en, ep in tex_map.items():
        if en.lower() == low:
            return ep
    return None


# Filename tokens that identify a base-color / albedo / diffuse texture.
# Per the UE material naming convention; applied only to the texture
# *binding* fallback, never to uasset parsing.
_BASE_COLOR_TOKENS = frozenset({'basecolor', 'bc', 'albedo', 'diffuse'})


def _name_looks_base_color(name: str) -> bool:
    """Return True if *name* looks like a base-color texture.

    Matches the documented UE naming convention (object name contains a
    ``_BaseColor``/``_BC``/``_Albedo``/``_Diffuse`` token).
    """
    if not name:
        return False
    tokens = set(name.lower().replace('-', '_').split('_'))
    return bool(tokens & _BASE_COLOR_TOKENS)


def _find_base_color_texture_in_imports(pkg: Package,
                                        tex_map: Dict[str, str]
                                        ) -> Optional[str]:
    """Scan a package's import table for a base-color texture.

    Covers materials that import their base-color ``Texture2D`` directly
    (e.g. ``T_Cardboard_Flat_Atlas_BaseColor``) rather than referencing it
    through ``TextureParameterValues``.  Only returns textures present in
    *tex_map*.
    """
    for imp in pkg.imports:
        if imp.class_name != 'Texture2D':
            continue
        if not _name_looks_base_color(imp.object_name):
            continue
        mapped = _lookup_texmap(imp.object_name, tex_map)
        if mapped is not None:
            return mapped
    return None


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
    """Resolve a base-colour texture for *material_name*.

    Layered binding strategy (UE naming conventions are used only as a
    *binding* fallback, never for uasset parsing):

    1. **Primary** — parse ``TextureParameterValues`` and pick the entry whose
       parameter name matches a base-color slot (``BaseColor``/``Albedo``/
       ``Diffuse``/…).
    2. **Import-table fallback** — scan the material's import table for a
       ``Texture2D`` whose name contains ``_BaseColor``/``_BC``/``_Albedo``/
       ``_Diffuse`` and is present in *tex_map*.
    3. **Parent chain** — repeat 1–2 walking ``Mi_*`` → ``M_*`` parents
       (uniform recursion; the master ``Material`` is also walked so its
       ``BaseColor`` expression / imports can be resolved).
    4. **Master-material expression** — last resort, trace the
       ``MaterialEditorOnlyData`` ``BaseColor`` input to a texture sample.

    Args:
        material_name:  Material asset name.
        uasset_index:   Index from :func:`_build_uasset_index`.
        tex_map:        Mapping of texture asset name → exported identifier.
        _depth:         Recursion guard (max 8).

    Returns:
        The resolved texture identifier from *tex_map*, or ``None``.
    """
    if _depth > 8:
        return None

    filepath = uasset_index.get(material_name)
    if filepath is None:
        return None

    try:
        pkg = Package(filepath)
    except Exception as e:
        logger.debug("Failed to open material package for '%s': %s",
                     material_name, e)
        return None

    # --- 1. Primary: TextureParameterValues base-color parameter ---------
    tex_params = _parse_texture_parameter_values(pkg)
    logger.debug("Material '%s' texture params: %s", material_name, tex_params)
    for param_name, tex_asset_name in tex_params.items():
        if param_name in _BASE_COLOR_PARAM_NAMES:
            mapped = _lookup_texmap(tex_asset_name, tex_map)
            if mapped is not None:
                logger.debug("  '%s': base color param '%s' -> '%s'",
                             material_name, param_name, tex_asset_name)
                return mapped

    # --- 2. Fallback: base-color texture in the import table -------------
    mapped = _find_base_color_texture_in_imports(pkg, tex_map)
    if mapped is not None:
        logger.debug("  '%s': base color via import table -> '%s'",
                     material_name, mapped)
        return mapped

    # --- 3. Walk the parent material chain (uniform recursion) -----------
    parent_name = _find_parent_material_name(pkg, material_name)
    if parent_name is not None:
        result = _get_base_color_texture_from_material(
            parent_name, uasset_index, tex_map, _depth + 1)
        if result is not None:
            return result

    # --- 4. Master-material BaseColor expression -------------------------
    tex_name = _get_base_color_from_master_material(pkg)
    if tex_name is not None:
        mapped = _lookup_texmap(tex_name, tex_map)
        if mapped is not None:
            logger.debug("  '%s': master-material BaseColor -> '%s'",
                         material_name, tex_name)
            return mapped

    return None


def _get_base_color_factor_from_material(material_name: str,
                                         uasset_index: Dict[str, str],
                                         _depth: int = 0) -> Optional[List[float]]:
    """Return an RGBA base-colour tint (list of 4 floats, 0..1, *linear*) taken
    from the material's ``VectorParameterValues``.

    Final layer of the binding strategy.  Procedural metals / plastics whose
    colour is a shader constant (a VectorParameter) have *no* albedo texture,
    so without this they export as plain white.  UE stores the authored colour
    as an ``FLinearColor`` in linear space, which is exactly what glTF's
    ``baseColorFactor`` expects, so no gamma conversion is applied.

    Selection: only non-white, non-black colour params are candidates; a name
    containing BaseColor/Colour/Color/Albedo/Diffuse/Tint is preferred, and a
    more saturated colour wins ties.  The parent chain is walked when the
    current material yields nothing.  Returns ``None`` (=> stay white) for
    brushed metals whose only colour params are white masks.

    The alpha channel is forced to 1.0 (opaque) — tint-only materials are
    opaque; transparency always comes through a texture / alpha-mask.
    """
    import math
    if _depth > 8:
        return None
    filepath = uasset_index.get(material_name)
    if filepath is None:
        return None
    try:
        pkg = Package(filepath)
    except Exception:
        return None

    mi_idx = None
    for i in range(pkg.export_count):
        try:
            if pkg.get_export_class_name(i) in (
                    'MaterialInstanceConstant', 'MaterialInstance'):
                mi_idx = i
                break
        except Exception:
            continue
    if mi_idx is None:
        return None

    try:
        r = pkg.get_export_data(mi_idx)
        props = read_properties(r, pkg.name_map, pkg.file_version_ue5) or {}
        elems = _parse_struct_array(pkg, props, 'VectorParameterValues')
    except Exception:
        elems = []

    best = None  # (score, name, (r,g,b,a))
    for e in elems:
        try:
            pname = (_extract_parameter_name(e, pkg) or '').lower()
            val = e.get('ParameterValue')
            if not isinstance(val, (bytes, bytearray)) or len(val) < 16:
                continue
            cr, cg, cb, _ca = struct.unpack_from('<4f', val, 0)
        except Exception:
            continue
        if not all(math.isfinite(c) for c in (cr, cg, cb)):
            continue
        mx, mn = max(cr, cg, cb), min(cr, cg, cb)
        if mx < 1e-3:                                   # pure black = unused
            continue
        is_white = (abs(cr - 1.0) < 0.02 and abs(cg - 1.0) < 0.02
                    and abs(cb - 1.0) < 0.02)
        if is_white:                                    # white mask, skip
            continue
        score = 0.0
        for key in ('basecolor', 'albedo', 'diffuse', 'colour', 'color',
                    'tint', 'base'):
            if key in pname:
                score += 10.0
                break
        score += (mx - mn) * 2.0                        # prefer saturated
        if best is None or score > best[0]:
            best = (score, pname, (cr, cg, cb))

    if best is not None:
        cr, cg, cb = best[2]
        clamp = lambda v: max(0.0, min(1.0, v))
        return [clamp(cr), clamp(cg), clamp(cb), 1.0]

    parent = _find_parent_material_name(pkg, material_name)
    if parent is not None:
        return _get_base_color_factor_from_material(
            parent, uasset_index, _depth + 1)
    return None


def _get_base_color_from_master_material(pkg: Package) -> Optional[str]:
    """Parse MaterialEditorOnlyData to find the BaseColor texture.

    Walks the chain:
      MaterialEditorOnlyData -> BaseColor (FColorMaterialInput)
        -> Expression (FPackageIndex -> MaterialExpressionTextureSample export)
          -> Texture (FPackageIndex -> Texture2D import)

    Returns the Texture2D import object name, or None.
    """
    from .properties import (
        TAG_HasArrayIndex, TAG_HasPropertyGuid, TAG_HasPropertyExtensions,
    )
    from .package import UE5_PROPERTY_TAG_EXTENSION

    # Find the MaterialEditorOnlyData export
    eod_idx = None
    for i in range(pkg.export_count):
        if pkg.get_export_class_name(i) == 'MaterialEditorOnlyData':
            eod_idx = i
            break
    if eod_idx is None:
        return None

    reader = pkg.get_export_data(eod_idx)
    if reader is None:
        return None

    use_ext = pkg.file_version_ue5 >= UE5_PROPERTY_TAG_EXTENSION

    # Skip SerializationControl byte
    reader.seek(1)

    # Read tagged properties until we find BaseColor
    while reader.can_read(8):
        name_idx = reader.read_int32()
        _name_num = reader.read_int32()
        name = pkg.name_map[name_idx] if 0 <= name_idx < len(pkg.name_map) else f"#{name_idx}"

        if name == 'None':
            break

        # Read type tree
        total_nodes = 1
        type_names: List[str] = []
        i = 0
        while i < total_nodes:
            t_idx = reader.read_int32()
            _t_num = reader.read_int32()
            inner_count = reader.read_int32()
            t_name = pkg.name_map[t_idx] if 0 <= t_idx < len(pkg.name_map) else f"#{t_idx}"
            type_names.append(t_name)
            total_nodes += inner_count
            i += 1

        size = reader.read_int32()
        flags = reader.read_uint8()

        if flags & TAG_HasArrayIndex:
            reader.skip(4)

        value_start = reader.position()

        if name == 'BaseColor' and len(type_names) >= 2 and type_names[0] == 'StructProperty':
            # Parse FColorMaterialInput (FMaterialInput base + UseConstant + Constant)
            # FMaterialInput:
            #   Expression: FPackageIndex (int32)
            #   OutputIndex: int32
            #   InputName: FName (8 bytes)
            #   Mask, MaskR, MaskG, MaskB, MaskA: 5 x int32
            # FColorMaterialInput:
            #   UseConstant: uint32
            #   Constant: FColor (4 bytes)
            if size < 40:
                reader.skip(size)
            else:
                expression_fpi = reader.read_int32()
                # Skip remaining fields: OutputIndex(4) + InputName(8) +
                # Mask(4)*5 + UseConstant(4) + Constant(4) = 40 bytes total
                # We already read 4 (expression_fpi), skip the rest
                reader.skip(size - 4)

                # Resolve Expression FPackageIndex to an export
                if expression_fpi > 0:
                    exp_idx = expression_fpi - 1
                    if 0 <= exp_idx < len(pkg.exports):
                        cn = pkg.get_export_class_name(exp_idx)
                        if cn == 'MaterialExpressionTextureSample':
                            tex_name = _get_texture_from_expression(pkg, exp_idx)
                            if tex_name is not None:
                                return tex_name
        else:
            reader.skip(size)

        if flags & TAG_HasPropertyGuid:
            reader.skip(16)
        if use_ext and (flags & TAG_HasPropertyExtensions):
            ext = reader.read_uint8()
            if ext & 0x01:
                reader.skip(1 + 4)

        # Ensure alignment
        remaining = size - (reader.position() - value_start)
        if remaining > 0:
            reader.skip(remaining)

    return None


def _get_texture_from_expression(pkg: Package, exp_idx: int) -> Optional[str]:
    """Parse a MaterialExpressionTextureSample export to find its Texture reference.

    The export data layout (after SerializationControl byte) is native-serialized.
    The Texture FPackageIndex is located by scanning for references to Texture2D imports.

    Returns the Texture2D import object name, or None.
    """
    import struct as _struct

    reader = pkg.get_export_data(exp_idx)
    if reader is None:
        return None

    data = reader.data

    # Build set of Texture2D import FPackageIndex values
    tex_imports = {}
    for imp_i, imp in enumerate(pkg.imports):
        if imp.class_name == 'Texture2D':
            fpi = -(imp_i + 1)
            tex_imports[fpi] = imp.object_name

    if not tex_imports:
        return None

    # Scan for FPackageIndex values that reference Texture2D imports.
    # The Texture property in MaterialExpressionTextureSample is typically
    # the first Texture2D reference after the UObject header.
    for offset in range(0, len(data) - 3):
        val = _struct.unpack_from('<i', data, offset)[0]
        if val in tex_imports:
            return tex_imports[val]

    return None

