"""Asset-based texture resolution for UE5 mesh export.

Provides utilities to walk the UE asset import chain
(mesh → material → texture) and resolve base-color textures
for GLB export.  Used by main.py during the export pipeline.

Legacy preview code (MatplotlibPreviewer, PygletPreviewer, build_preview_scene,
show_preview) has been removed in favour of the browser-based preview
(preview_server.py + preview.html).
"""
import os
import logging
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Asset-based texture resolution
# ---------------------------------------------------------------------------

_BASE_COLOR_SUFFIXES = ('_BC', '_B', '_D', '_bc', '_b', '_d')


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
    from .package import Package
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


def _get_base_color_texture_from_material(material_name: str,
                                          uasset_index: Dict[str, str],
                                          tex_map: Dict[str, str],
                                          _depth: int = 0) -> Optional[str]:
    """Resolve a base-colour texture name from a material's import chain.

    Opens the material's .uasset package, looks for ``Texture2D`` imports
    whose name ends with a base-colour suffix (``_BC``, ``_B``, ``_D``, …),
    and returns the mapped texture name from *tex_map*.

    If no texture is found directly, follows the parent
    ``MaterialInstanceConstant`` import chain recursively (up to 8 levels).

    Args:
        material_name:  Material asset name.
        uasset_index:   Index from :func:`_build_uasset_index`.
        tex_map:        Mapping of texture asset name → exported name / identifier.

    Returns:
        The resolved texture name from *tex_map*, or ``None``.
    """
    if _depth > 8:
        return None
    from .package import Package
    filepath = uasset_index.get(material_name)
    if filepath is None:
        return None
    try:
        pkg = Package(filepath)
        parent_name = None
        for imp in pkg.imports:
            if imp.class_name == 'Texture2D':
                tex_name = imp.object_name
                if any(tex_name.endswith(sfx) for sfx in _BASE_COLOR_SUFFIXES):
                    if tex_name in tex_map:
                        return tex_map[tex_name]
                    for en, ep in tex_map.items():
                        if en.lower() == tex_name.lower():
                            return ep
            # Remember the first MaterialInstanceConstant parent
            if (parent_name is None
                    and imp.class_name in ('MaterialInstanceConstant', 'Material')
                    and imp.object_name != material_name):
                parent_name = imp.object_name
        # No texture found directly — try the parent material
        if parent_name is not None:
            return _get_base_color_texture_from_material(
                parent_name, uasset_index, tex_map, _depth + 1)
        return None
    except Exception as e:
        logger.debug(f"Failed to read material package for '{material_name}': {e}")
        return None
