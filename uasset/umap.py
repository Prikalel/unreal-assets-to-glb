"""Level/Map parser for UE5 .umap files.

Parses .umap files (same format as .uasset) to extract actor placements
and static mesh references for level preview.
"""
import os
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field

from .package import Package
from .properties import read_properties


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LevelActor:
    """Represents an actor placed in a level."""
    name: str = ""               # Actor label or object name
    mesh_name: str = ""          # Static mesh asset name (e.g., "SM_Barrel1") or ""
    location: tuple = (0.0, 0.0, 0.0)   # (x, y, z) in UE space
    rotation: tuple = (0.0, 0.0, 0.0)   # (pitch, yaw, roll) in degrees
    scale: tuple = (1.0, 1.0, 1.0)      # (x, y, z)
    parent: str = ""             # Parent actor name or ""
    component_props: dict = field(default_factory=dict)  # Raw component properties


@dataclass
class LevelData:
    """Parsed level/map data."""
    map_name: str = ""
    actors: List[LevelActor] = field(default_factory=list)
    camera_location: tuple = (0.0, 0.0, 0.0)   # From PlayerStart/CameraActor
    camera_rotation: tuple = (0.0, 0.0, 0.0)   # (pitch, yaw, roll) in degrees
    has_camera: bool = False                     # Whether a camera actor was found


# ---------------------------------------------------------------------------
# FPackageIndex resolution
# ---------------------------------------------------------------------------

def resolve_package_index(pkg: Package, index: int) -> str:
    """Resolve FPackageIndex to a name.

    index > 0: export (index - 1), return export object name
    index < 0: import (-index - 1), return import object name
    index == 0: null, return ""
    """
    if index > 0:
        exp_idx = index - 1
        if 0 <= exp_idx < len(pkg.exports):
            return pkg.exports[exp_idx].object_name
    elif index < 0:
        imp_idx = -index - 1
        if 0 <= imp_idx < len(pkg.imports):
            return pkg.imports[imp_idx].object_name
    return ""


def resolve_import_path(pkg: Package, index: int) -> str:
    """Resolve FPackageIndex to an import's class_package (asset path).

    Only meaningful for imports (index < 0).
    Returns the class_package string (e.g., "/Game/TheAbandonedTunnel/Meshes/SM_Barrel1").
    """
    if index < 0:
        imp_idx = -index - 1
        if 0 <= imp_idx < len(pkg.imports):
            return pkg.imports[imp_idx].class_package
    return ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_vector(props: dict, key: str, default=(0.0, 0.0, 0.0)) -> tuple:
    """Extract (x, y, z) from a Vector struct property."""
    v = props.get(key)
    if v and isinstance(v, dict) and v.get("_type") == "Vector":
        return (v.get("x", 0.0), v.get("y", 0.0), v.get("z", 0.0))
    return default


def _get_rotator(props: dict, key: str, default=(0.0, 0.0, 0.0)) -> tuple:
    """Extract (pitch, yaw, roll) from a Rotator struct property."""
    v = props.get(key)
    if v and isinstance(v, dict) and v.get("_type") == "Rotator":
        return (v.get("pitch", 0.0), v.get("yaw", 0.0), v.get("roll", 0.0))
    return default


def _read_export_properties(pkg: Package, export_index: int) -> dict:
    """Read properties from an export, handling native serialization prefix.

    UE5 exports typically have a 1-byte SerializationControl prefix before
    property tags. We try multiple offsets to find where properties start,
    validating against script_serialization_end_offset when available.
    """
    entry = pkg.exports[export_index]
    data = pkg.get_export_data(export_index)
    if data is None:
        return {}

    # Use script serialization offsets if available and non-zero
    if entry.script_serialization_start_offset > 0:
        data.seek(entry.script_serialization_start_offset)
        return read_properties(data, pkg.name_map, pkg.file_version_ue5)

    # Expected end position (where "None" FName + trailing data ends)
    expected_end = entry.script_serialization_end_offset if entry.script_serialization_end_offset > 0 else None

    # Try common offsets where property tags typically start:
    # Offset 0: no native prefix (rare)
    # Offset 1: 1-byte SerializationControl prefix (most common)
    # Then scan further if needed (up to 256 bytes of native prefix)
    best_props = {}
    best_off = -1
    data_len = len(data.data)

    for off in range(min(data_len, 256)):
        data.seek(off)
        try:
            props = read_properties(data, pkg.name_map, pkg.file_version_ue5)
            end_pos = data.position()

            if len(props) == 0:
                continue

            # If we know the expected end position, validate against it
            if expected_end is not None and expected_end > 0:
                if end_pos == expected_end:
                    # Perfect match — this is the correct offset
                    return props
                # Close match (within 8 bytes — might be trailing "None" alignment)
                if abs(end_pos - expected_end) <= 8 and len(props) > len(best_props):
                    best_props = props
                    best_off = off
                    continue

            # No expected_end — use heuristic: most properties wins
            if len(props) > len(best_props):
                best_props = props
                best_off = off

            # If we found 2+ properties and no expected_end, good enough
            if len(props) >= 2 and expected_end is None:
                break

        except Exception:
            continue

    return best_props


# ---------------------------------------------------------------------------
# Level parser
# ---------------------------------------------------------------------------

def parse_level(filepath: str) -> LevelData:
    """Parse a .umap file and extract actor placements with mesh references.

    Args:
        filepath: Path to the .umap file

    Returns:
        LevelData with actors that have valid StaticMesh references
    """
    pkg = Package(filepath)
    map_name = os.path.splitext(os.path.basename(filepath))[0]

    # Phase 1: Read properties for ALL exports
    export_props: Dict[int, dict] = {}
    for i in range(pkg.export_count):
        class_name = pkg.get_export_class_name(i)
        # Only read properties for relevant classes to save time
        if class_name in ("StaticMeshActor", "StaticMeshComponent",
                          "SceneComponent", "ModelComponent", "Actor",
                          "PlayerStart", "CameraActor", "PlayerStartPIE",
                          "SpringArmComponent", "CameraComponent"):
            try:
                props = _read_export_properties(pkg, i)
                if props:
                    export_props[i] = props
            except Exception:
                pass

    # Phase 2: Build export_index → class_name map
    export_classes: Dict[int, str] = {}
    for i in range(pkg.export_count):
        export_classes[i] = pkg.get_export_class_name(i)

    # Phase 2.5: Find camera / player start actors for initial camera position
    camera_location = (0.0, 0.0, 0.0)
    camera_rotation = (0.0, 0.0, 0.0)
    has_camera = False
    _camera_classes = ("PlayerStart", "CameraActor", "PlayerStartPIE")

    for i in range(pkg.export_count):
        if export_classes.get(i) not in _camera_classes:
            continue
        actor_props = export_props.get(i, {})
        root_comp_idx = actor_props.get("RootComponent")
        if not isinstance(root_comp_idx, int) or root_comp_idx == 0:
            continue
        comp_export_idx = root_comp_idx - 1 if root_comp_idx > 0 else -1
        if comp_export_idx < 0 or comp_export_idx >= pkg.export_count:
            continue
        comp_props = export_props.get(comp_export_idx)
        if comp_props is None:
            try:
                comp_props = _read_export_properties(pkg, comp_export_idx)
            except Exception:
                comp_props = {}
        if comp_props:
            loc = _get_vector(comp_props, "RelativeLocation", (0.0, 0.0, 0.0))
            rot = _get_rotator(comp_props, "RelativeRotation", (0.0, 0.0, 0.0))
            camera_location = loc
            camera_rotation = rot
            has_camera = True
            break  # Use first camera found

    # Phase 3: Find StaticMeshActor exports and resolve their components
    actors: List[LevelActor] = []

    for i in range(pkg.export_count):
        if export_classes.get(i) != "StaticMeshActor":
            continue

        actor_props = export_props.get(i, {})
        actor_name = actor_props.get("ActorLabel",
                                     actor_props.get("Name",
                                                     pkg.exports[i].object_name))
        if isinstance(actor_name, bytes):
            actor_name = pkg.exports[i].object_name

        # Get RootComponent → FPackageIndex
        root_comp_idx = actor_props.get("RootComponent")
        if not isinstance(root_comp_idx, int) or root_comp_idx == 0:
            continue

        # Resolve RootComponent to export index
        comp_export_idx = root_comp_idx - 1 if root_comp_idx > 0 else -1
        if comp_export_idx < 0 or comp_export_idx >= pkg.export_count:
            continue

        # Read component properties (might already be cached)
        comp_props = export_props.get(comp_export_idx)
        if comp_props is None:
            try:
                comp_props = _read_export_properties(pkg, comp_export_idx)
            except Exception:
                comp_props = {}

        if not comp_props:
            continue

        # Extract transform
        location = _get_vector(comp_props, "RelativeLocation", (0.0, 0.0, 0.0))
        rotation = _get_rotator(comp_props, "RelativeRotation", (0.0, 0.0, 0.0))
        scale = _get_vector(comp_props, "RelativeScale3D", (1.0, 1.0, 1.0))

        # Extract StaticMesh reference
        mesh_fp_idx = comp_props.get("StaticMesh")
        mesh_name = ""
        if isinstance(mesh_fp_idx, int) and mesh_fp_idx != 0:
            mesh_name = resolve_package_index(pkg, mesh_fp_idx)

        # Skip actors without a valid mesh reference
        if not mesh_name:
            continue

        # Resolve parent from AttachParent
        parent_name = ""
        attach_parent_idx = comp_props.get("AttachParent")
        if isinstance(attach_parent_idx, int) and attach_parent_idx != 0:
            parent_name = resolve_package_index(pkg, attach_parent_idx)

        actor = LevelActor(
            name=actor_name,
            mesh_name=mesh_name,
            location=location,
            rotation=rotation,
            scale=scale,
            parent=parent_name,
            component_props=comp_props,
        )
        actors.append(actor)

    return LevelData(
        map_name=map_name,
        actors=actors,
        camera_location=camera_location,
        camera_rotation=camera_rotation,
        has_camera=has_camera,
    )
