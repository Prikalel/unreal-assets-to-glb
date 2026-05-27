"""Level/Map parser for UE5 .umap files.

Parses .umap files (same format as .uasset) to extract actor placements
and static mesh references for level preview.
"""
import os
import math
from typing import List, Optional, Dict, Tuple, Set
from dataclasses import dataclass, field

import numpy as np

from .package import Package
from .properties import read_properties
from .transform import rotator_to_matrix


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LevelActor:
    """Represents an actor placed in a level."""
    name: str = ""               # Actor label or object name
    mesh_name: str = ""          # Static mesh asset name (e.g., "SM_Barrel1") or ""
    location: tuple = (0.0, 0.0, 0.0)   # (x, y, z) local/relative in UE space
    rotation: tuple = (0.0, 0.0, 0.0)   # (pitch, yaw, roll) local in degrees
    scale: tuple = (1.0, 1.0, 1.0)      # (x, y, z) local
    parent: str = ""             # Parent actor name or ""
    component_props: dict = field(default_factory=dict)  # Raw component properties
    # World-space transforms (computed from parent chain)
    world_location: tuple = (0.0, 0.0, 0.0)
    world_rotation: tuple = (0.0, 0.0, 0.0)
    world_scale: tuple = (1.0, 1.0, 1.0)


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
# UE transform math (pure UE space, no coordinate conversion)
# ---------------------------------------------------------------------------

def _make_ue_transform(location: tuple, rotation: tuple, scale: tuple) -> np.ndarray:
    """Build 4x4 transform matrix in UE space (no coordinate conversion)."""
    R = rotator_to_matrix(*rotation)
    s = np.array([scale[0], scale[1], scale[2]])

    T = np.eye(4, dtype=float)
    T[:3, :3] = R * s[np.newaxis, :]
    T[0, 3] = location[0]
    T[1, 3] = location[1]
    T[2, 3] = location[2]

    return T


def _matrix_to_rotator(R: np.ndarray) -> tuple:
    """Extract UE FRotator (degrees) from 3x3 rotation matrix."""
    sp = -R[2, 0]
    if abs(sp) < 0.99999:
        pitch = math.asin(sp)
        yaw = math.atan2(R[1, 0], R[0, 0])
        roll = math.atan2(R[2, 1], R[2, 2])
    else:
        pitch = math.copysign(math.pi / 2, sp)
        yaw = math.atan2(-R[0, 1], R[1, 1])
        roll = 0.0
    return (math.degrees(pitch), math.degrees(yaw), math.degrees(roll))


def _decompose_ue_transform(M: np.ndarray) -> tuple:
    """Decompose 4x4 UE transform matrix to (location, rotation, scale)."""
    loc = (float(M[0, 3]), float(M[1, 3]), float(M[2, 3]))

    # Extract scale as column norms
    sx = float(np.linalg.norm(M[:3, 0]))
    sy = float(np.linalg.norm(M[:3, 1]))
    sz = float(np.linalg.norm(M[:3, 2]))

    # Avoid zero scale
    sx = max(sx, 1e-7)
    sy = max(sy, 1e-7)
    sz = max(sz, 1e-7)

    # Normalize columns to get rotation matrix
    R = M[:3, :3] / np.array([sx, sy, sz])[np.newaxis, :]

    rot = _matrix_to_rotator(R)
    return loc, rot, (sx, sy, sz)


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
                    return props
                if abs(end_pos - expected_end) <= 8 and len(props) > len(best_props):
                    best_props = props
                    best_off = off
                    continue

            # No expected_end — use heuristic: most properties wins
            if len(props) > len(best_props):
                best_props = props
                best_off = off

            if len(props) >= 2 and expected_end is None:
                break

        except Exception:
            continue

    return best_props


# ---------------------------------------------------------------------------
# LevelInstance helpers
# ---------------------------------------------------------------------------

def _extract_world_asset_path(world_asset_value, pkg: Package) -> str:
    """Extract UE asset path from a WorldAsset property value.

    Handles SoftObjectProperty (tuple), string, and ObjectProperty (int) values.
    Returns the asset path like '/Game/Maps/Sublevels/L_SomeLevel' or "".
    """
    path = ""
    if isinstance(world_asset_value, tuple):
        # SoftObjectProperty: (path, subpath)
        path = world_asset_value[0] if world_asset_value else ""
    elif isinstance(world_asset_value, str):
        path = world_asset_value
    elif isinstance(world_asset_value, int) and world_asset_value != 0:
        # ObjectProperty: FPackageIndex → resolve import path
        path = resolve_import_path(pkg, world_asset_value)

    if not path:
        return ""

    # Handle format like "World'/Game/Maps/Sublevels/L_SomeLevel.L_SomeLevel'"
    if "'" in path:
        start = path.index("'") + 1
        end = path.rindex("'")
        if start < end:
            path = path[start:end]

    # Remove trailing ".Name" duplicate (e.g., ".L_SomeLevel")
    last_slash = path.rfind('/')
    dot_pos = path.rfind('.')
    if dot_pos > last_slash and dot_pos > 0:
        path = path[:dot_pos]

    return path


def _resolve_umap_filesystem_path(asset_path: str, current_umap_path: str) -> str:
    """Convert a UE asset path to a filesystem path.

    Converts '/Game/Maps/Sublevels/L_SomeLevel' to
    '<game_root>/Content/Maps/Sublevels/L_SomeLevel.umap'.
    Returns "" if the path cannot be resolved.
    """
    if not asset_path.startswith('/Game/'):
        return ""

    # Find the game root by looking for 'Content' directory in current path
    current_abs = os.path.normpath(os.path.abspath(current_umap_path))
    parts = current_abs.replace('\\', '/').split('/')

    content_idx = -1
    for idx, part in enumerate(parts):
        if part.lower() == 'content':
            content_idx = idx
            break

    if content_idx < 0:
        return ""

    game_root = '/'.join(parts[:content_idx])
    relative = asset_path[6:]  # Remove '/Game/'

    fs_path = os.path.join(game_root, 'Content', relative + '.umap')
    return os.path.normpath(fs_path)


# ---------------------------------------------------------------------------
# Level parser
# ---------------------------------------------------------------------------

def parse_level(filepath: str, parent_transform: Optional[np.ndarray] = None,
                 _visited: Optional[Set[str]] = None,
                 _depth: int = 0) -> LevelData:
    """Parse a .umap file and extract actor placements with mesh references.

    Args:
        filepath: Path to the .umap file

    Returns:
        LevelData with actors that have valid StaticMesh references
    """
    pkg = Package(filepath)
    map_name = os.path.splitext(os.path.basename(filepath))[0]

    # Circular-reference guard and recursion depth limit
    if _depth > 10:
        return LevelData(map_name=map_name)
    if _visited is None:
        _visited = set()
    current_abs = os.path.normpath(os.path.abspath(filepath))
    _visited.add(current_abs)

    # Phase 1: Read properties for ALL exports
    export_props: Dict[int, dict] = {}
    for i in range(pkg.export_count):
        class_name = pkg.get_export_class_name(i)
        if class_name in ("StaticMeshActor", "StaticMeshComponent",
                          "SceneComponent", "ModelComponent", "Actor",
                          "PlayerStart", "CameraActor", "PlayerStartPIE",
                          "SpringArmComponent", "CameraComponent",
                          "LevelInstance"):
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

    # Phase 2.5: Build component info map for parent-chain transform resolution.
    comp_info: Dict[int, dict] = {}  # comp_export_idx → {loc, rot, scl, parent_comp_idx}
    for i in range(pkg.export_count):
        class_name = export_classes.get(i)
        if class_name in ("StaticMeshComponent", "SceneComponent",
                          "ModelComponent", "SpringArmComponent",
                          "CameraComponent"):
            props = export_props.get(i)
            if props is None:
                continue
            loc = _get_vector(props, "RelativeLocation", (0.0, 0.0, 0.0))
            rot = _get_rotator(props, "RelativeRotation", (0.0, 0.0, 0.0))
            scl = _get_vector(props, "RelativeScale3D", (1.0, 1.0, 1.0))
            parent_comp_idx = -1
            attach_parent = props.get("AttachParent")
            if isinstance(attach_parent, int) and attach_parent > 0:
                parent_comp_idx = attach_parent - 1
            comp_info[i] = {
                'loc': loc, 'rot': rot, 'scl': scl,
                'parent_comp_idx': parent_comp_idx,
            }

    # Build component children map: parent_comp_idx → [child_comp_idx, ...]
    comp_children_map: Dict[int, List[int]] = {}
    for ci, info in comp_info.items():
        pidx = info['parent_comp_idx']
        if pidx >= 0:
            comp_children_map.setdefault(pidx, []).append(ci)

    # Build component export index → owning actor label map.
    comp_to_actor: Dict[int, str] = {}
    _actor_classes = ("StaticMeshActor", "Actor", "PlayerStart",
                      "CameraActor", "PlayerStartPIE")
    for i in range(pkg.export_count):
        if export_classes.get(i) not in _actor_classes:
            continue
        actor_props = export_props.get(i, {})
        root_comp_idx = actor_props.get("RootComponent")
        if isinstance(root_comp_idx, int) and root_comp_idx > 0:
            comp_idx = root_comp_idx - 1
            name = actor_props.get("ActorLabel",
                                   actor_props.get("Name",
                                                   pkg.exports[i].object_name))
            if isinstance(name, bytes):
                name = pkg.exports[i].object_name
            comp_to_actor[comp_idx] = name

    # World-transform cache
    _wt_cache: Dict[int, np.ndarray] = {}

    def _get_world_transform(comp_idx: int, depth: int = 0) -> np.ndarray:
        if depth > 32:
            return np.eye(4)
        if comp_idx in _wt_cache:
            return _wt_cache[comp_idx]
        info = comp_info.get(comp_idx)
        if info is None:
            result = np.eye(4)
            _wt_cache[comp_idx] = result
            return result
        local = _make_ue_transform(info['loc'], info['rot'], info['scl'])
        pidx = info['parent_comp_idx']
        if pidx >= 0 and pidx in comp_info:
            parent_world = _get_world_transform(pidx, depth + 1)
            result = parent_world @ local
        else:
            result = local
        _wt_cache[comp_idx] = result
        return result

    # Phase 2.7: Find camera / player start
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
        world_mat = _get_world_transform(comp_export_idx)
        world_loc, world_rot, _ = _decompose_ue_transform(world_mat)
        camera_location = world_loc
        camera_rotation = world_rot
        has_camera = True
        break

    # Phase 3: Find StaticMeshActor exports (standalone actors)
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

        if not comp_props:
            continue

        location = _get_vector(comp_props, "RelativeLocation", (0.0, 0.0, 0.0))
        rotation = _get_rotator(comp_props, "RelativeRotation", (0.0, 0.0, 0.0))
        scale = _get_vector(comp_props, "RelativeScale3D", (1.0, 1.0, 1.0))

        mesh_fp_idx = comp_props.get("StaticMesh")
        mesh_name = ""
        if isinstance(mesh_fp_idx, int) and mesh_fp_idx != 0:
            mesh_name = resolve_package_index(pkg, mesh_fp_idx)

        if not mesh_name:
            continue

        world_mat = _get_world_transform(comp_export_idx)
        world_loc, world_rot, world_scl = _decompose_ue_transform(world_mat)

        parent_actor_name = ""
        attach_parent_idx = comp_props.get("AttachParent")
        if isinstance(attach_parent_idx, int) and attach_parent_idx > 0:
            parent_comp_idx = attach_parent_idx - 1
            parent_actor_name = comp_to_actor.get(parent_comp_idx, "")

        actor = LevelActor(
            name=actor_name,
            mesh_name=mesh_name,
            location=location,
            rotation=rotation,
            scale=scale,
            parent=parent_actor_name,
            component_props=comp_props,
            world_location=world_loc,
            world_rotation=world_rot,
            world_scale=world_scl,
        )
        actors.append(actor)

    # Phase 4: Process composed actors (Actor exports with mesh sub-components).
    # These are empty Actor exports that serve as containers for multiple
    # StaticMeshComponents attached in a hierarchy.
    _composed_names: Set[str] = set(a.name for a in actors)

    for i in range(pkg.export_count):
        if export_classes.get(i) != "Actor":
            continue
        actor_props = export_props.get(i, {})
        actor_label = actor_props.get("ActorLabel", pkg.exports[i].object_name)
        if isinstance(actor_label, bytes):
            actor_label = pkg.exports[i].object_name

        root_comp_idx = actor_props.get("RootComponent")
        if not isinstance(root_comp_idx, int) or root_comp_idx == 0:
            continue
        root_comp_export_idx = root_comp_idx - 1

        # Walk the component tree from root, find all StaticMeshComponents
        _name_counter: Dict[str, int] = {}

        def _walk_composed_tree(comp_idx: int, parent_name: str, depth: int = 0):
            if depth > 32:
                return
            children = comp_children_map.get(comp_idx, [])
            for child_idx in children:
                child_cn = export_classes.get(child_idx)
                if child_cn not in ("StaticMeshComponent", "SceneComponent",
                                    "ModelComponent"):
                    continue

                child_props = export_props.get(child_idx)
                if child_props is None:
                    continue

                mesh_fp_idx = child_props.get("StaticMesh")
                mesh_name = ""
                if isinstance(mesh_fp_idx, int) and mesh_fp_idx != 0:
                    mesh_name = resolve_package_index(pkg, mesh_fp_idx)

                comp_obj_name = pkg.exports[child_idx].object_name

                if mesh_name:
                    world_mat = _get_world_transform(child_idx)
                    world_loc, world_rot, world_scl = _decompose_ue_transform(world_mat)

                    info = comp_info.get(child_idx)
                    loc = info['loc'] if info else (0.0, 0.0, 0.0)
                    rot = info['rot'] if info else (0.0, 0.0, 0.0)
                    scl = info['scl'] if info else (1.0, 1.0, 1.0)

                    # Ensure unique name
                    base_name = f"{actor_label}/{comp_obj_name}"
                    if base_name in _name_counter:
                        _name_counter[base_name] += 1
                        unique_name = f"{base_name}_{_name_counter[base_name]}"
                    else:
                        _name_counter[base_name] = 0
                        unique_name = base_name

                    # Make sure it's globally unique
                    if unique_name in _composed_names:
                        idx = 2
                        while f"{unique_name}_{idx}" in _composed_names:
                            idx += 1
                        unique_name = f"{unique_name}_{idx}"

                    _composed_names.add(unique_name)

                    actors.append(LevelActor(
                        name=unique_name,
                        mesh_name=mesh_name,
                        location=loc,
                        rotation=rot,
                        scale=scl,
                        parent=parent_name,
                        component_props=child_props,
                        world_location=world_loc,
                        world_rotation=world_rot,
                        world_scale=world_scl,
                    ))

                    _walk_composed_tree(child_idx, unique_name, depth + 1)
                else:
                    # No mesh — recurse with same parent name
                    _walk_composed_tree(child_idx, parent_name, depth + 1)

        _walk_composed_tree(root_comp_export_idx, actor_label)

    # Phase 5: Process LevelInstance actors (recursive sub-levels).
    # Record how many actors came from this level before recursing.
    local_actor_count = len(actors)

    for i in range(pkg.export_count):
        if export_classes.get(i) != "LevelInstance":
            continue

        li_props = export_props.get(i, {})
        if not li_props:
            continue

        # Extract WorldAsset path
        world_asset = li_props.get("WorldAsset")
        if not world_asset:
            continue

        asset_path = _extract_world_asset_path(world_asset, pkg)
        if not asset_path:
            continue

        # Resolve to filesystem path
        sub_umap_path = _resolve_umap_filesystem_path(asset_path, filepath)
        if not sub_umap_path:
            continue

        sub_abs = os.path.normpath(os.path.abspath(sub_umap_path))
        if not os.path.isfile(sub_abs):
            print(f"[umap] LevelInstance WorldAsset not found: {sub_umap_path}")
            continue

        if sub_abs in _visited:
            print(f"[umap] LevelInstance circular reference skipped: {sub_umap_path}")
            continue

        # Get LevelInstance world transform from its RootComponent
        root_comp_idx = li_props.get("RootComponent")
        li_world = np.eye(4)
        if isinstance(root_comp_idx, int) and root_comp_idx > 0:
            comp_export_idx = root_comp_idx - 1
            li_world = _get_world_transform(comp_export_idx)

        # Compose: parent_transform @ li_world_transform
        pt = parent_transform if parent_transform is not None else np.eye(4)
        child_parent_transform = pt @ li_world

        # Recursively parse sub-level
        try:
            sub_level = parse_level(sub_umap_path, child_parent_transform,
                                    _visited, _depth + 1)
        except Exception as exc:
            print(f"[umap] Failed to parse sub-level {sub_umap_path}: {exc}")
            continue

        # Get LevelInstance actor name for parent reference
        li_name = li_props.get("ActorLabel", pkg.exports[i].object_name)
        if isinstance(li_name, bytes):
            li_name = pkg.exports[i].object_name

        # Set parent for sub-level actors that don't already have one
        for actor in sub_level.actors:
            if not actor.parent:
                actor.parent = li_name

        actors.extend(sub_level.actors)

    # Apply parent_transform to actors parsed directly from this level
    if parent_transform is not None and not np.allclose(parent_transform, np.eye(4)):
        for actor in actors[:local_actor_count]:
            local = _make_ue_transform(actor.world_location,
                                       actor.world_rotation,
                                       actor.world_scale)
            world = parent_transform @ local
            loc, rot, scl = _decompose_ue_transform(world)
            actor.world_location = loc
            actor.world_rotation = rot
            actor.world_scale = scl

    return LevelData(
        map_name=map_name,
        actors=actors,
        camera_location=camera_location,
        camera_rotation=camera_rotation,
        has_camera=has_camera,
    )
