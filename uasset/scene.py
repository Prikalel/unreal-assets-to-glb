"""Trimesh Scene Renderer for UE5 level preview with tkinter + matplotlib.

Builds a trimesh Scene from LevelData using exported OBJ + PNG files,
converting UE coordinate system to trimesh (OpenGL) coordinate system.

Features:
- Camera position parsed from map data (PlayerStart/CameraActor)
- Tkinter window with embedded matplotlib 3D canvas (CPU rendering)
- Tree view showing all rendered meshes
- Double-click to focus camera on a mesh
- Single-click to view/edit mesh transform
- Camera transform display
- Optional pyglet/OpenGL renderer (--gl flag)
"""
import os
import math
import logging
import time
from tkinter import ttk
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field

import numpy as np

try:
    import trimesh
except ImportError:
    trimesh = None

try:
    from PIL import Image
except ImportError:
    Image = None

from .umap import LevelData, LevelActor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# UE → trimesh coordinate conversion
# ---------------------------------------------------------------------------

COORD_CONVERT = np.array([
    [1,  0,  0, 0],
    [0,  0,  1, 0],
    [0, -1,  0, 0],
    [0,  0,  0, 1]
], dtype=float)


def rotator_to_matrix(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """Convert UE FRotator (degrees) to 3x3 rotation matrix."""
    p = math.radians(pitch)
    y = math.radians(yaw)
    r = math.radians(roll)

    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    cr, sr = math.cos(r), math.sin(r)

    R = np.array([
        [cy * cp,  cy * sp * sr - sy * cr,  cy * sp * cr + sy * sr],
        [sy * cp,  sy * sp * sr + cy * cr,  sy * sp * cr - cy * sr],
        [-sp,      cp * sr,                 cp * cr],
    ])
    return R


def ue_transform_to_matrix(location: tuple, rotation: tuple, scale: tuple) -> np.ndarray:
    """Build 4x4 transform matrix from UE transform data."""
    R = rotator_to_matrix(*rotation)
    s = np.array([scale[0], scale[1], scale[2]])

    T = np.eye(4, dtype=float)
    T[:3, :3] = R * s[np.newaxis, :]
    T[0, 3] = location[0]
    T[1, 3] = location[1]
    T[2, 3] = location[2]

    return COORD_CONVERT @ T


# ---------------------------------------------------------------------------
# Asset-based texture resolution
# ---------------------------------------------------------------------------

_BASE_COLOR_SUFFIXES = ('_BC', '_B', '_D', '_bc', '_b', '_d')


def _build_uasset_index(input_dir: str) -> Dict[str, str]:
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
                                          tex_map: Dict[str, str]) -> Optional[str]:
    from .package import Package
    filepath = uasset_index.get(material_name)
    if filepath is None:
        return None
    try:
        pkg = Package(filepath)
        for imp in pkg.imports:
            if imp.class_name == 'Texture2D':
                tex_name = imp.object_name
                if any(tex_name.endswith(sfx) for sfx in _BASE_COLOR_SUFFIXES):
                    if tex_name in tex_map:
                        return tex_map[tex_name]
                    for en, ep in tex_map.items():
                        if en.lower() == tex_name.lower():
                            return ep
        return None
    except Exception as e:
        logger.debug(f"Failed to read material package for '{material_name}': {e}")
        return None


def _resolve_mesh_texture(mesh_name: str,
                          uasset_index: Dict[str, str],
                          tex_map: Dict[str, str]) -> Optional[str]:
    materials = _get_material_names_from_mesh(mesh_name, uasset_index)
    if not materials:
        return None
    for mat_name in materials:
        tex_path = _get_base_color_texture_from_material(mat_name, uasset_index, tex_map)
        if tex_path is not None:
            return tex_path
    return None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ActorInfo:
    """Information about an actor in the scene for the tree view."""
    name: str
    mesh_name: str
    location: tuple
    rotation: tuple
    scale: tuple


@dataclass
class SceneData:
    """Complete scene data for preview."""
    scene: object = None                      # trimesh.Scene
    actors: List[ActorInfo] = field(default_factory=list)
    mesh_cache: Dict[str, object] = field(default_factory=dict)
    camera_location: tuple = (0.0, 0.0, 0.0)
    camera_rotation: tuple = (0.0, 0.0, 0.0)
    has_camera: bool = False


# ---------------------------------------------------------------------------
# Scene builder
# ---------------------------------------------------------------------------

def build_preview_scene(level_data: LevelData,
                        export_dir: str,
                        input_dir: Optional[str] = None) -> SceneData:
    """Build a trimesh Scene from level data using exported OBJ+PNG files."""
    if trimesh is None:
        raise ImportError("trimesh is required. Install with: pip install trimesh")

    meshes_dir = os.path.join(export_dir, "Meshes")
    textures_dir = os.path.join(export_dir, "Textures")

    obj_map: Dict[str, str] = {}
    if os.path.isdir(meshes_dir):
        for f in os.listdir(meshes_dir):
            if f.lower().endswith('.obj'):
                obj_map[os.path.splitext(f)[0]] = os.path.join(meshes_dir, f)

    tex_map: Dict[str, str] = {}
    if os.path.isdir(textures_dir):
        for f in os.listdir(textures_dir):
            if f.lower().endswith('.png'):
                tex_map[os.path.splitext(f)[0]] = os.path.join(textures_dir, f)

    uasset_index: Dict[str, str] = {}
    if input_dir is not None:
        uasset_index = _build_uasset_index(input_dir)

    scene = trimesh.Scene()
    loaded_meshes: Dict[str, trimesh.Trimesh] = {}
    actors_info: List[ActorInfo] = []

    for actor in level_data.actors:
        obj_path = obj_map.get(actor.mesh_name)
        if obj_path is None:
            for name, path in obj_map.items():
                if name.lower() == actor.mesh_name.lower():
                    obj_path = path
                    break
        if obj_path is None:
            continue

        if actor.mesh_name not in loaded_meshes:
            try:
                loaded_mesh = trimesh.load(obj_path, force='mesh')
                if isinstance(loaded_mesh, trimesh.Trimesh):
                    loaded_meshes[actor.mesh_name] = loaded_mesh
                elif isinstance(loaded_mesh, trimesh.Scene):
                    geo = trimesh.util.concatenate(loaded_mesh.geometry.values())
                    loaded_meshes[actor.mesh_name] = geo
                else:
                    continue
            except Exception as e:
                logger.warning(f"Failed to load OBJ '{obj_path}': {e}")
                continue

        mesh = loaded_meshes[actor.mesh_name].copy()

        texture_applied = False
        if Image is not None and uasset_index:
            tex_path = _resolve_mesh_texture(actor.mesh_name, uasset_index, tex_map)
            if tex_path:
                try:
                    img = Image.open(tex_path)
                    if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
                        mesh.visual = trimesh.visual.texture.TextureVisuals(
                            uv=mesh.visual.uv, image=img)
                        texture_applied = True
                except Exception:
                    pass

        if not texture_applied:
            gray = np.full((len(mesh.vertices), 4), [160, 160, 160, 255],
                           dtype=np.uint8)
            mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=gray)

        transform = ue_transform_to_matrix(actor.world_location, actor.world_rotation, actor.world_scale)
        if not np.all(np.isfinite(transform)):
            transform = np.eye(4)

        mesh.apply_transform(transform)
        scene.add_geometry(mesh, node_name=actor.name, geom_name=actor.name)

        actors_info.append(ActorInfo(
            name=actor.name,
            mesh_name=actor.mesh_name,
            location=actor.world_location,
            rotation=actor.world_rotation,
            scale=actor.world_scale,
        ))

    logger.info(f"Scene built with {len(scene.geometry)} objects")
    return SceneData(
        scene=scene,
        actors=actors_info,
        mesh_cache=loaded_meshes,
        camera_location=level_data.camera_location,
        camera_rotation=level_data.camera_rotation,
        has_camera=level_data.has_camera,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _matrix_to_euler_degrees(R: np.ndarray) -> Tuple[float, float, float]:
    """Extract Euler angles (degrees) from a 3x3 rotation matrix (YXZ order)."""
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0.0
    return (math.degrees(x), math.degrees(y), math.degrees(z))


# ---------------------------------------------------------------------------
# Matplotlib + Tkinter previewer (CPU rendering)
# ---------------------------------------------------------------------------

# Maximum faces per mesh for matplotlib rendering
_MAX_FACES = 4000

# Color palette for meshes (distinct colors for visual separation)
_MESH_COLORS = [
    (0.70, 0.70, 0.70),   # light grey
    (0.55, 0.60, 0.70),   # blue-grey
    (0.65, 0.60, 0.55),   # warm grey
    (0.60, 0.70, 0.60),   # green-grey
    (0.70, 0.65, 0.55),   # tan
    (0.55, 0.55, 0.70),   # lavender-grey
    (0.70, 0.55, 0.55),   # rose-grey
    (0.55, 0.70, 0.70),   # teal-grey
    (0.68, 0.68, 0.58),   # khaki-grey
    (0.58, 0.68, 0.68),   # cyan-grey
]


class MatplotlibPreviewer:
    """CPU-based level previewer using matplotlib embedded in tkinter."""

    def __init__(self, scene_data: SceneData):
        self.scene_data = scene_data
        self.scene = scene_data.scene
        self.actors_info = scene_data.actors
        self.mesh_cache = scene_data.mesh_cache
        self._selected_actor: Optional[ActorInfo] = None

        self._actor_by_name: Dict[str, ActorInfo] = {
            a.name: a for a in self.actors_info
        }
        self._mesh_to_actors: Dict[str, List[ActorInfo]] = {}
        for actor in self.actors_info:
            self._mesh_to_actors.setdefault(actor.mesh_name, []).append(actor)

        # Assign colors per mesh name
        self._mesh_colors: Dict[str, Tuple] = {}
        for i, mn in enumerate(sorted(self._mesh_to_actors.keys())):
            self._mesh_colors[mn] = _MESH_COLORS[i % len(_MESH_COLORS)]

        # Actor visibility
        self._actor_visible: Dict[str, bool] = {
            a.name: True for a in self.actors_info
        }

        # Tkinter widgets (set in _create_window)
        self.tk_root = None
        self.tree = None
        self.ax = None
        self.canvas = None
        self.fig = None

        # Transform entry vars
        self.loc_x = self.loc_y = self.loc_z = None
        self.rot_p = self.rot_y = self.rot_r = None
        self.scale_x = self.scale_y = self.scale_z = None

        # Camera labels
        self.camera_pos_label = None
        self.camera_rot_label = None

    # ------------------------------------------------------------------ run

    def run(self):
        """Create the tkinter window and start the mainloop."""
        import tkinter as tk
        from tkinter import ttk

        self.tk_root = tk.Tk()
        self.tk_root.title("UE5 Level Preview — CPU Renderer (matplotlib)")
        self.tk_root.geometry("1500x920")

        self._create_window()
        self._render_all()
        self._setup_camera()

        self.tk_root.mainloop()

    # --------------------------------------------------------- create window

    def _create_window(self):
        import tkinter as tk
        from tkinter import ttk
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        # Main horizontal paned window
        main_pane = ttk.PanedWindow(self.tk_root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)

        # === LEFT: matplotlib 3D canvas ===
        fig_frame = ttk.Frame(main_pane)
        main_pane.add(fig_frame, weight=3)

        self.fig = Figure(figsize=(10, 8), dpi=100, facecolor='#2b2b2b')
        self.ax = self.fig.add_subplot(111, projection='3d', facecolor='#1e1e1e')
        self.ax.set_xlabel('X', color='white', fontsize=8)
        self.ax.set_ylabel('Y', color='white', fontsize=8)
        self.ax.set_zlabel('Z', color='white', fontsize=8)
        self.ax.tick_params(colors='gray', labelsize=6)
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        self.canvas = FigureCanvasTkAgg(self.fig, master=fig_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # === RIGHT: tree + transform + camera ===
        right_frame = ttk.Frame(main_pane, width=480)
        main_pane.add(right_frame, weight=1)

        # Status bar
        status_frame = ttk.Frame(right_frame)
        status_frame.pack(fill=tk.X, padx=5, pady=(5, 0))
        n_actors = len(self.actors_info)
        n_meshes = len(self._mesh_to_actors)
        ttk.Label(status_frame,
                  text=f"Actors: {n_actors}  |  Mesh types: {n_meshes}",
                  foreground="gray").pack(side=tk.LEFT)

        # Tree view
        tree_frame = ttk.Frame(right_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.tree = ttk.Treeview(
            tree_frame, columns=("mesh", "loc"),
            show="tree headings", selectmode="browse",
        )
        self.tree.heading("#0", text="Actor / Mesh", anchor=tk.W)
        self.tree.heading("mesh", text="Mesh Asset", anchor=tk.W)
        self.tree.heading("loc", text="Location", anchor=tk.W)
        self.tree.column("#0", width=190, minwidth=130)
        self.tree.column("mesh", width=140, minwidth=90)
        self.tree.column("loc", width=130, minwidth=70)

        vsb = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        for mesh_name in sorted(self._mesh_to_actors.keys()):
            actors = self._mesh_to_actors[mesh_name]
            color = self._mesh_colors.get(mesh_name, (0.6, 0.6, 0.6))
            hex_col = '#%02x%02x%02x' % (
                int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
            mesh_node = self.tree.insert(
                '', 'end', text=f"📁 {mesh_name}",
                values=(f"{len(actors)} actors", ""),
                tags=('mesh_group',), open=False,
            )
            for actor in sorted(actors, key=lambda a: a.name):
                loc_str = (f"({actor.location[0]:.0f}, "
                           f"{actor.location[1]:.0f}, "
                           f"{actor.location[2]:.0f})")
                self.tree.insert(
                    mesh_node, 'end',
                    text=f"  {actor.name}",
                    values=(actor.mesh_name, loc_str),
                    tags=('actor',),
                )

        self.tree.tag_bind('actor', '<Double-1>', self._on_double_click)
        self.tree.tag_bind('actor', '<<TreeviewSelect>>', self._on_select)

        # Transform editor
        tf = ttk.LabelFrame(right_frame, text="Transform (UE space)", padding=5)
        tf.pack(fill=tk.X, padx=5, pady=5)

        row_loc = ttk.Frame(tf); row_loc.pack(fill=tk.X, pady=2)
        ttk.Label(row_loc, text="Location:", width=10).pack(side=tk.LEFT)
        self.loc_x = self._entry(row_loc, "X")
        self.loc_y = self._entry(row_loc, "Y")
        self.loc_z = self._entry(row_loc, "Z")

        row_rot = ttk.Frame(tf); row_rot.pack(fill=tk.X, pady=2)
        ttk.Label(row_rot, text="Rotation:", width=10).pack(side=tk.LEFT)
        self.rot_p = self._entry(row_rot, "P")
        self.rot_y = self._entry(row_rot, "Y")
        self.rot_r = self._entry(row_rot, "R")

        row_sc = ttk.Frame(tf); row_sc.pack(fill=tk.X, pady=2)
        ttk.Label(row_sc, text="Scale:", width=10).pack(side=tk.LEFT)
        self.scale_x = self._entry(row_sc, "X")
        self.scale_y = self._entry(row_sc, "Y")
        self.scale_z = self._entry(row_sc, "Z")

        btn_row = ttk.Frame(tf); btn_row.pack(fill=tk.X, pady=5)
        ttk.Button(btn_row, text="Apply", command=self._apply_transform).pack(
            side=tk.LEFT, padx=3)
        ttk.Button(btn_row, text="Reset", command=self._reset_transform).pack(
            side=tk.LEFT, padx=3)
        ttk.Button(btn_row, text="Focus", command=self._focus_selected).pack(
            side=tk.LEFT, padx=3)
        ttk.Button(btn_row, text="Fit All", command=self._fit_all).pack(
            side=tk.LEFT, padx=3)

        # Camera display
        cf = ttk.LabelFrame(right_frame, text="Camera", padding=5)
        cf.pack(fill=tk.X, padx=5, pady=5)
        self.camera_pos_label = ttk.Label(cf, text="Position: —", anchor=tk.W)
        self.camera_pos_label.pack(fill=tk.X)
        self.camera_rot_label = ttk.Label(cf, text="Rotation: —", anchor=tk.W)
        self.camera_rot_label.pack(fill=tk.X)

        # Update camera display when user rotates the 3D view
        self.canvas.mpl_connect('button_release_event', self._on_view_changed)

    def _entry(self, parent, label):
        import tkinter as tk
        f = ttk.Frame(parent); f.pack(side=tk.LEFT, padx=3)
        ttk.Label(f, text=label, width=2).pack(side=tk.LEFT)
        var = tk.StringVar(value="0.0")
        ttk.Entry(f, textvariable=var, width=10).pack(side=tk.LEFT)
        return var

    # ----------------------------------------------------------- rendering

    def _render_all(self):
        """Render all visible actor meshes into the matplotlib 3D axes."""
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        self.ax.cla()
        self.ax.set_facecolor('#1e1e1e')
        self.ax.set_xlabel('X', color='white', fontsize=8)
        self.ax.set_ylabel('Y', color='white', fontsize=8)
        self.ax.set_zlabel('Z', color='white', fontsize=8)
        self.ax.tick_params(colors='gray', labelsize=6)

        for actor in self.actors_info:
            if not self._actor_visible.get(actor.name, True):
                continue
            mesh = self.scene.geometry.get(actor.name)
            if mesh is None:
                continue
            color = self._mesh_colors.get(actor.mesh_name, (0.6, 0.6, 0.6))
            self._draw_mesh(mesh, color, actor.name == (
                self._selected_actor.name if self._selected_actor else ""))

        self._update_axes_limits()
        self.canvas.draw_idle()

    def _draw_mesh(self, mesh, color: tuple, selected: bool = False):
        """Draw a single mesh into the 3D axes."""
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        verts = mesh.vertices
        faces = mesh.faces

        if len(faces) == 0:
            return

        # Subsample faces if too many
        if len(faces) > _MAX_FACES:
            indices = np.random.choice(len(faces), _MAX_FACES, replace=False)
            faces = faces[indices]

        triangles = verts[faces]

        ec = (0.9, 0.3, 0.3) if selected else (color[0] * 0.7, color[1] * 0.7, color[2] * 0.7)
        lw = 0.3 if not selected else 0.6
        alpha = 0.95 if not selected else 1.0

        poly = Poly3DCollection(triangles, alpha=alpha, linewidth=lw)
        poly.set_facecolor(color)
        poly.set_edgecolor(ec)
        self.ax.add_collection3d(poly)

    def _update_axes_limits(self):
        """Set axes limits to fit the scene."""
        bounds = self.scene.bounds
        if bounds is None:
            return
        margin = 0.05
        for i, (lo, hi) in enumerate(zip(bounds[0], bounds[1])):
            span = hi - lo
            if span < 1e-6:
                span = 100.0
            lo -= span * margin
            hi += span * margin
            if i == 0:
                self.ax.set_xlim(lo, hi)
            elif i == 1:
                self.ax.set_ylim(lo, hi)
            else:
                self.ax.set_zlim(lo, hi)

    # ------------------------------------------------------------ camera

    def _setup_camera(self):
        """Set camera from level data or scene bounds."""
        if self.scene_data.has_camera:
            self._setup_camera_from_ue()
        else:
            self.ax.view_init(elev=25, azim=-60)
        self._update_camera_display()
        self.canvas.draw_idle()

    def _setup_camera_from_ue(self):
        """Set camera from UE camera data."""
        cam_loc = self.scene_data.camera_location
        cam_rot = self.scene_data.camera_rotation

        # Convert UE location to trimesh space
        loc_ue = np.array(cam_loc)
        loc_tri = COORD_CONVERT[:3, :3] @ loc_ue

        # Convert UE rotation to trimesh forward direction
        R = rotator_to_matrix(*cam_rot)
        fwd_ue = R @ np.array([1.0, 0.0, 0.0])
        fwd_tri = COORD_CONVERT[:3, :3] @ fwd_ue

        # Convert forward direction to matplotlib elevation/azimuth
        # matplotlib: elev = angle above XY plane, azim = angle in XY plane from X axis
        x, y, z = fwd_tri
        azim = math.degrees(math.atan2(y, x))
        r_xy = math.sqrt(x ** 2 + y ** 2)
        elev = math.degrees(math.atan2(z, r_xy))

        # matplotlib camera looks from (elev, azim) toward center
        # We want it to look in the forward direction, so invert
        self.ax.view_init(elev=-elev, azim=180 + azim)

    def _focus_on_actor(self, actor: ActorInfo):
        """Focus camera on an actor's mesh."""
        mesh = self.scene.geometry.get(actor.name)
        if mesh is None:
            return
        bounds = mesh.bounds
        if bounds is None:
            return

        center = (bounds[0] + bounds[1]) / 2.0
        size = bounds[1] - bounds[0]
        max_size = float(np.max(size))
        if max_size < 1e-6:
            max_size = 100.0

        margin = max_size * 0.6
        self.ax.set_xlim(center[0] - margin, center[0] + margin)
        self.ax.set_ylim(center[1] - margin, center[1] + margin)
        self.ax.set_zlim(center[2] - margin, center[2] + margin)

        self._render_all()

    def _fit_all(self):
        """Fit all meshes in view."""
        self._render_all()

    def _on_view_changed(self, _event=None):
        """Called when user rotates/zooms the 3D view."""
        self._update_camera_display()

    def _update_camera_display(self):
        """Update camera info labels."""
        try:
            elev = self.ax.elev
            azim = self.ax.azim
            self.camera_pos_label.config(
                text=f"Elevation: {elev:.1f}°  |  Azimuth: {azim:.1f}°"
            )
            # Get current axes limits as "position"
            xlim = self.ax.get_xlim3d()
            ylim = self.ax.get_ylim3d()
            zlim = self.ax.get_zlim3d()
            cx = (xlim[0] + xlim[1]) / 2
            cy = (ylim[0] + ylim[1]) / 2
            cz = (zlim[0] + zlim[1]) / 2
            self.camera_rot_label.config(
                text=f"Center: ({cx:.0f}, {cy:.0f}, {cz:.0f})"
            )
        except Exception:
            pass

    # -------------------------------------------------------- tree events

    def _get_selected_actor(self) -> Optional[ActorInfo]:
        sel = self.tree.selection()
        if not sel:
            return None
        item = sel[0]
        if 'actor' not in self.tree.item(item, 'tags'):
            return None
        name = self.tree.item(item, 'text').strip()
        return self._actor_by_name.get(name)

    def _on_double_click(self, _event):
        actor = self._get_selected_actor()
        if actor:
            self._focus_on_actor(actor)

    def _on_select(self, _event):
        actor = self._get_selected_actor()
        if actor:
            self._selected_actor = actor
            self._display_transform(actor)
            # Highlight selected actor
            self._render_all()

    def _focus_selected(self):
        actor = self._get_selected_actor()
        if actor:
            self._focus_on_actor(actor)

    # ------------------------------------------------------ transform edit

    def _display_transform(self, actor: ActorInfo):
        self.loc_x.set(f"{actor.location[0]:.2f}")
        self.loc_y.set(f"{actor.location[1]:.2f}")
        self.loc_z.set(f"{actor.location[2]:.2f}")
        self.rot_p.set(f"{actor.rotation[0]:.2f}")
        self.rot_y.set(f"{actor.rotation[1]:.2f}")
        self.rot_r.set(f"{actor.rotation[2]:.2f}")
        self.scale_x.set(f"{actor.scale[0]:.4f}")
        self.scale_y.set(f"{actor.scale[1]:.4f}")
        self.scale_z.set(f"{actor.scale[2]:.4f}")

    def _apply_transform(self):
        if self._selected_actor is None:
            return
        try:
            loc = (float(self.loc_x.get()), float(self.loc_y.get()),
                   float(self.loc_z.get()))
            rot = (float(self.rot_p.get()), float(self.rot_y.get()),
                   float(self.rot_r.get()))
            scale = (float(self.scale_x.get()), float(self.scale_y.get()),
                     float(self.scale_z.get()))
        except ValueError:
            return

        actor = self._selected_actor
        base_mesh = self.mesh_cache.get(actor.mesh_name)
        if base_mesh is None:
            return

        old_mesh = self.scene.geometry.get(actor.name)
        new_mesh = base_mesh.copy()

        # Preserve visual
        if old_mesh is not None:
            try:
                if isinstance(old_mesh.visual, trimesh.visual.texture.TextureVisuals):
                    new_mesh.visual = trimesh.visual.texture.TextureVisuals(
                        uv=getattr(old_mesh.visual, 'uv', None),
                        image=getattr(old_mesh.visual, 'image', None))
                elif isinstance(old_mesh.visual, trimesh.visual.ColorVisuals):
                    vc = getattr(old_mesh.visual, 'vertex_colors', None)
                    if vc is not None:
                        new_mesh.visual = trimesh.visual.ColorVisuals(
                            new_mesh, vertex_colors=vc)
            except Exception:
                gray = np.full((len(new_mesh.vertices), 4),
                               [160, 160, 160, 255], dtype=np.uint8)
                new_mesh.visual = trimesh.visual.ColorVisuals(
                    new_mesh, vertex_colors=gray)

        transform = ue_transform_to_matrix(loc, rot, scale)
        if not np.all(np.isfinite(transform)):
            return

        new_mesh.apply_transform(transform)
        self.scene.geometry[actor.name] = new_mesh

        actor.location = loc
        actor.rotation = rot
        actor.scale = scale

        self._render_all()

    def _reset_transform(self):
        if self._selected_actor:
            self._display_transform(self._selected_actor)


# ---------------------------------------------------------------------------
# Pyglet/OpenGL previewer (optional, for systems with good GPU support)
# ---------------------------------------------------------------------------

class PygletPreviewer:
    """GPU-based level previewer using pyglet/OpenGL (requires compatible GPU)."""

    def __init__(self, scene_data: SceneData):
        self.scene_data = scene_data
        self.scene = scene_data.scene
        self.actors_info = scene_data.actors
        self.mesh_cache = scene_data.mesh_cache
        self.viewer = None
        self.tk_root = None
        self.running = True
        self._selected_actor = None

        self._actor_by_name = {a.name: a for a in self.actors_info}
        self._mesh_to_actors: Dict[str, List[ActorInfo]] = {}
        for actor in self.actors_info:
            self._mesh_to_actors.setdefault(actor.mesh_name, []).append(actor)

    def run(self):
        import pyglet
        from trimesh.viewer import SceneViewer as _SceneViewer

        self._setup_camera()

        self.viewer = _SceneViewer(
            self.scene, visible=True, resolution=(1920, 1080),
            start_loop=False,
        )

        @self.viewer.event
        def on_close():
            self.running = False
            try:
                self.tk_root.destroy()
            except Exception:
                pass
            return pyglet.event.EVENT_HANDLED

        self._create_tkinter_window()
        self._run_loop()

    def _setup_camera(self):
        if self.scene_data.has_camera:
            T = ue_transform_to_matrix(
                self.scene_data.camera_location,
                self.scene_data.camera_rotation, (1, 1, 1))
            forward, right, up, pos = T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3]
            ct = np.eye(4)
            ct[:3, 0] = right
            ct[:3, 1] = up
            ct[:3, 2] = -forward
            ct[:3, 3] = pos
            self.scene.camera_transform = ct
        else:
            if self.scene.is_empty:
                return
            self.scene.camera.resolution = [1920, 1080]
            centroid = self.scene.centroid
            max_ext = max(float(np.max(self.scene.extents)), 1.0)
            dist = max_ext / (2 * math.tan(math.radians(self.scene.camera.fov[0]) / 2)) / 0.6
            cam_pos = centroid + np.array([dist * 0.5, dist * 0.6, dist * 0.8])
            fwd = centroid - cam_pos
            fwd /= np.linalg.norm(fwd)
            up = np.array([0., 1., 0.])
            right = np.cross(fwd, up)
            if np.linalg.norm(right) < 1e-6:
                up = np.array([0., 0., 1.])
                right = np.cross(fwd, up)
            right /= np.linalg.norm(right)
            up = np.cross(right, fwd)
            up /= np.linalg.norm(up)
            ct = np.eye(4)
            ct[:3, 0] = right
            ct[:3, 1] = up
            ct[:3, 2] = -fwd
            ct[:3, 3] = cam_pos
            self.scene.camera_transform = ct

    def _create_tkinter_window(self):
        import tkinter as tk
        from tkinter import ttk

        self.tk_root = tk.Tk()
        self.tk_root.title("Mesh Browser — UE5 Level Preview (OpenGL)")
        self.tk_root.geometry("520x920")
        self.tk_root.protocol("WM_DELETE_WINDOW", self._on_tk_close)

        status = ttk.Frame(self.tk_root)
        status.pack(fill=tk.X, padx=5, pady=(5, 0))
        ttk.Label(status, text=f"Actors: {len(self.actors_info)}  |  "
                  f"Meshes: {len(self._mesh_to_actors)}",
                  foreground="gray").pack(side=tk.LEFT)

        tf = ttk.Frame(self.tk_root)
        tf.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.tree = ttk.Treeview(tf, columns=("mesh", "loc"),
                                 show="tree headings", selectmode="browse")
        self.tree.heading("#0", text="Actor")
        self.tree.heading("mesh", text="Mesh")
        self.tree.heading("loc", text="Location")
        self.tree.column("#0", width=200)
        self.tree.column("mesh", width=160)
        self.tree.column("loc", width=140)
        sb = ttk.Scrollbar(tf, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        for mn in sorted(self._mesh_to_actors):
            actors = self._mesh_to_actors[mn]
            node = self.tree.insert('', 'end', text=f"📁 {mn}",
                                    values=(f"{len(actors)} actors", ""), open=False)
            for a in sorted(actors, key=lambda x: x.name):
                self.tree.insert(node, 'end', text=f"  {a.name}",
                                 values=(a.mesh_name,
                                         f"({a.location[0]:.0f}, {a.location[1]:.0f}, {a.location[2]:.0f})"),
                                 tags=('actor',))

        self.tree.tag_bind('actor', '<Double-1>', self._on_dbl)
        self.tree.tag_bind('actor', '<<TreeviewSelect>>', self._on_sel)

        ef = ttk.LabelFrame(self.tk_root, text="Transform (UE)", padding=5)
        ef.pack(fill=tk.X, padx=5, pady=5)
        r1 = ttk.Frame(ef); r1.pack(fill=tk.X, pady=2)
        ttk.Label(r1, text="Location:", width=10).pack(side=tk.LEFT)
        self.loc_x = self._ent(r1, "X"); self.loc_y = self._ent(r1, "Y"); self.loc_z = self._ent(r1, "Z")
        r2 = ttk.Frame(ef); r2.pack(fill=tk.X, pady=2)
        ttk.Label(r2, text="Rotation:", width=10).pack(side=tk.LEFT)
        self.rot_p = self._ent(r2, "P"); self.rot_y = self._ent(r2, "Y"); self.rot_r = self._ent(r2, "R")
        r3 = ttk.Frame(ef); r3.pack(fill=tk.X, pady=2)
        ttk.Label(r3, text="Scale:", width=10).pack(side=tk.LEFT)
        self.sx = self._ent(r3, "X"); self.sy = self._ent(r3, "Y"); self.sz = self._ent(r3, "Z")
        bf = ttk.Frame(ef); bf.pack(fill=tk.X, pady=5)
        ttk.Button(bf, text="Apply", command=self._apply).pack(side=tk.LEFT, padx=3)
        ttk.Button(bf, text="Focus", command=self._focus).pack(side=tk.LEFT, padx=3)

        cf = ttk.LabelFrame(self.tk_root, text="Camera", padding=5)
        cf.pack(fill=tk.X, padx=5, pady=5)
        self.cam_lbl = ttk.Label(cf, text="—", anchor=tk.W)
        self.cam_lbl.pack(fill=tk.X)

    def _ent(self, parent, label):
        import tkinter as tk
        f = ttk.Frame(parent); f.pack(side=tk.LEFT, padx=3)
        ttk.Label(f, text=label, width=2).pack(side=tk.LEFT)
        v = tk.StringVar(value="0.0")
        ttk.Entry(f, textvariable=v, width=10).pack(side=tk.LEFT)
        return v

    def _get_actor(self):
        sel = self.tree.selection()
        if not sel:
            return None
        if 'actor' not in self.tree.item(sel[0], 'tags'):
            return None
        return self._actor_by_name.get(self.tree.item(sel[0], 'text').strip())

    def _on_dbl(self, _e):
        a = self._get_actor()
        if a:
            self._focus_actor(a)

    def _on_sel(self, _e):
        a = self._get_actor()
        if a:
            self._selected_actor = a
            self.loc_x.set(f"{a.location[0]:.2f}")
            self.loc_y.set(f"{a.location[1]:.2f}")
            self.loc_z.set(f"{a.location[2]:.2f}")
            self.rot_p.set(f"{a.rotation[0]:.2f}")
            self.rot_y.set(f"{a.rotation[1]:.2f}")
            self.rot_r.set(f"{a.rotation[2]:.2f}")
            self.sx.set(f"{a.scale[0]:.4f}")
            self.sy.set(f"{a.scale[1]:.4f}")
            self.sz.set(f"{a.scale[2]:.4f}")

    def _focus_actor(self, a):
        m = self.scene.geometry.get(a.name)
        if m is None or m.bounds is None:
            return
        center = (m.bounds[0] + m.bounds[1]) / 2
        sz = max(float(np.max(m.bounds[1] - m.bounds[0])), 1.0)
        dist = sz / (2 * math.tan(math.radians(self.scene.camera.fov[0]) / 2)) / 0.6
        pos = center + np.array([dist * .5, dist * .6, dist * .8])
        fwd = center - pos; fwd /= np.linalg.norm(fwd)
        up = np.array([0., 1., 0.])
        right = np.cross(fwd, up)
        if np.linalg.norm(right) < 1e-6:
            up = np.array([0., 0., 1.]); right = np.cross(fwd, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd); up /= np.linalg.norm(up)
        ct = np.eye(4)
        ct[:3, 0] = right; ct[:3, 1] = up; ct[:3, 2] = -fwd; ct[:3, 3] = pos
        self.scene.camera_transform = ct
        self._redraw()

    def _focus(self):
        a = self._get_actor()
        if a:
            self._focus_actor(a)

    def _apply(self):
        a = self._selected_actor
        if a is None:
            return
        try:
            loc = (float(self.loc_x.get()), float(self.loc_y.get()), float(self.loc_z.get()))
            rot = (float(self.rot_p.get()), float(self.rot_y.get()), float(self.rot_r.get()))
            sc = (float(self.sx.get()), float(self.sy.get()), float(self.sz.get()))
        except ValueError:
            return
        base = self.mesh_cache.get(a.mesh_name)
        if base is None:
            return
        old = self.scene.geometry.get(a.name)
        nm = base.copy()
        if old:
            try:
                if isinstance(old.visual, trimesh.visual.texture.TextureVisuals):
                    nm.visual = trimesh.visual.texture.TextureVisuals(
                        uv=getattr(old.visual, 'uv', None),
                        image=getattr(old.visual, 'image', None))
                elif isinstance(old.visual, trimesh.visual.ColorVisuals):
                    vc = getattr(old.visual, 'vertex_colors', None)
                    if vc is not None:
                        nm.visual = trimesh.visual.ColorVisuals(nm, vertex_colors=vc)
            except Exception:
                pass
        T = ue_transform_to_matrix(loc, rot, sc)
        if not np.all(np.isfinite(T)):
            return
        nm.apply_transform(T)
        self.scene.geometry[a.name] = nm
        a.location, a.rotation, a.scale = loc, rot, sc
        self._redraw()

    def _redraw(self):
        if self.viewer:
            try:
                self.viewer.dispatch_event('on_draw')
                self.viewer.flip()
            except Exception:
                pass

    def _on_tk_close(self):
        self.running = False
        try:
            self.viewer.close()
        except Exception:
            pass

    def _run_loop(self):
        import pyglet
        import tkinter as tk
        last = 0.0
        while self.running:
            pyglet.clock.tick()
            for w in list(pyglet.app.windows):
                try:
                    w.switch_to(); w.dispatch_events()
                except Exception:
                    pass
            try:
                self.tk_root.update_idletasks(); self.tk_root.update()
            except tk.TclError:
                break
            now = time.time()
            if now - last > 0.2:
                ct = self.scene.camera_transform
                if ct is not None:
                    p = ct[:3, 3]
                    self.cam_lbl.config(text=f"Pos: ({p[0]:.0f}, {p[1]:.0f}, {p[2]:.0f})")
                last = now
            time.sleep(0.005)
        try:
            self.tk_root.destroy()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def show_preview(scene_data: SceneData, use_gl: bool = False):
    """Show the scene preview.

    Args:
        scene_data: SceneData from build_preview_scene()
        use_gl: If True, use pyglet/OpenGL renderer (requires compatible GPU).
                If False (default), use matplotlib CPU renderer.
    """
    if trimesh is None:
        raise ImportError("trimesh is required. Install with: pip install trimesh")

    if scene_data.scene.is_empty:
        logger.warning("Scene is empty, nothing to preview")
        return

    if use_gl:
        previewer = PygletPreviewer(scene_data)
    else:
        previewer = MatplotlibPreviewer(scene_data)
    previewer.run()
