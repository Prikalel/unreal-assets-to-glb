"""Three.js-based browser preview server for UE5 levels.

Starts a local HTTP server that serves a Three.js scene viewer.
Uses only stdlib http.server — no external Python dependencies.

Usage (called from main.py):
    from preview_server import start_server
    start_server(umap_path, export_dir, content_dir, port=3050)
"""
import http.server
import json
import os
import sys
import mimetypes
import urllib.parse
from pathlib import Path

# ---------------------------------------------------------------------------
# Scene data builder
# ---------------------------------------------------------------------------

def build_scene_json(umap_path, export_dir, content_dir):
    """Parse .umap and build scene data JSON for the viewer.

    Args:
        umap_path:   Path to the .umap file.
        export_dir:  Path to the Export/ folder (contains Meshes/ and Textures/).
        content_dir: Path to the Content/ folder (or project root) for asset resolution.

    Returns:
        dict with 'actors' and 'camera' keys.
    """
    from uasset.umap import parse_level

    print(f"Parsing level: {umap_path}")
    level_data = parse_level(umap_path)
    print(f"Found {len(level_data.actors)} actors with mesh references")

    # Build index of available exported GLB meshes
    meshes_dir = os.path.join(export_dir, "Meshes")
    exported_glbs = set()
    if os.path.isdir(meshes_dir):
        for f in os.listdir(meshes_dir):
            if f.lower().endswith('.glb'):
                exported_glbs.add(os.path.splitext(f)[0])

    actors_json = []
    for actor in level_data.actors:
        # Check if a GLB file exists for this mesh (textures are embedded)
        has_glb = actor.mesh_name in exported_glbs

        actors_json.append({
            "name": actor.name,
            "mesh_name": actor.mesh_name,
            "location": {
                "x": actor.world_location[0],
                "y": actor.world_location[1],
                "z": actor.world_location[2],
            },
            "rotation": {
                "pitch": actor.world_rotation[0],
                "yaw": actor.world_rotation[1],
                "roll": actor.world_rotation[2],
            },
            "scale": {
                "x": actor.world_scale[0],
                "y": actor.world_scale[1],
                "z": actor.world_scale[2],
            },
            "parent": actor.parent,
            "has_glb": has_glb,
        })

    camera_json = {
        "location": {
            "x": level_data.camera_location[0],
            "y": level_data.camera_location[1],
            "z": level_data.camera_location[2],
        },
        "rotation": {
            "pitch": level_data.camera_rotation[0],
            "yaw": level_data.camera_rotation[1],
            "roll": level_data.camera_rotation[2],
        },
        "has_camera": level_data.has_camera,
    }

    return {"actors": actors_json, "camera": camera_json}


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class PreviewHandler(http.server.BaseHTTPRequestHandler):
    """Serves preview.html, scene API, and exported assets."""

    # Class-level attributes set by start_server()
    scene_json = None
    export_dir = None
    html_path = None

    def log_message(self, fmt, *args):
        """Quieter logging — only show errors."""
        if args and '404' not in str(args[0]):
            super().log_message(fmt, *args)

    # ---- routing --------------------------------------------------------

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == '/' or path == '/index.html':
            self._serve_html()
        elif path == '/api/scene':
            self._serve_json()
        elif path.startswith('/Export/Meshes/') or path.startswith('/meshes/'):
            self._serve_export_file(path, 'Meshes')
        elif path.startswith('/Export/Textures/'):
            self._serve_export_file(path, 'Textures')
        else:
            self.send_error(404, 'Not Found')

    # ---- handlers -------------------------------------------------------

    def _serve_html(self):
        try:
            with open(self.html_path, 'r', encoding='utf-8') as f:
                data = f.read().encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except FileNotFoundError:
            self.send_error(404, 'preview.html not found')

    def _serve_json(self):
        data = json.dumps(self.scene_json).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(data)

    def _serve_export_file(self, url_path, subdir):
        """Serve a file from Export/<subdir>/."""
        # Extract filename from URL
        # /Export/Meshes/name.glb  ->  name.glb
        # /meshes/name             ->  name.glb
        parts = url_path.split('/')
        filename = parts[-1]

        # /meshes/<name> without extension → default to .glb
        if subdir == 'Meshes' and '.' not in filename:
            filename += '.glb'

        filepath = os.path.join(self.export_dir, subdir, filename)
        if not os.path.isfile(filepath):
            self.send_error(404, f'{filename} not found')
            return

        # Explicit MIME types for glTF formats
        ext = os.path.splitext(filename)[1].lower()
        if ext == '.glb':
            mime = 'model/gltf-binary'
        elif ext == '.gltf':
            mime = 'model/gltf+json'
        else:
            mime, _ = mimetypes.guess_type(filename)
            if mime is None:
                mime = 'application/octet-stream'

        try:
            with open(filepath, 'rb') as f:
                data = f.read()
            self.send_response(200)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except IOError:
            self.send_error(500, 'Error reading file')


# ---------------------------------------------------------------------------
# Server entry point
# ---------------------------------------------------------------------------

def start_server(umap_path, export_dir, content_dir, port=3050):
    """Build scene data and start the preview HTTP server.

    Args:
        umap_path:   Path to the .umap file.
        export_dir:  Path to the Export/ folder.
        content_dir: Path to the project root (containing Content/) for texture resolution.
        port:        TCP port to listen on (default 3050).
    """
    # Build scene data
    scene_data = build_scene_json(umap_path, export_dir, content_dir)

    # Count unique meshes
    mesh_names = set(a['mesh_name'] for a in scene_data['actors'])
    with_glb = sum(1 for a in scene_data['actors'] if a['has_glb'])
    print(f"Scene: {len(scene_data['actors'])} actors, "
          f"{len(mesh_names)} unique meshes, {with_glb} with GLB")

    # Resolve path to preview.html (same directory as this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(script_dir, 'preview.html')

    # Configure handler class attributes
    PreviewHandler.scene_json = scene_data
    PreviewHandler.export_dir = os.path.abspath(export_dir)
    PreviewHandler.html_path = html_path

    server = http.server.HTTPServer(('0.0.0.0', port), PreviewHandler)
    print(f"\nPreview available at http://localhost:{port}")
    print("Press Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down preview server.")
        server.shutdown()
