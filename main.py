"""
UE 5.5 UAsset Parser and Exporter
Exports static meshes as OBJ and base color textures as PNG from .uasset files.
"""
import os
import sys
import traceback

# Add workspace root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from uasset.package import Package
from uasset.mesh import StaticMesh, export_obj
from uasset.texture import Texture2D, export_png, is_base_color_texture

CONTENT_DIR = os.path.join(os.path.dirname(__file__), "Content", "TheAbandonedTunnel")
MESH_DIR = os.path.join(CONTENT_DIR, "Meshes")
TEXTURE_DIR = os.path.join(CONTENT_DIR, "Textures")
EXPORT_DIR = os.path.join(os.path.dirname(__file__), "Export")


def process_meshes():
    mesh_export_dir = os.path.join(EXPORT_DIR, "Meshes")
    os.makedirs(mesh_export_dir, exist_ok=True)

    if not os.path.exists(MESH_DIR):
        print(f"Mesh directory not found: {MESH_DIR}")
        return

    files = [f for f in os.listdir(MESH_DIR) if f.endswith('.uasset')]
    print(f"Found {len(files)} mesh files")

    success = 0
    for i, filename in enumerate(sorted(files)):
        filepath = os.path.join(MESH_DIR, filename)
        name = os.path.splitext(filename)[0]
        print(f"[{i+1}/{len(files)}] Processing mesh: {name}")
        try:
            pkg = Package(filepath)
            mesh = StaticMesh.from_package(pkg)
            if mesh and mesh.vertices:
                obj_path = os.path.join(mesh_export_dir, f"{name}.obj")
                export_obj(mesh, obj_path)
                print(f"  -> Exported {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")
                success += 1
            else:
                print(f"  -> No mesh data found")
        except Exception as e:
            print(f"  -> ERROR: {e}")
            traceback.print_exc()

    print(f"Mesh export complete: {success}/{len(files)} successful")


def process_textures(only_base_color=True):
    """Export textures as PNG.

    Args:
        only_base_color: If True, only export diffuse/base color textures
            (filenames ending with _BC, _B, or _D).
    """
    tex_export_dir = os.path.join(EXPORT_DIR, "Textures")
    os.makedirs(tex_export_dir, exist_ok=True)

    if not os.path.exists(TEXTURE_DIR):
        print(f"Texture directory not found: {TEXTURE_DIR}")
        return

    all_files = [f for f in os.listdir(TEXTURE_DIR) if f.endswith('.uasset')]

    if only_base_color:
        files = [f for f in all_files if is_base_color_texture(f)]
        print(f"Found {len(files)} base color textures (out of {len(all_files)} total)")
    else:
        files = all_files
        print(f"Found {len(files)} texture files")

    success = 0
    for i, filename in enumerate(sorted(files)):
        filepath = os.path.join(TEXTURE_DIR, filename)
        name = os.path.splitext(filename)[0]
        print(f"[{i+1}/{len(files)}] Processing texture: {name}")
        try:
            pkg = Package(filepath)
            texture = Texture2D.from_package(pkg)
            if texture and texture.pixels is not None:
                png_path = os.path.join(tex_export_dir, f"{name}.png")
                export_png(texture, png_path)
                print(f"  -> Exported {texture.width}x{texture.height} "
                      f"({texture.format_str})")
                success += 1
            else:
                print(f"  -> No texture data found")
        except Exception as e:
            print(f"  -> ERROR: {e}")
            traceback.print_exc()

    print(f"Texture export complete: {success}/{len(files)} successful")


if __name__ == "__main__":
    os.makedirs(EXPORT_DIR, exist_ok=True)

    print("=" * 60)
    print("UE 5.5 UAsset Parser and Exporter")
    print("=" * 60)

    process_meshes()
    print()
    process_textures(only_base_color=True)

    print("\nDone!")
