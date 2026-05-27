"""
UE 5.5 UAsset Parser, Exporter, and Level Previewer

Usage:
  python main.py [INPUT_DIR]           # Extract meshes + base color textures
  python main.py [INPUT_DIR] --preview LEVEL.umap  # Extract + show level preview in browser
  python main.py ./Input --skip-export --preview L_Showcase.umap  # Skip export, just preview
  python main.py ./Input --skip-textures  # Embed textures in GLB but skip separate PNG export

Arguments:
  INPUT_DIR    Path to folder containing .uproject and Content/ (default: ./Input)

Options:
  --preview LEVEL.umap  Parse a .umap level file and show 3D preview in browser (port 3050)
  --export-dir DIR      Output directory (default: ./Export)
  --skip-export         Skip export step, use existing Export/ directory
  --skip-textures       Skip separate PNG export (textures still embedded in GLB)
"""
import argparse
import os
import sys
import json
import traceback
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from uasset.package import Package
from uasset.mesh import StaticMesh, export_glb
import pickle
import hashlib
from uasset.texture import Texture2D, export_png
from uasset.properties import read_properties
from uasset.scene import (
    _build_uasset_index,
    _get_material_names_from_mesh,
    _get_base_color_texture_from_material,
)


def find_uproject(input_dir):
    """Find .uproject file in input directory. Verify UE version."""
    for f in os.listdir(input_dir):
        if f.endswith('.uproject'):
            path = os.path.join(input_dir, f)
            with open(path, 'r') as fh:
                data = json.load(fh)
            engine = data.get('EngineAssociation', '')
            print(f"Found project: {f} (Engine: {engine})")
            if not engine.startswith('5.'):
                print(f"WARNING: Engine version {engine} may not be compatible (expected 5.x)")
            return path
    print("WARNING: No .uproject file found in input directory")
    return None


def find_uasset_files(input_dir):
    """Recursively find all .uasset and .umap files."""
    uassets = []
    umaps = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            full = os.path.join(root, f)
            if f.endswith('.uasset'):
                uassets.append(full)
            elif f.endswith('.umap'):
                umaps.append(full)
    return uassets, umaps


def classify_uasset(filepath, input_dir):
    """Determine asset type by parsing package header.
    Returns: ('mesh', name) or ('texture', name) or ('material', name) or ('other', name)
    """
    try:
        pkg = Package(filepath)
        # Check exports for known types
        for i in range(pkg.export_count):
            class_name = pkg.get_export_class_name(i)
            if class_name == "StaticMesh":
                name = os.path.splitext(os.path.basename(filepath))[0]
                return 'mesh', name
            elif class_name in ("Texture2D", "TextureCube", "VolumeTexture"):
                name = os.path.splitext(os.path.basename(filepath))[0]
                return 'texture', name
        return 'other', os.path.splitext(os.path.basename(filepath))[0]
    except Exception:
        return 'other', os.path.splitext(os.path.basename(filepath))[0]


def process_assets(input_dir, export_dir, skip_textures=False, mesh_filter=None):
    """Find and export all meshes as GLB and base color textures as PNG.

    Args:
        mesh_filter: If set, only export meshes whose name contains this
            substring (case-insensitive).  Textures are still fully cached.
    """
    os.makedirs(os.path.join(export_dir, "Meshes"), exist_ok=True)
    os.makedirs(os.path.join(export_dir, "Textures"), exist_ok=True)

    uassets, umaps = find_uasset_files(input_dir)
    print(f"Found {len(uassets)} .uasset files, {len(umaps)} .umap files")

    meshes = []
    textures = []
    others = []

    for filepath in uassets:
        asset_type, name = classify_uasset(filepath, input_dir)
        if asset_type == 'mesh':
            meshes.append((filepath, name))
        elif asset_type == 'texture':
            textures.append((filepath, name))
        else:
            others.append((filepath, name))

    print(f"Classified: {len(meshes)} meshes, {len(textures)} textures, {len(others)} other")

    # Build uasset index for texture resolution (mesh → material → texture chain)
    uasset_index = _build_uasset_index(input_dir)

    # ------------------------------------------------------------------
    # Export base color textures as PNG  (also cache pixel data for GLB)
    # ------------------------------------------------------------------
    # Use pickle cache to avoid re-parsing textures when Input/ is unchanged
    cache_path = os.path.join(export_dir, "texture_cache.pkl")

    # Compute a quick fingerprint of the Input/ texture files
    def _input_fingerprint():
        h = hashlib.md5()
        for fp, n in sorted(textures):
            h.update(n.encode())
            h.update(str(os.path.getmtime(fp)).encode())
            h.update(str(os.path.getsize(fp)).encode())
        return h.hexdigest()

    tex_success = 0
    texture_cache = {}  # texture_name -> numpy RGBA pixels
    tex_name_map = {}

    base_color_textures = [(fp, n) for fp, n in textures]  # export ALL textures
    print(f"Textures to export: {len(base_color_textures)}")

    # Try loading from pickle cache
    fp_hash = _input_fingerprint()
    cache_loaded = False
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            if isinstance(cached, dict) and cached.get("fingerprint") == fp_hash:
                texture_cache = cached["textures"]
                tex_success = len(texture_cache)
                cache_loaded = True
                print(f"  Loaded {tex_success} textures from pickle cache")
        except Exception:
            pass  # Corrupt cache — rebuild

    if not cache_loaded:
        for filepath, name in tqdm(sorted(base_color_textures), desc="Exporting textures", unit="tex"):
            try:
                pkg = Package(filepath)
                texture = Texture2D.from_package(pkg)
                if texture and texture.pixels is not None:
                    if not skip_textures:
                        png_path = os.path.join(export_dir, "Textures", f"{name}.png")
                        export_png(texture, png_path)
                    texture_cache[name] = texture.pixels
                    tex_success += 1
            except Exception as e:
                tqdm.write(f"  {name}: ERROR {e}")

        # Save pickle cache
        try:
            with open(cache_path, "wb") as f:
                pickle.dump({"fingerprint": fp_hash, "textures": texture_cache}, f)
            print(f"  Saved texture cache ({tex_success} textures) to {cache_path}")
        except Exception as e:
            print(f"  Warning: could not save texture cache: {e}")

    if skip_textures:
        print(f"  (PNG file export skipped --skip-textures, {tex_success} textures cached for GLB)")

    # Identity map so _get_base_color_texture_from_material returns the texture name
    tex_name_map = {name: name for name in texture_cache}

    # ------------------------------------------------------------------
    # Export meshes as GLB with embedded textures
    # ------------------------------------------------------------------
    if mesh_filter:
        filtered = [(fp, n) for fp, n in meshes
                    if mesh_filter.lower() in n.lower()]
        print(f"Mesh filter '{mesh_filter}': {len(filtered)} of {len(meshes)} meshes match")
        meshes_to_export = filtered
    else:
        meshes_to_export = meshes

    mesh_success = 0
    for filepath, name in tqdm(sorted(meshes_to_export), desc="Exporting meshes", unit="mesh"):
        try:
            pkg = Package(filepath)
            mesh = StaticMesh.from_package(pkg)
            if mesh and mesh.vertices:
                # Resolve textures for each polygon group via the real
                # material-slot mapping parsed from the StaticMaterials
                # export property.  Each polygon group has an
                # ImportedMaterialSlotName which maps to a concrete material
                # import name through the StaticMaterials array.
                mesh_textures = []

                if mesh.material_slots and mesh.material_slot_names:
                    # Real data path: material_slots is an ordered list of
                    # (slot_name, material_name) indexed by StaticMaterials
                    # slot index.  Use SectionInfoMap to map each polygon
                    # group to the correct material slot index.
                    section_map = getattr(mesh, 'section_info_map', None)
                    for pg_idx, slot_name in enumerate(mesh.material_slot_names):
                        if slot_name is None:
                            continue
                        # Resolve pg_idx → material slot index
                        slot_idx = (section_map[pg_idx]
                                    if section_map and pg_idx < len(section_map)
                                    else pg_idx)
                        if slot_idx < len(mesh.material_slots):
                            mat_name = mesh.material_slots[slot_idx][1]
                        else:
                            mat_name = None
                        if mat_name is None:
                            continue
                        tex_name = _get_base_color_texture_from_material(
                            mat_name, uasset_index, tex_name_map)
                        if tex_name and tex_name in texture_cache:
                            mesh_textures.append(
                                (pg_idx, texture_cache[tex_name]))
                else:
                    # Fallback: no StaticMaterials data — assume polygon
                    # group index == material import index.
                    material_names = _get_material_names_from_mesh(
                        name, uasset_index)
                    for mat_idx, mat_name in enumerate(material_names):
                        tex_name = _get_base_color_texture_from_material(
                            mat_name, uasset_index, tex_name_map)
                        if tex_name and tex_name in texture_cache:
                            mesh_textures.append(
                                (mat_idx, texture_cache[tex_name]))

                glb_path = os.path.join(export_dir, "Meshes", f"{name}.glb")
                export_glb(mesh, glb_path,
                           textures=mesh_textures if mesh_textures else None)
                mesh_success += 1
        except Exception as e:
            tqdm.write(f"  {name}: ERROR {e}")

    print(f"Export complete: {mesh_success} meshes, {tex_success} textures")
    return mesh_success, tex_success


def find_umap_path(input_dir, umap_filename):
    """Find a .umap file in the input directory by filename."""
    uassets, umaps = find_uasset_files(input_dir)
    for u in umaps:
        if os.path.basename(u).lower() == umap_filename.lower():
            return u
    # Try with .uasset extension (umap files might be stored as uasset)
    for u in uassets:
        if os.path.basename(u).lower() == umap_filename.lower().replace('.umap', '.uasset'):
            return u
    return None


def preview_level(input_dir, export_dir, umap_filename):
    """Parse a .umap file and show a 3D preview in the browser."""
    from preview_server import start_server

    # Find the umap file
    umap_path = find_umap_path(input_dir, umap_filename)
    if not umap_path:
        print(f"ERROR: Level file '{umap_filename}' not found in {input_dir}")
        sys.exit(1)

    start_server(
        umap_path=umap_path,
        export_dir=export_dir,
        content_dir=input_dir,
        port=3050,
    )


def main():
    parser = argparse.ArgumentParser(
        description="UE 5.5 UAsset Parser, Exporter, and Level Previewer"
    )
    parser.add_argument(
        'input_dir', nargs='?', default='./Input',
        help='Path to folder with .uproject and Content/ (default: ./Input)'
    )
    parser.add_argument(
        '--preview', metavar='LEVEL.umap',
        help='Parse a .umap level and show 3D preview'
    )
    parser.add_argument(
        '--export-dir', default='./Export',
        help='Output directory (default: ./Export)'
    )
    parser.add_argument(
        '--skip-export', action='store_true',
        help='Skip export step, use existing Export/ directory'
    )
    parser.add_argument(
        '--skip-textures', action='store_true',
        help='Skip texture export and embedding (meshes only, no PNGs)'
    )
    parser.add_argument(
        '--filter', metavar='SUBSTRING', dest='mesh_filter',
        help='Only export meshes whose name contains this substring (case-insensitive)'
    )
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    export_dir = os.path.abspath(args.export_dir)

    if not os.path.isdir(input_dir):
        print(f"ERROR: Input directory not found: {input_dir}")
        sys.exit(1)

    print("=" * 60)
    print("UE 5.5 UAsset Parser, Exporter, and Level Previewer")
    print("=" * 60)
    print(f"Input:  {input_dir}")
    print(f"Output: {export_dir}")

    # Check .uproject
    find_uproject(input_dir)

    # Export assets
    if not args.skip_export:
        process_assets(input_dir, export_dir, skip_textures=args.skip_textures,
                       mesh_filter=args.mesh_filter)
    else:
        print("Skipping export (using existing Export/ directory)")

    # Preview if requested
    if args.preview:
        print(f"\n{'=' * 60}")
        preview_level(input_dir, export_dir, args.preview)


if __name__ == "__main__":
    main()
