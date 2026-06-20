# UE 5.5 UAsset Parser & Exporter

Extracts static meshes (glb) and base color textures (PNG) from Unreal Engine 5.5 `.uasset` files. Supports browser-based level preview with `--preview`.

## Requirements

- Python 3.10+
- numpy
- Pillow
- pyooz (provides the `ooz` Oodle-decompression module)
- pygltflib
- tqdm

## Usage

```bash
# Extract meshes and textures
python main.py ./Input

# Extract and preview a level in browser (port 3050)
python main.py ./Input --preview L_Showcase.umap

# Preview without re-exporting
python main.py ./Input --skip-export --preview L_Showcase.umap

# Export without textures
python main.py ./Input --skip-textures

# Export only meshes with certain filesname (all containing Pipe in name)
python main.py ./Input --filter Pipe
```

The input directory should contain a `.uproject` file and a `Content/` folder with `.uasset` / `.umap` files.

The output is `./Export` folder created in current workspace.

## Features

- level parts (level isntansing) included in preview
- material parent recursive search
- material slot indexes recognition
- only 1 UV channel
- texture override in material instance

## Not included

- lights,
- colliders,
- PBR textures,
- vertex colors,
- tangents,
- LOD
- shaders,
- texture baking,
- Nanite,
- animation decompression
- skeletons/bones
