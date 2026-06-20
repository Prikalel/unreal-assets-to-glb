# UE 5.5 UAsset Parser & Exporter

Extracts static meshes (glb) and base color textures (PNG) from Unreal Engine 5.5 `.uasset` files. Supports browser-based level preview with `--preview`.

installation:

```
pip install unreal-assets-to-glb        # gets the latest = for UE 5.5 assets (pip package version 5.5.0)
pip install unreal-assets-to-glb==4.27.2.0   # if you want to parse UE 4.27 assets
```

## Example

### Here is how the asset looks in unreal:

<img width="894" height="488" alt="пример" src="https://github.com/user-attachments/assets/14148b30-db4b-4921-8f14-431a75b60231" />

### Here is how it looks in preview:

https://github.com/user-attachments/assets/b8468b8b-f110-46e9-b798-d624e33d885f

You probably may notice some textures applied wrong thats because there was a little bug that already fixed, the video is made for olden version plus you can find the limitations at the end of that page but I will list some already now:

- complex materials with graph-based shaders not supported
- Only for Unreal 5.5 / 4.27.2, usage arguents shown below

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
