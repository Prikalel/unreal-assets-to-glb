# UE 5.x UAsset Parser & Exporter

Extracts static meshes (glb) and textures (PNG) from Unreal Engine 5.0 to 5.5 `.uasset` files. Supports browser-based level preview with `--preview`. Not requires Unreal engine installation!

This is a fork of [Prikalel/unreal-assets-to-glb](https://github.com/Prikalel/unreal-assets-to-glb); see [Changes in this fork](#changes-in-this-fork).

installation:

```
pip install git+https://github.com/Hellmark/unreal-assets-to-glb@5.5   # this fork, UE 5.0 to 5.5 assets
pip install unreal-assets-to-glb==4.27.2.0   # upstream's package, if you want to parse UE 4.27 assets
```

## Example

### Here is how the asset looks in unreal:

<img width="894" height="488" alt="пример" src="https://github.com/user-attachments/assets/14148b30-db4b-4921-8f14-431a75b60231" />

### Here is how it looks in preview:

https://github.com/user-attachments/assets/b8468b8b-f110-46e9-b798-d624e33d885f

You probably may notice some textures applied wrong thats because there was a little bug that already fixed, the video is made for olden version plus you can find the limitations at the end of that page but I will list some already now:

- complex materials with graph-based shaders not supported
- Only for Unreal 5.0 to 5.5 (this fork) / 4.27.2 (upstream's 4.27.2.0 package), usage arguents shown below

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

# Export meshes only: no separate PNGs (the GLBs still embed their textures)
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

## Changes in this fork

| Change | Details |
| --- | --- |
| UE 5.0 to 5.3 packages | Reads the older property tag layout (before UE 5.4's complete type names) and the SerializationControl byte only where it exists. UE 5.0 packages have no package trailer, so mesh and texture source data is found inline after the exports. Uncompressed ("method 0") compressed buffers are read too. |
| Texture colors | UE stores 8-bit BGRA8 texture sources as PNGs without reordering the channels, so they came out with red and blue swapped. 8-bit RGB/RGBA PNG sources are now swapped back; 16-bit and greyscale sources are left as they are. A quick check: a normal map should average blue-purple (about 128, 128, 255), not salmon. |
| PNG sources in UE 5.2 | A texture source is taken as PNG by its signature, since UE 5.2 sets `CompressionFormat = TSCF_PNG` without the `bPNGCompressed` flag. |
| Base color parameters | Material parameters are matched as base color however the pack spells them ("00_BaseColor", "BaseColour", "Base Color Texture", albedo, diffuse). |
| Memory use and resuming | Textures are written to PNG one at a time and read back when a mesh embeds them (the last few are kept in memory), in place of the pickle cache that held every decoded texture. A large pack no longer runs out of RAM, and a PNG that already exists and verifies is not decoded again, so an interrupted run picks up where it stopped. |
| `--skip-textures` | Writes the PNGs to a temporary folder that is removed afterwards, so the GLBs still embed their textures. |

Tested on a UE 5.0 project (Tire Repair Shop) and a UE 5.2 one (Auto Repair Shop).
