# UE 5.5 UAsset Parser & Exporter

Extracts static meshes (OBJ) and base color textures (PNG) from Unreal Engine 5.5 `.uasset` files. Supports browser-based level preview with `--preview`.

## Requirements

- Python 3.10+
- numpy
- Pillow
- ooz-python
- tqdm

```
pip install numpy Pillow ooz-python tqdm
```

## Usage

```bash
# Extract meshes and textures
python main.py ./Input

# Extract and preview a level in browser (port 3050)
python main.py ./Input --preview L_Showcase.umap

# Preview without re-exporting
python main.py ./Input --skip-export --preview L_Showcase.umap
```

The input directory should contain a `.uproject` file and a `Content/` folder with `.uasset` / `.umap` files.
