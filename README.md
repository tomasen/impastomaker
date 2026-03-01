# ImpastoMaker

Convert 2D paintings into multi-color 3D relief prints for FDM 3D printers.

ImpastoMaker takes any painting or image and generates a multi-color 3MF file with sculpted relief — like a thick impasto oil painting rendered in plastic. Each color is a separate filament, heights follow the painting's tonal values, and brushstroke textures add surface detail.

Designed for **Bambu Lab H2C** with AMS (up to 8 filaments), but the 3MF output works with any multi-material FDM printer supported by Bambu Studio.

![Hokusai's Great Wave — preview](examples/tsunami_3d_preview.png)

## Features

- **Automatic color matching** — Maps image colors to the Bambu PLA Basic filament palette using hue-aware perceptual distance in CIE Lab + HSV color spaces
- **Smart 8-color budget** — Palette diversity check prevents redundant similar shades; ordered dithering (Bayer 4x4) creates intermediate tones from fewer filaments
- **Sculptural relief** — Brightness-based height mapping turns tonal values into physical depth, with configurable contrast and smoothing
- **Brushstroke texture** — Medial-axis swept stroke ridges add painterly surface detail at configurable intensity
- **Paint layer stacking** — Colors stack in Z like real paint (background at bottom, accents on top)
- **Bambu Studio native 3MF** — Output includes filament color assignments, extruder mapping, and project settings recognized by Bambu Studio

## Quick Start

```bash
pip install -r requirements.txt

# Basic usage — outputs a 3MF file ready for Bambu Studio
python impasto_maker.py painting.jpg

# 8 colors with heavy brushstroke texture
python impasto_maker.py painting.jpg -c 8 --texture heavy -o output.3mf

# Custom size and relief settings
python impasto_maker.py painting.jpg --output-width 200 --max-height 8.0 --steepness 3.0
```

## Usage

```
python impasto_maker.py [options] input_image
```

| Option | Default | Description |
|--------|---------|-------------|
| `-c, --colors` | 8 | Number of filament colors |
| `-r, --resolution` | 1525 | Max pixels on longest edge (0.2mm/pixel at 305mm) |
| `--output-width` | 305.0 | Output width in mm (H2C right nozzle limit) |
| `--output-height` | auto | Output height in mm (auto-capped at 320mm) |
| `--min-height` | 0.5 | Minimum sculpture height in mm |
| `--max-height` | 5.0 | Maximum sculpture height in mm |
| `--layer-step` | 0.4 | Z offset between paint layers in mm |
| `--canvas` | 0.6 | Canvas base plate thickness in mm |
| `--strategy` | brightness | Height mapping: `brightness`, `frequency`, or `equal` |
| `--texture` | none | Brushstroke amplitude: `none`, `light`, `medium`, `heavy`, `extreme`, or mm value |
| `--steepness` | 2.0 | Relief contrast exponent |
| `--smooth` | auto | Gaussian smoothing sigma |
| `--no-preview` | | Skip preview PNG generation |
| `-o, --output` | auto | Output 3MF file path |

## How It Works

1. **Color quantization** — The image is quantized to `n` colors from the Bambu PLA Basic palette using a greedy selection algorithm with hue-aware distance. A diversity check replaces redundant similar colors, and ordered dithering blends close shades spatially.

2. **Height mapping** — Pixel brightness maps to physical relief height. Brighter areas are taller, creating a bas-relief effect. The `--steepness` parameter controls contrast.

3. **Brushstroke generation** — For each color region, the medial axis (skeleton) is computed. Stroke ridges are swept along the skeleton with rounded cross-sections, adding painterly texture.

4. **Mesh generation** — Each color becomes a separate 3D mesh. Colors stack in Z order (most common at bottom, least common on top), with the canvas base plate sharing the background color's filament slot.

5. **3MF packaging** — Meshes are packaged in Bambu Studio's native 3MF format with proper filament color definitions, extruder assignments, and project settings.

## Examples

**Hokusai — The Great Wave off Kanagawa**

![Great Wave preview](examples/tsunami_3d_preview.png)

**Pollock — Number 1, 1949**

![Pollock preview](examples/pollock1_3d_preview.png)

## Requirements

- Python 3.8+
- OpenCV, NumPy, Pillow, SciPy, scikit-image

```bash
pip install -r requirements.txt
```

## License

MIT
