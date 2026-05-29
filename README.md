# Astro Stacker

Astro Stacker is a desktop application for astronomical image stacking, calibration, preview stretching, and basic post-processing. It is designed for deep-sky image sequences from DSLR/mirrorless cameras, dedicated astro cameras, and smart telescopes.

The application focuses on a practical workflow: load a folder of light frames, optionally apply calibration frames, align the sequence, stack it, inspect rejected frames, and export either a linear FIT/FITS result or a stretched visual image.

Current application version: **2.3**

## Main Features

- Light-frame stacking with translation, ECC affine, star alignment + RANSAC, sequential alignment, comet alignment, and calibration-frame stacking.
- Mean, median, and sigma-clipped mean stacking.
- Robust star alignment with frame rejection when alignment fails.
- Automatic or manual reference-frame selection.
- Optional frame-quality review before stacking, including manual exclusion with the Space key.
- Optional satellite-trail detection in the frame-quality table.
- Flat, Bias, and Dark calibration.
- Automatic detection of `Flat`, `Bias`, and `Dark` subfolders.
- Cached `MasterFlat_AS.fit`, `MasterBias_AS.fit`, and `MasterDark_AS.fit` files for faster repeated processing.
- Manual calibration-frame selection.
- RAW only mode to exclude JPG/PNG/BMP/TIFF preview files while keeping FIT/FITS and camera RAW files.
- Bayer FIT handling with Auto, Mono, RGGB, BGGR, GRBG, and GBRG modes.
- GPU stacking support where available.
- CPU multiprocessing and tiled CPU stacking for large datasets.
- PixInsight wrapper support.
- Simple and Advanced UI modes.
- Preview tools: Balance, Auto WB, crop, background neutralization, histogram, flip/rotate, color correction, synthetic flat, vignette correction, SCNR Green, highlight compression, and Astro Denoise.
- Linear FIT/FITS export plus visual TIFF/PNG export.

## Supported Formats

Input formats include:

- FIT / FITS
- CR2 / CR3 / RAW / NEF / ARW / DNG / ORF / RW2 / RAF
- TIFF / PNG / JPG / BMP

Recommended output formats:

- **FIT/FITS** for linear astronomical data and further processing.
- **TIFF** for 16-bit stretched visual export.
- **PNG** for 8-bit preview/share export.

## Basic Workflow

1. Choose a folder containing Light frames.
2. If the folder contains preview images, enable **RAW only**.
3. Add Flat, Bias, and Dark frames manually, or place them in subfolders named `Flat`, `Bias`, and `Dark`.
4. Use **Star alignment + RANSAC** for normal deep-sky sequences.
5. Enable **Review frames before stacking** if you want to inspect quality scores and manually exclude frames.
6. Start stacking.
7. Use the right panel to adjust the preview.
8. Save a linear FIT/FITS file for processing, or export a stretched TIFF/PNG image.

## Calibration

Astro Stacker supports standard calibration with Flat, Bias, and Dark frames.

When automatic calibration folders are used, the application creates cached master files:

- `MasterFlat_AS.fit`
- `MasterBias_AS.fit`
- `MasterDark_AS.fit`

These files are reused on later runs if the source calibration folder has not changed. The **Clear cache quality frames** command can remove cache files when needed.

The flat calibration path uses a normalized master flat:

```text
calibrated = (Light - Bias/Dark) / normalized(Flat - Bias)
```

For most workflows, this preserves image brightness while correcting vignetting and dust shadows.

## Preview and Post-Processing

The right panel changes the visual preview and visual exports, not the linear FIT output.

Important tools:

- **Balance**: automatic preview stretch and background balance.
- **Auto WB**: automatic white balance.
- **Crop edges**: crop border artifacts after alignment.
- **Neutralize background**: reduce color cast in the background.
- **Synthetic flat**: approximate smooth background correction when no real flat is available.
- **Color background correction**: remove strong color casts, useful for smart-telescope data.
- **Astro Denoise**: gentle denoising with star and nebula-structure protection.
- **Show stacked image**: return to the original stacked image without preview edits.

## Comet Stacking

Astro Stacker can stack on a moving comet. For difficult sequences, mark the comet in the first and last frame with:

- **Comet First**
- **Comet Last**
- **Comet Clear**

The application interpolates comet motion between the two marked frames and can refine the comet position locally.

## PixInsight Wrapper

The `AS_Stacker_PI_Plugin` folder contains a PixInsight wrapper script and a CLI bridge. The wrapper allows Astro Stacker processing from inside PixInsight and can open the resulting FIT file after completion.

Main files:

- `AS_Stacker_PI_Plugin/AS_Stacker_PI.js`
- `AS_Stacker_PI_Plugin/astro_stacker_cli.py`
- `AS_Stacker_PI_Plugin/astro_stacker_app.py`

See `AS_Stacker_PI_Plugin/README_INSTALL.txt` for wrapper installation notes.

## Running from Source

Install Python 3 and the required packages in your Python environment.

Typical dependencies include:

```bash
pip install numpy opencv-python pillow astropy rawpy PySide6
```

Optional GPU-related packages depend on your platform and hardware.

Run:

```bash
python astro_stacker_app.py
```

## Building Standalone Packages

The project can be packaged with PyInstaller. Example macOS command:

```bash
python -m PyInstaller \
  --windowed \
  --onedir \
  --name AstroStacker \
  --hidden-import PySide6.QtCore \
  --hidden-import PySide6.QtGui \
  --hidden-import PySide6.QtWidgets \
  --add-data "AS_balance_icon.png:." \
  --add-data "AstroStacker_intro.png:." \
  astro_stacker_app.py
```

On Windows, include GPU/CuPy packages only if you intend to distribute NVIDIA GPU support.

## Documentation

Full user manuals are included:

- `MANUAL_EN.md`
- `MANUAL_EN.html`
- `MANUAL_CZ.md`
- `MANUAL_CZ.html`

## Author

Astro Stacker by **Josef Ladra**.

