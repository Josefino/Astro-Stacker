# Astro Stacker

Astro Stacker is a desktop application for astronomical image stacking, calibration, preview stretching, and basic post-processing. It is designed for deep-sky image sequences from DSLR/mirrorless cameras, dedicated astro cameras, and smart telescopes.

The application focuses on a practical workflow: load a folder of light frames, optionally apply calibration frames, align the sequence, stack it, inspect rejected frames, and export either a linear FIT/FITS result or a stretched visual image.

Current application version: **3.1**

## Main Features

- Light-frame stacking with translation, ECC affine, star alignment + RANSAC, comet alignment, optional expanded-canvas mosaic mode, and calibration-frame stacking.
- Mean, median, and sigma-clipped mean stacking.
- Robust star alignment with frame rejection when alignment fails.
- Automatic or manual reference-frame selection.
- Optional frame-quality review before stacking, including manual exclusion with the Space key.
- Frame-quality heading with Light, Dark, Flat, and Bias input totals.
- Optional satellite-trail detection in the frame-quality table.
- Flat, Bias, and Dark calibration.
- Automatic detection of `Flat`, `Bias`, and `Dark` subfolders.
- Cached `MasterFlat_AS.fit`, `MasterBias_AS.fit`, and `MasterDark_AS.fit` files for faster repeated processing.
- Manual calibration selection from a finished Master file or any folder with individual frames.
- RAW only mode to exclude JPG/PNG/BMP/TIFF preview files while keeping XISF, FIT/FITS and camera RAW files.
- Bayer FIT handling with Auto, Mono, RGGB, BGGR, GRBG, and GBRG modes.
- GPU stacking support where available. Aligned frames are streamed to CUDA or Metal/MPS in row tiles to avoid a second full-stack RAM copy.
- CPU multiprocessing and tiled CPU stacking for large datasets.
- ASCOM Alpaca control for one or more cameras, including synchronized capture into a shared Live Stack folder.
- Per-camera file identifiers in file names and FIT metadata for distinguishing frames from identical cameras.
- PixInsight wrapper support.
- Simple and Advanced UI modes.
- Preview tools: Balance, adjustable STF strength, Auto WB, crop, background neutralization, polynomial gradient removal, histogram, flip/rotate, color correction, synthetic flat, vignette correction, SCNR Green, and Astro Denoise.
- Linear FIT/FITS export plus visual TIFF/PNG export.

## Supported Formats

Input formats include:

- XISF, including compressed PixInsight XISF files
- FIT / FITS
- CR2 / CR3 / RAW / NEF / ARW / DNG / ORF / RW2 / RAF
- TIFF / PNG / JPG / BMP

Recommended output formats:

- **XISF** for compressed 32-bit linear data and direct use in PixInsight.
- **FIT/FITS** for linear astronomical data and further processing.
- **TIFF** for 16-bit stretched visual export. External 16-bit TIFF files are also loaded without reducing their tonal depth.
- **PNG** for 8-bit preview/share export.

## Basic Workflow

1. Choose a folder containing Light frames.
2. If the folder contains preview images, enable **RAW only**.
3. Add Flat, Bias, and Dark calibration manually as a Master file or an arbitrary frame folder, or place them in subfolders named `Flat`, `Bias`, and `Dark`.
4. Use **Star alignment + RANSAC** for normal deep-sky sequences.
5. Enable **Review frames before stacking** if you want to inspect quality scores and manually exclude frames.
6. For smart-telescope mosaic sequences, enable **Mosaic - expand canvas** in Advanced mode.
7. Start stacking.
8. Use the right panel to adjust the preview.
9. Save a linear FIT/FITS file for processing, or export a stretched TIFF/PNG image.

## ASCOM Alpaca Cameras

Open **Camera > ASCOM Alpaca cameras...** to control one camera or a synchronized
group. Discover the cameras, select them in the **Use** column, connect them,
set exposure and cooling parameters, and select a shared output folder.

The **Camera/ID file** column shows the camera together with the identifier used
in generated file names and the FIT `CAMID` keyword. This keeps frames from
several identical cameras distinguishable while allowing all cameras to feed
the same Live Stack folder.

Alpaca support uses standard network components and does not require an extra
Python package.

## Mosaic Mode

**Mosaic - expand canvas** preserves aligned image areas outside the reference frame and creates an output larger than the native camera resolution. It is intended for smart-telescope mosaic sequences such as Vespera captures.

Partially covered mosaic edges are ignored during pixel integration instead of being treated as black pixels. With GPU enabled, mosaic integration uploads row tiles directly to CUDA or Apple Metal/MPS VRAM. If GPU processing is unavailable, Astro Stacker uses the RAM-protected parallel tiled CPU path.

## Calibration

Astro Stacker supports standard calibration with Flat, Bias, and Dark frames.

When automatic calibration folders are used, the application creates cached master files:

- `MasterFlat_AS.fit`
- `MasterBias_AS.fit`
- `MasterDark_AS.fit`

These files are reused on later runs if the source calibration folder has not changed. The **Clear cache quality frames** command can remove cache files when needed.

The right-panel Flat, Bias, and Dark buttons also accept an arbitrary folder with individual calibration frames. The folder does not need a special name. Astro Stacker integrates it and stores a reusable Master cache directly inside it.

Bias, Flat, and Dark master frames are integrated with an arithmetic average to use the signal from all calibration exposures.

The flat calibration path subtracts the bias before normalizing the correction:

```text
calibrated = (Light - Bias/Dark) / normalized(MasterFlat - MasterBias)
```

For most workflows, this preserves image brightness while correcting vignetting and dust shadows.

## Preview and Post-Processing

The right panel changes the visual preview and visual exports, not the linear FIT output.
Both side panels can be collapsed with the small arrow in the top-left corner of
each panel, leaving more room for the preview and Frame Quality table.

Important tools:

- **Balance**: automatic preview stretch and background balance.
- **Auto WB**: automatic white balance.
- **Crop edges**: crop border artifacts after alignment.
- **Select crop**: drag a rectangle in the preview and keep only the selected area.
- **Neutralize background**: reduce color cast in the background.
- **Remove gradient**: subtract a smooth polynomial background model. This works especially well for galaxies; use it carefully with large nebulae.
- **Synthetic flat**: approximate smooth background correction when no real flat is available.
- **Color background correction**: remove strong color casts, useful for smart-telescope data.
- **Astro Denoise**: Classic CPU denoising or AI DRUNet ONNX denoising with an adaptive cached noise map.
- **AI Star Deconvolution**: cached stellar-model sharpening with an adjustable star-only mask. It affects preview and PNG/TIFF export while linear FIT remains unchanged.
- **Show stacked image**: return to the original stacked image without preview edits.

## Comet Stacking

Astro Stacker can stack on a moving comet. For difficult sequences, mark the comet in the first and last frame with:

- **Comet First**
- **Comet Last**
- **Comet Clear**

The application interpolates comet motion between the two marked frames and can refine the comet position locally.

## PixInsight Wrapper

The `AS_Stacker_PI_Plugin` folder contains a PixInsight wrapper script, CLI bridge, current stacking engine, both bundled DRUNet models, and the AI Star Deconvolution stellar model. The wrapper allows Astro Stacker processing from inside PixInsight and can open the resulting linear FIT file after completion.

Main files:

- `AS_Stacker_PI_Plugin/AS_Stacker_PI.js`
- `AS_Stacker_PI_Plugin/astro_stacker_cli.py`
- `AS_Stacker_PI_Plugin/astro_stacker_app.py`

See `AS_Stacker_PI_Plugin/README_INSTALL.txt` for wrapper installation notes.

## Running from Source

Install Python 3 and the required packages in your Python environment.

Typical dependencies include:

```bash
pip install numpy opencv-python pillow astropy rawpy PySide6 xisf
```

Optional GPU-related packages depend on your platform and hardware. For a
Windows source installation with NVIDIA CUDA, install the CTK extra so the
required runtime DLL packages are included:

```bash
pip install "cupy-cuda12x[ctk]" onnxruntime-gpu
```

For DRUNet ONNX denoising, install ONNX Runtime. The distribution contains:

- `models/drunet_color.onnx` for RGB images
- `models/drunet_gray.onnx` for monochrome images
- `models/cosmic_clarity_stellar.onnx` for AI Star Deconvolution

The application selects the color model automatically. A different bundled
model can be selected from the right panel:

```bash
pip install onnxruntime
```

Run:

```bash
python astro_stacker_app.py
```

## Building Standalone Packages

Ready-to-run installer build scripts are available in the `packaging` folder.
They create:

- `AstroStacker31_Setup.exe` for Windows, with optional NVIDIA CUDA support.
- `AstroStacker31_Lite_Setup.exe` for the compact Windows CPU-only edition.
- `AstroStacker31_macOS.dmg` for macOS, with Apple Metal/MPS support.

See [`packaging/README_BUILD.md`](packaging/README_BUILD.md) for the exact
requirements, commands, CUDA packaging, macOS signing, and notarization.


## Documentation

Full user manuals are included:

- `MANUAL_EN.html`
- `MANUAL_CZ.html`

## Author

Astro Stacker by **Josef Ladra**.

<img width="1800" height="971" alt="AS23" src="https://github.com/user-attachments/assets/af08d9dc-7c84-4192-934d-a944d8ad5c2c" />
