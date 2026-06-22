AS_Stacker 3.1 PixInsight Wrapper
=================================

This folder is a self-contained PixInsight wrapper package.

Included files:
- AS_Stacker_PI.js             PixInsight Feature Script wrapper
- astro_stacker_cli.py         command-line bridge
- astro_stacker_app.py         stacking/calibration/alignment engine
- requirements.txt             Python dependencies
- MANUAL_EN.html               English user manual
- MANUAL_CZ.html               Czech user manual
- models/drunet_color.onnx     DRUNet model for RGB images
- models/drunet_gray.onnx      DRUNet model for monochrome images
- models/cosmic_clarity_stellar.onnx
                                AI Star Deconvolution stellar model
- models/COSMIC_CLARITY_STELLAR_LICENSE.txt
                                Stellar model license and attribution

PixInsight installation:
1. Keep this complete folder together.
2. Open PixInsight.
3. Open SCRIPT > Feature Scripts...
4. Click Add.
5. Select AS_Stacker_PI.js from this folder.
6. Click Done.
7. Run SCRIPT > Utilities > AS_Stacker.

Wrapper features:
- Flat, Bias and Dark master files or folders with individual frames
- comet alignment from first/last frame coordinates
- separate Star + Comet FIT outputs

Python environment:
1. Install Python 3.11 or 3.12.
2. Create a virtual environment if desired.
3. Install the dependencies:

   python -m pip install -r requirements.txt

4. In the wrapper, select the Python executable from that environment.

GPU behavior:
- Apple Silicon: ONNX Runtime prefers CoreML/Metal when available.
- Windows NVIDIA: install onnxruntime-gpu in the selected Python environment
  if CUDA inference is required.
- Without Metal, CUDA, or DirectML, DRUNet runs on the CPU.
- AI Star Deconvolution uses the same ONNX Runtime provider selection.
- The wrapper stacking engine still works when AI denoising is not used.

Notes:
- Keep the models folder next to astro_stacker_app.py.
- RGB images use drunet_color.onnx by default.
- Monochrome processing can use drunet_gray.onnx.
- AI Star Deconvolution uses cosmic_clarity_stellar.onnx.
- CLI stacking writes linear FIT data and does not apply preview-only denoise
  or AI Star Deconvolution to the FIT output.
- Output files start with AS_.
- The wrapper remembers the last Python path, folders, and stacking settings.
- Console output is ASCII because PixInsight can display accented characters
  incorrectly.
