# Astro Stacker Installers

This directory builds end-user installers. Users of the resulting packages do
not need Python or any Python packages.

## Windows installer

### Requirements on the build computer

- 64-bit Windows 10 or Windows 11
- 64-bit Python 3.12
- Internet connection while dependencies are installed
- Inno Setup 6

Install Inno Setup once:

```bat
winget install JRSoftware.InnoSetup
```

Then run from Command Prompt:

```bat
packaging\build_windows_installer.bat
```

If Inno Setup is missing, the build script now installs it automatically
through `winget`. Windows may display a confirmation or administrator prompt.

The script:

1. Creates an isolated `.venv-build-windows` environment.
2. Removes stale CUDA packages and builds a clean `AstroStacker_CPU`.
3. Installs CuPy with CUDA 12 for stacking and DirectML for DRUNet AI denoise.
4. Builds `AstroStacker_CUDA`.
5. Downloads and embeds the current Microsoft Visual C++ v14 x64 Runtime.
6. Verifies the splash image, DRUNet models, manuals, wrapper, stellar AI
   model, and CUDA DLL payloads.
7. Creates one installer with an **Install NVIDIA CUDA GPU support** checkbox.

Result:

```text
release\AstroStacker31_Setup.exe
```

If the CPU and CUDA folders were already built but Inno Setup was missing, do
not rebuild everything. Run:

```bat
packaging\create_windows_installer_only.bat
```

This installs/finds Inno Setup and only creates the final installer.

The build computer does not need an NVIDIA GPU. The CUDA application must,
however, be tested on a real Windows computer with an NVIDIA GPU and a current
NVIDIA display driver before publishing.

The installer contains both CPU and CUDA payloads so that the selected version
is complete and independent. This makes the installer larger, but avoids
fragile partial CUDA installations.
The bundled Microsoft Visual C++ Runtime installer is invoked silently. It
updates an older runtime when needed and otherwise completes without changing
the system.

## Windows Lite installer

For a significantly smaller CPU-only package without CUDA, PyTorch, ONNX
Runtime, DRUNet models, or the PixInsight wrapper, run:

```bat
packaging\build_windows_lite_installer.bat
```

Result:

```text
release\AstroStacker31_Lite_Setup.exe
```

The Lite interface hides the GPU option and AI DRUNet method. Classic denoise
and all ordinary CPU stacking and editing functions remain available. The
build script also checks that no ONNX models, ONNX Runtime, CuPy, CUDA, or
PyTorch binaries leaked into the final package.

## macOS DMG

### Requirements on the build computer

- Apple Silicon Mac for an arm64 release, or Intel Mac for an x86_64 release
- Python 3
- `hdiutil`, `codesign`, `sips`, and `iconutil` from macOS

Run:

```bash
chmod +x packaging/build_macos_dmg.command
./packaging/build_macos_dmg.command
```

Result:

```text
release/AstroStacker31_macOS.dmg
```

The script installs PyTorch so that Apple Metal/MPS GPU stacking is available.
The common requirements install ONNX Runtime for DRUNet AI denoising. Both
`models/drunet_color.onnx` and `models/drunet_gray.onnx` are embedded in the
standalone application, together with the stellar AI deconvolution model, the
startup splash image, the English/Czech HTML manuals, and the PixInsight
wrapper package.
The resulting build matches the architecture of the Python interpreter used to
run the build.
The build also verifies the Python/host architecture and checks the complete
application payload before signing. The finished DMG is verified with
`hdiutil verify`.

### Public signed and notarized DMG

An Apple Developer account and a **Developer ID Application** certificate are
required to avoid Gatekeeper warnings on other Macs.

Store notarization credentials once:

```bash
xcrun notarytool store-credentials "AstroStackerNotary" \
  --apple-id "YOUR_APPLE_ID" \
  --team-id "YOUR_TEAM_ID" \
  --password "APP_SPECIFIC_PASSWORD"
```

Build with:

```bash
export CODESIGN_IDENTITY="Developer ID Application: YOUR NAME (TEAMID)"
export APPLE_NOTARY_PROFILE="AstroStackerNotary"
./packaging/build_macos_dmg.command
```

Without these variables, the script creates an ad-hoc signed DMG suitable for
local testing. Users may then need to remove quarantine manually.

## Files

- `requirements-base.txt`: common application/build dependencies.
- `requirements-windows-cuda.txt`: optional Windows CUDA dependencies.
- `requirements-windows-lite.txt`: minimal Windows CPU dependencies.
- `AstroStacker-Windows-CPU.spec`: CPU-only PyInstaller build.
- `AstroStacker-Windows-CUDA.spec`: NVIDIA CUDA PyInstaller build.
- `AstroStacker-Windows-Lite.spec`: small CPU build without AI/CUDA.
- `AstroStacker.iss`: Inno Setup installer.
- `AstroStacker-Lite.iss`: Inno Setup definition for the Lite installer.
- `AstroStacker-macOS.spec`: macOS PyInstaller app bundle.
- `macos-entitlements.plist`: hardened-runtime permissions needed by PyTorch.
- `make_icons.py`: creates Windows ICO and macOS ICNS files.
- `../AstroStacker_intro.png`: startup splash image.
- `../MANUAL_EN.html`: English user guide.
- `../MANUAL_CZ.html`: Czech user guide.
- `../AS_Stacker_PI_Plugin/`: PixInsight wrapper package.
- `../models/drunet_color.onnx`: bundled RGB DRUNet model.
- `../models/drunet_gray.onnx`: bundled monochrome DRUNet model.
- `../models/cosmic_clarity_stellar.onnx`: bundled stellar AI deconvolution model.
- `../models/COSMIC_CLARITY_STELLAR_LICENSE.txt`: license notice for the stellar model.
