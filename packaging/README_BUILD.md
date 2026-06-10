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
2. Builds `AstroStacker_CPU`.
3. Installs CuPy with its CUDA 12 runtime packages.
4. Builds `AstroStacker_CUDA`.
5. Creates one installer with an **Install NVIDIA CUDA GPU support** checkbox.

Result:

```text
release\AstroStacker28_Setup.exe
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
release/AstroStacker28_macOS.dmg
```

The script installs PyTorch so that Apple Metal/MPS GPU stacking is available.
The resulting build matches the architecture of the Python interpreter used to
run the build.

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
- `AstroStacker-Windows-CPU.spec`: CPU-only PyInstaller build.
- `AstroStacker-Windows-CUDA.spec`: NVIDIA CUDA PyInstaller build.
- `AstroStacker.iss`: Inno Setup installer.
- `AstroStacker-macOS.spec`: macOS PyInstaller app bundle.
- `macos-entitlements.plist`: hardened-runtime permissions needed by PyTorch.
- `make_icons.py`: creates Windows ICO and macOS ICNS files.
