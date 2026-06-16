#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
PACKAGING="$ROOT/packaging"
VENV="$ROOT/.venv-build-macos"
PY="$VENV/bin/python3"
DIST="$ROOT/dist_installer_macos"
WORK="$ROOT/build_installer_macos"
RELEASE="$ROOT/release"
STAGING="$WORK/dmg"
DMG="$RELEASE/AstroStacker30_macOS.dmg"

echo "============================================================"
echo "Astro Stacker 3.0 - macOS DMG build"
echo "============================================================"
echo

if [ ! -f "$ROOT/astro_stacker_app.py" ]; then
  echo "ERROR: astro_stacker_app.py was not found."
  exit 1
fi
for MODEL in drunet_color.onnx drunet_gray.onnx cosmic_clarity_stellar.onnx; do
  if [ ! -f "$ROOT/models/$MODEL" ]; then
    echo "ERROR: models/$MODEL was not found."
    exit 1
  fi
done
if [ ! -f "$ROOT/models/COSMIC_CLARITY_STELLAR_LICENSE.txt" ]; then
  echo "ERROR: models/COSMIC_CLARITY_STELLAR_LICENSE.txt was not found."
  exit 1
fi
for REQUIRED in \
  "$ROOT/MANUAL_EN.html" \
  "$ROOT/MANUAL_CZ.html" \
  "$ROOT/AS_Stacker_PI_Plugin/AS_Stacker_PI.js" \
  "$ROOT/AS_Stacker_PI_Plugin/astro_stacker_cli.py" \
  "$ROOT/AS_Stacker_PI_Plugin/astro_stacker_app.py"; do
  if [ ! -f "$REQUIRED" ]; then
    echo "ERROR: required release file was not found: $REQUIRED"
    exit 1
  fi
done
if ! cmp -s "$ROOT/astro_stacker_app.py" "$ROOT/AS_Stacker_PI_Plugin/astro_stacker_app.py"; then
  echo "ERROR: PixInsight wrapper contains an outdated astro_stacker_app.py"
  exit 1
fi
if ! cmp -s "$ROOT/astro_stacker_cli.py" "$ROOT/AS_Stacker_PI_Plugin/astro_stacker_cli.py"; then
  echo "ERROR: PixInsight wrapper contains an outdated astro_stacker_cli.py"
  exit 1
fi

if [ ! -x "$PY" ]; then
  echo "Creating build environment..."
  python3 -m venv "$VENV"
fi

"$PY" -m pip install --upgrade pip setuptools wheel
"$PY" -m pip install --upgrade -r "$PACKAGING/requirements-base.txt"
"$PY" -m pip install --upgrade torch

"$PY" - <<'PYCODE'
from pathlib import Path
import onnxruntime as ort

root = Path.cwd()
model = root / "models" / "cosmic_clarity_stellar.onnx"
session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
shape = list(session.get_inputs()[0].shape)
if shape != [1, 3, 256, 256]:
    raise SystemExit(
        f"ERROR: unexpected AI Star Deconvolution model input shape: {shape}"
    )
print(
    "AI Star Deconvolution ONNX preflight passed:",
    model.name,
    shape,
)
PYCODE

"$PY" "$PACKAGING/make_icons.py"

BUILD_ARCH="$("$PY" -c 'import platform; print(platform.machine())')"
HOST_ARCH="$(uname -m)"
echo "Build architecture: Python=$BUILD_ARCH, host=$HOST_ARCH"
if [ "$HOST_ARCH" = "arm64" ] && [ "$BUILD_ARCH" != "arm64" ]; then
  echo "ERROR: Apple Silicon build is using a non-arm64 Python interpreter."
  echo "Install native arm64 Python and remove .venv-build-macos before rebuilding."
  exit 1
fi

rm -rf "$DIST" "$WORK"
mkdir -p "$DIST" "$WORK" "$RELEASE"

"$PY" -m PyInstaller --noconfirm --clean \
  --distpath "$DIST" \
  --workpath "$WORK/pyinstaller" \
  "$PACKAGING/AstroStacker-macOS.spec"

APP="$DIST/AstroStacker.app"
if [ ! -d "$APP" ]; then
  echo "ERROR: PyInstaller did not create AstroStacker.app"
  exit 1
fi

for REQUIRED_NAME in \
  drunet_color.onnx \
  drunet_gray.onnx \
  cosmic_clarity_stellar.onnx \
  COSMIC_CLARITY_STELLAR_LICENSE.txt \
  MANUAL_EN.html \
  MANUAL_CZ.html \
  AS_Stacker_PI.js; do
  if ! find "$APP" -type f -name "$REQUIRED_NAME" -print -quit | grep -q .; then
    echo "ERROR: AstroStacker.app does not contain $REQUIRED_NAME"
    exit 1
  fi
done

STELLAR_BUNDLED="$(
  find "$APP" -type f -name "cosmic_clarity_stellar.onnx" -print -quit
)"
if [ -z "$STELLAR_BUNDLED" ]; then
  echo "ERROR: AI Star Deconvolution model path was not found in AstroStacker.app"
  exit 1
fi
if ! cmp -s "$ROOT/models/cosmic_clarity_stellar.onnx" "$STELLAR_BUNDLED"; then
  echo "ERROR: bundled AI Star Deconvolution model differs from the source model"
  exit 1
fi
echo "AI Star Deconvolution payload verified:"
echo "  $STELLAR_BUNDLED"
echo "Application payload verification passed."

SIGN_IDENTITY="${CODESIGN_IDENTITY:--}"
echo "Signing app with identity: $SIGN_IDENTITY"
codesign --force --deep --options runtime \
  --entitlements "$PACKAGING/macos-entitlements.plist" \
  --sign "$SIGN_IDENTITY" "$APP"
codesign --verify --deep --strict --verbose=2 "$APP"

mkdir -p "$STAGING"
cp -R "$APP" "$STAGING/"
ln -s /Applications "$STAGING/Applications"

rm -f "$DMG"
hdiutil create \
  -volname "Astro Stacker 3.0" \
  -srcfolder "$STAGING" \
  -ov \
  -format UDZO \
  "$DMG"

if [ -n "${APPLE_NOTARY_PROFILE:-}" ]; then
  if [ "$SIGN_IDENTITY" = "-" ]; then
    echo "ERROR: notarization requires CODESIGN_IDENTITY with a Developer ID Application certificate."
    exit 1
  fi
  codesign --force --sign "$SIGN_IDENTITY" "$DMG"
  echo "Submitting DMG for Apple notarization..."
  xcrun notarytool submit "$DMG" --keychain-profile "$APPLE_NOTARY_PROFILE" --wait
  xcrun stapler staple "$DMG"
fi

hdiutil verify "$DMG"

echo
echo "Build complete:"
echo "  $DMG"
echo
if [ "$SIGN_IDENTITY" = "-" ]; then
  echo "The DMG is ad-hoc signed. For public distribution, configure"
  echo "CODESIGN_IDENTITY and APPLE_NOTARY_PROFILE as described in README_BUILD.md."
fi
