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
DMG="$RELEASE/AstroStacker28_macOS.dmg"

echo "============================================================"
echo "Astro Stacker 2.8 - macOS DMG build"
echo "============================================================"
echo

if [ ! -f "$ROOT/astro_stacker_app.py" ]; then
  echo "ERROR: astro_stacker_app.py was not found."
  exit 1
fi

if [ ! -x "$PY" ]; then
  echo "Creating build environment..."
  python3 -m venv "$VENV"
fi

"$PY" -m pip install --upgrade pip setuptools wheel
"$PY" -m pip install --upgrade -r "$PACKAGING/requirements-base.txt"
"$PY" -m pip install --upgrade torch

"$PY" "$PACKAGING/make_icons.py"

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
  -volname "Astro Stacker 2.8" \
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

echo
echo "Build complete:"
echo "  $DMG"
echo
if [ "$SIGN_IDENTITY" = "-" ]; then
  echo "The DMG is ad-hoc signed. For public distribution, configure"
  echo "CODESIGN_IDENTITY and APPLE_NOTARY_PROFILE as described in README_BUILD.md."
fi
