#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "Astro Stacker macOS installer"
echo "Working folder: $(pwd)"
echo

if [ ! -f "astro_stacker_app.py" ]; then
  echo "ERROR: astro_stacker_app.py was not found in this folder."
  echo "Put this installer next to astro_stacker_app.py and run it again."
  read -r -p "Press Enter to close..."
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 was not found."
  if command -v brew >/dev/null 2>&1; then
    echo "Installing Python 3 with Homebrew..."
    brew install python
  else
    echo "Homebrew was not found."
    echo "Install Python 3 from https://www.python.org/downloads/macos/ or install Homebrew first:"
    echo '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
    read -r -p "Press Enter to close..."
    exit 1
  fi
fi

PYTHON_BIN="$(command -v python3)"
echo "Using Python: $PYTHON_BIN"
"$PYTHON_BIN" --version

VENV_DIR=".venv-astrostacker"
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment: $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

PY="$PWD/$VENV_DIR/bin/python3"

echo "Upgrading pip/setuptools/wheel..."
"$PY" -m pip install --upgrade pip setuptools wheel

echo "Installing Astro Stacker dependencies..."
"$PY" -m pip install --upgrade \
  numpy \
  opencv-python \
  pillow \
  astropy \
  rawpy \
  PySide6

echo
read -r -p "Install optional PyTorch for Apple Metal/MPS GPU stacking? [y/N] " INSTALL_TORCH
case "$INSTALL_TORCH" in
  y|Y|yes|YES)
    echo "Installing PyTorch..."
    "$PY" -m pip install --upgrade torch
    ;;
  *)
    echo "Skipping PyTorch."
    ;;
esac

echo
echo "Verifying imports..."
"$PY" - <<'PYCODE'
import cv2
import numpy
from PIL import Image
from astropy.io import fits
import rawpy
from PySide6.QtWidgets import QApplication
print("Base dependencies OK")
try:
    import torch
    print("PyTorch OK:", torch.__version__)
except Exception as exc:
    print("PyTorch not installed or not available:", exc)
PYCODE

cat > run_astro_stacker_macos.command <<'EOF'
#!/bin/bash
cd "$(dirname "$0")"
exec "$PWD/.venv-astrostacker/bin/python3" "$PWD/astro_stacker_app.py"
EOF
chmod +x run_astro_stacker_macos.command

echo
echo "Installation complete."
echo "Run Astro Stacker with:"
echo "  ./run_astro_stacker_macos.command"
echo
read -r -p "Press Enter to close..."
