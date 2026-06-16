# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

from PyInstaller.utils.hooks import collect_all


ROOT = Path(SPECPATH).resolve().parent

datas = [
    (str(ROOT / "AS_balance_icon.png"), "."),
    (str(ROOT / "AstroStacker_intro.png"), "."),
    (str(ROOT / "models" / "drunet_color.onnx"), "models"),
    (str(ROOT / "models" / "drunet_gray.onnx"), "models"),
    (str(ROOT / "models" / "cosmic_clarity_stellar.onnx"), "models"),
    (str(ROOT / "models" / "COSMIC_CLARITY_STELLAR_LICENSE.txt"), "models"),
    (str(ROOT / "MANUAL_EN.html"), "Documentation"),
    (str(ROOT / "MANUAL_CZ.html"), "Documentation"),
    (str(ROOT / "AS_Stacker_PI_Plugin" / "AS_Stacker_PI.js"), "AS_Stacker_PI_Plugin"),
    (str(ROOT / "AS_Stacker_PI_Plugin" / "astro_stacker_cli.py"), "AS_Stacker_PI_Plugin"),
    (str(ROOT / "AS_Stacker_PI_Plugin" / "astro_stacker_app.py"), "AS_Stacker_PI_Plugin"),
    (str(ROOT / "AS_Stacker_PI_Plugin" / "README_INSTALL.txt"), "AS_Stacker_PI_Plugin"),
    (str(ROOT / "AS_Stacker_PI_Plugin" / "requirements.txt"), "AS_Stacker_PI_Plugin"),
]
binaries = []
hiddenimports = [
    "graphlib",
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "xisf",
    "lz4.frame",
    "zstandard",
    "onnxruntime",
]

for package in ("xisf", "lz4", "zstandard", "onnxruntime", "cupy", "cupyx", "cupy_backends"):
    package_datas, package_binaries, package_hidden = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hidden

# CuPy's CTK wheels place runtime DLLs and supporting data in these NVIDIA
# namespace packages. Keep their directory layout so CuPy can resolve all
# runtime components in the frozen application.
for package in (
    "nvidia.cuda_runtime",
    "nvidia.cuda_nvrtc",
    "nvidia.cublas",
    "nvidia.nvjitlink",
    "nvidia.cuda_cccl",
):
    try:
        package_datas, package_binaries, package_hidden = collect_all(package)
        datas += package_datas
        binaries += package_binaries
        hiddenimports += package_hidden
    except Exception:
        pass

a = Analysis(
    [str(ROOT / "astro_stacker_app.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["torch"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AstroStacker",
    debug=False,
    strip=False,
    upx=False,
    console=False,
    icon=str(ROOT / "packaging" / "icons" / "AstroStacker.ico"),
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="AstroStacker_CUDA",
)
