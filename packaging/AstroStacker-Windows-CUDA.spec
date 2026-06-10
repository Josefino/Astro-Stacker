# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

from PyInstaller.utils.hooks import collect_all


ROOT = Path(SPECPATH).resolve().parent

datas = [
    (str(ROOT / "AS_balance_icon.png"), "."),
    (str(ROOT / "AstroStacker_intro.png"), "."),
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
]

for package in ("xisf", "lz4", "zstandard", "cupy", "cupyx", "cupy_backends"):
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
