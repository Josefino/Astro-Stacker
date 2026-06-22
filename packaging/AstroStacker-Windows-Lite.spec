# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

from PyInstaller.utils.hooks import collect_all


ROOT = Path(SPECPATH).resolve().parent

datas = [
    (str(ROOT / "AS_balance_icon.png"), "."),
    (str(ROOT / "AstroStacker_intro.png"), "."),
]
for manual in ("MANUAL_EN.html", "MANUAL_CZ.html"):
    manual_path = ROOT / manual
    if not manual_path.is_file():
        manual_path = ROOT / "AS_Stacker_PI_Plugin" / manual
    if manual_path.is_file():
        datas.append((str(manual_path), "Documentation"))
binaries = []
hiddenimports = [
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "xisf",
    "lz4.frame",
    "zstandard",
]

for package in ("xisf", "lz4", "zstandard"):
    package_datas, package_binaries, package_hidden = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hidden

a = Analysis(
    [str(ROOT / "astro_stacker_app.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    runtime_hooks=[str(ROOT / "packaging" / "runtime_lite.py")],
    excludes=[
        "onnxruntime",
        "onnxruntime_gpu",
        "cupy",
        "cupyx",
        "cupy_backends",
        "nvidia",
        "torch",
    ],
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
    name="AstroStacker_Lite",
)
