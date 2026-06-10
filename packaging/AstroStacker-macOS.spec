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
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "xisf",
    "lz4.frame",
    "zstandard",
    "torch",
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
    excludes=["cupy", "cupyx", "cupy_backends", "nvidia"],
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
    argv_emulation=False,
    icon=str(ROOT / "packaging" / "icons" / "AstroStacker.icns"),
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="AstroStacker",
)
app = BUNDLE(
    coll,
    name="AstroStacker.app",
    icon=str(ROOT / "packaging" / "icons" / "AstroStacker.icns"),
    bundle_identifier="cz.josefladra.astrostacker",
    info_plist={
        "CFBundleName": "Astro Stacker",
        "CFBundleDisplayName": "Astro Stacker",
        "CFBundleShortVersionString": "2.8",
        "CFBundleVersion": "28",
        "NSHighResolutionCapable": True,
    },
)
