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
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "xisf",
    "lz4.frame",
    "zstandard",
    "torch",
    "onnxruntime",
]
for package in ("xisf", "lz4", "zstandard", "onnxruntime"):
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
        "CFBundleShortVersionString": "3.0",
        "CFBundleVersion": "300",
        "NSHighResolutionCapable": True,
    },
)
