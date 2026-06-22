"""
Astro Stacker GUI - desktop application for astronomical image stacking.

English overview:
- loads image sequences and standalone preview files: XISF, FIT/FITS, camera RAW,
  TIFF, PNG, JPG, and BMP
- aligns frames by translation, affine ECC, stars + RANSAC, or a moving comet
- stacks with mean, median, sigma-clipped mean, or high-rejection mean
- supports automatic/manual reference selection, quality review, satellite
  trail detection, and large EAA drift
- calibrates Lights with Flat, Bias, and Dark frames before debayering when
  raw sensor data are available
- accepts automatic calibration subfolders, arbitrary manually selected
  calibration folders, or finished Master files
- handles monochrome and Bayer XISF/FIT/FITS data; Auto mode debayers only
  when explicit Bayer/CFA metadata are present
- provides preview adjustments, background tools, zoom, flips, rotations,
  visual PNG/JPG/TIFF export, and linear unstretched FIT/FITS export
- supports CPU multiprocessing and optional NVIDIA CUDA/CuPy or Apple
  Metal/MPS acceleration with an automatic CPU fallback

Cesky prehled:
- nacita sekvence snimku i samostatne nahledy: XISF, FIT/FITS, foto RAW, TIFF, PNG,
  JPG a BMP
- zarovnava snimky posunem, afinnim ECC, podle hvezd + RANSAC nebo na kometu
- sklada prumerem, medianem, sigma-clipped prumerem nebo high-rejection mean
- podporuje automatickou/rucni referenci, kontrolu kvality, detekci
  satelitnich stop a velky EAA drift
- kalibruje Light snimky pomoci Flat, Bias a Dark pred debayeringem, pokud
  jsou dostupna raw senzorova data
- umi automaticke kalibracni podslozky, libovolne rucne vybrane kalibracni
  slozky i hotove Master soubory
- rozlisuje monochromaticke a Bayer XISF/FIT/FITS snimky; rezim Auto
  debayeruje jen pri nalezeni explicitnich Bayer/CFA metadat
- nabizi upravy nahledu, nastroje pro pozadi, zoom, flipy, rotace, vizualni
  PNG/JPG/TIFF export a linearni nestretchovany FIT/FITS export
- podporuje CPU multiprocessing a volitelnou NVIDIA CUDA/CuPy nebo Apple
  Metal/MPS akceleraci s automatickym fallbackem na CPU

Installation / Instalace:
    pip install PySide6 opencv-python numpy pillow astropy rawpy xisf
    optional NVIDIA GPU acceleration / volitelna NVIDIA GPU akcelerace:
    pip install "cupy-cuda12x[ctk]"
    optional Apple Silicon Metal/MPS acceleration / volitelna Apple Metal/MPS akcelerace:
    pip install torch
    optional DRUNet ONNX AI denoising / volitelne AI odsumeni DRUNet ONNX:
    pip install onnxruntime

Run / Spusteni:
    python astro_stacker_app.py
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import sys
import re
import traceback
import multiprocessing as mp
import socket
import warnings
import subprocess
import tempfile
import ctypes
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np
from PIL import Image


LOG_PATH: Optional[Path] = None
APP_VERSION = "3.1"
LITE_BUILD = os.environ.get("ASTRO_STACKER_LITE", "").strip() == "1"
APP_DISPLAY_VERSION = f"{APP_VERSION} Lite" if LITE_BUILD else APP_VERSION
GITHUB_RELEASES_API_URL = "https://api.github.com/repos/Josefino/Astro-Stacker/releases/latest"
GITHUB_RELEASES_PAGE_URL = "https://github.com/Josefino/Astro-Stacker/releases/latest"


def parse_version_tuple(value: str) -> Tuple[int, ...]:
    text = str(value or "").strip().lower()
    text = text.lstrip("v")
    if "." not in text:
        digits_only = re.sub(r"\D+", "", text)
        if len(digits_only) == 2:
            return (int(digits_only[0]), int(digits_only[1]))
        if len(digits_only) == 3:
            return (int(digits_only[0]), int(digits_only[1:]))
    parts = re.findall(r"\d+", text)
    return tuple(int(part) for part in parts) if parts else (0,)


def is_newer_version(candidate: str, current: str) -> bool:
    cand = list(parse_version_tuple(candidate))
    cur = list(parse_version_tuple(current))
    width = max(len(cand), len(cur))
    cand.extend([0] * (width - len(cand)))
    cur.extend([0] * (width - len(cur)))
    return tuple(cand) > tuple(cur)


def display_version(value: str) -> str:
    parsed = parse_version_tuple(value)
    if parsed == (0,):
        return str(value or "").strip().lstrip("v") or "?"
    if len(parsed) >= 2:
        return f"{parsed[0]}.{parsed[1]}"
    return str(parsed[0])


def init_log_path() -> Path:
    global LOG_PATH
    if LOG_PATH is not None:
        return LOG_PATH

    candidates = []
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).resolve().parent)
    candidates.append(Path.cwd())
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        candidates.append(Path(local_appdata) / "AstroStacker")

    for folder in candidates:
        try:
            folder.mkdir(parents=True, exist_ok=True)
            test_path = folder / "astro_stacker_gpu.log"
            with test_path.open("a", encoding="utf-8") as fh:
                fh.write("")
            LOG_PATH = test_path
            return LOG_PATH
        except Exception:
            continue

    LOG_PATH = Path("astro_stacker_gpu.log")
    return LOG_PATH


def log_debug(message: str) -> None:
    try:
        path = init_log_path()
        with path.open("a", encoding="utf-8") as fh:
            fh.write(message.rstrip() + "\n")
    except Exception:
        pass


def bundled_file_path(filename: str) -> Optional[Path]:
    candidates = []
    if getattr(sys, "frozen", False):
        candidates.append(Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent)))
        candidates.append(Path(sys.executable).resolve().parent)
    candidates.append(Path(__file__).resolve().parent)
    candidates.append(Path.cwd())

    for folder in candidates:
        path = folder / filename
        if path.exists():
            return path
    return None


_CUDA_DLL_DIRECTORY_HANDLES = []


def configure_cuda_runtime_path() -> None:
    """Make CUDA Toolkit DLLs visible to CuPy on Windows before importing cupy."""
    log_debug("=" * 72)
    log_debug("Astro Stacker startup")
    log_debug(f"frozen={getattr(sys, 'frozen', False)}")
    log_debug(f"sys.executable={sys.executable}")
    log_debug(f"cwd={Path.cwd()}")
    log_debug(f"initial CUDA_PATH={os.environ.get('CUDA_PATH')}")
    log_debug(f"initial CUPY_CACHE_DIR={os.environ.get('CUPY_CACHE_DIR')}")

    if "CUPY_CACHE_DIR" not in os.environ:
        cache_candidates = []
        local_appdata = os.environ.get("LOCALAPPDATA")
        if local_appdata:
            cache_candidates.append(Path(local_appdata) / "AstroStacker" / "cupy_cache")
        cache_candidates.append(Path.cwd() / ".cupy_cache")
        for cache_dir in cache_candidates:
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                os.environ["CUPY_CACHE_DIR"] = str(cache_dir)
                log_debug(f"set CUPY_CACHE_DIR={cache_dir}")
                break
            except Exception:
                continue

    if os.name != "nt":
        return

    candidates: List[Path] = []
    if getattr(sys, "frozen", False):
        internal_dir = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
        candidates.append(internal_dir)
        candidates.append(Path(sys.executable).parent / "_internal")
        log_debug(f"frozen internal candidates: {internal_dir}, {Path(sys.executable).parent / '_internal'}")

    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        candidates.append(Path(cuda_path))

    root = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if root.exists():
        candidates.extend(sorted(root.glob("v*"), reverse=True))

    # CUDA wheels installed through cupy-cuda12x[ctk] keep individual runtime
    # DLL groups in separate nvidia/*/bin directories. Search those directories
    # both in a PyInstaller bundle and in a normal virtual environment.
    runtime_roots = list(candidates[:2]) if getattr(sys, "frozen", False) else []
    for entry in sys.path:
        try:
            nvidia_root = Path(entry) / "nvidia"
            if nvidia_root.is_dir():
                runtime_roots.append(nvidia_root)
        except (OSError, TypeError):
            continue

    dll_patterns = (
        "cudart*.dll",
        "nvrtc*.dll",
        "nvJitLink*.dll",
        "cublas*.dll",
    )
    dll_dirs = set()
    for runtime_root in runtime_roots:
        if not runtime_root.exists():
            continue
        for pattern in dll_patterns:
            try:
                dll_dirs.update(path.parent for path in runtime_root.rglob(pattern))
            except OSError:
                continue

    for dll_dir in sorted(dll_dirs, key=lambda path: str(path).lower()):
        dll_dir_text = str(dll_dir)
        path_parts = os.environ.get("PATH", "").split(os.pathsep)
        if dll_dir_text not in path_parts:
            os.environ["PATH"] = dll_dir_text + os.pathsep + os.environ.get("PATH", "")
        try:
            handle = os.add_dll_directory(dll_dir_text)
            _CUDA_DLL_DIRECTORY_HANDLES.append(handle)
            log_debug(f"added bundled CUDA DLL directory: {dll_dir}")
        except Exception:
            log_debug(f"add bundled DLL directory failed for {dll_dir}:\n{traceback.format_exc()}")

    if dll_dirs:
        log_debug(f"bundled CUDA DLL directories found: {len(dll_dirs)}")

    for cuda_root in candidates:
        bin_dir = cuda_root / "bin"
        if not bin_dir.exists() and any(cuda_root.glob("nvrtc*.dll")):
            bin_dir = cuda_root
        if not bin_dir.exists():
            log_debug(f"CUDA candidate skipped, bin missing: {cuda_root}")
            continue
        if not any(bin_dir.glob("nvrtc*.dll")):
            log_debug(f"CUDA candidate skipped, nvrtc missing: {bin_dir}")
            continue
        os.environ.setdefault("CUDA_PATH", str(cuda_root))
        path_parts = os.environ.get("PATH", "").split(os.pathsep)
        if str(bin_dir) not in path_parts:
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
        try:
            handle = os.add_dll_directory(str(bin_dir))
            _CUDA_DLL_DIRECTORY_HANDLES.append(handle)
            log_debug(f"added DLL directory: {bin_dir}")
        except Exception:
            log_debug(f"add_dll_directory failed for {bin_dir}:\n{traceback.format_exc()}")
        log_debug(f"selected CUDA root={cuda_root}")
        log_debug(f"selected CUDA bin={bin_dir}")
        log_debug(f"nvrtc dlls={[p.name for p in bin_dir.glob('nvrtc*.dll')]}")
        return

    log_debug("No usable CUDA runtime path found.")


configure_cuda_runtime_path()

try:
    import cupy as cp
    CUPY_IMPORT_ERROR = None
    log_debug(f"CuPy import OK: {getattr(cp, '__version__', '?')}")
except Exception as exc:
    cp = None
    CUPY_IMPORT_ERROR = traceback.format_exc()
    log_debug(f"CuPy import failed:\n{CUPY_IMPORT_ERROR}")

GPU_AVAILABLE_ERROR = None
GPU_AVAILABLE_DETAIL = ""

try:
    import torch
    TORCH_IMPORT_ERROR = None
    log_debug(f"PyTorch import OK: {getattr(torch, '__version__', '?')}")
except Exception:
    torch = None
    TORCH_IMPORT_ERROR = traceback.format_exc()
    log_debug(f"PyTorch import failed:\n{TORCH_IMPORT_ERROR}")

MPS_AVAILABLE_ERROR = None
MPS_AVAILABLE_DETAIL = ""

try:
    import onnxruntime as ort
    ONNXRUNTIME_IMPORT_ERROR = None
    available_ort_providers = ort.get_available_providers()
    if (
        os.name == "nt"
        and "CUDAExecutionProvider" in available_ort_providers
        and hasattr(ort, "preload_dlls")
    ):
        try:
            ort.preload_dlls(directory="")
            log_debug("ONNX Runtime optional GPU DLL preload completed.")
        except Exception:
            log_debug(
                "ONNX Runtime optional GPU DLL preload failed:\n"
                + traceback.format_exc()
            )
    log_debug(
        f"ONNX Runtime import OK: {getattr(ort, '__version__', '?')}; "
        f"providers={available_ort_providers}"
    )
except Exception:
    ort = None
    ONNXRUNTIME_IMPORT_ERROR = traceback.format_exc()
    log_debug(f"ONNX Runtime import failed:\n{ONNXRUNTIME_IMPORT_ERROR}")

try:
    from astropy.io import fits
    from astropy.io.fits.verify import VerifyWarning
except ImportError:  # aplikace poběží i bez FITS podpory, ale FITS nepůjde načíst/uložit
    fits = None
    VerifyWarning = Warning

try:
    import rawpy
except ImportError:
    rawpy = None

try:
    from xisf import XISF
except ImportError:
    XISF = None
from PySide6.QtCore import QEventLoop, QPoint, QPointF, QRectF, QSize, QSettings, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QBrush, QColor, QCursor, QDesktopServices, QIcon, QImage, QPainter, QPen, QPixmap, QPolygon
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSlider,
    QSizePolicy,
    QSplashScreen,
    QSpinBox,
    QTabWidget,
    QAbstractItemView,
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

RAW_EXTENSIONS = {".cr2", ".cr3", ".raw", ".nef", ".arw", ".dng", ".orf", ".rw2", ".raf"}
XISF_EXTENSIONS = {".xisf"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".fits", ".fit"} | XISF_EXTENSIONS | RAW_EXTENSIONS
FITS_EXTENSIONS = {".fits", ".fit"}
LINEAR_ASTRO_EXTENSIONS = FITS_EXTENSIONS | XISF_EXTENSIONS
RAW_STACK_EXTENSIONS = LINEAR_ASTRO_EXTENSIONS | RAW_EXTENSIONS

LAST_STACK_SELECTION: Dict[str, Any] = {}
ALIGNMENT_CACHE_VERSION = "aligned-v8-full-satellite-mask"
QUALITY_CACHE_VERSION = "quality-v15-star-shape-reference"
CALIBRATION_SIGNATURE_CACHE: Dict[Tuple[Any, ...], Tuple[Any, ...]] = {}
MP_WORKER_CONTEXT: Dict[str, Any] = {}
GRADIENT_REMOVAL_ALGORITHM_VERSION = "gradient-v2-robust-polynomial"
DRUNET_MODEL_FILENAMES = (
    "drunet_color.onnx",
    "drunet_color_fp32.onnx",
    "drunet_rgb.onnx",
    "drunet_gray.onnx",
    "drunet_grayscale.onnx",
)
DRUNET_SESSION_CACHE: Dict[Tuple[str, Tuple[str, ...]], Any] = {}
STELLAR_MODEL_FILENAMES = (
    "cosmic_clarity_stellar.onnx",
    "deep_sharp_stellar_cnn.onnx",
    "deep_sharp_stellar_cnn_AI3_5s.onnx",
)
STELLAR_SESSION_CACHE: Dict[Tuple[str, Tuple[str, ...]], Any] = {}


class ArrowComboBox(QComboBox):
    """QComboBox with a consistently visible down triangle over the styled button."""

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        color = QColor("#9aa3b2") if not self.isEnabled() else self.palette().color(self.foregroundRole())
        painter.setBrush(color)
        painter.setPen(Qt.NoPen)
        cx = self.width() - 12
        cy = self.height() // 2 + 1
        painter.drawPolygon(QPolygon([
            QPoint(cx - 5, cy - 3),
            QPoint(cx + 5, cy - 3),
            QPoint(cx, cy + 4),
        ]))


class ArrowSpinBox(QSpinBox):
    """QSpinBox with explicit up/down triangles; QSS native arrows are unreliable."""

    def paintEvent(self, event):
        super().paintEvent(event)
        self._paint_arrows()

    def _paint_arrows(self):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        color = QColor("#9aa3b2") if not self.isEnabled() else self.palette().color(self.foregroundRole())
        painter.setBrush(color)
        painter.setPen(Qt.NoPen)
        cx = self.width() - 12
        up_y = max(7, self.height() // 4)
        down_y = min(self.height() - 7, (self.height() * 3) // 4)
        painter.drawPolygon(QPolygon([
            QPoint(cx - 5, up_y + 3),
            QPoint(cx + 5, up_y + 3),
            QPoint(cx, up_y - 4),
        ]))
        painter.drawPolygon(QPolygon([
            QPoint(cx - 5, down_y - 3),
            QPoint(cx + 5, down_y - 3),
            QPoint(cx, down_y + 4),
        ]))


class ArrowDoubleSpinBox(QDoubleSpinBox):
    """QDoubleSpinBox with the same visible arrows as ArrowSpinBox."""

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        color = QColor("#9aa3b2") if not self.isEnabled() else self.palette().color(self.foregroundRole())
        painter.setBrush(color)
        painter.setPen(Qt.NoPen)
        cx = self.width() - 12
        if self.objectName() == "sliderValueBox":
            cy = self.height() // 2
            painter.drawPolygon(QPolygon([
                QPoint(cx - 5, cy - 4),
                QPoint(cx + 5, cy - 4),
                QPoint(cx, cy - 11),
            ]))
            painter.drawPolygon(QPolygon([
                QPoint(cx - 5, cy + 4),
                QPoint(cx + 5, cy + 4),
                QPoint(cx, cy + 11),
            ]))
            return
        up_y = max(7, self.height() // 4)
        down_y = min(self.height() - 7, (self.height() * 3) // 4)
        painter.drawPolygon(QPolygon([
            QPoint(cx - 5, up_y + 3),
            QPoint(cx + 5, up_y + 3),
            QPoint(cx, up_y - 4),
        ]))
        painter.drawPolygon(QPolygon([
            QPoint(cx - 5, down_y - 3),
            QPoint(cx + 5, down_y - 3),
            QPoint(cx, down_y + 4),
        ]))


class ProcessingCancelled(Exception):
    pass

def open_fits_safely(path: Path, memmap: bool = False):
    """Otevře FIT/FITS tolerantně k nevalidním hlavičkám.

    Některé kamery/programy uloží kartu BAYERPAT ne zcela podle FITS
    standardu. Astropy pak hlásí "Unparsable card (BAYERPAT), fix it first
    with .verify('fix')". Tady se proto hlavička po otevření hned opraví.
    """
    if fits is None:
        raise RuntimeError("Pro FITS podporu nainstaluj: pip install astropy")

    hdul = fits.open(path, memmap=memmap, ignore_missing_end=True)
    try:
        hdul.verify("fix")
    except Exception:
        # Některé vadné karty nemusí jít opravit dokonale, ale většinou lze
        # stále bezpečně pokračovat a přečíst obrazová data.
        pass
    return hdul


def get_first_fits_header_safely(path: Path):
    """Vrátí kopii první obrazové FITS hlavičky s tolerantním verify('fix')."""
    if fits is None:
        return None
    try:
        with open_fits_safely(path, memmap=False) as hdul:
            for hdu in hdul:
                if getattr(hdu, "data", None) is not None:
                    return hdu.header.copy()
    except Exception:
        return None
    return None



CALIBRATION_ALGORITHM_VERSION = "calibration-v6-average-masters"


@dataclass
class StackSettings:
    align_mode: str = "translation"  # calibration | translation | ecc_affine | star_affine | comet
    stack_mode: str = "sigma"        # mean | median | sigma | high_rejection
    sigma: float = 2.5
    max_images: int = 0               # 0 = all
    raw_only: bool = False            # použije XISF, FIT/FITS a foto RAW; vynechá JPG/PNG/BMP/TIFF
    fit_only: bool = False            # zpětná kompatibilita se starými profily
    downscale_for_alignment: float = 0.5
    normalize_background: bool = True
    auto_reference: bool = True       # vybere nejostřejší snímek jako referenci
    manual_reference_path: Optional[str] = None  # ručně zvolená reference pro běžné zarovnání
    mosaic_mode: bool = False           # rozšíří výstupní plátno podle transformací přijatých snímků
    quality_filter: bool = True       # vyřadí nejhorší snímky podle skóre
    keep_percent: int = 80            # kolik % nejlepších snímků ponechat
    manual_excluded_paths: Tuple[str, ...] = ()  # ručně vyřazené light snímky z review tabulky
    preselected_paths: Tuple[str, ...] = ()  # cesty vybrané v review kroku; stack pak přeskočí nový quality výběr
    preselected_reference_path: Optional[str] = None
    max_star_shift: int = 180          # max. drift hvězd vůči referenci v pixelech
    star_border_margin: int = 120       # okraj obrazu ignorovaný při detekci hvězd, pomáhá proti stromům/větvím; lze nastavit i stovky až tisíce px
    strict_star_filter: bool = True     # přísnější filtr tvaru hvězd, potlačí větve, dráty a rozmazané fleky
    satellite_trail_filter: bool = False  # detekuje a maskuje dlouhé rovné satelitní stopy
    bayer_pattern: str = "auto"          # auto | mono | RGGB | BGGR | GRBG | GBRG
    max_comet_shift: int = 800         # maximální očekávaný pohyb komety vůči referenci v pixelech
    comet_refine: bool = True          # po dvoubodové predikci jemně dohledá kometu lokální korelací
    comet_refine_patch: int = 45       # poloměr referenční šablony komety v pixelech
    comet_refine_search: int = 90      # jak daleko od predikce se smí hledat jádro komety
    manual_comet_xy: Optional[Tuple[float, float]] = None  # ručně označené jádro komety v prvním/referenčním snímku, souřadnice v plném rozlišení
    manual_comet_reference_path: Optional[str] = None      # soubor, ve kterém byla první poloha komety ručně označena
    manual_comet_end_xy: Optional[Tuple[float, float]] = None  # ručně označené jádro komety v posledním snímku
    manual_comet_end_path: Optional[str] = None               # soubor, ve kterém byla poslední poloha komety ručně označena
    comet_mask_radius: int = 120       # poloměr masky komety pro kombinovaný star+comet stack
    comet_mask_softness: int = 60      # měkkost okraje masky komety
    flat_frame_path: Optional[str] = None  # volitelný master flat nebo složka Flat snímků
    bias_frame_path: Optional[str] = None  # volitelný master bias nebo složka Bias snímků
    dark_frame_path: Optional[str] = None  # volitelný master dark nebo složka Dark snímků
    source_folder: Optional[str] = None  # složka light snímků pro automatické hledání Bias/Flat/Dark podsložek
    output_dir: Optional[str] = None  # volitelná výstupní složka pro CLI/wrapper režimy s více soubory
    use_gpu: bool = False                  # optional CUDA/CuPy or Apple Metal/MPS acceleration for stacking
    use_aligned_cache: bool = False         # cache zarovnaných snímků; default vypnuto kvůli I/O brzdě při alignmentu
    live_stack_reject_outliers: bool = True   # online kappa-sigma rejection jednotlivých pixelů v Live stacku (letadla, satelity, kosmické záření)
    live_stack_reject_sigma: float = 3.0      # práh odmítnutí pro Live stack (násobek průběžné směrodatné odchylky)
    live_stack_reject_min_frames: int = 4     # počet úvodních snímků bez odmítání, než se ustálí průběžný odhad šumu
    live_stack_reject_elongated_stars: bool = True  # vyřadí celý Live stack snímek s výrazně protaženými hvězdami
    live_stack_min_roundness: float = 0.50  # minor/major; 0.50 odpovídá elongaci cca 2:1
    live_stack_min_shape_stars: int = 8  # minimum spolehlivě změřených hvězd pro kontrolu elongace
    language: str = "en"


@dataclass
class StretchSettings:
    black: int = 0
    white: int = 65535
    gamma: float = 1.5
    stf_strength: float = 5.0
    vignette_removal: float = 0.0
    synthetic_flat: float = 0.0
    color_background_correction: float = 0.0
    denoise_strength: float = 0.0
    denoise_mode: str = "classic"
    ai_denoise_model_path: Optional[str] = None
    star_deconvolution_strength: float = 0.0
    star_deconvolution_size: float = 5.0
    star_deconvolution_model_path: Optional[str] = None
    contrast: float = 2.1
    saturation: float = 1.0
    red: float = 1.0
    green: float = 1.0
    blue: float = 1.0
    scnr_green_strength: int = 0


def normalize_array_to_float(arr: np.ndarray) -> np.ndarray:
    """Normalize arbitrary image array to float32 0..1 for ordinary image preview/input.

    Pozor: tato funkce používá robustní percentily, tedy dělá náhledový stretch.
    Pro běžné obrazové formáty je to v pořádku; FIT/FITS data se ale
    načítají přes normalize_fits_linear_to_float(), aby export FIT zůstal lineární.
    """
    arr = np.asarray(arr).astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    if arr.size == 0:
        raise ValueError("Prázdný obrazový soubor.")

    # Robustní normalizace pro běžné obrazové formáty/náhled.
    lo = np.percentile(arr, 0.1)
    hi = np.percentile(arr, 99.9)
    if hi <= lo:
        lo = float(np.min(arr))
        hi = float(np.max(arr))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)

    arr = (arr - lo) / (hi - lo)
    return np.clip(arr, 0, 1).astype(np.float32)


def midtones_transfer(m: float, x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)
    m = float(np.clip(m, 1e-5, 0.99999))
    denom = (2.0 * m - 1.0) * x - m
    return np.clip(((m - 1.0) * x) / np.where(np.abs(denom) < 1e-6, -1e-6, denom), 0.0, 1.0).astype(np.float32)


def display_preview_limits(img: np.ndarray) -> Tuple[float, ...]:
    arr = np.asarray(img, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.size == 0:
        return 0.0, 1.0, 0.25, 0.16, 0.85, 0.55

    if arr.ndim == 3 and arr.shape[2] == 3:
        lum = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    else:
        lum = arr

    finite = lum[np.isfinite(lum)]
    if finite.size < 10:
        return 0.0, 1.0, 0.25, 0.16, 0.85, 0.55

    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    if not np.isfinite(median):
        return 0.0, 1.0, 0.25, 0.16, 0.85, 0.55

    # Standard linked astronomical STF: clip only the low background tail
    # using median/MAD, then move the remaining background median to a
    # moderate display level. Bright pixels retain the complete 0..1 range;
    # no high percentile is promoted into the highlight region.
    lo = max(0.0, median - 2.8 * mad)
    hi = 1.0
    normalized_median = float(np.clip((median - lo) / max(hi - lo, 1e-6), 0.0, 1.0))
    # A restrained background target keeps single light frames from looking
    # excessively noisy while preserving the standard STF highlight behavior.
    target = 0.16
    if normalized_median <= 1e-6:
        mtf = 0.25
    else:
        denom = 2.0 * normalized_median * target - target - normalized_median
        mtf = (
            normalized_median * (target - 1.0) / denom
            if abs(denom) > 1e-8
            else 0.25
        )
    mtf = float(np.clip(mtf, 1e-4, 0.999))
    return lo, hi, mtf, target, hi, 1.0


def apply_display_preview_limits(img: np.ndarray, limits: Tuple[float, ...]) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if len(limits) >= 6:
        lo, hi, mtf, _target, _knee, _stf_weight = map(float, limits[:6])
        if hi <= lo:
            return np.zeros_like(arr, dtype=np.float32)
        x = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
        return midtones_transfer(mtf, x)
    elif len(limits) >= 4:
        lo, hi, mtf, _target = map(float, limits[:4])
        if hi <= lo:
            return np.zeros_like(arr, dtype=np.float32)
        x = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
        return midtones_transfer(mtf, x)
    elif len(limits) >= 3:
        lo, knee, hi = map(float, limits[:3])
    else:
        # Backward compatibility with profiles/Undo states created earlier.
        lo, hi = float(limits[0]), float(limits[1])
        knee = hi
    if knee <= lo:
        return np.zeros_like(arr, dtype=np.float32)

    # Leave a little more headroom for galaxy nuclei and bright stellar cores.
    # The highlight tail still reaches white smoothly, but starts lower so its
    # internal structure remains easier to see in previews and visual exports.
    knee_level = 0.78
    low = (arr - lo) / (knee - lo)
    out = low * knee_level
    if hi > knee:
        tail = np.clip((arr - knee) / (hi - knee), 0.0, 1.0)
        # Rise toward white gradually. A power shoulder preserves separation
        # inside bright galaxy nuclei instead of making the lower highlight
        # tail brighter, as the previous logarithmic mapping did.
        tail = np.power(tail, 1.45)
        out = np.where(arr > knee, knee_level + (1.0 - knee_level) * tail, out)
    return np.clip(out, 0, 1).astype(np.float32)


def make_display_preview_base(img: np.ndarray) -> np.ndarray:
    """Vytvoří dočasný obraz pouze pro náhled v GUI.

    DŮLEŽITÉ:
    - Tato funkce se NESMÍ používat pro FIT export ani pro stackovací výpočty.
    - Slouží jen k tomu, aby lineární FIT/stack data byla na obrazovce viditelná.
    - Zachovává barevné poměry tím, že percentily počítá z luminance a stejnou
      lineární transformaci aplikuje na všechny RGB kanály.
    """
    # Robustní displejový stretch. Je záměrně pouze pro zobrazení.
    return apply_display_preview_limits(img, display_preview_limits(img))


def normalize_fits_linear_to_float(arr: np.ndarray) -> np.ndarray:
    """Lineární normalizace FIT/FITS dat bez percentilového stretche.

    Cíl: zachovat lineární vztahy jasů. Nepoužívá black/white point, gamma ani
    percentilové oříznutí. Pro celočíselná data používá rozsah datového typu,
    pro float data používá pouze lineární posun/škálování podle skutečného min/max.
    """
    original = np.asarray(arr)
    if original.size == 0:
        raise ValueError("Prázdný obrazový soubor.")

    if np.issubdtype(original.dtype, np.integer):
        info = np.iinfo(original.dtype)
        data = original.astype(np.float32)
        # U většiny astro FIT je fyzikální nula opravdu 0. Signed data posuneme
        # podle rozsahu typu, pořád čistě lineárně.
        data = (data - float(info.min)) / max(1.0, float(info.max - info.min))
        return np.clip(np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0), 0, 1).astype(np.float32)

    data = np.asarray(original, dtype=np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(np.min(data))
    hi = float(np.max(data))
    if hi <= lo:
        return np.zeros_like(data, dtype=np.float32)

    # A normalized 32-bit floating-point FIT is already linear 0..1 data.
    # Stretching it again from its own min/max destroys its physical brightness
    # scale and makes bright galaxy cores appear much larger than they are.
    if lo >= 0.0 and hi <= 1.000001:
        return np.clip(data, 0, 1).astype(np.float32)

    # Non-normalized non-negative float FIT data are commonly stored as ADU.
    # Scale by the sensor container range without subtracting the frame minimum.
    if lo >= 0.0 and hi <= 65535.0:
        return np.clip(data / 65535.0, 0, 1).astype(np.float32)

    # Last-resort linear mapping for signed/calibrated floating-point data.
    data = (data - lo) / (hi - lo)
    return np.clip(data, 0, 1).astype(np.float32)


BAYER_PATTERNS = {"RGGB", "BGGR", "GRBG", "GBRG"}
BAYER_PATTERN_OVERRIDE = "auto"  # auto | mono | RGGB | BGGR | GRBG | GBRG

def normalize_bayer_pattern_value(value) -> Optional[str]:
    """Robustně přečte Bayer pattern z hodnoty nebo komentáře FITS karty.

    Rozumí zápisům jako:
    RGGB, 'RGGB', "RGGB", BAYERPAT=RGGB, Bayer pattern: RGGB.
    """
    if value is None:
        return None

    raw = str(value).upper().strip()

    m = re.search(r"\b(RGGB|BGGR|GRBG|GBRG)\b", raw)
    if m:
        return m.group(1)

    cleaned = "".join(ch for ch in raw if ch in "RGB")
    for pat in BAYER_PATTERNS:
        if pat in cleaned:
            return pat

    return None



def set_bayer_pattern_override(pattern: Optional[str]) -> None:
    """Nastaví ruční override Bayer masky pro načítání FIT/FITS.

    Hodnoty:
    - auto: použije FIT hlavičku
    - mono: nikdy nedebayeruje, 2D FIT bere jako monochrom
    - RGGB/BGGR/GRBG/GBRG: ručně vynutí Bayer masku pro 2D FIT data
    """
    global BAYER_PATTERN_OVERRIDE
    p = str(pattern or "auto").strip().upper()
    if p in BAYER_PATTERNS:
        BAYER_PATTERN_OVERRIDE = p
    elif p == "MONO":
        BAYER_PATTERN_OVERRIDE = "mono"
    else:
        BAYER_PATTERN_OVERRIDE = "auto"


def get_bayer_pattern_override() -> str:
    return BAYER_PATTERN_OVERRIDE


def effective_bayer_pattern_from_header(header) -> Optional[str]:
    """Vrátí Bayer pattern s ohledem na ruční nastavení v UI."""
    override = get_bayer_pattern_override()
    if isinstance(override, str):
        if override.upper() in BAYER_PATTERNS:
            return override.upper()
        if override.lower() == "mono":
            return None
    return detect_bayer_pattern_from_header(header)


def detect_bayer_pattern_from_header(header) -> Optional[str]:
    """Vrátí Bayer/CFA pattern z FIT hlavičky, např. RGGB/BGGR/GRBG/GBRG.

    Funkce je tolerantní k různým zápisům:
    - RGGB
    - 'RGGB'
    - "RGGB"
    - BAYERPAT = 'RGGB'
    - CFA_PATTERN: RGGB
    """
    if header is None:
        return None

    keys = (
        "BAYERPAT", "BAYER_PATTERN", "BAYER",
        "CFA", "CFA_PAT", "CFA_PATTERN", "CFAPAT", "COLORTYP", "COLORTYPE",
        "FILTERPAT", "FILTER_PATTERN",
    )

    # 1) Přímé známé klíče + jejich komentáře.
    for key in keys:
        if key in header:
            pat = normalize_bayer_pattern_value(header.get(key))
            if pat:
                return pat
            try:
                pat = normalize_bayer_pattern_value(header.comments[key])
                if pat:
                    return pat
            except Exception:
                pass

    # 2) Projdi jen relevantní FITS karty. U monochromatických kamer bývají v
    # hlavičce různé texty s řetězci podobnými RGB; celý header proto nesmíme
    # prohledávat naslepo, jinak by Auto režim falešně debayeroval mono snímky.
    try:
        cards = list(header.cards)
    except Exception:
        cards = []

    for card in cards:
        keyword = str(getattr(card, "keyword", "")).upper()
        comment = str(getattr(card, "comment", "")).upper()
        relevant = any(token in keyword for token in ("BAYER", "CFA", "COLOR", "COLOUR", "FILTERPAT"))
        relevant = relevant or any(token in comment for token in ("BAYER", "CFA", "COLOR FILTER", "COLOUR FILTER"))
        if not relevant:
            continue
        for value in (
            getattr(card, "value", None),
            getattr(card, "comment", None),
            f"{getattr(card, 'keyword', '')} {getattr(card, 'value', '')} {getattr(card, 'comment', '')}",
        ):
            pat = normalize_bayer_pattern_value(value)
            if pat:
                return pat

    # Bez explicitní Bayer/CFA informace je FIT v Auto režimu monochromatický.
    return None


def xisf_image_metadata(path: Path) -> Dict[str, Any]:
    """Read metadata for the first image block without decoding image pixels."""
    if XISF is None:
        raise RuntimeError("Pro XISF podporu nainstaluj: pip install xisf")
    reader = XISF(str(path))
    metadata = reader.get_images_metadata()
    if not metadata:
        raise ValueError(f"XISF soubor neobsahuje obrazová data: {Path(path).name}")
    return dict(metadata[0])


def xisf_metadata_values(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Flatten XISF FITS keywords and properties for metadata/CFA detection."""
    flattened: Dict[str, Any] = {}
    if not isinstance(metadata, dict):
        return flattened

    fits_keywords = metadata.get("FITSKeywords", {})
    if isinstance(fits_keywords, dict):
        for key, entries in fits_keywords.items():
            value = entries
            if isinstance(entries, list) and entries:
                first = entries[0]
                value = first.get("value") if isinstance(first, dict) else first
            elif isinstance(entries, dict):
                value = entries.get("value")
            flattened[str(key).upper()] = value

    properties = metadata.get("XISFProperties", {})
    if isinstance(properties, dict):
        for key, prop in properties.items():
            value = prop.get("value") if isinstance(prop, dict) else prop
            flattened[str(key).upper()] = value
    return flattened


def bayer_pattern_from_xisf_metadata(metadata: Optional[Dict[str, Any]]) -> Optional[str]:
    override = get_bayer_pattern_override()
    if str(override).upper() in BAYER_PATTERNS:
        return str(override).upper()
    if str(override).lower() == "mono":
        return None

    values = xisf_metadata_values(metadata)
    direct_keys = (
        "BAYERPAT", "BAYER_PATTERN", "BAYER", "CFA", "CFA_PAT",
        "CFA_PATTERN", "CFAPAT", "COLORTYP", "COLORTYPE",
        "FILTERPAT", "FILTER_PATTERN", "PCL:CFASOURCEPATTERN",
        "PCL:CFAPATTERN",
    )
    for key in direct_keys:
        if key in values:
            pattern = normalize_bayer_pattern_value(values[key])
            if pattern:
                return pattern

    for key, value in values.items():
        key_upper = str(key).upper()
        if any(token in key_upper for token in ("BAYER", "CFA", "FILTERPAT")):
            pattern = normalize_bayer_pattern_value(value)
            if pattern:
                return pattern
    return None


def bayer_pattern_for_fits_path(path: Path, override: Optional[str] = None) -> Optional[str]:
    """Rychle zjistí, jestli se FIT/FITS bude debayerovat.

    Bere v úvahu ruční nastavení Bayer masky. Pokud je zvoleno Mono,
    vrací None. Pokud je ručně zvolen konkrétní pattern, vrací ho pro
    2D FIT/cube data bez ohledu na hlavičku. V režimu Auto čte hlavičku.
    Obrazová data se zde záměrně nenačítají: tato funkce se volá pro celou
    sekvenci ještě před multiprocessingem a načtení každého velkého FITu by
    vytvořilo zbytečnou sériovou fázi před paralelním debayeringem.
    """
    suffix = Path(path).suffix.lower()
    if suffix in XISF_EXTENSIONS:
        selected = str(override if override is not None else get_bayer_pattern_override()).strip()
        if selected.upper() in BAYER_PATTERNS:
            return selected.upper()
        if selected.lower() == "mono":
            return None
        try:
            return bayer_pattern_from_xisf_metadata(xisf_image_metadata(path))
        except Exception:
            return None

    if fits is None or suffix not in FITS_EXTENSIONS:
        return None

    selected = str(override if override is not None else get_bayer_pattern_override()).strip()
    selected_upper = selected.upper()
    selected_lower = selected.lower()

    try:
        with open_fits_safely(path, memmap=True) as hdul:
            for hdu in hdul:
                header = getattr(hdu, "header", None)
                if header is None:
                    continue
                naxis = int(header.get("NAXIS", 0) or 0)
                if naxis <= 0:
                    continue
                # Ruční Mono: nedebayerovat.
                if selected_lower == "mono":
                    return None
                # Ruční pattern: debayerovat jen 2D raw nebo první rovinu cube.
                if selected_upper in BAYER_PATTERNS:
                    if naxis == 2:
                        return selected_upper
                    if naxis == 3:
                        # NumPy tvar FIT cube je (NAXIS3, NAXIS2, NAXIS1).
                        axis1 = int(header.get("NAXIS1", 0) or 0)
                        axis3 = int(header.get("NAXIS3", 0) or 0)
                        if axis1 not in (3, 4) and axis3 not in (3, 4):
                            return selected_upper
                    return None
                # Auto: použít FIT hlavičku.
                return detect_bayer_pattern_from_header(header)
    except Exception:
        return None
    return None


def report_bayer_conversion_if_needed(path: Path, progress_callback, progress_value: int = 0) -> None:
    """Zobrazí informaci, že tento konkrétní FIT projde raw načtením, kalibrací a debayeringem."""
    if not progress_callback:
        return
    pattern = bayer_pattern_for_fits_path(path)
    if pattern:
        progress_callback(progress_value, f"Načítám/kalibruji/debayeruji Bayer {pattern}")


def format_elapsed(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    rest = seconds - minutes * 60
    if minutes < 60:
        return f"{minutes}m {rest:04.1f}s"
    hours = minutes // 60
    minutes = minutes % 60
    return f"{hours}h {minutes:02d}m {rest:04.1f}s"


def format_bytes_short(num_bytes: float) -> str:
    value = max(0.0, float(num_bytes))
    units = ("B", "KB", "MB", "GB", "TB")
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{value:.0f} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024.0


def linear_array_to_uint16_for_demosaic(arr: np.ndarray) -> np.ndarray:
    """Převede lineární raw Bayer data na uint16 pro OpenCV demosaic.

    Převod je lineární, bez percentilového stretche.
    """
    original = np.asarray(arr)
    if original.dtype == np.uint16:
        return np.ascontiguousarray(original)
    if original.dtype == np.uint8:
        return np.ascontiguousarray(original.astype(np.uint16) * np.uint16(257))
    if np.issubdtype(original.dtype, np.integer):
        info = np.iinfo(original.dtype)
        data = original.astype(np.float32)
        data = (data - float(info.min)) / max(1.0, float(info.max - info.min))
    else:
        data = np.asarray(original, dtype=np.float32)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        lo = float(np.min(data))
        hi = float(np.max(data))
        if hi <= lo:
            return np.zeros_like(data, dtype=np.uint16)
        data = (data - lo) / (hi - lo)

    return (np.clip(data, 0, 1) * 65535.0).astype(np.uint16)


def debayer_fits_to_rgb_float(raw: np.ndarray, pattern: str) -> np.ndarray:
    """Převede 2D Bayer FIT data na lineární RGB float 0..1.

    Podporované patterny: RGGB, BGGR, GRBG, GBRG.
    Výstup je H x W x 3 v RGB pořadí.
    """
    pattern = (pattern or "").upper()
    code_map = {
        # These OpenCV aliases preserve the same RGB channel order as the
        # previous RGB conversion followed by a channel reversal.
        "RGGB": cv2.COLOR_BayerRG2BGR,
        "BGGR": cv2.COLOR_BayerBG2BGR,
        "GRBG": cv2.COLOR_BayerGR2BGR,
        "GBRG": cv2.COLOR_BayerGB2BGR,
    }
    if pattern not in code_map:
        raise ValueError(f"Nepodporovaný Bayer pattern: {pattern}")

    raw16 = linear_array_to_uint16_for_demosaic(raw)
    rgb16 = cv2.cvtColor(raw16, code_map[pattern])
    return np.ascontiguousarray(rgb16.astype(np.float32) * np.float32(1.0 / 65535.0))


def debayer_sensor_mosaic_to_rgb_float(raw: np.ndarray, pattern: str) -> np.ndarray:
    """Debayer an already normalized 2D sensor mosaic without rescaling it."""
    pattern = (pattern or "").upper()
    code_map = {
        "RGGB": cv2.COLOR_BayerRG2BGR,
        "BGGR": cv2.COLOR_BayerBG2BGR,
        "GRBG": cv2.COLOR_BayerGR2BGR,
        "GBRG": cv2.COLOR_BayerGB2BGR,
    }
    if pattern not in code_map:
        raise ValueError(f"Nepodporovaný Bayer pattern: {pattern}")
    source = np.asarray(raw, dtype=np.float32)
    work = np.empty_like(source, dtype=np.float32)
    np.clip(source, 0, 1, out=work)
    np.multiply(work, np.float32(65535.0), out=work)
    raw16 = work.astype(np.uint16)
    del work
    rgb16 = cv2.cvtColor(raw16, code_map[pattern])
    return np.ascontiguousarray(rgb16.astype(np.float32) * np.float32(1.0 / 65535.0))


def normalize_sensor_mosaic_to_float(arr: np.ndarray) -> np.ndarray:
    """Normalize raw sensor values linearly while preserving calibration ratios."""
    original = np.asarray(arr)
    if np.issubdtype(original.dtype, np.integer):
        info = np.iinfo(original.dtype)
        data = original.astype(np.float32)
        return np.clip((data - float(info.min)) / max(1.0, float(info.max - info.min)), 0, 1).astype(np.float32)
    data = np.nan_to_num(np.asarray(original, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    hi = float(np.max(data)) if data.size else 0.0
    if hi <= 1.000001:
        return np.clip(data, 0, 1).astype(np.float32)
    scale = 65535.0 if hi <= 65535.0 else max(1.0, hi)
    return np.clip(data / scale, 0, 1).astype(np.float32)


def load_fits_as_float(path: Path) -> np.ndarray:
    if fits is None:
        raise RuntimeError("Pro FITS podporu nainstaluj: pip install astropy")

    header = None
    try:
        with open_fits_safely(path, memmap=False) as hdul:
            data = None
            for hdu in hdul:
                if getattr(hdu, "data", None) is not None:
                    data = hdu.data
                    header = getattr(hdu, "header", None)
                    break
    except Exception as exc:
        log_debug("FITS load failed")
        log_debug(f"path={path}")
        try:
            import astropy
            log_debug(f"astropy={getattr(astropy, '__version__', '?')} file={getattr(astropy, '__file__', '?')}")
        except Exception:
            pass
        log_debug(f"numpy={getattr(np, '__version__', '?')} file={getattr(np, '__file__', '?')}")
        log_debug(traceback.format_exc())
        raise RuntimeError(f"{exc}. Diagnostic log: {init_log_path()}") from exc

    if data is None:
        raise ValueError(f"FITS soubor neobsahuje obrazová data: {path.name}")

    data = np.asarray(data)
    bayer_pattern = effective_bayer_pattern_from_header(header)

    # FITS může být:
    # - 2D monochromatický obraz
    # - 2D Bayer/CFA raw obraz, který převedeme na RGB
    # - 3D RGB obraz nebo cube
    # DŮLEŽITÉ: FITS načítáme lineárně, bez percentilového stretche.
    if data.ndim == 2:
        if bayer_pattern:
            img = debayer_fits_to_rgb_float(data, bayer_pattern)
        else:
            img = normalize_fits_linear_to_float(data)
            img = np.repeat(img[..., None], 3, axis=2)
    elif data.ndim == 3:
        # Častý tvar: channels, height, width
        if data.shape[0] in (3, 4) and data.shape[1] > 16 and data.shape[2] > 16:
            img = np.moveaxis(data[:3], 0, -1)
            img = normalize_fits_linear_to_float(img)
        # Alternativní tvar: height, width, channels
        elif data.shape[-1] in (3, 4):
            img = data[..., :3]
            img = normalize_fits_linear_to_float(img)
        else:
            # FIT cube: použij první rovinu. Pokud hlavička říká Bayer, debayeruj ji.
            plane = data[0]
            if bayer_pattern and plane.ndim == 2:
                img = debayer_fits_to_rgb_float(plane, bayer_pattern)
            else:
                img = normalize_fits_linear_to_float(plane)
                img = np.repeat(img[..., None], 3, axis=2)
    else:
        raise ValueError(f"Nepodporovaný FITS rozměr {data.ndim}D v souboru {path.name}")

    return np.ascontiguousarray(img.astype(np.float32))


def load_xisf_data_and_metadata(path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    if XISF is None:
        raise RuntimeError("Pro XISF podporu nainstaluj: pip install xisf")
    metadata: Dict[str, Any] = {}
    try:
        data = XISF.read(str(path), n=0, image_metadata=metadata)
    except Exception as exc:
        log_debug(f"XISF load failed: {path}\n{traceback.format_exc()}")
        raise RuntimeError(f"XISF se nepodařilo načíst: {exc}") from exc
    if data is None:
        raise ValueError(f"XISF soubor neobsahuje obrazová data: {path.name}")
    return np.asarray(data), metadata


def normalize_xisf_linear_to_float(arr: np.ndarray) -> np.ndarray:
    """Preserve native normalized XISF floats; scale integer samples by dtype."""
    original = np.asarray(arr)
    if original.size == 0:
        raise ValueError("Prázdný obrazový soubor.")
    if np.issubdtype(original.dtype, np.integer):
        info = np.iinfo(original.dtype)
        data = original.astype(np.float32)
        data = (data - float(info.min)) / max(1.0, float(info.max - info.min))
        return np.ascontiguousarray(np.clip(data, 0, 1).astype(np.float32))

    data = np.nan_to_num(np.asarray(original, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(np.min(data))
    hi = float(np.max(data))
    if lo >= 0.0 and hi <= 1.000001:
        return np.ascontiguousarray(np.clip(data, 0, 1).astype(np.float32))
    return np.ascontiguousarray(normalize_fits_linear_to_float(data))


def load_xisf_as_float(path: Path) -> np.ndarray:
    """Load the first XISF image as linear float32 RGB in the 0..1 range."""
    data, metadata = load_xisf_data_and_metadata(path)
    pattern = bayer_pattern_from_xisf_metadata(metadata)

    if data.ndim == 2:
        if pattern:
            img = debayer_fits_to_rgb_float(data, pattern)
        else:
            mono = normalize_xisf_linear_to_float(data)
            img = np.repeat(mono[..., None], 3, axis=2)
    elif data.ndim == 3:
        if data.shape[-1] == 1:
            plane = data[..., 0]
            if pattern:
                img = debayer_fits_to_rgb_float(plane, pattern)
            else:
                mono = normalize_xisf_linear_to_float(plane)
                img = np.repeat(mono[..., None], 3, axis=2)
        elif data.shape[-1] in (3, 4):
            img = normalize_xisf_linear_to_float(data[..., :3])
        elif data.shape[0] in (3, 4):
            img = normalize_xisf_linear_to_float(np.moveaxis(data[:3], 0, -1))
        else:
            plane = data[0]
            if pattern and plane.ndim == 2:
                img = debayer_fits_to_rgb_float(plane, pattern)
            else:
                mono = normalize_xisf_linear_to_float(plane)
                img = np.repeat(mono[..., None], 3, axis=2)
    else:
        raise ValueError(f"Nepodporovaný XISF rozměr {data.ndim}D v souboru {path.name}")

    return np.ascontiguousarray(np.asarray(img, dtype=np.float32))



def rawpy_bayer_pattern(raw) -> Optional[str]:
    """Zjistí Bayer pattern z rawpy objektu.

    Nejdřív používá raw.raw_colors_visible, protože to odpovídá skutečné
    viditelné RAW matici po ořezu. To je důležité hlavně u starších CR2,
    kde raw.raw_pattern může popisovat celý senzor, ale raw_image_visible
    začíná o pixel jinak.

    rawpy barevné indexy:
    0 = R, 1 = G, 2 = B, 3 = G2.
    """
    try:
        color_desc = raw.color_desc.decode("ascii", errors="ignore")
    except Exception:
        color_desc = "RGBG"

    def idx_to_char(idx: int) -> str:
        try:
            ch = color_desc[int(idx)].upper()
        except Exception:
            ch = "G"
        if ch not in ("R", "G", "B"):
            ch = "G"
        return ch

    # 1) Nejlepší zdroj: skutečné barvy viditelné RAW matice.
    try:
        colors = np.asarray(raw.raw_colors_visible)
        if colors.shape[0] >= 2 and colors.shape[1] >= 2:
            pat = (
                idx_to_char(colors[0, 0]) +
                idx_to_char(colors[0, 1]) +
                idx_to_char(colors[1, 0]) +
                idx_to_char(colors[1, 1])
            )
            if pat in BAYER_PATTERNS:
                return pat
    except Exception:
        pass

    # 2) Fallback: obecný raw_pattern.
    try:
        pattern = np.asarray(raw.raw_pattern)
        if pattern.shape[0] >= 2 and pattern.shape[1] >= 2:
            pat = (
                idx_to_char(pattern[0, 0]) +
                idx_to_char(pattern[0, 1]) +
                idx_to_char(pattern[1, 0]) +
                idx_to_char(pattern[1, 1])
            )
            if pat in BAYER_PATTERNS:
                return pat
    except Exception:
        pass

    return None


def load_raw_sensor_mosaic_as_float(path: Path) -> Tuple[np.ndarray, str]:
    """Load DSLR/MILC RAW as an undebayered normalized sensor mosaic."""
    if rawpy is None:
        raise RuntimeError("RAW podpora vyžaduje rawpy. Nainstaluj: pip install rawpy")
    with rawpy.imread(str(path)) as raw:
        pattern = rawpy_bayer_pattern(raw) or "RGGB"
        mosaic = np.asarray(raw.raw_image_visible, dtype=np.float32)
        try:
            white = float(raw.white_level)
        except Exception:
            white = float(np.max(mosaic))
    return np.ascontiguousarray(np.clip(mosaic / max(1.0, white), 0, 1).astype(np.float32)), pattern


def load_fits_sensor_mosaic_as_float(path: Path) -> Optional[Tuple[np.ndarray, str]]:
    """Load a Bayer FIT/FITS frame as a 2D sensor mosaic when metadata allows it."""
    if fits is None:
        raise RuntimeError("Pro FITS podporu nainstaluj: pip install astropy")
    with open_fits_safely(path, memmap=False) as hdul:
        for hdu in hdul:
            data = getattr(hdu, "data", None)
            if data is None:
                continue
            pattern = effective_bayer_pattern_from_header(getattr(hdu, "header", None))
            if not pattern:
                return None
            arr = np.asarray(data)
            if arr.ndim == 3 and arr.shape[0] not in (3, 4) and arr.shape[-1] not in (3, 4):
                arr = arr[0]
            if arr.ndim != 2:
                return None
            return np.ascontiguousarray(normalize_sensor_mosaic_to_float(arr)), pattern
    return None


def load_sensor_mosaic_as_float(path: Path) -> Optional[Tuple[np.ndarray, str]]:
    """Return an undebayered sensor mosaic for RAW/Bayer FIT/XISF, otherwise None."""
    suffix = Path(path).suffix.lower()
    if suffix in RAW_EXTENSIONS:
        return load_raw_sensor_mosaic_as_float(path)
    if suffix in FITS_EXTENSIONS:
        return load_fits_sensor_mosaic_as_float(path)
    if suffix in XISF_EXTENSIONS:
        data, metadata = load_xisf_data_and_metadata(path)
        pattern = bayer_pattern_from_xisf_metadata(metadata)
        if not pattern:
            return None
        if data.ndim == 3 and data.shape[-1] == 1:
            data = data[..., 0]
        if data.ndim != 2:
            return None
        return np.ascontiguousarray(normalize_sensor_mosaic_to_float(data)), pattern
    return None


def load_raw_bayer_manual_as_float(path: Path) -> np.ndarray:
    """Ruční načtení RAW Bayer dat a debayer přes OpenCV.

    Používáme hlavně pro problematické Sony ARW, kde rawpy.postprocess()
    může u některých souborů dávat špatně interpretované barvy/debayer.
    """
    if rawpy is None:
        raise RuntimeError("RAW podpora vyžaduje rawpy. Nainstaluj: pip install rawpy")

    with rawpy.imread(str(path)) as raw:
        # Pro Canon CR2 použijeme natvrdo RGGB podle požadavku.
        # Ostatní RAW v ručním režimu používají pattern z rawpy metadat.
        if path.suffix.lower() == ".cr2":
            pattern = "RGGB"
        else:
            pattern = rawpy_bayer_pattern(raw)

        if not pattern:
            # Když metadata selžou, použijeme nejčastější bezpečný fallback.
            pattern = "RGGB"

        raw_img = np.asarray(raw.raw_image_visible).astype(np.float32)

        # Odečtení black levelu. Většina Bayer senzorů má 4 hodnoty pro 2x2 pattern.
        try:
            black = np.asarray(raw.black_level_per_channel, dtype=np.float32)
            if black.size >= 4:
                black2 = np.array([[black[0], black[1]], [black[2], black[3]]], dtype=np.float32)
                tiled_black = np.tile(
                    black2,
                    (raw_img.shape[0] // 2 + 1, raw_img.shape[1] // 2 + 1),
                )[: raw_img.shape[0], : raw_img.shape[1]]
                raw_img = raw_img - tiled_black
            elif black.size == 1:
                raw_img = raw_img - float(black[0])
        except Exception:
            pass

        try:
            white = float(raw.white_level)
        except Exception:
            white = float(np.max(raw_img))

        raw_img = np.clip(raw_img, 0, None)
        maxv = max(1.0, white)
        raw16 = (np.clip(raw_img / maxv, 0, 1) * 65535.0).astype(np.uint16)

    # Použijeme stejnou logiku jako u Bayer FIT.
    return debayer_fits_to_rgb_float(raw16, pattern)

def load_raw_as_float(path: Path) -> np.ndarray:
    """Načte DSLR/MILC RAW soubory (CR2/CR3/NEF/ARW/DNG...) jako lineární RGB float 0..1.

    Vyžaduje:
        pip install rawpy

    Speciální režimy:
    - Sony ARW: ruční Bayer debayer z raw_image_visible.
    - Starší Canon CR2: ruční Bayer debayer z raw_image_visible.
      Pattern se bere z raw.raw_colors_visible, ne natvrdo, aby seděl i po cropu.
    - Ostatní RAW: rawpy.postprocess.
    """
    if rawpy is None:
        raise RuntimeError("RAW podpora vyžaduje rawpy. Nainstaluj: pip install rawpy")

    suffix = path.suffix.lower()

    # ARW a starší CR2: ruční Bayer režim, protože rawpy.postprocess nebo
    # špatný offset Bayer patternu může dát falešné barvy.
    if suffix in {".arw", ".cr2"}:
        return load_raw_bayer_manual_as_float(path)

    with rawpy.imread(str(path)) as raw:
        rgb16 = raw.postprocess(
            output_bps=16,
            no_auto_bright=True,
            use_camera_wb=False,
            gamma=(1, 1),
            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
        )

    rgb = np.asarray(rgb16, dtype=np.float32) / 65535.0
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(rgb, 0, 1).astype(np.float32)


def load_tiff_as_float(path: Path) -> np.ndarray:
    """Load TIFF without silently reducing 16-bit data to 8 bits."""
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ValueError(f"TIFF soubor se nepodařilo načíst: {path.name}")

    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    else:
        raise ValueError(f"Nepodporovaný TIFF rozměr v souboru {path.name}: {arr.shape}")

    if np.issubdtype(arr.dtype, np.integer):
        scale = float(np.iinfo(arr.dtype).max)
        out = arr.astype(np.float32) / max(1.0, scale)
    else:
        out = np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
        hi = float(np.max(out)) if out.size else 0.0
        if hi > 1.000001:
            out /= 65535.0 if hi <= 65535.0 else hi

    return np.ascontiguousarray(np.clip(out, 0, 1).astype(np.float32))


def load_image_as_float(path: Path) -> np.ndarray:
    """Load image as float32 RGB array in range 0..1."""
    suffix = path.suffix.lower()

    if suffix in {".fits", ".fit"}:
        return load_fits_as_float(path)

    if suffix in XISF_EXTENSIONS:
        return load_xisf_as_float(path)

    if suffix in RAW_EXTENSIONS:
        return load_raw_as_float(path)

    if suffix in {".tif", ".tiff"}:
        return load_tiff_as_float(path)

    with Image.open(path) as im:
        im = im.convert("RGB")
        arr = np.asarray(im).astype(np.float32) / 255.0
    return arr


def stackable_extensions(settings: Optional[StackSettings] = None) -> set[str]:
    if settings is not None and (getattr(settings, "raw_only", False) or getattr(settings, "fit_only", False)):
        return RAW_STACK_EXTENSIONS
    return IMAGE_EXTENSIONS


def prepare_calibration_frame(calib: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Připraví flat/bias frame na rozměr light snímku a RGB tvar."""
    arr = np.asarray(calib, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[..., :3]
    h, w = target_shape
    if arr.shape[:2] != (h, w):
        arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_AREA)
    return np.clip(arr.astype(np.float32), 0, 1)


def calibrate_light_frame(
    light: np.ndarray,
    flat: Optional[np.ndarray] = None,
    bias: Optional[np.ndarray] = None,
    dark: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Kalibrace light snímku pomocí volitelného Dark/Flat/Bias.

    Podporované režimy:
    - jen Dark: Light - Dark
    - jen Bias: Light - Bias
    - Flat + Bias: (Light - Bias) / (Flat - Bias)
    - Flat + Dark + Bias: (Light - Dark) / (Flat - Bias)

    Poznámka: Master dark obvykle už obsahuje bias složku, proto má při
    odečítání light snímku přednost před samotným biasem.
    """
    light = np.asarray(light, dtype=np.float32)
    h, w = light.shape[:2]

    if dark is not None:
        subtract_frame = prepare_calibration_frame(dark, (h, w))
    elif bias is not None:
        subtract_frame = prepare_calibration_frame(bias, (h, w))
    else:
        subtract_frame = None

    if subtract_frame is not None:
        numerator = light - subtract_frame
    else:
        numerator = light.copy()

    if flat is not None:
        flat_frame = prepare_calibration_frame(flat, (h, w))
        flat_bias = prepare_calibration_frame(bias, (h, w)) if bias is not None else np.zeros_like(flat_frame)
        denominator = flat_frame - flat_bias
        denom_median = float(np.median(denominator))
        if abs(denom_median) < 1e-8:
            denom_median = float(np.mean(denominator))
        if abs(denom_median) < 1e-8:
            denom_median = 1.0
        denominator = denominator / denom_median
        eps = max(1e-6, float(np.percentile(np.abs(denominator), 5)) * 0.1)
        denominator = np.where(np.abs(denominator) < eps, eps, denominator)
        calibrated = numerator / denominator
    else:
        calibrated = numerator

    calibrated = np.nan_to_num(calibrated, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(calibrated, 0, 1).astype(np.float32)


def prepare_sensor_calibration_frame(calib: np.ndarray, target_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    """Prepare a 2D calibration master for calibration before debayering."""
    arr = np.asarray(calib, dtype=np.float32)
    if arr.ndim != 2:
        return None
    h, w = target_shape
    if arr.shape != (h, w):
        arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_AREA)
    return np.clip(arr.astype(np.float32), 0, 1)


def calibrate_sensor_mosaic(
    light: np.ndarray,
    flat: Optional[np.ndarray] = None,
    bias: Optional[np.ndarray] = None,
    dark: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Calibrate a 2D Bayer mosaic before debayering; return None for RGB masters."""
    light = np.asarray(light, dtype=np.float32)
    if light.ndim != 2:
        return None
    h, w = light.shape

    flat2 = prepare_sensor_calibration_frame(flat, (h, w)) if flat is not None else None
    bias2 = prepare_sensor_calibration_frame(bias, (h, w)) if bias is not None else None
    dark2 = prepare_sensor_calibration_frame(dark, (h, w)) if dark is not None else None
    if (flat is not None and flat2 is None) or (bias is not None and bias2 is None) or (dark is not None and dark2 is None):
        return None

    numerator = light - (dark2 if dark2 is not None else bias2 if bias2 is not None else 0.0)
    if flat2 is not None:
        denominator = flat2 - (bias2 if bias2 is not None else 0.0)
        denom_median = float(np.median(denominator))
        if abs(denom_median) < 1e-8:
            denom_median = float(np.mean(denominator))
        if abs(denom_median) < 1e-8:
            denom_median = 1.0
        denominator = denominator / denom_median
        eps = max(1e-6, float(np.percentile(np.abs(denominator), 5)) * 0.1)
        denominator = np.where(np.abs(denominator) < eps, eps, denominator)
        numerator = numerator / denominator
    return np.clip(np.nan_to_num(numerator, nan=0.0, posinf=1.0, neginf=0.0), 0, 1).astype(np.float32)


def load_calibrated_image_as_float(
    path: Path,
    flat: Optional[np.ndarray] = None,
    bias: Optional[np.ndarray] = None,
    dark: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Load and calibrate before debayering when raw sensor data are available."""
    total_start = time.perf_counter()
    sensor_start = time.perf_counter()
    sensor = load_sensor_mosaic_as_float(path)
    sensor_elapsed = time.perf_counter() - sensor_start
    if sensor is not None:
        mosaic, pattern = sensor
        calibration_start = time.perf_counter()
        calibrated_mosaic = calibrate_sensor_mosaic(mosaic, flat, bias, dark)
        calibration_elapsed = time.perf_counter() - calibration_start
        if calibrated_mosaic is not None:
            debayer_start = time.perf_counter()
            result = debayer_sensor_mosaic_to_rgb_float(calibrated_mosaic, pattern)
            debayer_elapsed = time.perf_counter() - debayer_start
            log_debug(
                "Bayer load/calibrate/debayer "
                f"{Path(path).name}: pattern={pattern} "
                f"load={sensor_elapsed:.3f}s "
                f"calibrate={calibration_elapsed:.3f}s "
                f"debayer={debayer_elapsed:.3f}s "
                f"total={time.perf_counter() - total_start:.3f}s "
                f"shape={result.shape} "
                f"dark={dark is not None} flat={flat is not None} bias={bias is not None}"
            )
            return result

    load_start = time.perf_counter()
    img = load_image_as_float(path)
    load_elapsed = time.perf_counter() - load_start
    calibration_start = time.perf_counter()
    result = calibrate_light_frame(img, flat, bias, dark)
    calibration_elapsed = time.perf_counter() - calibration_start
    log_debug(
        "Image load/calibrate "
        f"{Path(path).name}: sensor_probe={sensor_elapsed:.3f}s "
        f"load={load_elapsed:.3f}s "
        f"calibrate={calibration_elapsed:.3f}s "
        f"total={time.perf_counter() - total_start:.3f}s "
        f"shape={result.shape} "
        f"dark={dark is not None} flat={flat is not None} bias={bias is not None}"
    )
    return result


def load_stack_frame_for_settings(
    path: Path,
    settings: StackSettings,
    flat: Optional[np.ndarray] = None,
    bias: Optional[np.ndarray] = None,
    dark: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Load a frame for stacking while respecting calibration-master mode."""
    if getattr(settings, "align_mode", "") == "calibration":
        sensor = load_sensor_mosaic_as_float(path)
        if sensor is not None:
            return np.ascontiguousarray(sensor[0].astype(np.float32))
        if Path(path).suffix.lower() in FITS_EXTENSIONS:
            return load_calibration_master_fit(path, preserve_mosaic=True)
        return load_image_as_float(path)
    return load_calibrated_image_as_float(path, flat, bias, dark)



def find_calibration_subfolder(folder: Path, names: Tuple[str, ...]) -> Optional[Path]:
    """Najde podsložku kalibračních snímků bez ohledu na velikost písmen."""
    if folder is None:
        return None
    try:
        subdirs = [p for p in Path(folder).iterdir() if p.is_dir()]
    except Exception:
        return None

    wanted = {n.lower() for n in names}
    for sub in subdirs:
        if sub.name.lower() in wanted:
            return sub
    return None


def stack_calibration_folder(
    folder: Path,
    kind: str,
    settings: Optional[StackSettings] = None,
    progress_callback=None,
) -> Optional[np.ndarray]:
    """Složí kalibrační snímky pixel na pixel bez zarovnání.

    kind: Bias / Flat / Dark — pouze pro text ve statusu.
    Používá průměr, který nejlépe využije signál ze všech kalibračních snímků.
    """
    folder = Path(folder)
    paths = calibration_folder_input_paths(folder, settings)
    if not paths:
        return None

    cached_master = load_calibration_master_cache(folder, kind, settings, paths, progress_callback)
    if cached_master is not None:
        return cached_master

    frames = []
    reference_shape = None
    total = len(paths)
    sensor_frames = []
    for path in paths:
        try:
            sensor = load_sensor_mosaic_as_float(path)
        except Exception:
            sensor = None
        if sensor is None:
            sensor_frames = []
            break
        sensor_frames.append(sensor[0])
    use_sensor_mosaics = len(sensor_frames) == total

    for idx, path in enumerate(paths):
        if progress_callback:
            progress_callback(0, f"Skládám {kind} ({idx + 1}/{total}): {path.name}")

        img = sensor_frames[idx] if use_sensor_mosaics else load_image_as_float(path)

        if reference_shape is None:
            reference_shape = img.shape[:2]
        elif img.shape[:2] != reference_shape:
            h, w = reference_shape
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

        frames.append(img.astype(np.float32))

    if not frames:
        return None

    if progress_callback:
        progress_callback(0, f"Skladam {kind} master na CPU po blocich RAM...")

    def calibration_stack_progress(_value: int, message: str):
        if progress_callback:
            progress_callback(0, f"{kind}: {message}")

    master = stack_frames_cpu_tiled_from_sequence(
        frames,
        "mean",
        3.0,
        calibration_stack_progress,
        force_tiled=True,
    ).astype(np.float32)
    save_calibration_master_cache(folder, kind, settings, paths, master)
    return master


def auto_master_calibration_from_subfolders(folder: Path, settings: Optional[StackSettings] = None, progress_callback=None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Automaticky najde podsložky Bias/Flat/Dark a složí master framy."""
    if folder is None:
        return None, None, None

    folder = Path(folder)
    bias_dir = find_calibration_subfolder(folder, ("bias", "biases", "offset", "offsets"))
    flat_dir = find_calibration_subfolder(folder, ("flat", "flats"))
    dark_dir = find_calibration_subfolder(folder, ("dark", "darks"))

    bias = stack_calibration_folder(bias_dir, "Bias", settings, progress_callback) if bias_dir else None
    flat = stack_calibration_folder(flat_dir, "Flat", settings, progress_callback) if flat_dir else None
    dark = stack_calibration_folder(dark_dir, "Dark", settings, progress_callback) if dark_dir else None

    if progress_callback and (bias is not None or flat is not None or dark is not None):
        parts = []
        if bias is not None:
            parts.append("Bias")
        if flat is not None:
            parts.append("Flat")
        if dark is not None:
            parts.append("Dark")
        progress_callback(0, "Automatická kalibrace aktivní: " + ", ".join(parts))

    return flat, bias, dark


def load_manual_calibration_frame(path: Path) -> np.ndarray:
    """Load a manually selected calibration frame.

    FIT/FITS calibration masters must keep their exact linear 0..1 scale.
    Normal image formats are still loaded through the general image path.
    """
    suffix = Path(path).suffix.lower()
    if suffix in FITS_EXTENSIONS:
        return load_calibration_master_fit(path)
    if suffix in XISF_EXTENSIONS:
        data, _metadata = load_xisf_data_and_metadata(path)
        if data.ndim == 3 and data.shape[-1] == 1:
            data = data[..., 0]
        if data.ndim == 3 and data.shape[0] in (3, 4) and data.shape[-1] not in (3, 4):
            data = np.moveaxis(data[:3], 0, -1)
        if data.ndim not in (2, 3):
            raise ValueError(f"Nepodporovaný Master XISF rozměr: {data.shape}")
        return normalize_xisf_linear_to_float(data)
    return load_image_as_float(path)


def load_manual_calibration_source(
    path: Path,
    kind: str,
    settings: Optional[StackSettings] = None,
    progress_callback=None,
) -> Optional[np.ndarray]:
    """Load a selected master file or stack an arbitrary calibration folder."""
    source = Path(path)
    if source.is_dir():
        if progress_callback:
            progress_callback(0, f"Skládám ručně vybranou složku {kind}: {source.name}")
        return stack_calibration_folder(source, kind, settings, progress_callback)
    return load_manual_calibration_frame(source)


def load_calibration_frames(settings: StackSettings, progress_callback=None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Načte ručně zvolené mastery/složky nebo automaticky složí podsložky Bias/Flat/Dark.

    Ručně vybrané zdroje mají přednost. Pokud ručně vybraný zdroj chybí,
    program zkusí najít podsložku Bias, Flat nebo Dark ve složce light snímků.
    """
    flat = None
    bias = None
    dark = None

    # 1) Automatické master framy z podsložek.
    source_folder = getattr(settings, "source_folder", None)
    auto_flat = auto_bias = auto_dark = None
    if source_folder:
        auto_flat, auto_bias, auto_dark = auto_master_calibration_from_subfolders(Path(source_folder), settings, progress_callback)

    # 2) Ruční Flat má přednost před auto Flat.
    flat_path = getattr(settings, "flat_frame_path", None)
    if flat_path:
        p = Path(flat_path)
        if p.exists():
            if progress_callback:
                progress_callback(0, f"Načítám Flat Frame: {p.name}")
            flat = load_manual_calibration_source(p, "Flat", settings, progress_callback)
    if flat is None:
        flat = auto_flat

    # 3) Ruční Bias má přednost před auto Bias.
    bias_path = getattr(settings, "bias_frame_path", None)
    if bias_path:
        p = Path(bias_path)
        if p.exists():
            if progress_callback:
                progress_callback(0, f"Načítám Bias Frame: {p.name}")
            bias = load_manual_calibration_source(p, "Bias", settings, progress_callback)
    if bias is None:
        bias = auto_bias

    # 4) Ruční Dark má přednost před auto Dark.
    dark_path = getattr(settings, "dark_frame_path", None)
    if dark_path:
        p = Path(dark_path)
        if p.exists():
            if progress_callback:
                progress_callback(0, f"Načítám Dark Frame: {p.name}")
            dark = load_manual_calibration_source(p, "Dark", settings, progress_callback)
    if dark is None:
        dark = auto_dark

    if flat is not None and bias is None and progress_callback:
        progress_callback(0, "Flat Frame aktivní, Bias Frame nepoužit — používám Bias = 0")
    if dark is not None and progress_callback:
        progress_callback(0, "Dark Frame aktivní — odečítám Dark od Light snímků")

    return flat, bias, dark


def to_gray_float(img: np.ndarray) -> np.ndarray:
    """Create a grayscale image for alignment/star detection/display analysis.

    Important: this function may use a *temporary* robust normalization so that
    very dark linear FIT data are usable for detection/preview logic. It does
    not modify the original stack data and is never used for FIT export.
    """
    img = np.asarray(img, dtype=np.float32)
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

    if img.ndim == 2:
        gray = img.astype(np.float32)
    else:
        # Work in float directly. Do not quantize to uint8 here, because linear
        # FIT data can occupy only a small part of 0..1 and would look black.
        gray = (0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]).astype(np.float32)

    # Temporary robust scaling for analysis only. This makes the preview/auto-stretch
    # and star/comet detection behave well even with unstretched linear FIT data.
    lo = float(np.percentile(gray, 0.5))
    hi = float(np.percentile(gray, 99.8))
    if hi <= lo:
        lo = float(np.min(gray))
        hi = float(np.max(gray))
    if hi > lo:
        gray = (gray - lo) / (hi - lo)
    gray = np.clip(gray, 0, 1)

    # Log stretch helps stars stand out without over-weighting bright cores.
    gray = np.log1p(gray * 20.0) / np.log1p(20.0)
    return gray.astype(np.float32)


def resize_for_alignment(gray: np.ndarray, scale: float) -> np.ndarray:
    if scale >= 0.999:
        return gray
    h, w = gray.shape[:2]
    return cv2.resize(gray, (max(32, int(w * scale)), max(32, int(h * scale))), interpolation=cv2.INTER_AREA)


def estimate_translation(reference_gray: np.ndarray, moving_gray: np.ndarray, scale: float) -> np.ndarray:
    ref = resize_for_alignment(reference_gray, scale)
    mov = resize_for_alignment(moving_gray, scale)
    window = cv2.createHanningWindow((ref.shape[1], ref.shape[0]), cv2.CV_32F)
    shift, _response = cv2.phaseCorrelate(ref * window, mov * window)
    dx, dy = shift
    # phaseCorrelate returns shift from ref to moving; inverse it to align moving back to ref
    dx = -dx / scale
    dy = -dy / scale
    matrix = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    return matrix


def estimate_ecc_affine(reference_gray: np.ndarray, moving_gray: np.ndarray, scale: float) -> np.ndarray:
    ref = resize_for_alignment(reference_gray, scale)
    mov = resize_for_alignment(moving_gray, scale)

    # Initialize with phase correlation translation for better convergence.
    init = estimate_translation(reference_gray, moving_gray, scale).copy()
    init[:, 2] *= scale
    warp = init.astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-5)
    try:
        _cc, warp = cv2.findTransformECC(
            ref,
            mov,
            warp,
            cv2.MOTION_AFFINE,
            criteria,
            inputMask=None,
            gaussFiltSize=5,
        )
        # findTransformECC returns transform mapping moving to reference for WARP_INVERSE_MAP workflow.
        warp_full = warp.copy()
        warp_full[:, 2] /= scale
        return warp_full.astype(np.float32)
    except cv2.error:
        # Fall back to simple translation if ECC fails.
        return estimate_translation(reference_gray, moving_gray, scale)




def detect_stars(gray: np.ndarray, max_stars: int = 250, threshold_percentile: float = 99.5, min_area: int = 2, max_area: int = 300, border_margin: int = 0, strict_shape: bool = True) -> np.ndarray:
    """Detect bright star-like centroids in a grayscale 0..1 image.

    Vrací pole bodů tvaru N x 2 ve formátu [x, y]. Je to záměrně jednoduchá
    detekce bez SciPy/skimage, aby aplikace neměla další závislosti.
    """
    gray = np.asarray(gray, dtype=np.float32)
    gray = np.nan_to_num(gray, nan=0.0, posinf=0.0, neginf=0.0)

    # Odstraň pomalé pozadí a zvýrazni hvězdy.
    blurred = cv2.GaussianBlur(gray, (0, 0), 3.0)
    highpass = gray - blurred
    highpass = highpass - np.min(highpass)
    max_v = float(np.max(highpass))
    if max_v <= 1e-6:
        return np.empty((0, 2), dtype=np.float32)
    highpass /= max_v

    threshold = np.percentile(highpass, threshold_percentile)
    threshold = max(float(threshold), 0.08)
    mask = (highpass >= threshold).astype(np.uint8)

    # Malé morfologické očištění.
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    stars = []
    height, width = gray.shape[:2]
    # Okraj může být klidně 1000+ px u velkých FIT snímků.
    # Zároveň ho omezíme, aby u menších náhledů neodřízl úplně celé pole.
    margin = int(min(max(0, int(border_margin)), max(0, min(width, height) * 0.48)))

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue

        x, y = centroids[label]
        x0 = int(stats[label, cv2.CC_STAT_LEFT])
        y0 = int(stats[label, cv2.CC_STAT_TOP])
        ww = int(stats[label, cv2.CC_STAT_WIDTH])
        hh = int(stats[label, cv2.CC_STAT_HEIGHT])

        # Stromy, větve, okraje snímku a vinětace často vytvářejí jasné fragmenty u hran.
        # Pro star alignment je bezpečnější je vůbec nenabízet matcheru.
        if margin > 0 and (x < margin or y < margin or x > width - margin or y > height - margin):
            continue

        if strict_shape:
            max_dim = max(ww, hh)
            min_dim = max(1, min(ww, hh))
            aspect = max_dim / min_dim
            fill = area / max(1, ww * hh)

            # Hvězdy jsou malé a relativně kompaktní. Větve/dráty jsou protáhlé;
            # kometa/galaxie jsou příliš velké a difuzní. Saturace malé hvězdy projde.
            if max_dim > 18:
                continue
            if aspect > 2.6:
                continue
            if fill < 0.12 or fill > 0.95:
                continue

        patch = highpass[y0:y0 + hh, x0:x0 + ww]
        brightness = float(np.max(patch)) if patch.size else 0.0

        # Preferuj bodové, kontrastní objekty; trestáme velké/protáhlé komponenty.
        compactness_bonus = area / max(1, ww * hh)
        score = brightness * (0.7 + 0.3 * compactness_bonus)
        stars.append((score, float(x), float(y)))

    if not stars:
        return np.empty((0, 2), dtype=np.float32)

    stars.sort(reverse=True, key=lambda t: t[0])
    pts = np.array([[x, y] for _brightness, x, y in stars[:max_stars]], dtype=np.float32)
    return pts



def estimate_star_translation_by_offsets(reference_pts: np.ndarray, moving_pts: np.ndarray, max_shift: float = 180.0, bin_size: float = 8.0) -> Optional[np.ndarray]:
    """Hrubý odhad velkého posunu podle nejčastějšího offsetu mezi hvězdami.

    Je to užitečné pro EAA sekvence, kde se pole mezi první a poslední expozicí
    posune třeba o 100+ px a phase correlation/ECC už nemusí dát spolehlivý první odhad.
    """
    if len(reference_pts) < 3 or len(moving_pts) < 3:
        return None

    ref = reference_pts[: min(len(reference_pts), 80)]
    mov = moving_pts[: min(len(moving_pts), 80)]
    bins: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}

    max_shift2 = max_shift * max_shift
    for mx, my in mov:
        diffs = ref - np.array([mx, my], dtype=np.float32)
        d2 = np.sum(diffs * diffs, axis=1)
        valid = d2 <= max_shift2
        for dx, dy in diffs[valid]:
            bx = int(round(float(dx) / bin_size))
            by = int(round(float(dy) / bin_size))
            bins.setdefault((bx, by), []).append((float(dx), float(dy)))

    if not bins:
        return None

    best_offsets = max(bins.values(), key=len)
    if len(best_offsets) < 3:
        return None

    arr = np.array(best_offsets, dtype=np.float32)
    dx, dy = np.median(arr, axis=0)
    return np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)


def estimate_star_translation_candidates_by_offsets(
    reference_pts: np.ndarray,
    moving_pts: np.ndarray,
    max_shift: float = 180.0,
    bin_size: float = 8.0,
    max_candidates: int = 8,
) -> List[np.ndarray]:
    """Return several strong translation hypotheses from star-offset voting.

    Dithering can jump in different directions. Keeping only the single best
    offset can miss the true displacement when a few bright stars create a
    false bin, so RANSAC gets multiple plausible starts.
    """
    if len(reference_pts) < 3 or len(moving_pts) < 3:
        return []

    ref = reference_pts[: min(len(reference_pts), 140)]
    mov = moving_pts[: min(len(moving_pts), 140)]
    bins: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}

    max_shift2 = max_shift * max_shift
    for mx, my in mov:
        diffs = ref - np.array([mx, my], dtype=np.float32)
        d2 = np.sum(diffs * diffs, axis=1)
        valid = d2 <= max_shift2
        for dx, dy in diffs[valid]:
            bx = int(round(float(dx) / bin_size))
            by = int(round(float(dy) / bin_size))
            bins.setdefault((bx, by), []).append((float(dx), float(dy)))

    if not bins:
        return []

    ranked = sorted(bins.values(), key=len, reverse=True)
    candidates: List[np.ndarray] = []
    seen: set[Tuple[int, int]] = set()
    for offsets in ranked:
        if len(offsets) < 3:
            continue
        arr = np.array(offsets, dtype=np.float32)
        dx, dy = np.median(arr, axis=0)
        key = (int(round(float(dx) / max(1.0, bin_size))), int(round(float(dy) / max(1.0, bin_size))))
        if key in seen:
            continue
        seen.add(key)
        candidates.append(np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32))
        if len(candidates) >= max_candidates:
            break
    return candidates


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Transform Nx2 points with either a 2x3 affine or 3x3 perspective matrix."""
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float32)
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape == (3, 3):
        projected = cv2.perspectiveTransform(points.reshape(-1, 1, 2).astype(np.float32), matrix)
        return projected.reshape(-1, 2).astype(np.float32)
    ones = np.ones((len(points), 1), dtype=np.float32)
    points_h = np.hstack([points.astype(np.float32), ones])
    return (points_h @ matrix[:2, :].T).astype(np.float32)


def match_stars_nearest(reference_pts: np.ndarray, moving_pts: np.ndarray, initial_matrix: np.ndarray, max_distance: float = 40.0) -> Tuple[np.ndarray, np.ndarray]:
    """Pair moving stars to reference stars after an initial transform estimate."""
    if len(reference_pts) < 3 or len(moving_pts) < 3:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    moving_pred = transform_points(moving_pts, initial_matrix)
    max_d2 = max_distance * max_distance

    ref_arr = np.asarray(reference_pts, dtype=np.float32)
    pred_arr = np.asarray(moving_pred, dtype=np.float32)
    diff = pred_arr[:, None, :] - ref_arr[None, :, :]
    d2 = np.einsum("mri,mri->mr", diff, diff, optimize=True)
    nearest_idx = np.argmin(d2, axis=1)
    nearest_d2 = d2[np.arange(len(pred_arr)), nearest_idx]

    ref_used = set()
    src = []
    dst = []

    # Řadíme od jasnějších hvězd, protože detect_stars vrací body podle jasu.
    for i, j_raw in enumerate(nearest_idx):
        j = int(j_raw)
        if float(nearest_d2[i]) <= max_d2 and j not in ref_used:
            ref_used.add(j)
            src.append(moving_pts[i])
            dst.append(reference_pts[j])

    if len(src) < 3:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    return np.array(src, dtype=np.float32), np.array(dst, dtype=np.float32)


def _star_affine_match_is_excellent(inliers: int, ratio: float, matches: int, ref_count: int, mov_count: int) -> bool:
    """Return True when a candidate is strong enough to skip the remaining search."""
    available = max(1, min(int(ref_count), int(mov_count)))
    min_inliers = max(16, min(70, int(round(available * 0.35))))
    return int(inliers) >= min_inliers and int(matches) >= min_inliers and float(ratio) >= 0.82


def _fit_star_affine_candidate(ref_pts: np.ndarray, mov_pts: np.ndarray, initial_small: np.ndarray, max_distance: float) -> Tuple[Optional[np.ndarray], int, float, int]:
    src, dst = match_stars_nearest(ref_pts, mov_pts, initial_small, max_distance=max_distance)
    if len(src) < 6:
        return None, 0, 0.0, len(src)

    matrix_small, inliers = cv2.estimateAffinePartial2D(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=4.0,
        maxIters=5000,
        confidence=0.997,
        refineIters=30,
    )
    if matrix_small is None or inliers is None:
        return None, 0, 0.0, len(src)

    inlier_count = int(np.sum(inliers))
    inlier_ratio = inlier_count / max(1, len(src))
    return matrix_small.astype(np.float32), inlier_count, inlier_ratio, len(src)


def centered_rotation_matrix(shape: Tuple[int, int], angle_deg: float) -> np.ndarray:
    """Affine matrix for rotating the moving image around its center."""
    h, w = shape[:2]
    center = (float(w) * 0.5, float(h) * 0.5)
    return cv2.getRotationMatrix2D(center, float(angle_deg), 1.0).astype(np.float32)



def detect_comet_center(gray: np.ndarray, border_margin: int = 0, expected_center: Optional[Tuple[float, float]] = None, search_radius: float = 0.0) -> Optional[Tuple[float, float]]:
    """Najde přibližný střed/jádro komety v šedém obraze 0..1.

    Na rozdíl od detekce hvězd hledá spíš jasnější difuzní objekt. Proto se hodí pro
    sekvence, kde je málo hvězd a kamera stojí, ale kometa se mezi expozicemi posouvá.
    """
    gray = np.asarray(gray, dtype=np.float32)
    gray = np.nan_to_num(gray, nan=0.0, posinf=0.0, neginf=0.0)
    if gray.size == 0:
        return None

    h, w = gray.shape[:2]
    margin = int(min(max(0, int(border_margin)), max(0, min(w, h) * 0.48)))

    # Jemné vyhlazení potlačí šum a velké pozadí odečteme, aby vylezlo jádro/kometa.
    smooth = cv2.GaussianBlur(gray, (0, 0), 2.0)
    background = cv2.GaussianBlur(gray, (0, 0), 35.0)
    enhanced = smooth - background
    enhanced = enhanced - np.percentile(enhanced, 5)
    enhanced = np.clip(enhanced, 0, None)
    max_v = float(np.max(enhanced))
    if max_v <= 1e-8:
        return None
    enhanced = enhanced / max_v

    valid = np.ones((h, w), dtype=np.uint8)
    if margin > 0:
        valid[:margin, :] = 0
        valid[-margin:, :] = 0
        valid[:, :margin] = 0
        valid[:, -margin:] = 0

    if expected_center is not None and search_radius and search_radius > 0:
        cx, cy = expected_center
        yy, xx = np.ogrid[:h, :w]
        roi = ((xx - cx) ** 2 + (yy - cy) ** 2) <= float(search_radius) ** 2
        valid &= roi.astype(np.uint8)

    vals = enhanced[valid.astype(bool)]
    if vals.size < 100:
        return None

    # Adaptivní práh: kometa bývá výraznější než pozadí, ale ne nutně bodová.
    thr = max(float(np.percentile(vals, 99.2)), 0.18)
    mask = ((enhanced >= thr) & (valid > 0)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best = None
    best_score = -1.0

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 12 or area > max(20000, int(0.08 * h * w)):
            continue
        x0 = int(stats[label, cv2.CC_STAT_LEFT])
        y0 = int(stats[label, cv2.CC_STAT_TOP])
        ww = int(stats[label, cv2.CC_STAT_WIDTH])
        hh = int(stats[label, cv2.CC_STAT_HEIGHT])

        # Neber komponenty dotýkající se okraje validní oblasti; často to jsou větve/okraje.
        if margin > 0 and (x0 <= margin or y0 <= margin or x0 + ww >= w - margin or y0 + hh >= h - margin):
            continue

        component = labels[y0:y0 + hh, x0:x0 + ww] == label
        patch = enhanced[y0:y0 + hh, x0:x0 + ww]
        weights = np.where(component, patch, 0.0)
        total = float(np.sum(weights))
        if total <= 1e-8:
            continue

        yy, xx = np.mgrid[y0:y0 + hh, x0:x0 + ww]
        cx = float(np.sum(xx * weights) / total)
        cy = float(np.sum(yy * weights) / total)
        peak = float(np.max(patch[component]))

        # Difuzní jasný objekt dostane vyšší skóre než izolovaná malá hvězda.
        score = peak * (area ** 0.45) * (1.0 + min(total, 5.0) * 0.08)
        if expected_center is not None:
            ex, ey = expected_center
            dist = math.hypot(cx - ex, cy - ey)
            score *= max(0.25, 1.0 - dist / max(1.0, float(search_radius) * 1.5))

        if score > best_score:
            best_score = score
            best = (cx, cy)

    return best


def estimate_comet_translation(reference_gray: np.ndarray, moving_gray: np.ndarray, scale: float, border_margin_px: int = 0, max_shift_px: int = 800, reference_center_px: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """Zarovnání na kometu: najde centrum komety v referenci a v pohyblivém snímku.

    Výsledkem je čistý posun. Je vhodný pro komety, kde hvězdy/pozadí nemají být ostré,
    ale kometa ano. Když se kometa nenajde, vrací bezpečný hrubý translation fallback.
    """
    fallback = estimate_translation(reference_gray, moving_gray, scale)

    ref_small = resize_for_alignment(reference_gray, scale)
    mov_small = resize_for_alignment(moving_gray, scale)
    border_small = max(0, int(border_margin_px * scale))
    max_shift_small = max(20.0, float(max_shift_px) * scale)

    if reference_center_px is not None:
        ref_center = (float(reference_center_px[0]) * scale, float(reference_center_px[1]) * scale)
    else:
        ref_center = detect_comet_center(ref_small, border_margin=border_small)
    if ref_center is None:
        return fallback

    # Pro moving snímek preferujeme objekt v okolí referenční pozice, ale s povoleným pohybem komety.
    mov_center = detect_comet_center(
        mov_small,
        border_margin=border_small,
        expected_center=ref_center,
        search_radius=max_shift_small,
    )
    if mov_center is None:
        mov_center = detect_comet_center(mov_small, border_margin=border_small)
    if mov_center is None:
        return fallback

    dx_small = ref_center[0] - mov_center[0]
    dy_small = ref_center[1] - mov_center[1]
    if abs(dx_small) > max_shift_small * 1.25 or abs(dy_small) > max_shift_small * 1.25:
        return fallback

    dx = dx_small / scale
    dy = dy_small / scale
    return np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)



def sequence_paths_for_folder(folder: Path, max_images: int = 0, settings: Optional[StackSettings] = None) -> List[Path]:
    """Vrátí snímky v pořadí sekvence, omezené podle Max. snímků."""
    extensions = stackable_extensions(settings)
    paths = sorted([p for p in folder.iterdir() if p.suffix.lower() in extensions])
    if max_images > 0:
        paths = paths[:max_images]
    return paths


def interpolate_manual_comet_position_for_path(path: Path, sequence_paths: List[Path], settings: StackSettings) -> Optional[Tuple[float, float]]:
    """Lineární poloha komety pro daný snímek z ručně označeného prvního a posledního bodu.

    Tohle je nejspolehlivější režim pro slabou kometu za soumraku: program už nehledá jádro v obraze,
    ale použije pohyb vypočtený ze dvou ručních bodů.
    """
    if settings.manual_comet_xy is None:
        return None

    start_path = Path(settings.manual_comet_reference_path) if settings.manual_comet_reference_path else None
    end_path = Path(settings.manual_comet_end_path) if settings.manual_comet_end_path else None

    if start_path is None or start_path not in sequence_paths:
        return None

    # Jen jeden ruční bod: pozici známe pouze pro referenci, ostatní musí použít automatickou detekci.
    if settings.manual_comet_end_xy is None or end_path is None or end_path not in sequence_paths:
        return settings.manual_comet_xy if path == start_path else None

    i0 = sequence_paths.index(start_path)
    i1 = sequence_paths.index(end_path)
    ip = sequence_paths.index(path) if path in sequence_paths else i0
    if i1 == i0:
        return settings.manual_comet_xy

    t = (ip - i0) / float(i1 - i0)
    x0, y0 = settings.manual_comet_xy
    x1, y1 = settings.manual_comet_end_xy
    return (float(x0) + t * (float(x1) - float(x0)), float(y0) + t * (float(y1) - float(y0)))


def manual_comet_translation_matrix(reference_xy: Tuple[float, float], moving_xy: Tuple[float, float]) -> np.ndarray:
    """Čistý posun, který přesune předpokládanou pozici komety na referenční pozici."""
    dx = float(reference_xy[0]) - float(moving_xy[0])
    dy = float(reference_xy[1]) - float(moving_xy[1])
    return np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)


def crop_with_bounds(gray: np.ndarray, center_xy: Tuple[float, float], radius: int) -> Optional[Tuple[np.ndarray, int, int]]:
    """Vrátí čtvercový výřez a jeho levý horní roh, nebo None když je mimo obraz."""
    h, w = gray.shape[:2]
    cx, cy = float(center_xy[0]), float(center_xy[1])
    r = max(4, int(radius))
    x0 = max(0, int(round(cx)) - r)
    y0 = max(0, int(round(cy)) - r)
    x1 = min(w, int(round(cx)) + r + 1)
    y1 = min(h, int(round(cy)) + r + 1)
    if x1 - x0 < max(9, r) or y1 - y0 < max(9, r):
        return None
    return gray[y0:y1, x0:x1].astype(np.float32), x0, y0


def preprocess_comet_patch(patch: np.ndarray) -> np.ndarray:
    """Připraví lokální výřez komety pro korelaci: potlačí gradient soumraku a větve."""
    patch = np.nan_to_num(np.asarray(patch, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    # Jemné vyhlazení pomůže slabé difuzní kometě, high-pass potlačí pozadí.
    smooth = cv2.GaussianBlur(patch, (0, 0), 1.2)
    background = cv2.GaussianBlur(smooth, (0, 0), max(6.0, min(patch.shape[:2]) / 4.0))
    out = smooth - background
    out -= float(np.median(out))
    std = float(np.std(out))
    if std > 1e-6:
        out /= std
    return out.astype(np.float32)


def refine_comet_position_by_template(
    reference_gray: np.ndarray,
    moving_gray: np.ndarray,
    reference_xy: Tuple[float, float],
    predicted_xy: Tuple[float, float],
    patch_radius: int = 45,
    search_radius: int = 90,
) -> Optional[Tuple[float, float, float]]:
    """Jemně doladí pozici komety kolem dvoubodové predikce pomocí lokální korelace.

    Vrací (x, y, score) v souřadnicích moving snímku. Když je korelace nedůvěryhodná,
    vrátí None a použije se původní dvoubodová predikce.
    """
    patch_radius = max(8, int(patch_radius))
    search_radius = max(patch_radius + 4, int(search_radius))

    ref_crop = crop_with_bounds(reference_gray, reference_xy, patch_radius)
    mov_crop = crop_with_bounds(moving_gray, predicted_xy, patch_radius + search_radius)
    if ref_crop is None or mov_crop is None:
        return None

    template, _tx0, _ty0 = ref_crop
    search, sx0, sy0 = mov_crop
    if search.shape[0] <= template.shape[0] + 2 or search.shape[1] <= template.shape[1] + 2:
        return None

    tmpl = preprocess_comet_patch(template)
    srch = preprocess_comet_patch(search)

    # Slabá kometa může být nízkokontrastní; pokud šablona nemá strukturu, nedolaďujeme.
    if float(np.std(tmpl)) < 0.15 or float(np.std(srch)) < 0.05:
        return None

    res = cv2.matchTemplate(srch, tmpl, cv2.TM_CCOEFF_NORMED)
    _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(res)
    if not np.isfinite(max_val) or max_val < 0.12:
        return None

    # Subpixelové zpřesnění přes kvadratickou interpolaci okolí maxima.
    mx, my = max_loc
    sub_dx = 0.0
    sub_dy = 0.0
    if 0 < mx < res.shape[1] - 1:
        left, center, right = float(res[my, mx - 1]), float(res[my, mx]), float(res[my, mx + 1])
        denom = left - 2.0 * center + right
        if abs(denom) > 1e-6:
            sub_dx = 0.5 * (left - right) / denom
    if 0 < my < res.shape[0] - 1:
        top, center, bottom = float(res[my - 1, mx]), float(res[my, mx]), float(res[my + 1, mx])
        denom = top - 2.0 * center + bottom
        if abs(denom) > 1e-6:
            sub_dy = 0.5 * (top - bottom) / denom

    x = sx0 + mx + sub_dx + template.shape[1] / 2.0
    y = sy0 + my + sub_dy + template.shape[0] / 2.0

    # Bezpečnostní kontrola: doladění nesmí odskočit mimo povolené hledání.
    if abs(x - float(predicted_xy[0])) > search_radius or abs(y - float(predicted_xy[1])) > search_radius:
        return None

    return float(x), float(y), float(max_val)


def comet_alignment_matrix_with_optional_refine(
    reference_gray: np.ndarray,
    moving_gray: np.ndarray,
    settings: StackSettings,
    predicted_xy: Tuple[float, float],
) -> np.ndarray:
    """Matice pro comet alignment: dvoubodová predikce + volitelné lokální doladění."""
    if settings.manual_comet_xy is None:
        return estimate_comet_translation(
            reference_gray,
            moving_gray,
            settings.downscale_for_alignment,
            settings.star_border_margin,
            settings.max_comet_shift,
            settings.manual_comet_xy,
        )

    moving_xy = predicted_xy
    if settings.comet_refine:
        refined = refine_comet_position_by_template(
            reference_gray,
            moving_gray,
            settings.manual_comet_xy,
            predicted_xy,
            patch_radius=settings.comet_refine_patch,
            search_radius=settings.comet_refine_search,
        )
        if refined is not None:
            moving_xy = (refined[0], refined[1])

    return manual_comet_translation_matrix(settings.manual_comet_xy, moving_xy)

def estimate_star_affine_detailed(reference_gray: np.ndarray, moving_gray: np.ndarray, scale: float, max_shift_px: int = 180, border_margin_px: int = 40, strict_star_filter: bool = True) -> Tuple[np.ndarray, str, Dict[str, Any]]:
    """Estimate alignment from detected stars and RANSAC partial affine transform.

    Verze pro EAA: počítá s tím, že mezi referencí a pozdějšími snímky může být drift
    klidně 100+ px. Proto kromě phase correlation zkouší i hrubý posun z hvězdných offsetů.
    """
    initial = estimate_translation(reference_gray, moving_gray, scale)

    ref_small = resize_for_alignment(reference_gray, scale)
    mov_small = resize_for_alignment(moving_gray, scale)

    border_small = max(0, int(border_margin_px * scale))
    ref_pts = detect_stars(ref_small, max_stars=500, threshold_percentile=99.30, border_margin=border_small, strict_shape=strict_star_filter)
    mov_pts = detect_stars(mov_small, max_stars=500, threshold_percentile=99.30, border_margin=border_small, strict_shape=strict_star_filter)

    if len(ref_pts) < 6 or len(mov_pts) < 6:
        return initial, "fallback_insufficient_stars", {
            "reference_stars": int(len(ref_pts)),
            "moving_stars": int(len(mov_pts)),
            "inliers": 0,
            "inlier_ratio": 0.0,
            "matches": 0,
        }

    max_shift_small = max(24.0, float(max_shift_px) * scale)
    search_radius = max(18.0, min(max_shift_small, max(ref_small.shape[:2]) * 0.55))

    candidates: List[np.ndarray] = []
    initial_small = initial.copy()
    initial_small[:, 2] *= scale
    candidates.append(initial_small.astype(np.float32))

    offset_initial = estimate_star_translation_by_offsets(ref_pts, mov_pts, max_shift=max_shift_small, bin_size=max(4.0, 8.0 * scale))
    if offset_initial is not None:
        candidates.append(offset_initial.astype(np.float32))
    offset_candidates = estimate_star_translation_candidates_by_offsets(
        ref_pts,
        mov_pts,
        max_shift=max_shift_small,
        bin_size=max(3.0, 6.0 * scale),
        max_candidates=14,
    )
    coarse_offset_candidates = estimate_star_translation_candidates_by_offsets(
        ref_pts,
        mov_pts,
        max_shift=max_shift_small,
        bin_size=max(8.0, 18.0 * scale),
        max_candidates=8,
    )
    candidates.extend(cand.astype(np.float32) for cand in offset_candidates)
    candidates.extend(cand.astype(np.float32) for cand in coarse_offset_candidates)

    jitter = max(12.0, min(max_shift_small * 0.18, 80.0 * scale))
    for dx in (-jitter, 0.0, jitter):
        for dy in (-jitter, 0.0, jitter):
            if dx == 0.0 and dy == 0.0:
                continue
            shifted = initial_small.astype(np.float32).copy()
            shifted[:, 2] += np.array([dx, dy], dtype=np.float32)
            candidates.append(shifted)

    for angle in range(-90, 91, 5):
        if angle == 0:
            continue
        rot = centered_rotation_matrix(ref_small.shape, angle)
        candidates.append(rot)
        if offset_initial is not None:
            rot_shift = rot.copy()
            rot_shift[:, 2] += offset_initial[:, 2]
            candidates.append(rot_shift.astype(np.float32))
        for offset_candidate in (offset_candidates + coarse_offset_candidates)[:6]:
            rot_shift = rot.copy()
            rot_shift[:, 2] += offset_candidate[:, 2]
            candidates.append(rot_shift.astype(np.float32))

    best_matrix_small = None
    best_score = -1.0
    best_inliers = 0
    best_ratio = 0.0
    best_matches = 0
    candidate_tests = 0
    stop_search = False

    for cand in candidates:
        # Nejdřív užší radius, potom širší. Širší zachrání velký drift, užší snižuje falešné páry.
        radii = (min(22.0, search_radius), min(55.0, search_radius), search_radius)
        for radius in dict.fromkeys(float(r) for r in radii):
            candidate_tests += 1
            matrix_small, inliers, ratio, matches = _fit_star_affine_candidate(ref_pts, mov_pts, cand, max_distance=radius)
            score = inliers * max(0.2, ratio)
            if matrix_small is not None and score > best_score:
                best_matrix_small = matrix_small
                best_score = score
                best_inliers = inliers
                best_ratio = ratio
                best_matches = matches
                if _star_affine_match_is_excellent(inliers, ratio, matches, len(ref_pts), len(mov_pts)):
                    stop_search = True
                    break
        if stop_search:
            break

    flip_180_used = False
    if best_matrix_small is None or best_inliers < 7 or best_ratio < 0.28:
        # Běžný rozsah úhlů výše pokrývá jen -90..+90°, takže meridian flip mountu
        # (typicky přesně ~180° otočení pole) jím neprojde. Než to vzdáme, zkusíme
        # cíleně jen kandidáty kolem 180° - znovu použijeme už nalezené hvězdné body,
        # takže to nestojí žádnou další detekci hvězd, jen pár dalších RANSAC fitů.
        flip_candidates: List[np.ndarray] = []
        for delta_angle in (0.0, -6.0, -3.0, 3.0, 6.0):
            rot_flip = centered_rotation_matrix(ref_small.shape, 180.0 + delta_angle)
            flip_candidates.append(rot_flip)
            if offset_initial is not None:
                rot_flip_shift = rot_flip.copy()
                rot_flip_shift[:, 2] += offset_initial[:, 2]
                flip_candidates.append(rot_flip_shift.astype(np.float32))
            for offset_candidate in (offset_candidates + coarse_offset_candidates)[:8]:
                rot_flip_shift = rot_flip.copy()
                rot_flip_shift[:, 2] += offset_candidate[:, 2]
                flip_candidates.append(rot_flip_shift.astype(np.float32))

        for cand in flip_candidates:
            radii = (min(22.0, search_radius), min(55.0, search_radius), search_radius)
            for radius in dict.fromkeys(float(r) for r in radii):
                candidate_tests += 1
                matrix_small, inliers, ratio, matches = _fit_star_affine_candidate(ref_pts, mov_pts, cand, max_distance=radius)
                score = inliers * max(0.2, ratio)
                if matrix_small is not None and score > best_score:
                    best_matrix_small = matrix_small
                    best_score = score
                    best_inliers = inliers
                    best_ratio = ratio
                    best_matches = matches
                    flip_180_used = True
                    if _star_affine_match_is_excellent(inliers, ratio, matches, len(ref_pts), len(mov_pts)):
                        stop_search = True
                        break
            if stop_search:
                break

    if best_matrix_small is None or best_inliers < 7 or best_ratio < 0.28:
        # U star alignmentu je bezpečnější použít čistý hrubý posun než ECC, které se na slabých EAA datech
        # může chytit galaxie/gradientu a vyrobit jeden snímek mimo.
        return initial, "fallback_ransac_failed", {
            "reference_stars": int(len(ref_pts)),
            "moving_stars": int(len(mov_pts)),
            "inliers": int(best_inliers),
            "inlier_ratio": float(best_ratio),
            "matches": int(best_matches),
            "candidate_tests": int(candidate_tests),
        }

    matrix = best_matrix_small.astype(np.float32).copy()
    matrix[:, 2] /= scale

    a, b = float(matrix[0, 0]), float(matrix[0, 1])
    c, d = float(matrix[1, 0]), float(matrix[1, 1])
    approx_scale = math.sqrt(max(1e-9, abs(a * d - b * c)))
    h, w = reference_gray.shape[:2]
    cx, cy = float(w) * 0.5, float(h) * 0.5
    mapped_cx = a * cx + b * cy + float(matrix[0, 2])
    mapped_cy = c * cx + d * cy + float(matrix[1, 2])
    center_shift = math.hypot(mapped_cx - cx, mapped_cy - cy)
    max_allowed_center_shift = max(float(max_shift_px) * 1.6, max(w, h) * 0.35)
    # Po 180° flipu je posun středu pole typicky výrazně větší než běžný "max_star_shift"
    # (mount po meridian flipu zacílí na stejný objekt, ale rám i kabeláž se otočí),
    # takže pro tento případ povolíme větší toleranci posunu.
    if flip_180_used:
        max_allowed_center_shift = max(max_allowed_center_shift, max(w, h) * 0.75)

    if approx_scale < 0.85 or approx_scale > 1.18 or center_shift > max_allowed_center_shift:
        return initial, "fallback_sanity_failed", {
            "reference_stars": int(len(ref_pts)),
            "moving_stars": int(len(mov_pts)),
            "inliers": int(best_inliers),
            "inlier_ratio": float(best_ratio),
            "matches": int(best_matches),
            "scale": float(approx_scale),
            "center_shift": float(center_shift),
            "candidate_tests": int(candidate_tests),
        }

    return matrix.astype(np.float32), "ransac", {
        "reference_stars": int(len(ref_pts)),
        "moving_stars": int(len(mov_pts)),
        "inliers": int(best_inliers),
        "inlier_ratio": float(best_ratio),
        "matches": int(best_matches),
        "scale": float(approx_scale),
        "center_shift": float(center_shift),
        "candidate_tests": int(candidate_tests),
        "matrix": matrix.astype(np.float32).tolist(),
        "flip_180": flip_180_used,
    }



def estimate_star_affine(reference_gray: np.ndarray, moving_gray: np.ndarray, scale: float, max_shift_px: int = 180, border_margin_px: int = 40, strict_star_filter: bool = True, allow_fallback: bool = True) -> Optional[np.ndarray]:
    matrix, status, _detail = estimate_star_affine_detailed(
        reference_gray,
        moving_gray,
        scale,
        max_shift_px,
        border_margin_px,
        strict_star_filter,
    )
    if status != "ransac" and not allow_fallback:
        return None
    return matrix.astype(np.float32)


def star_alignment_support_ratio(ref_small: np.ndarray, mov_small: np.ndarray, matrix_small: np.ndarray, ref_pts: np.ndarray) -> float:
    """Ověří, že zarovnaný snímek má skutečné hvězdy v místech referenčních hvězd.

    U velké field rotation není validní hodnotit hvězdy v rozích, které po
    transformaci neleží v překryvu s moving snímkem. Tyto body by falešně
    snižovaly support ratio a vyřazovaly správně zarovnané snímky.
    """
    if len(ref_pts) == 0:
        return 0.0

    def highpass_norm(gray: np.ndarray) -> np.ndarray:
        hp = gray.astype(np.float32) - cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 3.0)
        hp -= float(np.min(hp))
        max_v = float(np.max(hp))
        if max_v <= 1e-6:
            return np.zeros_like(gray, dtype=np.float32)
        return (hp / max_v).astype(np.float32)

    h, w = ref_small.shape[:2]
    mh, mw = mov_small.shape[:2]
    mov_aligned = warp_to_reference(mov_small.astype(np.float32), matrix_small.astype(np.float32), (h, w))
    mov_hp = highpass_norm(mov_aligned)
    ref_hp = highpass_norm(ref_small)
    matrix_small = np.asarray(matrix_small, dtype=np.float32)
    if matrix_small.shape == (3, 3):
        try:
            inverse_matrix = np.linalg.inv(matrix_small).astype(np.float32)
        except np.linalg.LinAlgError:
            return 0.0
    else:
        inverse_matrix = cv2.invertAffineTransform(matrix_small.astype(np.float32))

    sample = ref_pts[: min(len(ref_pts), 80)]
    hits = 0
    checked = 0
    for x, y in sample:
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        if xi < 3 or yi < 3 or xi >= w - 3 or yi >= h - 3:
            continue
        src = transform_points(np.array([[float(x), float(y)]], dtype=np.float32), inverse_matrix)[0]
        src_x = float(src[0])
        src_y = float(src[1])
        if src_x < 3 or src_y < 3 or src_x >= mw - 3 or src_y >= mh - 3:
            continue
        ref_patch = ref_hp[yi - 2:yi + 3, xi - 2:xi + 3]
        if float(np.max(ref_patch)) < 0.12:
            continue
        mov_patch = mov_hp[yi - 2:yi + 3, xi - 2:xi + 3]
        checked += 1
        if float(np.max(mov_patch)) >= 0.12:
            hits += 1

    if checked < 6:
        return 0.0
    return hits / float(checked)


def refresh_last_stack_selection_after_alignment(used_paths: List[Path]) -> None:
    """Aktualizuje UI souhrn podle snímků, které opravdu prošly zarovnáním."""
    global LAST_STACK_SELECTION
    if not LAST_STACK_SELECTION:
        return
    used_resolved = []
    used_set = set()
    for path in used_paths:
        try:
            resolved = str(Path(path).resolve())
        except Exception:
            resolved = str(path)
        used_resolved.append(resolved)
        used_set.add(resolved)

    all_paths = list(LAST_STACK_SELECTION.get("all_paths", []))
    selected_paths = list(LAST_STACK_SELECTION.get("selected_paths", []))
    selected_set = set(selected_paths)
    LAST_STACK_SELECTION["used_paths"] = used_resolved
    LAST_STACK_SELECTION["excluded_paths"] = [p for p in all_paths if p not in used_set]
    LAST_STACK_SELECTION["alignment_rejected_paths"] = [p for p in selected_paths if p not in used_set]
    LAST_STACK_SELECTION["quality_excluded_paths"] = [p for p in all_paths if p not in selected_set]


def warp_to_reference(img: np.ndarray, matrix: np.ndarray, shape: Tuple[int, int], border_value: float = 0.0) -> np.ndarray:
    h, w = shape
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape == (3, 3):
        warped = cv2.warpPerspective(
            img,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_value,
        )
    else:
        warped = cv2.warpAffine(
            img,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_value,
        )
    return warped.astype(np.float32)


def warp_valid_coverage(source_shape: Tuple[int, int], matrix: np.ndarray, output_shape: Tuple[int, int]) -> np.ndarray:
    """Return per-pixel source coverage for a warp.

    Normal star alignment keeps the reference frame size, so rotations/translations
    leave invalid borders. Those borders must not participate in median stacking
    or background normalization, otherwise they can create dark/colored bands.
    """
    source_h, source_w = source_shape
    mask = np.ones((source_h, source_w), dtype=np.float32)
    return warp_to_reference(mask, matrix, output_shape, border_value=0.0)


def mark_invalid_warp_pixels(img: np.ndarray, matrix: np.ndarray, output_shape: Tuple[int, int], threshold: float = 0.999) -> np.ndarray:
    coverage = warp_valid_coverage(img.shape[:2], matrix, output_shape)
    valid = coverage >= float(threshold)
    out = np.asarray(img, dtype=np.float32).copy()
    if out.ndim == 3:
        out[~valid, :] = np.nan
    else:
        out[~valid] = np.nan
    return out


def matrix_to_homogeneous(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape == (3, 3):
        return matrix
    if matrix.shape == (2, 3):
        return np.vstack([matrix, [0, 0, 1]]).astype(np.float32)
    raise ValueError(f"Unsupported alignment matrix shape: {matrix.shape}")


def mosaic_canvas_for_matrices(shape: Tuple[int, int], matrices: List[np.ndarray]) -> Tuple[Tuple[int, int], List[np.ndarray]]:
    """Return an expanded mosaic canvas and transforms shifted into its coordinates."""
    h, w = shape
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=np.float32).T
    transformed_corners = []
    homogeneous = []
    for matrix in matrices:
        full = matrix_to_homogeneous(matrix)
        projected = full @ corners
        divisor = np.where(np.abs(projected[2]) > 1e-8, projected[2], 1.0)
        transformed_corners.append((projected[:2] / divisor).T)
        homogeneous.append(full)

    all_points = np.concatenate(transformed_corners, axis=0)
    min_x = float(np.floor(np.min(all_points[:, 0])))
    min_y = float(np.floor(np.min(all_points[:, 1])))
    max_x = float(np.ceil(np.max(all_points[:, 0])))
    max_y = float(np.ceil(np.max(all_points[:, 1])))
    canvas_w = max(1, int(max_x - min_x))
    canvas_h = max(1, int(max_y - min_y))

    # A bad transform must not attempt an enormous allocation.
    if canvas_w > w * 4 or canvas_h > h * 4 or canvas_w * canvas_h > w * h * 16:
        raise ValueError(
            f"Mozaika by vytvořila podezřele velké plátno {canvas_w} × {canvas_h} px. "
            "Zkontroluj alignment nebo sniž maximální drift hvězd."
        )

    offset = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], dtype=np.float32)
    shifted = [(offset @ matrix).astype(np.float32) for matrix in homogeneous]
    return (canvas_h, canvas_w), shifted


def warp_mosaic_records(
    records: List[Tuple[Path, np.ndarray, np.ndarray, Optional[np.ndarray]]],
    reference_shape: Tuple[int, int],
    progress_callback=None,
) -> List[np.ndarray]:
    canvas_shape, shifted_matrices = mosaic_canvas_for_matrices(reference_shape, [record[2] for record in records])
    canvas_h, canvas_w = canvas_shape
    if progress_callback:
        progress_callback(72, f"Mozaika: vytvářím plátno {canvas_w} × {canvas_h} px...")
    total = len(records)
    cpu_count = max(1, os.cpu_count() or 1)
    desired_workers = max(1, min(total, cpu_count - 1 if cpu_count > 2 else cpu_count))
    available = available_system_memory_bytes()
    bytes_per_worker = max(1, canvas_h * canvas_w * np.dtype(np.float32).itemsize * 5)
    max_by_memory = max(1, int((available * 0.30) / bytes_per_worker)) if available > 0 else min(desired_workers, 4)
    workers = max(1, min(desired_workers, max_by_memory))
    aligned: List[Optional[np.ndarray]] = [None] * total

    log_debug(
        f"Mosaic canvas warp: frames={total}, canvas={canvas_w}x{canvas_h}, "
        f"workers={workers}, available={available}, bytes_per_worker={bytes_per_worker}"
    )
    if progress_callback:
        progress_callback(72, f"Mozaika: převádím snímky na plátno ({workers} vláken)...")

    def warp_one(index: int) -> Tuple[int, np.ndarray]:
        _path, img, _matrix, trail_mask = records[index]
        shifted = shifted_matrices[index]
        warped = warp_to_reference(img, shifted, canvas_shape, border_value=0.0)
        coverage = warp_to_reference(np.ones(img.shape[:2], dtype=np.float32), shifted, canvas_shape, border_value=0.0)
        valid = coverage > 1e-4
        if warped.ndim == 3:
            warped = warped / np.maximum(coverage[..., None], 1e-6)
            warped[~valid, :] = np.nan
        else:
            warped = warped / np.maximum(coverage, 1e-6)
            warped[~valid] = np.nan
        warped = apply_warped_exclusion_mask(warped, trail_mask, shifted, canvas_shape)
        return index, warped.astype(np.float32)

    if workers <= 1:
        for completed, index in enumerate(range(total), start=1):
            result_index, warped = warp_one(index)
            aligned[result_index] = warped
            if progress_callback and (completed == total or completed % max(1, total // 10) == 0):
                progress_callback(72 + int(completed / max(1, total) * 6), f"Mozaika: převádím snímky na plátno ({completed}/{total})...")
    else:
        completed = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(warp_one, index) for index in range(total)]
            for future in as_completed(futures):
                result_index, warped = future.result()
                aligned[result_index] = warped
                completed += 1
                if progress_callback and (completed == total or completed % max(1, total // 10) == 0):
                    progress_callback(72 + int(completed / max(1, total) * 6), f"Mozaika: převádím snímky na plátno ({completed}/{total})...")
    return [frame for frame in aligned if frame is not None]


def normalize_background(img: np.ndarray, reference_median: np.ndarray) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim == 2:
        flat = arr.reshape(-1, 1)
    else:
        flat = arr.reshape(-1, arr.shape[-1])

    finite = np.all(np.isfinite(flat), axis=1)
    if np.count_nonzero(finite) < 100:
        return arr

    # Ignore warp borders and clipped black pixels. Real astro backgrounds are
    # normally above zero after calibration; constant zeros mostly mean "no data".
    lum = np.mean(flat, axis=1)
    valid = finite & (lum > 1e-6)
    if np.count_nonzero(valid) < 100:
        valid = finite

    med = np.median(flat[valid], axis=0).astype(np.float32)
    ref = np.asarray(reference_median, dtype=np.float32).reshape(-1)
    if ref.size == 1 and med.size > 1:
        ref = np.full(med.shape, float(ref[0]), dtype=np.float32)
    elif ref.size != med.size:
        ref = np.resize(ref, med.shape).astype(np.float32)

    corrected = arr - med + ref
    return np.clip(corrected, 0, 1)


def sigma_clip_mean(stack: np.ndarray, sigma: float) -> np.ndarray:
    use_nan = not np.isfinite(stack).all()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nanmedian(stack, axis=0) if use_nan else np.median(stack, axis=0)
        std = (np.nanstd(stack, axis=0) if use_nan else np.std(stack, axis=0)) + 1e-6
    keep = np.abs(stack - median) <= sigma * std
    masked = np.where(keep, stack, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        result = np.nanmean(masked, axis=0)
    result = np.where(np.isnan(result), np.nan_to_num(median, nan=0.0), result)
    return result.astype(np.float32)


def high_rejection_mean(stack: np.ndarray, sigma: float) -> np.ndarray:
    """Average frames after rejecting only unusually bright temporal outliers."""
    use_nan = not np.isfinite(stack).all()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nanmedian(stack, axis=0) if use_nan else np.median(stack, axis=0)
        robust_sigma = (np.nanmedian(np.abs(stack - median), axis=0) if use_nan else np.median(np.abs(stack - median), axis=0)) * 1.4826 + 1e-6
    keep = np.isfinite(stack) & (stack <= median + max(0.5, float(sigma)) * robust_sigma)
    counts = np.sum(keep, axis=0)
    summed = np.sum(np.where(keep, stack, 0.0), axis=0)
    result = summed / np.maximum(counts, 1)
    return np.where(counts > 0, result, np.nan_to_num(median, nan=0.0)).astype(np.float32)


def stack_temp_factor(mode: str) -> float:
    if mode in {"sigma", "high_rejection"}:
        return 6.5
    if mode == "median":
        return 3.5
    return 1.8


def available_system_memory_bytes() -> int:
    """Best-effort estimate of currently available RAM without extra dependencies."""
    # Linux / macOS can usually provide this through sysconf. On macOS this is
    # not perfect, but still better than blindly allocating a huge stack.
    try:
        if hasattr(os, "sysconf"):
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            avail_pages_name = "SC_AVPHYS_PAGES"
            if avail_pages_name in os.sysconf_names:
                return max(0, int(os.sysconf(avail_pages_name)) * page_size)
    except Exception:
        pass

    if sys.platform.startswith("win"):
        try:
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return int(stat.ullAvailPhys)
        except Exception:
            pass

    return 0


def limit_processes_for_memory(processes: int, total: int, reference: np.ndarray, progress_callback=None) -> int:
    processes = max(1, min(int(processes), int(total)))
    if not sys.platform.startswith("win"):
        return processes

    available = available_system_memory_bytes()
    if not available or reference is None:
        return processes

    # Worker může dočasně držet načtený snímek, kalibrovaný snímek, gray kopii,
    # warped výstup a serializovanou referenci. Omezení je konzervativní hlavně
    # pro Windows, kde se při spawn režimu paměť kopíruje výrazněji.
    per_process_bytes = max(1, int(np.asarray(reference).nbytes * 5.0))
    max_by_memory = max(1, int((available * 0.60) / per_process_bytes))
    limited = max(1, min(processes, max_by_memory))
    if limited < processes:
        message = (
            f"CPU procesy omezeny kvuli RAM: {processes} -> {limited} "
            f"(volna RAM ~ {available // (1024 * 1024)} MB)"
        )
        log_debug(message)
        if progress_callback:
            progress_callback(5, message)
    return limited


def estimate_stack_bytes(frame_shape: Tuple[int, ...], frame_count: int) -> int:
    if frame_count <= 0:
        return 0
    pixels = int(np.prod(frame_shape))
    return int(frame_count) * pixels * np.dtype(np.float32).itemsize


def should_stack_tiled_on_cpu(frame_shape: Tuple[int, ...], frame_count: int, mode: str) -> Tuple[bool, int, int]:
    stack_bytes = estimate_stack_bytes(frame_shape, frame_count)
    available = available_system_memory_bytes()
    required = int(stack_bytes * stack_temp_factor(mode))

    # Even if available RAM cannot be detected, avoid huge temporary allocations.
    hard_limit = 768 * 1024 * 1024
    if available <= 0:
        return required > hard_limit, required, available

    # Same conservative rule as app5: the aligned list already lives in RAM, so
    # a second full np.stack copy is allowed only when clearly safe.
    safe_budget = int(available * 0.35)
    return required >= safe_budget or stack_bytes >= hard_limit, required, available


def minimum_valid_stack_count(frame_count: int, require_majority_coverage: bool) -> int:
    if not require_majority_coverage:
        return 1
    return max(1, int(frame_count) // 2 + 1)


def apply_numpy_minimum_coverage(result: np.ndarray, counts: np.ndarray, frame_count: int, require_majority_coverage: bool) -> np.ndarray:
    if not require_majority_coverage:
        return result
    min_count = minimum_valid_stack_count(frame_count, True)
    return np.where(counts >= min_count, result, 0.0).astype(np.float32)


def normalize_angle_90(angle_deg: float) -> float:
    """Normalize an angle to the nearest horizontal/vertical axis."""
    angle = float(angle_deg)
    while angle <= -90.0:
        angle += 180.0
    while angle > 90.0:
        angle -= 180.0
    if angle > 45.0:
        angle -= 90.0
    elif angle < -45.0:
        angle += 90.0
    return angle


def estimate_valid_frame_angle(img: np.ndarray) -> Optional[float]:
    """Estimate the tilt of the valid stacked-image footprint from dark borders."""
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim == 3:
        lum = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
        finite = np.all(np.isfinite(arr[..., :3]), axis=2)
    else:
        lum = arr
        finite = np.isfinite(arr)
    h, w = lum.shape[:2]
    if h < 64 or w < 64:
        return None

    finite_lum = lum[finite]
    finite_lum = finite_lum[np.isfinite(finite_lum)]
    if finite_lum.size < 1000:
        return None

    positive = finite_lum[finite_lum > 1e-6]
    if positive.size < finite_lum.size * 0.10:
        return None

    center = lum[h // 4: (3 * h) // 4, w // 4: (3 * w) // 4]
    center = center[np.isfinite(center)]
    center_positive = center[center > 1e-6]
    thresholds = [max(1e-6, float(np.percentile(positive, 0.5)) * 0.25)]
    if center_positive.size > 100:
        center_median = float(np.median(center_positive))
        thresholds.extend([
            max(1e-6, float(np.percentile(center_positive, 10.0)) * 0.25),
            max(1e-6, float(np.percentile(center_positive, 20.0)) * 0.35),
            max(1e-6, center_median * 0.15),
            max(1e-6, center_median * 0.25),
            max(1e-6, center_median * 0.40),
        ])

    kernel = np.ones((7, 7), np.uint8)
    selected_contour = None
    selected_area = 0.0
    frame_area = float(h * w)
    for threshold in sorted(set(float(t) for t in thresholds)):
        mask = ((lum > threshold) & finite).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        largest = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(largest))
        area_ratio = area / frame_area if frame_area > 0 else 0.0
        if 0.20 <= area_ratio <= 0.985 and area > selected_area:
            selected_contour = largest
            selected_area = area

    if selected_contour is None:
        return None

    box = cv2.boxPoints(cv2.minAreaRect(selected_contour)).astype(np.float32)
    edge_angles = []
    for idx in range(4):
        p0 = box[idx]
        p1 = box[(idx + 1) % 4]
        dx = float(p1[0] - p0[0])
        dy = float(p1[1] - p0[1])
        length = math.hypot(dx, dy)
        if length < 8:
            continue
        edge_angles.append((length, normalize_angle_90(math.degrees(math.atan2(dy, dx)))))
    if not edge_angles:
        return None

    # Prefer the edge that is closest to horizontal. It gives the visible frame
    # tilt regardless of whether the footprint is portrait or landscape.
    _length, angle = min(edge_angles, key=lambda item: abs(item[1]))
    if abs(angle) < 0.15 or abs(angle) > 12.0:
        return None
    return float(angle)


def rotate_image_expand(img: np.ndarray, angle_deg: float) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    h, w = arr.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, float(angle_deg), 1.0).astype(np.float32)
    cos_a = abs(float(matrix[0, 0]))
    sin_a = abs(float(matrix[0, 1]))
    new_w = int(math.ceil(h * sin_a + w * cos_a))
    new_h = int(math.ceil(h * cos_a + w * sin_a))
    matrix[0, 2] += new_w / 2.0 - center[0]
    matrix[1, 2] += new_h / 2.0 - center[1]
    rotated = cv2.warpAffine(
        arr,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    return np.clip(rotated, 0, 1).astype(np.float32)


def auto_rotate_stacked_footprint(img: np.ndarray) -> Tuple[np.ndarray, Optional[float]]:
    angle = estimate_valid_frame_angle(img)
    if angle is None:
        return np.asarray(img, dtype=np.float32), None
    # Rotate in the opposite direction of the detected frame tilt.
    return rotate_image_expand(img, -angle), -angle


def stack_frames_cpu_tiled_from_sequence(frames: List[np.ndarray], mode: str, sigma: float, progress_callback=None, force_tiled: bool = False, require_majority_coverage: bool = False) -> np.ndarray:
    if not frames:
        raise ValueError("No frames to stack.")
    if len(frames) == 1:
        return np.asarray(frames[0], dtype=np.float32).copy()

    first = np.asarray(frames[0], dtype=np.float32)
    n = len(frames)
    h, w = first.shape[:2]
    channels = int(np.prod(first.shape[2:])) if first.ndim > 2 else 1
    full_stack_bytes = sum(int(np.asarray(frame).nbytes) for frame in frames)
    available = available_system_memory_bytes()
    temp_factor = stack_temp_factor(mode)

    safe_full_stack = bool(available and full_stack_bytes * temp_factor < available * 0.35)
    if not force_tiled and safe_full_stack and full_stack_bytes < 768 * 1024 * 1024:
        log_debug(
            f"CPU stack uses full array: mode={mode}, frames={n}, shape={first.shape}, "
            f"stack_bytes={full_stack_bytes}, available_mem={available}"
        )
        return stack_frames_cpu(
            np.stack(frames, axis=0),
            mode,
            sigma,
            require_majority_coverage=require_majority_coverage,
        )

    bytes_per_row = max(1, n * w * channels * np.dtype(np.float32).itemsize)
    cpu_count = max(1, os.cpu_count() or 1)
    worker_limit = cpu_count if force_tiled else 8
    desired_workers = max(1, min(cpu_count - 1 if cpu_count > 2 else cpu_count, worker_limit))
    parallel_budget = int((available or 768 * 1024 * 1024) * 0.30)
    target_bytes = int(parallel_budget / max(1, desired_workers))
    target_bytes = max(16 * 1024 * 1024, min(target_bytes, 256 * 1024 * 1024))
    rows_per_tile = max(4, int(target_bytes / max(1, bytes_per_row * temp_factor)))
    rows_per_tile = min(h, rows_per_tile)
    if force_tiled and desired_workers > 1:
        # Mosaic mode should not collapse into one large NumPy operation when
        # RAM is plentiful. Create enough row ranges to keep CPU cores busy.
        rows_per_tile = min(rows_per_tile, max(4, int(math.ceil(h / desired_workers))))
    result = np.empty(first.shape, dtype=np.float32)
    tile_ranges = [(y0, min(h, y0 + rows_per_tile)) for y0 in range(0, h, rows_per_tile)]
    tile_temp_bytes = int(bytes_per_row * rows_per_tile * temp_factor)
    workers = cpu_stack_tile_worker_count(tile_temp_bytes, len(tile_ranges), available)

    log_debug(
        f"CPU tiled stack: mode={mode}, frames={n}, shape={first.shape}, "
        f"rows_per_tile={rows_per_tile}, stack_bytes={full_stack_bytes}, "
        f"available={available}, bytes_per_row={bytes_per_row}, workers={workers}, "
        f"tiles={len(tile_ranges)}, tile_temp_bytes={tile_temp_bytes}"
    )
    if progress_callback:
        progress_callback(80, f"Skladam na CPU po blocich RAM ({workers} vlaken, {rows_per_tile} radku)...")

    def compute_tile(y0: int, y1: int) -> Tuple[int, int, np.ndarray]:
        tile = np.stack([np.asarray(frame[y0:y1, ...], dtype=np.float32) for frame in frames], axis=0)
        tile_result = stack_frames_cpu(tile, mode, sigma, require_majority_coverage=require_majority_coverage)
        return y0, y1, tile_result

    if workers <= 1 or len(tile_ranges) <= 1:
        for tile_idx, (y0, y1) in enumerate(tile_ranges, start=1):
            _, _, tile_result = compute_tile(y0, y1)
            result[y0:y1, ...] = tile_result
            if progress_callback:
                pct = 80 + int((tile_idx / max(1, len(tile_ranges))) * 18)
                progress_callback(min(98, pct), f"Skladam na CPU po blocich RAM ({tile_idx}/{len(tile_ranges)})...")
            del tile_result
    else:
        completed = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(compute_tile, y0, y1) for y0, y1 in tile_ranges]
            for future in as_completed(futures):
                y0, y1, tile_result = future.result()
                result[y0:y1, ...] = tile_result
                completed += 1
                if progress_callback:
                    pct = 80 + int((completed / max(1, len(tile_ranges))) * 18)
                    progress_callback(min(98, pct), f"Skladam na CPU po blocich RAM ({completed}/{len(tile_ranges)})...")
                del tile_result
    return np.clip(result, 0, 1).astype(np.float32)


def stack_frames_cpu_mean_streaming(frames: List[np.ndarray], progress_callback=None) -> np.ndarray:
    if not frames:
        raise ValueError("No frames to stack.")
    result = np.zeros_like(np.asarray(frames[0], dtype=np.float32), dtype=np.float32)
    total = len(frames)
    for idx, frame in enumerate(frames, start=1):
        result += np.asarray(frame, dtype=np.float32)
        if progress_callback and (idx == total or idx % max(1, total // 20) == 0):
            pct = 80 + int(idx / max(1, total) * 18)
            progress_callback(min(98, pct), f"Skladam prumer prubezne ({idx}/{total})...")
    result /= max(1, total)
    return np.clip(result, 0, 1).astype(np.float32)


def gpu_available() -> bool:
    global GPU_AVAILABLE_ERROR, GPU_AVAILABLE_DETAIL
    GPU_AVAILABLE_ERROR = None
    GPU_AVAILABLE_DETAIL = ""
    if cp is None:
        log_debug("gpu_available: cp is None")
        return False
    try:
        count = int(cp.cuda.runtime.getDeviceCount())
        GPU_AVAILABLE_DETAIL = f"CUDA devices: {count}"
        log_debug(GPU_AVAILABLE_DETAIL)
        if count <= 0:
            return False
        try:
            props = cp.cuda.runtime.getDeviceProperties(0)
            name = props.get("name", b"")
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            GPU_AVAILABLE_DETAIL = f"CUDA devices: {count}, GPU: {name}"
            log_debug(GPU_AVAILABLE_DETAIL)
        except Exception as prop_exc:
            GPU_AVAILABLE_DETAIL = f"CUDA devices: {count}, GPU name error: {prop_exc}"
            log_debug(GPU_AVAILABLE_DETAIL)
        return True
    except Exception as exc:
        GPU_AVAILABLE_ERROR = traceback.format_exc()
        log_debug(f"gpu_available failed:\n{GPU_AVAILABLE_ERROR}")
        return False


def mps_available() -> bool:
    """Vrátí True, pokud je dostupné Apple Metal/MPS přes PyTorch."""
    global MPS_AVAILABLE_ERROR, MPS_AVAILABLE_DETAIL
    MPS_AVAILABLE_ERROR = None
    MPS_AVAILABLE_DETAIL = ""
    if torch is None:
        log_debug("mps_available: torch is None")
        return False
    try:
        available = bool(torch.backends.mps.is_available())
        built = bool(torch.backends.mps.is_built())
        MPS_AVAILABLE_DETAIL = f"PyTorch MPS available={available}, built={built}, torch={getattr(torch, '__version__', '?')}"
        log_debug(MPS_AVAILABLE_DETAIL)
        return available and built
    except Exception:
        MPS_AVAILABLE_ERROR = traceback.format_exc()
        log_debug(f"mps_available failed:\n{MPS_AVAILABLE_ERROR}")
        return False


def stack_frames_cpu(arr: np.ndarray, mode: str, sigma: float, require_majority_coverage: bool = False) -> np.ndarray:
    use_nan = not np.isfinite(arr).all()
    counts = np.sum(np.isfinite(arr), axis=0) if use_nan else np.full(arr.shape[1:], arr.shape[0], dtype=np.int32)
    if mode == "median":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            result = np.nanmedian(arr, axis=0) if use_nan else np.median(arr, axis=0)
        result = np.nan_to_num(result, nan=0.0).astype(np.float32)
        return apply_numpy_minimum_coverage(result, counts, arr.shape[0], require_majority_coverage)
    if mode == "mean":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            result = np.nanmean(arr, axis=0) if use_nan else np.mean(arr, axis=0)
        result = np.nan_to_num(result, nan=0.0).astype(np.float32)
        return apply_numpy_minimum_coverage(result, counts, arr.shape[0], require_majority_coverage)
    if mode == "high_rejection":
        result = high_rejection_mean(arr, sigma)
        return apply_numpy_minimum_coverage(result, counts, arr.shape[0], require_majority_coverage)
    result = sigma_clip_mean(arr, sigma)
    return apply_numpy_minimum_coverage(result, counts, arr.shape[0], require_majority_coverage)


def cpu_stack_tile_worker_count(tile_temp_bytes: int, tile_count: int, available: int) -> int:
    cpu_count = max(1, os.cpu_count() or 1)
    desired = max(1, min(tile_count, cpu_count - 1 if cpu_count > 2 else cpu_count))
    if desired <= 1:
        return 1
    if tile_temp_bytes <= 0 or available <= 0:
        return min(desired, 4)
    # Aligned frames already live in RAM; leave a generous reserve for the OS
    # and for NumPy temporaries inside median/sigma calculations.
    memory_budget = int(available * 0.30)
    max_by_memory = max(1, int(memory_budget / max(1, tile_temp_bytes)))
    return max(1, min(desired, max_by_memory))


def torch_masked_median(tensor, valid=None):
    """NumPy-compatible masked median without torch.nanmedian() on Apple MPS."""
    if valid is None:
        valid = torch.isfinite(tensor)
    counts = valid.sum(dim=0)
    fill = torch.full((), float("inf"), dtype=tensor.dtype, device=tensor.device)
    sorted_values = torch.sort(torch.where(valid, tensor, fill), dim=0).values
    lower_indices = torch.clamp((counts - 1) // 2, min=0).unsqueeze(0)
    upper_indices = torch.clamp(counts // 2, min=0).unsqueeze(0)
    lower = torch.gather(sorted_values, 0, lower_indices).squeeze(0)
    upper = torch.gather(sorted_values, 0, upper_indices).squeeze(0)
    median = (lower + upper) * 0.5
    zero = torch.zeros((), dtype=tensor.dtype, device=tensor.device)
    return torch.where(counts > 0, median, zero)


def stack_frames_torch_tensor(tensor, mode: str, sigma: float, require_majority_coverage: bool = False):
    valid = torch.isfinite(tensor)
    zero = torch.zeros((), dtype=tensor.dtype, device=tensor.device)
    safe_tensor = torch.where(valid, tensor, zero)
    counts = valid.sum(dim=0)
    safe_counts = torch.clamp(counts, min=1).to(dtype=tensor.dtype)
    min_count = minimum_valid_stack_count(int(tensor.shape[0]), require_majority_coverage)

    def apply_min_coverage(result):
        if not require_majority_coverage:
            return result
        return torch.where(counts >= min_count, result, zero)

    if mode == "mean":
        result = safe_tensor.sum(dim=0) / safe_counts
        return apply_min_coverage(torch.where(counts > 0, result, zero))

    median = torch_masked_median(tensor, valid)
    median_safe = torch.nan_to_num(median, nan=0.0)
    if mode == "median":
        return apply_min_coverage(median_safe)

    if mode == "high_rejection":
        deviations = torch.where(valid, torch.abs(tensor - median), zero)
        robust_sigma = torch_masked_median(deviations, valid) * 1.4826
        robust_sigma = torch.nan_to_num(robust_sigma, nan=0.0) + 1e-6
        keep = valid & (tensor <= median + max(0.5, float(sigma)) * robust_sigma)
        accepted_counts = keep.sum(dim=0)
        summed = torch.where(keep, tensor, zero).sum(dim=0)
        safe_accepted_counts = torch.clamp(accepted_counts, min=1).to(dtype=tensor.dtype)
        return apply_min_coverage(torch.where(accepted_counts > 0, summed / safe_accepted_counts, median_safe))

    mean = safe_tensor.sum(dim=0) / safe_counts
    centered = torch.where(valid, tensor - mean, zero)
    std = torch.sqrt((centered * centered).sum(dim=0) / safe_counts) + 1e-6
    keep = valid & (torch.abs(tensor - median) <= float(sigma) * std)
    accepted_counts = keep.sum(dim=0)
    summed = torch.where(keep, tensor, zero).sum(dim=0)
    safe_accepted_counts = torch.clamp(accepted_counts, min=1).to(dtype=tensor.dtype)
    result = summed / safe_accepted_counts
    return apply_min_coverage(torch.where(accepted_counts > 0, result, median_safe))


def stack_frames_mps(arr: np.ndarray, mode: str, sigma: float) -> np.ndarray:
    if torch is None:
        raise RuntimeError("PyTorch neni nainstalovany.")
    if not mps_available():
        raise RuntimeError("Apple Metal/MPS neni dostupne.")

    arr = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
    if arr.ndim < 3:
        raise ValueError("MPS stacking expects stack array with shape (frames, height, width[, channels]).")

    # Metal/MPS používá sdílenou paměť, ale median/sigma vytvářejí velké dočasné
    # tenzory. Nad tento limit zpracujeme obraz po řádkových dlaždicích.
    safety_factor = 5.0 if mode in {"median", "sigma", "high_rejection"} else 2.5
    target_bytes = 768 * 1024 * 1024
    if arr.nbytes * safety_factor > target_bytes:
        log_debug(
            f"MPS stack uses tiled mode: mode={mode}, arr={arr.shape}, "
            f"arr_bytes={arr.nbytes}, target_bytes={target_bytes}"
        )
        return stack_frames_mps_tiled(arr, mode, sigma, target_bytes)

    device = torch.device("mps")
    with torch.no_grad():
        tensor = torch.from_numpy(arr).to(device)
        result = stack_frames_torch_tensor(tensor, mode, sigma)
        result = torch.clamp(result, 0, 1).to(dtype=torch.float32)
        out = result.cpu().numpy()
        del tensor, result
    try:
        torch.mps.empty_cache()
    except Exception:
        pass
    return out.astype(np.float32)


def stack_frames_mps_tiled(arr: np.ndarray, mode: str, sigma: float, target_bytes: int) -> np.ndarray:
    if torch is None:
        raise RuntimeError("PyTorch neni nainstalovany.")

    n, h, w = arr.shape[:3]
    channels = int(np.prod(arr.shape[3:])) if arr.ndim > 3 else 1
    bytes_per_row = max(1, n * w * channels * np.dtype(np.float32).itemsize)
    temp_factor = 6.0 if mode in {"sigma", "high_rejection"} else 4.0 if mode == "median" else 2.0
    rows_per_tile = max(8, int(target_bytes / max(1, bytes_per_row * temp_factor)))
    rows_per_tile = min(h, rows_per_tile)
    result_cpu = np.empty(arr.shape[1:], dtype=np.float32)
    device = torch.device("mps")

    log_debug(
        f"MPS tiled params: rows_per_tile={rows_per_tile}, h={h}, w={w}, "
        f"frames={n}, channels={channels}, bytes_per_row={bytes_per_row}"
    )

    with torch.no_grad():
        for y0 in range(0, h, rows_per_tile):
            y1 = min(h, y0 + rows_per_tile)
            tile = torch.from_numpy(np.ascontiguousarray(arr[:, y0:y1, ...])).to(device)
            tile_result = stack_frames_torch_tensor(tile, mode, sigma)
            tile_result = torch.clamp(tile_result, 0, 1).to(dtype=torch.float32)
            result_cpu[y0:y1, ...] = tile_result.cpu().numpy()
            del tile, tile_result

    # Reuse Metal buffers between tiles; clearing per tile forces a costly
    # synchronization and repeated allocations.
    try:
        torch.mps.empty_cache()
    except Exception:
        pass

    return result_cpu.astype(np.float32)


def stack_frames_gpu(arr: np.ndarray, mode: str, sigma: float) -> np.ndarray:
    if cp is None:
        raise RuntimeError("CuPy neni nainstalovane.")

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim < 3:
        raise ValueError("GPU stacking expects stack array with shape (frames, height, width[, channels]).")

    try:
        free_mem, total_mem = cp.cuda.runtime.memGetInfo()
    except Exception:
        free_mem, total_mem = 0, 0

    # Median/sigma need temporary arrays in addition to the input tile. If the
    # full stack is too large for VRAM, process spatial tiles instead.
    safety_factor = 5.0 if mode in {"median", "sigma", "high_rejection"} else 2.5
    if free_mem and arr.nbytes * safety_factor > free_mem:
        log_debug(
            f"GPU stack uses tiled mode: mode={mode}, arr={arr.shape}, "
            f"arr_bytes={arr.nbytes}, free_mem={free_mem}, total_mem={total_mem}"
        )
        return stack_frames_gpu_tiled(arr, mode, sigma, free_mem)

    with cp.cuda.Device(0):
        gpu_arr = cp.asarray(arr, dtype=cp.float32)
        valid = cp.isfinite(gpu_arr)
        if mode == "median":
            result = cp.nanmedian(gpu_arr, axis=0)
        elif mode == "mean":
            counts = cp.sum(valid, axis=0)
            summed = cp.sum(cp.where(valid, gpu_arr, cp.float32(0.0)), axis=0)
            result = cp.where(counts > 0, summed / cp.maximum(counts, 1), cp.float32(0.0))
        elif mode == "high_rejection":
            median = cp.nanmedian(gpu_arr, axis=0)
            median_safe = cp.nan_to_num(median, nan=0.0)
            robust_sigma = cp.nanmedian(cp.abs(gpu_arr - median), axis=0) * cp.float32(1.4826)
            robust_sigma = cp.nan_to_num(robust_sigma, nan=0.0) + cp.float32(1e-6)
            keep = valid & (gpu_arr <= median + cp.float32(max(0.5, float(sigma))) * robust_sigma)
            counts = cp.sum(keep, axis=0)
            summed = cp.sum(cp.where(keep, gpu_arr, cp.float32(0.0)), axis=0)
            result = cp.where(counts > 0, summed / cp.maximum(counts, 1), median_safe)
        else:
            median = cp.nanmedian(gpu_arr, axis=0)
            median_safe = cp.nan_to_num(median, nan=0.0)
            std = cp.nanstd(gpu_arr, axis=0) + cp.float32(1e-6)
            keep = valid & (cp.abs(gpu_arr - median) <= cp.float32(float(sigma)) * std)
            counts = cp.sum(keep, axis=0)
            summed = cp.sum(cp.where(keep, gpu_arr, cp.float32(0.0)), axis=0)
            result = cp.where(counts > 0, summed / cp.maximum(counts, 1), median_safe)
        result = cp.clip(cp.nan_to_num(result, nan=0.0), 0, 1).astype(cp.float32)
        out = cp.asnumpy(result)

    try:
        cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass
    return out.astype(np.float32)


def stack_frames_gpu_tiled(arr: np.ndarray, mode: str, sigma: float, free_mem: int = 0) -> np.ndarray:
    if cp is None:
        raise RuntimeError("CuPy neni nainstalovane.")

    n, h, w = arr.shape[:3]
    channels = int(np.prod(arr.shape[3:])) if arr.ndim > 3 else 1
    bytes_per_row = max(1, n * w * channels * np.dtype(np.float32).itemsize)
    temp_factor = 6.0 if mode in {"sigma", "high_rejection"} else 4.0 if mode == "median" else 2.0
    target_bytes = min(512 * 1024 * 1024, int((free_mem or 512 * 1024 * 1024) * 0.40))
    rows_per_tile = max(8, int(target_bytes / max(1, bytes_per_row * temp_factor)))
    rows_per_tile = min(h, rows_per_tile)

    result_cpu = np.empty(arr.shape[1:], dtype=np.float32)
    log_debug(
        f"GPU tiled params: rows_per_tile={rows_per_tile}, h={h}, w={w}, "
        f"frames={n}, channels={channels}, bytes_per_row={bytes_per_row}"
    )

    with cp.cuda.Device(0):
        for y0 in range(0, h, rows_per_tile):
            y1 = min(h, y0 + rows_per_tile)
            tile = cp.asarray(arr[:, y0:y1, ...], dtype=cp.float32)
            valid = cp.isfinite(tile)
            if mode == "median":
                tile_result = cp.nanmedian(tile, axis=0)
            elif mode == "mean":
                counts = cp.sum(valid, axis=0)
                summed = cp.sum(cp.where(valid, tile, cp.float32(0.0)), axis=0)
                tile_result = cp.where(counts > 0, summed / cp.maximum(counts, 1), cp.float32(0.0))
            elif mode == "high_rejection":
                median = cp.nanmedian(tile, axis=0)
                median_safe = cp.nan_to_num(median, nan=0.0)
                robust_sigma = cp.nanmedian(cp.abs(tile - median), axis=0) * cp.float32(1.4826)
                robust_sigma = cp.nan_to_num(robust_sigma, nan=0.0) + cp.float32(1e-6)
                keep = valid & (tile <= median + cp.float32(max(0.5, float(sigma))) * robust_sigma)
                counts = cp.sum(keep, axis=0)
                summed = cp.sum(cp.where(keep, tile, cp.float32(0.0)), axis=0)
                tile_result = cp.where(counts > 0, summed / cp.maximum(counts, 1), median_safe)
            else:
                median = cp.nanmedian(tile, axis=0)
                median_safe = cp.nan_to_num(median, nan=0.0)
                std = cp.nanstd(tile, axis=0) + cp.float32(1e-6)
                keep = valid & (cp.abs(tile - median) <= cp.float32(float(sigma)) * std)
                counts = cp.sum(keep, axis=0)
                summed = cp.sum(cp.where(keep, tile, cp.float32(0.0)), axis=0)
                tile_result = cp.where(counts > 0, summed / cp.maximum(counts, 1), median_safe)

            tile_result = cp.clip(cp.nan_to_num(tile_result, nan=0.0), 0, 1).astype(cp.float32)
            result_cpu[y0:y1, ...] = cp.asnumpy(tile_result)
            del tile, tile_result

        # Let CuPy reuse tile buffers and release the pool once the operation
        # is complete instead of synchronizing after every tile.
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass

    return result_cpu.astype(np.float32)


def stack_frames_gpu_tiled_from_sequence(frames: List[np.ndarray], mode: str, sigma: float, progress_callback=None, require_majority_coverage: bool = False) -> np.ndarray:
    """Upload row tiles directly from aligned frames instead of building a full RAM stack."""
    if cp is None:
        raise RuntimeError("CuPy neni nainstalovane.")
    if not frames:
        raise ValueError("No frames to stack.")

    first = np.asarray(frames[0], dtype=np.float32)
    n = len(frames)
    h, w = first.shape[:2]
    channels = int(np.prod(first.shape[2:])) if first.ndim > 2 else 1
    bytes_per_row = max(1, n * w * channels * np.dtype(np.float32).itemsize)
    temp_factor = 6.0 if mode in {"sigma", "high_rejection"} else 4.0 if mode == "median" else 2.0
    try:
        free_mem, _total_mem = cp.cuda.runtime.memGetInfo()
    except Exception:
        free_mem = 0
    target_bytes = min(512 * 1024 * 1024, int((free_mem or 512 * 1024 * 1024) * 0.40))
    rows_per_tile = max(8, int(target_bytes / max(1, bytes_per_row * temp_factor)))
    rows_per_tile = min(h, rows_per_tile)
    result_cpu = np.empty(first.shape, dtype=np.float32)

    started_at = time.perf_counter()
    log_debug(
        f"GPU sequence tiles: mode={mode}, frames={n}, shape={first.shape}, "
        f"rows_per_tile={rows_per_tile}, free_mem={free_mem}"
    )
    with cp.cuda.Device(0):
        for tile_idx, y0 in enumerate(range(0, h, rows_per_tile), start=1):
            y1 = min(h, y0 + rows_per_tile)
            host_tile = np.stack([np.asarray(frame[y0:y1, ...], dtype=np.float32) for frame in frames], axis=0)
            tile = cp.asarray(host_tile, dtype=cp.float32)
            valid = cp.isfinite(tile)
            coverage_counts = cp.sum(valid, axis=0)
            if mode == "median":
                tile_result = cp.nanmedian(tile, axis=0)
            elif mode == "mean":
                summed = cp.sum(cp.where(valid, tile, cp.float32(0.0)), axis=0)
                tile_result = cp.where(coverage_counts > 0, summed / cp.maximum(coverage_counts, 1), cp.float32(0.0))
            elif mode == "high_rejection":
                median = cp.nanmedian(tile, axis=0)
                median_safe = cp.nan_to_num(median, nan=0.0)
                robust_sigma = cp.nanmedian(cp.abs(tile - median), axis=0) * cp.float32(1.4826)
                robust_sigma = cp.nan_to_num(robust_sigma, nan=0.0) + cp.float32(1e-6)
                keep = valid & (tile <= median + cp.float32(max(0.5, float(sigma))) * robust_sigma)
                counts = cp.sum(keep, axis=0)
                summed = cp.sum(cp.where(keep, tile, cp.float32(0.0)), axis=0)
                tile_result = cp.where(counts > 0, summed / cp.maximum(counts, 1), median_safe)
            else:
                median = cp.nanmedian(tile, axis=0)
                median_safe = cp.nan_to_num(median, nan=0.0)
                std = cp.nanstd(tile, axis=0) + cp.float32(1e-6)
                keep = valid & (cp.abs(tile - median) <= cp.float32(float(sigma)) * std)
                counts = cp.sum(keep, axis=0)
                summed = cp.sum(cp.where(keep, tile, cp.float32(0.0)), axis=0)
                tile_result = cp.where(counts > 0, summed / cp.maximum(counts, 1), median_safe)

            if require_majority_coverage:
                min_count = minimum_valid_stack_count(n, True)
                tile_result = cp.where(coverage_counts >= min_count, tile_result, cp.float32(0.0))
            tile_result = cp.clip(cp.nan_to_num(tile_result, nan=0.0), 0, 1).astype(cp.float32)
            result_cpu[y0:y1, ...] = cp.asnumpy(tile_result)
            del host_tile, tile, tile_result
            if progress_callback:
                tiles = max(1, int(math.ceil(h / rows_per_tile)))
                progress_callback(min(98, 80 + int(tile_idx / tiles * 18)), f"Skladam na GPU po blocich VRAM ({tile_idx}/{tiles})...")
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass
    log_debug(f"GPU sequence stack finished in {time.perf_counter() - started_at:.3f} s")
    return result_cpu.astype(np.float32)


def stack_frames_mps_tiled_from_sequence(frames: List[np.ndarray], mode: str, sigma: float, progress_callback=None, require_majority_coverage: bool = False) -> np.ndarray:
    """Upload row tiles directly from aligned frames to Apple Metal/MPS."""
    if torch is None:
        raise RuntimeError("PyTorch neni nainstalovany.")
    if not frames:
        raise ValueError("No frames to stack.")

    first = np.asarray(frames[0], dtype=np.float32)
    n = len(frames)
    h, w = first.shape[:2]
    channels = int(np.prod(first.shape[2:])) if first.ndim > 2 else 1
    bytes_per_row = max(1, n * w * channels * np.dtype(np.float32).itemsize)
    temp_factor = 6.0 if mode in {"sigma", "high_rejection"} else 4.0 if mode == "median" else 2.0
    target_bytes = 384 * 1024 * 1024
    rows_per_tile = max(8, int(target_bytes / max(1, bytes_per_row * temp_factor)))
    rows_per_tile = min(h, rows_per_tile)
    result_cpu = np.empty(first.shape, dtype=np.float32)
    device = torch.device("mps")

    started_at = time.perf_counter()
    log_debug(f"MPS sequence tiles: mode={mode}, frames={n}, shape={first.shape}, rows_per_tile={rows_per_tile}")
    with torch.no_grad():
        for tile_idx, y0 in enumerate(range(0, h, rows_per_tile), start=1):
            y1 = min(h, y0 + rows_per_tile)
            host_tile = np.stack([np.asarray(frame[y0:y1, ...], dtype=np.float32) for frame in frames], axis=0)
            tile = torch.from_numpy(host_tile).to(device)
            tile_result = stack_frames_torch_tensor(tile, mode, sigma, require_majority_coverage=require_majority_coverage)
            tile_result = torch.clamp(tile_result, 0, 1).to(dtype=torch.float32)
            result_cpu[y0:y1, ...] = tile_result.cpu().numpy()
            del host_tile, tile, tile_result
            if progress_callback:
                tiles = max(1, int(math.ceil(h / rows_per_tile)))
                progress_callback(min(98, 80 + int(tile_idx / tiles * 18)), f"Skladam na GPU po blocich sdilene pameti ({tile_idx}/{tiles})...")
    try:
        torch.mps.empty_cache()
    except Exception:
        pass
    log_debug(f"MPS sequence stack finished in {time.perf_counter() - started_at:.3f} s")
    return result_cpu.astype(np.float32)


def stack_aligned_frames(aligned: List[np.ndarray], settings: StackSettings, progress_callback=None) -> np.ndarray:
    if not aligned:
        raise ValueError("No aligned frames to stack.")

    first_shape = tuple(np.asarray(aligned[0]).shape)
    use_tiled_cpu, required_bytes, available_bytes = should_stack_tiled_on_cpu(first_shape, len(aligned), settings.stack_mode)
    mosaic_mode = bool(getattr(settings, "mosaic_mode", False))
    use_gpu = bool(getattr(settings, "use_gpu", False))
    require_majority_coverage = not mosaic_mode

    if not use_gpu:
        if mosaic_mode:
            if progress_callback:
                progress_callback(80, "Mozaika: skládám paralelně na CPU po blocích RAM...")
            return stack_frames_cpu_tiled_from_sequence(
                aligned,
                settings.stack_mode,
                settings.sigma,
                progress_callback,
                force_tiled=True,
                require_majority_coverage=False,
            )
        if settings.stack_mode == "mean":
            if progress_callback:
                progress_callback(80, "Skladam prumer na CPU po blocich RAM...")
            return stack_frames_cpu_tiled_from_sequence(
                aligned,
                settings.stack_mode,
                settings.sigma,
                progress_callback,
                require_majority_coverage=require_majority_coverage,
            )

        # Robust rejection modes need access to all frames per pixel. Process
        # row tiles directly from the aligned list so we do not spend a long,
        # single-threaded pause building one huge np.stack array first.
        if use_tiled_cpu or len(aligned) > 8 or settings.stack_mode in {"median", "sigma", "high_rejection"}:
            if progress_callback:
                progress_callback(80, "Skladam na CPU po blocich RAM...")
            return stack_frames_cpu_tiled_from_sequence(
                aligned,
                settings.stack_mode,
                settings.sigma,
                progress_callback,
                require_majority_coverage=require_majority_coverage,
            )

        try:
            if progress_callback:
                progress_callback(78, "Pripravuji stack v RAM...")
            arr = np.stack(aligned, axis=0).astype(np.float32, copy=False)
        except MemoryError:
            if progress_callback:
                progress_callback(80, "RAM ochrana: nedostatek pameti pro cely stack, skladam po castech...")
            return stack_frames_cpu_tiled_from_sequence(
                aligned,
                settings.stack_mode,
                settings.sigma,
                progress_callback,
                require_majority_coverage=require_majority_coverage,
            )
        if progress_callback:
            progress_callback(80, "Skladam snimky...")
        return stack_frames_cpu(arr, settings.stack_mode, settings.sigma, require_majority_coverage=require_majority_coverage)

    gpu_failed_message = None
    gpu_unavailable_detail = ""
    if use_gpu and gpu_available():
        if progress_callback:
            progress_callback(80, "Posilam stack po blocich primo do GPU (CUDA/CuPy)...")
        try:
            return stack_frames_gpu_tiled_from_sequence(
                aligned,
                settings.stack_mode,
                settings.sigma,
                progress_callback,
                require_majority_coverage=require_majority_coverage,
            )
        except Exception as exc:
            gpu_failed_message = str(exc).splitlines()[0]
            log_debug(f"GPU stack failed:\n{traceback.format_exc()}")
            if progress_callback:
                progress_callback(80, f"GPU vypocet selhal ({gpu_failed_message}); pokracuji na CPU...")
    elif use_gpu and cp is None:
        gpu_unavailable_detail = f"CuPy nelze nacist ({CUPY_IMPORT_ERROR})" if CUPY_IMPORT_ERROR else "CuPy nelze nacist"
    elif use_gpu:
        detail = f" ({GPU_AVAILABLE_ERROR})" if GPU_AVAILABLE_ERROR else f" ({GPU_AVAILABLE_DETAIL})" if GPU_AVAILABLE_DETAIL else ""
        gpu_unavailable_detail = f"CUDA/CuPy neni dostupne{detail}"

    if use_gpu and mps_available():
        if progress_callback:
            progress_callback(80, "Posilam stack po blocich primo do GPU (Apple Metal/MPS)...")
        try:
            return stack_frames_mps_tiled_from_sequence(
                aligned,
                settings.stack_mode,
                settings.sigma,
                progress_callback,
                require_majority_coverage=require_majority_coverage,
            )
        except Exception as exc:
            gpu_failed_message = str(exc).splitlines()[0]
            log_debug(f"MPS stack failed:\n{traceback.format_exc()}")
            if progress_callback:
                progress_callback(80, f"GPU vypocet selhal ({gpu_failed_message}); pokracuji na CPU...")
    elif use_gpu and torch is None:
        detail = f" ({TORCH_IMPORT_ERROR})" if TORCH_IMPORT_ERROR else ""
        gpu_unavailable_detail = (gpu_unavailable_detail + "; " if gpu_unavailable_detail else "") + f"PyTorch/MPS nelze nacist{detail}"
    elif use_gpu:
        detail = f" ({MPS_AVAILABLE_ERROR})" if MPS_AVAILABLE_ERROR else f" ({MPS_AVAILABLE_DETAIL})" if MPS_AVAILABLE_DETAIL else ""
        gpu_unavailable_detail = (gpu_unavailable_detail + "; " if gpu_unavailable_detail else "") + f"Apple Metal/MPS neni dostupne{detail}"

    if progress_callback:
        if gpu_failed_message:
            progress_callback(80, f"GPU vypocet selhal ({gpu_failed_message}); skladam na CPU...")
        elif use_gpu:
            detail = f" ({gpu_unavailable_detail})" if gpu_unavailable_detail else ""
            progress_callback(80, f"GPU neni dostupne{detail}; Python: {sys.executable}; skladam na CPU...")
        else:
            progress_callback(80, "Skladam snimky...")
    if progress_callback:
        progress_callback(80, "CPU fallback skladam po castech...")
    return stack_frames_cpu_tiled_from_sequence(
        aligned,
        settings.stack_mode,
        settings.sigma,
        progress_callback,
        force_tiled=mosaic_mode,
        require_majority_coverage=require_majority_coverage,
    )


def frame_quality_preview_gray(gray: np.ndarray, center_fraction: float = 0.72, max_edge: int = 900) -> np.ndarray:
    """Return a smaller central preview used only for fast quality/reference scoring."""
    gray = np.asarray(gray, dtype=np.float32)
    if gray.ndim != 2 or gray.size == 0:
        return gray
    h, w = gray.shape[:2]
    frac = max(0.25, min(1.0, float(center_fraction)))
    crop_h = max(32, int(h * frac))
    crop_w = max(32, int(w * frac))
    y0 = max(0, (h - crop_h) // 2)
    x0 = max(0, (w - crop_w) // 2)
    preview = gray[y0:y0 + crop_h, x0:x0 + crop_w]
    ph, pw = preview.shape[:2]
    largest = max(ph, pw)
    if largest > max_edge:
        scale = float(max_edge) / float(largest)
        preview = cv2.resize(
            preview,
            (max(1, int(round(pw * scale))), max(1, int(round(ph * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    return np.asarray(preview, dtype=np.float32)


def measure_star_shape_metrics(gray: np.ndarray, stars: np.ndarray, radius: int = 5) -> Dict[str, float]:
    """Measure median stellar roundness and FWHM from small local profiles."""
    gray = np.nan_to_num(np.asarray(gray, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    h, w = gray.shape[:2]
    roundness_values: List[float] = []
    fwhm_values: List[float] = []

    for x, y in np.asarray(stars, dtype=np.float32)[:120]:
        cx, cy = int(round(float(x))), int(round(float(y)))
        if cx - radius < 0 or cy - radius < 0 or cx + radius >= w or cy + radius >= h:
            continue
        patch = gray[cy - radius:cy + radius + 1, cx - radius:cx + radius + 1].astype(np.float32)
        background = float(np.percentile(patch, 25))
        weights = np.clip(patch - background, 0.0, None)
        peak = float(np.max(weights))
        total = float(np.sum(weights))
        if peak <= 1e-6 or total <= peak * 1.8:
            continue

        yy, xx = np.indices(weights.shape, dtype=np.float32)
        mx = float(np.sum(weights * xx) / total)
        my = float(np.sum(weights * yy) / total)
        dx = xx - mx
        dy = yy - my
        cov_xx = float(np.sum(weights * dx * dx) / total)
        cov_yy = float(np.sum(weights * dy * dy) / total)
        cov_xy = float(np.sum(weights * dx * dy) / total)
        trace = cov_xx + cov_yy
        determinant = max(0.0, cov_xx * cov_yy - cov_xy * cov_xy)
        disc = max(0.0, trace * trace - 4.0 * determinant)
        major_var = max(1e-6, 0.5 * (trace + math.sqrt(disc)))
        minor_var = max(1e-6, 0.5 * (trace - math.sqrt(disc)))
        major_sigma = math.sqrt(major_var)
        minor_sigma = math.sqrt(minor_var)
        fwhm = 2.355 * math.sqrt(major_sigma * minor_sigma)
        roundness = min(1.0, minor_sigma / max(1e-6, major_sigma))

        if 0.55 <= fwhm <= 12.0:
            roundness_values.append(float(roundness))
            fwhm_values.append(float(fwhm))

    if not fwhm_values:
        return {"roundness": 0.45, "fwhm": 4.0, "shape_star_count": 0.0}
    return {
        "roundness": float(np.median(roundness_values)),
        "fwhm": float(np.median(fwhm_values)),
        "shape_star_count": float(len(fwhm_values)),
    }


def frame_quality_metrics_from_gray(gray: np.ndarray) -> Dict[str, float]:
    """Reference score: image sharpness plus measured stellar size and roundness.

    Vyšší číslo = vhodnější snímek pro referenci a stack.
    """
    gray = frame_quality_preview_gray(gray)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    stars = detect_stars(gray, max_stars=180, threshold_percentile=99.4)
    star_bonus = min(len(stars), 180) / 180.0
    star_shape = measure_star_shape_metrics(gray, stars)
    roundness = float(star_shape["roundness"])
    fwhm = float(star_shape["fwhm"])
    measured_stars = float(star_shape["shape_star_count"])

    # Laplacian keeps sensitivity to focus, while the stellar profile terms
    # reject references with sharp noise or visibly elongated stars.
    roundness_factor = 0.35 + 0.65 * roundness
    size_factor = 1.0 / max(1.0, fwhm * fwhm)
    measured_bonus = min(measured_stars, 80.0) / 80.0
    score = lap_var * (1.0 + 0.20 * star_bonus) * roundness_factor * size_factor * (0.80 + 0.20 * measured_bonus)
    return {
        "score": float(score),
        "sharpness": float(lap_var),
        "star_count": float(len(stars)),
        "roundness": roundness,
        "fwhm": fwhm,
        "shape_star_count": measured_stars,
    }


def detect_satellite_trail_from_gray(gray: np.ndarray) -> Dict[str, float]:
    """Fast heuristic for long straight bright trails in a frame-quality preview."""
    try:
        preview = frame_quality_preview_gray(gray, center_fraction=1.0, max_edge=1100)
        preview = np.asarray(preview, dtype=np.float32)
        if preview.ndim != 2 or min(preview.shape[:2]) < 80:
            return {"satellite_trail": 0.0, "trail_score": 0.0, "trail_count": 0.0}

        finite = np.isfinite(preview)
        if not np.any(finite):
            return {"satellite_trail": 0.0, "trail_score": 0.0, "trail_count": 0.0}
        vals = preview[finite]
        lo = float(np.percentile(vals, 5))
        hi = float(np.percentile(vals, 99.8))
        if hi <= lo:
            return {"satellite_trail": 0.0, "trail_score": 0.0, "trail_count": 0.0}
        norm = np.clip((preview - lo) / (hi - lo), 0.0, 1.0)

        h, w = norm.shape[:2]
        blur_size = min(61, max(21, int(round(min(h, w) / 28.0)) | 1))
        background = cv2.medianBlur((norm * 255).astype(np.uint8), blur_size).astype(np.float32) / 255.0
        residual = np.clip(norm - background, 0.0, 1.0)
        res_vals = residual[np.isfinite(residual)]
        if res_vals.size == 0:
            return {"satellite_trail": 0.0, "trail_score": 0.0, "trail_count": 0.0}
        threshold = max(float(np.percentile(res_vals, 99.8)), float(res_vals.mean() + 4.2 * res_vals.std()), 0.085)
        mask = (residual > threshold).astype(np.uint8) * 255
        if int(mask.sum()) <= 0:
            return {"satellite_trail": 0.0, "trail_score": 0.0, "trail_count": 0.0}

        candidate_threshold = max(float(np.percentile(res_vals, 99.35)), float(res_vals.mean() + 2.9 * res_vals.std()), 0.048)
        candidate_mask = (residual > candidate_threshold).astype(np.uint8) * 255
        candidate_edges = cv2.Canny(candidate_mask, 20, 80)
        hough_threshold = max(58, int(round(min(h, w) * 0.10)))
        hough_lines = cv2.HoughLines(candidate_edges, 1, np.pi / 180.0, threshold=hough_threshold)
        if hough_lines is not None:
            line_thickness = max(3, int(round(min(h, w) / 260.0)))
            min_support = max(150, int(round(min(h, w) * 0.34)))
            best_score = 0.0
            best_support = 0
            for line in hough_lines[:80, 0, :]:
                rho, theta = float(line[0]), float(line[1])
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(round(x0 + 2000 * (-b)))
                y1 = int(round(y0 + 2000 * a))
                x2 = int(round(x0 - 2000 * (-b)))
                y2 = int(round(y0 - 2000 * a))
                line_mask = np.zeros_like(candidate_mask)
                cv2.line(line_mask, (x1, y1), (x2, y2), 255, line_thickness)
                line_pixels = int(np.count_nonzero(line_mask))
                if line_pixels <= 0:
                    continue
                support = int(np.count_nonzero(cv2.bitwise_and(candidate_mask, line_mask)))
                support_ratio = float(support) / float(line_pixels)
                mean_residual = float(cv2.mean(residual, mask=line_mask)[0])
                line_score = support_ratio * max(0.0, mean_residual / max(1e-6, candidate_threshold))
                best_score = max(best_score, line_score)
                best_support = max(best_support, support)
                strong_line = support >= min_support and support_ratio >= 0.13 and mean_residual >= max(0.085, candidate_threshold * 0.36)
                faint_line = (
                    support >= max(240, int(round(min(h, w) * 0.28)))
                    and support_ratio >= 0.055
                    and mean_residual >= max(0.032, candidate_threshold * 0.22)
                )
                if strong_line or faint_line:
                    return {
                        "satellite_trail": 1.0,
                        "trail_score": float(line_score),
                        "trail_count": 1.0,
                    }

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        residual_u8 = np.clip(residual * 255.0, 0, 255).astype(np.uint8)
        edges = cv2.Canny(residual_u8, 28, 90)
        edges = cv2.bitwise_or(edges, cv2.Canny(mask, 28, 90))
        line_source = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
        edges = cv2.bitwise_or(edges, line_source)
        min_len = max(80, int(round(min(h, w) * 0.35)))
        max_gap = 2
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=24, minLineLength=min_len, maxLineGap=max_gap)
        if lines is None:
            return {"satellite_trail": 0.0, "trail_score": 0.0, "trail_count": 0.0}

        diagonal = float(math.hypot(w, h))
        good_lengths: List[float] = []
        for line in lines[:, 0, :]:
            x1, y1, x2, y2 = [int(v) for v in line]
            length = float(math.hypot(x2 - x1, y2 - y1))
            if length < min_len:
                continue
            line_mask = np.zeros_like(mask)
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, max(2, int(round(min(h, w) / 320.0))))
            support = float(np.count_nonzero(cv2.bitwise_and(mask, line_mask)))
            support_ratio = support / max(1.0, length)
            mean_residual = float(cv2.mean(residual, mask=line_mask)[0])
            if support_ratio >= 0.09 and mean_residual >= threshold * 0.65:
                good_lengths.append(length)

        if not good_lengths:
            return {"satellite_trail": 0.0, "trail_score": 0.0, "trail_count": 0.0}
        longest = max(good_lengths)
        score = float(longest / max(1.0, diagonal))
        suspicious = 1.0 if score >= 0.25 or longest / max(1.0, min(h, w)) >= 0.35 else 0.0
        return {
            "satellite_trail": suspicious,
            "trail_score": score,
            "trail_count": float(len(good_lengths)),
        }
    except Exception as exc:
        log_debug(f"Satellite trail detection failed: {exc}")
        return {"satellite_trail": 0.0, "trail_score": 0.0, "trail_count": 0.0}


def satellite_trail_mask_from_gray(gray: np.ndarray, confirmed: bool = False) -> Optional[np.ndarray]:
    """Return a full-resolution mask for a confirmed long satellite trail.

    The existing conservative trail detector decides whether a frame is
    suspicious. Only then do we locate the dominant line and create a slightly
    wider mask that also covers the trail halo. Masked pixels are excluded from
    integration as NaN; the rest of the frame remains usable.
    """
    if not confirmed:
        metrics = detect_satellite_trail_from_gray(gray)
        if float(metrics.get("satellite_trail", 0.0)) < 0.5:
            return None

    try:
        source = np.asarray(gray, dtype=np.float32)
        source_h, source_w = source.shape[:2]
        preview = frame_quality_preview_gray(source, center_fraction=1.0, max_edge=1100)
        finite = np.isfinite(preview)
        if preview.ndim != 2 or not np.any(finite):
            return None

        values = preview[finite]
        lo = float(np.percentile(values, 5.0))
        hi = float(np.percentile(values, 99.8))
        if hi <= lo + 1e-8:
            return None
        norm = np.clip((preview - lo) / (hi - lo), 0.0, 1.0)
        h, w = norm.shape[:2]

        blur_size = min(61, max(21, int(round(min(h, w) / 28.0)) | 1))
        background = cv2.medianBlur((norm * 255).astype(np.uint8), blur_size).astype(np.float32) / 255.0
        residual = np.clip(norm - background, 0.0, 1.0)
        residual_values = residual[np.isfinite(residual)]
        threshold = max(
            float(np.percentile(residual_values, 99.25)),
            float(residual_values.mean() + 2.7 * residual_values.std()),
            0.042,
        )
        candidate = (residual > threshold).astype(np.uint8) * 255
        candidate = cv2.morphologyEx(
            candidate,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1,
        )
        edges = cv2.Canny(candidate, 18, 72)

        min_len = max(70, int(round(min(h, w) * 0.25)))
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 360.0,
            threshold=max(20, int(round(min(h, w) * 0.035))),
            minLineLength=min_len,
            maxLineGap=3,
        )

        best_line = None
        best_score = -1.0
        if lines is not None:
            support_thickness = max(3, int(round(min(h, w) / 260.0)))
            for x1, y1, x2, y2 in lines[:, 0, :]:
                length = float(math.hypot(int(x2) - int(x1), int(y2) - int(y1)))
                if length < min_len:
                    continue
                probe = np.zeros((h, w), dtype=np.uint8)
                cv2.line(probe, (int(x1), int(y1)), (int(x2), int(y2)), 255, support_thickness)
                support = float(np.count_nonzero(cv2.bitwise_and(candidate, probe)))
                mean_residual = float(cv2.mean(residual, mask=probe)[0])
                score = length * (1.0 + support / max(1.0, length)) * (1.0 + mean_residual)
                if score > best_score:
                    best_score = score
                    best_line = (int(x1), int(y1), int(x2), int(y2))

        if best_line is None:
            standard_lines = cv2.HoughLines(
                edges,
                1,
                np.pi / 360.0,
                threshold=max(48, int(round(min(h, w) * 0.085))),
            )
            if standard_lines is None:
                return None
            rho, theta = [float(v) for v in standard_lines[0, 0, :]]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            span = int(math.ceil(math.hypot(w, h) * 1.5))
            best_line = (
                int(round(x0 + span * (-b))),
                int(round(y0 + span * a)),
                int(round(x0 - span * (-b))),
                int(round(y0 - span * a)),
            )

        # HoughLinesP usually returns only the brightest continuous segment of
        # a trail. Extend its direction beyond both endpoints so the exclusion
        # mask covers the complete straight trail across the frame.
        x1, y1, x2, y2 = best_line
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        line_length = math.hypot(dx, dy)
        if line_length > 1e-6:
            ux = dx / line_length
            uy = dy / line_length
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            span = float(math.hypot(w, h) * 1.5)
            best_line = (
                int(round(cx - ux * span)),
                int(round(cy - uy * span)),
                int(round(cx + ux * span)),
                int(round(cy + uy * span)),
            )

        mask_preview = np.zeros((h, w), dtype=np.uint8)
        halo_width = max(7, int(round(min(h, w) / 115.0)))
        cv2.line(
            mask_preview,
            (best_line[0], best_line[1]),
            (best_line[2], best_line[3]),
            255,
            halo_width,
            cv2.LINE_AA,
        )
        mask_preview = cv2.dilate(
            mask_preview,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )
        full_mask = cv2.resize(mask_preview, (source_w, source_h), interpolation=cv2.INTER_NEAREST)
        return full_mask > 0
    except Exception as exc:
        log_debug(f"Satellite trail mask creation failed: {exc}")
        return None


def apply_warped_exclusion_mask(
    img: np.ndarray,
    source_mask: Optional[np.ndarray],
    matrix: np.ndarray,
    output_shape: Tuple[int, int],
) -> np.ndarray:
    """Warp a source-space exclusion mask and mark matching output pixels NaN."""
    if source_mask is None or not np.any(source_mask):
        return np.asarray(img, dtype=np.float32)
    warped_mask = warp_to_reference(
        np.asarray(source_mask, dtype=np.float32),
        matrix,
        output_shape,
        border_value=0.0,
    )
    excluded = warped_mask > 0.05
    out = np.asarray(img, dtype=np.float32).copy()
    if out.ndim == 3:
        out[excluded, :] = np.nan
    else:
        out[excluded] = np.nan
    return out


def frame_quality_score_from_gray(gray: np.ndarray) -> float:
    return float(frame_quality_metrics_from_gray(gray).get("score", -1.0))


def frame_quality_metrics(path: Path, detect_satellite_trails: bool = False) -> Dict[str, float]:
    cached = load_frame_quality_cache(path)
    if cached is not None and (not detect_satellite_trails or cached.get("satellite_trail_checked", 0.0) >= 0.5):
        cached = dict(cached)
        cached["cached"] = 1.0
        return cached
    try:
        img = load_image_as_float(path)
        gray = to_gray_float(img)
        metrics = dict(cached) if cached is not None else frame_quality_metrics_from_gray(gray)
        if detect_satellite_trails:
            metrics.update(detect_satellite_trail_from_gray(gray))
            metrics["satellite_trail_checked"] = 1.0
        save_frame_quality_cache(path, metrics)
        return metrics
    except Exception:
        return {"score": -1.0, "sharpness": 0.0, "star_count": 0.0, "roundness": 0.0, "fwhm": 0.0, "shape_star_count": 0.0}


def frame_quality_score(path: Path) -> float:
    return float(frame_quality_metrics(path).get("score", -1.0))


def evaluate_frame_quality_batch(paths: List[Path], progress_callback=None, detect_satellite_trails: bool = False) -> Tuple[Dict[Path, float], Dict[Path, Dict[str, float]]]:
    scores: Dict[Path, float] = {}
    metrics: Dict[Path, Dict[str, float]] = {}
    total = len(paths)
    if total <= 0:
        return scores, metrics

    workers = min(total, max(1, int(round((os.cpu_count() or 1) * 0.75))))
    if total < 8 or workers <= 1:
        for idx, path in enumerate(paths):
            path_metrics = frame_quality_metrics(path, detect_satellite_trails=detect_satellite_trails)
            metrics[path] = path_metrics
            scores[path] = float(path_metrics.get("score", -1.0))
            if progress_callback:
                label = "Kvalita z cache" if path_metrics.get("cached") else "Hodnotím kvalitu"
                progress_callback(int((idx + 1) / total * 10), f"{label} ({idx + 1}/{total}): {path.name}")
        return scores, metrics

    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_path = {executor.submit(frame_quality_metrics, path, detect_satellite_trails): path for path in paths}
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                path_metrics = future.result()
            except Exception:
                path_metrics = {"score": -1.0, "sharpness": 0.0, "star_count": 0.0, "roundness": 0.0, "fwhm": 0.0, "shape_star_count": 0.0}
            metrics[path] = path_metrics
            scores[path] = float(path_metrics.get("score", -1.0))
            completed += 1
            if progress_callback:
                label = "Kvalita z cache" if path_metrics.get("cached") else "Hodnotím kvalitu paralelně"
                progress_callback(int(completed / total * 10), f"{label} ({completed}/{total}): {path.name}")
    return scores, metrics


def file_cache_signature(path: Optional[Path]) -> Tuple[str, int, int]:
    if not path:
        return ("", 0, 0)
    try:
        p = Path(path)
        stat = p.stat()
        return (str(p.resolve()), int(stat.st_mtime_ns), int(stat.st_size))
    except Exception:
        return (str(path), 0, 0)


def calibration_folder_input_paths(folder: Path, settings: Optional[StackSettings] = None) -> List[Path]:
    extensions = stackable_extensions(settings)
    return sorted(
        p for p in Path(folder).iterdir()
        if p.is_file() and p.suffix.lower() in extensions and not p.name.lower().startswith("master")
    )


def calibration_master_paths(folder: Path, kind: str) -> Tuple[Path, Path]:
    safe_kind = "".join(ch for ch in str(kind).title() if ch.isalnum()) or "Calibration"
    base = Path(folder) / f"Master{safe_kind}_AS"
    return base.with_suffix(".fit"), base.with_suffix(".json")


def calibration_master_signature(
    folder: Path,
    kind: str,
    settings: Optional[StackSettings],
    paths: List[Path],
) -> Dict[str, Any]:
    return {
        "version": CALIBRATION_ALGORITHM_VERSION,
        "kind": str(kind),
        "raw_only": bool(settings is not None and (getattr(settings, "raw_only", False) or getattr(settings, "fit_only", False))),
        "bayer_pattern": getattr(settings, "bayer_pattern", "auto") if settings is not None else "auto",
        "files": [list(file_cache_signature(path)) for path in paths],
    }


def load_calibration_master_fit(path: Path, preserve_mosaic: bool = True) -> np.ndarray:
    """Load cached calibration masters without any FITS auto-normalization.

    MasterBias/MasterFlat/MasterDark are calibration data, not display images.
    Their absolute 0..1 scale must be preserved; otherwise a cached flat can be
    re-scaled and break the light-frame calibration.
    """
    if fits is None:
        raise RuntimeError("Pro FITS podporu nainstaluj: pip install astropy")

    with open_fits_safely(path, memmap=False) as hdul:
        data = None
        for hdu in hdul:
            if getattr(hdu, "data", None) is not None:
                data = hdu.data
                break
    if data is None:
        raise ValueError(f"FITS soubor neobsahuje obrazová data: {path.name}")

    arr = np.asarray(data)
    if arr.ndim == 2:
        if not preserve_mosaic:
            arr = np.repeat(arr[..., None], 3, axis=2)
    elif arr.ndim == 3:
        if arr.shape[0] in (3, 4) and arr.shape[1] > 16 and arr.shape[2] > 16:
            arr = np.moveaxis(arr[:3], 0, -1)
        elif arr.shape[-1] in (3, 4):
            arr = arr[..., :3]
        else:
            arr = np.repeat(arr[0][..., None], 3, axis=2)
    else:
        raise ValueError(f"Nepodporovaný FITS rozměr {arr.ndim}D v souboru {path.name}")

    if np.issubdtype(arr.dtype, np.integer):
        arr = normalize_fits_linear_to_float(arr)
    else:
        arr = np.asarray(arr, dtype=np.float32)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            arr = np.zeros_like(arr, dtype=np.float32)
        else:
            lo = float(np.min(finite))
            hi = float(np.max(finite))
            if lo >= -1e-6 and hi <= 1.000001:
                arr = np.clip(np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0), 0, 1).astype(np.float32)
            else:
                arr = normalize_fits_linear_to_float(arr)
    return np.ascontiguousarray(arr.astype(np.float32))


def load_calibration_master_cache(
    folder: Path,
    kind: str,
    settings: Optional[StackSettings],
    paths: List[Path],
    progress_callback=None,
) -> Optional[np.ndarray]:
    master_path, meta_path = calibration_master_paths(folder, kind)
    if not master_path.exists() or not meta_path.exists():
        return None
    try:
        expected = calibration_master_signature(folder, kind, settings, paths)
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        if data != expected:
            return None
        if progress_callback:
            progress_callback(0, f"Nacitam Master{kind} z cache: {master_path.name}")
        return load_calibration_master_fit(master_path)
    except Exception as exc:
        log_debug(f"Calibration master cache read failed for {kind}: {exc}")
        return None


def save_calibration_master_cache(
    folder: Path,
    kind: str,
    settings: Optional[StackSettings],
    paths: List[Path],
    master: np.ndarray,
) -> None:
    if fits is None:
        return
    master_path, meta_path = calibration_master_paths(folder, kind)
    try:
        stack_info = {
            "align_mode": "calibration",
            "stack_mode": "mean",
            "num_images": len(paths),
            "bayer_pattern": getattr(settings, "bayer_pattern", "auto") if settings is not None else "auto",
        }
        save_stack_fits(master_path, master, source_header=None, stack_info=stack_info)
        meta_path.write_text(json.dumps(calibration_master_signature(folder, kind, settings, paths), indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        log_debug(f"Calibration master cache write failed for {kind}: {exc}")


def frame_quality_cache_key(path: Path) -> Tuple[Any, ...]:
    return (
        QUALITY_CACHE_VERSION,
        file_cache_signature(path),
        get_bayer_pattern_override(),
        bayer_pattern_for_fits_path(path) if Path(path).suffix.lower() in FITS_EXTENSIONS else "",
    )


def frame_quality_cache_path(path: Path) -> Path:
    key = repr(frame_quality_cache_key(path)).encode("utf-8", errors="replace")
    digest = hashlib.sha1(key).hexdigest()
    return Path(path).parent / "astro_stacker_cache" / f"quality_{digest}.json"


def load_frame_quality_cache(path: Path) -> Optional[Dict[str, float]]:
    cache_path = frame_quality_cache_path(path)
    if not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        if data.get("version") != QUALITY_CACHE_VERSION:
            return None
        if data.get("key") != repr(frame_quality_cache_key(path)):
            return None
        metrics = {
            "score": float(data.get("score", -1.0)),
            "sharpness": float(data.get("sharpness", 0.0)),
            "star_count": float(data.get("star_count", 0.0)),
            "roundness": float(data.get("roundness", 0.0)),
            "fwhm": float(data.get("fwhm", 0.0)),
            "shape_star_count": float(data.get("shape_star_count", 0.0)),
        }
        if "satellite_trail" in data:
            metrics.update({
                "satellite_trail": float(data.get("satellite_trail", 0.0)),
                "trail_score": float(data.get("trail_score", 0.0)),
                "trail_count": float(data.get("trail_count", 0.0)),
                "satellite_trail_checked": float(data.get("satellite_trail_checked", 0.0)),
            })
        return metrics
    except Exception as exc:
        log_debug(f"Frame quality cache read failed for {path}: {exc}")
        return None


def save_frame_quality_cache(path: Path, metrics: Dict[str, float]) -> None:
    try:
        cache_path = frame_quality_cache_path(path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": QUALITY_CACHE_VERSION,
            "key": repr(frame_quality_cache_key(path)),
            "score": float(metrics.get("score", -1.0)),
            "sharpness": float(metrics.get("sharpness", 0.0)),
            "star_count": float(metrics.get("star_count", 0.0)),
            "roundness": float(metrics.get("roundness", 0.0)),
            "fwhm": float(metrics.get("fwhm", 0.0)),
            "shape_star_count": float(metrics.get("shape_star_count", 0.0)),
        }
        if metrics.get("satellite_trail_checked", 0.0) >= 0.5:
            data.update({
                "satellite_trail": float(metrics.get("satellite_trail", 0.0)),
                "trail_score": float(metrics.get("trail_score", 0.0)),
                "trail_count": float(metrics.get("trail_count", 0.0)),
                "satellite_trail_checked": 1.0,
            })
        temp_path = cache_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
        try:
            temp_path.replace(cache_path)
        except Exception:
            cache_path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass
    except Exception as exc:
        log_debug(f"Frame quality cache write failed for {path}: {exc}")


def folder_cache_signature(path: Optional[Path], settings: Optional[StackSettings] = None) -> Tuple[Any, ...]:
    if not path:
        return ("",)
    try:
        p = Path(path)
        if not p.is_dir():
            return file_cache_signature(p)
        files = calibration_folder_input_paths(p, settings)
        return ("folder", str(p.resolve()), tuple(file_cache_signature(child) for child in files))
    except Exception:
        return (str(path),)


def settings_calibration_cache_signature(settings: StackSettings) -> Tuple[Any, ...]:
    cache_key = (
        settings.flat_frame_path or "",
        settings.bias_frame_path or "",
        settings.dark_frame_path or "",
        settings.source_folder or "",
        bool(getattr(settings, "raw_only", False) or getattr(settings, "fit_only", False)),
    )
    cached = CALIBRATION_SIGNATURE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    source = Path(settings.source_folder) if settings.source_folder else None
    auto_bias = find_calibration_subfolder(source, ("bias", "biases", "offset", "offsets")) if source else None
    auto_flat = find_calibration_subfolder(source, ("flat", "flats")) if source else None
    auto_dark = find_calibration_subfolder(source, ("dark", "darks")) if source else None
    signature = (
        CALIBRATION_ALGORITHM_VERSION,
        folder_cache_signature(Path(settings.flat_frame_path) if settings.flat_frame_path else None, settings),
        folder_cache_signature(Path(settings.bias_frame_path) if settings.bias_frame_path else None, settings),
        folder_cache_signature(Path(settings.dark_frame_path) if settings.dark_frame_path else None, settings),
        folder_cache_signature(auto_flat, settings),
        folder_cache_signature(auto_bias, settings),
        folder_cache_signature(auto_dark, settings),
    )
    CALIBRATION_SIGNATURE_CACHE[cache_key] = signature
    return signature


def aligned_cache_dir(settings: StackSettings, path: Path) -> Path:
    source = getattr(settings, "source_folder", None)
    base = Path(source) if source else Path(path).parent
    return base / "astro_stacker_cache"


def alignment_cache_key(path: Path, reference_path: Path, settings: StackSettings, predicted_xy: Optional[Tuple[float, float]] = None) -> Tuple[Any, ...]:
    predicted = None
    if predicted_xy is not None:
        predicted = (round(float(predicted_xy[0]), 3), round(float(predicted_xy[1]), 3))
    return (
        ALIGNMENT_CACHE_VERSION,
        file_cache_signature(path),
        file_cache_signature(reference_path),
        getattr(settings, "align_mode", ""),
        float(getattr(settings, "downscale_for_alignment", 0.5)),
        int(getattr(settings, "max_star_shift", 0)),
        int(getattr(settings, "max_comet_shift", 0)),
        int(getattr(settings, "star_border_margin", 0)),
        bool(getattr(settings, "strict_star_filter", True)),
        bool(getattr(settings, "satellite_trail_filter", False)),
        bool(getattr(settings, "comet_refine", True)),
        int(getattr(settings, "comet_refine_patch", 0)),
        int(getattr(settings, "comet_refine_search", 0)),
        getattr(settings, "bayer_pattern", "auto"),
        bool(getattr(settings, "normalize_background", False)),
        predicted,
        settings_calibration_cache_signature(settings),
    )


def aligned_frame_cache_path(path: Path, reference_path: Path, settings: StackSettings, predicted_xy: Optional[Tuple[float, float]] = None) -> Path:
    key = repr(alignment_cache_key(path, reference_path, settings, predicted_xy)).encode("utf-8", errors="replace")
    digest = hashlib.sha1(key).hexdigest()
    return aligned_cache_dir(settings, path) / f"aligned_{digest}.npy"


def load_aligned_frame_cache(path: Path, reference_path: Path, settings: StackSettings, predicted_xy: Optional[Tuple[float, float]] = None) -> Optional[np.ndarray]:
    if not bool(getattr(settings, "use_aligned_cache", False)) or bool(getattr(settings, "mosaic_mode", False)):
        return None
    cache_path = aligned_frame_cache_path(path, reference_path, settings, predicted_xy)
    if not cache_path.exists():
        return None
    try:
        return np.load(cache_path, allow_pickle=False).astype(np.float32, copy=False)
    except Exception as exc:
        log_debug(f"Aligned frame cache read failed for {path}: {exc}")
        return None


def save_aligned_frame_cache(path: Path, reference_path: Path, settings: StackSettings, predicted_xy: Optional[Tuple[float, float]], img: np.ndarray) -> None:
    if not bool(getattr(settings, "use_aligned_cache", False)) or bool(getattr(settings, "mosaic_mode", False)):
        return
    cache_path = aligned_frame_cache_path(path, reference_path, settings, predicted_xy)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = cache_path.with_suffix(".tmp.npy")
        np.save(temp_path, np.asarray(img, dtype=np.float32))
        try:
            temp_path.replace(cache_path)
        except Exception:
            np.save(cache_path, np.asarray(img, dtype=np.float32))
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass
    except Exception as exc:
        log_debug(f"Aligned frame cache write failed for {path}: {exc}")


def remove_cache_dir_safely(cache_dir: Path) -> Tuple[int, int]:
    """Remove files from an AstroStacker cache directory only.

    Returns (removed_files, failed_files). The directory must be named exactly
    astro_stacker_cache so this helper cannot accidentally delete image folders.
    """
    cache_dir = Path(cache_dir)
    if cache_dir.name != "astro_stacker_cache" or not cache_dir.exists() or not cache_dir.is_dir():
        return 0, 0

    removed = 0
    failed = 0
    for item in sorted(cache_dir.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        try:
            if item.is_file() or item.is_symlink():
                item.unlink()
                removed += 1
            elif item.is_dir():
                item.rmdir()
        except Exception as exc:
            failed += 1
            log_debug(f"Cache delete failed for {item}: {exc}")
    try:
        cache_dir.rmdir()
    except Exception:
        pass
    return removed, failed


def find_astrostacker_cache_dirs(root: Path, recursive: bool = False) -> List[Path]:
    """Find AstroStacker cache directories below a user-selected root."""
    try:
        root = Path(root)
        if not root.exists():
            return []
        if recursive:
            found = [p for p in root.rglob("astro_stacker_cache") if p.is_dir() and p.name == "astro_stacker_cache"]
        else:
            candidate = root / "astro_stacker_cache"
            found = [candidate] if candidate.is_dir() else []
        return sorted(set(found), key=lambda p: str(p))
    except Exception as exc:
        log_debug(f"Cache directory scan failed for {root}: {exc}")
        return []


def looks_like_calibration_frame(path: Path) -> bool:
    text = " ".join(part.lower() for part in Path(path).parts)
    name = Path(path).stem.lower()
    tokens = ("dark", "darks", "bias", "biases", "offset", "offsets", "flat", "flats")
    return any(token in text or token in name for token in tokens)


def is_effectively_black_frame(img: np.ndarray) -> bool:
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim == 2:
        gray = arr
    else:
        gray = (0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]).astype(np.float32)
    gray = np.nan_to_num(gray, nan=0.0, posinf=0.0, neginf=0.0)
    p50 = float(np.percentile(gray, 50))
    p99 = float(np.percentile(gray, 99))
    p999 = float(np.percentile(gray, 99.9))
    dynamic = p999 - p50
    upper_signal = p99 - p50
    return p999 < 0.01 or (dynamic < 0.003 and upper_signal < 0.0015)


def prepare_stack_paths(folder: Path, settings: StackSettings, progress_callback=None) -> Tuple[List[Path], Path, Dict[Path, float]]:
    global LAST_STACK_SELECTION
    """Načte seznam snímků, volitelně vybere nejlepší referenci a vyfiltruje horší snímky."""
    if getattr(settings, "preselected_paths", ()):
        all_paths_raw = sorted([p for p in folder.iterdir() if p.suffix.lower() in stackable_extensions(settings)])
        if settings.max_images > 0:
            all_paths_raw = all_paths_raw[: settings.max_images]
        all_resolved = [str(p.resolve()) for p in all_paths_raw]
        by_resolved = {str(p.resolve()): p for p in all_paths_raw}
        selected = []
        for path_str in settings.preselected_paths:
            try:
                key = str(Path(path_str).resolve())
            except Exception:
                key = str(path_str)
            path = by_resolved.get(key)
            if path is not None:
                selected.append(path)
        if not selected:
            raise ValueError("No frames selected for stacking.")
        reference_path = selected[0]
        explicit_reference = settings.preselected_reference_path
        if not getattr(settings, "auto_reference", True) and getattr(settings, "manual_reference_path", None):
            explicit_reference = settings.manual_reference_path
        if explicit_reference:
            try:
                ref_key = str(Path(explicit_reference).resolve())
                reference_path = by_resolved.get(ref_key, reference_path)
            except Exception:
                pass
        if reference_path not in selected:
            selected.insert(0, reference_path)
        paths = [reference_path] + [p for p in selected if p != reference_path]
        used_set = {str(p.resolve()) for p in paths}
        LAST_STACK_SELECTION = {
            "folder": str(Path(folder).resolve()),
            "all_paths": all_resolved,
            "light_paths": list(LAST_STACK_SELECTION.get("light_paths", all_resolved)),
            "selected_paths": [str(p.resolve()) for p in paths],
            "used_paths": [str(p.resolve()) for p in paths],
            "excluded_paths": [p for p in all_resolved if p not in used_set],
            "quality_excluded_paths": list(LAST_STACK_SELECTION.get("quality_excluded_paths", [])),
            "alignment_rejected_paths": [],
            "manual_excluded_paths": list(getattr(settings, "manual_excluded_paths", ()) or ()),
            "reference_path": str(reference_path.resolve()),
            "scores": dict(LAST_STACK_SELECTION.get("scores", {})),
            "quality_metrics": dict(LAST_STACK_SELECTION.get("quality_metrics", {})),
            "quality_filter": bool(LAST_STACK_SELECTION.get("quality_filter", False)),
            "keep_percent": int(getattr(settings, "keep_percent", 100)),
        }
        return paths, reference_path, {}

    extensions = stackable_extensions(settings)
    paths = sorted([p for p in folder.iterdir() if p.suffix.lower() in extensions])
    if settings.max_images > 0:
        paths = paths[: settings.max_images]
    if not paths:
        if getattr(settings, "raw_only", False) or getattr(settings, "fit_only", False):
            raise ValueError("Ve složce nejsou žádné XISF, FIT/FITS ani RAW snímky. Vypni volbu Pouze RAW, pokud chceš skládat i PNG/JPG/TIFF/BMP.")
        raise ValueError("Ve složce nejsou žádné podporované obrázky. Podporované formáty zahrnují XISF, FIT/FITS, CR2/CR3/RAW, TIFF, PNG, JPG a BMP.")

    all_paths = list(paths)
    if settings.align_mode != "calibration":
        paths = [p for p in paths if not looks_like_calibration_frame(p)]
        if not paths:
            raise ValueError("Ve složce nezbyly žádné light snímky. Zkontroluj, zda nejsou soubory označené jako Dark/Bias/Flat.")
    light_paths = list(paths)

    is_calibration_stack = settings.align_mode == "calibration"
    need_scores = (
        not is_calibration_stack
        and (
            settings.auto_reference
            or settings.quality_filter
            or bool(getattr(settings, "satellite_trail_filter", False))
        )
    )
    scores: Dict[Path, float] = {}
    metrics: Dict[Path, Dict[str, float]] = {}

    if need_scores:
        scores, metrics = evaluate_frame_quality_batch(paths, progress_callback, detect_satellite_trails=bool(getattr(settings, "satellite_trail_filter", False)))
    else:
        scores = {p: 0.0 for p in paths}
        metrics = {
            p: {
                "score": 0.0,
                "sharpness": 0.0,
                "star_count": 0.0,
                "roundness": 0.0,
                "fwhm": 0.0,
                "shape_star_count": 0.0,
                "satellite_trail": 0.0,
                "trail_score": 0.0,
                "trail_count": 0.0,
            }
            for p in paths
        }

    manual_ref = Path(settings.manual_comet_reference_path) if settings.manual_comet_reference_path else None
    manual_stack_ref = Path(settings.manual_reference_path) if getattr(settings, "manual_reference_path", None) else None
    if is_calibration_stack:
        reference_path = paths[0]
    elif settings.align_mode == "comet" and settings.manual_comet_xy is not None and manual_ref in paths:
        # U kometárního dvoubodového zarovnání musí být reference přesně ten snímek,
        # ve kterém uživatel označil první polohu komety.
        reference_path = manual_ref
    elif not settings.auto_reference and manual_stack_ref is not None and manual_stack_ref in paths:
        reference_path = manual_stack_ref
    else:
        if settings.auto_reference:
            reference_candidates = paths
            if settings.align_mode == "star_affine" and len(paths) >= 20:
                start = max(0, int(len(paths) * 0.25))
                end = min(len(paths), max(start + 1, int(len(paths) * 0.75)))
                reference_candidates = paths[start:end]
            reference_path = max(reference_candidates, key=lambda p: scores.get(p, 0.0))
        else:
            reference_path = paths[0]

    if not is_calibration_stack and settings.quality_filter and len(paths) > 3:
        keep_percent = max(10, min(100, int(settings.keep_percent)))
        keep_count = max(3, int(round(len(paths) * keep_percent / 100.0)))
        ranked = sorted(paths, key=lambda p: scores.get(p, -1.0), reverse=True)
        keep_set = set(ranked[:keep_count])
        keep_set.add(reference_path)
        manual_end = Path(settings.manual_comet_end_path) if settings.manual_comet_end_path else None
        if manual_end in paths:
            keep_set.add(manual_end)
        paths = [p for p in paths if p in keep_set]

    manual_excluded = set()
    for path_str in getattr(settings, "manual_excluded_paths", ()) or ():
        try:
            manual_excluded.add(str(Path(path_str).resolve()))
        except Exception:
            manual_excluded.add(str(path_str))

    if manual_excluded:
        paths = [p for p in paths if str(p.resolve()) not in manual_excluded or p == reference_path]

    # Referenční snímek dáme na začátek, aby se nezarovnával a byl vždy ve stacku.
    paths = [reference_path] + [p for p in paths if p != reference_path]
    selected_paths = list(paths)
    used_set = {p.resolve() for p in paths}
    selected_resolved = [str(p.resolve()) for p in selected_paths]
    light_resolved = [str(p.resolve()) for p in light_paths]
    selected_set = set(selected_resolved)
    LAST_STACK_SELECTION = {
        "folder": str(Path(folder).resolve()),
        "all_paths": [str(p.resolve()) for p in all_paths],
        "light_paths": light_resolved,
        "selected_paths": selected_resolved,
        "used_paths": [str(p.resolve()) for p in paths],
        "excluded_paths": [str(p.resolve()) for p in all_paths if p.resolve() not in used_set],
        "quality_excluded_paths": [str(p.resolve()) for p in all_paths if str(p.resolve()) not in selected_set],
        "alignment_rejected_paths": [],
        "manual_excluded_paths": [str(p.resolve()) for p in all_paths if str(p.resolve()) in manual_excluded],
        "reference_path": str(reference_path.resolve()),
        "scores": {str(p.resolve()): float(scores.get(p, 0.0)) for p in all_paths},
        "quality_metrics": {
            str(p.resolve()): {
                "score": float(metrics.get(p, {}).get("score", scores.get(p, 0.0))),
                "sharpness": float(metrics.get(p, {}).get("sharpness", 0.0)),
                "star_count": int(metrics.get(p, {}).get("star_count", 0)),
                "roundness": float(metrics.get(p, {}).get("roundness", 0.0)),
                "fwhm": float(metrics.get(p, {}).get("fwhm", 0.0)),
                "shape_star_count": int(metrics.get(p, {}).get("shape_star_count", 0)),
                "satellite_trail": float(metrics.get(p, {}).get("satellite_trail", 0.0)),
                "trail_score": float(metrics.get(p, {}).get("trail_score", 0.0)),
                "trail_count": int(metrics.get(p, {}).get("trail_count", 0)),
            }
            for p in all_paths
        },
        "quality_filter": bool(not is_calibration_stack and settings.quality_filter and len(all_paths) > 3),
        "keep_percent": int(getattr(settings, "keep_percent", 100)),
    }
    return paths, reference_path, scores


def soft_comet_mask(shape: Tuple[int, int], center_xy: Tuple[float, float], radius: int, softness: int) -> np.ndarray:
    """Vytvoří měkkou kruhovou masku pro vložení komety do hvězdného stacku."""
    h, w = shape
    cx, cy = center_xy
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.sqrt((xx - float(cx)) ** 2 + (yy - float(cy)) ** 2)
    radius = max(5, int(radius))
    softness = max(1, int(softness))
    inner = max(1.0, float(radius - softness))
    mask = np.ones((h, w), dtype=np.float32)
    fade = (float(radius) - dist) / max(1.0, float(radius) - inner)
    mask = np.where(dist <= inner, 1.0, np.clip(fade, 0.0, 1.0))
    mask = np.where(dist <= float(radius), mask, 0.0)
    if softness > 1:
        k = max(3, int(softness // 2) * 2 + 1)
        mask = cv2.GaussianBlur(mask, (k, k), softness / 3.0)
    return np.clip(mask, 0, 1).astype(np.float32)


def adaptive_comet_mask(star_stack: np.ndarray, comet_stack: np.ndarray, center_xy: Tuple[float, float], radius: int, softness: int) -> np.ndarray:
    """Maska komety: kombinuje bezpečnou kruhovou masku s rozdílem comet-stack minus star-stack.

    Když je kometa slabá a rozdílová maska nevyjde spolehlivě, zůstane použitelná
    měkká kruhová maska. To je u komet za soumraku robustnější než čistá automatika.
    """
    h, w = star_stack.shape[:2]
    circular = soft_comet_mask((h, w), center_xy, radius, softness)

    star_lum = 0.2126 * star_stack[..., 0] + 0.7152 * star_stack[..., 1] + 0.0722 * star_stack[..., 2]
    comet_lum = 0.2126 * comet_stack[..., 0] + 0.7152 * comet_stack[..., 1] + 0.0722 * comet_stack[..., 2]
    diff = np.maximum(comet_lum - star_lum, 0.0) * circular

    if np.max(diff) > 1e-6 and np.count_nonzero(circular > 0.05) > 20:
        roi_vals = diff[circular > 0.05]
        hi = float(np.percentile(roi_vals, 99.5))
        lo = float(np.percentile(roi_vals, 65.0))
        if hi > lo + 1e-6:
            adaptive = np.clip((diff - lo) / (hi - lo), 0, 1)
            k = max(5, int(max(softness, 15) // 2) * 2 + 1)
            adaptive = cv2.GaussianBlur(adaptive.astype(np.float32), (k, k), max(softness, 15) / 4.0)
            # Kruhová maska garantuje, že slabé jádro nezmizí; adaptive zpřesní okraje, když to jde.
            core = soft_comet_mask((h, w), center_xy, max(8, radius // 3), max(4, softness // 3))
            return np.clip(np.maximum(core, adaptive * circular), 0, 1).astype(np.float32)

    return circular[..., None].astype(np.float32) if circular.ndim == 3 else circular.astype(np.float32)


def comet_trail_mask(shape: Tuple[int, int], start_xy: Tuple[float, float], end_xy: Tuple[float, float], radius: int, softness: int) -> np.ndarray:
    """Maska celé dráhy komety ve hvězdně zarovnaném stacku.

    Ve star stacku je kometa rozmazaná podél své dráhy vůči hvězdám. Když pak vložíme
    ostrou kometu z comet stacku, star stack by jinak nechal viditelného ducha komety.
    Tato maska pokrývá úsečku mezi prvním a posledním označením komety.
    """
    h, w = shape
    x0, y0 = float(start_xy[0]), float(start_xy[1])
    x1, y1 = float(end_xy[0]), float(end_xy[1])
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    vx = x1 - x0
    vy = y1 - y0
    denom = vx * vx + vy * vy
    if denom < 1e-6:
        return soft_comet_mask(shape, start_xy, radius, softness)

    t = ((xx - x0) * vx + (yy - y0) * vy) / denom
    t = np.clip(t, 0.0, 1.0)
    px = x0 + t * vx
    py = y0 + t * vy
    dist = np.sqrt((xx - px) ** 2 + (yy - py) ** 2)

    radius = max(5, int(radius))
    softness = max(1, int(softness))
    inner = max(1.0, float(radius - softness))
    fade = (float(radius) - dist) / max(1.0, float(radius) - inner)
    mask = np.where(dist <= inner, 1.0, np.clip(fade, 0.0, 1.0))
    mask = np.where(dist <= float(radius), mask, 0.0)

    if softness > 1:
        k = max(3, int(softness // 2) * 2 + 1)
        mask = cv2.GaussianBlur(mask.astype(np.float32), (k, k), softness / 3.0)
    return np.clip(mask, 0, 1).astype(np.float32)


def remove_comet_ghost_from_star_stack(star_stack: np.ndarray, start_xy: Tuple[float, float], end_xy: Tuple[float, float], radius: int, softness: int) -> np.ndarray:
    """Odstraní ze star stacku rozmazaného ducha komety podél její dráhy.

    Místo duchů vloží lokálně vyhlazené pozadí. Není to astrometricky dokonalé,
    ale pro vizuální sloučení ostré komety + ostrých hvězd je to výrazně lepší.
    """
    h, w = star_stack.shape[:2]
    trail2d = comet_trail_mask((h, w), start_xy, end_xy, max(8, int(radius)), max(4, int(softness)))
    trail = trail2d[..., None]

    # Velký blur odhadne lokální pozadí bez jádra komety; kernel musí být lichý.
    k = max(21, int(radius * 2 + softness) | 1)
    if k > 401:
        k = 401
    background = cv2.GaussianBlur(star_stack.astype(np.float32), (k, k), max(5.0, radius / 2.0))
    cleaned = star_stack * (1.0 - trail) + background * trail
    return np.clip(cleaned, 0, 1).astype(np.float32)



def transform_point_affine(matrix: np.ndarray, xy: Tuple[float, float]) -> Tuple[float, float]:
    """Transformuje bod [x, y] afinní 2x3 maticí."""
    x, y = float(xy[0]), float(xy[1])
    tx = float(matrix[0, 0]) * x + float(matrix[0, 1]) * y + float(matrix[0, 2])
    ty = float(matrix[1, 0]) * x + float(matrix[1, 1]) * y + float(matrix[1, 2])
    return tx, ty


def comet_positions_mask(shape: Tuple[int, int], positions: List[Tuple[float, float]], radius: int, softness: int) -> np.ndarray:
    """Maska celé skutečné dráhy komety ve star-aligned stacku.

    Důležité: ve star stacku nejsou polohy komety prostě raw souřadnice z prvního/posledního
    snímku. Každý snímek se před složením posune/otočí podle hvězd. Proto se musí ručně
    predikovaná poloha komety v každém snímku nejdřív transformovat stejnou star-alignment
    maticí a teprve z těchto bodů vytvořit masku.
    """
    h, w = shape
    positions = [(float(x), float(y)) for x, y in positions if -radius <= x <= w + radius and -radius <= y <= h + radius]
    if not positions:
        return np.zeros((h, w), dtype=np.float32)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist_min = np.full((h, w), np.inf, dtype=np.float32)

    # Vzdálenost k úsečkám mezi postupnými polohami komety.
    if len(positions) == 1:
        x0, y0 = positions[0]
        dist_min = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)
    else:
        for (x0, y0), (x1, y1) in zip(positions[:-1], positions[1:]):
            vx = x1 - x0
            vy = y1 - y0
            denom = vx * vx + vy * vy
            if denom < 1e-6:
                dist = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)
            else:
                t = ((xx - x0) * vx + (yy - y0) * vy) / denom
                t = np.clip(t, 0.0, 1.0)
                px = x0 + t * vx
                py = y0 + t * vy
                dist = np.sqrt((xx - px) ** 2 + (yy - py) ** 2)
            dist_min = np.minimum(dist_min, dist)

    radius = max(5, int(radius))
    softness = max(1, int(softness))
    inner = max(1.0, float(radius - softness))
    fade = (float(radius) - dist_min) / max(1.0, float(radius) - inner)
    mask = np.where(dist_min <= inner, 1.0, np.clip(fade, 0.0, 1.0))
    mask = np.where(dist_min <= float(radius), mask, 0.0)
    if softness > 1:
        k = max(3, int(softness // 2) * 2 + 1)
        mask = cv2.GaussianBlur(mask.astype(np.float32), (k, k), softness / 3.0)
    return np.clip(mask, 0, 1).astype(np.float32)


def remove_comet_ghost_from_star_stack_positions(star_stack: np.ndarray, positions: List[Tuple[float, float]], radius: int, softness: int) -> np.ndarray:
    """Odstraní ducha komety ze star stacku podle reálné star-aligned dráhy."""
    h, w = star_stack.shape[:2]
    trail2d = comet_positions_mask((h, w), positions, max(8, int(radius)), max(4, int(softness)))
    trail = trail2d[..., None]
    k = max(21, int(radius * 2 + softness) | 1)
    if k > 501:
        k = 501
    background = cv2.GaussianBlur(star_stack.astype(np.float32), (k, k), max(5.0, radius / 2.0))
    cleaned = star_stack * (1.0 - trail) + background * trail
    return np.clip(cleaned, 0, 1).astype(np.float32)


def stack_folder_star_with_comet_positions(folder: Path, settings: StackSettings, progress_callback=None) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    set_bayer_pattern_override(getattr(settings, "bayer_pattern", "auto"))
    flat_frame, bias_frame, dark_frame = load_calibration_frames(settings, progress_callback)
    """Star stack + polohy komety po aplikaci star alignment transformací.

    Tato funkce je určená pro Star + Comet merge. Vrací hvězdně zarovnaný stack a seznam
    poloh komety v tomto hvězdném souřadném systému. Díky tomu lze odstranit ducha komety
    přesně tam, kde ve star stacku opravdu leží.
    """
    star_settings = replace(settings, align_mode="star_affine")
    paths, reference_path, scores, sequence_paths = prepare_paths_for_alignment_mode(folder, star_settings, progress_callback)

    report_bayer_conversion_if_needed(reference_path, progress_callback, 0)
    reference = load_calibrated_image_as_float(reference_path, flat_frame, bias_frame, dark_frame)
    h, w = reference.shape[:2]
    reference_gray = to_gray_float(reference)
    reference_median = np.median(reference.reshape(-1, 3), axis=0)

    aligned: List[np.ndarray] = []
    comet_positions: List[Tuple[float, float]] = []
    total = len(paths)

    for idx, path in enumerate(paths):
        report_bayer_conversion_if_needed(path, progress_callback, 10 + int((idx + 1) / total * 60))
        img = load_calibrated_image_as_float(path, flat_frame, bias_frame, dark_frame)
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

        predicted_xy = interpolate_manual_comet_position_for_path(path, sequence_paths, settings)

        if path == reference_path:
            matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
            warped = img
        else:
            moving_gray = to_gray_float(img)
            if progress_callback:
                progress_callback(10 + int((idx + 1) / total * 60), f"Zarovnávám hvězdy ({idx + 1}/{total}): {path.name}")
            matrix = estimate_star_affine(
                reference_gray,
                moving_gray,
                star_settings.downscale_for_alignment,
                star_settings.max_star_shift,
                star_settings.star_border_margin,
                star_settings.strict_star_filter,
                allow_fallback=False,
            )
            if matrix is None:
                if progress_callback:
                    progress_callback(10 + int((idx + 1) / total * 60), f"Vyřazuji bez platného star alignmentu ({idx + 1}/{total}): {path.name}")
                continue
            warped = warp_to_reference(img, matrix, (h, w))

        if predicted_xy is not None:
            comet_positions.append(transform_point_affine(matrix, predicted_xy))

        if star_settings.normalize_background:
            warped = normalize_background(warped, reference_median)

        aligned.append(warped)
        if progress_callback:
            progress_callback(10 + int((idx + 1) / total * 60), f"Zarovnáno hvězdně ({idx + 1}/{total}): {path.name}")

    if not aligned:
        raise ValueError("Žádný snímek neprošel zarovnáním. Zkontroluj, zda složka Light neobsahuje Dark/Bias snímky nebo zda jsou ve snímcích detekovatelné hvězdy.")

    result = stack_aligned_frames(aligned, star_settings, progress_callback)

    return np.clip(result, 0, 1), comet_positions


def blend_star_and_comet_stack_positions(
    star_stack: np.ndarray,
    comet_stack: np.ndarray,
    center_xy: Tuple[float, float],
    radius: int,
    softness: int,
    star_aligned_comet_positions: List[Tuple[float, float]],
) -> np.ndarray:
    """Sloučí ostrý star stack a ostrý comet stack s korektní opravou ducha ve star stacku."""
    trail_radius = max(radius, int(radius * 1.35))
    trail_softness = max(softness, int(softness * 1.35))
    star_base = remove_comet_ghost_from_star_stack_positions(
        star_stack,
        star_aligned_comet_positions,
        trail_radius,
        trail_softness,
    )

    # Pro samotné vložení použij raději stabilní měkkou kruhovou masku okolo ostré komety.
    # Adaptivní rozdílová maska může u soumrakových dat a zeleného pozadí vyrobit fleky.
    mask2d = soft_comet_mask(star_stack.shape[:2], center_xy, radius, softness)
    mask = mask2d[..., None]
    result = star_base * (1.0 - mask) + comet_stack * mask
    return np.clip(result, 0, 1).astype(np.float32)


def blend_star_and_comet_stack(
    star_stack: np.ndarray,
    comet_stack: np.ndarray,
    center_xy: Tuple[float, float],
    radius: int,
    softness: int,
    comet_end_xy: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    # Nejdřív odstraň ze star stacku ducha komety po celé dráze. Jinak vznikne dvojitá kometa:
    # jedna ostrá z comet stacku a druhá rozmazaná/posunutá ze star stacku.
    star_base = star_stack
    if comet_end_xy is not None:
        trail_radius = max(radius, int(radius * 1.25))
        trail_softness = max(softness, int(softness * 1.25))
        star_base = remove_comet_ghost_from_star_stack(star_stack, center_xy, comet_end_xy, trail_radius, trail_softness)

    mask2d = adaptive_comet_mask(star_base, comet_stack, center_xy, radius, softness)
    if mask2d.ndim == 2:
        mask = mask2d[..., None]
    else:
        mask = mask2d
    result = star_base * (1.0 - mask) + comet_stack * mask
    return np.clip(result, 0, 1).astype(np.float32)






def read_fits_header_copy(path: Path):
    """Vrátí kopii první obrazové FIT/FITS hlavičky, nebo None."""
    if fits is None or path is None:
        return None
    path = Path(path)
    if path.suffix.lower() not in {".fits", ".fit"} or not path.exists():
        return None

    try:
        with open_fits_safely(path, memmap=False) as hdul:
            for hdu in hdul:
                if getattr(hdu, "data", None) is not None:
                    return hdu.header.copy()
    except Exception:
        return None
    return None


def find_reference_fits_header(folder: Optional[Path], settings: Optional[StackSettings] = None):
    """Najde vhodnou původní FIT hlavičku pro export stacku.

    Preferuje ručně označený referenční snímek komety, jinak vezme první FIT/FITS
    soubor ve zvolené složce. Metadata kamery, expozice, gainu, teploty apod.
    tak zůstanou zachována i ve výsledném stacku.
    """
    if folder is None:
        return None

    folder = Path(folder)
    candidates: List[Path] = []

    if settings is not None:
        for attr in ("manual_reference_path", "manual_comet_reference_path", "manual_comet_end_path"):
            value = getattr(settings, attr, None)
            if value:
                p = Path(value)
                if p.suffix.lower() in {".fits", ".fit"} and p.exists():
                    candidates.append(p)

    if folder.exists():
        candidates.extend(sorted(p for p in folder.iterdir() if p.suffix.lower() in {".fits", ".fit"}))

    seen = set()
    for p in candidates:
        if str(p) in seen:
            continue
        seen.add(str(p))
        header = read_fits_header_copy(p)
        if header is not None:
            return header
    return None


FITS_STANDARD_KEY_RE = re.compile(r"^[A-Z0-9_-]{1,8}$")


def sanitized_fits_header_for_output(source_header=None):
    """Vrátí kopii FITS hlavičky s klíči bezpečnými pro nový zápis.

    Některé kamery nebo programy ukládají nestandardní klíče s malými písmeny
    (např. medNR-1). Astropy je při zápisu nové HDU odmítne, proto je pro
    výstup převedeme na standardní uppercase formu a neplatné položky přeskočíme.
    """
    if fits is None:
        return None

    clean = fits.Header()
    if source_header is None:
        return clean

    for card in source_header.cards:
        key = str(getattr(card, "keyword", "") or "")
        if not key:
            continue
        try:
            if key in {"COMMENT", "HISTORY"}:
                clean[key] = card.value
                continue

            safe_key = key.upper()
            if not FITS_STANDARD_KEY_RE.match(safe_key):
                continue
            clean[safe_key] = (card.value, card.comment)
        except Exception:
            continue
    return clean


def prepare_output_fits_header(source_header=None, stack_info: Optional[Dict[str, Any]] = None):
    """Připraví FIT hlavičku pro výstup se zachováním původních metadat."""
    if fits is None:
        return None

    header = sanitized_fits_header_for_output(source_header)

    # Strukturní klíče musí odpovídat novému stacku; Astropy je doplní správně.
    for key in (
        "SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "NAXIS3",
        "EXTEND", "PCOUNT", "GCOUNT", "BSCALE", "BZERO",
    ):
        if key in header:
            try:
                del header[key]
            except Exception:
                pass

    preserve_bayer_header = bool(stack_info and stack_info.get("align_mode") == "calibration")
    if not preserve_bayer_header:
        # Po debayeringu/stacku už výstup není raw Bayer mozaika.
        for key in (
            "BAYERPAT", "BAYER_PATTERN", "BAYER", "PATTERN",
            "CFA", "CFA_PAT", "CFA_PATTERN", "CFAPAT",
            "FILTERPAT", "FILTER_PATTERN",
        ):
            if key in header:
                try:
                    del header[key]
                except Exception:
                    pass

    header["CREATOR"] = "Astro Stacker GUI"
    header["BUNIT"] = "normalized"
    header["STACKED"] = (True, "Created by Astro Stacker GUI")
    header["HISTORY"] = "Linear unstretched stack exported by Astro Stacker GUI"
    header["HISTORY"] = "Original FITS header was preserved where possible"
    if preserve_bayer_header:
        header["HISTORY"] = "Calibration stack exported without debayering"
    else:
        header["HISTORY"] = "RGB channel order is R,G,B when NAXIS3=3"

    if stack_info:
        mapping = {
            "align_mode": "ALIGN",
            "stack_mode": "STACKMOD",
            "sigma": "SIGMA",
            "num_images": "NSTACK",
            "bayer_pattern": "BAYERIN",
        }
        for src_key, fits_key in mapping.items():
            if src_key in stack_info and stack_info[src_key] is not None:
                try:
                    header[fits_key] = stack_info[src_key]
                except Exception:
                    pass

    return header


def save_stack_fits(
    path: Path,
    img: np.ndarray,
    source_header=None,
    stack_info: Optional[Dict[str, Any]] = None,
) -> None:
    """Uloží lineární RGB stack jako 32bit FIT/FITS bez jakéhokoliv stretche.

    Pokud je předána původní FIT hlavička, zachová metadata kamery/expozice/gainu
    a doplní jen informace o stackování.
    """
    if fits is None:
        raise RuntimeError("Pro uložení FIT/FITS nainstaluj: pip install astropy")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = np.asarray(img, dtype=np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    data = np.clip(data, 0, 1)
    if data.ndim == 3 and data.shape[2] == 3:
        data = np.moveaxis(data, -1, 0)  # RGB: H x W x 3 -> 3 x H x W

    header = prepare_output_fits_header(source_header, stack_info)
    hdu = fits.PrimaryHDU(data.astype(np.float32), header=header)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", VerifyWarning)
        hdu.writeto(str(path), overwrite=True)


def save_stack_xisf(
    path: Path,
    img: np.ndarray,
    stack_info: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a linear RGB stack as compressed 32-bit floating-point XISF."""
    if XISF is None:
        raise RuntimeError("Pro uložení XISF nainstaluj: pip install xisf")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(img, dtype=np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    data = np.ascontiguousarray(np.clip(data, 0, 1).astype(np.float32))

    fits_keywords: Dict[str, List[Dict[str, Any]]] = {}
    keyword_map = {
        "align_mode": "ASALIGN",
        "stack_mode": "ASSTACK",
        "sigma": "ASSIGMA",
        "num_images": "ASCOUNT",
        "bayer_pattern": "ASBAYER",
    }
    for source_key, keyword in keyword_map.items():
        if stack_info and source_key in stack_info:
            fits_keywords[keyword] = [{"value": str(stack_info[source_key]), "comment": "Astro Stacker"}]
    image_metadata = {"FITSKeywords": fits_keywords} if fits_keywords else None
    XISF.write(
        str(path),
        data,
        creator_app=f"Astro Stacker {APP_VERSION}",
        image_metadata=image_metadata,
        codec="zstd",
        shuffle=True,
    )


def stack_star_and_comet_two_outputs(folder: Path, settings: StackSettings, use_multiprocessing: bool = False, processes: int = 1, progress_callback=None) -> None:
    set_bayer_pattern_override(getattr(settings, "bayer_pattern", "auto"))
    """Režim Star + Comet: pouze uloží dva samostatné soubory.

    Nevytváří žádný merge, nic nemaskuje a nevrací výsledný složený obraz.
    Výstupy se ukládají jako lineární FIT soubory bez stretche do podsložky:
        astro_stacker_output/01_star_stack.fit
        astro_stacker_output/02_comet_stack.fit
    """
    if settings.manual_comet_xy is None or settings.manual_comet_end_xy is None:
        raise ValueError("Star + Comet výstupy vyžadují označení komety v prvním i posledním snímku.")

    output_dir = (
        Path(settings.output_dir)
        if getattr(settings, "output_dir", None)
        else folder / "astro_stacker_output"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_star = output_dir / "01_star_stack.fit"
    output_comet = output_dir / "02_comet_stack.fit"

    source_header = find_reference_fits_header(folder, settings)
    sequence_paths = sequence_paths_for_folder(folder, settings.max_images, settings)
    stack_info = {
        "align_mode": settings.align_mode,
        "stack_mode": settings.stack_mode,
        "sigma": float(settings.sigma),
        "num_images": len(sequence_paths),
        "bayer_pattern": getattr(settings, "bayer_pattern", "auto"),
    }

    star_settings = replace(settings, align_mode="star_affine", auto_reference=False)
    comet_settings = replace(settings, align_mode="comet", auto_reference=False)

    if progress_callback:
        progress_callback(0, "Star + Comet: skládám pouze hvězdy…")
    if use_multiprocessing and processes > 1:
        star_stack = stack_folder_multiprocessing(
            folder,
            star_settings,
            processes,
            lambda p, m: progress_callback(int(p * 0.45), "Hvězdy: " + m) if progress_callback else None,
        )
    else:
        star_stack = stack_folder(
            folder,
            star_settings,
            lambda p, m: progress_callback(int(p * 0.45), "Hvězdy: " + m) if progress_callback else None,
        )

    if progress_callback:
        progress_callback(46, f"Ukládám {output_star.name}…")
    save_stack_fits(output_star, star_stack, source_header=source_header, stack_info=stack_info)

    if progress_callback:
        progress_callback(50, "Star + Comet: skládám pouze kometu…")
    if use_multiprocessing and processes > 1:
        comet_stack = stack_folder_multiprocessing(
            folder,
            comet_settings,
            processes,
            lambda p, m: progress_callback(50 + int(p * 0.45), "Kometa: " + m) if progress_callback else None,
        )
    else:
        comet_stack = stack_folder(
            folder,
            comet_settings,
            lambda p, m: progress_callback(50 + int(p * 0.45), "Kometa: " + m) if progress_callback else None,
        )

    if progress_callback:
        progress_callback(96, f"Ukládám {output_comet.name}…")
    save_stack_fits(output_comet, comet_stack, source_header=source_header, stack_info=stack_info)

    if progress_callback:
        progress_callback(100, f"Hotovo — uloženo: {output_star}, {output_comet}")

def prepare_paths_for_alignment_mode(folder: Path, settings: StackSettings, progress_callback=None) -> Tuple[List[Path], Path, Dict[Path, float], List[Path]]:
    """Připraví snímky a referenci podle režimu zarovnání.

    Důležitá oprava pro komety:
    - Star alignment může používat automaticky vybranou nejostřejší referenci.
    - Comet alignment NESMÍ automaticky přepnout referenci na jiný snímek,
      protože ručně označený pohyb komety je definovaný vůči prvnímu bodu.
    """
    sequence_paths = sequence_paths_for_folder(folder, settings.max_images, settings)

    if settings.align_mode == "comet":
        comet_settings = replace(settings, auto_reference=False)
        paths, reference_path, scores = prepare_stack_paths(folder, comet_settings, progress_callback)

        manual_ref = Path(settings.manual_comet_reference_path) if settings.manual_comet_reference_path else None
        if manual_ref is not None and manual_ref in sequence_paths:
            reference_path = manual_ref
            if reference_path not in paths:
                paths.insert(0, reference_path)
            else:
                paths = [reference_path] + [p for p in paths if p != reference_path]

        return paths, reference_path, scores, sequence_paths

    paths, reference_path, scores = prepare_stack_paths(folder, settings, progress_callback)
    return paths, reference_path, scores, sequence_paths


def selected_satellite_trail_paths() -> set:
    """Return resolved paths marked as trail suspects by the quality pass."""
    metrics = LAST_STACK_SELECTION.get("quality_metrics", {})
    return {
        str(path)
        for path, values in metrics.items()
        if isinstance(values, dict) and float(values.get("satellite_trail", 0.0)) >= 0.5
    }


def stack_folder(folder: Path, settings: StackSettings, progress_callback=None) -> Optional[np.ndarray]:
    CALIBRATION_SIGNATURE_CACHE.clear()
    set_bayer_pattern_override(getattr(settings, "bayer_pattern", "auto"))
    if settings.align_mode == "calibration":
        flat_frame, bias_frame, dark_frame = None, None, None
    else:
        flat_frame, bias_frame, dark_frame = load_calibration_frames(settings, progress_callback)
    if settings.align_mode == "comet_merge":
        stack_star_and_comet_two_outputs(
            folder,
            settings,
            use_multiprocessing=False,
            processes=1,
            progress_callback=progress_callback,
        )
        return None

    paths, reference_path, scores, sequence_paths = prepare_paths_for_alignment_mode(folder, settings, progress_callback)
    satellite_trail_paths = selected_satellite_trail_paths()
    if bool(getattr(settings, "satellite_trail_filter", False)):
        LAST_STACK_SELECTION["satellite_masked_paths"] = sorted(satellite_trail_paths)

    reference = load_stack_frame_for_settings(reference_path, settings, flat_frame, bias_frame, dark_frame)
    h, w = reference.shape[:2]
    reference_gray = to_gray_float(reference)
    reference_median = np.median(reference.reshape(-1, 3), axis=0) if reference.ndim == 3 else float(np.median(reference))

    aligned: List[np.ndarray] = []
    mosaic_records: List[Tuple[Path, np.ndarray, np.ndarray, Optional[np.ndarray]]] = []
    used_paths: List[Path] = []
    total = len(paths)

    for idx, path in enumerate(paths):
        predicted_xy = interpolate_manual_comet_position_for_path(path, sequence_paths, settings) if settings.align_mode == "comet" else None
        if path != reference_path and settings.align_mode != "calibration":
            cached_warped = load_aligned_frame_cache(path, reference_path, settings, predicted_xy)
            if cached_warped is not None and cached_warped.shape[:2] == (h, w):
                aligned.append(np.clip(cached_warped, 0, 1).astype(np.float32))
                used_paths.append(path)
                if progress_callback:
                    progress_callback(10 + int((idx + 1) / total * 60), f"Zarovnáno z cache ({idx + 1}/{total}): {path.name}")
                continue

        img = load_stack_frame_for_settings(path, settings, flat_frame, bias_frame, dark_frame)
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        trail_mask = None
        if (
            bool(getattr(settings, "satellite_trail_filter", False))
            and settings.align_mode != "calibration"
            and str(path.resolve()) in satellite_trail_paths
        ):
            trail_mask = satellite_trail_mask_from_gray(to_gray_float(img), confirmed=True)
            if trail_mask is not None:
                log_debug(f"Satellite trail masked: {path}")

        if path == reference_path or settings.align_mode == "calibration":
            # Kalibrační snímky skládáme pixel na pixel bez zarovnání.
            warped = img
            matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        else:
            moving_gray = to_gray_float(img)
            if progress_callback:
                progress_callback(10 + int((idx + 1) / total * 60), f"Zarovnávám ({idx + 1}/{total}): {path.name}")
            if settings.align_mode == "comet":
                if settings.manual_comet_xy is not None and predicted_xy is not None:
                    matrix = comet_alignment_matrix_with_optional_refine(reference_gray, moving_gray, settings, predicted_xy)
                else:
                    matrix = estimate_comet_translation(reference_gray, moving_gray, settings.downscale_for_alignment, settings.star_border_margin, settings.max_comet_shift, settings.manual_comet_xy)
            elif settings.align_mode == "star_affine":
                matrix = estimate_star_affine(reference_gray, moving_gray, settings.downscale_for_alignment, settings.max_star_shift, settings.star_border_margin, settings.strict_star_filter, allow_fallback=False)
                if matrix is None:
                    if progress_callback:
                        if is_effectively_black_frame(img):
                            progress_callback(10 + int((idx + 1) / total * 60), f"Vyřazuji černý snímek bez hvězd ({idx + 1}/{total}): {path.name}")
                        else:
                            progress_callback(10 + int((idx + 1) / total * 60), f"Vyřazuji bez platného star alignmentu ({idx + 1}/{total}): {path.name}")
                    continue
            elif settings.align_mode == "ecc_affine":
                matrix = estimate_ecc_affine(reference_gray, moving_gray, settings.downscale_for_alignment)
            else:
                matrix = estimate_translation(reference_gray, moving_gray, settings.downscale_for_alignment)
            if getattr(settings, "mosaic_mode", False):
                warped = img
            else:
                warped = warp_to_reference(img, matrix, (h, w))
                warped = mark_invalid_warp_pixels(warped, matrix, (h, w))
        if not getattr(settings, "mosaic_mode", False):
            warped = apply_warped_exclusion_mask(warped, trail_mask, matrix, (h, w))

        if settings.normalize_background and settings.align_mode != "calibration" and not getattr(settings, "mosaic_mode", False):
            warped = normalize_background(warped, reference_median)
        if path != reference_path and settings.align_mode != "calibration":
            save_aligned_frame_cache(path, reference_path, settings, predicted_xy, warped)

        if getattr(settings, "mosaic_mode", False):
            if settings.normalize_background and settings.align_mode != "calibration":
                img = normalize_background(img, reference_median)
            mosaic_records.append((path, img, matrix, trail_mask))
        else:
            aligned.append(warped)
        used_paths.append(path)
        if progress_callback:
            progress_callback(10 + int((idx + 1) / total * 60), f"Zarovnáno ({idx + 1}/{total}): {path.name}")

    if getattr(settings, "mosaic_mode", False):
        aligned = warp_mosaic_records(mosaic_records, (h, w), progress_callback)

    if not aligned:
        raise ValueError("Žádný snímek neprošel zarovnáním. Zkontroluj, zda složka Light neobsahuje Dark/Bias snímky nebo zda jsou ve snímcích detekovatelné hvězdy.")
    refresh_last_stack_selection_after_alignment(used_paths)
    result = stack_aligned_frames(aligned, settings, progress_callback)

    if progress_callback:
        progress_callback(100, "Hotovo")
    return np.clip(result, 0, 1)



def apply_scnr_green(img: np.ndarray, strength: int = 0) -> np.ndarray:
    """SCNR Green — šetrné potlačení zeleného přebytku.

    strength:
    0 = vypnuto
    1 = velmi jemné
    10 = silné

    Používá princip Maximum Neutral: G se omezuje směrem k silnějšímu
    z kanálů R/B. Oproti průměru R/B zachová více skutečného zeleného
    signálu a zelený přebytek se barevně posune k dominantnímu R nebo B.
    """
    strength = int(max(0, min(10, strength)))
    if strength <= 0:
        return img

    out = np.asarray(img, dtype=np.float32).copy()
    r = out[..., 0]
    g = out[..., 1]
    b = out[..., 2]

    # Maximum Neutral je šetrnější než Average Neutral. Pokud je například
    # modrý kanál silný, zelená se přiblíží modré místo zbytečného stažení
    # až k průměru R/B, které může poškodit zelenomodrou mlhovinu.
    neutral_green = np.maximum(r, b)

    # Míra zásahu: 1..10 -> 10..100 %
    amount = strength / 10.0

    # Zasahuj pouze do skutečného zeleného přebytku. R a B zůstávají beze
    # změny; částečná síla plynule míchá původní a neutralizovanou hodnotu.
    g_new = np.where(
        g > neutral_green,
        g * (1.0 - amount) + neutral_green * amount,
        g,
    )
    out[..., 1] = g_new

    return np.clip(out, 0, 1).astype(np.float32)


def apply_vignette_removal(img: np.ndarray, strength: float = 0.0) -> np.ndarray:
    """Jednoduchá radiální korekce vinětace pro vizuální náhled a PNG/JPG/TIFF export."""
    amount = max(0.0, min(1.0, float(strength)))
    if amount <= 1e-6:
        return img

    arr = np.asarray(img, dtype=np.float32)
    h, w = arr.shape[:2]
    if h < 4 or w < 4:
        return arr

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx = (w - 1) * 0.5
    cy = (h - 1) * 0.5
    rx = max(1.0, cx)
    ry = max(1.0, cy)
    r2 = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    r2 = np.clip(r2 / 2.0, 0.0, 1.0)

    gain2d = 1.0 + amount * (1.45 * r2 + 0.55 * r2 * r2)
    gain = gain2d[..., None] if arr.ndim == 3 else gain2d
    corrected = arr * gain

    before = float(np.median(arr))
    after = float(np.median(corrected))
    if before > 1e-6 and after > 1e-6:
        corrected *= before / after
    return np.clip(corrected, 0, 1).astype(np.float32)


def apply_synthetic_flat(img: np.ndarray, strength: float = 0.0) -> np.ndarray:
    """Synthetic flat z hladkého modelu pozadí pro náhled a PNG/JPG/TIFF export."""
    amount = max(0.0, min(1.0, float(strength)))
    if amount <= 1e-6:
        return img

    arr = np.asarray(img, dtype=np.float32)
    h, w = arr.shape[:2]
    if h < 32 or w < 32:
        return arr

    if arr.ndim == 2:
        lum = arr
    else:
        lum = (0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]).astype(np.float32)
    lum = np.nan_to_num(lum, nan=0.0, posinf=0.0, neginf=0.0)

    # Maskuj hvězdy a jasné objekty, aby model pozadí nesnědl reálný signál.
    smooth = cv2.GaussianBlur(lum, (0, 0), 1.0)
    highpass = smooth - cv2.GaussianBlur(smooth, (0, 0), max(8.0, min(h, w) / 80.0))
    star_thr = np.percentile(highpass, 98.0)
    bright_thr = np.percentile(lum, 86.0)
    mask = (highpass > star_thr) | (lum > bright_thr)
    mask = cv2.dilate(mask.astype(np.uint8), np.ones((7, 7), np.uint8), iterations=2).astype(bool)

    bg_values = lum[~mask]
    if bg_values.size < max(256, int(0.05 * lum.size)):
        bg_values = lum.reshape(-1)
    fill = float(np.median(bg_values))
    bg_seed = lum.copy()
    bg_seed[mask] = fill

    sigma = max(18.0, min(h, w) / 18.0)
    background = cv2.GaussianBlur(bg_seed, (0, 0), sigma)
    background = cv2.GaussianBlur(background, (0, 0), sigma)

    bg_med = float(np.median(background))
    if bg_med <= 1e-6:
        return arr
    flat = np.clip(background / bg_med, 0.25, 4.0)
    correction = 1.0 / flat
    correction = 1.0 + amount * (correction - 1.0)
    gain = correction[..., None] if arr.ndim == 3 else correction
    corrected = arr * gain

    before = float(np.median(arr))
    after = float(np.median(corrected))
    if before > 1e-6 and after > 1e-6:
        corrected *= before / after
    return np.clip(corrected, 0, 1).astype(np.float32)


def apply_color_background_correction(img: np.ndarray, strength: float = 0.0) -> np.ndarray:
    """Suppress a smooth per-channel color background cast for preview/export."""
    amount = max(0.0, min(1.0, float(strength)))
    arr = np.asarray(img, dtype=np.float32)
    if amount <= 1e-6 or arr.ndim != 3 or arr.shape[2] < 3:
        return arr

    h, w = arr.shape[:2]
    if h < 32 or w < 32:
        return arr

    max_bg_edge = 900
    scale = min(1.0, float(max_bg_edge) / float(max(h, w)))
    if scale < 1.0:
        sw = max(32, int(round(w * scale)))
        sh = max(32, int(round(h * scale)))
        work = cv2.resize(arr, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32)
    else:
        work = arr
        sh, sw = h, w

    lum = (0.2126 * work[..., 0] + 0.7152 * work[..., 1] + 0.0722 * work[..., 2]).astype(np.float32)
    smooth = cv2.GaussianBlur(lum, (0, 0), 1.0)
    highpass = smooth - cv2.GaussianBlur(smooth, (0, 0), max(6.0, min(sh, sw) / 80.0))
    star_thr = np.percentile(highpass, 97.5)
    bright_thr = np.percentile(lum, 82.0)
    dark_thr = np.percentile(lum, 2.0)
    mask = (highpass > star_thr) | (lum > bright_thr) | (lum < dark_thr)
    mask = cv2.dilate(mask.astype(np.uint8), np.ones((7, 7), np.uint8), iterations=2).astype(bool)

    backgrounds = []
    sigma = max(14.0, min(sh, sw) / 14.0)
    for channel in range(3):
        plane = work[..., channel].astype(np.float32)
        bg_values = plane[~mask]
        if bg_values.size < max(256, int(0.04 * plane.size)):
            bg_values = plane.reshape(-1)
        fill = float(np.median(bg_values))
        seed = plane.copy()
        seed[mask] = fill
        bg = cv2.GaussianBlur(seed, (0, 0), sigma)
        bg = cv2.GaussianBlur(bg, (0, 0), sigma)
        backgrounds.append(bg)

    bg_stack = np.stack(backgrounds, axis=2).astype(np.float32)
    if bg_stack.shape[:2] != (h, w):
        bg_stack = cv2.resize(bg_stack, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    neutral_bg = np.median(bg_stack, axis=2, keepdims=True)
    color_cast = bg_stack - neutral_bg
    corrected = arr - amount * color_cast

    before = float(np.median(lum))
    after_lum = 0.2126 * corrected[..., 0] + 0.7152 * corrected[..., 1] + 0.0722 * corrected[..., 2]
    after = float(np.median(after_lum))
    if before > 1e-6 and after > 1e-6:
        corrected *= before / after
    return np.clip(corrected, 0, 1).astype(np.float32)


def apply_polynomial_gradient_removal(img: np.ndarray) -> np.ndarray:
    """Remove a strong smooth RGB gradient with the validated production model."""
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return arr
    h, w = arr.shape[:2]
    if h < 48 or w < 48:
        return arr

    max_edge = 720
    scale = min(1.0, float(max_edge) / float(max(h, w)))
    if scale < 0.999:
        sw = max(48, int(round(w * scale)))
        sh = max(48, int(round(h * scale)))
        work = cv2.resize(
            arr[..., :3], (sw, sh), interpolation=cv2.INTER_AREA
        ).astype(np.float32)
    else:
        work = arr[..., :3]
        sh, sw = h, w

    lum = (
        0.2126 * work[..., 0]
        + 0.7152 * work[..., 1]
        + 0.0722 * work[..., 2]
    ).astype(np.float32)
    finite = np.all(np.isfinite(work), axis=2) & np.isfinite(lum)
    if np.count_nonzero(finite) < 256:
        return arr

    low_lum = float(np.percentile(lum[finite], 1))
    positive = finite & (np.min(work, axis=2) > max(1e-7, low_lum * 0.15))
    if np.count_nonzero(positive) < 256:
        positive = finite

    samples_xy: List[Tuple[float, float]] = []
    samples_rgb: List[np.ndarray] = []
    grid_x, grid_y = 18, 14
    for gy in range(grid_y):
        y0 = int(round(gy * sh / grid_y))
        y1 = max(y0 + 1, int(round((gy + 1) * sh / grid_y)))
        for gx in range(grid_x):
            x0 = int(round(gx * sw / grid_x))
            x1 = max(x0 + 1, int(round((gx + 1) * sw / grid_x)))
            valid = positive[y0:y1, x0:x1]
            if np.count_nonzero(valid) < 12:
                continue
            cell_lum = lum[y0:y1, x0:x1][valid]
            lo = float(np.percentile(cell_lum, 12))
            hi = float(np.percentile(cell_lum, 48))
            local_lum = lum[y0:y1, x0:x1]
            background = valid & (local_lum >= lo) & (local_lum <= hi)
            if np.count_nonzero(background) < 8:
                continue
            cell = work[y0:y1, x0:x1]
            samples_xy.append(
                ((x0 + x1 - 1) * 0.5, (y0 + y1 - 1) * 0.5)
            )
            samples_rgb.append(
                np.median(cell[background], axis=0).astype(np.float32)
            )

    if len(samples_xy) < 18:
        return arr

    coords = np.asarray(samples_xy, dtype=np.float64)
    values = np.asarray(samples_rgb, dtype=np.float64)
    nx = coords[:, 0] / max(1.0, float(sw - 1)) * 2.0 - 1.0
    ny = coords[:, 1] / max(1.0, float(sh - 1)) * 2.0 - 1.0
    design = np.stack(
        [np.ones_like(nx), nx, ny, nx * nx, nx * ny, ny * ny], axis=1
    )

    coeffs = []
    for channel in range(3):
        channel_values = values[:, channel]
        keep = np.ones(len(channel_values), dtype=bool)
        coef = np.linalg.lstsq(design, channel_values, rcond=None)[0]
        for _ in range(4):
            residual = channel_values - design @ coef
            center = float(np.median(residual[keep]))
            mad = (
                float(np.median(np.abs(residual[keep] - center))) * 1.4826
                + 1e-8
            )
            new_keep = np.abs(residual - center) <= 2.8 * mad
            if np.count_nonzero(new_keep) < 12:
                break
            keep = new_keep
            coef = np.linalg.lstsq(
                design[keep], channel_values[keep], rcond=None
            )[0]
        coeffs.append(coef)

    fx = (
        np.arange(w, dtype=np.float32)[None, :]
        / max(1.0, float(w - 1))
        * 2.0
        - 1.0
    )
    fy = (
        np.arange(h, dtype=np.float32)[:, None]
        / max(1.0, float(h - 1))
        * 2.0
        - 1.0
    )
    corrected = arr[..., :3].copy()
    for channel, coef in enumerate(coeffs):
        c = np.asarray(coef, dtype=np.float32)
        background = (
            c[0]
            + c[1] * fx
            + c[2] * fy
            + c[3] * fx * fx
            + c[4] * fx * fy
            + c[5] * fy * fy
        )
        target = float(np.median(design @ coef))
        corrected[..., channel] -= background - target

    log_debug(
        f"{GRADIENT_REMOVAL_ALGORITHM_VERSION}: "
        f"samples={len(samples_xy)}, size={w}x{h}"
    )
    return np.clip(
        np.nan_to_num(corrected, nan=0.0, posinf=1.0, neginf=0.0),
        0.0,
        1.0,
    ).astype(np.float32)


def apply_astro_denoise(img: np.ndarray, strength: float = 0.0) -> np.ndarray:
    """Preview/export denoise that protects stars and bright nebula detail.

    This is intentionally conservative: it smooths low-contrast background,
    plus stronger color speckle noise in chroma channels, while a luminance/edge
    mask keeps stars and sharper structure close to the original stretched image.
    """
    amount = max(0.0, min(1.0, float(strength)))
    arr = np.asarray(img, dtype=np.float32)
    if amount <= 1e-6 or arr.ndim != 3 or arr.shape[2] < 3:
        return arr

    h, w = arr.shape[:2]
    if h < 8 or w < 8:
        return arr

    work = np.clip(np.nan_to_num(arr[..., :3], nan=0.0, posinf=1.0, neginf=0.0), 0, 1).astype(np.float32)
    lum = (0.2126 * work[..., 0] + 0.7152 * work[..., 1] + 0.0722 * work[..., 2]).astype(np.float32)

    # Protect stars, sharp edges, and stronger nebulosity gradients.
    small_blur = cv2.GaussianBlur(lum, (0, 0), 1.0)
    large_blur = cv2.GaussianBlur(lum, (0, 0), max(6.0, min(h, w) / 90.0))
    highpass = np.abs(lum - small_blur)
    structure = np.abs(lum - large_blur)
    grad_x = cv2.Sobel(lum, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(lum, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(grad_x, grad_y)

    try:
        star_thr = np.percentile(highpass, 98.6)
        bright_thr = np.percentile(lum, 96.5)
        struct_thr = np.percentile(structure, 93.0)
        grad_thr = np.percentile(gradient, 95.0)
    except Exception:
        return work
    protect = (
        (highpass > star_thr)
        | (lum > bright_thr)
        | (structure > struct_thr)
        | (gradient > grad_thr)
    ).astype(np.uint8)
    protect = cv2.dilate(protect, np.ones((3, 3), np.uint8), iterations=1).astype(np.float32)
    protect = cv2.GaussianBlur(protect, (0, 0), 1.4)
    protect = np.clip(protect, 0.0, 1.0)

    # Luminance denoise: bilateral keeps gentle edges better than pure blur.
    sigma_color = 0.018 + amount * 0.055
    sigma_space = 1.2 + amount * 3.8
    lum_denoised = cv2.bilateralFilter(lum, d=0, sigmaColor=sigma_color, sigmaSpace=sigma_space)

    # Chroma denoise: operate in Lab so random red/green/blue speckles can be
    # reduced without flattening the luminance detail that defines stars and
    # nebula structure. Median catches isolated color pixels, bilateral keeps
    # broad color transitions, and a wider Gaussian removes fine chroma grain.
    try:
        lab = cv2.cvtColor(work, cv2.COLOR_RGB2LAB).astype(np.float32)
        chroma = lab[..., 1:3]
        chroma_med = np.empty_like(chroma)
        for c in range(2):
            chroma_med[..., c] = cv2.medianBlur(chroma[..., c], 3)
        chroma_bilateral = cv2.bilateralFilter(
            chroma_med,
            d=0,
            sigmaColor=4.0 + amount * 16.0,
            sigmaSpace=1.5 + amount * 5.0,
        )
        chroma_blur = cv2.GaussianBlur(chroma_bilateral, (0, 0), 0.9 + amount * 3.2)
        lab_smooth = lab.copy()
        chroma_blend = np.clip(amount * 1.25 * (1.0 - 0.70 * protect), 0.0, 1.0)
        lab_smooth[..., 1:3] = (
            lab[..., 1:3] * (1.0 - chroma_blend[..., None])
            + chroma_blur * chroma_blend[..., None]
        )
        chroma_denoised = np.clip(cv2.cvtColor(lab_smooth, cv2.COLOR_LAB2RGB), 0, 1).astype(np.float32)
    except Exception:
        chroma_residual = work - lum[..., None]
        chroma_smooth = cv2.GaussianBlur(chroma_residual, (0, 0), 0.9 + amount * 3.2)
        chroma_blend = np.clip(amount * 1.15 * (1.0 - 0.70 * protect), 0.0, 1.0)
        chroma_denoised = work * (1.0 - chroma_blend[..., None]) + (lum[..., None] + chroma_smooth) * chroma_blend[..., None]

    lum_mix = np.clip(amount * 0.75 * (1.0 - 0.92 * protect), 0.0, 1.0)
    out_lum = lum * (1.0 - lum_mix) + lum_denoised * lum_mix
    chroma_lum = (0.2126 * chroma_denoised[..., 0] + 0.7152 * chroma_denoised[..., 1] + 0.0722 * chroma_denoised[..., 2]).astype(np.float32)
    out = chroma_denoised + (out_lum - chroma_lum)[..., None]
    return np.clip(out, 0, 1).astype(np.float32)


def find_drunet_model(explicit_path: Optional[str] = None) -> Optional[Path]:
    """Find a bundled, user-selected, or locally installed DRUNet ONNX model."""
    candidates: List[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())

    roots: List[Path] = []
    if getattr(sys, "frozen", False):
        roots.extend(
            [
                Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent)),
                Path(sys.executable).resolve().parent,
            ]
        )
    module_root = Path(__file__).resolve().parent
    roots.extend([module_root, module_root.parent, Path.cwd()])

    if sys.platform == "darwin":
        roots.append(Path.home() / "Library" / "Application Support" / "AstroStacker" / "models")
    elif os.name == "nt":
        local_appdata = os.environ.get("LOCALAPPDATA")
        if local_appdata:
            roots.append(Path(local_appdata) / "AstroStacker" / "models")
    else:
        roots.append(Path.home() / ".local" / "share" / "AstroStacker" / "models")

    for root in roots:
        for filename in DRUNET_MODEL_FILENAMES:
            candidates.append(root / filename)
            candidates.append(root / "models" / filename)
            # An external PixInsight wrapper installed next to a PyInstaller
            # one-folder application can reuse the bundled models without
            # installing another 250 MB copy.
            candidates.append(root / "_internal" / "models" / filename)

    seen = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            resolved = candidate
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file() and candidate.suffix.lower() == ".onnx":
            return candidate
    return None


def drunet_execution_providers() -> List[str]:
    if ort is None:
        return []
    available = set(ort.get_available_providers())
    preferred: List[str] = []
    if sys.platform == "darwin":
        preferred.append("CoreMLExecutionProvider")
    elif os.name == "nt":
        preferred.extend(["CUDAExecutionProvider", "DmlExecutionProvider"])
    preferred.append("CPUExecutionProvider")
    return [provider for provider in preferred if provider in available]


def get_drunet_session(model_path: Path):
    if ort is None:
        raise RuntimeError("ONNX Runtime is not installed. Install: pip install onnxruntime")
    providers = drunet_execution_providers()
    if not providers:
        raise RuntimeError("ONNX Runtime has no usable execution provider.")
    key = (
        str(Path(model_path).resolve()),
        tuple(repr(provider) for provider in providers),
    )
    cached = DRUNET_SESSION_CACHE.get(key)
    if cached is not None:
        return cached

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.log_severity_level = 3
    if "DmlExecutionProvider" in providers:
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=providers)
    active_providers = session.get_providers()
    missing_providers = [provider for provider in providers if provider not in active_providers]
    if missing_providers:
        log_debug(
            "DRUNet ONNX provider fallback: "
            f"requested={providers}; active={active_providers}; "
            f"unavailable={missing_providers}."
        )
    DRUNET_SESSION_CACHE[key] = session
    log_debug(f"DRUNet ONNX loaded: {model_path}; providers={active_providers}")
    return session


def find_stellar_deconvolution_model(
    explicit_path: Optional[str] = None,
) -> Optional[Path]:
    """Find the optional Cosmic Clarity stellar sharpening ONNX model."""
    candidates: List[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())

    roots: List[Path] = []
    if getattr(sys, "frozen", False):
        roots.extend(
            [
                Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent)),
                Path(sys.executable).resolve().parent,
            ]
        )
    module_root = Path(__file__).resolve().parent
    roots.extend([module_root, module_root.parent, Path.cwd()])
    for root in roots:
        for filename in STELLAR_MODEL_FILENAMES:
            candidates.extend(
                [
                    root / filename,
                    root / "models" / filename,
                    root / "_internal" / "models" / filename,
                ]
            )

    seen = set()
    for candidate in candidates:
        try:
            key = str(candidate.resolve())
        except Exception:
            key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file() and candidate.suffix.lower() == ".onnx":
            return candidate
    return None


def get_stellar_deconvolution_session(model_path: Path):
    if ort is None:
        raise RuntimeError("ONNX Runtime is not installed. Install: pip install onnxruntime")
    providers = cosmic_clarity_execution_providers()
    if not providers:
        raise RuntimeError("ONNX Runtime has no usable execution provider.")
    key = (
        str(Path(model_path).resolve()),
        tuple(repr(provider) for provider in providers),
    )
    cached = STELLAR_SESSION_CACHE.get(key)
    if cached is not None:
        return cached

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.log_severity_level = 3
    if "DmlExecutionProvider" in providers:
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    try:
        session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=providers,
        )
    except Exception:
        if providers == ["CPUExecutionProvider"]:
            raise
        log_debug(
            "Stellar ONNX GPU provider initialization failed; "
            "falling back to CPU:\n"
            + traceback.format_exc()
        )
        session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
    STELLAR_SESSION_CACHE[key] = session
    log_debug(
        f"Cosmic Clarity stellar ONNX loaded: {model_path}; "
        f"providers={session.get_providers()}"
    )
    return session


def cosmic_clarity_execution_providers():
    """Prefer the Apple GPU explicitly for the fixed-size Cosmic models."""
    if ort is None:
        return []
    available = set(ort.get_available_providers())
    providers: List[Any] = []
    if sys.platform == "darwin" and "CoreMLExecutionProvider" in available:
        cache_dir = (
            Path.home()
            / "Library"
            / "Caches"
            / "AstroStacker"
            / "CoreML"
        )
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            cache_dir = Path(tempfile.gettempdir()) / "AstroStacker-CoreML"
            cache_dir.mkdir(parents=True, exist_ok=True)
        providers.append(
            (
                "CoreMLExecutionProvider",
                {
                    "ModelFormat": "MLProgram",
                    "MLComputeUnits": "CPUAndGPU",
                    "RequireStaticInputShapes": "1",
                    "EnableOnSubgraphs": "0",
                    "SpecializationStrategy": "FastPrediction",
                    "ModelCacheDirectory": str(cache_dir),
                },
            )
        )
    elif os.name == "nt":
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        if "DmlExecutionProvider" in available:
            providers.append("DmlExecutionProvider")
    if "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")
    return providers


def _drunet_tile_positions(length: int, tile: int, overlap: int) -> List[int]:
    if length <= tile:
        return [0]
    step = max(1, tile - overlap)
    positions = list(range(0, max(1, length - tile + 1), step))
    last = length - tile
    if not positions or positions[-1] != last:
        positions.append(last)
    return positions


def _drunet_tile_weight(
    tile_h: int,
    tile_w: int,
    overlap: int,
    top_edge: bool,
    bottom_edge: bool,
    left_edge: bool,
    right_edge: bool,
) -> np.ndarray:
    wy = np.ones(tile_h, dtype=np.float32)
    wx = np.ones(tile_w, dtype=np.float32)
    ramp_h = min(overlap, tile_h // 2)
    ramp_w = min(overlap, tile_w // 2)
    if ramp_h > 0:
        ramp = np.sin(np.linspace(0.0, np.pi / 2.0, ramp_h, dtype=np.float32)) ** 2
        if not top_edge:
            wy[:ramp_h] = ramp
        if not bottom_edge:
            wy[-ramp_h:] = ramp[::-1]
    if ramp_w > 0:
        ramp = np.sin(np.linspace(0.0, np.pi / 2.0, ramp_w, dtype=np.float32)) ** 2
        if not left_edge:
            wx[:ramp_w] = ramp
        if not right_edge:
            wx[-ramp_w:] = ramp[::-1]
    return wy[:, None] * wx[None, :]


def stellar_deconvolution_mask(img: np.ndarray, star_size: float) -> np.ndarray:
    """Build a soft star-only mask while excluding broad nebula/galaxy structure."""
    rgb = np.clip(np.asarray(img, dtype=np.float32)[..., :3], 0.0, 1.0)
    lum = (
        0.2126 * rgb[..., 0]
        + 0.7152 * rgb[..., 1]
        + 0.0722 * rgb[..., 2]
    ).astype(np.float32)
    local_background = cv2.GaussianBlur(lum, (0, 0), 3.2)
    compact = np.maximum(lum - local_background, 0.0)
    noise = float(np.median(np.abs(compact - np.median(compact))) * 1.4826)
    threshold = max(float(np.percentile(compact, 97.8)), noise * 3.5, 1e-4)
    core = ((compact >= threshold) & (lum >= np.percentile(lum, 65.0))).astype(np.uint8)

    radius = max(1, int(round(np.clip(star_size, 1.0, 10.0) * 0.65)))
    kernel_size = radius * 2 + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    expanded = cv2.dilate(core, kernel, iterations=1).astype(np.float32)
    sigma = max(0.8, radius * 0.55)
    return np.clip(cv2.GaussianBlur(expanded, (0, 0), sigma), 0.0, 1.0)


def run_stellar_deconvolution_onnx(
    img: np.ndarray,
    model_path: Path,
    tile_size: int = 256,
    overlap: int = 64,
) -> np.ndarray:
    """Run Cosmic Clarity stellar sharpening on luminance and return raw output."""
    source = np.clip(
        np.nan_to_num(np.asarray(img, dtype=np.float32), nan=0.0, posinf=1.0),
        0.0,
        1.0,
    )
    if source.ndim != 3 or source.shape[2] < 3:
        raise ValueError("AI Star Deconvolution requires an RGB display image.")

    session = get_stellar_deconvolution_session(Path(model_path))
    input_meta = session.get_inputs()[0]
    input_shape = list(input_meta.shape)
    fixed_h = input_shape[2] if len(input_shape) == 4 and isinstance(input_shape[2], int) else None
    fixed_w = input_shape[3] if len(input_shape) == 4 and isinstance(input_shape[3], int) else None
    tile_h = int(fixed_h or tile_size)
    tile_w = int(fixed_w or tile_size)
    overlap = max(16, min(int(overlap), tile_h // 3, tile_w // 3))

    rgb = np.ascontiguousarray(source[..., :3])
    lum = (
        0.2126 * rgb[..., 0]
        + 0.7152 * rgb[..., 1]
        + 0.0722 * rgb[..., 2]
    ).astype(np.float32)
    h, w = lum.shape
    padded_h = max(h, tile_h)
    padded_w = max(w, tile_w)
    padded = np.pad(
        lum,
        ((0, padded_h - h), (0, padded_w - w)),
        mode="reflect",
    )
    out_sum = np.zeros((padded_h, padded_w), dtype=np.float32)
    weight_sum = np.zeros((padded_h, padded_w), dtype=np.float32)
    input_name = input_meta.name
    output_name = session.get_outputs()[0].name

    for y0 in _drunet_tile_positions(padded_h, tile_h, overlap):
        for x0 in _drunet_tile_positions(padded_w, tile_w, overlap):
            tile = padded[y0:y0 + tile_h, x0:x0 + tile_w]
            tensor = np.repeat(tile[None, None], 3, axis=1).astype(np.float32)
            prediction = session.run([output_name], {input_name: tensor})[0]
            if prediction.ndim != 4 or prediction.shape[1] < 1:
                raise ValueError(
                    f"Unexpected stellar model output shape: {prediction.shape}"
                )
            tile_out = np.asarray(prediction[0, 0], dtype=np.float32)
            weight = _drunet_tile_weight(
                tile_h,
                tile_w,
                overlap,
                y0 == 0,
                y0 + tile_h >= padded_h,
                x0 == 0,
                x0 + tile_w >= padded_w,
            )
            out_sum[y0:y0 + tile_h, x0:x0 + tile_w] += tile_out * weight
            weight_sum[y0:y0 + tile_h, x0:x0 + tile_w] += weight

    sharpened_lum = out_sum / np.maximum(weight_sum, 1e-6)
    sharpened_lum = np.clip(sharpened_lum[:h, :w], 0.0, 1.0)
    delta = sharpened_lum - lum
    return np.clip(rgb + delta[..., None], 0.0, 1.0).astype(np.float32)


def astro_detail_protection_mask(img: np.ndarray) -> np.ndarray:
    """Soft mask protecting stars and high-contrast astronomical structures."""
    work = np.clip(np.asarray(img, dtype=np.float32)[..., :3], 0, 1)
    lum = 0.2126 * work[..., 0] + 0.7152 * work[..., 1] + 0.0722 * work[..., 2]
    blur = cv2.GaussianBlur(lum, (0, 0), 1.1)
    highpass = np.abs(lum - blur)
    grad_x = cv2.Sobel(lum, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(lum, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(grad_x, grad_y)
    star_thr = float(np.percentile(highpass, 98.2))
    grad_thr = float(np.percentile(gradient, 96.0))
    bright_thr = float(np.percentile(lum, 98.5))
    mask = ((highpass > star_thr) | (gradient > grad_thr) | (lum > bright_thr)).astype(np.uint8)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1).astype(np.float32)
    return np.clip(cv2.GaussianBlur(mask, (0, 0), 1.5), 0, 1)


def apply_drunet_onnx(
    img: np.ndarray,
    strength: float,
    model_path: Path,
    tile_size: int = 512,
    overlap: int = 48,
) -> np.ndarray:
    """Run color DRUNet ONNX tile inference on a stretched RGB image."""
    amount = float(np.clip(strength, 0.0, 1.0))
    source = np.clip(np.nan_to_num(np.asarray(img, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0), 0, 1)
    if amount <= 1e-6:
        return source
    if source.ndim != 3 or source.shape[2] < 3:
        raise ValueError("DRUNet requires an RGB display image.")

    session = get_drunet_session(Path(model_path))
    input_meta = session.get_inputs()[0]
    input_shape = list(input_meta.shape)
    input_channels = input_shape[1] if len(input_shape) == 4 and isinstance(input_shape[1], int) else 4
    fixed_h = input_shape[2] if len(input_shape) == 4 and isinstance(input_shape[2], int) else None
    fixed_w = input_shape[3] if len(input_shape) == 4 and isinstance(input_shape[3], int) else None
    tile_h = int(fixed_h or tile_size)
    tile_w = int(fixed_w or tile_size)
    tile_h = max(32, tile_h - tile_h % 8)
    tile_w = max(32, tile_w - tile_w % 8)
    overlap = max(8, min(int(overlap), tile_h // 3, tile_w // 3))

    rgb = np.ascontiguousarray(source[..., :3])
    grayscale_model = input_channels in (1, 2)
    if input_channels not in (1, 2, 3, 4):
        raise ValueError(f"Unsupported DRUNet input channel count: {input_channels}")
    if grayscale_model:
        model_input = (
            0.2126 * rgb[..., 0]
            + 0.7152 * rgb[..., 1]
            + 0.0722 * rgb[..., 2]
        )[..., None].astype(np.float32)
    else:
        model_input = rgb
    h, w = rgb.shape[:2]
    padded_h = max(h, tile_h)
    padded_w = max(w, tile_w)
    pad_bottom = padded_h - h
    pad_right = padded_w - w
    padded = np.pad(
        model_input,
        ((0, pad_bottom), (0, pad_right), (0, 0)),
        mode="reflect",
    )
    output_channels = 1 if grayscale_model else 3
    out_sum = np.zeros(
        (padded_h, padded_w, output_channels),
        dtype=np.float32,
    )
    weight_sum = np.zeros(padded.shape[:2], dtype=np.float32)

    noise_sigma = (5.0 + 45.0 * amount) / 255.0
    y_positions = _drunet_tile_positions(padded_h, tile_h, overlap)
    x_positions = _drunet_tile_positions(padded_w, tile_w, overlap)
    input_name = input_meta.name
    output_name = session.get_outputs()[0].name

    for y0 in y_positions:
        for x0 in x_positions:
            tile = padded[y0:y0 + tile_h, x0:x0 + tile_w, :]
            tensor = np.transpose(tile, (2, 0, 1))[None].astype(np.float32)
            if input_channels in (2, 4):
                noise_map = np.full((1, 1, tile_h, tile_w), noise_sigma, dtype=np.float32)
                tensor = np.concatenate((tensor, noise_map), axis=1)

            prediction = session.run([output_name], {input_name: tensor})[0]
            if prediction.ndim != 4:
                raise ValueError(f"Unexpected DRUNet output shape: {prediction.shape}")
            if prediction.shape[1] < output_channels:
                raise ValueError(f"Unexpected DRUNet output shape: {prediction.shape}")
            tile_out = np.transpose(
                prediction[0, :output_channels],
                (1, 2, 0),
            )
            tile_out = np.clip(tile_out[:tile_h, :tile_w], 0, 1).astype(np.float32)
            weight = _drunet_tile_weight(
                tile_h,
                tile_w,
                overlap,
                y0 == 0,
                y0 + tile_h >= padded_h,
                x0 == 0,
                x0 + tile_w >= padded_w,
            )
            out_sum[y0:y0 + tile_h, x0:x0 + tile_w] += tile_out * weight[..., None]
            weight_sum[y0:y0 + tile_h, x0:x0 + tile_w] += weight

    denoised = out_sum / np.maximum(weight_sum[..., None], 1e-6)
    denoised = denoised[:h, :w]
    if grayscale_model:
        original_luminance = model_input[:h, :w, 0]
        luminance_delta = denoised[..., 0] - original_luminance
        denoised = np.clip(
            rgb + luminance_delta[..., None],
            0.0,
            1.0,
        ).astype(np.float32)
    protect = astro_detail_protection_mask(rgb)
    blend = amount * (1.0 - 0.82 * protect)
    result = rgb * (1.0 - blend[..., None]) + denoised * blend[..., None]
    return np.clip(result, 0, 1).astype(np.float32)


def apply_stretch(img: np.ndarray, s: StretchSettings) -> np.ndarray:
    out = img.astype(np.float32).copy()
    mono_visual = out.ndim == 2 or (out.ndim == 3 and out.shape[2] == 1)
    if out.ndim == 2:
        out = np.repeat(out[..., None], 3, axis=2)
    elif out.ndim == 3 and out.shape[2] == 1:
        out = np.repeat(out, 3, axis=2)

    out = apply_vignette_removal(out, getattr(s, "vignette_removal", 0.0))
    out = apply_synthetic_flat(out, getattr(s, "synthetic_flat", 0.0))
    out = apply_color_background_correction(out, getattr(s, "color_background_correction", 0.0))

    black = s.black / 65535.0
    white = max((s.white / 65535.0), black + 1e-6)
    out = (out - black) / (white - black)
    out = np.clip(out, 0, 1)

    gamma = max(0.05, s.gamma)
    out = np.power(out, 1.0 / gamma)

    # Adjustable additional STF. Keep the direction unambiguous:
    # 0 = no extra STF, 5 = medium, 10 = strongest reveal of faint structures.
    stf_strength = float(np.clip(getattr(s, "stf_strength", 5.0), 0.0, 10.0))
    normalized_strength = stf_strength / 10.0
    if normalized_strength > 1e-6:
        mtf = 0.5 - 0.30 * np.power(normalized_strength, 0.85)
        out = midtones_transfer(float(mtf), out)

    out = (out - 0.5) * s.contrast + 0.5
    out = np.clip(out, 0, 1)

    if not mono_visual:
        # RGB balance
        out[..., 0] *= s.red
        out[..., 1] *= s.green
        out[..., 2] *= s.blue

    # Preserve highlight structure after RGB/Balance gains. A hard per-channel
    # clip turns bright galaxy nuclei into a flat white or colored patch as soon
    # as one channel exceeds 1.0. Compress the per-pixel peak smoothly instead
    # and scale all channels together so their color ratios remain unchanged.
    peak = np.max(out, axis=2)
    shoulder = 0.85
    bright = peak > shoulder
    if np.any(bright):
        span = 1.0 - shoulder
        rolled_peak = shoulder + span * (
            1.0 - np.exp(-(peak - shoulder) / span)
        )
        scale = np.ones_like(peak, dtype=np.float32)
        scale[bright] = rolled_peak[bright] / np.maximum(peak[bright], 1e-6)
        out *= scale[..., None]
    out = np.clip(out, 0, 1)

    if not mono_visual:
        # SCNR Green — astro potlačení zeleného nádechu po RGB balance.
        out = apply_scnr_green(out, getattr(s, "scnr_green_strength", 0))

        # Saturation in floating-point HSV. Keep the processing pipeline at
        # float32 precision; uint8 conversion belongs only to the final GUI/PNG
        # output and would otherwise reduce even 16-bit TIFF exports to 256 levels.
        hsv = cv2.cvtColor(np.ascontiguousarray(out.astype(np.float32)), cv2.COLOR_RGB2HSV)
        hsv[..., 1] *= s.saturation
        hsv[..., 1] = np.clip(hsv[..., 1], 0.0, 1.0)
        out = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32)
        out = np.clip(out, 0.0, 1.0)
    if getattr(s, "denoise_mode", "classic") != "drunet":
        out = apply_astro_denoise(out, getattr(s, "denoise_strength", 0.0))
    return out


def numpy_to_qpixmap(img: np.ndarray, max_size: Optional[Tuple[int, int]] = None, zoom: float = 1.0) -> QPixmap:
    """Convert RGB float image 0..1 to QPixmap.

    - max_size: fit-to-window mode
    - zoom: fixed zoom mode, e.g. 1.0 = 100 % / 1:1
    """
    img8 = visual_float_to_uint8(img)
    if img8.ndim == 2:
        img8 = np.repeat(img8[..., None], 3, axis=2)
    elif img8.ndim == 3 and img8.shape[2] == 1:
        img8 = np.repeat(img8, 3, axis=2)
    h, w, ch = img8.shape
    qimg = QImage(img8.data, w, h, ch * w, QImage.Format_RGB888).copy()
    pixmap = QPixmap.fromImage(qimg)

    if max_size is not None:
        return pixmap.scaled(max_size[0], max_size[1], Qt.KeepAspectRatio, Qt.SmoothTransformation)

    if abs(zoom - 1.0) < 1e-6:
        return pixmap

    new_w = max(1, int(w * zoom))
    new_h = max(1, int(h * zoom))
    return pixmap.scaled(new_w, new_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)


def visual_float_to_uint8(img: np.ndarray) -> np.ndarray:
    """Shared visual 8-bit conversion for preview, PNG and JPEG export."""
    return np.clip(
        np.rint(np.clip(np.asarray(img, dtype=np.float32), 0.0, 1.0) * 255.0),
        0,
        255,
    ).astype(np.uint8)


def visual_float_to_uint16(img: np.ndarray) -> np.ndarray:
    """Shared visual 16-bit conversion for TIFF export."""
    return np.clip(
        np.rint(np.clip(np.asarray(img, dtype=np.float32), 0.0, 1.0) * 65535.0),
        0,
        65535,
    ).astype(np.uint16)


def make_histogram_pixmap(img: np.ndarray, width: int = 280, height: int = 120) -> QPixmap:
    """Create a compact LRGB histogram pixmap from the currently displayed preview."""
    arr = np.asarray(img, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim != 3 or arr.shape[2] != 3 or arr.size == 0:
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        qimg = QImage(canvas.data, width, height, width * 3, QImage.Format_RGB888).copy()
        return QPixmap.fromImage(qimg)

    step = max(1, int(math.sqrt(max(1, arr.shape[0] * arr.shape[1] // 250000))))
    sample = np.clip(arr[::step, ::step], 0, 1)
    lum = 0.2126 * sample[..., 0] + 0.7152 * sample[..., 1] + 0.0722 * sample[..., 2]

    channels = [
        (lum.ravel(), (235, 235, 235)),
        (sample[..., 0].ravel(), (255, 70, 70)),
        (sample[..., 1].ravel(), (80, 230, 100)),
        (sample[..., 2].ravel(), (90, 150, 255)),
    ]
    bins = 256
    hists = []
    max_count = 1.0
    for values, _color in channels:
        hist, _ = np.histogram(values, bins=bins, range=(0.0, 1.0))
        hist = np.log1p(hist.astype(np.float32))
        hists.append(hist)
        max_count = max(max_count, float(hist.max()))

    canvas = np.full((height, width, 3), 18, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), (70, 70, 70), 1)
    axis_h = 14
    plot_bottom = height - axis_h
    tick_color = (82, 86, 94)
    grid_color = (34, 37, 42)
    text_color = (128, 134, 145)
    for pct in (0, 25, 50, 75, 100):
        x = int(round(2 + (width - 5) * (pct / 100.0)))
        cv2.line(canvas, (x, 2), (x, plot_bottom - 1), grid_color, 1)
        cv2.line(canvas, (x, plot_bottom), (x, plot_bottom + 4), tick_color, 1)
        label = str(pct)
        text_size, _baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
        text_x = max(2, min(width - text_size[0] - 2, x - text_size[0] // 2))
        cv2.putText(canvas, label, (text_x, height - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_color, 1, cv2.LINE_AA)
    cv2.line(canvas, (2, plot_bottom), (width - 3, plot_bottom), tick_color, 1)

    plot_h = max(10, plot_bottom - 6)
    for hist, (_values, color) in zip(hists, channels):
        xs = np.linspace(2, width - 3, bins).astype(np.int32)
        ys = plot_bottom - 2 - np.clip((hist / max_count) * plot_h, 0, plot_h).astype(np.int32)
        pts = np.column_stack([xs, ys]).reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts], False, color, 1, cv2.LINE_AA)

    qimg = QImage(canvas.data, width, height, width * 3, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


def apply_dark_theme(app: QApplication) -> None:
    app.setStyleSheet(
        """
        QMainWindow, QWidget {
            background-color: #202124;
            color: #e8eaed;
        }
        QMenuBar, QMenu {
            background-color: #24262a;
            color: #e8eaed;
            border: 1px solid #3a3d42;
        }
        QMenuBar::item:selected, QMenu::item:selected {
            background-color: #343842;
        }
        QFrame {
            background-color: #24262a;
            border: 1px solid #3a3d42;
        }
        QLabel {
            color: #e8eaed;
            border: none;
        }
        QPushButton {
            background-color: #30343b;
            color: #f1f3f4;
            border: 1px solid #4a4f58;
            border-radius: 4px;
            padding: 5px 8px;
        }
        QPushButton:hover {
            background-color: #3a404a;
            border-color: #6b7280;
        }
        QPushButton:pressed {
            background-color: #4a5568;
        }
        QPushButton:disabled {
            background-color: #26282c;
            color: #777c85;
            border-color: #36393f;
        }
        QComboBox, QSpinBox, QDoubleSpinBox {
            background-color: #191b1f;
            color: #f1f3f4;
            border: 1px solid #4a4f58;
            border-radius: 4px;
            padding: 3px 26px 3px 6px;
        }
        QComboBox::drop-down {
            border-left: 1px solid #4a4f58;
            background-color: #2b3038;
            border-top-right-radius: 4px;
            border-bottom-right-radius: 4px;
            width: 22px;
        }
        QComboBox::down-arrow {
            width: 0;
            height: 0;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 6px solid #e8eaed;
            margin-right: 6px;
        }
        QSpinBox::up-button, QSpinBox::down-button,
        QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
            subcontrol-origin: border;
            width: 22px;
            background-color: #2b3038;
            border-left: 1px solid #4a4f58;
        }
        QSpinBox::up-button, QDoubleSpinBox::up-button {
            subcontrol-position: top right;
            border-top-right-radius: 4px;
        }
        QSpinBox::down-button, QDoubleSpinBox::down-button {
            subcontrol-position: bottom right;
            border-bottom-right-radius: 4px;
        }
        QSpinBox::up-button:hover, QSpinBox::down-button:hover,
        QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {
            background-color: #3a404a;
        }
        QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
            image: none;
            width: 9px;
            height: 9px;
        }
        QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
            image: none;
            width: 9px;
            height: 9px;
        }
        QComboBox QAbstractItemView {
            background-color: #1f2228;
            color: #f1f3f4;
            selection-background-color: #3b4a63;
            border: 1px solid #4a4f58;
        }
        QCheckBox {
            color: #e8eaed;
            spacing: 6px;
        }
        QCheckBox::indicator {
            width: 15px;
            height: 15px;
            border: 1px solid #8a93a3;
            border-radius: 3px;
            background-color: #101216;
        }
        QCheckBox::indicator:hover {
            border-color: #c3d4ff;
            background-color: #1b2230;
        }
        QCheckBox::indicator:checked {
            background-color: #5f8cff;
            border-color: #c3d4ff;
        }
        QCheckBox::indicator:checked:hover {
            background-color: #7aa2ff;
        }
        QSlider::groove:horizontal {
            height: 5px;
            background: #3a3d42;
            border-radius: 2px;
        }
        QSlider::handle:horizontal {
            background: #9ab4f8;
            border: 1px solid #c3d4ff;
            width: 14px;
            margin: -5px 0;
            border-radius: 7px;
        }
        QProgressBar {
            background-color: #191b1f;
            color: #f1f3f4;
            border: 1px solid #4a4f58;
            border-radius: 4px;
            text-align: center;
        }
        QProgressBar::chunk {
            background-color: #5f8cff;
            border-radius: 3px;
        }
        QScrollArea {
            background-color: #202124;
            border: none;
        }
        QScrollBar:vertical, QScrollBar:horizontal {
            background: #111318;
            border: 1px solid #343943;
            margin: 0;
            width: 14px;
            height: 14px;
        }
        QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
            background: #687386;
            border: 1px solid #9aa6bb;
            border-radius: 6px;
            min-height: 24px;
            min-width: 24px;
        }
        QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {
            background: #8fa3c4;
            border-color: #c3d4ff;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
            background: #22262e;
            border: 1px solid #3e4652;
            width: 14px;
            height: 14px;
        }
        QScrollBar::add-line:vertical:hover, QScrollBar::sub-line:vertical:hover,
        QScrollBar::add-line:horizontal:hover, QScrollBar::sub-line:horizontal:hover {
            background: #3a4454;
        }
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical,
        QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
            background: #151820;
        }
        QMessageBox {
            background-color: #24262a;
            color: #e8eaed;
        }
        """
    )


def apply_light_theme(app: QApplication) -> None:
    app.setStyleSheet(
        """
        QMainWindow, QWidget {
            background-color: #d9dde3;
            color: #1f2933;
        }
        QMenuBar, QMenu {
            background-color: #e2e5ea;
            color: #1f2933;
            border: 1px solid #c8ced8;
        }
        QMenuBar::item:selected, QMenu::item:selected {
            background-color: #dbeafe;
        }
        QFrame {
            background-color: #e4e7ec;
            border: 1px solid #c8ced8;
        }
        QLabel {
            color: #1f2933;
            border: none;
            background-color: transparent;
        }
        QPushButton {
            background-color: #edf0f4;
            color: #111827;
            border: 1px solid #aab4c2;
            border-radius: 4px;
            padding: 5px 8px;
        }
        QPushButton:hover {
            background-color: #e8f0fe;
            border-color: #5b7cba;
        }
        QPushButton:pressed {
            background-color: #dbeafe;
        }
        QPushButton:disabled {
            background-color: #e5e7eb;
            color: #8a94a3;
            border-color: #c8ced8;
        }
        QComboBox, QSpinBox, QDoubleSpinBox {
            background-color: #eef1f5;
            color: #111827;
            border: 1px solid #9aa6b8;
            border-radius: 4px;
            padding: 3px 26px 3px 6px;
        }
        QComboBox::drop-down {
            border-left: 1px solid #9aa6b8;
            background-color: #e8eef7;
            border-top-right-radius: 4px;
            border-bottom-right-radius: 4px;
            width: 22px;
        }
        QComboBox::down-arrow {
            width: 0;
            height: 0;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 6px solid #1f2933;
            margin-right: 6px;
        }
        QSpinBox::up-button, QSpinBox::down-button,
        QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
            subcontrol-origin: border;
            width: 22px;
            background-color: #e8eef7;
            border-left: 1px solid #9aa6b8;
        }
        QSpinBox::up-button, QDoubleSpinBox::up-button {
            subcontrol-position: top right;
            border-top-right-radius: 4px;
        }
        QSpinBox::down-button, QDoubleSpinBox::down-button {
            subcontrol-position: bottom right;
            border-bottom-right-radius: 4px;
        }
        QSpinBox::up-button:hover, QSpinBox::down-button:hover,
        QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {
            background-color: #dbeafe;
        }
        QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
            image: none;
            width: 9px;
            height: 9px;
        }
        QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
            image: none;
            width: 9px;
            height: 9px;
        }
        QComboBox QAbstractItemView {
            background-color: #eef1f5;
            color: #111827;
            selection-background-color: #cfe1ff;
            border: 1px solid #9aa6b8;
        }
        QCheckBox {
            color: #1f2933;
            spacing: 6px;
        }
        QCheckBox::indicator {
            width: 15px;
            height: 15px;
            border: 1px solid #6b7280;
            border-radius: 3px;
            background-color: #eef1f5;
        }
        QCheckBox::indicator:hover {
            border-color: #2563eb;
            background-color: #eff6ff;
        }
        QCheckBox::indicator:checked {
            background-color: #2563eb;
            border-color: #1d4ed8;
        }
        QSlider::groove:horizontal {
            height: 5px;
            background: #c8ced8;
            border-radius: 2px;
        }
        QSlider::handle:horizontal {
            background: #2563eb;
            border: 1px solid #1d4ed8;
            width: 14px;
            margin: -5px 0;
            border-radius: 7px;
        }
        QProgressBar {
            background-color: #eef1f5;
            color: #111827;
            border: 1px solid #9aa6b8;
            border-radius: 4px;
            text-align: center;
        }
        QProgressBar::chunk {
            background-color: #2563eb;
            border-radius: 3px;
        }
        QScrollArea {
            background-color: #cbd1da;
            border: none;
        }
        QScrollArea > QWidget > QWidget {
            background-color: #cbd1da;
        }
        QScrollBar:vertical, QScrollBar:horizontal {
            background: #cbd1da;
            border: 1px solid #aab4c2;
            margin: 0;
            width: 14px;
            height: 14px;
        }
        QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
            background: #7b8798;
            border: 1px solid #4b5563;
            border-radius: 6px;
            min-height: 24px;
            min-width: 24px;
        }
        QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {
            background: #4f6f9f;
            border-color: #1d4ed8;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
            background: #cbd1da;
            border: none;
            width: 0;
            height: 0;
        }
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical,
        QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
            background: #cbd1da;
        }
        QMessageBox {
            background-color: #e4e7ec;
            color: #1f2933;
        }
        """
    )



def init_process_single_image_mp(context: Dict[str, Any]) -> None:
    """Initialize shared worker state once per spawned process."""
    global MP_WORKER_CONTEXT
    cv2.setNumThreads(1)
    MP_WORKER_CONTEXT = dict(context or {})


def process_single_image_mp(args: Dict[str, Any]) -> Optional[Tuple[Any, ...]]:
    """Worker funkce pro multiprocessing. Musí být top-level kvůli Windows/macOS spawn režimu."""
    cv2.setNumThreads(1)

    context = MP_WORKER_CONTEXT
    path = Path(args["path"])
    reference_path = Path(args.get("reference_path", context["reference_path"]))
    reference = args.get("reference", context["reference"])
    reference_gray = args.get("reference_gray", context["reference_gray"])
    reference_median = args.get("reference_median", context["reference_median"])
    settings = args.get("settings", context["settings"])
    set_bayer_pattern_override(getattr(settings, "bayer_pattern", "auto"))
    h, w = reference.shape[:2]
    predicted_xy = args.get("manual_comet_predicted_xy")

    if path != reference_path and settings.align_mode != "calibration":
        cached_warped = load_aligned_frame_cache(path, reference_path, settings, predicted_xy)
        if cached_warped is not None and cached_warped.shape[:2] == (h, w):
            return str(path), np.clip(cached_warped, 0, 1).astype(np.float32)

    img = load_stack_frame_for_settings(
        path,
        settings,
        args.get("flat_frame", context.get("flat_frame")),
        args.get("bias_frame", context.get("bias_frame")),
        args.get("dark_frame", context.get("dark_frame")),
    )
    if img.shape[:2] != (h, w):
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    trail_mask = None
    satellite_trail_paths = set(context.get("satellite_trail_paths", ()))
    if (
        bool(getattr(settings, "satellite_trail_filter", False))
        and settings.align_mode != "calibration"
        and str(path.resolve()) in satellite_trail_paths
    ):
        trail_mask = satellite_trail_mask_from_gray(to_gray_float(img), confirmed=True)
        if trail_mask is not None:
            log_debug(f"Satellite trail masked: {path}")

    if path == reference_path or settings.align_mode == "calibration":
        # Kalibrační snímky skládáme pixel na pixel bez zarovnání.
        warped = img
        matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    else:
        moving_gray = to_gray_float(img)
        if settings.align_mode == "comet":
            if settings.manual_comet_xy is not None and predicted_xy is not None:
                matrix = comet_alignment_matrix_with_optional_refine(reference_gray, moving_gray, settings, predicted_xy)
            else:
                matrix = estimate_comet_translation(reference_gray, moving_gray, settings.downscale_for_alignment, settings.star_border_margin, settings.max_comet_shift, settings.manual_comet_xy)
        elif settings.align_mode == "star_affine":
            matrix = estimate_star_affine(reference_gray, moving_gray, settings.downscale_for_alignment, settings.max_star_shift, settings.star_border_margin, settings.strict_star_filter, allow_fallback=False)
            if matrix is None:
                return None
        elif settings.align_mode == "ecc_affine":
            matrix = estimate_ecc_affine(reference_gray, moving_gray, settings.downscale_for_alignment)
        else:
            matrix = estimate_translation(reference_gray, moving_gray, settings.downscale_for_alignment)
        if getattr(settings, "mosaic_mode", False):
            warped = img
        else:
            warped = warp_to_reference(img, matrix, (h, w))
            warped = mark_invalid_warp_pixels(warped, matrix, (h, w))
    if not getattr(settings, "mosaic_mode", False):
        warped = apply_warped_exclusion_mask(warped, trail_mask, matrix, (h, w))

    if settings.normalize_background and settings.align_mode != "calibration" and not getattr(settings, "mosaic_mode", False):
        warped = normalize_background(warped, reference_median)
    if getattr(settings, "mosaic_mode", False):
        if settings.normalize_background and settings.align_mode != "calibration":
            img = normalize_background(img, reference_median)
        return str(path), np.clip(img, 0, 1).astype(np.float32), np.asarray(matrix, dtype=np.float32), trail_mask
    if path != reference_path and settings.align_mode != "calibration":
        save_aligned_frame_cache(path, reference_path, settings, predicted_xy, warped)

    return str(path), np.clip(warped, 0, 1).astype(np.float32)


def stack_folder_multiprocessing(folder: Path, settings: StackSettings, processes: int, progress_callback=None) -> Optional[np.ndarray]:
    CALIBRATION_SIGNATURE_CACHE.clear()
    set_bayer_pattern_override(getattr(settings, "bayer_pattern", "auto"))
    if settings.align_mode == "calibration":
        flat_frame, bias_frame, dark_frame = None, None, None
    else:
        flat_frame, bias_frame, dark_frame = load_calibration_frames(settings, progress_callback)
    if settings.align_mode == "comet_merge":
        stack_star_and_comet_two_outputs(
            folder,
            settings,
            use_multiprocessing=True,
            processes=processes,
            progress_callback=progress_callback,
        )
        return None

    paths, reference_path, scores, sequence_paths = prepare_paths_for_alignment_mode(folder, settings, progress_callback)
    satellite_trail_paths = selected_satellite_trail_paths()
    if bool(getattr(settings, "satellite_trail_filter", False)):
        LAST_STACK_SELECTION["satellite_masked_paths"] = sorted(satellite_trail_paths)

    cv2.setNumThreads(1)

    if settings.align_mode != "calibration":
        report_bayer_conversion_if_needed(reference_path, progress_callback, 0)
    reference = load_stack_frame_for_settings(reference_path, settings, flat_frame, bias_frame, dark_frame)
    h, w = reference.shape[:2]
    reference_gray = to_gray_float(reference)
    reference_median = np.median(reference.reshape(-1, 3), axis=0) if reference.ndim == 3 else float(np.median(reference))

    total = len(paths)
    requested_processes = max(1, min(int(processes), total))
    processes = limit_processes_for_memory(processes, total, reference, progress_callback)
    if progress_callback and processes > 1:
        if processes == requested_processes:
            progress_callback(5, f"CPU alignment procesy: {processes}")
        else:
            progress_callback(5, f"CPU alignment procesy: {requested_processes} -> {processes}")

    if progress_callback:
        bayer_patterns = sorted(
            {
                pattern
                for path in paths
                for pattern in [bayer_pattern_for_fits_path(path)]
                if pattern
            }
        )
        bayer_suffix = f", Bayer {'/'.join(bayer_patterns)}" if bayer_patterns else ""
        progress_callback(5, f"Načítám a zarovnávám snímky paralelně{bayer_suffix}...")

    worker_context = {
        "reference_path": str(reference_path),
        "reference": reference,
        "reference_gray": reference_gray,
        "reference_median": reference_median,
        "settings": settings,
        "flat_frame": flat_frame,
        "bias_frame": bias_frame,
        "dark_frame": dark_frame,
        "satellite_trail_paths": tuple(sorted(satellite_trail_paths)),
    }
    tasks = [
        {
            "path": str(path),
            "manual_comet_predicted_xy": interpolate_manual_comet_position_for_path(path, sequence_paths, settings) if settings.align_mode == "comet" else None,
        }
        for path in paths
    ]

    aligned: List[np.ndarray] = []
    mosaic_records: List[Tuple[Path, np.ndarray, np.ndarray, Optional[np.ndarray]]] = []
    used_paths: List[Path] = []

    # Chunksize 1 dává plynulejší progress bar; pro velmi mnoho snímků lze zvýšit.
    with mp.Pool(processes=processes, initializer=init_process_single_image_mp, initargs=(worker_context,)) as pool:
        for idx, item in enumerate(pool.imap(process_single_image_mp, tasks, chunksize=1)):
            name = paths[idx].name
            if item is None:
                if progress_callback:
                    progress_callback(10 + int((idx + 1) / total * 60), f"Vyřazuji bez platného star alignmentu ({idx + 1}/{total}): {name}")
                continue
            if getattr(settings, "mosaic_mode", False):
                used_path, img, matrix, trail_mask = item
                mosaic_records.append((Path(used_path), img, matrix, trail_mask))
            else:
                used_path, warped = item
                aligned.append(warped)
            used_paths.append(Path(used_path))
            if progress_callback:
                progress_callback(10 + int((idx + 1) / total * 60), f"Zarovnáno paralelně ({idx + 1}/{total}): {name}")
        if progress_callback:
            progress_callback(72, "Dokoncuji paralelni alignment...")

    if getattr(settings, "mosaic_mode", False):
        aligned = warp_mosaic_records(mosaic_records, (h, w), progress_callback)

    if not aligned:
        raise ValueError("Žádný snímek neprošel zarovnáním. Zkontroluj, zda složka Light neobsahuje Dark/Bias snímky nebo zda jsou ve snímcích detekovatelné hvězdy.")
    refresh_last_stack_selection_after_alignment(used_paths)
    result = stack_aligned_frames(aligned, settings, progress_callback)

    if progress_callback:
        progress_callback(100, "Hotovo")
    return np.clip(result, 0, 1)



class ClickableImageLabel(QLabel):
    """QLabel, který umí vrátit souřadnici kliknutí uvnitř zobrazeného pixmapu.

    Emituje souřadnice v pixmapu, ne v původním snímku. Převod na původní
    obraz řeší hlavní okno podle aktuálního měřítka náhledu.
    """
    image_clicked = Signal(float, float)
    crop_selected = Signal(float, float, float, float)
    image_double_clicked = Signal()
    wheel_zoomed = Signal(float, float, float)
    drag_started = Signal(float, float)
    drag_moved = Signal(float, float)
    drag_finished = Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._press_pos = None
        self._last_drag_pos = None
        self._last_drag_global = None
        self._dragging = False
        self._marking_mode = False
        self._crop_selection_mode = False
        self._crop_start_pos = None
        self._crop_current_pos = None
        self._overlay_markers = []
        self._overlay_image_size = (1, 1)
        self.setCursor(Qt.OpenHandCursor)

    def set_overlay_markers(self, markers, image_size):
        self._overlay_markers = list(markers or [])
        self._overlay_image_size = (
            max(1, int(image_size[0])),
            max(1, int(image_size[1])),
        )
        self.update()

    def _interaction_cursor(self):
        if self._marking_mode or self._crop_selection_mode:
            return Qt.CrossCursor
        return Qt.OpenHandCursor

    def set_marking_mode(self, enabled: bool):
        self._marking_mode = bool(enabled)
        if self._marking_mode:
            self._crop_selection_mode = False
            self._crop_start_pos = None
            self._crop_current_pos = None
            self.update()
        self.setCursor(self._interaction_cursor())

    def set_crop_selection_mode(self, enabled: bool):
        self._crop_selection_mode = bool(enabled)
        if self._crop_selection_mode:
            self._marking_mode = False
        self._crop_start_pos = None
        self._crop_current_pos = None
        self.setCursor(self._interaction_cursor())
        self.update()

    def _widget_to_pixmap_point(self, point, clamp: bool = False):
        pixmap = self.pixmap()
        if pixmap is None or pixmap.isNull():
            return None
        pw = float(pixmap.width())
        ph = float(pixmap.height())
        ox = max(0.0, (self.width() - pw) / 2.0)
        oy = max(0.0, (self.height() - ph) / 2.0)
        px = float(point.x()) - ox
        py = float(point.y()) - oy
        if clamp:
            px = max(0.0, min(pw, px))
            py = max(0.0, min(ph, py))
            return px, py
        if 0 <= px < pw and 0 <= py < ph:
            return px, py
        return None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self._crop_selection_mode:
                if self._widget_to_pixmap_point(event.position()) is None:
                    event.accept()
                    return
                self._crop_start_pos = event.position()
                self._crop_current_pos = event.position()
                self.update()
                event.accept()
                return
            self._press_pos = event.position()
            self._last_drag_pos = event.position()
            self._last_drag_global = event.globalPosition()
            self._dragging = False
            if self._marking_mode:
                self.setCursor(Qt.CrossCursor)
            else:
                self.setCursor(Qt.ClosedHandCursor)
                self.drag_started.emit(float(event.position().x()), float(event.position().y()))
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._crop_selection_mode and self._crop_start_pos is not None:
            self._crop_current_pos = event.position()
            self.update()
            event.accept()
            return
        if self._press_pos is not None and not self._marking_mode:
            dx = float(event.position().x() - self._press_pos.x())
            dy = float(event.position().y() - self._press_pos.y())
            if self._dragging or abs(dx) > 3 or abs(dy) > 3:
                self._dragging = True
                last_global = self._last_drag_global or event.globalPosition()
                step_dx = float(event.globalPosition().x() - last_global.x())
                step_dy = float(event.globalPosition().y() - last_global.y())
                self._last_drag_pos = event.position()
                self._last_drag_global = event.globalPosition()
                self.drag_moved.emit(step_dx, step_dy)
                event.accept()
                return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._crop_selection_mode:
            start_pos = self._crop_start_pos
            end_pos = event.position()
            self._crop_start_pos = None
            self._crop_current_pos = None
            start = self._widget_to_pixmap_point(start_pos, clamp=True) if start_pos is not None else None
            end = self._widget_to_pixmap_point(end_pos, clamp=True)
            self.set_crop_selection_mode(False)
            if start is not None and end is not None:
                self.crop_selected.emit(float(start[0]), float(start[1]), float(end[0]), float(end[1]))
            event.accept()
            return
        if event.button() == Qt.LeftButton and self._press_pos is not None:
            was_dragging = self._dragging
            press_pos = self._press_pos
            self._press_pos = None
            self._last_drag_pos = None
            self._last_drag_global = None
            self._dragging = False
            self.setCursor(self._interaction_cursor())
            if not self._marking_mode:
                self.drag_finished.emit()
            if was_dragging:
                event.accept()
                return

            pixmap = self.pixmap()
            if pixmap is not None and not pixmap.isNull():
                x = float(press_pos.x())
                y = float(press_pos.y())
                pw = float(pixmap.width())
                ph = float(pixmap.height())
                ox = max(0.0, (self.width() - pw) / 2.0)
                oy = max(0.0, (self.height() - ph) / 2.0)
                px = x - ox
                py = y - oy
                if 0 <= px < pw and 0 <= py < ph:
                    self.image_clicked.emit(px, py)
                    event.accept()
                    return
        self.setCursor(self._interaction_cursor())
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            pixmap = self.pixmap()
            if pixmap is not None and not pixmap.isNull():
                self.image_double_clicked.emit()
                event.accept()
                return
        super().mouseDoubleClickEvent(event)

    def leaveEvent(self, event):
        if not self._dragging:
            self.setCursor(self._interaction_cursor())
        super().leaveEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        pixmap = self.pixmap()
        if self._overlay_markers and pixmap is not None and not pixmap.isNull():
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing, True)
            pw = float(pixmap.width())
            ph = float(pixmap.height())
            ox = max(0.0, (float(self.width()) - pw) / 2.0)
            oy = max(0.0, (float(self.height()) - ph) / 2.0)
            image_w, image_h = self._overlay_image_size
            marker_scale = min(pw / float(image_w), ph / float(image_h))
            for nx, ny, color, outer_radius, inner_radius in self._overlay_markers:
                center = QPointF(ox + float(nx) * pw, oy + float(ny) * ph)
                painter.setPen(QPen(QColor.fromRgbF(*color), 1.0))
                painter.drawEllipse(center, max(7.0, outer_radius * marker_scale), max(7.0, outer_radius * marker_scale))
                painter.drawEllipse(center, max(3.5, inner_radius * marker_scale), max(3.5, inner_radius * marker_scale))

        if self._crop_selection_mode and self._crop_start_pos is not None and self._crop_current_pos is not None:
            rect = QRectF(self._crop_start_pos, self._crop_current_pos).normalized()
            if rect.width() >= 2 and rect.height() >= 2:
                painter = QPainter(self)
                painter.setRenderHint(QPainter.Antialiasing, False)
                painter.fillRect(rect, QColor(80, 160, 255, 45))
                painter.setPen(QPen(QColor(140, 205, 255), 2, Qt.DashLine))
                painter.drawRect(rect)

    def _legacy_mousePressEvent(self, event):
        pixmap = self.pixmap()
        if pixmap is not None and not pixmap.isNull():
            x = float(event.position().x())
            y = float(event.position().y())
            pw = float(pixmap.width())
            ph = float(pixmap.height())
            ox = max(0.0, (self.width() - pw) / 2.0)
            oy = max(0.0, (self.height() - ph) / 2.0)
            px = x - ox
            py = y - oy
            if 0 <= px < pw and 0 <= py < ph:
                self.image_clicked.emit(px, py)
                return
        super().mousePressEvent(event)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta == 0:
            super().wheelEvent(event)
            return
        factor = 1.25 if delta > 0 else 1.0 / 1.25
        self.wheel_zoomed.emit(factor, float(event.position().x()), float(event.position().y()))
        event.accept()


class FullscreenPreviewWindow(QWidget):
    """Celoobrazovkový náhled aktuálně zobrazeného snímku se zoomem a posunem."""

    def __init__(self, pixmap: QPixmap, parent=None):
        super().__init__(parent)
        self.original_pixmap = QPixmap(pixmap)
        self.zoom_factor = 1.0
        self.zoom_mode = "fit"
        self.pan_scroll = (0.0, 0.0)
        self.pan_velocity = (0.0, 0.0)
        self.setWindowTitle("AstroStacker Preview")
        self.setWindowFlag(Qt.Window, True)
        self.setStyleSheet("background: #050505;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        preview_stack = QWidget()
        preview_stack_layout = QGridLayout(preview_stack)
        preview_stack_layout.setContentsMargins(0, 0, 0, 0)
        preview_stack_layout.setSpacing(0)

        self.label = ClickableImageLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("background: #050505;")
        self.label.wheel_zoomed.connect(self.on_wheel_zoom)
        self.label.drag_started.connect(self.on_drag_started)
        self.label.drag_moved.connect(self.on_drag_moved)
        self.label.drag_finished.connect(self.on_drag_finished)
        self.label.image_double_clicked.connect(self.close)

        self.scroll = QScrollArea()
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setAlignment(Qt.AlignCenter)
        self.scroll.setWidget(self.label)
        preview_stack_layout.addWidget(self.scroll, 0, 0)

        self.zoom_overlay_timer = QTimer(self)
        self.zoom_overlay_timer.setSingleShot(True)
        self.zoom_overlay_timer.timeout.connect(self.hide_zoom_overlay)
        self.pan_inertia_timer = QTimer(self)
        self.pan_inertia_timer.setInterval(16)
        self.pan_inertia_timer.timeout.connect(self.run_pan_inertia)
        self.zoom_overlay = QLabel("")
        self.zoom_overlay.setStyleSheet(
            "background: rgba(0, 0, 0, 150); color: #f8fafc; "
            "border: 1px solid rgba(255, 255, 255, 90); border-radius: 4px; "
            "padding: 4px 8px; font-size: 13px; font-weight: bold;"
        )
        self.zoom_overlay.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.zoom_overlay.hide()
        preview_stack_layout.addWidget(self.zoom_overlay, 0, 0, Qt.AlignTop | Qt.AlignLeft)
        layout.addWidget(preview_stack)
        self.update_pixmap()

    def fit_zoom_factor(self) -> float:
        if self.original_pixmap.isNull():
            return 1.0
        viewport = self.scroll.viewport().size()
        if viewport.width() <= 1 or viewport.height() <= 1:
            return 1.0
        return min(
            float(viewport.width()) / max(1.0, float(self.original_pixmap.width())),
            float(viewport.height()) / max(1.0, float(self.original_pixmap.height())),
            1.0,
        )

    def update_pixmap(self):
        if self.original_pixmap.isNull():
            return
        if self.zoom_mode == "fit":
            self.scroll.setWidgetResizable(True)
            scale = self.fit_zoom_factor()
        else:
            self.scroll.setWidgetResizable(False)
            scale = self.zoom_factor

        width = max(1, int(round(self.original_pixmap.width() * scale)))
        height = max(1, int(round(self.original_pixmap.height() * scale)))
        pixmap = self.original_pixmap.scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.label.setPixmap(pixmap)
        if self.zoom_mode == "fit":
            self.label.setMinimumSize(1, 1)
        else:
            self.label.resize(pixmap.size())
            self.label.setMinimumSize(pixmap.size())

    def on_wheel_zoom(self, factor: float, widget_x: float, widget_y: float):
        self.stop_pan_inertia()
        pixmap = self.label.pixmap()
        if pixmap is None or pixmap.isNull():
            return

        hbar = self.scroll.horizontalScrollBar()
        vbar = self.scroll.verticalScrollBar()
        ox = max(0.0, (self.label.width() - pixmap.width()) / 2.0)
        oy = max(0.0, (self.label.height() - pixmap.height()) / 2.0)
        pix_x = max(0.0, min(float(pixmap.width()), float(widget_x) - ox))
        pix_y = max(0.0, min(float(pixmap.height()), float(widget_y) - oy))
        rel_x = pix_x / max(1.0, float(pixmap.width()))
        rel_y = pix_y / max(1.0, float(pixmap.height()))
        viewport_pos = self.label.mapTo(self.scroll.viewport(), QPoint(int(round(widget_x)), int(round(widget_y))))
        viewport_x = float(viewport_pos.x())
        viewport_y = float(viewport_pos.y())

        old_zoom = self.fit_zoom_factor() if self.zoom_mode == "fit" else self.zoom_factor
        self.zoom_mode = "fixed"
        self.zoom_factor = max(0.05, min(12.0, old_zoom * factor))
        self.update_pixmap()

        new_pixmap = self.label.pixmap()
        if new_pixmap is None or new_pixmap.isNull():
            return
        new_ox = max(0.0, (self.label.width() - new_pixmap.width()) / 2.0)
        new_oy = max(0.0, (self.label.height() - new_pixmap.height()) / 2.0)
        target_x = new_ox + rel_x * float(new_pixmap.width()) - viewport_x
        target_y = new_oy + rel_y * float(new_pixmap.height()) - viewport_y

        def apply_anchor():
            hbar.setValue(int(round(target_x)))
            vbar.setValue(int(round(target_y)))

        apply_anchor()
        QTimer.singleShot(0, apply_anchor)
        self.show_zoom_overlay()

    def show_zoom_overlay(self):
        text = "Fit" if self.zoom_mode == "fit" else f"{int(round(self.zoom_factor * 100))} %"
        self.zoom_overlay.setText(text)
        self.zoom_overlay.adjustSize()
        self.zoom_overlay.show()
        self.zoom_overlay.raise_()
        self.zoom_overlay_timer.start(1800)

    def hide_zoom_overlay(self):
        self.zoom_overlay.hide()

    def on_drag_started(self, _x: float, _y: float):
        self.stop_pan_inertia()
        self.pan_velocity = (0.0, 0.0)
        self.pan_scroll = (
            float(self.scroll.horizontalScrollBar().value()),
            float(self.scroll.verticalScrollBar().value()),
        )

    def on_drag_moved(self, dx: float, dy: float):
        hbar = self.scroll.horizontalScrollBar()
        vbar = self.scroll.verticalScrollBar()
        pan_x, pan_y = self.pan_scroll
        step_x = -float(dx)
        step_y = -float(dy)
        pan_x += step_x
        pan_y += step_y
        old_vx, old_vy = self.pan_velocity
        self.pan_velocity = (
            old_vx * 0.35 + step_x * 0.65,
            old_vy * 0.35 + step_y * 0.65,
        )
        self.pan_scroll = (pan_x, pan_y)
        hbar.setValue(int(round(pan_x)))
        vbar.setValue(int(round(pan_y)))

    def on_drag_finished(self):
        self.pan_scroll = (
            float(self.scroll.horizontalScrollBar().value()),
            float(self.scroll.verticalScrollBar().value()),
        )
        self.start_pan_inertia()

    def start_pan_inertia(self):
        if self.zoom_mode != "fixed":
            return
        vx, vy = self.pan_velocity
        speed = math.hypot(vx, vy)
        if speed < 1.2:
            self.pan_velocity = (0.0, 0.0)
            return
        max_step = 34.0
        if speed > max_step:
            scale = max_step / speed
            self.pan_velocity = (vx * scale, vy * scale)
        self.pan_inertia_timer.start()

    def stop_pan_inertia(self):
        if hasattr(self, "pan_inertia_timer"):
            self.pan_inertia_timer.stop()
        self.pan_velocity = (0.0, 0.0)

    def run_pan_inertia(self):
        if self.zoom_mode != "fixed":
            self.stop_pan_inertia()
            return
        vx, vy = self.pan_velocity
        if math.hypot(vx, vy) < 0.35:
            self.stop_pan_inertia()
            return

        hbar = self.scroll.horizontalScrollBar()
        vbar = self.scroll.verticalScrollBar()
        old_h = hbar.value()
        old_v = vbar.value()
        hbar.setValue(int(round(old_h + vx)))
        vbar.setValue(int(round(old_v + vy)))
        new_h = hbar.value()
        new_v = vbar.value()
        self.pan_scroll = (float(new_h), float(new_v))

        hit_x_edge = new_h == old_h and abs(vx) > 0.01
        hit_y_edge = new_v == old_v and abs(vy) > 0.01
        vx = 0.0 if hit_x_edge else vx * 0.88
        vy = 0.0 if hit_y_edge else vy * 0.88
        self.pan_velocity = (vx, vy)

    def resizeEvent(self, event):
        self.stop_pan_inertia()
        if self.zoom_mode == "fit":
            self.update_pixmap()
        super().resizeEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
            event.accept()
            return
        super().keyPressEvent(event)

    def mouseDoubleClickEvent(self, event):
        self.stop_pan_inertia()
        self.close()
        event.accept()


class NumericTableWidgetItem(QTableWidgetItem):
    def __lt__(self, other):
        left = self.data(Qt.UserRole)
        right = other.data(Qt.UserRole) if other is not None else None
        if left is not None and right is not None:
            try:
                return float(left) < float(right)
            except Exception:
                pass
        return super().__lt__(other)


class FrameQualityTable(QTableWidget):
    toggle_requested = Signal()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.toggle_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class AutoStackCalibrationWorker(QThread):
    """Prepare manual or automatic calibration masters once per live session."""

    progress = Signal(str)
    masters_ready = Signal(object, object, object)
    failed = Signal(str)

    def __init__(self, settings: StackSettings):
        super().__init__()
        self.settings = settings

    def run(self):
        try:
            def report(_value: int, message: str):
                if self.isInterruptionRequested():
                    raise ProcessingCancelled()
                self.progress.emit(str(message))

            flat, bias, dark = load_calibration_frames(self.settings, report)
            if self.isInterruptionRequested():
                return
            self.masters_ready.emit(flat, bias, dark)
        except ProcessingCancelled:
            return
        except Exception as exc:
            self.failed.emit(str(exc))


class AutoStackFrameWorker(QThread):
    """Load and align one newly completed frame without blocking the GUI."""

    frame_ready = Signal(str, object, object, object)
    failed = Signal(str, str)

    def __init__(
        self,
        path: Path,
        settings: StackSettings,
        reference_gray: Optional[np.ndarray],
        reference_shape: Optional[Tuple[int, ...]],
        flat: Optional[np.ndarray] = None,
        bias: Optional[np.ndarray] = None,
        dark: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.path = Path(path)
        self.settings = settings
        self.reference_gray = reference_gray
        self.reference_shape = reference_shape
        self.flat = flat
        self.bias = bias
        self.dark = dark

    def run(self):
        try:
            if self.isInterruptionRequested():
                return
            set_bayer_pattern_override(getattr(self.settings, "bayer_pattern", "auto"))
            image = np.ascontiguousarray(
                load_calibrated_image_as_float(
                    self.path,
                    self.flat,
                    self.bias,
                    self.dark,
                ).astype(np.float32)
            )
            if image.ndim not in (2, 3):
                raise ValueError(f"Unsupported image shape: {image.shape}")
            if image.ndim == 3 and image.shape[2] not in (1, 3):
                image = np.ascontiguousarray(image[..., :3])

            valid = np.all(np.isfinite(image), axis=2) if image.ndim == 3 else np.isfinite(image)
            image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

            if self.reference_gray is None:
                coverage = valid.astype(np.float32)
                self.frame_ready.emit(
                    str(self.path),
                    image,
                    coverage,
                    {"status": "reference"},
                )
                return

            if self.reference_shape is None or tuple(image.shape) != tuple(self.reference_shape):
                raise ValueError(
                    f"Frame dimensions differ from the reference: {image.shape} != {self.reference_shape}"
                )
            if self.isInterruptionRequested():
                return

            moving_gray = to_gray_float(image)
            matrix, status, detail = estimate_star_affine_detailed(
                self.reference_gray,
                moving_gray,
                float(getattr(self.settings, "downscale_for_alignment", 0.5)),
                int(getattr(self.settings, "max_star_shift", 180)),
                int(getattr(self.settings, "star_border_margin", 40)),
                bool(getattr(self.settings, "strict_star_filter", True)),
            )
            if status != "ransac":
                raise ValueError(f"RANSAC alignment failed ({status})")
            if self.isInterruptionRequested():
                return

            height, width = self.reference_gray.shape[:2]
            aligned = cv2.warpAffine(
                image,
                matrix.astype(np.float32),
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            ).astype(np.float32)
            coverage = cv2.warpAffine(
                valid.astype(np.float32),
                matrix.astype(np.float32),
                (width, height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            coverage = (coverage > 0.5).astype(np.float32)
            detail = dict(detail or {})
            detail["status"] = status
            self.frame_ready.emit(str(self.path), aligned, coverage, detail)
        except Exception as exc:
            self.failed.emit(str(self.path), str(exc))


class StackWorker(QThread):
    progress = Signal(int, str)
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, folder: Path, settings: StackSettings, use_multiprocessing: bool = False, processes: int = 1):
        super().__init__()
        self.folder = folder
        self.settings = settings
        self.use_multiprocessing = use_multiprocessing
        self.processes = processes

    def _progress(self, value: int, message: str):
        if self.isInterruptionRequested():
            raise ProcessingCancelled()
        if getattr(self.settings, "language", "cz") == "en":
            message = self._translate_progress_message(message)
        self.progress.emit(value, message)

    def _translate_progress_message(self, message: str) -> str:
        replacements = [
            ("Kvalita z cache", "Quality from cache"),
            ("Hodnotím kvalitu paralelně", "Scoring quality in parallel"),
            ("Hodnotim kvalitu paralelne", "Scoring quality in parallel"),
            ("Hodnotím kvalitu", "Scoring quality"),
            ("Hodnotim kvalitu", "Scoring quality"),
            ("Zarovnáno z cache", "Aligned from cache"),
            ("Zarovnano z cache", "Aligned from cache"),
            ("Zarovnáno paralelně", "Aligned in parallel"),
            ("Zarovnáno hvězdně", "Star aligned"),
            ("Zarovnáno", "Aligned"),
            ("Zarovnávám paralelně", "Aligning in parallel"),
            ("Zarovnavam paralelne", "Aligning in parallel"),
            ("Dokoncuji paralelni alignment", "Finishing parallel alignment"),
            ("Zarovnávám hvězdy", "Aligning stars"),
            ("Zarovnávám", "Aligning"),
            ("Zarovnavam", "Aligning"),
            ("Skládám snímky na GPU", "Stacking frames on GPU"),
            ("Skladam snimky na GPU", "Stacking frames on GPU"),
            ("Posilam stack po blocich primo do GPU", "Sending stack tiles directly to GPU"),
            ("Skladam na GPU po blocich VRAM", "Stacking on GPU in VRAM tiles"),
            ("Skladam na GPU po blocich sdilene pameti", "Stacking on GPU in shared-memory tiles"),
            ("Apple Metal/MPS neni dostupne", "Apple Metal/MPS is not available"),
            ("PyTorch/MPS nelze nacist", "PyTorch/MPS cannot be loaded"),
            ("CUDA/CuPy neni dostupne", "CUDA/CuPy is not available"),
            ("Pripravuji stack v RAM", "Preparing stack in RAM"),
            ("Skladam na CPU po blocich RAM", "Stacking on CPU in RAM tiles"),
            ("Skladam Bias master na CPU po blocich RAM", "Stacking Bias master on CPU in RAM tiles"),
            ("Skladam Flat master na CPU po blocich RAM", "Stacking Flat master on CPU in RAM tiles"),
            ("Skladam Dark master na CPU po blocich RAM", "Stacking Dark master on CPU in RAM tiles"),
            ("Skládám ručně vybranou složku", "Stacking manually selected folder"),
            ("Nacitam MasterBias z cache", "Loading MasterBias from cache"),
            ("Nacitam MasterFlat z cache", "Loading MasterFlat from cache"),
            ("Nacitam MasterDark z cache", "Loading MasterDark from cache"),
            ("Skladam prumer na CPU po blocich RAM", "Stacking mean on CPU in RAM tiles"),
            ("Skladam prumer prubezne", "Stacking mean incrementally"),
            ("CPU fallback skladam po castech", "CPU fallback stacking in tiles"),
            ("CPU alignment procesy", "CPU alignment processes"),
            ("CPU procesy omezeny kvuli RAM", "CPU processes limited because of RAM"),
            ("RAM ochrana", "RAM protection"),
            ("skladam po castech", "stacking in tiles"),
            ("Skladam snimky po castech", "Stacking frames in tiles"),
            ("nedostatek pameti pro cely stack", "not enough memory for the full stack"),
            ("odhad", "estimated"),
            ("volne", "available"),
            ("radky", "rows"),
            ("radku", "rows"),
            ("vlaken", "threads"),
            ("Časy", "Times"),
            ("Načítám a zarovnávám snímky paralelně", "Loading and aligning frames in parallel"),
            ("Načítám a zarovnávám snímky", "Loading and aligning frames"),
            ("Načítám/kalibruji/debayeruji Bayer snímky", "Loading/calibrating/debayering Bayer frames"),
            ("Načítám/kalibruji/debayeruji Bayer", "Loading/calibrating/debayering Bayer"),
            ("debayering/načtení", "debayering/loading"),
            ("debayering+alignment", "debayering+alignment"),
            ("alignment", "alignment"),
            ("skládání", "stacking"),
            ("celkem", "total"),
            ("Skládám snímky", "Stacking frames"),
            ("Skladam snimky", "Stacking frames"),
            ("Skládám hvězdy", "Stacking stars"),
            ("skládám pouze hvězdy", "stacking stars only"),
            ("skládám pouze kometu", "stacking comet only"),
            ("Hvězdy", "Stars"),
            ("Kometa", "Comet"),
            ("Ukládám", "Saving"),
            ("uloženo", "saved"),
            ("uloženo:", "saved:"),
            ("Skládám", "Stacking"),
            ("Automatická kalibrace aktivní", "Automatic calibration active"),
            ("Načítám Flat Frame", "Loading Flat Frame"),
            ("Načítám Bias Frame", "Loading Bias Frame"),
            ("Načítám Dark Frame", "Loading Dark Frame"),
            ("Flat Frame aktivní, Bias Frame nepoužit", "Flat Frame active, Bias Frame not used"),
            ("používám Bias = 0", "using Bias = 0"),
            ("Dark Frame aktivní", "Dark Frame active"),
            ("odečítám Dark od Light snímků", "subtracting Dark from Light frames"),
            ("Konvertuji z Bayer masky", "Converting from Bayer pattern"),
            ("Vyřazuji bez platného star alignmentu", "Rejecting frame without valid star alignment"),
            ("Vyřazuji černý snímek bez hvězd", "Rejecting black frame without stars"),
            ("Mozaika: vytvářím plátno", "Mosaic: creating canvas"),
            ("Mozaika: převádím snímky na plátno", "Mosaic: warping frames to canvas"),
            ("Mozaika: skládám paralelně na CPU po blocích RAM", "Mosaic: stacking in parallel on CPU in RAM tiles"),
            ("Mozaika by vytvořila podezřele velké plátno", "Mosaic would create a suspiciously large canvas"),
            ("Zkontroluj alignment nebo sniž maximální drift hvězd.", "Check alignment or reduce the maximum star drift."),
            ("GPU vypocet selhal", "GPU computation failed"),
            ("pokracuji na CPU", "continuing on CPU"),
            ("skladam na CPU", "stacking on CPU"),
            ("GPU neni dostupne", "GPU is not available"),
            ("CuPy nelze nacist", "CuPy cannot be loaded"),
            ("Hotovo", "Done"),
            ("Žádný snímek neprošel zarovnáním. Zkontroluj, zda složka Light neobsahuje Dark/Bias snímky nebo zda jsou ve snímcích detekovatelné hvězdy.", "No frame passed alignment. Check whether the Light folder contains Dark/Bias frames or whether detectable stars are present."),
            ("Star + Comet výstupy vyžadují označení komety v prvním i posledním snímku.", "Star + Comet outputs require marking the comet in both the first and last frame."),
            ("Ve složce nejsou žádné XISF, FIT/FITS ani RAW snímky. Vypni volbu Pouze RAW, pokud chceš skládat i PNG/JPG/TIFF/BMP.", "The folder contains no XISF, FIT/FITS or RAW frames. Disable RAW only if you also want to stack PNG/JPG/TIFF/BMP files."),
            ("Ve složce nejsou žádné podporované obrázky. Podporované formáty zahrnují XISF, FIT/FITS, CR2/CR3/RAW, TIFF, PNG, JPG a BMP.", "The folder contains no supported images. Supported formats include XISF, FIT/FITS, CR2/CR3/RAW, TIFF, PNG, JPG and BMP."),
            ("Pro XISF podporu nainstaluj", "Install for XISF support"),
            ("Ve složce nezbyly žádné light snímky. Zkontroluj, zda nejsou soubory označené jako Dark/Bias/Flat.", "No light frames remain in the folder. Check whether files are marked as Dark/Bias/Flat."),
            ("Pro FITS podporu nainstaluj", "Install for FITS support"),
            ("RAW podpora vyžaduje rawpy. Nainstaluj", "RAW support requires rawpy. Install"),
            ("Prázdný obrazový soubor.", "Empty image file."),
        ]
        for src, dst in replacements:
            message = message.replace(src, dst)
        return message

    def run(self):
        try:
            if self.use_multiprocessing and self.processes > 1:
                result = stack_folder_multiprocessing(
                    self.folder,
                    self.settings,
                    self.processes,
                    self._progress,
                )
            else:
                result = stack_folder(self.folder, self.settings, self._progress)
            self.finished.emit(result)
        except ProcessingCancelled:
            self.cancelled.emit()
        except Exception as exc:
            message = str(exc)
            if getattr(self.settings, "language", "cz") == "en":
                message = self._translate_progress_message(message)
            self.failed.emit(message)

class FrameAnalysisWorker(QThread):
    progress = Signal(int, str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, folder: Path, settings: StackSettings):
        super().__init__()
        self.folder = folder
        self.settings = settings

    def _translate_progress_message(self, message: str) -> str:
        return StackWorker._translate_progress_message(self, message)

    def _progress(self, value: int, message: str):
        if getattr(self.settings, "language", "cz") == "en":
            message = self._translate_progress_message(message)
        self.progress.emit(value, message)

    def run(self):
        global LAST_STACK_SELECTION
        try:
            prepare_paths_for_alignment_mode(self.folder, self.settings, self._progress)
            self.finished.emit(dict(LAST_STACK_SELECTION))
        except Exception as exc:
            message = str(exc)
            if getattr(self.settings, "language", "cz") == "en":
                message = self._translate_progress_message(message)
            self.failed.emit(message)


class UpdateCheckWorker(QThread):
    finished_ok = Signal(dict)
    failed = Signal(str)

    def __init__(self, api_url: str, current_version: str, timeout_s: float = 6.0):
        super().__init__()
        self.api_url = api_url
        self.current_version = current_version
        self.timeout_s = float(timeout_s)

    def run(self):
        try:
            request = urllib.request.Request(
                self.api_url,
                headers={
                    "Accept": "application/vnd.github+json",
                    "User-Agent": f"Astro-Stacker/{APP_VERSION}",
                },
            )
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8", errors="replace"))
            tag = str(payload.get("tag_name") or payload.get("name") or "").strip()
            self.finished_ok.emit(
                {
                    "tag": tag,
                    "url": str(payload.get("html_url") or GITHUB_RELEASES_PAGE_URL),
                    "body": str(payload.get("body") or ""),
                    "is_newer": is_newer_version(tag, self.current_version),
                }
            )
        except Exception as exc:
            self.failed.emit(str(exc))


class AlpacaClient:
    """Small dependency-free client for the ASCOM Alpaca Camera API."""

    DISCOVERY_PORT = 32227
    DISCOVERY_MESSAGE = b"alpacadiscovery1"
    IMAGE_BYTES_DTYPES = {
        1: np.dtype("<i2"),
        2: np.dtype("<i4"),
        3: np.dtype("<f8"),
        4: np.dtype("<f4"),
        5: np.dtype("<u8"),
        6: np.dtype("u1"),
        7: np.dtype("<i8"),
        8: np.dtype("<u2"),
        9: np.dtype("<u4"),
    }

    def __init__(self, host: str, port: int, timeout: float = 8.0):
        host = str(host).strip()
        if not host:
            raise ValueError("Alpaca host is empty.")
        if "://" not in host:
            host = "http://" + host
        parsed = urllib.parse.urlsplit(host)
        hostname = parsed.hostname or parsed.path
        scheme = parsed.scheme or "http"
        selected_port = parsed.port or int(port)
        self.base_url = f"{scheme}://{hostname}:{selected_port}"
        self.timeout = float(timeout)
        self.client_id = max(1, os.getpid() & 0x7FFFFFFF)
        self.transaction_id = 0

    def _next_transaction(self) -> int:
        self.transaction_id += 1
        return self.transaction_id

    def request(
        self,
        method: str,
        path: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Any:
        params = dict(parameters or {})
        params["ClientID"] = self.client_id
        params["ClientTransactionID"] = self._next_transaction()
        url = self.base_url + "/" + str(path).lstrip("/")
        data = None
        if method.upper() == "GET":
            url += "?" + urllib.parse.urlencode(params)
        else:
            data = urllib.parse.urlencode(params).encode("ascii")
        headers = {
            "Accept": "application/json",
            "User-Agent": f"Astro-Stacker/{APP_VERSION}",
        }
        if data is not None:
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        request = urllib.request.Request(
            url,
            data=data,
            method=method.upper(),
            headers=headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Alpaca HTTP {exc.code}: {detail or exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Cannot reach Alpaca server {self.base_url}: {exc.reason}") from exc

        error_number = int(payload.get("ErrorNumber", 0) or 0)
        if error_number:
            message = str(payload.get("ErrorMessage", "") or "Unknown Alpaca error")
            raise RuntimeError(f"Alpaca error {error_number}: {message}")
        return payload.get("Value")

    def get(self, path: str, parameters: Optional[Dict[str, Any]] = None) -> Any:
        return self.request("GET", path, parameters)

    def put(self, path: str, parameters: Optional[Dict[str, Any]] = None) -> Any:
        return self.request("PUT", path, parameters)

    def configured_cameras(self) -> List[Dict[str, Any]]:
        devices = self.get("/management/v1/configureddevices")
        return [
            device for device in (devices or [])
            if str(device.get("DeviceType", "")).lower() == "camera"
        ]

    @classmethod
    def discover_servers(
        cls,
        timeout: float = 2.5,
        target_host: Optional[str] = None,
    ) -> List[Tuple[str, int]]:
        """Discover Alpaca servers on the local network using the standard UDP probe."""
        discovered: Dict[Tuple[str, int], None] = {}
        deadline = time.monotonic() + max(0.2, float(timeout))
        targets = [("255.255.255.255", cls.DISCOVERY_PORT)]
        if target_host:
            host = str(target_host).strip()
            if "://" not in host:
                host = "http://" + host
            parsed = urllib.parse.urlsplit(host)
            hostname = parsed.hostname or parsed.path
            if hostname and hostname != "255.255.255.255":
                targets.append((hostname, cls.DISCOVERY_PORT))
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp_socket:
            udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            udp_socket.bind(("", 0))
            for target in targets:
                try:
                    udp_socket.sendto(cls.DISCOVERY_MESSAGE, target)
                except OSError:
                    continue
            while time.monotonic() < deadline:
                udp_socket.settimeout(max(0.05, deadline - time.monotonic()))
                try:
                    response, address = udp_socket.recvfrom(4096)
                except socket.timeout:
                    break
                try:
                    payload = json.loads(response.decode("utf-8"))
                    port = int(payload.get("AlpacaPort", 0))
                except (UnicodeDecodeError, ValueError, TypeError, json.JSONDecodeError):
                    continue
                if 1 <= port <= 65535:
                    discovered[(str(address[0]), port)] = None
        return list(discovered)

    def camera_path(self, device_number: int, command: str) -> str:
        return f"/api/v1/camera/{int(device_number)}/{command.lower()}"

    def camera_get(self, device_number: int, command: str) -> Any:
        return self.get(self.camera_path(device_number, command))

    def camera_put(
        self,
        device_number: int,
        command: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Any:
        return self.put(self.camera_path(device_number, command), parameters)

    def camera_image_array(self, device_number: int) -> np.ndarray:
        """Download ImageArray in either Alpaca ImageBytes or JSON format."""
        params = {
            "ClientID": self.client_id,
            "ClientTransactionID": self._next_transaction(),
        }
        url = self.base_url + self.camera_path(device_number, "imagearray")
        url += "?" + urllib.parse.urlencode(params)
        request = urllib.request.Request(
            url,
            method="GET",
            headers={
                "Accept": "application/imagebytes, application/json",
                "User-Agent": f"Astro-Stacker/{APP_VERSION}",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=max(60.0, self.timeout)) as response:
                content_type = response.headers.get_content_type().lower()
                payload = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Alpaca HTTP {exc.code}: {detail or exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Cannot download the Alpaca image: {exc.reason}") from exc

        if content_type == "application/imagebytes" or not payload.lstrip().startswith((b"{", b"[")):
            return self._decode_image_bytes(payload)

        response_payload = json.loads(payload.decode("utf-8"))
        error_number = int(response_payload.get("ErrorNumber", 0) or 0)
        if error_number:
            message = str(response_payload.get("ErrorMessage", "") or "Unknown Alpaca error")
            raise RuntimeError(f"Alpaca error {error_number}: {message}")
        return np.asarray(response_payload.get("Value"))

    @classmethod
    def _decode_image_bytes(cls, payload: bytes) -> np.ndarray:
        if len(payload) < 44:
            raise RuntimeError(
                f"Invalid Alpaca ImageBytes response: only {len(payload)} bytes received."
            )
        error_number = int.from_bytes(payload[4:8], "little", signed=True)
        data_start = int.from_bytes(payload[16:20], "little", signed=False)
        transmission_type = int.from_bytes(payload[24:28], "little", signed=False)
        rank = int.from_bytes(payload[28:32], "little", signed=False)
        dimensions = [
            int.from_bytes(payload[32:36], "little", signed=False),
            int.from_bytes(payload[36:40], "little", signed=False),
            int.from_bytes(payload[40:44], "little", signed=False),
        ]
        if error_number:
            message_start = data_start if 44 <= data_start < len(payload) else 44
            message = payload[message_start:].decode("utf-8", errors="replace")
            raise RuntimeError(f"Alpaca error {error_number}: {message}")
        if rank not in (2, 3):
            raise RuntimeError(f"Unsupported Alpaca ImageBytes rank: {rank}.")
        shape = tuple(dimensions[:rank])
        if not all(shape):
            raise RuntimeError(f"Invalid Alpaca ImageBytes dimensions: {shape}.")
        dtype = cls.IMAGE_BYTES_DTYPES.get(transmission_type)
        if dtype is None:
            raise RuntimeError(
                f"Unsupported Alpaca ImageBytes element type: {transmission_type}."
            )
        if data_start < 44 or data_start > len(payload):
            raise RuntimeError(f"Invalid Alpaca ImageBytes data offset: {data_start}.")
        expected_pixels = math.prod(shape)
        available_pixels = (len(payload) - data_start) // dtype.itemsize
        if available_pixels < expected_pixels:
            raise RuntimeError(
                "Incomplete Alpaca image: "
                f"expected {expected_pixels} pixels, received {available_pixels}."
            )
        image = np.frombuffer(
            payload,
            dtype=dtype,
            count=expected_pixels,
            offset=data_start,
        )
        return image.reshape(shape)


class AlpacaCaptureWorker(QThread):
    progress = Signal(int, str)
    frame_saved = Signal(str)
    failed = Signal(str)
    connection_changed = Signal(bool)

    def __init__(
        self,
        host: str,
        port: int,
        device_number: int,
        output_folder: Path,
        exposure_seconds: float,
        frame_count: int,
        interval_seconds: float,
        gain: int,
        binning: int,
        cooler_enabled: bool,
        target_temperature: float,
        camera_id: str = "Camera",
    ):
        super().__init__()
        self.host = host
        self.port = int(port)
        self.device_number = int(device_number)
        self.output_folder = Path(output_folder)
        self.exposure_seconds = float(exposure_seconds)
        self.frame_count = int(frame_count)
        self.interval_seconds = float(interval_seconds)
        self.gain = int(gain)
        self.binning = int(binning)
        self.cooler_enabled = bool(cooler_enabled)
        self.target_temperature = float(target_temperature)
        safe_camera_id = re.sub(r"[^A-Za-z0-9_-]+", "_", str(camera_id)).strip("_")
        self.camera_id = safe_camera_id or f"Camera_{self.device_number}"
        self.camera_name = ""
        self.sensor_type: Optional[int] = None
        self.bayer_pattern: Optional[str] = None
        self.max_adu: Optional[int] = None

    def _wait_interruptible(self, seconds: float):
        deadline = time.monotonic() + max(0.0, float(seconds))
        while time.monotonic() < deadline:
            if self.isInterruptionRequested():
                raise ProcessingCancelled()
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))

    def _image_to_fits_array(self, value: Any) -> np.ndarray:
        image = np.asarray(value)
        if image.ndim == 2:
            image = image.T
        elif image.ndim == 3:
            image = np.transpose(image, (1, 0, 2))
            if image.shape[2] in (3, 4):
                image = np.moveaxis(image[..., :3], -1, 0)
        else:
            raise ValueError(f"Unsupported Alpaca ImageArray shape: {image.shape}")
        if image.dtype.kind not in "uif":
            image = image.astype(np.float32)
        elif image.dtype.kind in "iu" and image.size:
            minimum = int(np.min(image))
            maximum = int(np.max(image))
            if minimum >= 0 and maximum <= np.iinfo(np.uint16).max:
                image = image.astype(np.uint16, copy=False)
        return np.ascontiguousarray(image)

    @staticmethod
    def _bayer_pattern_from_alpaca(
        sensor_type: Optional[int],
        offset_x: int,
        offset_y: int,
    ) -> Optional[str]:
        if sensor_type != 2:
            return None
        patterns = {
            (0, 0): "RGGB",
            (1, 0): "GRBG",
            (0, 1): "GBRG",
            (1, 1): "BGGR",
        }
        return patterns[(int(offset_x) & 1, int(offset_y) & 1)]

    def _read_camera_metadata(self, client: AlpacaClient):
        try:
            self.camera_name = str(
                client.camera_get(self.device_number, "name") or ""
            )
        except Exception:
            self.camera_name = ""
        try:
            self.sensor_type = int(
                client.camera_get(self.device_number, "sensortype")
            )
        except Exception:
            self.sensor_type = None
        offset_x = 0
        offset_y = 0
        if self.sensor_type == 2:
            try:
                offset_x = int(client.camera_get(self.device_number, "bayeroffsetx"))
                offset_y = int(client.camera_get(self.device_number, "bayeroffsety"))
            except Exception:
                offset_x = 0
                offset_y = 0
        self.bayer_pattern = self._bayer_pattern_from_alpaca(
            self.sensor_type,
            offset_x,
            offset_y,
        )
        try:
            self.max_adu = int(client.camera_get(self.device_number, "maxadu"))
        except Exception:
            self.max_adu = None

    def _save_frame(
        self,
        image: np.ndarray,
        index: int,
        sensor_temperature: Optional[float] = None,
    ) -> Path:
        if fits is None:
            raise RuntimeError("Astropy is required to save Alpaca camera frames as FIT.")
        self.output_folder.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        path = (
            self.output_folder
            / f"Alpaca_{self.camera_id}_Light_{timestamp}_{index:04d}.fit"
        )
        header = fits.Header()
        header["IMAGETYP"] = ("LIGHT", "Frame type")
        header["EXPTIME"] = (self.exposure_seconds, "Exposure time in seconds")
        header["XBINNING"] = self.binning
        header["YBINNING"] = self.binning
        if self.gain >= 0:
            header["GAIN"] = self.gain
        if self.camera_name:
            header["INSTRUME"] = self.camera_name
        header["CAMID"] = (self.camera_id, "Multi-camera source identifier")
        if self.max_adu is not None:
            header["MAXADU"] = self.max_adu
        if self.bayer_pattern:
            header["BAYERPAT"] = (self.bayer_pattern, "Bayer CFA pattern")
        if self.cooler_enabled:
            header["SET-TEMP"] = (self.target_temperature, "Cooler target in deg C")
        if sensor_temperature is not None:
            header["CCD-TEMP"] = (sensor_temperature, "Sensor temperature in deg C")
        header["ORIGIN"] = f"Astro Stacker {APP_VERSION} Alpaca"
        fits.PrimaryHDU(data=image, header=header).writeto(path, overwrite=False)
        return path

    def run(self):
        client = AlpacaClient(self.host, self.port, timeout=15.0)
        try:
            client.camera_put(self.device_number, "connected", {"Connected": "true"})
            self.connection_changed.emit(True)
            self._read_camera_metadata(client)
            if self.binning > 0:
                client.camera_put(self.device_number, "binx", {"BinX": self.binning})
                client.camera_put(self.device_number, "biny", {"BinY": self.binning})
            if self.gain >= 0:
                client.camera_put(self.device_number, "gain", {"Gain": self.gain})
            if self.cooler_enabled:
                client.camera_put(
                    self.device_number,
                    "setccdtemperature",
                    {"SetCCDTemperature": self.target_temperature},
                )
                client.camera_put(
                    self.device_number,
                    "cooleron",
                    {"CoolerOn": "true"},
                )

            for index in range(1, self.frame_count + 1):
                if self.isInterruptionRequested():
                    raise ProcessingCancelled()
                self.progress.emit(
                    int((index - 1) * 100 / max(1, self.frame_count)),
                    f"Starting exposure {index}/{self.frame_count} ({self.exposure_seconds:.2f} s)…",
                )
                client.camera_put(
                    self.device_number,
                    "startexposure",
                    {"Duration": self.exposure_seconds, "Light": "true"},
                )
                deadline = time.monotonic() + max(30.0, self.exposure_seconds + 120.0)
                while not bool(client.camera_get(self.device_number, "imageready")):
                    if self.isInterruptionRequested():
                        try:
                            client.camera_put(self.device_number, "abortexposure")
                        except Exception:
                            pass
                        raise ProcessingCancelled()
                    if time.monotonic() > deadline:
                        raise TimeoutError("The Alpaca camera did not finish the exposure in time.")
                    self._wait_interruptible(0.25)

                self.progress.emit(
                    int((index - 0.25) * 100 / max(1, self.frame_count)),
                    f"Downloading exposure {index}/{self.frame_count}…",
                )
                image = self._image_to_fits_array(
                    client.camera_image_array(self.device_number)
                )
                sensor_temperature = None
                try:
                    sensor_temperature = float(
                        client.camera_get(self.device_number, "ccdtemperature")
                    )
                except Exception:
                    pass
                path = self._save_frame(image, index, sensor_temperature)
                self.frame_saved.emit(str(path))
                self.progress.emit(
                    int(index * 100 / max(1, self.frame_count)),
                    f"Saved {path.name}",
                )
                if index < self.frame_count:
                    self._wait_interruptible(self.interval_seconds)
        except ProcessingCancelled:
            pass
        except Exception as exc:
            self.failed.emit(str(exc))


class AlpacaCameraDialog(QDialog):
    CAMERA_TRANSLATIONS = {
        "cz": {
            "window_title": "Camera - ASCOM Alpaca Multi Camera",
            "alpaca_server": "Alpaca server",
            "discover": "Najít",
            "cameras": "Kamery",
            "refresh": "Obnovit",
            "select_all": "Vybrat vše",
            "select_none": "Zrušit výběr",
            "table_use": "Použít",
            "table_camera_id": "Camera/ID file",
            "table_server": "Alpaca server",
            "table_status": "Stav",
            "connect_selected": "Připojit vybrané",
            "disconnect_selected": "Odpojit vybrané",
            "connect_remaining": "Připojit zbývající",
            "disconnect_all": "Odpojit vše",
            "all_selected_connected": "Vše vybrané připojeno",
            "selected_disconnected": "Vybrané odpojeno",
            "connecting_selected": "Připojuji vybrané...",
            "exposure": "Expozice",
            "gain": "Gain",
            "unchanged": "Beze změny",
            "cooler": "Chlazení",
            "apply": "Použít",
            "sensor_temperature": "Teplota senzoru",
            "current_temperature": "Aktuální: -- °C",
            "current_temperature_value": "Aktuální: {value:.1f} °C",
            "current_temperature_unavailable": "Aktuální: nedostupná",
            "binning": "Binning",
            "frames": "Snímky",
            "interval": "Interval",
            "live_stack_folder": "Složka Live stack",
            "select": "Vybrat",
            "auto_live": "Automaticky spustit Live stack",
            "start_sequence": "Spustit synchronizovanou sérii",
            "stop_all": "Zastavit vše",
            "initial_status": "Zadej Alpaca server a stiskni Obnovit.",
            "disconnected": "Odpojeno",
            "connected": "Připojeno",
            "connecting": "Připojuji...",
            "disconnecting": "Odpojuji...",
            "error": "Chyba",
            "settings_applied": "Nastavení použito",
            "settings_error": "Chyba nastavení",
            "starting": "Startuji...",
            "stopping": "Zastavuji...",
            "completed": "Dokončeno",
            "stopped": "Zastaveno",
            "no_server_discovered": "Nebyl nalezen žádný Alpaca server. Zkontroluj, že je počítač připojený k Wi-Fi kamery.",
            "discovered_cameras": "Nalezeno {cameras} Alpaca kamer na {servers} serverech.",
            "found_servers_no_cameras": "Nalezeno {servers} Alpaca serverů, ale žádný nehlásí kameru.",
            "discovery_failed": "Vyhledání Alpaca selhalo: {error}",
            "select_camera_first": "Nejprve vyber alespoň jednu Alpaca kameru.",
            "found_cameras": "Nalezeno {count} Alpaca kamer.",
            "no_cameras_reported": "Tento server nehlásí žádné Alpaca kamery.",
            "cameras_connected": "{count} kamer připojeno.",
            "cameras_disconnected": "{count} kamer odpojeno.",
            "settings_applied_to": "Nastavení použito pro {count} kamer.",
            "select_folder_title": "Vybrat složku Live stack",
            "select_output_folder": "Vyber výstupní složku.",
            "started_sequence": "Spuštěna synchronizovaná série na {count} kamerách.",
            "stopping_all": "Zastavuji všechny kamery po dokončení aktuální operace...",
            "starting_exposure": "Spouštím expozici",
            "exposing": "Exponuji",
            "downloading_exposure": "Stahuji expozici",
            "saved": "Uloženo",
            "saved_status": "{name}: uloženo {file}; Live stack ho zařadí do fronty.",
            "all_completed": "Všechny kamerové série dokončeny.",
            "sequence_stopped": "Multi-camera série zastavena.",
        },
        "en": {},
    }

    def __init__(self, main_window: "AstroStackerWindow"):
        super().__init__(main_window)
        self.main_window = main_window
        self.capture_workers: Dict[str, AlpacaCaptureWorker] = {}
        self.camera_states: Dict[str, bool] = {}
        self.camera_progress: Dict[str, int] = {}
        self.setWindowTitle(self.tr_cam("window_title", "Camera - ASCOM Alpaca Multi Camera"))
        self.setMinimumWidth(720)
        self.setModal(False)

        layout = QVBoxLayout(self)
        connection_form = QFormLayout()
        connection_form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.host_edit = QLineEdit("10.42.42.1")
        self.port_spin = ArrowSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(80)
        server_row = QHBoxLayout()
        server_row.setContentsMargins(0, 0, 0, 0)
        server_row.addWidget(self.host_edit, 1)
        server_row.addWidget(self.port_spin)
        self.discover_btn = QPushButton(self.tr_cam("discover", "Discover"))
        self.discover_btn.clicked.connect(self.discover_cameras)
        server_row.addWidget(self.discover_btn)
        server_widget = QWidget()
        server_widget.setLayout(server_row)
        connection_form.addRow(self.tr_cam("alpaca_server", "Alpaca server"), server_widget)

        camera_buttons = QHBoxLayout()
        camera_buttons.setContentsMargins(0, 0, 0, 0)
        self.refresh_btn = QPushButton(self.tr_cam("refresh", "Refresh"))
        self.refresh_btn.clicked.connect(self.refresh_cameras)
        self.select_all_btn = QPushButton(self.tr_cam("select_all", "Select all"))
        self.select_none_btn = QPushButton(self.tr_cam("select_none", "Select none"))
        self.select_all_btn.clicked.connect(lambda: self.set_all_cameras_checked(True))
        self.select_none_btn.clicked.connect(lambda: self.set_all_cameras_checked(False))
        camera_buttons.addWidget(self.refresh_btn)
        camera_buttons.addWidget(self.select_all_btn)
        camera_buttons.addWidget(self.select_none_btn)
        camera_buttons.addStretch(1)
        camera_buttons_widget = QWidget()
        camera_buttons_widget.setLayout(camera_buttons)
        connection_form.addRow(self.tr_cam("cameras", "Cameras"), camera_buttons_widget)
        layout.addLayout(connection_form)

        self.camera_table = QTableWidget(0, 4)
        self.camera_table.setHorizontalHeaderLabels(
            [
                self.tr_cam("table_use", "Use"),
                self.tr_cam("table_camera_id", "Camera/ID file"),
                self.tr_cam("table_server", "Alpaca server"),
                self.tr_cam("table_status", "Status"),
            ]
        )
        self.camera_table.verticalHeader().setVisible(False)
        self.camera_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.camera_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.camera_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.camera_table.setMinimumHeight(150)
        header = self.camera_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        layout.addWidget(self.camera_table)

        connect_row = QHBoxLayout()
        self.connect_btn = QPushButton(self.tr_cam("connect_selected", "Connect selected"))
        self.disconnect_btn = QPushButton(self.tr_cam("disconnect_selected", "Disconnect selected"))
        self.connect_btn.clicked.connect(lambda: self.set_connected(True))
        self.disconnect_btn.clicked.connect(lambda: self.set_connected(False))
        connect_row.addWidget(self.connect_btn)
        connect_row.addWidget(self.disconnect_btn)
        layout.addLayout(connect_row)

        capture_form = QFormLayout()
        self.exposure_spin = ArrowDoubleSpinBox()
        self.exposure_spin.setRange(0.001, 86400.0)
        self.exposure_spin.setDecimals(3)
        self.exposure_spin.setValue(10.0)
        self.exposure_spin.setSuffix(" s")
        capture_form.addRow(self.tr_cam("exposure", "Exposure"), self.exposure_spin)

        self.gain_spin = ArrowSpinBox()
        self.gain_spin.setRange(-1, 100000)
        self.gain_spin.setValue(-1)
        self.gain_spin.setSpecialValueText(self.tr_cam("unchanged", "Unchanged"))
        capture_form.addRow(self.tr_cam("gain", "Gain"), self.gain_spin)

        temperature_row = QHBoxLayout()
        self.cooler_check = QCheckBox(self.tr_cam("cooler", "Cooler"))
        self.target_temperature_spin = ArrowDoubleSpinBox()
        self.target_temperature_spin.setRange(-80.0, 40.0)
        self.target_temperature_spin.setDecimals(1)
        self.target_temperature_spin.setValue(-10.0)
        self.target_temperature_spin.setSuffix(" °C")
        self.apply_camera_settings_btn = QPushButton(self.tr_cam("apply", "Apply"))
        self.apply_camera_settings_btn.clicked.connect(self.apply_camera_settings)
        temperature_row.addWidget(self.cooler_check)
        temperature_row.addWidget(self.target_temperature_spin)
        temperature_row.addWidget(self.apply_camera_settings_btn)
        temperature_widget = QWidget()
        temperature_widget.setLayout(temperature_row)
        capture_form.addRow(self.tr_cam("sensor_temperature", "Sensor temperature"), temperature_widget)

        self.current_temperature_label = QLabel(self.tr_cam("current_temperature", "Current: -- °C"))
        capture_form.addRow("", self.current_temperature_label)

        self.bin_spin = ArrowSpinBox()
        self.bin_spin.setRange(1, 16)
        self.bin_spin.setValue(1)
        capture_form.addRow(self.tr_cam("binning", "Binning"), self.bin_spin)

        self.count_spin = ArrowSpinBox()
        self.count_spin.setRange(1, 100000)
        self.count_spin.setValue(10)
        capture_form.addRow(self.tr_cam("frames", "Frames"), self.count_spin)

        self.interval_spin = ArrowDoubleSpinBox()
        self.interval_spin.setRange(0.0, 3600.0)
        self.interval_spin.setValue(5.0)
        self.interval_spin.setSuffix(" s")
        capture_form.addRow(self.tr_cam("interval", "Interval"), self.interval_spin)

        folder_row = QHBoxLayout()
        self.folder_edit = QLineEdit(
            str(main_window.folder) if main_window.folder is not None else ""
        )
        self.folder_btn = QPushButton(self.tr_cam("select", "Select"))
        self.folder_btn.clicked.connect(self.choose_output_folder)
        folder_row.addWidget(self.folder_edit, 1)
        folder_row.addWidget(self.folder_btn)
        folder_widget = QWidget()
        folder_widget.setLayout(folder_row)
        capture_form.addRow(self.tr_cam("live_stack_folder", "Live stack folder"), folder_widget)
        layout.addLayout(capture_form)

        self.auto_live_check = QCheckBox(self.tr_cam("auto_live", "Start Live stacking automatically"))
        self.auto_live_check.setChecked(True)
        layout.addWidget(self.auto_live_check)

        action_row = QHBoxLayout()
        self.start_btn = QPushButton(self.tr_cam("start_sequence", "Start synchronized sequence"))
        self.stop_btn = QPushButton(self.tr_cam("stop_all", "Stop all"))
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start_sequence)
        self.stop_btn.clicked.connect(self.stop_sequence)
        action_row.addWidget(self.start_btn)
        action_row.addWidget(self.stop_btn)
        layout.addLayout(action_row)

        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        self.status_label = QLabel(self.tr_cam("initial_status", "Enter the Alpaca server and press Refresh."))
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self.update_connection_indicator(False)

    def tr_cam(self, key: str, fallback: str, **kwargs) -> str:
        lang = getattr(self.main_window, "language", "en")
        text = self.CAMERA_TRANSLATIONS.get(lang, {}).get(key, fallback)
        if kwargs:
            try:
                return text.format(**kwargs)
            except Exception:
                return text
        return text

    def camera_status_text(self, text: str) -> str:
        mapping = {
            "Disconnected": self.tr_cam("disconnected", "Disconnected"),
            "Connected": self.tr_cam("connected", "Connected"),
            "Connecting...": self.tr_cam("connecting", "Connecting..."),
            "Disconnecting...": self.tr_cam("disconnecting", "Disconnecting..."),
            "Error": self.tr_cam("error", "Error"),
            "Settings applied": self.tr_cam("settings_applied", "Settings applied"),
            "Settings error": self.tr_cam("settings_error", "Settings error"),
            "Starting...": self.tr_cam("starting", "Starting..."),
            "Stopping...": self.tr_cam("stopping", "Stopping..."),
            "Completed": self.tr_cam("completed", "Completed"),
            "Stopped": self.tr_cam("stopped", "Stopped"),
        }
        return mapping.get(str(text), str(text))

    @staticmethod
    def camera_key(host: str, port: int, device_number: int) -> str:
        return f"{host}:{int(port)}/camera/{int(device_number)}"

    @staticmethod
    def camera_file_id(name: str, host: str, device_number: int) -> str:
        safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", str(name)).strip("_")
        safe_host = re.sub(r"[^A-Za-z0-9_-]+", "_", str(host)).strip("_")
        return f"{safe_name or 'Camera'}_{safe_host}_D{int(device_number)}"

    def clear_camera_table(self):
        self.camera_table.setRowCount(0)
        self.camera_states.clear()
        self.camera_progress.clear()
        self.update_connection_indicator(False)

    def add_camera_row(
        self,
        name: str,
        host: str,
        port: int,
        device_number: int,
        checked: bool = True,
    ):
        key = self.camera_key(host, port, device_number)
        camera_id = self.camera_file_id(name, host, device_number)
        for row in range(self.camera_table.rowCount()):
            item = self.camera_table.item(row, 0)
            if item is not None and item.data(Qt.UserRole) == key:
                return
        row = self.camera_table.rowCount()
        self.camera_table.insertRow(row)
        use_item = QTableWidgetItem()
        use_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
        use_item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
        use_item.setData(Qt.UserRole, key)
        use_item.setData(
            Qt.UserRole + 1,
            (str(host), int(port), int(device_number), str(name), camera_id),
        )
        self.camera_table.setItem(row, 0, use_item)
        self.camera_table.setItem(row, 1, QTableWidgetItem(str(name)))
        self.camera_table.setItem(row, 2, QTableWidgetItem(f"{host}:{int(port)}"))
        status_item = QTableWidgetItem(self.tr_cam("disconnected", "Disconnected"))
        status_item.setForeground(QBrush(QColor("#ef4444")))
        self.camera_table.setItem(row, 3, status_item)
        self.camera_states[key] = False
        self.camera_progress[key] = 0

    def camera_entries(self, selected_only: bool = True) -> List[Dict[str, Any]]:
        entries = []
        for row in range(self.camera_table.rowCount()):
            use_item = self.camera_table.item(row, 0)
            if use_item is None:
                continue
            if selected_only and use_item.checkState() != Qt.Checked:
                continue
            data = use_item.data(Qt.UserRole + 1)
            if not data:
                continue
            host, port, device_number, name, camera_id = data
            entries.append(
                {
                    "row": row,
                    "key": str(use_item.data(Qt.UserRole)),
                    "host": str(host),
                    "port": int(port),
                    "device_number": int(device_number),
                    "name": str(name),
                    "camera_id": str(camera_id),
                }
            )
        return entries

    def set_all_cameras_checked(self, checked: bool):
        for row in range(self.camera_table.rowCount()):
            item = self.camera_table.item(row, 0)
            if item is not None:
                item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
        self.update_aggregate_connection_indicator()

    def set_camera_status(
        self,
        key: str,
        text: str,
        color: str,
        connected: Optional[bool] = None,
        progress: Optional[int] = None,
    ):
        for entry in self.camera_entries(selected_only=False):
            if entry["key"] != key:
                continue
            item = self.camera_table.item(entry["row"], 3)
            if item is not None:
                item.setText(self.camera_status_text(text))
                item.setForeground(QBrush(QColor(color)))
            break
        if connected is not None:
            self.camera_states[key] = bool(connected)
        if progress is not None:
            self.camera_progress[key] = int(progress)
        self.update_aggregate_connection_indicator()

    def update_aggregate_connection_indicator(self):
        selected = self.camera_entries()
        if not selected:
            self.update_connection_indicator(False)
            return
        states = [self.camera_states.get(entry["key"], False) for entry in selected]
        if all(states):
            self.update_connection_indicator(True)
        elif any(states):
            self.connect_btn.setText(self.tr_cam("connect_remaining", "Connect remaining"))
            self.connect_btn.setEnabled(True)
            self.connect_btn.setStyleSheet("font-weight: 600; padding: 6px 12px;")
            self.disconnect_btn.setText(self.tr_cam("disconnect_all", "Disconnect all"))
            self.disconnect_btn.setEnabled(True)
            self.disconnect_btn.setStyleSheet(
                """
                QPushButton {
                    font-weight: 600;
                    padding: 6px 12px;
                    color: white;
                    background-color: #b45309;
                    border: 1px solid #f59e0b;
                }
                """
            )
        else:
            self.update_connection_indicator(False)

    def update_connection_indicator(
        self,
        connected: Optional[bool],
    ):
        base_style = "font-weight: 600; padding: 6px 12px;"
        if connected is True:
            self.connect_btn.setText(self.tr_cam("all_selected_connected", "All selected connected"))
            self.connect_btn.setEnabled(False)
            self.connect_btn.setStyleSheet(
                base_style
                + """
                QPushButton, QPushButton:disabled {
                    color: white;
                    background-color: #2563eb;
                    border: 1px solid #60a5fa;
                }
                """
            )
            self.disconnect_btn.setText(self.tr_cam("disconnect_selected", "Disconnect selected"))
            self.disconnect_btn.setEnabled(True)
            self.disconnect_btn.setStyleSheet(base_style)
        elif connected is False:
            self.connect_btn.setText(self.tr_cam("connect_selected", "Connect selected"))
            self.connect_btn.setEnabled(True)
            self.connect_btn.setStyleSheet(base_style)
            self.disconnect_btn.setText(self.tr_cam("selected_disconnected", "Selected disconnected"))
            self.disconnect_btn.setEnabled(False)
            self.disconnect_btn.setStyleSheet(
                base_style
                + """
                QPushButton, QPushButton:disabled {
                    color: white;
                    background-color: #b91c1c;
                    border: 1px solid #ef4444;
                }
                """
            )
        else:
            self.connect_btn.setText(self.tr_cam("connecting_selected", "Connecting selected..."))
            self.connect_btn.setEnabled(False)
            self.connect_btn.setStyleSheet(
                base_style
                + """
                QPushButton, QPushButton:disabled {
                    color: #111827;
                    background-color: #f59e0b;
                    border: 1px solid #fbbf24;
                }
                """
            )
            self.disconnect_btn.setText(self.tr_cam("disconnect_selected", "Disconnect selected"))
            self.disconnect_btn.setEnabled(False)
            self.disconnect_btn.setStyleSheet(base_style)

    def discover_cameras(self):
        self.discover_btn.setEnabled(False)
        self.refresh_btn.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            servers = AlpacaClient.discover_servers(
                target_host=self.host_edit.text(),
            )
            if not servers:
                self.status_label.setText(
                    self.tr_cam(
                        "no_server_discovered",
                        "No Alpaca server was discovered. Check that the computer is connected to the camera Wi-Fi.",
                    )
                )
                return

            total_cameras = 0
            self.clear_camera_table()
            first_server: Optional[Tuple[str, int]] = None
            errors = []
            for host, port in servers:
                try:
                    cameras = AlpacaClient(host, port).configured_cameras()
                except Exception as exc:
                    errors.append(f"{host}:{port}: {exc}")
                    continue
                if first_server is None and cameras:
                    first_server = (host, port)
                for camera in cameras:
                    number = int(camera.get("DeviceNumber", 0))
                    name = str(camera.get("DeviceName") or f"Camera {number}")
                    self.add_camera_row(name, host, port, number)
                    total_cameras += 1

            if first_server is not None:
                self.host_edit.setText(first_server[0])
                self.port_spin.setValue(first_server[1])
            if total_cameras:
                self.refresh_connection_indicator()
                self.status_label.setText(
                    self.tr_cam(
                        "discovered_cameras",
                        "Discovered {cameras} Alpaca camera(s) on {servers} server(s).",
                        cameras=total_cameras,
                        servers=len(servers),
                    )
                )
            elif errors:
                self.status_label.setText(errors[0])
            else:
                self.status_label.setText(
                    self.tr_cam(
                        "found_servers_no_cameras",
                        "Found {servers} Alpaca server(s), but none reported a camera.",
                        servers=len(servers),
                    )
                )
        except Exception as exc:
            self.status_label.setText(
                self.tr_cam("discovery_failed", "Alpaca discovery failed: {error}", error=exc)
            )
        finally:
            QApplication.restoreOverrideCursor()
            self.discover_btn.setEnabled(True)
            self.refresh_btn.setEnabled(True)

    def current_client_and_device(self) -> Tuple[AlpacaClient, int]:
        entries = self.camera_entries()
        if not entries:
            raise RuntimeError(self.tr_cam("select_camera_first", "Select at least one Alpaca camera first."))
        entry = entries[0]
        return (
            AlpacaClient(entry["host"], entry["port"]),
            entry["device_number"],
        )

    def refresh_cameras(self):
        self.refresh_btn.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            client = AlpacaClient(self.host_edit.text(), self.port_spin.value())
            cameras = client.configured_cameras()
            self.clear_camera_table()
            for camera in cameras:
                number = int(camera.get("DeviceNumber", 0))
                name = str(camera.get("DeviceName") or f"Camera {number}")
                self.add_camera_row(
                    name,
                    self.host_edit.text().strip(),
                    self.port_spin.value(),
                    number,
                )
            if cameras:
                self.refresh_connection_indicator()
                self.status_label.setText(
                    self.tr_cam("found_cameras", "Found {count} Alpaca camera(s).", count=len(cameras))
                )
            else:
                self.status_label.setText(
                    self.tr_cam("no_cameras_reported", "No Alpaca cameras were reported by this server.")
                )
        except Exception as exc:
            self.status_label.setText(str(exc))
        finally:
            QApplication.restoreOverrideCursor()
            self.refresh_btn.setEnabled(True)

    def refresh_connection_indicator(self):
        for entry in self.camera_entries():
            try:
                state = bool(
                    AlpacaClient(entry["host"], entry["port"]).camera_get(
                        entry["device_number"],
                        "connected",
                    )
                )
            except Exception:
                state = False
            self.set_camera_status(
                entry["key"],
                "Connected" if state else "Disconnected",
                "#60a5fa" if state else "#ef4444",
                connected=state,
            )

    def set_connected(self, connected: bool):
        self.update_connection_indicator(None)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            entries = self.camera_entries()
            if not entries:
                raise RuntimeError(self.tr_cam("select_camera_first", "Select at least one Alpaca camera first."))
            failures = []
            for index, entry in enumerate(entries):
                client = AlpacaClient(entry["host"], entry["port"])
                try:
                    self.set_camera_status(
                        entry["key"],
                        "Connecting..." if connected else "Disconnecting...",
                        "#f59e0b",
                    )
                    client.camera_put(
                        entry["device_number"],
                        "connected",
                        {"Connected": "true" if connected else "false"},
                    )
                    try:
                        state = bool(
                            client.camera_get(entry["device_number"], "connected")
                        )
                    except Exception:
                        state = bool(connected)
                    if state and index == 0:
                        self.read_camera_settings(client, entry["device_number"])
                    self.set_camera_status(
                        entry["key"],
                        "Connected" if state else "Disconnected",
                        "#60a5fa" if state else "#ef4444",
                        connected=state,
                    )
                except Exception as exc:
                    failures.append(f"{entry['name']}: {exc}")
                    self.set_camera_status(
                        entry["key"],
                        "Error",
                        "#ef4444",
                        connected=False,
                    )
            if failures:
                self.status_label.setText("; ".join(failures))
            else:
                self.status_label.setText(
                    self.tr_cam(
                        "cameras_connected" if connected else "cameras_disconnected",
                        "{count} camera(s) connected." if connected else "{count} camera(s) disconnected.",
                        count=len(entries),
                    )
                )
        except Exception as exc:
            self.status_label.setText(str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    def read_camera_settings(
        self,
        client: Optional[AlpacaClient] = None,
        device_number: Optional[int] = None,
    ):
        if client is None or device_number is None:
            client, device_number = self.current_client_and_device()
        details = []
        try:
            gain_min = int(client.camera_get(device_number, "gainmin"))
            gain_max = int(client.camera_get(device_number, "gainmax"))
            if gain_max >= gain_min:
                self.gain_spin.setRange(gain_min, gain_max)
        except Exception:
            pass
        try:
            gain = int(client.camera_get(device_number, "gain"))
            self.gain_spin.setValue(gain)
            details.append(f"gain {gain}")
        except Exception:
            pass
        try:
            temperature = float(client.camera_get(device_number, "ccdtemperature"))
            self.current_temperature_label.setText(
                self.tr_cam("current_temperature_value", "Current: {value:.1f} °C", value=temperature)
            )
            details.append(f"sensor {temperature:.1f} °C")
        except Exception:
            self.current_temperature_label.setText(
                self.tr_cam("current_temperature_unavailable", "Current: unavailable")
            )
        try:
            cooler_on = bool(client.camera_get(device_number, "cooleron"))
            self.cooler_check.setChecked(cooler_on)
        except Exception:
            pass
        try:
            target = float(client.camera_get(device_number, "setccdtemperature"))
            self.target_temperature_spin.setValue(target)
        except Exception:
            pass
        return details

    def apply_camera_settings(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            entries = self.camera_entries()
            if not entries:
                raise RuntimeError(self.tr_cam("select_camera_first", "Select at least one Alpaca camera first."))
            failures = []
            for index, entry in enumerate(entries):
                client = AlpacaClient(entry["host"], entry["port"])
                try:
                    if self.gain_spin.value() >= 0:
                        client.camera_put(
                            entry["device_number"],
                            "gain",
                            {"Gain": self.gain_spin.value()},
                        )
                    if self.cooler_check.isChecked():
                        client.camera_put(
                            entry["device_number"],
                            "setccdtemperature",
                            {
                                "SetCCDTemperature":
                                    self.target_temperature_spin.value()
                            },
                        )
                    client.camera_put(
                        entry["device_number"],
                        "cooleron",
                        {
                            "CoolerOn":
                                "true" if self.cooler_check.isChecked() else "false"
                        },
                    )
                    if index == 0:
                        self.read_camera_settings(client, entry["device_number"])
                    self.set_camera_status(
                        entry["key"],
                        "Settings applied",
                        "#60a5fa",
                        connected=True,
                    )
                except Exception as exc:
                    failures.append(f"{entry['name']}: {exc}")
                    self.set_camera_status(entry["key"], "Settings error", "#ef4444")
            if failures:
                self.status_label.setText("; ".join(failures))
            else:
                self.status_label.setText(
                    self.tr_cam(
                        "settings_applied_to",
                        "Settings applied to {count} camera(s).",
                        count=len(entries),
                    )
                )
        except Exception as exc:
            self.status_label.setText(str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    def choose_output_folder(self):
        selected = QFileDialog.getExistingDirectory(
            self,
            self.tr_cam("select_folder_title", "Select Live stack folder"),
            self.folder_edit.text(),
        )
        if selected:
            self.folder_edit.setText(selected)

    def start_sequence(self):
        if self.capture_workers:
            return
        try:
            entries = self.camera_entries()
            if not entries:
                raise RuntimeError(self.tr_cam("select_camera_first", "Select at least one Alpaca camera first."))
            output_folder_text = self.folder_edit.text().strip()
            if not output_folder_text:
                raise RuntimeError(self.tr_cam("select_output_folder", "Select an output folder."))
            output_folder = Path(output_folder_text).expanduser()
            output_folder.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            QMessageBox.warning(self, "Camera", str(exc))
            return

        self.main_window.folder = output_folder
        self.main_window.folder_label.setText(
            f"{self.main_window.tr_ui('folder_prefix')}: {output_folder}"
        )
        if self.auto_live_check.isChecked() and not self.main_window.auto_stack_active:
            self.main_window.start_auto_stacking()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress.setValue(0)
        self.update_connection_indicator(None)
        self.camera_progress = {entry["key"]: 0 for entry in entries}

        for entry in entries:
            worker = AlpacaCaptureWorker(
                entry["host"],
                entry["port"],
                entry["device_number"],
                output_folder,
                self.exposure_spin.value(),
                self.count_spin.value(),
                self.interval_spin.value(),
                self.gain_spin.value(),
                self.bin_spin.value(),
                self.cooler_check.isChecked(),
                self.target_temperature_spin.value(),
                camera_id=entry["camera_id"],
            )
            key = entry["key"]
            name = entry["name"]
            self.capture_workers[key] = worker
            self.set_camera_status(key, "Starting...", "#f59e0b", progress=0)
            worker.progress.connect(
                lambda value, message, camera_key=key, camera_name=name:
                    self.on_capture_progress(
                        camera_key,
                        camera_name,
                        value,
                        message,
                    )
            )
            worker.frame_saved.connect(
                lambda path, camera_key=key, camera_name=name:
                    self.on_frame_saved(camera_key, camera_name, path)
            )
            worker.failed.connect(
                lambda message, camera_key=key, camera_name=name:
                    self.on_capture_failed(camera_key, camera_name, message)
            )
            worker.connection_changed.connect(
                lambda connected, camera_key=key:
                    self.set_camera_status(
                        camera_key,
                        "Connected" if connected else "Disconnected",
                        "#60a5fa" if connected else "#ef4444",
                        connected=connected,
                    )
            )
            worker.finished.connect(
                lambda camera_key=key, finished_worker=worker:
                    self.on_capture_finished(camera_key, finished_worker)
            )

        for worker in list(self.capture_workers.values()):
            worker.start()
        self.status_label.setText(
            self.tr_cam(
                "started_sequence",
                "Started synchronized sequence on {count} camera(s).",
                count=len(entries),
            )
        )

    def stop_sequence(self):
        if self.capture_workers:
            for key, worker in list(self.capture_workers.items()):
                worker.requestInterruption()
                self.set_camera_status(key, "Stopping...", "#f59e0b")
            self.stop_btn.setEnabled(False)
            self.status_label.setText(
                self.tr_cam("stopping_all", "Stopping all cameras after their current operation...")
            )

    def translate_capture_message(self, message: str) -> str:
        text = str(message or "")
        replacements = [
            ("Starting exposure", self.tr_cam("starting_exposure", "Starting exposure")),
            ("Downloading exposure", self.tr_cam("downloading_exposure", "Downloading exposure")),
            ("Saved", self.tr_cam("saved", "Saved")),
        ]
        for source, target in replacements:
            if text.startswith(source):
                return target + text[len(source):]
        return text

    def on_capture_progress(
        self,
        key: str,
        name: str,
        value: int,
        message: str,
    ):
        self.camera_progress[key] = int(value)
        active_values = [
            self.camera_progress.get(active_key, 0)
            for active_key in self.capture_workers
        ]
        if active_values:
            self.progress.setValue(round(sum(active_values) / len(active_values)))
        display_message = self.translate_capture_message(message)
        short_message = display_message.replace(
            self.tr_cam("starting_exposure", "Starting exposure"),
            self.tr_cam("exposing", "Exposing"),
        )
        self.set_camera_status(key, short_message, "#60a5fa", progress=value)
        self.status_label.setText(f"{name}: {display_message}")

    def on_frame_saved(self, key: str, name: str, path: str):
        file_name = Path(path).name
        self.set_camera_status(key, f"{self.tr_cam('saved', 'Saved')} {file_name}", "#22c55e")
        self.status_label.setText(
            self.tr_cam(
                "saved_status",
                "{name}: saved {file}; Live stack will queue it.",
                name=name,
                file=file_name,
            )
        )

    def on_capture_failed(self, key: str, name: str, message: str):
        self.set_camera_status(key, "Error", "#ef4444")
        self.status_label.setText(f"{name}: {message}")

    def on_capture_finished(self, key: str, worker: AlpacaCaptureWorker):
        current_worker = self.capture_workers.get(key)
        if current_worker is not worker:
            worker.deleteLater()
            return
        self.capture_workers.pop(key, None)
        completed = self.camera_progress.get(key, 0) >= 100
        self.set_camera_status(
            key,
            "Completed" if completed else "Stopped",
            "#22c55e" if completed else "#f59e0b",
            progress=100 if completed else self.camera_progress.get(key, 0),
        )
        worker.deleteLater()
        if not self.capture_workers:
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            if self.camera_progress and all(
                value >= 100 for value in self.camera_progress.values()
            ):
                self.progress.setValue(100)
                self.status_label.setText(self.tr_cam("all_completed", "All camera sequences completed."))
            else:
                self.status_label.setText(self.tr_cam("sequence_stopped", "Multi-camera sequence stopped."))

    def closeEvent(self, event):
        if self.capture_workers:
            self.hide()
            event.ignore()
            return
        super().closeEvent(event)


class AstroStackerWindow(QMainWindow):
    TRANSLATIONS = {
        "cz": {
            "window_title": f"Astro Stacker {APP_DISPLAY_VERSION} - skládání astronomických snímků",
            "settings": "Setup", "folder_none": "Složka: nevybrána", "choose_folder": "Vybrat složku",
            "open_image": "Otevřít obrázek", "language": "Jazyk", "align": "Zarovnání",
            "pixinsight": "PixInsight",
            "pixinsight_tooltip": "Otevře PixInsight a zaregistruje AS_Stacker wrapper do menu Script > Utilities.",
            "stacking": "Skládání", "max_images": "Max. snímků", "sigma": "Sigma",
            "fit_only": "Pouze RAW",
            "normalize_bg": "Normalizovat pozadí", "mp_cpu": "Použít multiprocessing CPU",
            "gpu": "Použít GPU (CUDA/Metal)", "cpu_processes": "CPU procesy",
            "auto_ref": "Automaticky vybrat nejlepší referenci", "quality_filter": "Použít jen nejlepší snímky",
            "review_frames": "Zkontrolovat snímky před skládáním",
            "manual_reference": "Použít aktuální snímek jako referenci",
            "manual_reference_tooltip": "Ručně zvolený referenční snímek. Automatická reference je vypnutá.",
            "mosaic_mode": "Mozaika - rozšířit plátno",
            "mosaic_mode_tooltip": "Rozšíří výsledné plátno podle polohy všech zarovnaných snímků. Vhodné pro mozaikové sekvence chytrých dalekohledů. Při zapnuté GPU volbě se mozaika skládá po dlaždicích VRAM; jinak použije paralelní CPU cestu.",
            "keep": "Ponechat", "max_star_drift": "Max. drift hvězd", "max_comet_move": "Max. pohyb komety",
            "comet_refine": "Jemně doladit kometu", "comet_template": "Šablona komety",
            "comet_search": "Hledání komety", "ignore_edge": "Ignorovat okraj",
            "strict_stars": "Přísný filtr hvězd", "bayer_fit": "Bayer FIT",
            "satellite_trail_filter": "Satelitní stopa",
            "satellite_trail_tooltip": "Detekuje dlouhé rovné satelitní stopy, při skládání zamaskuje jejich pixely a podezřelé snímky označí v tabulce kvality.",
            "mark_comet_start": "Kometa první", "mark_comet_end": "Kometa poslední",
            "clear_comet_marks": "Kometa smaž",
            "clear_comet_marks_done": "Poloha komety byla smazána.",
            "comet_mark_first_label": "prvním",
            "comet_mark_last_label": "posledním",
            "comet_click_status": "Klikni do náhledu na jádro komety v {label} snímku: {name}",
            "comet_first_marked": "První poloha komety označena: x={x:.1f}, y={y:.1f}.",
            "comet_last_marked": "Poslední poloha komety označena: x={x:.1f}, y={y:.1f}.",
            "comet_two_point_ready": " Dvoubodový comet alignment je připravený.",
            "comet_mark_other": " Označ ještě druhý bod komety v druhém snímku.",
            "missing_folder_title": "Chybí složka",
            "missing_folder_message": "Nejdřív vyber složku se snímky.",
            "missing_frames_title": "Chybí snímky",
            "missing_frames_message": "Ve složce nejsou žádné podporované obrázky. Podporované formáty zahrnují XISF, FIT/FITS, CR2/CR3/RAW, TIFF, PNG, JPG a BMP.",
            "comet_not_marked_title": "Kometa není označena ve dvou snímcích",
            "comet_not_marked_message": "Pro tento typ dat označ kometu v prvním i posledním snímku. Jediný bod nestačí, protože slabé jádro za soumraku se v dalších snímcích nemusí automaticky najít.",
            "star_comet_done": "Hotovo — uloženo do astro_stacker_output: 01_star_stack.fit a 02_comet_stack.fit",
            "comet_first_tooltip": "Načte první snímek a po kliknutí uloží první polohu jádra komety.",
            "comet_last_tooltip": "Načte poslední snímek a po kliknutí uloží poslední polohu jádra komety. Program pak skládá podle vypočteného pohybu komety.",
            "comet_clear_tooltip": "Smaže první i poslední označenou polohu komety.",
            "max_comet_tooltip": "Maximální očekávaný pohyb komety vůči referenci. Pro kometu po západu Slunce zkus 500-1500 px.",
            "comet_refine_tooltip": "Po dvoubodové predikci dohledá jádro komety lokální korelací v každém snímku. Pomáhá, když jsou některé snímky o pár pixelů mimo.",
            "comet_template_tooltip": "Velikost šablony kolem jádra komety. Pro slabou difuzní kometu zkus 35-80 px.",
            "comet_search_tooltip": "Jak daleko od předpokládané pozice se smí kometa dohledat. Pro menší chyby zkus 50-120 px.",
            "start_stack": "Spustit skládání", "stop_stack": "Stop", "ready": "Připraveno",
            "calibration": "Kalibrace", "flat_unused": "Flat: nepoužit", "bias_unused": "Bias: nepoužit",
            "dark_unused": "Dark: nepoužit", "reset_calib": "Reset kalibrace",
            "show_stacked": "Zobraz složený obraz",
            "show_stacked_tooltip": "Vrátí do náhledu původní složený obraz bez ořezu, neutralizace a vizuálních úprav.",
            "show_stacked_done": "Zobrazen původní složený obraz bez úprav.",
            "undo": "Zpět",
            "undo_tooltip": "Vrátí poslední úpravu náhledu zpět. Uchovává poslední dva kroky.",
            "undo_empty": "Není co vrátit.",
            "undo_done": "Vrácen předchozí stav náhledu.",
            "check_updates": "Zkontrolovat aktualizace…",
            "update_available_title": "Je dostupná nová verze",
            "update_available_message": "Je dostupná nová verze Astro Stacker {version}.\n\nAktuální verze: {current}\n\nOtevřít stránku stažení?",
            "update_current_title": "Astro Stacker je aktuální",
            "update_current_message": "Používáš aktuální verzi Astro Stacker {current}.",
            "update_failed_title": "Kontrola aktualizace selhala",
            "update_failed_message": "Nepodařilo se zkontrolovat novou verzi:\n{error}",
            "clear_cache": "Smazat cache",
            "clear_cache_title": "Smazat cache",
            "clear_cache_message": "Smazat cache kvality a zarovnaných snímků?",
            "clear_cache_current": "Pouze aktuální složka",
            "clear_cache_all": "Celá cache v této složce",
            "clear_cache_none": "Nebyla nalezena žádná cache ke smazání.",
            "clear_cache_done": "Cache smazána: {files} souborů",
            "clear_cache_failed": "; nepodařilo se smazat {failed} položek",
            "clear_cache_tooltip": "Smaže pouze adresáře astro_stacker_cache. Snímky ve složce zůstanou nedotčené.",
            "hide_left_panel": "Skrýt levý panel",
            "show_left_panel": "Zobrazit levý panel",
            "hide_right_panel": "Skrýt pravý panel",
            "show_right_panel": "Zobrazit pravý panel",
            "tab_setting": "Setting",
            "tab_stars_alignment": "Stars Alignment",
            "tab_comet_alignment": "Comet Alignment",
            "tab_color": "Barva",
            "tab_correction": "Korekce",
            "tab_ai_tools": "AI tools",
            "curves": "Křivky / barvy / kontrast", "stf_strength": "Síla STF",
            "stf_strength_tooltip": "0 = silné STF pro velmi slabé snímky; 5 = standardní náhled; 10 = jemné STF.",
            "auto_stretch": "Balance",
            "auto_stretch_tooltip": "Neutralizuje pozadí a nastaví black point/gamma pouze pro náhled. Lineární FIT výstup se nemění.",
            "auto_stretch_done": "Balance nastaven pouze pro náhled.",
            "vignette": "Odstranění vinětace",
            "synthetic_flat": "Umělý flat",
            "color_background": "Korekce barevného pozadí",
            "astro_denoise": "Astro odšumění",
            "astro_denoise_tooltip": "Jemné odšumění náhledu/exportu s ochranou hvězd a struktur mlhovin. Potlačuje luminanční i barevný chroma šum. Lineární FIT výstup zůstává beze změny.",
            "denoise_mode": "Metoda odšumění",
            "denoise_classic": "Klasické",
            "denoise_drunet": "AI DRUNet",
            "select_drunet_model": "Vybrat model",
            "select_drunet_title": "Vybrat DRUNet ONNX model",
            "drunet_model_missing_title": "Chybí DRUNet model",
            "drunet_model_missing_message": "Vyber DRUNet model ve formátu ONNX. Podporovaný je barevný RGB model i šedý model, s volitelnou mapou šumu.",
            "onnxruntime_missing_title": "Chybí ONNX Runtime",
            "onnxruntime_missing_message": "Pro AI DRUNet nainstaluj balíček onnxruntime.",
            "drunet_model_selected": "DRUNet model nastaven: {name}",
            "drunet_tooltip": "AI odšumění přes DRUNet ONNX. Na macOS preferuje CoreML, ve Windows CUDA nebo DirectML. Model pracuje jen s vizuálním náhledem a PNG/JPG/TIFF exportem.",
            "star_deconvolution": "AI dekonvoluce hvězd",
            "star_deconvolution_size": "Velikost hvězd",
            "star_deconvolution_tooltip": "Doostří a zmenší hvězdy pomocí stellar modelu Cosmic Clarity. Ovlivňuje pouze náhled a PNG/JPG/TIFF export; lineární FIT zůstává beze změny.",
            "select_stellar_model": "Vybrat",
            "select_stellar_title": "Vybrat Cosmic Clarity stellar ONNX model",
            "stellar_model_missing_title": "Chybí stellar ONNX model",
            "stellar_model_missing_message": "Vyber převedený Cosmic Clarity stellar model ve formátu ONNX.",
            "stellar_model_selected": "Stellar model nastaven: {name}",
            "ai_stellar_preview": "AI dekonvoluce hvězd v náhledu…",
            "ai_stellar_export": "AI dekonvoluce hvězd v plném rozlišení…",
            "ai_stellar_done": "AI dekonvoluce hvězd dokončena.",
            "contrast": "Kontrast", "saturation": "Saturace", "red": "Červená", "green": "Zelená", "blue": "Modrá",
            "histogram": "Histogram L/R/G/B", "neutralize": "Neutralizovat pozadí",
            "clear_neutralize": "Zrušit neutralizaci", "flip_h": "Flip H",
            "flip_v": "Flip V", "rotate_left": "Left 90", "rotate_right": "Right 90",
            "preview_view": "Zobrazení náhledu", "fit": "Přizpůsobit",
            "reset": "Reset úprav", "starting": "Startuji…", "cancel_requested": "Zastavuji po dokončení aktuálního kroku…",
            "cancelled": "Skládání zastaveno.", "done_edit": "Složeno lineárně. Pro zviditelnění náhledu použij Balance.",
            "activity_title": "Aktuální operace",
            "ai_denoise_preview": "AI odšumění náhledu…",
            "ai_denoise_export": "AI odšumění v plném rozlišení…",
            "ai_denoise_done": "AI odšumění dokončeno.",
            "auto_rotated": "Automaticky pootočeno {angle:+.2f}°.",
            "failed": "Chyba při skládání.",
            "preview_prompt": "Vyber složku a spusť skládání.",
            "max_images_suffix": " snímků; 0 = vše",
            "processes_suffix": " procesů",
            "cpu_auto": "Auto",
            "cpu_manual": "Ručně",
            "folder_prefix": "Složka",
            "found_images": "Nalezeno {count} podporovaných obrázků.",
            "calibration_off": "Flat/Bias/Dark kalibrace vypnuta.",
            "open_preview_status": "Otevřen obrázek pro náhled: {name}",
            "stack_selection_summary": "Použito {used} / {total} light snímků; vyřazeno {excluded}.",
            "estimated_full_stack_ram": "Odhad RAM pro celý stack",
            "quality_score": "Skóre kvality: {score:.3g}",
            "stack_reference_tooltip": "Referenční snímek, použit ve stacku.",
            "stack_excluded_tooltip": "Vyřazeno filtrem kvality.",
            "stack_excluded_keep_tooltip": "Vyřazeno filtrem kvality; ponecháno nejlepších {keep} %.",
            "stack_used_tooltip": "Použit ve stacku.",
            "frame_quality_title": "Vyhodnocení snímků",
            "frame_quality_counts": "Lights: {lights}  |  Darks: {darks}  |  Flats: {flats}  |  Biases: {biases}",
            "frame_quality_headers": ["#", "Soubor", "Skóre", "Hvězdy", "Ostrost", "Kulatost", "FWHM px", "Stopa", "Stav", "Ref"],
            "frame_status_used": "Použit",
            "frame_status_reference": "Reference",
            "frame_status_quality": "Vyřazen kvalitou",
            "frame_status_alignment": "Selhal alignment",
            "frame_status_skipped": "Vynechán",
            "frame_status_manual_excluded": "Vyřazený",
            "trail_ok": "OK",
            "trail_suspect": "Podezřelá",
            "review_ready": "Výběr reference a kvality je hotový. Mezerníkem v tabulce vyřaď/povol snímky a potom pokračuj ve skládání.",
            "continue_stack": "Pokračovat ve skládání",
            "reference_cannot_exclude": "Referenční snímek nelze vyřadit. Nejprve vyber jinou referenci.",
            "choose_images_folder": "Vybrat složku se snímky",
            "open_image_title": "Otevřít obrázek",
            "all_files": "Všechny soubory",
            "metadata": "Metadata",
            "theme": "Motiv",
            "theme_dark": "Tmavý",
            "theme_light": "Světlý",
            "ui_mode": "Režim",
            "ui_simple": "Jednoduchý",
            "ui_advanced": "Pokročilý",
            "completion_sound": "Zvuk hotovo",
            "missing_image_title": "Chybí obraz",
            "missing_image_message": "Nejdřív slož snímky nebo otevři obrázek/FIT.",
            "save_image_title": "Uložit výsledek",
            "save_image_done": "Uloženo: {path}",
            "save_image_error_title": "Chyba při ukládání",
            "save_tiff_error": "TIFF se nepodařilo uložit přes OpenCV.",
            "neutralize_not_possible_title": "Nelze neutralizovat",
            "neutralize_rgb_required": "Neutralizace pozadí vyžaduje RGB obraz.",
            "not_enough_background_title": "Málo pozadí",
            "not_enough_background_message": "Nepodařilo se najít dostatek neutrálního pozadí.",
            "background_too_dark_title": "Pozadí je příliš tmavé",
            "background_too_dark_message": "Pozadí má příliš nízký signál pro neutralizaci.",
            "neutralization_applied": "Neutralizace pozadí aplikována: posun R {r:+.4f}, G {g:+.4f}, B {b:+.4f} (pozadí před: R {r_med:.4f}, G {g_med:.4f}, B {b_med:.4f})",
            "neutralization_cleared": "Neutralizace pozadí zrušena.",
            "remove_gradient": "Odstranit gradient",
            "clear_gradient": "Zrušit gradient",
            "gradient_removed": "Gradient pozadí byl odstraněn pouze pro náhled a PNG/JPG/TIFF export.",
            "gradient_cleared": "Odstranění gradientu bylo zrušeno.",
            "gradient_tooltip": "Odhadne hladký gradient pozadí robustním 2D polynomem. Lineární FIT výstup zůstává beze změny.",
            "crop_edges": "Turn angle",
            "crop_select": "Výběr",
            "auto_wb": "Auto WB",
            "crop_amount": "Úhel",
            "crop_tooltip": "Pootočí aktuální snímek o zadaný úhel ve stupních. Kladná hodnota otáčí proti směru hodinových ručiček.",
            "crop_select_tooltip": "Zapne ruční ořez: v náhledu myší označ oblast, která má zůstat.",
            "crop_select_status": "Myší označ v náhledu obdélník, který má zůstat po ořezu.",
            "crop_applied": "Pootočeno o {angle:+.1f}°. Nový rozměr: {w} × {h}.",
            "crop_selection_applied": "Oříznuto podle výběru. Nový rozměr: {w} × {h}.",
            "crop_too_small": "Vybraný ořez je příliš malý.",
            "awb_rgb_required": "AWB očekává RGB obraz.",
            "awb_not_enough_neutral": "Nepodařilo se najít dost neutrálních pixelů pro vyvážení bílé.",
            "awb_source_open": "otevřený obrázek",
            "awb_source_stack": "složený obraz",
            "awb_status": "AWB ({source}): R {r:.2f}×, G {g:.2f}×, B {b:.2f}×",
        },
        "en": {
            "window_title": f"Astro Stacker {APP_DISPLAY_VERSION} - astronomical image stacking",
            "settings": "Setup", "folder_none": "Folder: not selected", "choose_folder": "Choose folder",
            "open_image": "Open image", "language": "Language", "align": "Alignment",
            "pixinsight": "PixInsight",
            "pixinsight_tooltip": "Opens PixInsight and registers the AS_Stacker wrapper under Script > Utilities.",
            "stacking": "Stacking", "max_images": "Max. frames", "sigma": "Sigma",
            "fit_only": "RAW only",
            "normalize_bg": "Normalize background", "mp_cpu": "Use CPU multiprocessing",
            "gpu": "Use GPU (CUDA/Metal)", "cpu_processes": "CPU processes",
            "auto_ref": "Automatically choose best reference", "quality_filter": "Use only best frames",
            "review_frames": "Review frames before stacking",
            "manual_reference": "Use current frame as reference",
            "manual_reference_tooltip": "Manually selected reference frame. Automatic reference is disabled.",
            "mosaic_mode": "Mosaic - expand canvas",
            "mosaic_mode_tooltip": "Expands the output canvas to include all aligned frames. Useful for smart telescope mosaic sequences. With GPU enabled, mosaic integration uses VRAM tiles; otherwise it uses the parallel CPU path.",
            "keep": "Keep", "max_star_drift": "Max. star drift", "max_comet_move": "Max. comet motion",
            "comet_refine": "Refine comet position", "comet_template": "Comet template",
            "comet_search": "Comet search", "ignore_edge": "Ignore border",
            "strict_stars": "Strict star filter", "bayer_fit": "Bayer FIT",
            "satellite_trail_filter": "Satellite trail",
            "satellite_trail_tooltip": "Detects long straight satellite trails, masks their pixels during stacking, and marks suspect frames in the quality table.",
            "mark_comet_start": "Comet First", "mark_comet_end": "Comet Last",
            "clear_comet_marks": "Comet Clear",
            "clear_comet_marks_done": "Comet position cleared.",
            "comet_mark_first_label": "first",
            "comet_mark_last_label": "last",
            "comet_click_status": "Click the comet nucleus in the {label} frame preview: {name}",
            "comet_first_marked": "First comet position marked: x={x:.1f}, y={y:.1f}.",
            "comet_last_marked": "Last comet position marked: x={x:.1f}, y={y:.1f}.",
            "comet_two_point_ready": " Two-point comet alignment is ready.",
            "comet_mark_other": " Mark the second comet point in the other frame.",
            "missing_folder_title": "Missing folder",
            "missing_folder_message": "Choose an image folder first.",
            "missing_frames_title": "Missing frames",
            "missing_frames_message": "No supported images were found in the folder. Supported formats include XISF, FIT/FITS, CR2/CR3/RAW, TIFF, PNG, JPG, and BMP.",
            "comet_not_marked_title": "Comet is not marked in two frames",
            "comet_not_marked_message": "For this data type, mark the comet in both the first and last frame. A single point is not enough because a weak nucleus may not be detected automatically in later frames.",
            "star_comet_done": "Done - saved to astro_stacker_output: 01_star_stack.fit and 02_comet_stack.fit",
            "comet_first_tooltip": "Loads the first frame; click the comet nucleus to save its first position.",
            "comet_last_tooltip": "Loads the last frame; click the comet nucleus to save its last position. The program then stacks using the computed comet motion.",
            "comet_clear_tooltip": "Clears both marked comet positions.",
            "max_comet_tooltip": "Maximum expected comet motion relative to the reference. For a comet after sunset, try 500-1500 px.",
            "comet_refine_tooltip": "After two-point prediction, refines the comet nucleus by local correlation in each frame. Helps when some frames are a few pixels off.",
            "comet_template_tooltip": "Template size around the comet nucleus. For a weak diffuse comet, try 35-80 px.",
            "comet_search_tooltip": "How far from the predicted position the comet may be searched. For small errors, try 50-120 px.",
            "start_stack": "Start stacking", "stop_stack": "Stop", "ready": "Ready",
            "calibration": "Calibration", "flat_unused": "Flat: not used", "bias_unused": "Bias: not used",
            "dark_unused": "Dark: not used", "reset_calib": "Reset calibration",
            "show_stacked": "Show stacked image",
            "show_stacked_tooltip": "Returns the preview to the original stacked image without crop, neutralization, or visual adjustments.",
            "show_stacked_done": "Original stacked image shown without adjustments.",
            "undo": "Undo",
            "undo_tooltip": "Undo the last preview adjustment. Keeps the last two steps.",
            "undo_empty": "Nothing to undo.",
            "undo_done": "Previous preview state restored.",
            "check_updates": "Check for updates…",
            "update_available_title": "New version available",
            "update_available_message": "Astro Stacker {version} is available.\n\nCurrent version: {current}\n\nOpen the download page?",
            "update_current_title": "Astro Stacker is up to date",
            "update_current_message": "You are using the current Astro Stacker version {current}.",
            "update_failed_title": "Update check failed",
            "update_failed_message": "Could not check for a new version:\n{error}",
            "clear_cache": "Clear cache",
            "clear_cache_title": "Clear cache",
            "clear_cache_message": "Clear quality and aligned-frame cache?",
            "clear_cache_current": "Current folder only",
            "clear_cache_all": "All cache in this folder",
            "clear_cache_none": "No cache was found to clear.",
            "clear_cache_done": "Cache cleared: {files} files",
            "clear_cache_failed": "; failed to delete {failed} items",
            "clear_cache_tooltip": "Deletes only astro_stacker_cache folders. Images in the folder are not touched.",
            "hide_left_panel": "Hide left panel",
            "show_left_panel": "Show left panel",
            "hide_right_panel": "Hide right panel",
            "show_right_panel": "Show right panel",
            "tab_setting": "Setting",
            "tab_stars_alignment": "Stars Alignment",
            "tab_comet_alignment": "Comet Alignment",
            "tab_color": "Color",
            "tab_correction": "Correction",
            "tab_ai_tools": "AI tools",
            "curves": "Curves / color / contrast", "stf_strength": "STF strength",
            "stf_strength_tooltip": "0 = strong STF for very faint images; 5 = standard preview; 10 = gentle STF.",
            "auto_stretch": "Balance",
            "auto_stretch_tooltip": "Neutralizes background and sets black point/gamma for preview only. Linear FIT output is unchanged.",
            "auto_stretch_done": "Balance applied to preview only.",
            "vignette": "Vignette removal",
            "synthetic_flat": "Synthetic flat",
            "color_background": "Color background correction",
            "astro_denoise": "Astro Denoise",
            "astro_denoise_tooltip": "Gentle preview/export denoise with star and nebula-structure protection. Reduces luminance and color chroma noise. Linear FIT output is unchanged.",
            "denoise_mode": "Denoise method",
            "denoise_classic": "Classic",
            "denoise_drunet": "AI DRUNet",
            "select_drunet_model": "Select",
            "select_drunet_title": "Select DRUNet ONNX model",
            "drunet_model_missing_title": "DRUNet model missing",
            "drunet_model_missing_message": "Select a DRUNet model in ONNX format. RGB and grayscale models are supported, with an optional noise map.",
            "onnxruntime_missing_title": "ONNX Runtime missing",
            "onnxruntime_missing_message": "Install the onnxruntime package to use AI DRUNet.",
            "drunet_model_selected": "DRUNet model selected: {name}",
            "drunet_tooltip": "AI denoising through DRUNet ONNX. Prefers CoreML on macOS and CUDA or DirectML on Windows. It affects only the visual preview and PNG/JPG/TIFF export.",
            "star_deconvolution": "AI Star Deconvolution",
            "star_deconvolution_size": "Star size",
            "star_deconvolution_tooltip": "Sharpens and reduces stars with the Cosmic Clarity stellar model. It affects only the preview and PNG/JPG/TIFF export; linear FIT data is unchanged.",
            "select_stellar_model": "Select",
            "select_stellar_title": "Select Cosmic Clarity stellar ONNX model",
            "stellar_model_missing_title": "Stellar ONNX model missing",
            "stellar_model_missing_message": "Select a converted Cosmic Clarity stellar model in ONNX format.",
            "stellar_model_selected": "Stellar model selected: {name}",
            "ai_stellar_preview": "AI star deconvolution preview…",
            "ai_stellar_export": "AI star deconvolution at full resolution…",
            "ai_stellar_done": "AI star deconvolution complete.",
            "contrast": "Contrast", "saturation": "Saturation", "red": "Red", "green": "Green", "blue": "Blue",
            "histogram": "Histogram L/R/G/B", "neutralize": "Neutralize background",
            "clear_neutralize": "Clear neutralization", "flip_h": "Flip H",
            "flip_v": "Flip V", "rotate_left": "Rotate left", "rotate_right": "Rotate right",
            "preview_view": "Preview view", "fit": "Fit",
            "reset": "Reset adjustments", "starting": "Starting…", "cancel_requested": "Stopping after the current step…",
            "cancelled": "Stacking stopped.", "done_edit": "Linear stack complete. Use Balance to stretch the preview.",
            "activity_title": "Current operation",
            "ai_denoise_preview": "AI denoising preview…",
            "ai_denoise_export": "AI denoising at full resolution…",
            "ai_denoise_done": "AI denoising complete.",
            "auto_rotated": "Auto-rotated {angle:+.2f}°.",
            "failed": "Stacking failed.",
            "preview_prompt": "Choose a folder and start stacking.",
            "max_images_suffix": " frames; 0 = all",
            "processes_suffix": " processes",
            "cpu_auto": "Auto",
            "cpu_manual": "Manual",
            "folder_prefix": "Folder",
            "found_images": "Found {count} supported images.",
            "calibration_off": "Flat/Bias/Dark calibration disabled.",
            "open_preview_status": "Opened image for preview: {name}",
            "stack_selection_summary": "Used {used} / {total} light frames; excluded {excluded}.",
            "estimated_full_stack_ram": "Estimated full-stack RAM",
            "quality_score": "Quality score: {score:.3g}",
            "stack_reference_tooltip": "Reference frame, used in the stack.",
            "stack_excluded_tooltip": "Excluded by quality filter.",
            "stack_excluded_keep_tooltip": "Excluded by quality filter; kept the best {keep}%.",
            "stack_used_tooltip": "Used in the stack.",
            "frame_quality_title": "Frame quality",
            "frame_quality_counts": "Lights: {lights}  |  Darks: {darks}  |  Flats: {flats}  |  Biases: {biases}",
            "frame_quality_headers": ["#", "File", "Score", "Stars", "Sharpness", "Roundness", "FWHM px", "Trail", "Status", "Ref"],
            "frame_status_used": "Used",
            "frame_status_reference": "Reference",
            "frame_status_quality": "Excluded by quality",
            "frame_status_alignment": "Rejected alignment",
            "frame_status_skipped": "Skipped",
            "frame_status_manual_excluded": "Excluded",
            "trail_ok": "OK",
            "trail_suspect": "Suspect",
            "review_ready": "Reference and quality selection is ready. Press Space in the table to exclude/allow frames, then continue stacking.",
            "continue_stack": "Continue stacking",
            "reference_cannot_exclude": "The reference frame cannot be excluded. Select another reference first.",
            "choose_images_folder": "Choose image folder",
            "open_image_title": "Open image",
            "all_files": "All files",
            "metadata": "Metadata",
            "theme": "Theme",
            "theme_dark": "Dark",
            "theme_light": "Light",
            "ui_mode": "Mode",
            "ui_simple": "Simple",
            "ui_advanced": "Advanced",
            "completion_sound": "Finish Sound",
            "missing_image_title": "Missing image",
            "missing_image_message": "Stack frames first or open an image/FIT.",
            "save_image_title": "Save result",
            "save_image_done": "Saved: {path}",
            "save_image_error_title": "Save error",
            "save_tiff_error": "TIFF could not be saved with OpenCV.",
            "neutralize_not_possible_title": "Cannot neutralize",
            "neutralize_rgb_required": "Background neutralization requires an RGB image.",
            "not_enough_background_title": "Not enough background",
            "not_enough_background_message": "Could not find enough neutral background.",
            "background_too_dark_title": "Background is too dark",
            "background_too_dark_message": "The background signal is too low for neutralization.",
            "neutralization_applied": "Background neutralization applied: offsets R {r:+.4f}, G {g:+.4f}, B {b:+.4f} (background before: R {r_med:.4f}, G {g_med:.4f}, B {b_med:.4f})",
            "neutralization_cleared": "Background neutralization cleared.",
            "remove_gradient": "Remove gradient",
            "clear_gradient": "Clear gradient",
            "gradient_removed": "Background gradient removed for preview and PNG/JPG/TIFF export only.",
            "gradient_cleared": "Gradient removal cleared.",
            "gradient_tooltip": "Estimates a smooth background gradient with a robust 2D polynomial. Linear FIT output remains unchanged.",
            "crop_edges": "Turn angle",
            "crop_select": "Select",
            "auto_wb": "Auto WB",
            "crop_amount": "Angle",
            "crop_tooltip": "Rotates the current image by the selected angle in degrees. Positive values rotate counterclockwise.",
            "crop_select_tooltip": "Manual crop: drag a rectangle in the preview to keep that area.",
            "crop_select_status": "Drag a rectangle in the preview to keep that area after cropping.",
            "crop_applied": "Rotated by {angle:+.1f}°. New size: {w} × {h}.",
            "crop_selection_applied": "Cropped to the selected area. New size: {w} × {h}.",
            "crop_too_small": "The selected crop area is too small.",
            "awb_rgb_required": "AWB requires an RGB image.",
            "awb_not_enough_neutral": "Could not find enough neutral pixels for white balance.",
            "awb_source_open": "opened image",
            "awb_source_stack": "stacked image",
            "awb_status": "AWB ({source}): R {r:.2f}×, G {g:.2f}×, B {b:.2f}×",
        },
    }

    def tr_ui(self, key: str) -> str:
        return self.TRANSLATIONS.get(getattr(self, "language", "en"), self.TRANSLATIONS["en"]).get(key, key)

    def optimal_cpu_processes(self) -> int:
        cpu_count = max(1, os.cpu_count() or 1)
        if cpu_count <= 2:
            return 1
        return max(1, min(cpu_count - 1, int(round(cpu_count * 0.75))))

    def __init__(self):
        super().__init__()
        self.language = "en"
        self.setWindowTitle(self.tr_ui("window_title"))
        self.resize(1300, 850)

        self.folder: Optional[Path] = None
        self.flat_frame_path: Optional[Path] = None
        self.bias_frame_path: Optional[Path] = None
        self.dark_frame_path: Optional[Path] = None
        self.linear_result: Optional[np.ndarray] = None
        self.original_linear_result: Optional[np.ndarray] = None
        self.worker: Optional[StackWorker] = None
        self.zoom_mode: str = "fit"  # fit | fixed
        self.zoom_factor: float = 0.5
        self.preview_override: Optional[np.ndarray] = None
        self.preview_source_shape: Optional[Tuple[int, int]] = None
        self.flip_horizontal: bool = False
        self.flip_vertical: bool = False
        self.preview_rotation_degrees: int = 0
        self.preview_display_cache: Optional[np.ndarray] = None
        self.preview_display_cache_source_id: Optional[int] = None
        self.preview_display_cache_neutralized: bool = False
        self.preview_display_cache_stretched: bool = True
        self.neutralized_preview_layer: Optional[np.ndarray] = None
        self.neutralized_preview_base_source_id: Optional[int] = None
        self.gradient_preview_layer: Optional[np.ndarray] = None
        self.gradient_preview_base_source_id: Optional[int] = None
        self.preview_display_cache_edge: int = 0
        self.preview_display_scale: float = 1.0
        self.preview_fast_display_cache: Optional[np.ndarray] = None
        self.preview_fast_display_cache_key: Optional[Tuple[int, int]] = None
        self.preview_display_limits: Optional[Tuple[float, ...]] = None
        # A completed stack starts as a true linear preview. Balance enables
        # the robust display-only stretch without modifying linear_result.
        self.preview_auto_display_stretch: bool = True
        self.preview_heavy_cache: Optional[np.ndarray] = None
        self.preview_heavy_cache_key: Optional[Tuple[Any, ...]] = None
        detected_drunet_model = find_drunet_model()
        self.ai_denoise_model_path: Optional[str] = (
            str(detected_drunet_model) if detected_drunet_model is not None else None
        )
        self.preview_ai_denoise_cache: Optional[np.ndarray] = None
        self.preview_ai_denoise_cache_key: Optional[Tuple[Any, ...]] = None
        self.ai_denoise_layer_cache: Dict[Tuple[Any, ...], Dict[str, np.ndarray]] = {}
        self.ai_denoise_in_progress: bool = False
        self.pending_ai_denoise_completion_sound: bool = False
        detected_stellar_model = find_stellar_deconvolution_model()
        self.star_deconvolution_model_path: Optional[str] = (
            str(detected_stellar_model)
            if detected_stellar_model is not None
            else None
        )
        self.star_deconvolution_layer_cache: Dict[
            Tuple[Any, ...], Dict[str, np.ndarray]
        ] = {}
        self.star_deconvolution_in_progress: bool = False
        self.preview_render_array: Optional[np.ndarray] = None
        self.preview_fast_zoom_scale: float = 1.0
        self.showing_intro_preview: bool = False
        self.preview_interactive: bool = False
        self.preview_slider_zoom_state: Optional[Dict[str, Any]] = None
        self.preview_render_pending_final: bool = False
        self.preview_position_generation: int = 0
        self.preview_render_timer = QTimer(self)
        self.preview_render_timer.setSingleShot(True)
        self.preview_render_timer.timeout.connect(self._run_scheduled_preview_update)
        self.zoom_overlay_timer = QTimer(self)
        self.zoom_overlay_timer.setSingleShot(True)
        self.zoom_overlay_timer.timeout.connect(self.hide_zoom_overlay)
        self.preview_pan_inertia_timer = QTimer(self)
        self.preview_pan_inertia_timer.setInterval(16)
        self.preview_pan_inertia_timer.timeout.connect(self.run_preview_pan_inertia)
        self.preview_undo_stack: List[Dict[str, Any]] = []
        self.preview_undo_limit: int = 2
        self.restoring_preview_undo: bool = False
        self.preview_pan_velocity = (0.0, 0.0)
        self.awaiting_comet_click: bool = False
        self.comet_click_mode: Optional[str] = None  # "start" | "end"
        self.preview_override_path: Optional[str] = None
        self.preview_sequence_paths: List[Path] = []
        self.preview_sequence_loading: bool = False
        self.settings_store = QSettings("AstroStacker", "AstroStacker")
        self.recent_image_actions: List[QAction] = []
        self.slider_value_labels: Dict[QSlider, ArrowDoubleSpinBox] = {}
        self.simple_mode_hidden_widgets: List[QWidget] = []
        self.stack_selection_info: Dict[str, Any] = {}
        self.stack_used_paths: set[str] = set()
        self.stack_excluded_paths: set[str] = set()
        self.stack_reference_path: Optional[str] = None
        self.stack_quality_scores: Dict[str, float] = {}
        self.last_total_processing_time: Optional[float] = None
        self.last_estimated_used_ram_bytes: Optional[int] = None
        self.manual_excluded_paths: set[str] = set()
        self.review_ready: bool = False
        self.analysis_worker: Optional[FrameAnalysisWorker] = None
        self.update_check_worker: Optional[UpdateCheckWorker] = None
        self.update_check_manual: bool = False
        self.manual_reference_path: Optional[str] = None
        self.manual_comet_xy: Optional[Tuple[float, float]] = None
        self.manual_comet_reference_path: Optional[str] = None
        self.manual_comet_end_xy: Optional[Tuple[float, float]] = None
        self.manual_comet_end_path: Optional[str] = None
        self.auto_stack_active: bool = False
        self.auto_stack_timer = QTimer(self)
        self.auto_stack_timer.setInterval(1000)
        self.auto_stack_timer.timeout.connect(self.scan_auto_stack_folder)
        self.auto_stack_calibration_worker: Optional[AutoStackCalibrationWorker] = None
        self.auto_stack_worker: Optional[AutoStackFrameWorker] = None
        self.auto_stack_pending: Dict[str, Tuple[int, int, int]] = {}
        self.auto_stack_queued_paths: set[str] = set()
        self.auto_stack_ready_queue: List[Path] = []
        self.auto_stack_reference_gray: Optional[np.ndarray] = None
        self.auto_stack_reference_shape: Optional[Tuple[int, ...]] = None
        self.auto_stack_sum: Optional[np.ndarray] = None
        self.auto_stack_weight: Optional[np.ndarray] = None
        self.auto_stack_sumsq: Optional[np.ndarray] = None  # suma čtverců pro průběžný (online) odhad rozptylu
        self.auto_stack_reject_outliers: bool = True   # zapnutí kappa-sigma rejection pixelů v Live stacku
        self.auto_stack_reject_sigma: float = 3.0       # práh odmítnutí, násobek průběžné std. odchylky
        self.auto_stack_reject_min_frames: int = 4      # warm-up: počet snímků na daný pixel, než se začne odmítat
        self.auto_stack_reject_elongated_stars: bool = True
        self.auto_stack_min_roundness: float = 0.50
        self.auto_stack_min_shape_stars: int = 8
        self.auto_stack_last_quality_metrics: Optional[Dict[str, float]] = None
        self.auto_stack_balance_applied: bool = False
        self.auto_stack_rejected_pixels_last: int = 0   # kolik pixelů bylo odmítnuto v posledním zpracovaném snímku
        self.auto_stack_flip_180_count: int = 0   # kolik snímků bylo zarovnáno přes fallback s 180° otočením (meridian flip)
        self.auto_stack_flat: Optional[np.ndarray] = None
        self.auto_stack_bias: Optional[np.ndarray] = None
        self.auto_stack_dark: Optional[np.ndarray] = None
        self.auto_stack_accepted: int = 0
        self.auto_stack_rejected: int = 0
        self.alpaca_camera_dialog: Optional[AlpacaCameraDialog] = None

        self._build_ui()
        self._build_menu()
        self.apply_language()
        QTimer.singleShot(0, self.show_intro_preview)
        QTimer.singleShot(3500, lambda: self.check_for_updates(manual=False))

    def _panel_toggle_button(self, text: str, tooltip: str) -> QPushButton:
        button = QPushButton(text)
        button.setFixedSize(24, 24)
        button.setToolTip(tooltip)
        button.setFocusPolicy(Qt.NoFocus)
        button.setStyleSheet("font-weight: bold; padding: 0;")
        return button

    def _make_panel_rail(self, button: QPushButton) -> QFrame:
        rail = QFrame()
        rail.setFrameShape(QFrame.StyledPanel)
        rail.setFixedWidth(30)
        rail_layout = QVBoxLayout(rail)
        rail_layout.setContentsMargins(2, 8, 2, 2)
        rail_layout.setSpacing(0)
        rail_layout.addWidget(button, 0, Qt.AlignTop | Qt.AlignHCenter)
        rail_layout.addStretch(1)
        return rail

    def _refresh_preview_after_panel_toggle(self):
        if hasattr(self, "image_label"):
            if self.linear_result is not None or self.preview_override is not None:
                QTimer.singleShot(0, self.update_preview)
            elif getattr(self, "showing_intro_preview", False):
                QTimer.singleShot(0, self.show_intro_preview)

    def set_left_panel_collapsed(self, collapsed: bool):
        self.left_panel_collapsed = bool(collapsed)
        if hasattr(self, "left_panel_scroll"):
            self.left_panel_scroll.setVisible(not collapsed)
        if hasattr(self, "left_panel_rail"):
            self.left_panel_rail.setVisible(collapsed)
        self._refresh_preview_after_panel_toggle()

    def set_right_panel_collapsed(self, collapsed: bool):
        self.right_panel_collapsed = bool(collapsed)
        if hasattr(self, "right_panel_scroll"):
            self.right_panel_scroll.setVisible(not collapsed)
        if hasattr(self, "right_panel_rail"):
            self.right_panel_rail.setVisible(collapsed)
        self._refresh_preview_after_panel_toggle()

    def _build_menu(self):
        self.file_menu = self.menuBar().addMenu("Soubor")
        file_menu = self.file_menu

        self.open_action = QAction("Vybrat složku…", self)
        self.open_action.triggered.connect(self.choose_folder)
        file_menu.addAction(self.open_action)

        self.open_image_action = QAction("Otevřít obrázek…", self)
        self.open_image_action.triggered.connect(self.open_image_file)
        file_menu.addAction(self.open_image_action)

        self.recent_images_menu = file_menu.addMenu("Nedávné obrázky")
        self.clear_recent_images_action = QAction("Smazat historii", self)
        self.clear_recent_images_action.triggered.connect(self.clear_recent_images)
        self.update_recent_images_menu()

        self.save_action = QAction("Uložit výsledek jako…", self)
        self.save_action.triggered.connect(self.save_preview)
        file_menu.addAction(self.save_action)

        file_menu.addSeparator()

        self.save_profile_action = QAction("Uložit profil nastavení…", self)
        self.save_profile_action.triggered.connect(self.save_settings_profile)
        file_menu.addAction(self.save_profile_action)

        self.load_profile_action = QAction("Načíst profil nastavení…", self)
        self.load_profile_action.triggered.connect(self.load_settings_profile)
        file_menu.addAction(self.load_profile_action)

        file_menu.addSeparator()

        self.comet_action = QAction("Označit kometu v prvním snímku…", self)
        self.comet_action.triggered.connect(self.select_comet_start_point)
        file_menu.addAction(self.comet_action)

        self.comet_end_action = QAction("Označit kometu v posledním snímku…", self)
        self.comet_end_action.triggered.connect(self.select_comet_end_point)
        file_menu.addAction(self.comet_end_action)

        self.view_menu = self.menuBar().addMenu("Zobrazení")
        view_menu = self.view_menu

        self.fit_action = QAction("Přizpůsobit", self)
        self.fit_action.setShortcut("Ctrl+0")
        self.fit_action.triggered.connect(self.zoom_fit)
        view_menu.addAction(self.fit_action)

        self.actual_action = QAction("Zobrazit 1:1", self)
        self.actual_action.setShortcut("Ctrl+1")
        self.actual_action.triggered.connect(self.zoom_actual_size)
        view_menu.addAction(self.actual_action)

        self.zoom_in_action = QAction("Přiblížit", self)
        self.zoom_in_action.setShortcut("Ctrl++")
        self.zoom_in_action.triggered.connect(self.zoom_in)
        view_menu.addAction(self.zoom_in_action)

        self.zoom_out_action = QAction("Oddálit", self)
        self.zoom_out_action.setShortcut("Ctrl+-")
        self.zoom_out_action.triggered.connect(self.zoom_out)
        view_menu.addAction(self.zoom_out_action)

        self.auto_stack_menu = self.menuBar().addMenu("Live stack")
        self.auto_stack_start_action = QAction("Live stacking", self)
        self.auto_stack_start_action.triggered.connect(self.start_auto_stacking)
        self.auto_stack_menu.addAction(self.auto_stack_start_action)

        self.auto_stack_stop_action = QAction("Stop Live stacking", self)
        self.auto_stack_stop_action.triggered.connect(self.stop_auto_stacking)
        self.auto_stack_stop_action.setEnabled(False)
        self.auto_stack_menu.addAction(self.auto_stack_stop_action)

        self.auto_stack_menu.addSeparator()
        self.auto_stack_reject_action = QAction("Reject outliers (sigma)", self)
        self.auto_stack_reject_action.setCheckable(True)
        self.auto_stack_reject_action.setChecked(True)
        self.auto_stack_reject_action.toggled.connect(self.on_auto_stack_reject_toggled)
        self.auto_stack_menu.addAction(self.auto_stack_reject_action)

        self.auto_stack_reject_elongated_action = QAction("Reject elongated stars", self)
        self.auto_stack_reject_elongated_action.setCheckable(True)
        self.auto_stack_reject_elongated_action.setChecked(True)
        self.auto_stack_reject_elongated_action.toggled.connect(self.on_auto_stack_reject_elongated_toggled)
        self.auto_stack_menu.addAction(self.auto_stack_reject_elongated_action)

        self.camera_menu = self.menuBar().addMenu("Camera")
        self.camera_control_action = QAction("ASCOM Alpaca cameras…", self)
        self.camera_control_action.triggered.connect(self.show_alpaca_camera_dialog)
        self.camera_menu.addAction(self.camera_control_action)

        self.help_menu = self.menuBar().addMenu("Nápověda")
        help_menu = self.help_menu

        self.user_guide_action = QAction("Nápověda k programu…", self)
        self.user_guide_action.triggered.connect(self.show_user_guide_dialog)
        help_menu.addAction(self.user_guide_action)

        self.check_updates_action = QAction("Zkontrolovat aktualizace…", self)
        self.check_updates_action.triggered.connect(lambda: self.check_for_updates(manual=True))
        help_menu.addAction(self.check_updates_action)

        help_menu.addSeparator()

        self.help_about_action = QAction("O programu…", self)
        try:
            self.help_about_action.setMenuRole(QAction.MenuRole.NoRole)
        except AttributeError:
            self.help_about_action.setMenuRole(QAction.NoRole)
        self.help_about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(self.help_about_action)

        file_menu.addSeparator()

        self.open_log_action = QAction("Zobrazit / smazat logy…", self)
        self.open_log_action.triggered.connect(self.show_logs_dialog)
        file_menu.addAction(self.open_log_action)

        self.quit_action = QAction("Konec", self)
        self.quit_action.triggered.connect(self.close)
        file_menu.addAction(self.quit_action)

    def show_alpaca_camera_dialog(self):
        if self.alpaca_camera_dialog is None:
            self.alpaca_camera_dialog = AlpacaCameraDialog(self)
        if self.folder is not None and not self.alpaca_camera_dialog.folder_edit.text():
            self.alpaca_camera_dialog.folder_edit.setText(str(self.folder))
        self.alpaca_camera_dialog.show()
        self.alpaca_camera_dialog.raise_()
        self.alpaca_camera_dialog.activateWindow()

    def _build_ui(self):
        root = QWidget()
        layout = QHBoxLayout(root)
        self.setCentralWidget(root)

        side_content_width = 370
        side_scroll_width = 400
        self.left_panel_collapsed = False
        self.right_panel_collapsed = False

        self.left_expand_btn = self._panel_toggle_button("▶", self.tr_ui("show_left_panel"))
        self.left_expand_btn.clicked.connect(lambda: self.set_left_panel_collapsed(False))
        self.left_panel_rail = self._make_panel_rail(self.left_expand_btn)
        self.left_panel_rail.hide()
        layout.addWidget(self.left_panel_rail)

        left = QFrame()
        left.setFrameShape(QFrame.StyledPanel)
        left.setFixedWidth(side_content_width)
        left_layout = QVBoxLayout(left)

        left_panel_scroll = QScrollArea()
        left_panel_scroll.setWidgetResizable(True)
        left_panel_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_panel_scroll.setWidget(left)
        left_panel_scroll.setFixedWidth(side_scroll_width)
        layout.addWidget(left_panel_scroll)
        self.left_panel_scroll = left_panel_scroll

        title = QLabel("Nastavení")
        self.title_label = title
        title.setAlignment(Qt.AlignCenter)
        title.setWordWrap(True)
        title.setStyleSheet("font-size: 18px; font-weight: bold;")

        left_header = QHBoxLayout()
        left_header.setContentsMargins(0, 0, 0, 0)
        left_header.setSpacing(4)
        self.left_collapse_btn = self._panel_toggle_button("◀", self.tr_ui("hide_left_panel"))
        self.left_collapse_btn.clicked.connect(lambda: self.set_left_panel_collapsed(True))
        left_header.addWidget(self.left_collapse_btn, 0, Qt.AlignTop)
        left_header.addWidget(title, 1)
        left_header.addSpacing(24)
        left_layout.addLayout(left_header)

        self.folder_label = QLabel("Složka: nevybrána")
        self.folder_label.setWordWrap(True)
        left_layout.addWidget(self.folder_label)

        file_buttons_row = QHBoxLayout()
        file_buttons_row.setContentsMargins(0, 0, 0, 0)
        file_buttons_row.setSpacing(6)

        self.choose_btn = QPushButton("Vybrat složku")
        self.choose_btn.clicked.connect(self.choose_folder)
        file_buttons_row.addWidget(self.choose_btn)


        self.open_image_btn = QPushButton("Otevřít obrázek")
        self.open_image_btn.clicked.connect(self.open_image_file)
        file_buttons_row.addWidget(self.open_image_btn)
        left_layout.addLayout(file_buttons_row)

        self.pixinsight_btn = QPushButton("PixInsight")
        self.pixinsight_btn.setToolTip(self.tr_ui("pixinsight_tooltip"))
        self.pixinsight_btn.clicked.connect(self.launch_pixinsight_wrapper)
        self.pixinsight_btn.hide()

        self.preview_file_combo = ArrowComboBox()
        self.preview_file_combo.setMinimumWidth(1)
        self.preview_file_combo.setMinimumContentsLength(28)
        self.preview_file_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.preview_file_combo.currentIndexChanged.connect(self.on_preview_file_selected)
        left_layout.addWidget(self.preview_file_combo)
        self.stack_selection_label = QLabel("")
        self.stack_selection_label.setWordWrap(True)
        self.stack_selection_label.setStyleSheet("font-size: 11px; color: #9aa3b2;")
        self.stack_selection_label.setMinimumHeight(34)
        left_layout.addWidget(self.stack_selection_label)

        preview_nav = QHBoxLayout()
        self.preview_prev_btn = QPushButton("◀")
        self.preview_prev_btn.clicked.connect(self.preview_previous_image)
        preview_nav.addWidget(self.preview_prev_btn)
        self.preview_next_btn = QPushButton("▶")
        self.preview_next_btn.clicked.connect(self.preview_next_image)
        preview_nav.addWidget(self.preview_next_btn)
        left_layout.addLayout(preview_nav)

        self.language_combo = ArrowComboBox()
        self.language_combo.addItem("Čeština", "cz")
        self.language_combo.addItem("English", "en")
        self.language_combo.setCurrentIndex(max(0, self.language_combo.findData(self.language)))
        self.language_combo.currentIndexChanged.connect(self.change_language)

        self.theme_combo = ArrowComboBox()
        self.theme_combo.setFixedWidth(102)
        self.theme_combo.addItem(self.tr_ui("theme_dark"), "dark")
        self.theme_combo.addItem(self.tr_ui("theme_light"), "light")
        self.theme_combo.currentIndexChanged.connect(self.change_theme)

        self.completion_sound_check = QCheckBox("Zvuk hotovo")
        self.completion_sound_check.setChecked(True)
        self.completion_sound_check.setMinimumWidth(150)

        self.align_combo = ArrowComboBox()
        self.align_combo.setMinimumContentsLength(24)
        self.align_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.align_combo.addItem("Pouze posun", "translation")
        self.align_combo.addItem("Kalibrační snímky — bez zarovnání", "calibration")
        self.align_combo.addItem("Afinní ECC — posun/rotace/měřítko", "ecc_affine")
        self.align_combo.addItem("Star alignment — hvězdy + RANSAC", "star_affine")
        self.align_combo.addItem("Comet alignment — skládat na kometu", "comet")
        self.align_combo.addItem("Star + Comet — uložit zvlášť hvězdy a kometu", "comet_merge")
        self.align_combo.setCurrentIndex(self.align_combo.findData("star_affine"))

        self.stack_combo = ArrowComboBox()
        self.stack_combo.setMinimumContentsLength(18)
        self.stack_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.stack_combo.addItem("Sigma-clipped průměr", "sigma")
        self.stack_combo.addItem("Průměr s odmítnutím jasných pixelů", "high_rejection")
        self.stack_combo.addItem("Průměr", "mean")
        self.stack_combo.addItem("Medián", "median")
        self.stack_combo.setCurrentIndex(self.stack_combo.findData("median"))

        self.max_images_spin = ArrowSpinBox()
        self.max_images_spin.setRange(0, 100000)
        self.max_images_spin.setValue(0)
        self.max_images_spin.setSuffix(" snímků; 0 = vše")

        self.sigma_spin = ArrowSpinBox()
        self.sigma_spin.setRange(1, 5)
        self.sigma_spin.setValue(3)

        self.fit_only_check = QCheckBox("Pouze RAW")
        self.fit_only_check.setChecked(False)
        self.fit_only_check.setToolTip("Při skládání použije pouze XISF, FIT/FITS a foto RAW soubory; ignoruje JPG/PNG/BMP/TIFF ve stejné složce.")
        self.fit_only_check.stateChanged.connect(lambda _state: self.update_frame_quality_title())

        self.normalize_check = QCheckBox("Normalizovat pozadí")
        self.normalize_check.setChecked(True)

        self.mp_check = QCheckBox("Použít multiprocessing CPU")
        self.mp_check.setChecked(True)

        self.gpu_check = QCheckBox("Pouzit GPU (CUDA/Metal)")
        self.gpu_check.setChecked(False)
        self.gpu_check.setToolTip("Pouzije NVIDIA CUDA/CuPy nebo Apple Metal/MPS pres PyTorch. Pokud GPU neni dostupne, aplikace automaticky pouzije CPU.")

        self.process_mode_combo = ArrowComboBox()
        self.process_mode_combo.addItem(self.tr_ui("cpu_auto"), "auto")
        self.process_mode_combo.addItem(self.tr_ui("cpu_manual"), "manual")
        self.process_mode_combo.currentIndexChanged.connect(self.update_process_mode)

        self.processes_spin = ArrowSpinBox()
        max_cpu = max(1, os.cpu_count() or 1)
        self.processes_spin.setRange(1, max_cpu)
        self.processes_spin.setValue(self.optimal_cpu_processes())
        self.processes_spin.setSuffix(" procesů")
        self.processes_spin.setEnabled(False)

        self.auto_reference_check = QCheckBox("Automaticky vybrat nejlepší referenci")
        self.auto_reference_check.setChecked(True)
        self.auto_reference_check.stateChanged.connect(self.on_auto_reference_changed)

        self.manual_reference_btn = QPushButton("Použít aktuální snímek jako referenci")
        self.manual_reference_btn.clicked.connect(self.set_current_preview_as_reference)

        self.mosaic_mode_check = QCheckBox("Mozaika - rozšířit plátno")
        self.mosaic_mode_check.setChecked(False)
        self.mosaic_mode_check.setToolTip(self.tr_ui("mosaic_mode_tooltip"))

        self.quality_filter_check = QCheckBox("Použít jen nejlepší snímky")
        self.quality_filter_check.setChecked(False)

        self.review_frames_check = QCheckBox("Zkontrolovat snímky před skládáním")
        self.review_frames_check.setChecked(False)

        self.keep_percent_spin = ArrowSpinBox()
        self.keep_percent_spin.setRange(10, 100)
        self.keep_percent_spin.setValue(80)
        self.keep_percent_spin.setSuffix(" %")

        self.max_star_shift_spin = ArrowSpinBox()
        self.max_star_shift_spin.setRange(20, 3000)
        self.max_star_shift_spin.setValue(180)
        self.max_star_shift_spin.setSuffix(" px")
        self.max_star_shift_spin.setToolTip("Maximum expected star drift against the reference. For strong EAA dithering try 1200-1800 px.")

        self.max_comet_shift_spin = ArrowSpinBox()
        self.max_comet_shift_spin.setRange(20, 5000)
        self.max_comet_shift_spin.setValue(800)
        self.max_comet_shift_spin.setSuffix(" px")

        self.comet_refine_check = QCheckBox("Jemně doladit kometu")
        self.comet_refine_check.setChecked(True)

        self.comet_refine_patch_spin = ArrowSpinBox()
        self.comet_refine_patch_spin.setRange(10, 300)
        self.comet_refine_patch_spin.setValue(45)
        self.comet_refine_patch_spin.setSuffix(" px")

        self.comet_refine_search_spin = ArrowSpinBox()
        self.comet_refine_search_spin.setRange(10, 800)
        self.comet_refine_search_spin.setValue(90)
        self.comet_refine_search_spin.setSuffix(" px")

        self.star_border_margin_spin = ArrowSpinBox()
        self.star_border_margin_spin.setRange(0, 5000)
        self.star_border_margin_spin.setValue(120)
        self.star_border_margin_spin.setSuffix(" px")
        self.star_border_margin_spin.setToolTip("Kolik pixelů z každého okraje ignorovat při hledání hvězd. Pro snímky s větvemi/kometou zkus 500–1000 px.")

        self.strict_star_filter_check = QCheckBox("Přísný filtr hvězd")
        self.strict_star_filter_check.setChecked(True)

        self.satellite_trail_check = QCheckBox("Satelitní stopa")
        self.satellite_trail_check.setChecked(False)
        self.satellite_trail_check.setToolTip(self.tr_ui("satellite_trail_tooltip"))

        self.bayer_combo = ArrowComboBox()
        self.bayer_combo.setMinimumContentsLength(20)
        self.bayer_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.bayer_combo.addItem("Auto podle FIT hlavičky", "auto")
        self.bayer_combo.addItem("Mono / nedebayerovat", "mono")
        self.bayer_combo.addItem("RGGB", "RGGB")
        self.bayer_combo.addItem("BGGR", "BGGR")
        self.bayer_combo.addItem("GRBG", "GRBG")
        self.bayer_combo.addItem("GBRG", "GBRG")
        self.bayer_combo.setToolTip("Ruční override Bayer masky pro XISF/FIT/FITS. Auto použije metadata; Mono vynutí monochrom.")

        self.left_settings_tabs = QTabWidget()
        self.left_settings_tabs.setDocumentMode(True)
        self.left_settings_tabs.setUsesScrollButtons(False)
        self.left_settings_tabs.tabBar().setExpanding(True)
        self.left_settings_tabs.setStyleSheet(
            """
            QTabWidget::pane {
                border: 1px solid #303744;
                border-radius: 4px;
                background: #050505;
                top: -1px;
            }
            QTabWidget::tab-bar {
                background: #050505;
            }
            QTabBar {
                background: #050505;
                qproperty-drawBase: 0;
            }
            QTabBar::base {
                background: #050505;
                border: none;
            }
            QTabBar::tab {
                background: #050505;
                color: #f5f7fb;
                border: 1px solid #303744;
                border-bottom: none;
                padding: 7px 7px;
                margin-right: 2px;
                font-weight: 600;
                min-width: 52px;
            }
            QTabBar::tab:selected {
                background: #1f2937;
                color: #ffffff;
                border-color: #6aa7ff;
                border-top: 3px solid #6aa7ff;
                padding-top: 5px;
            }
            QTabBar::tab:hover:!selected {
                background: #111827;
                border-color: #4b5563;
            }
            """
        )
        left_layout.addWidget(self.left_settings_tabs)

        def make_left_tab(title_key: str) -> Tuple[QWidget, QVBoxLayout, QFormLayout]:
            page = QWidget()
            page_layout = QVBoxLayout(page)
            page_layout.setContentsMargins(4, 8, 4, 4)
            page_layout.setSpacing(8)
            page_form = QFormLayout()
            page_form.setContentsMargins(0, 0, 0, 0)
            page_form.setSpacing(6)
            page_layout.addLayout(page_form)
            page_layout.addStretch(1)
            self.left_settings_tabs.addTab(page, self.tr_ui(title_key))
            return page, page_layout, page_form

        self.setting_tab, setting_tab_layout, setting_form = make_left_tab("tab_setting")
        self.stars_alignment_tab, stars_alignment_tab_layout, stars_alignment_form = make_left_tab("tab_stars_alignment")
        self.comet_alignment_tab, comet_alignment_tab_layout, comet_alignment_form = make_left_tab("tab_comet_alignment")

        self.metadata_title = QLabel("Metadata")
        self.metadata_title.setStyleSheet("font-size: 14px; font-weight: bold; margin-top: 8px;")
        left_layout.addWidget(self.metadata_title)
        self.metadata_label = QLabel("-")
        self.metadata_label.setWordWrap(False)
        self.metadata_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.metadata_label.setStyleSheet("font-family: Consolas, monospace; font-size: 11px; color: #ddd; background: #151515; border: 1px solid #444; border-radius: 4px; padding: 6px;")
        self.metadata_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        self.metadata_scroll = QScrollArea()
        self.metadata_scroll.setWidgetResizable(True)
        self.metadata_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.metadata_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.metadata_scroll.setFrameShape(QFrame.NoFrame)
        self.metadata_scroll.setWidget(self.metadata_label)
        metadata_lines = 10
        metadata_height = self.metadata_label.fontMetrics().lineSpacing() * metadata_lines + 18
        self.metadata_scroll.setMinimumHeight(metadata_height)
        self.metadata_scroll.setMaximumHeight(metadata_height)
        left_layout.addWidget(self.metadata_scroll)

        def _mark_simple_hidden(*widgets):
            for widget in widgets:
                if widget is not None and widget not in self.simple_mode_hidden_widgets:
                    self.simple_mode_hidden_widgets.append(widget)

        def add_tab_form_row(form: QFormLayout, label_text, field_widget, advanced: bool = False):
            form.addRow(label_text, field_widget)
            if advanced:
                _mark_simple_hidden(form.labelForField(field_widget), field_widget)

        def add_tab_form_layout_row(form: QFormLayout, label_text, row_layout, advanced: bool = False):
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_widget = QWidget()
            row_widget.setLayout(row_layout)
            add_tab_form_row(form, label_text, row_widget, advanced=advanced)
            return row_widget

        def add_setting_row(label_text, field_widget, advanced: bool = False):
            add_tab_form_row(setting_form, label_text, field_widget, advanced)

        def add_setting_layout_row(label_text, row_layout, advanced: bool = False):
            return add_tab_form_layout_row(setting_form, label_text, row_layout, advanced)

        def add_stars_row(label_text, field_widget, advanced: bool = False):
            add_tab_form_row(stars_alignment_form, label_text, field_widget, advanced)

        def add_comet_row(label_text, field_widget, advanced: bool = False):
            add_tab_form_row(comet_alignment_form, label_text, field_widget, advanced)

        add_setting_row("Jazyk", self.language_combo)
        theme_row = QHBoxLayout()
        theme_row.addWidget(self.theme_combo)
        theme_row.addWidget(self.completion_sound_check)
        theme_row.addStretch(1)
        add_setting_layout_row("Motiv", theme_row)
        add_setting_row("Zarovnání", self.align_combo)
        add_setting_row("Skládání", self.stack_combo)
        add_setting_row("Max. snímků", self.max_images_spin)
        add_setting_row("", self.fit_only_check)
        add_setting_row("Sigma", self.sigma_spin)
        add_setting_row("", self.normalize_check, advanced=True)
        add_setting_row("", self.mp_check, advanced=True)
        if not LITE_BUILD:
            add_setting_row("", self.gpu_check, advanced=True)
        process_row = QHBoxLayout()
        process_row.addWidget(self.process_mode_combo)
        process_row.addWidget(self.processes_spin)
        add_setting_layout_row("CPU procesy", process_row, advanced=True)
        add_setting_row("Bayer FIT", self.bayer_combo, advanced=True)

        add_stars_row("", self.auto_reference_check)
        add_stars_row("", self.review_frames_check, advanced=True)
        add_stars_row("", self.manual_reference_btn, advanced=True)
        add_stars_row("", self.mosaic_mode_check, advanced=True)
        add_stars_row("", self.quality_filter_check, advanced=True)
        add_stars_row("Ponechat", self.keep_percent_spin, advanced=True)
        add_stars_row("Max. drift hvězd", self.max_star_shift_spin)
        add_stars_row("Ignorovat okraj", self.star_border_margin_spin, advanced=True)
        add_stars_row("", self.strict_star_filter_check)
        add_stars_row("", self.satellite_trail_check, advanced=True)

        add_comet_row("Max. pohyb komety", self.max_comet_shift_spin, advanced=True)
        add_comet_row("", self.comet_refine_check, advanced=True)
        add_comet_row("Šablona komety", self.comet_refine_patch_spin, advanced=True)
        add_comet_row("Hledání komety", self.comet_refine_search_spin, advanced=True)

        comet_row = QHBoxLayout()
        comet_row.setSpacing(4)

        self.comet_select_btn = QPushButton("První")
        self.comet_select_btn.setStyleSheet("font-weight: bold; padding: 4px 6px;")
        self.comet_select_btn.setToolTip(self.tr_ui("comet_first_tooltip"))
        self.comet_select_btn.clicked.connect(self.select_comet_start_point)
        comet_row.addWidget(self.comet_select_btn)

        self.comet_end_btn = QPushButton("Poslední")
        self.comet_end_btn.setStyleSheet("font-weight: bold; padding: 4px 6px;")
        self.comet_end_btn.setToolTip(self.tr_ui("comet_last_tooltip"))
        self.comet_end_btn.clicked.connect(self.select_comet_end_point)
        comet_row.addWidget(self.comet_end_btn)

        self.clear_comet_marks_btn = QPushButton("Smazat")
        self.clear_comet_marks_btn.setStyleSheet("font-weight: bold; padding: 4px 6px;")
        self.clear_comet_marks_btn.setToolTip(self.tr_ui("comet_clear_tooltip"))
        self.clear_comet_marks_btn.clicked.connect(self.clear_comet_marks)
        comet_row.addWidget(self.clear_comet_marks_btn)
        comet_row_widget = QWidget()
        comet_row_widget.setLayout(comet_row)
        comet_alignment_tab_layout.insertWidget(0, comet_row_widget)
        _mark_simple_hidden(comet_row_widget)

        self.stack_btn = QPushButton("Spustit skládání")
        self.stack_btn.setStyleSheet("font-weight: bold; padding: 8px;")
        self.stack_btn.setFixedHeight(38)
        self.stack_btn.clicked.connect(self.start_stack)

        self.clear_cache_btn = QPushButton("Smazat cache")
        self.clear_cache_btn.setFixedHeight(38)
        self.clear_cache_btn.setToolTip(self.tr_ui("clear_cache_tooltip"))
        self.clear_cache_btn.clicked.connect(self.clear_cache_dialog)
        _mark_simple_hidden(self.clear_cache_btn)

        stack_action_row = QHBoxLayout()
        stack_action_row.setSpacing(4)
        stack_action_row.addWidget(self.stack_btn, 1)
        stack_action_row.addWidget(self.clear_cache_btn, 1)
        left_layout.addLayout(stack_action_row)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_stack)
        left_layout.addWidget(self.stop_btn)

        # Progress is presented in the central activity overlay. Keep this
        # internal meter for worker state without duplicating it in the panel.
        self.progress = QProgressBar(left)
        self.progress.hide()
        self.status_label = QLabel("Připraveno")
        self.status_label.setWordWrap(True)
        left_layout.addWidget(self.status_label)

        # Pravý panel: úpravy obrazu a zoom.
        # Dříve byly tyto ovládací prvky v levém panelu, ale u menších obrazovek
        # je pohodlnější mít workflow/stack vlevo a post-processing vpravo.
        right_panel = QFrame()
        right_panel.setFrameShape(QFrame.StyledPanel)
        right_panel.setFixedWidth(side_content_width)
        right_panel_layout = QVBoxLayout(right_panel)

        right_panel_layout.setContentsMargins(8, 8, 8, 8)
        right_panel_layout.setSpacing(8)

        self.calib_title = QLabel("Kalibrace")
        self.calib_title.setAlignment(Qt.AlignCenter)
        self.calib_title.setStyleSheet("font-size: 18px; font-weight: bold; margin: 0; padding: 0;")

        right_header = QHBoxLayout()
        right_header.setContentsMargins(0, 0, 0, 0)
        right_header.setSpacing(4)
        self.right_collapse_btn = self._panel_toggle_button("▶", self.tr_ui("hide_right_panel"))
        self.right_collapse_btn.clicked.connect(lambda: self.set_right_panel_collapsed(True))
        right_header.addWidget(self.right_collapse_btn, 0, Qt.AlignTop)
        right_header.addWidget(self.calib_title, 1)
        right_header.addSpacing(28)
        right_panel_layout.addLayout(right_header)

        flat_row = QHBoxLayout()
        self.flat_label = QLabel("Flat: nepoužit")
        flat_row.addWidget(self.flat_label, 1)
        flat_btn = QPushButton("Flat")
        flat_btn.setFixedWidth(64)
        flat_btn.clicked.connect(self.choose_flat_frame)
        flat_row.addWidget(flat_btn)
        right_panel_layout.addLayout(flat_row)

        bias_row = QHBoxLayout()
        self.bias_label = QLabel("Bias: nepoužit")
        bias_row.addWidget(self.bias_label, 1)
        bias_btn = QPushButton("Bias")
        bias_btn.setFixedWidth(64)
        bias_btn.clicked.connect(self.choose_bias_frame)
        bias_row.addWidget(bias_btn)
        right_panel_layout.addLayout(bias_row)

        dark_row = QHBoxLayout()
        self.dark_label = QLabel("Dark: nepoužit")
        dark_row.addWidget(self.dark_label, 1)
        dark_btn = QPushButton("Dark")
        dark_btn.setFixedWidth(64)
        dark_btn.clicked.connect(self.choose_dark_frame)
        dark_row.addWidget(dark_btn)
        right_panel_layout.addLayout(dark_row)

        self.clear_calib_btn = QPushButton("Reset kalibrace")
        self.clear_calib_btn.clicked.connect(self.clear_calibration_frames)
        right_panel_layout.addWidget(self.clear_calib_btn)

        self.show_stacked_btn = QPushButton("Zobraz složený obraz")
        self.show_stacked_btn.setToolTip(self.tr_ui("show_stacked_tooltip"))
        self.show_stacked_btn.clicked.connect(self.show_original_stacked_image)
        right_panel_layout.addWidget(self.show_stacked_btn)

        self.adjustment_tabs = QTabWidget()
        self.adjustment_tabs.setDocumentMode(True)
        self.adjustment_tabs.setUsesScrollButtons(False)
        self.adjustment_tabs.tabBar().setExpanding(True)
        self.adjustment_tabs.setStyleSheet(
            """
            QTabWidget::pane {
                border: 1px solid #303744;
                border-radius: 4px;
                background: #050505;
                top: -1px;
            }
            QTabWidget::tab-bar {
                background: #050505;
            }
            QTabBar {
                background: #050505;
                qproperty-drawBase: 0;
            }
            QTabBar::base {
                background: #050505;
                border: none;
            }
            QTabBar::tab {
                background: #050505;
                color: #f5f7fb;
                border: 1px solid #303744;
                border-bottom: none;
                padding: 7px 8px;
                margin-right: 2px;
                font-weight: 600;
                min-width: 58px;
            }
            QTabBar::tab:selected {
                background: #1f2937;
                color: #ffffff;
                border-color: #6aa7ff;
                border-top: 3px solid #6aa7ff;
                padding-top: 5px;
            }
            QTabBar::tab:hover:!selected {
                background: #111827;
                border-color: #4b5563;
            }
            QTabBar::tab:first {
                border-top-left-radius: 4px;
            }
            QTabBar::tab:last {
                border-top-right-radius: 4px;
            }
            """
        )
        right_panel_layout.addWidget(self.adjustment_tabs)

        def make_adjustment_tab(title_key: str) -> Tuple[QWidget, QVBoxLayout, QFormLayout]:
            page = QWidget()
            page_layout = QVBoxLayout(page)
            page_layout.setContentsMargins(4, 8, 4, 4)
            page_layout.setSpacing(8)
            form = QFormLayout()
            form.setContentsMargins(0, 0, 0, 0)
            form.setSpacing(6)
            page_layout.addLayout(form)
            page_layout.addStretch(1)
            self.adjustment_tabs.addTab(page, self.tr_ui(title_key))
            return page, page_layout, form

        self.color_tab, color_tab_layout, color_form = make_adjustment_tab("tab_color")
        self.correction_tab, correction_tab_layout, correction_form = make_adjustment_tab("tab_correction")
        self.ai_tools_tab, ai_tab_layout, ai_form = make_adjustment_tab("tab_ai_tools")

        stretch_header = QHBoxLayout()
        stretch_title = QLabel("Křivky / barvy / kontrast")
        stretch_title.setStyleSheet("font-size: 16px; font-weight: bold;")
        stretch_header.addWidget(stretch_title, 1)

        self.black_slider = self._slider(0, 65535, 0)
        self.white_slider = self._slider(1, 65535, 65535)
        curve_step = int(round(65535.0 * 0.05 / 10.0))
        for curve_slider in (self.black_slider, self.white_slider):
            curve_slider.setSingleStep(curve_step)
            curve_slider.setPageStep(curve_step)
        self.gamma_slider = self._slider(10, 400, 68)
        self.contrast_slider = self._slider(10, 300, 71)
        self.saturation_slider = self._slider(0, 300, 100)
        self.red_slider = self._slider(0, 300, 100)
        self.green_slider = self._slider(0, 300, 100)
        self.blue_slider = self._slider(0, 300, 100)

        def add_tab_row(form: QFormLayout, label_text, field_widget, advanced: bool = False):
            form.addRow(label_text, field_widget)
            if advanced:
                _mark_simple_hidden(form.labelForField(field_widget), field_widget)

        def add_color_row(label_text, field_widget, advanced: bool = False):
            add_tab_row(color_form, label_text, field_widget, advanced)

        def add_correction_row(label_text, field_widget, advanced: bool = False):
            add_tab_row(correction_form, label_text, field_widget, advanced)

        def add_ai_row(label_text, field_widget, advanced: bool = False):
            add_tab_row(ai_form, label_text, field_widget, advanced)

        add_color_row("Black point", self.slider_with_value(self.black_slider))
        add_color_row("White point", self.slider_with_value(self.white_slider))
        add_color_row("Gamma", self.slider_with_value(self.gamma_slider))

        self.auto_stretch_btn = QPushButton("")
        self.auto_stretch_btn.setToolTip(self.tr_ui("auto_stretch_tooltip"))
        balance_icon_path = bundled_file_path("AS_balance_icon.png")
        if balance_icon_path is not None:
            self.auto_stretch_btn.setIcon(QIcon(str(balance_icon_path)))
            self.auto_stretch_btn.setIconSize(QSize(45, 45))
            self.auto_stretch_btn.setFixedSize(49, 49)
            self.auto_stretch_btn.setStyleSheet("QPushButton { border: none; background: transparent; padding: 2px; } QPushButton:hover { background: rgba(255,255,255,0.08); border-radius: 6px; }")
        self.auto_stretch_btn.clicked.connect(self.auto_stretch_preview)
        stretch_header.addWidget(self.auto_stretch_btn, 0, Qt.AlignRight | Qt.AlignVCenter)
        stretch_header.addSpacing(24)
        color_tab_layout.insertLayout(0, stretch_header)

        self.stf_strength_slider = QSlider(Qt.Horizontal)

        self.stf_strength_slider.setRange(0, 100)

        self.stf_strength_slider.setValue(50)

        self._connect_preview_slider(self.stf_strength_slider)
        self.stf_strength_slider.setToolTip(self.tr_ui("stf_strength_tooltip"))

        add_correction_row("Síla STF", self.slider_with_value(self.stf_strength_slider), advanced=True)

        self.vignette_removal_slider = QSlider(Qt.Horizontal)
        self.vignette_removal_slider.setRange(0, 100)
        self.vignette_removal_slider.setValue(0)
        self._connect_preview_slider(self.vignette_removal_slider)
        self.vignette_removal_slider.setToolTip("Zesvětlí okraje a rohy pro jemné potlačení vinětace. Maximum odpovídá asi 30 % původní síly.")
        add_correction_row("Odstranění vinětace", self.slider_with_value(self.vignette_removal_slider), advanced=True)

        self.synthetic_flat_slider = QSlider(Qt.Horizontal)
        self.synthetic_flat_slider.setRange(0, 100)
        self.synthetic_flat_slider.setValue(0)
        self._connect_preview_slider(self.synthetic_flat_slider)
        self.synthetic_flat_slider.setToolTip("Odhadne hladké pozadí ze složeného obrazu a použije ho jako jemný umělý flat pro náhled a PNG/JPG/TIFF export.")
        add_correction_row("Umělý flat", self.slider_with_value(self.synthetic_flat_slider), advanced=True)

        self.color_background_slider = QSlider(Qt.Horizontal)
        self.color_background_slider.setRange(0, 100)
        self.color_background_slider.setValue(0)
        self._connect_preview_slider(self.color_background_slider)
        self.color_background_slider.setToolTip("Potlačí hladký barevný závoj pozadí po jednotlivých RGB kanálech. Vhodné pro růžové/fialové pozadí z chytrých dalekohledů.")
        add_correction_row("Korekce barevného pozadí", self.slider_with_value(self.color_background_slider), advanced=True)

        self.denoise_slider = QSlider(Qt.Horizontal)
        self.denoise_slider.setRange(0, 100)
        self.denoise_slider.setValue(0)
        self._connect_preview_slider(self.denoise_slider)
        self.denoise_slider.setToolTip(self.tr_ui("astro_denoise_tooltip"))

        denoise_mode_widget = QWidget()
        denoise_mode_layout = QHBoxLayout(denoise_mode_widget)
        denoise_mode_layout.setContentsMargins(0, 0, 0, 0)
        denoise_mode_layout.setSpacing(6)
        self.denoise_mode_combo = ArrowComboBox()
        self.denoise_mode_combo.addItem(self.tr_ui("denoise_classic"), "classic")
        if not LITE_BUILD:
            self.denoise_mode_combo.addItem(self.tr_ui("denoise_drunet"), "drunet")
        self.denoise_mode_combo.setToolTip(self.tr_ui("drunet_tooltip"))
        self.denoise_mode_combo.currentIndexChanged.connect(self.on_denoise_mode_changed)
        denoise_mode_layout.addWidget(self.denoise_mode_combo, 1)
        self.select_drunet_model_btn = QPushButton(self.tr_ui("select_drunet_model"))
        self.select_drunet_model_btn.setToolTip(self.tr_ui("drunet_tooltip"))
        self.select_drunet_model_btn.clicked.connect(self.choose_selected_denoise_model)
        if not LITE_BUILD:
            denoise_mode_layout.addWidget(self.select_drunet_model_btn)
        add_ai_row(self.tr_ui("denoise_mode"), denoise_mode_widget, advanced=True)
        add_ai_row("Astro odšumění", self.slider_with_value(self.denoise_slider), advanced=True)

        self.star_deconvolution_slider = QSlider(Qt.Horizontal)
        self.star_deconvolution_slider.setRange(0, 100)
        self.star_deconvolution_slider.setValue(0)
        self._connect_preview_slider(self.star_deconvolution_slider)
        self.star_deconvolution_slider.setToolTip(
            self.tr_ui("star_deconvolution_tooltip")
        )
        stellar_control = QWidget()
        stellar_control_layout = QHBoxLayout(stellar_control)
        stellar_control_layout.setContentsMargins(0, 0, 0, 0)
        stellar_control_layout.setSpacing(6)
        stellar_control_layout.addWidget(
            self.slider_with_value(self.star_deconvolution_slider),
            1,
        )
        self.select_stellar_model_btn = QPushButton(
            self.tr_ui("select_stellar_model")
        )
        self.select_stellar_model_btn.setToolTip(
            self.tr_ui("star_deconvolution_tooltip")
        )
        self.select_stellar_model_btn.clicked.connect(
            self.choose_stellar_deconvolution_model
        )
        stellar_control_layout.addWidget(self.select_stellar_model_btn)
        add_ai_row(
            self.tr_ui("star_deconvolution"),
            stellar_control,
            advanced=True,
        )

        self.star_deconvolution_size_slider = QSlider(Qt.Horizontal)
        self.star_deconvolution_size_slider.setRange(1, 10)
        self.star_deconvolution_size_slider.setValue(5)
        self._connect_preview_slider(self.star_deconvolution_size_slider)
        self.star_deconvolution_size_slider.setToolTip(
            self.tr_ui("star_deconvolution_tooltip")
        )
        add_ai_row(
            self.tr_ui("star_deconvolution_size"),
            self.slider_with_value(self.star_deconvolution_size_slider),
            advanced=True,
        )

        self.scnr_green_slider = QSlider(Qt.Horizontal)


        self.scnr_green_slider.setRange(0, 10)


        self.scnr_green_slider.setValue(0)


        self._connect_preview_slider(self.scnr_green_slider)

        add_correction_row("SCNR Green", self.slider_with_value(self.scnr_green_slider), advanced=True)
        add_color_row("Kontrast", self.slider_with_value(self.contrast_slider), advanced=True)
        add_color_row("Saturace", self.slider_with_value(self.saturation_slider), advanced=True)
        add_color_row("Červená", self.slider_with_value(self.red_slider), advanced=True)
        add_color_row("Zelená", self.slider_with_value(self.green_slider), advanced=True)
        add_color_row("Modrá", self.slider_with_value(self.blue_slider), advanced=True)

        self.hist_title = QLabel("Histogram L/R/G/B")
        self.hist_title.setStyleSheet("font-size: 14px; font-weight: bold; margin-top: 8px;")
        right_panel_layout.addWidget(self.hist_title)
        self.histogram_label = QLabel()
        self.histogram_label.setMinimumHeight(120)
        self.histogram_label.setMaximumHeight(120)
        self.histogram_label.setStyleSheet("background: #121212; border: 1px solid #444; border-radius: 4px;")
        right_panel_layout.addWidget(self.histogram_label)

        compact_button_block = QVBoxLayout()
        compact_button_block.setContentsMargins(0, 0, 0, 0)
        compact_button_block.setSpacing(5)

        crop_transform_row = QHBoxLayout()
        crop_transform_row.setContentsMargins(0, 0, 0, 0)
        crop_transform_row.setSpacing(3)
        self.crop_edges_btn = QPushButton("Turn angle")
        self.crop_edges_btn.setToolTip(self.tr_ui("crop_tooltip"))
        self.crop_edges_btn.clicked.connect(self.rotate_current_image_by_crop_angle)
        crop_transform_row.addWidget(self.crop_edges_btn)

        self.crop_angle_spin = ArrowSpinBox()
        self.crop_angle_spin.setRange(-15, 15)
        self.crop_angle_spin.setValue(0)
        self.crop_angle_spin.setSuffix("°")
        self.crop_angle_spin.setToolTip(self.tr_ui("crop_tooltip"))
        self.crop_angle_spin.setMinimumWidth(64)
        crop_transform_row.addWidget(self.crop_angle_spin)

        self.crop_select_btn = QPushButton("Výběr")
        self.crop_select_btn.setMinimumWidth(58)
        self.crop_select_btn.setToolTip(self.tr_ui("crop_select_tooltip"))
        self.crop_select_btn.clicked.connect(self.start_manual_crop_selection)
        crop_transform_row.addWidget(self.crop_select_btn)
        crop_transform_row_widget = QWidget()
        crop_transform_row_widget.setLayout(crop_transform_row)
        compact_button_block.addWidget(crop_transform_row_widget)
        _mark_simple_hidden(crop_transform_row_widget)

        neutralize_row = QHBoxLayout()
        neutralize_row.setContentsMargins(0, 0, 0, 0)
        neutralize_row.setSpacing(4)
        self.neutral_bg_btn = QPushButton("Neutralizovat pozadí")
        self.neutral_bg_btn.clicked.connect(self.neutralize_background)
        neutralize_row.addWidget(self.neutral_bg_btn)

        self.clear_neutral_bg_btn = QPushButton("Zrušit neutralizaci")
        self.clear_neutral_bg_btn.clicked.connect(self.clear_background_neutralization)
        neutralize_row.addWidget(self.clear_neutral_bg_btn)
        neutralize_row_widget = QWidget()
        neutralize_row_widget.setLayout(neutralize_row)
        correction_tab_layout.insertWidget(0, neutralize_row_widget)
        _mark_simple_hidden(neutralize_row_widget)

        gradient_row = QHBoxLayout()
        gradient_row.setContentsMargins(0, 0, 0, 0)
        gradient_row.setSpacing(4)
        self.remove_gradient_btn = QPushButton("Odstranit gradient")
        self.remove_gradient_btn.setToolTip(self.tr_ui("gradient_tooltip"))
        self.remove_gradient_btn.clicked.connect(self.remove_background_gradient)
        gradient_row.addWidget(self.remove_gradient_btn)
        self.clear_gradient_btn = QPushButton("Zrušit gradient")
        self.clear_gradient_btn.setToolTip(self.tr_ui("gradient_tooltip"))
        self.clear_gradient_btn.clicked.connect(self.clear_background_gradient)
        gradient_row.addWidget(self.clear_gradient_btn)
        gradient_row_widget = QWidget()
        gradient_row_widget.setLayout(gradient_row)
        correction_tab_layout.insertWidget(1, gradient_row_widget)
        _mark_simple_hidden(gradient_row_widget)

        transform_row = QHBoxLayout()
        transform_row.setContentsMargins(0, 0, 0, 0)
        transform_row.setSpacing(4)
        self.flip_h_btn = QPushButton("Flip H")
        self.flip_h_btn.clicked.connect(self.toggle_flip_horizontal)
        transform_row.addWidget(self.flip_h_btn)

        self.flip_v_btn = QPushButton("Flip V")
        self.flip_v_btn.clicked.connect(self.toggle_flip_vertical)
        transform_row.addWidget(self.flip_v_btn)

        self.rotate_left_btn = QPushButton("Left 90")
        self.rotate_left_btn.clicked.connect(self.rotate_preview_left)
        transform_row.addWidget(self.rotate_left_btn)

        self.rotate_right_btn = QPushButton("Right 90")
        self.rotate_right_btn.clicked.connect(self.rotate_preview_right)
        transform_row.addWidget(self.rotate_right_btn)
        transform_row_widget = QWidget()
        transform_row_widget.setLayout(transform_row)
        compact_button_block.addWidget(transform_row_widget)
        _mark_simple_hidden(transform_row_widget)

        right_panel_layout.addLayout(compact_button_block)
        right_panel_layout.addSpacing(2)
        self.preview_view_title = QLabel("Zobrazení náhledu")
        self.preview_view_title.setStyleSheet("font-size: 14px; font-weight: bold; margin-top: 4px;")
        right_panel_layout.addWidget(self.preview_view_title)

        preview_view_row = QHBoxLayout()
        self.fit_btn = QPushButton("Fit")
        self.fit_btn.clicked.connect(self.zoom_fit)
        preview_view_row.addWidget(self.fit_btn)

        self.actual_btn = QPushButton("1:1")
        self.actual_btn.clicked.connect(self.zoom_actual_size)
        preview_view_row.addWidget(self.actual_btn)

        self.zoom_label = QLabel("Fit")
        self.zoom_label.setAlignment(Qt.AlignCenter)
        self.zoom_label.setMinimumWidth(72)
        self.zoom_label.setStyleSheet("font-size: 12px; font-weight: bold;")
        preview_view_row.addWidget(self.zoom_label)
        right_panel_layout.addLayout(preview_view_row)

        reset_row = QHBoxLayout()
        reset_row.setContentsMargins(0, 0, 0, 0)
        reset_row.setSpacing(4)
        self.undo_btn = QPushButton("Zpět")
        self.undo_btn.setToolTip(self.tr_ui("undo_tooltip"))
        self.undo_btn.clicked.connect(self.undo_preview_adjustment)
        reset_row.addWidget(self.undo_btn, 1)
        self.update_undo_button_state()

        self.reset_btn = QPushButton("Reset úprav")
        self.reset_btn.clicked.connect(self.reset_stretch)
        reset_row.addWidget(self.reset_btn, 1)
        right_panel_layout.addLayout(reset_row)
        right_panel_layout.addStretch()

        right = QVBoxLayout()
        layout.addLayout(right, stretch=1)

        self.image_label = ClickableImageLabel(self.tr_ui("preview_prompt"))
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(1, 1)
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.image_label.setStyleSheet("background: #111; color: #ddd; font-size: 18px;")
        self.image_label.image_clicked.connect(self.on_preview_clicked)
        self.image_label.crop_selected.connect(self.on_preview_crop_selected)
        self.image_label.image_double_clicked.connect(self.open_fullscreen_preview)
        self.image_label.wheel_zoomed.connect(self.on_preview_wheel_zoom)
        self.image_label.drag_started.connect(self.on_preview_drag_started)
        self.image_label.drag_moved.connect(self.on_preview_drag_moved)
        self.image_label.drag_finished.connect(self.on_preview_drag_finished)
        self.preview_drag_start_scroll = (0, 0)
        self.preview_pan_scroll = (0.0, 0.0)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setAlignment(Qt.AlignCenter)
        self.scroll.setWidget(self.image_label)
        preview_stack = QWidget()
        preview_stack_layout = QGridLayout(preview_stack)
        preview_stack_layout.setContentsMargins(0, 0, 0, 0)
        preview_stack_layout.setSpacing(0)
        preview_stack_layout.addWidget(self.scroll, 0, 0)

        self.zoom_overlay = QLabel("")
        self.zoom_overlay.setStyleSheet(
            "background: rgba(0, 0, 0, 150); color: #f8fafc; "
            "border: 1px solid rgba(255, 255, 255, 90); border-radius: 4px; "
            "padding: 4px 8px; font-size: 12px; font-weight: bold;"
        )
        self.zoom_overlay.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.zoom_overlay.hide()
        preview_stack_layout.addWidget(self.zoom_overlay, 0, 0, Qt.AlignTop | Qt.AlignLeft)
        right.addWidget(preview_stack, stretch=3)
        self.fullscreen_preview = None

        self.frame_quality_title = QLabel(self.tr_ui("frame_quality_title"))
        self.frame_quality_title.setStyleSheet("font-size: 13px; font-weight: bold; margin-top: 4px;")
        right.addWidget(self.frame_quality_title)
        self.update_frame_quality_title()

        self.frame_quality_table = FrameQualityTable(0, 10)
        self.frame_quality_table.setHorizontalHeaderLabels(self.tr_ui("frame_quality_headers"))
        self.frame_quality_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.frame_quality_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.frame_quality_table.setAlternatingRowColors(True)
        self.frame_quality_table.setSortingEnabled(True)
        self.frame_quality_table.verticalHeader().setVisible(False)
        self.frame_quality_table.verticalHeader().setDefaultSectionSize(20)
        self.frame_quality_table.verticalHeader().setMinimumSectionSize(18)
        self.frame_quality_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.frame_quality_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.frame_quality_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.frame_quality_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.frame_quality_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.frame_quality_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self.frame_quality_table.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeToContents)
        self.frame_quality_table.horizontalHeader().setSectionResizeMode(7, QHeaderView.ResizeToContents)
        self.frame_quality_table.horizontalHeader().setSectionResizeMode(8, QHeaderView.ResizeToContents)
        self.frame_quality_table.horizontalHeader().setSectionResizeMode(9, QHeaderView.ResizeToContents)
        self.frame_quality_table.setMinimumHeight(120)
        self.frame_quality_table.setMaximumHeight(220)
        self.frame_quality_table.setStyleSheet(
            "QTableWidget { background: #252a33; alternate-background-color: #2d333d; "
            "color: #d9dee7; gridline-color: #3f4754; selection-background-color: #3f5f8f; "
            "selection-color: #ffffff; border: 1px solid #454d59; } "
            "QHeaderView::section { background: #343b47; color: #f1f5f9; "
            "padding: 3px 5px; border: 1px solid #454d59; font-weight: bold; }"
        )
        self.frame_quality_table.itemSelectionChanged.connect(self.on_frame_quality_selection_changed)
        self.frame_quality_table.toggle_requested.connect(self.toggle_selected_frame_excluded)
        right.addWidget(self.frame_quality_table, stretch=1)

        right_panel_scroll = QScrollArea()
        right_panel_scroll.setWidgetResizable(True)
        right_panel_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        right_panel_scroll.setWidget(right_panel)
        right_panel_scroll.setFixedWidth(side_scroll_width)
        self.right_expand_btn = self._panel_toggle_button("◀", self.tr_ui("show_right_panel"))
        self.right_expand_btn.clicked.connect(lambda: self.set_right_panel_collapsed(False))
        self.right_panel_rail = self._make_panel_rail(self.right_expand_btn)
        self.right_panel_rail.hide()
        layout.addWidget(self.right_panel_rail)
        self.right_panel_scroll = right_panel_scroll
        layout.addWidget(right_panel_scroll)
        self.apply_ui_mode()
        self._build_activity_overlay()

    def _build_activity_overlay(self):
        self.activity_overlay_generation = 0
        self.activity_overlay = QFrame(self)
        self.activity_overlay.setObjectName("activityOverlay")
        self.activity_overlay.setAttribute(Qt.WA_StyledBackground, True)
        self.activity_overlay.setFixedWidth(440)

        overlay_layout = QVBoxLayout(self.activity_overlay)
        overlay_layout.setContentsMargins(18, 14, 18, 16)
        overlay_layout.setSpacing(9)

        self.activity_title_label = QLabel(self.tr_ui("activity_title"))
        self.activity_title_label.setObjectName("activityTitle")
        self.activity_title_label.setAlignment(Qt.AlignCenter)
        overlay_layout.addWidget(self.activity_title_label)

        self.activity_message_label = QLabel("")
        self.activity_message_label.setObjectName("activityMessage")
        self.activity_message_label.setAlignment(Qt.AlignCenter)
        self.activity_message_label.setWordWrap(True)
        self.activity_message_label.setMinimumHeight(34)
        overlay_layout.addWidget(self.activity_message_label)

        self.activity_progress = QProgressBar()
        self.activity_progress.setRange(0, 100)
        self.activity_progress.setValue(0)
        self.activity_progress.setTextVisible(True)
        overlay_layout.addWidget(self.activity_progress)

        self._update_activity_overlay_theme()
        self.activity_overlay.hide()

    def _update_activity_overlay_theme(self):
        if not hasattr(self, "activity_overlay"):
            return
        theme = self.theme_combo.currentData() if hasattr(self, "theme_combo") else "dark"
        if theme == "light":
            background, border, title, text = "#f7f9fc", "#8b98a8", "#18202a", "#344150"
        else:
            background, border, title, text = "#20242b", "#5c6673", "#ffffff", "#d8dde5"
        self.activity_overlay.setStyleSheet(
            f"""
            QFrame#activityOverlay {{
                background-color: {background};
                border: 1px solid {border};
                border-radius: 8px;
            }}
            QLabel#activityTitle {{
                color: {title};
                border: none;
                font-size: 15px;
                font-weight: bold;
            }}
            QLabel#activityMessage {{
                color: {text};
                border: none;
                font-size: 12px;
            }}
            """
        )

    def _position_activity_overlay(self):
        if not hasattr(self, "activity_overlay") or not self.activity_overlay.isVisible():
            return
        self.activity_overlay.adjustSize()
        width = self.activity_overlay.width()
        height = max(112, self.activity_overlay.height())
        self.activity_overlay.resize(width, height)
        x = max(12, (self.width() - width) // 2)
        y = max(self.menuBar().height() + 12, (self.height() - height) // 2)
        self.activity_overlay.move(x, y)
        self.activity_overlay.raise_()

    def begin_activity(self, message: str, indeterminate: bool = False):
        if not hasattr(self, "activity_overlay"):
            return
        self.activity_overlay_generation += 1
        self.activity_title_label.setText(self.tr_ui("activity_title"))
        self.activity_message_label.setText(str(message))
        if indeterminate:
            self.activity_progress.setRange(0, 0)
        else:
            self.activity_progress.setRange(0, 100)
            self.activity_progress.setValue(0)
        self.activity_overlay.show()
        self._position_activity_overlay()
        self.activity_overlay.repaint()

    def flush_activity_overlay(self):
        """Show the overlay before a blocking task without moving the preview."""
        zoom_state = (
            self.capture_preview_zoom_state()
            if hasattr(self, "capture_preview_zoom_state")
            else None
        )
        if hasattr(self, "preview_render_timer"):
            self.preview_render_timer.stop()

        QApplication.processEvents(
            QEventLoop.ExcludeUserInputEvents | QEventLoop.ExcludeSocketNotifiers
        )

        if zoom_state is not None:
            self.restore_preview_zoom_state(zoom_state)

    def update_activity(self, value: int, message: str):
        if not hasattr(self, "activity_overlay"):
            return
        self.activity_message_label.setText(str(message))
        self.activity_progress.setRange(0, 100)
        self.activity_progress.setValue(max(0, min(100, int(value))))
        if not self.activity_overlay.isVisible():
            self.activity_overlay.show()
        self._position_activity_overlay()

    def finish_activity(self, message: str, value: Optional[int] = None, delay_ms: int = 1200):
        if not hasattr(self, "activity_overlay"):
            return
        self.activity_message_label.setText(str(message))
        if value is not None:
            self.activity_progress.setRange(0, 100)
            self.activity_progress.setValue(max(0, min(100, int(value))))
        self.activity_overlay.show()
        self._position_activity_overlay()
        generation = self.activity_overlay_generation
        QTimer.singleShot(
            max(0, int(delay_ms)),
            lambda: (
                self.activity_overlay.hide()
                if generation == self.activity_overlay_generation
                else None
            ),
        )

    def _slider(self, min_v: int, max_v: int, value: int) -> QSlider:
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_v, max_v)
        slider.setValue(value)
        self._connect_preview_slider(slider)
        return slider

    def slider_position_text(self, slider: QSlider) -> str:
        span = max(1, slider.maximum() - slider.minimum())
        value = (slider.value() - slider.minimum()) / span * 10.0
        if slider in (getattr(self, "black_slider", None), getattr(self, "white_slider", None)):
            value = round(value * 20.0) / 20.0
            return f"{value:0.2f}"
        return f"{value:0.1f}"

    def curve_slider_raw_value(self, slider: QSlider) -> int:
        span = max(1, slider.maximum() - slider.minimum())
        display_value = (slider.value() - slider.minimum()) / span * 10.0
        display_value = round(display_value * 20.0) / 20.0
        normalized = np.clip(display_value / 10.0, 0.0, 1.0)
        return int(round(float(normalized) * 65535.0))

    def update_slider_value_label(self, slider: QSlider):
        value_box = getattr(self, "slider_value_labels", {}).get(slider)
        if value_box is not None:
            value_box.blockSignals(True)
            value_box.setValue(float(self.slider_position_text(slider)))
            value_box.blockSignals(False)

    def slider_display_value_to_position(self, slider: QSlider, display_value: float) -> int:
        display_value = float(np.clip(display_value, 0.0, 10.0))
        span = max(1, slider.maximum() - slider.minimum())
        position = slider.minimum() + int(round(display_value / 10.0 * span))
        return max(slider.minimum(), min(slider.maximum(), position))

    def set_slider_from_value_box(self, slider: QSlider, display_value: float):
        slider.setValue(self.slider_display_value_to_position(slider, display_value))

    def slider_with_value(self, slider: QSlider) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        value_box = ArrowDoubleSpinBox()
        value_box.setObjectName("sliderValueBox")
        value_box.setRange(0.0, 10.0)
        value_box.setDecimals(2 if slider in (getattr(self, "black_slider", None), getattr(self, "white_slider", None)) else 1)
        value_box.setSingleStep(0.05 if value_box.decimals() == 2 else 0.1)
        value_box.setKeyboardTracking(False)
        value_box.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        value_box.setFixedWidth(75 if value_box.decimals() == 2 else 68)
        value_box.setStyleSheet("""
            QDoubleSpinBox#sliderValueBox {
                font-family: Menlo, Consolas, monospace;
                font-size: 11px;
                padding: 2px 18px 0px 0px;
            }
            QDoubleSpinBox#sliderValueBox::up-button,
            QDoubleSpinBox#sliderValueBox::down-button {
                width: 18px;
            }
        """)
        try:
            value_box.lineEdit().setTextMargins(0, 0, 0, 0)
            value_box.lineEdit().setStyleSheet("padding: 0px; margin: 0px; background: transparent;")
        except Exception:
            pass
        value_box.setValue(float(self.slider_position_text(slider)))
        self.slider_value_labels[slider] = value_box
        slider.valueChanged.connect(lambda _value, s=slider: self.update_slider_value_label(s))
        value_box.valueChanged.connect(lambda value, s=slider: self.set_slider_from_value_box(s, value))
        layout.addWidget(slider, 1)
        layout.addWidget(value_box)
        return row

    def _undo_array(self, arr: Optional[np.ndarray], copy_pixels: bool = False) -> Optional[np.ndarray]:
        if arr is None:
            return None
        return np.array(arr, copy=True) if copy_pixels else arr

    def _stretch_slider_values(self) -> Dict[str, int]:
        values = {
            "black": self.black_slider.value(),
            "white": self.white_slider.value(),
            "gamma": self.gamma_slider.value(),
            "contrast": self.contrast_slider.value(),
            "saturation": self.saturation_slider.value(),
            "red": self.red_slider.value(),
            "green": self.green_slider.value(),
            "blue": self.blue_slider.value(),
        }
        optional = {
            "stf_strength": "stf_strength_slider",
            "vignette_removal": "vignette_removal_slider",
            "synthetic_flat": "synthetic_flat_slider",
            "color_background": "color_background_slider",
            "denoise": "denoise_slider",
            "star_deconvolution": "star_deconvolution_slider",
            "star_deconvolution_size": "star_deconvolution_size_slider",
            "scnr_green": "scnr_green_slider",
        }
        for key, attr in optional.items():
            slider = getattr(self, attr, None)
            if slider is not None:
                values[key] = slider.value()
        return values

    def _restore_stretch_slider_values(self, values: Dict[str, int]):
        mapping = {
            "black": "black_slider",
            "white": "white_slider",
            "gamma": "gamma_slider",
            "contrast": "contrast_slider",
            "saturation": "saturation_slider",
            "red": "red_slider",
            "green": "green_slider",
            "blue": "blue_slider",
            "stf_strength": "stf_strength_slider",
            "vignette_removal": "vignette_removal_slider",
            "synthetic_flat": "synthetic_flat_slider",
            "color_background": "color_background_slider",
            "denoise": "denoise_slider",
            "star_deconvolution": "star_deconvolution_slider",
            "star_deconvolution_size": "star_deconvolution_size_slider",
            "scnr_green": "scnr_green_slider",
        }
        for key, attr in mapping.items():
            slider = getattr(self, attr, None)
            if slider is None or key not in values:
                continue
            slider.blockSignals(True)
            slider.setValue(int(values[key]))
            slider.blockSignals(False)
            self.update_slider_value_label(slider)

    def push_preview_undo_state(self, copy_pixels: bool = False):
        if getattr(self, "restoring_preview_undo", False):
            return
        source = self.preview_override if self.preview_override is not None else self.linear_result
        state = {
            "linear_result": self._undo_array(self.linear_result, copy_pixels),
            "preview_override": self._undo_array(self.preview_override, copy_pixels),
            "preview_override_path": getattr(self, "preview_override_path", None),
            "preview_source_shape": self.preview_source_shape,
            "neutralized_preview_layer": self._undo_array(getattr(self, "neutralized_preview_layer", None), copy_pixels),
            "gradient_preview_layer": self._undo_array(getattr(self, "gradient_preview_layer", None), copy_pixels),
            "neutralized_matches_source": (
                getattr(self, "neutralized_preview_layer", None) is not None
                and source is not None
                and getattr(self, "neutralized_preview_base_source_id", None) == id(source)
            ),
            "gradient_matches_source": (
                getattr(self, "gradient_preview_layer", None) is not None
                and source is not None
                and getattr(self, "gradient_preview_base_source_id", None) == id(source)
            ),
            "flip_horizontal": bool(getattr(self, "flip_horizontal", False)),
            "flip_vertical": bool(getattr(self, "flip_vertical", False)),
            "preview_rotation_degrees": int(getattr(self, "preview_rotation_degrees", 0)),
            "preview_display_limits": self.preview_display_limits,
            "preview_auto_display_stretch": bool(self.preview_auto_display_stretch),
            "stretch": self._stretch_slider_values(),
        }
        stack = getattr(self, "preview_undo_stack", [])
        stack.append(state)
        limit = max(1, int(getattr(self, "preview_undo_limit", 2)))
        if len(stack) > limit:
            del stack[:-limit]
        self.preview_undo_stack = stack
        self.update_undo_button_state()

    def update_undo_button_state(self):
        button = getattr(self, "undo_btn", None)
        if button is not None:
            button.setEnabled(bool(getattr(self, "preview_undo_stack", [])))

    def undo_preview_adjustment(self):
        stack = getattr(self, "preview_undo_stack", [])
        if not stack:
            self.status_label.setText(self.tr_ui("undo_empty"))
            return
        state = stack.pop()
        self.preview_undo_stack = stack
        self.restoring_preview_undo = True
        try:
            self.linear_result = state.get("linear_result")
            self.preview_override = state.get("preview_override")
            self.preview_override_path = state.get("preview_override_path")
            self.preview_source_shape = state.get("preview_source_shape")
            self.neutralized_preview_layer = state.get("neutralized_preview_layer")
            self.gradient_preview_layer = state.get("gradient_preview_layer")
            source = self.preview_override if self.preview_override is not None else self.linear_result
            self.neutralized_preview_base_source_id = (
                id(source) if self.neutralized_preview_layer is not None and state.get("neutralized_matches_source") and source is not None else None
            )
            self.gradient_preview_base_source_id = (
                id(source) if self.gradient_preview_layer is not None and state.get("gradient_matches_source") and source is not None else None
            )
            self.flip_horizontal = bool(state.get("flip_horizontal", False))
            self.flip_vertical = bool(state.get("flip_vertical", False))
            self.preview_rotation_degrees = int(state.get("preview_rotation_degrees", 0))
            self.preview_display_limits = state.get("preview_display_limits")
            self.preview_auto_display_stretch = bool(state.get("preview_auto_display_stretch", True))
            self._restore_stretch_slider_values(state.get("stretch", {}))
        finally:
            self.restoring_preview_undo = False
        self.invalidate_preview_cache()
        self.update_undo_button_state()
        path_text = getattr(self, "preview_override_path", None)
        if path_text:
            try:
                self.update_metadata_panel(Path(path_text), self.preview_override)
            except Exception:
                pass
        self.status_label.setText(self.tr_ui("undo_done"))
        self.update_preview()

    def _connect_preview_slider(self, slider: QSlider):
        slider.sliderPressed.connect(lambda s=slider: self.push_preview_undo_state(copy_pixels=False))
        slider.sliderPressed.connect(self._preview_slider_pressed)
        slider.sliderReleased.connect(self._preview_slider_released)
        slider.valueChanged.connect(self.schedule_preview_update)

    def recent_image_paths(self) -> List[Path]:
        value = self.settings_store.value("recent_images", [])
        if isinstance(value, str):
            raw_paths = [value]
        elif isinstance(value, (list, tuple)):
            raw_paths = [str(item) for item in value if item]
        else:
            raw_paths = []
        paths: List[Path] = []
        seen: set[str] = set()
        for raw in raw_paths:
            try:
                path = Path(raw).expanduser()
                key = str(path)
            except Exception:
                continue
            if key not in seen:
                paths.append(path)
                seen.add(key)
        return paths[:10]

    def save_recent_image_paths(self, paths: List[Path]):
        self.settings_store.setValue("recent_images", [str(path) for path in paths[:10]])
        self.update_recent_images_menu()

    def add_recent_image_path(self, path: Path):
        path = Path(path).expanduser()
        paths = [p for p in self.recent_image_paths() if str(p) != str(path)]
        paths.insert(0, path)
        self.save_recent_image_paths(paths)

    def clear_recent_images(self):
        self.save_recent_image_paths([])

    def update_recent_images_menu(self):
        if not hasattr(self, "recent_images_menu"):
            return
        self.recent_images_menu.clear()
        self.recent_image_actions = []
        paths = self.recent_image_paths()
        if paths:
            for index, path in enumerate(paths, start=1):
                text = f"{index}. {path.name}"
                action = QAction(text, self)
                action.setToolTip(str(path))
                action.triggered.connect(lambda _checked=False, p=path: self.open_recent_image(p))
                self.recent_images_menu.addAction(action)
                self.recent_image_actions.append(action)
            self.recent_images_menu.addSeparator()
            self.clear_recent_images_action.setEnabled(True)
        else:
            empty_action = QAction(
                "Žádná historie" if self.language == "cz" else "No recent images",
                self,
            )
            empty_action.setEnabled(False)
            self.recent_images_menu.addAction(empty_action)
            self.clear_recent_images_action.setEnabled(False)
        self.recent_images_menu.addAction(self.clear_recent_images_action)

    def open_recent_image(self, path: Path):
        path = Path(path).expanduser()
        if not path.exists():
            QMessageBox.warning(
                self,
                self.tr_ui("missing_image_title"),
                (
                    f"Soubor už neexistuje:\n{path}"
                    if self.language == "cz"
                    else f"The file no longer exists:\n{path}"
                ),
            )
            self.save_recent_image_paths([p for p in self.recent_image_paths() if str(p) != str(path)])
            return
        self.open_image_path(path, add_to_recent=True)

    def open_image_path(self, path: Path, add_to_recent: bool = False):
        path = Path(path).expanduser()
        sibling_paths = sorted(
            [
                p
                for p in path.parent.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            ],
            key=lambda p: p.name.lower(),
        )
        self.set_preview_sequence(sibling_paths, path)
        if self.load_preview_image(path) and add_to_recent:
            self.add_recent_image_path(path)

    def _is_ai_preview_slider(self, slider) -> bool:
        return slider in (
            getattr(self, "denoise_slider", None),
            getattr(self, "star_deconvolution_slider", None),
            getattr(self, "star_deconvolution_size_slider", None),
        )

    def choose_drunet_model(self) -> bool:
        current_path = find_drunet_model(getattr(self, "ai_denoise_model_path", None))
        start_dir = str(current_path.parent if current_path is not None else Path.home())
        filename, _ = QFileDialog.getOpenFileName(
            self,
            self.tr_ui("select_drunet_title"),
            start_dir,
            "ONNX model (*.onnx)",
        )
        if not filename:
            return False
        candidate = Path(filename).expanduser()
        saved_preview_state = self.capture_preview_zoom_state()
        DRUNET_SESSION_CACHE.clear()
        if ort is not None:
            try:
                session = get_drunet_session(candidate)
                input_meta = session.get_inputs()[0]
                input_shape = list(input_meta.shape)
                input_channels = (
                    input_shape[1]
                    if len(input_shape) == 4 and isinstance(input_shape[1], int)
                    else 4
                )
                if input_channels not in (1, 2, 3, 4):
                    raise ValueError(
                        f"Unsupported DRUNet input channel count: {input_channels}"
                    )
            except Exception as exc:
                QMessageBox.critical(
                    self,
                    self.tr_ui("drunet_model_missing_title"),
                    str(exc),
                )
                return False
        self.preview_slider_zoom_state = saved_preview_state
        self.preview_render_pending_final = True
        self.ai_denoise_model_path = str(candidate)
        self.preview_ai_denoise_cache = None
        self.preview_ai_denoise_cache_key = None
        self.ai_denoise_layer_cache.clear()
        self.status_label.setText(
            self.tr_ui("drunet_model_selected").format(name=Path(filename).name)
        )
        self.schedule_preview_update()
        return True

    def choose_selected_denoise_model(self) -> bool:
        return self.choose_drunet_model()

    def choose_stellar_deconvolution_model(self) -> bool:
        current_path = find_stellar_deconvolution_model(
            getattr(self, "star_deconvolution_model_path", None)
        )
        start_dir = str(
            current_path.parent if current_path is not None else Path.home()
        )
        filename, _ = QFileDialog.getOpenFileName(
            self,
            self.tr_ui("select_stellar_title"),
            start_dir,
            "ONNX model (*.onnx)",
        )
        if not filename:
            return False
        candidate = Path(filename).expanduser()
        saved_preview_state = self.capture_preview_zoom_state()
        STELLAR_SESSION_CACHE.clear()
        if ort is not None:
            try:
                session = get_stellar_deconvolution_session(candidate)
                input_shape = list(session.get_inputs()[0].shape)
                channels = (
                    input_shape[1]
                    if len(input_shape) == 4 and isinstance(input_shape[1], int)
                    else 3
                )
                if channels != 3:
                    raise ValueError(
                        f"Stellar model must have three input channels, got {channels}."
                    )
            except Exception as exc:
                QMessageBox.critical(
                    self,
                    self.tr_ui("stellar_model_missing_title"),
                    str(exc),
                )
                return False
        self.preview_slider_zoom_state = saved_preview_state
        self.preview_render_pending_final = True
        self.star_deconvolution_model_path = str(candidate)
        self.star_deconvolution_layer_cache.clear()
        self.status_label.setText(
            self.tr_ui("stellar_model_selected").format(name=candidate.name)
        )
        self.schedule_preview_update()
        return True

    def on_denoise_mode_changed(self, *_args):
        mode = (
            self.denoise_mode_combo.currentData()
            if hasattr(self, "denoise_mode_combo")
            else "classic"
        )
        if mode == "drunet":
            if ort is None:
                QMessageBox.warning(
                    self,
                    self.tr_ui("onnxruntime_missing_title"),
                    self.tr_ui("onnxruntime_missing_message"),
                )
                self.denoise_mode_combo.blockSignals(True)
                self.denoise_mode_combo.setCurrentIndex(
                    max(0, self.denoise_mode_combo.findData("classic"))
                )
                self.denoise_mode_combo.blockSignals(False)
                return
            model_path = find_drunet_model(
                getattr(self, "ai_denoise_model_path", None)
            )
            if model_path is None:
                QMessageBox.information(
                    self,
                    self.tr_ui("drunet_model_missing_title"),
                    self.tr_ui("drunet_model_missing_message"),
                )
                selected = self.choose_drunet_model()
                if not selected:
                    self.denoise_mode_combo.blockSignals(True)
                    self.denoise_mode_combo.setCurrentIndex(
                        max(0, self.denoise_mode_combo.findData("classic"))
                    )
                    self.denoise_mode_combo.blockSignals(False)
                    return
            else:
                self.ai_denoise_model_path = str(model_path)
        if hasattr(self, "denoise_mode_combo"):
            self.denoise_mode_combo.setToolTip(self.tr_ui("drunet_tooltip"))
        self.preview_ai_denoise_cache = None
        self.preview_ai_denoise_cache_key = None
        self.ai_denoise_layer_cache.clear()
        self.schedule_preview_update()

    def change_ui_mode(self, *_args):
        self.apply_ui_mode()

    def apply_ui_mode(self):
        mode = self.ui_mode_combo.currentData() if hasattr(self, "ui_mode_combo") else "advanced"
        show_advanced = mode != "simple"
        for widget in getattr(self, "simple_mode_hidden_widgets", []):
            if widget is not None:
                widget.setVisible(show_advanced)

    def update_process_mode(self, *_args):
        mode = self.process_mode_combo.currentData() if hasattr(self, "process_mode_combo") else "auto"
        auto_mode = mode != "manual"
        if hasattr(self, "processes_spin"):
            self.processes_spin.setEnabled(not auto_mode)
            if auto_mode:
                self.processes_spin.setValue(self.optimal_cpu_processes())

    def selected_cpu_processes(self) -> int:
        if hasattr(self, "process_mode_combo") and self.process_mode_combo.currentData() == "auto":
            return self.optimal_cpu_processes()
        return self.processes_spin.value()

    def classify_frame_path(self, path: Path) -> str:
        text = " ".join(part.lower() for part in path.parts)
        name = path.stem.lower()
        if any(token in text or token in name for token in ("bias", "biases", "offset", "offsets")):
            return "Bias"
        if any(token in text or token in name for token in ("flat", "flats")):
            return "Flat"
        if any(token in text or token in name for token in ("dark", "darks")):
            return "Dark"
        return "Light"

    def collect_preview_paths_for_folder(self, folder: Path) -> List[Path]:
        if not folder:
            return []
        paths = [p for p in Path(folder).rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        order = {"Light": 0, "Flat": 1, "Bias": 2, "Dark": 3}
        return sorted(paths, key=lambda p: (order.get(self.classify_frame_path(p), 9), str(p).lower()))

    def frame_type_counts_for_folder(self) -> Dict[str, int]:
        counts = {"Light": 0, "Dark": 0, "Flat": 0, "Bias": 0}
        extensions = (
            RAW_STACK_EXTENSIONS
            if hasattr(self, "fit_only_check") and self.fit_only_check.isChecked()
            else IMAGE_EXTENSIONS
        )
        ignored_dirs = {"astro_stacker_cache", "astro_stacker_output", "__pycache__"}
        counted_paths: set[str] = set()

        def add_path(path: Path, kind: Optional[str] = None, allow_master: bool = False):
            try:
                path = Path(path)
                if not path.is_file() or path.suffix.lower() not in extensions:
                    return
                if not allow_master and path.name.lower().startswith("master"):
                    return
                if any(part.lower() in ignored_dirs for part in path.parts):
                    return
                key = str(path.resolve())
                if key in counted_paths:
                    return
                counted_paths.add(key)
                counts[kind or self.classify_frame_path(path)] += 1
            except Exception:
                pass

        if self.folder:
            try:
                for path in Path(self.folder).rglob("*"):
                    add_path(path)
            except Exception:
                pass

        # A manually selected calibration source may live outside the Light
        # folder and does not need a conventional Flat/Bias/Dark directory name.
        for kind, attr in (
            ("Flat", "flat_frame_path"),
            ("Bias", "bias_frame_path"),
            ("Dark", "dark_frame_path"),
        ):
            source = getattr(self, attr, None)
            if not source:
                continue
            source = Path(source)
            if source.is_dir():
                try:
                    for path in source.iterdir():
                        add_path(path, kind=kind)
                except Exception:
                    pass
            else:
                # A manually selected finished Master represents one active
                # calibration input even though automatic scans skip masters.
                add_path(source, kind=kind, allow_master=True)
        return counts

    def update_frame_quality_title(self):
        if not hasattr(self, "frame_quality_title"):
            return
        counts = self.frame_type_counts_for_folder()
        suffix = self.tr_ui("frame_quality_counts").format(
            lights=counts["Light"],
            darks=counts["Dark"],
            flats=counts["Flat"],
            biases=counts["Bias"],
        )
        self.frame_quality_title.setText(f"{self.tr_ui('frame_quality_title')}  |  {suffix}")

    def set_preview_sequence(self, paths: List[Path], current: Optional[Path] = None):
        self.preview_sequence_paths = list(paths)
        self.preview_sequence_loading = True
        self.preview_file_combo.clear()
        for idx, path in enumerate(self.preview_sequence_paths):
            kind = self.classify_frame_path(path)
            path_key = self.preview_path_key(path)
            prefix = "  "
            manual_reference_key = self.preview_path_key(Path(self.manual_reference_path)) if self.manual_reference_path else None
            if manual_reference_key is not None and path_key == manual_reference_key and not self.auto_reference_check.isChecked():
                prefix = "R "
            elif path_key == self.stack_reference_path:
                prefix = "* "
            elif path_key in self.stack_excluded_paths:
                prefix = "x "
            try:
                label = f"{prefix}{idx + 1:04d} [{kind}] {path.relative_to(self.folder) if self.folder else path.name}"
            except Exception:
                label = f"{prefix}{idx + 1:04d} [{kind}] {path.name}"
            self.preview_file_combo.addItem(label, str(path))
            tooltip = self.preview_stack_tooltip(path)
            self.preview_file_combo.setItemData(idx, tooltip, Qt.ToolTipRole)
            if path_key in self.stack_excluded_paths:
                self.preview_file_combo.setItemData(idx, QBrush(QColor("#7d8591")), Qt.ForegroundRole)
            elif manual_reference_key is not None and path_key == manual_reference_key and not self.auto_reference_check.isChecked():
                self.preview_file_combo.setItemData(idx, QBrush(QColor("#7bd88f")), Qt.ForegroundRole)
            elif path_key == self.stack_reference_path:
                self.preview_file_combo.setItemData(idx, QBrush(QColor("#f4d35e")), Qt.ForegroundRole)

        current_index = 0
        if current is not None:
            try:
                current_resolved = Path(current).resolve()
                for idx, path in enumerate(self.preview_sequence_paths):
                    if path.resolve() == current_resolved:
                        current_index = idx
                        break
            except Exception:
                pass
        if self.preview_sequence_paths:
            self.preview_file_combo.setCurrentIndex(current_index)
        self.preview_sequence_loading = False
        self.update_preview_nav_state()
        self.update_frame_quality_title()

    def preview_path_key(self, path: Path) -> str:
        try:
            return str(Path(path).resolve())
        except Exception:
            return str(path)

    def preview_stack_tooltip(self, path: Path) -> str:
        path_key = self.preview_path_key(path)
        score = self.stack_quality_scores.get(path_key)
        score_text = "\n" + self.tr_ui("quality_score").format(score=score) if score is not None else ""
        if path_key == self.stack_reference_path:
            return f"{self.tr_ui('stack_reference_tooltip')}{score_text}"
        manual_reference_key = self.preview_path_key(Path(self.manual_reference_path)) if self.manual_reference_path else None
        if manual_reference_key is not None and path_key == manual_reference_key and not self.auto_reference_check.isChecked():
            return self.tr_ui("manual_reference_tooltip")
        if path_key in self.stack_excluded_paths:
            keep = self.stack_selection_info.get("keep_percent")
            reason = (
                self.tr_ui("stack_excluded_keep_tooltip").format(keep=keep)
                if keep
                else self.tr_ui("stack_excluded_tooltip")
            )
            return f"{reason}{score_text}"
        if path_key in self.stack_used_paths:
            return f"{self.tr_ui('stack_used_tooltip')}{score_text}"
        return str(path)

    def update_frame_quality_table(self):
        table = getattr(self, "frame_quality_table", None)
        if table is None:
            return
        table.blockSignals(True)
        table.setSortingEnabled(False)
        table.setRowCount(0)

        all_paths = list(self.stack_selection_info.get("all_paths", []))
        if not all_paths:
            table.setSortingEnabled(True)
            table.blockSignals(False)
            return

        metrics = dict(self.stack_selection_info.get("quality_metrics", {}))
        used = set(self.stack_selection_info.get("used_paths", []))
        selected = set(self.stack_selection_info.get("selected_paths", []))
        light = set(self.stack_selection_info.get("light_paths", all_paths))
        reference = self.stack_selection_info.get("reference_path")
        manual_excluded = set(self.manual_excluded_paths)

        for idx, path_str in enumerate(all_paths):
            path = Path(path_str)
            row = table.rowCount()
            table.insertRow(row)
            frame_metrics = metrics.get(path_str, {})
            score = float(frame_metrics.get("score", self.stack_quality_scores.get(path_str, 0.0)))
            sharpness = float(frame_metrics.get("sharpness", 0.0))
            star_count = int(frame_metrics.get("star_count", 0))
            roundness = float(frame_metrics.get("roundness", 0.0))
            fwhm = float(frame_metrics.get("fwhm", 0.0))
            trail_score = float(frame_metrics.get("trail_score", 0.0))
            satellite_trail = float(frame_metrics.get("satellite_trail", 0.0)) >= 0.5

            if path_str in manual_excluded:
                status_key = "frame_status_manual_excluded"
            elif path_str == reference:
                status_key = "frame_status_reference"
            elif path_str in used:
                status_key = "frame_status_used"
            elif path_str in selected:
                status_key = "frame_status_alignment"
            elif path_str in light:
                status_key = "frame_status_quality"
            else:
                status_key = "frame_status_skipped"

            try:
                file_label = str(path.relative_to(self.folder)) if self.folder else path.name
            except Exception:
                file_label = path.name

            items = [
                NumericTableWidgetItem(str(idx + 1)),
                QTableWidgetItem(file_label),
                NumericTableWidgetItem(f"{score:.3g}"),
                NumericTableWidgetItem(str(star_count)),
                NumericTableWidgetItem(f"{sharpness:.3g}"),
                NumericTableWidgetItem(f"{roundness:.3f}"),
                NumericTableWidgetItem(f"{fwhm:.2f}"),
                NumericTableWidgetItem(self.tr_ui("trail_suspect") if satellite_trail else self.tr_ui("trail_ok")),
                QTableWidgetItem(self.tr_ui(status_key)),
                QTableWidgetItem("*" if path_str == reference else ""),
            ]
            numeric_values = [idx + 1, None, score, star_count, sharpness, roundness, fwhm, trail_score, None, 1 if path_str == reference else 0]
            for col, item in enumerate(items):
                if numeric_values[col] is not None:
                    item.setData(Qt.UserRole, numeric_values[col])
                item.setToolTip(path_str)
                table.setItem(row, col, item)
            table.item(row, 1).setData(Qt.UserRole, path_str)

            if path_str in manual_excluded:
                color = QColor("#ff5a5f")
            elif path_str == reference:
                color = QColor("#f4d35e")
            elif satellite_trail:
                color = QColor("#ff8a4c")
            elif path_str in used:
                color = QColor("#7bd88f")
            elif path_str in selected:
                color = QColor("#f59e0b")
            else:
                color = QColor("#7d8591")
            for col in range(table.columnCount()):
                table.item(row, col).setForeground(QBrush(color))

        table.setHorizontalHeaderLabels(self.tr_ui("frame_quality_headers"))
        table.setSortingEnabled(True)
        table.sortItems(2, Qt.AscendingOrder)
        table.blockSignals(False)

    def toggle_selected_frame_excluded(self):
        table = getattr(self, "frame_quality_table", None)
        if table is None:
            return
        row = table.currentRow()
        if row < 0:
            return
        item = table.item(row, 1)
        path_str = item.data(Qt.UserRole) if item is not None else None
        if not path_str:
            return
        if path_str == self.stack_reference_path:
            self.status_label.setText(self.tr_ui("reference_cannot_exclude"))
            return
        if path_str in self.manual_excluded_paths:
            self.manual_excluded_paths.remove(path_str)
        else:
            self.manual_excluded_paths.add(path_str)
        self.update_stack_selection_summary()
        self.update_frame_quality_table()

    def review_selected_paths_for_stack(self) -> List[str]:
        if not self.stack_selection_info:
            return []
        selected = list(self.stack_selection_info.get("selected_paths", []))
        reference = self.stack_selection_info.get("reference_path")
        if not self.auto_reference_check.isChecked() and self.manual_reference_path:
            try:
                reference = str(Path(self.manual_reference_path).resolve())
            except Exception:
                reference = str(self.manual_reference_path)
        selected = [p for p in selected if p not in self.manual_excluded_paths or p == reference]
        if reference and reference not in selected:
            selected.insert(0, reference)
        return selected

    def on_frame_quality_selection_changed(self):
        table = getattr(self, "frame_quality_table", None)
        if table is None:
            return
        row = table.currentRow()
        if row < 0:
            return
        item = table.item(row, 1)
        path_str = item.data(Qt.UserRole) if item is not None else None
        if not path_str:
            return
        for idx, path in enumerate(self.preview_sequence_paths):
            if self.preview_path_key(path) == path_str:
                self.preview_file_combo.setCurrentIndex(idx)
                break

    def apply_stack_selection_info(self, info: Dict[str, Any]):
        self.stack_selection_info = dict(info or {})
        self.stack_used_paths = set(self.stack_selection_info.get("used_paths", []))
        self.stack_excluded_paths = set(self.stack_selection_info.get("excluded_paths", []))
        self.stack_reference_path = self.stack_selection_info.get("reference_path")
        self.stack_quality_scores = dict(self.stack_selection_info.get("scores", {}))
        if "manual_excluded_paths" in self.stack_selection_info:
            self.manual_excluded_paths = set(self.stack_selection_info.get("manual_excluded_paths", []))
        total_time = self.stack_selection_info.get("total_processing_time")
        self.last_total_processing_time = float(total_time) if total_time is not None else None
        ram_bytes = self.stack_selection_info.get("estimated_used_ram_bytes")
        self.last_estimated_used_ram_bytes = int(ram_bytes) if ram_bytes is not None else None
        self.update_stack_selection_summary()
        self.update_frame_quality_table()
        if self.preview_sequence_paths:
            current = None
            idx = self.preview_file_combo.currentIndex()
            if 0 <= idx < len(self.preview_sequence_paths):
                current = self.preview_sequence_paths[idx]
            self.set_preview_sequence(self.preview_sequence_paths, current)

    def clear_stack_selection_info(self):
        self.stack_selection_info = {}
        self.stack_used_paths = set()
        self.stack_excluded_paths = set()
        self.stack_reference_path = None
        self.stack_quality_scores = {}
        self.last_total_processing_time = None
        self.last_estimated_used_ram_bytes = None
        self.review_ready = False
        self.update_stack_selection_summary()
        self.update_frame_quality_table()

    def update_stack_selection_summary(self):
        if not hasattr(self, "stack_selection_label"):
            return
        total = len(self.stack_selection_info.get("all_paths", []))
        used_paths = [p for p in self.stack_selection_info.get("used_paths", []) if p not in self.manual_excluded_paths]
        used = len(used_paths)
        excluded = max(0, total - used)
        if total:
            summary = self.tr_ui("stack_selection_summary").format(used=used, total=total, excluded=excluded)
            manual_count = len(self.manual_excluded_paths)
            if manual_count:
                summary += f" {self.tr_ui('frame_status_manual_excluded')}: {manual_count}."
            if self.last_total_processing_time is not None:
                summary += f"\nTotal time: {format_elapsed(self.last_total_processing_time)}"
                if self.last_estimated_used_ram_bytes is not None:
                    summary += f"; {self.tr_ui('estimated_full_stack_ram')}: ~{format_bytes_short(self.last_estimated_used_ram_bytes)}"
            self.stack_selection_label.setText(summary)
        else:
            self.stack_selection_label.setText("")

    def update_preview_nav_state(self):
        has_items = bool(self.preview_sequence_paths)
        idx = self.preview_file_combo.currentIndex() if hasattr(self, "preview_file_combo") else -1
        self.preview_prev_btn.setEnabled(has_items and idx > 0)
        self.preview_next_btn.setEnabled(has_items and idx >= 0 and idx < len(self.preview_sequence_paths) - 1)

    def on_preview_file_selected(self, index: int):
        if self.preview_sequence_loading or index < 0 or index >= len(self.preview_sequence_paths):
            self.update_preview_nav_state()
            return
        self.load_preview_image(self.preview_sequence_paths[index])

    def preview_previous_image(self):
        idx = self.preview_file_combo.currentIndex()
        if idx > 0:
            self.preview_file_combo.setCurrentIndex(idx - 1)

    def preview_next_image(self):
        idx = self.preview_file_combo.currentIndex()
        if 0 <= idx < len(self.preview_sequence_paths) - 1:
            self.preview_file_combo.setCurrentIndex(idx + 1)

    def current_preview_path(self) -> Optional[Path]:
        idx = self.preview_file_combo.currentIndex() if hasattr(self, "preview_file_combo") else -1
        if 0 <= idx < len(self.preview_sequence_paths):
            return self.preview_sequence_paths[idx]
        return None

    def set_current_preview_as_reference(self):
        path = self.current_preview_path()
        if path is None:
            QMessageBox.warning(
                self,
                "Chybí snímek" if self.language == "cz" else "No frame selected",
                "Nejdřív vyber snímek v seznamu náhledu." if self.language == "cz" else "First select a frame in the preview list.",
            )
            return
        if self.classify_frame_path(path) != "Light":
            QMessageBox.warning(
                self,
                "Nevhodná reference" if self.language == "cz" else "Invalid reference",
                "Jako referenci vyber Light snímek, ne Flat/Bias/Dark." if self.language == "cz" else "Choose a Light frame as the reference, not Flat/Bias/Dark.",
            )
            return
        try:
            path_str = str(Path(path).resolve())
        except Exception:
            path_str = str(path)
        self.manual_reference_path = str(path)
        self.auto_reference_check.setChecked(False)
        if self.stack_selection_info:
            self.stack_reference_path = path_str
            self.stack_selection_info["reference_path"] = path_str
            selected = list(self.stack_selection_info.get("selected_paths", []))
            if path_str not in selected:
                selected.insert(0, path_str)
            else:
                selected = [path_str] + [p for p in selected if p != path_str]
            self.stack_selection_info["selected_paths"] = selected
            used = list(self.stack_selection_info.get("used_paths", []))
            if path_str not in used:
                used.insert(0, path_str)
            else:
                used = [path_str] + [p for p in used if p != path_str]
            self.stack_selection_info["used_paths"] = used
            excluded = [p for p in self.stack_selection_info.get("excluded_paths", []) if p != path_str]
            quality_excluded = [p for p in self.stack_selection_info.get("quality_excluded_paths", []) if p != path_str]
            self.stack_selection_info["excluded_paths"] = excluded
            self.stack_selection_info["quality_excluded_paths"] = quality_excluded
            self.stack_excluded_paths.discard(path_str)
            self.manual_excluded_paths.discard(path_str)
            self.update_stack_selection_summary()
            self.update_frame_quality_table()
        self.set_preview_sequence(self.preview_sequence_paths, path)
        self.status_label.setText(
            f"Ruční reference nastavena: {path.name}"
            if self.language == "cz"
            else f"Manual reference set: {path.name}"
        )

    def on_auto_reference_changed(self, *_args):
        if self.preview_sequence_paths:
            current = self.current_preview_path()
            self.set_preview_sequence(self.preview_sequence_paths, current)

    def reset_denoise_for_new_preview_source(self):
        """Do not automatically run old AI processing settings on a new file."""
        if hasattr(self, "denoise_slider"):
            self.denoise_slider.blockSignals(True)
            self.denoise_slider.setValue(0)
            self.denoise_slider.blockSignals(False)
        self.preview_ai_denoise_cache = None
        self.preview_ai_denoise_cache_key = None
        self.ai_denoise_layer_cache.clear()
        if hasattr(self, "star_deconvolution_slider"):
            self.star_deconvolution_slider.blockSignals(True)
            self.star_deconvolution_slider.setValue(0)
            self.star_deconvolution_slider.blockSignals(False)
        self.star_deconvolution_layer_cache.clear()

    def reset_preview_render_state_for_new_source(self):
        """Discard delayed renders and positions that belong to another image."""
        if hasattr(self, "preview_render_timer"):
            self.preview_render_timer.stop()
        self.preview_interactive = False
        self.preview_render_pending_final = False
        self.preview_slider_zoom_state = None
        self.preview_fast_zoom_scale = 1.0
        self.preview_position_generation += 1
        self.stop_preview_pan_inertia()
        self.preview_pan_scroll = (0.0, 0.0)

    def load_preview_image(self, path: Path) -> bool:
        try:
            set_bayer_pattern_override(self.bayer_combo.currentData() if hasattr(self, "bayer_combo") else "auto")
            img = load_image_as_float(path)
        except Exception as exc:
            QMessageBox.critical(self, "Chyba" if self.language == "cz" else "Error", f"Snímek se nepodařilo načíst:\n{exc}" if self.language == "cz" else f"Could not load image:\n{exc}")
            return False

        self.reset_preview_render_state_for_new_source()
        self.preview_override = img
        self.preview_override_path = str(path)
        self.preview_source_shape = img.shape[:2]
        self.reset_denoise_for_new_preview_source()
        self.invalidate_preview_cache()
        self.preview_auto_display_stretch = True
        self.awaiting_comet_click = False
        self.comet_click_mode = None
        if hasattr(self, "image_label"):
            self.image_label.set_marking_mode(False)
        self.zoom_mode = "fit"
        self.zoom_factor = 1.0
        self.update_metadata_panel(path, img)
        self.apply_balance_to_current_preview(
            record_undo=False,
            refresh_preview=False,
            set_status=False,
        )
        self.status_label.setText(self.tr_ui("open_preview_status").format(name=path.name))
        self.update_preview()
        self.update_preview_nav_state()
        return True

    def format_metadata_for_path(self, path: Path, img: Optional[np.ndarray] = None) -> str:
        lines = []
        kind = self.classify_frame_path(path)
        stat = None
        try:
            stat = path.stat()
        except Exception:
            pass
        lines.append(f"Type: {kind}")
        lines.append(f"File: {path.name}")
        lines.append(f"Ext: {path.suffix.lower()}")
        if stat:
            lines.append(f"Size: {stat.st_size / (1024 * 1024):.1f} MB")
        if img is not None:
            lines.append(f"Loaded: {img.shape}, {img.dtype}")

        suffix = path.suffix.lower()
        if suffix in {".fit", ".fits"}:
            header = get_first_fits_header_safely(path)
            if header is not None:
                def hval(*keys):
                    for key in keys:
                        if key in header:
                            return header.get(key)
                    upper_map = {str(k).upper(): k for k in header.keys()}
                    for key in keys:
                        real_key = upper_map.get(str(key).upper())
                        if real_key is not None:
                            return header.get(real_key)
                    return None

                def add(label, *keys, unit: str = ""):
                    value = hval(*keys)
                    if value is not None and str(value).strip() != "":
                        lines.append(f"{label}: {value}{unit}")

                add("Frame", "IMAGETYP", "FRAME", "FRAMETYP", "PICTTYPE")
                add("Exposure", "EXPTIME", "EXPOSURE", "EXP_TIME", "EXPOSURE_TIME", "XPOSURE", "ONTIME", unit=" s")
                add("Gain", "GAIN", "EGAIN", "CCDGAIN", "CAMGAIN", "ZWO_GAIN")
                add("Offset", "OFFSET", "BLKLEVEL", "BLACKLEV", "PEDESTAL", "ZWO_OFFSET")
                add("Sensor temp", "CCD-TEMP", "CCD_TEMP", "SENSOR_TEMP", "CMOS-TEMP", "SET-TEMP", "TEMPERAT", unit=" C")
                add("Set temp", "SET-TEMP", "SET_TEMP", "CCD-TEMP", unit=" C")
                add("Binning", "XBINNING", "BINX", "XBIN")
                add("Y binning", "YBINNING", "BINY", "YBIN")
                add("Camera", "INSTRUME", "CAMERA", "CCDNAME", "DETECTOR", "SENSOR")
                add("Telescope", "TELESCOP", "SCOPE")
                add("Object", "OBJECT", "TARGET")
                add("Filter", "FILTER", "FILTNAME")
                add("Date obs", "DATE-OBS", "DATEOBS", "DATE")
                add("Readout", "READOUTM", "READOUT", "READMODE")
                add("Bayer", "BAYERPAT", "BAYER", "COLORTYP", "CFA")
                add("Pixel size X", "XPIXSZ", "PIXSIZE1", "XPIXSIZE", unit=" um")
                add("Pixel size Y", "YPIXSZ", "PIXSIZE2", "YPIXSIZE", unit=" um")
                add("Focal length", "FOCALLEN", "FOCLEN", "FOCUSPOS", unit=" mm")
                add("RA", "OBJCTRA", "RA", "TELRA")
                add("DEC", "OBJCTDEC", "DEC", "TELDEC")
                add("Software", "SWCREATE", "CREATOR", "PROGRAM")

                shown = {
                    "IMAGETYP", "FRAME", "FRAMETYP", "PICTTYPE", "EXPTIME", "EXPOSURE", "EXP_TIME",
                    "EXPOSURE_TIME", "XPOSURE", "ONTIME", "GAIN", "EGAIN", "CCDGAIN", "CAMGAIN",
                    "ZWO_GAIN", "OFFSET", "BLKLEVEL", "BLACKLEV", "PEDESTAL", "ZWO_OFFSET",
                    "CCD-TEMP", "CCD_TEMP", "SENSOR_TEMP", "CMOS-TEMP", "SET-TEMP", "SET_TEMP",
                    "TEMPERAT", "XBINNING", "BINX", "XBIN", "YBINNING", "BINY", "YBIN",
                    "INSTRUME", "CAMERA", "CCDNAME", "DETECTOR", "SENSOR", "TELESCOP", "SCOPE",
                    "OBJECT", "TARGET", "FILTER", "FILTNAME", "DATE-OBS", "DATEOBS", "DATE",
                    "READOUTM", "READOUT", "READMODE", "BAYERPAT", "BAYER", "COLORTYP", "CFA",
                    "XPIXSZ", "PIXSIZE1", "XPIXSIZE", "YPIXSZ", "PIXSIZE2", "YPIXSIZE",
                    "FOCALLEN", "FOCLEN", "FOCUSPOS", "OBJCTRA", "RA", "TELRA", "OBJCTDEC",
                    "DEC", "TELDEC", "SWCREATE", "CREATOR", "PROGRAM",
                }
                extra = []
                for key in header.keys():
                    key_s = str(key)
                    if key_s.upper() in shown or key_s.upper() in {"COMMENT", "HISTORY", ""}:
                        continue
                    value = header.get(key)
                    if value is not None and str(value).strip() != "":
                        extra.append(f"{key_s}: {value}")
                    if len(extra) >= 12:
                        break
                if extra:
                    lines.append("--- Other FITS ---")
                    lines.extend(extra)
        elif suffix in XISF_EXTENSIONS:
            try:
                metadata = xisf_image_metadata(path)
                geometry = metadata.get("geometry")
                if geometry:
                    lines.append(f"XISF geometry: {geometry}")
                lines.append(f"XISF sample: {metadata.get('sampleFormat', metadata.get('dtype', 'unknown'))}")
                values = xisf_metadata_values(metadata)
                preferred = (
                    "IMAGETYP", "EXPTIME", "EXPOSURE", "GAIN", "OFFSET",
                    "CCD-TEMP", "INSTRUME", "TELESCOP", "OBJECT", "FILTER",
                    "DATE-OBS", "BAYERPAT", "CFA", "PCL:CFASOURCEPATTERN",
                )
                shown = set()
                for key in preferred:
                    if key in values and values[key] not in (None, ""):
                        lines.append(f"{key}: {values[key]}")
                        shown.add(key)
                extra = [
                    f"{key}: {value}"
                    for key, value in values.items()
                    if key not in shown and value not in (None, "")
                ]
                if extra:
                    lines.append("--- Other XISF ---")
                    lines.extend(extra[:12])
            except Exception as exc:
                lines.append(f"XISF metadata error: {exc}")
        elif suffix in RAW_EXTENSIONS and rawpy is not None:
            try:
                with rawpy.imread(str(path)) as raw:
                    lines.append(f"RAW visible: {raw.raw_image_visible.shape}")
                    pattern = rawpy_bayer_pattern(raw)
                    if pattern:
                        lines.append(f"Bayer: {pattern}")
                    try:
                        lines.append(f"Camera WB: {tuple(round(float(x), 3) for x in raw.camera_whitebalance)}")
                    except Exception:
                        pass
            except Exception as exc:
                lines.append(f"RAW metadata error: {exc}")
        else:
            try:
                with Image.open(path) as im:
                    lines.append(f"Image: {im.size[0]} x {im.size[1]}")
                    lines.append(f"Mode: {im.mode}")
                    exif = im.getexif()
                    if exif:
                        exif_map = {
                            33434: "Exposure",
                            33437: "FNumber",
                            34855: "ISO",
                            37386: "Focal length",
                            36867: "Date taken",
                            271: "Camera make",
                            272: "Camera model",
                        }
                        for tag, label in exif_map.items():
                            value = exif.get(tag)
                            if value is not None:
                                lines.append(f"{label}: {value}")
            except Exception:
                pass

        try:
            lines.append(f"Folder: {path.parent}")
        except Exception:
            pass
        return "\n".join(lines)

    def update_metadata_panel(self, path: Optional[Path], img: Optional[np.ndarray] = None):
        if not hasattr(self, "metadata_label"):
            return
        if path is None:
            self.metadata_label.setText("-")
            return
        self.metadata_label.setText(self.format_metadata_for_path(Path(path), img))

    def choose_calibration_source(self, kind: str):
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Question)
        msg.setWindowTitle(
            f"Vybrat {kind}" if self.language == "cz" else f"Choose {kind}"
        )
        msg.setText(
            f"Vyber hotový Master {kind}, nebo složku s jednotlivými {kind} snímky."
            if self.language == "cz"
            else f"Choose a finished Master {kind} file or a folder with individual {kind} frames."
        )
        file_btn = msg.addButton(
            "Master soubor" if self.language == "cz" else "Master file",
            QMessageBox.AcceptRole,
        )
        folder_btn = msg.addButton(
            "Složka se snímky" if self.language == "cz" else "Frames folder",
            QMessageBox.ActionRole,
        )
        msg.addButton(QMessageBox.Cancel)
        msg.setDefaultButton(folder_btn)
        msg.exec()

        selected = ""
        if msg.clickedButton() is file_btn:
            selected, _ = QFileDialog.getOpenFileName(
                self,
                f"Vybrat Master {kind}" if self.language == "cz" else f"Choose Master {kind}",
                "",
        "Obrázky/XISF/FIT/RAW (*.xisf *.fits *.fit *.cr2 *.cr3 *.nef *.arw *.dng *.orf *.rw2 *.raf *.tif *.tiff *.png *.jpg *.jpeg *.bmp)",
            )
        elif msg.clickedButton() is folder_btn:
            selected = QFileDialog.getExistingDirectory(
                self,
                f"Vybrat složku {kind}" if self.language == "cz" else f"Choose {kind} frames folder",
                "",
            )
        if not selected:
            return

        source = Path(selected)
        setattr(self, f"{kind.lower()}_frame_path", source)
        label = getattr(self, f"{kind.lower()}_label", None)
        if label is not None:
            suffix = " (složka)" if self.language == "cz" and source.is_dir() else " (folder)" if source.is_dir() else ""
            label.setText(f"{kind}: {source.name}{suffix}")
        self.status_label.setText(
            f"{kind} složka vybrána: {source}" if self.language == "cz" and source.is_dir()
            else f"{kind} Frame vybrán: {source}" if self.language == "cz"
            else f"{kind} folder selected: {source}" if source.is_dir()
            else f"{kind} Frame selected: {source}"
        )
        self.update_frame_quality_title()

    def choose_flat_frame(self):
        self.choose_calibration_source("Flat")

    def choose_bias_frame(self):
        self.choose_calibration_source("Bias")

    def choose_dark_frame(self):
        self.choose_calibration_source("Dark")

    def clear_calibration_frames(self):
        self.flat_frame_path = None
        self.bias_frame_path = None
        self.dark_frame_path = None
        if hasattr(self, "flat_label"):
            self.flat_label.setText(self.tr_ui("flat_unused"))
        if hasattr(self, "bias_label"):
            self.bias_label.setText(self.tr_ui("bias_unused"))
        if hasattr(self, "dark_label"):
            self.dark_label.setText(self.tr_ui("dark_unused"))
        self.status_label.setText(self.tr_ui("calibration_off"))
        self.update_frame_quality_title()

    def clear_cache_dialog(self):
        if self.folder is None:
            QMessageBox.warning(self, self.tr_ui("missing_folder_title"), self.tr_ui("missing_folder_message"))
            return

        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Question)
        msg.setWindowTitle(self.tr_ui("clear_cache_title"))
        msg.setText(self.tr_ui("clear_cache_message"))
        current_btn = msg.addButton(self.tr_ui("clear_cache_current"), QMessageBox.AcceptRole)
        all_btn = msg.addButton(self.tr_ui("clear_cache_all"), QMessageBox.DestructiveRole)
        msg.addButton(QMessageBox.Cancel)
        msg.setDefaultButton(current_btn)
        msg.exec()

        clicked = msg.clickedButton()
        if clicked is current_btn:
            cache_dirs = find_astrostacker_cache_dirs(self.folder, recursive=False)
        elif clicked is all_btn:
            cache_dirs = find_astrostacker_cache_dirs(self.folder, recursive=True)
        else:
            return

        removed = 0
        failed = 0
        for cache_dir in cache_dirs:
            rm, fl = remove_cache_dir_safely(cache_dir)
            removed += rm
            failed += fl

        if removed <= 0 and failed <= 0:
            text = self.tr_ui("clear_cache_none")
        else:
            text = self.tr_ui("clear_cache_done").format(files=removed)
            if failed:
                text += self.tr_ui("clear_cache_failed").format(failed=failed)
        self.status_label.setText(text)
        QMessageBox.information(self, self.tr_ui("clear_cache_title"), text)

    def show_original_stacked_image(self):
        original = getattr(self, "original_linear_result", None)
        if original is None:
            original = self.linear_result
        if original is None:
            QMessageBox.information(self, self.tr_ui("missing_image_title"), self.tr_ui("missing_image_message"))
            return

        self.push_preview_undo_state(copy_pixels=True)
        self.linear_result = original
        self.preview_override = None
        self.preview_override_path = None
        self.preview_source_shape = original.shape[:2]
        self.neutralized_preview_layer = None
        self.neutralized_preview_base_source_id = None
        self.gradient_preview_layer = None
        self.gradient_preview_base_source_id = None
        self.preview_auto_display_stretch = False
        self.reset_preview_display_limits()
        self.flip_horizontal = False
        self.flip_vertical = False
        self.preview_rotation_degrees = 0
        self.reset_stretch(push_undo=False)
        self.status_label.setText(self.tr_ui("show_stacked_done"))

    def find_pixinsight_wrapper_script(self) -> Optional[Path]:
        roots = []
        try:
            roots.append(Path(__file__).resolve().parent)
        except Exception:
            pass
        if getattr(sys, "frozen", False):
            roots.append(Path(sys.executable).resolve().parent)
        roots.append(Path.cwd())
        roots.append(Path.home() / "Downloads")

        candidates = []
        for root in roots:
            candidates.extend([
                root / "AS_Stacker_PI_Plugin" / "AS_Stacker_PI.js",
                root / "AS_Stacker_PI.js",
            ])
        for path in candidates:
            if path.exists():
                return path
        return None

    def find_pixinsight_executable(self) -> Optional[Path]:
        env_candidates = [
            os.environ.get("PIXINSIGHT_EXECUTABLE"),
            os.environ.get("PIXINSIGHT_PATH"),
        ]
        candidates = [Path(p) for p in env_candidates if p]
        candidates.extend([
            Path("/Applications/PixInsight/PixInsight.app/Contents/MacOS/PixInsight"),
            Path("/Applications/PixInsight/PixInsight"),
            Path("/Applications/OLD_PixInsight/PixInsight.app/Contents/MacOS/PixInsight"),
            Path("/opt/PixInsight/bin/PixInsight"),
            Path("/usr/local/bin/PixInsight"),
            Path(r"C:\Program Files\PixInsight\bin\PixInsight.exe"),
            Path(r"C:\Program Files\PixInsight\PixInsight.exe"),
        ])
        for path in candidates:
            if path.exists():
                return path
        return None

    def pixinsight_is_running(self) -> bool:
        if sys.platform != "darwin":
            return False
        try:
            result = subprocess.run(
                ["pgrep", "-x", "PixInsight"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            return result.returncode == 0
        except Exception:
            return False

    def create_pixinsight_run_script(self, wrapper_path: Path) -> Path:
        script_file = str(wrapper_path).replace("\\", "\\\\").replace('"', '\\"')
        script_path = Path(tempfile.gettempdir()) / "AS_Stacker_run_registered.scp"
        script_path.write_text(
            f'run --execute-mode=auto "{script_file}"\n',
            encoding="utf-8",
        )
        return script_path

    def execute_pixinsight_run_script(self, pixinsight_exe: Path, run_script: Path):
        try:
            subprocess.Popen([str(pixinsight_exe), f"--execute={run_script}"])
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Chyba PixInsight" if self.language == "cz" else "PixInsight error",
                str(exc),
            )

    def launch_pixinsight_wrapper(self):
        wrapper_path = self.find_pixinsight_wrapper_script()
        if wrapper_path is None:
            filename, _ = QFileDialog.getOpenFileName(
                self,
                "Vybrat AS_Stacker_PI.js" if self.language == "cz" else "Choose AS_Stacker_PI.js",
                str(Path.cwd()),
                "PixInsight scripts (*.js);;All files (*)" if self.language == "en" else "PixInsight skripty (*.js);;Všechny soubory (*)",
            )
            if not filename:
                return
            wrapper_path = Path(filename)

        pixinsight_exe = self.find_pixinsight_executable()
        if pixinsight_exe is None:
            QMessageBox.warning(
                self,
                "PixInsight nenalezen" if self.language == "cz" else "PixInsight not found",
                (
                    "Nepodařilo se najít spustitelný PixInsight. Nastav proměnnou PIXINSIGHT_EXECUTABLE "
                    "nebo spusť wrapper ručně z PixInsight menu."
                    if self.language == "cz"
                    else "Could not find the PixInsight executable. Set the PIXINSIGHT_EXECUTABLE environment variable "
                    "or run the wrapper manually from the PixInsight Script menu."
                ),
            )
            return

        try:
            run_script = self.create_pixinsight_run_script(wrapper_path)
            if self.pixinsight_is_running():
                self.execute_pixinsight_run_script(pixinsight_exe, run_script)
            else:
                subprocess.Popen([str(pixinsight_exe)])
                QTimer.singleShot(
                    9000,
                    lambda exe=pixinsight_exe, script=run_script: self.execute_pixinsight_run_script(exe, script),
                )
            self.status_label.setText(
                "Spouštím PixInsight; wrapper se otevře automaticky po startu."
                if self.language == "cz"
                else "Launching PixInsight; the wrapper will open automatically after startup."
            )
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Chyba PixInsight" if self.language == "cz" else "PixInsight error",
                str(exc),
            )

    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, self.tr_ui("choose_images_folder"))
        if folder:
            self.folder = Path(folder)
            self.folder_label.setText(f"{self.tr_ui('folder_prefix')}: {self.folder}")
            count = len([p for p in self.folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS])
            self.status_label.setText(self.tr_ui("found_images").format(count=count))
            self.manual_excluded_paths = set()
            self.review_ready = False
            self.stack_btn.setText(self.tr_ui("start_stack"))
            self.clear_stack_selection_info()
            paths = self.collect_preview_paths_for_folder(self.folder)
            self.set_preview_sequence(paths)
            if paths:
                self.load_preview_image(paths[0])
            self.manual_reference_path = None
            self.manual_comet_xy = None
            self.manual_comet_reference_path = None
            self.manual_comet_end_xy = None
            self.manual_comet_end_path = None
            self.preview_override_path = None
        self.neutralized_preview_layer = None
        self.neutralized_preview_base_source_id = None
        self.gradient_preview_layer = None
        self.gradient_preview_base_source_id = None
        self.flip_horizontal = False
        self.flip_vertical = False
        self.preview_rotation_degrees = 0

    def auto_stack_status_text(self, last_name: Optional[str] = None) -> str:
        pixel_pct = 0.0
        if self.auto_stack_weight is not None and self.auto_stack_rejected_pixels_last > 0:
            total_px = max(1, int(self.auto_stack_weight.size))
            pixel_pct = 100.0 * self.auto_stack_rejected_pixels_last / total_px
        flip_count = getattr(self, "auto_stack_flip_180_count", 0)
        if self.language == "cz":
            text = (
                f"Live stacking: složeno {self.auto_stack_accepted}, "
                f"vyřazeno {self.auto_stack_rejected}"
            )
            if pixel_pct > 0:
                text += f", odlehlé pixely v posl. snímku: {pixel_pct:.2f}%"
            if flip_count > 0:
                text += f", meridian flip korekce: {flip_count}"
            if last_name:
                text += f"; poslední: {last_name}"
            return text
        text = (
            f"Live stacking: stacked {self.auto_stack_accepted}, "
            f"rejected {self.auto_stack_rejected}"
        )
        if pixel_pct > 0:
            text += f", outlier pixels in last frame: {pixel_pct:.2f}%"
        if flip_count > 0:
            text += f", meridian flip corrections: {flip_count}"
        if last_name:
            text += f"; last: {last_name}"
        return text

    def on_auto_stack_reject_toggled(self, checked: bool):
        self.auto_stack_reject_outliers = bool(checked)

    def on_auto_stack_reject_elongated_toggled(self, checked: bool):
        self.auto_stack_reject_elongated_stars = bool(checked)

    def start_auto_stacking(self):
        if self.auto_stack_calibration_worker is not None:
            return
        if self.auto_stack_worker is not None:
            return
        if self.folder is None:
            folder = QFileDialog.getExistingDirectory(self, self.tr_ui("choose_images_folder"))
            if not folder:
                return
            self.folder = Path(folder)
            self.folder_label.setText(f"{self.tr_ui('folder_prefix')}: {self.folder}")

        self.auto_stack_active = True
        self.auto_stack_pending = {}
        self.auto_stack_queued_paths = set()
        self.auto_stack_ready_queue = []
        self.auto_stack_reference_gray = None
        self.auto_stack_reference_shape = None
        self.auto_stack_sum = None
        self.auto_stack_weight = None
        self.auto_stack_sumsq = None
        self.auto_stack_rejected_pixels_last = 0
        self.auto_stack_flip_180_count = 0
        self.auto_stack_flat = None
        self.auto_stack_bias = None
        self.auto_stack_dark = None
        self.auto_stack_accepted = 0
        self.auto_stack_rejected = 0
        self.auto_stack_last_quality_metrics = None
        self.auto_stack_balance_applied = False

        self.linear_result = None
        self.original_linear_result = None
        self.preview_override = None
        self.preview_override_path = None
        self.neutralized_preview_layer = None
        self.neutralized_preview_base_source_id = None
        self.gradient_preview_layer = None
        self.gradient_preview_base_source_id = None
        self.preview_auto_display_stretch = True
        self.reset_preview_display_limits()
        self.zoom_mode = "fit"
        self.progress.setValue(0)
        self.status_label.setText(
            "Live stacking připravuje Flat/Bias/Dark kalibraci…"
            if self.language == "cz"
            else "Live stacking is preparing Flat/Bias/Dark calibration…"
        )
        self.auto_stack_start_action.setEnabled(False)
        self.auto_stack_stop_action.setEnabled(True)
        self.stack_btn.setEnabled(False)
        self.open_action.setEnabled(False)
        self.open_image_action.setEnabled(False)
        settings = self.current_stack_settings()
        settings = replace(settings, align_mode="star_affine", stack_mode="median")
        self.auto_stack_reject_outliers = bool(self.auto_stack_reject_action.isChecked())
        self.auto_stack_reject_sigma = float(getattr(settings, "live_stack_reject_sigma", 3.0))
        self.auto_stack_reject_min_frames = int(getattr(settings, "live_stack_reject_min_frames", 4))
        self.auto_stack_reject_elongated_stars = bool(
            getattr(self, "auto_stack_reject_elongated_action", None) is None
            or self.auto_stack_reject_elongated_action.isChecked()
        )
        self.auto_stack_min_roundness = float(getattr(settings, "live_stack_min_roundness", 0.50))
        self.auto_stack_min_shape_stars = int(getattr(settings, "live_stack_min_shape_stars", 8))
        self.auto_stack_calibration_worker = AutoStackCalibrationWorker(settings)
        self.auto_stack_calibration_worker.progress.connect(self.on_auto_stack_calibration_progress)
        self.auto_stack_calibration_worker.masters_ready.connect(self.on_auto_stack_calibration_ready)
        self.auto_stack_calibration_worker.failed.connect(self.on_auto_stack_calibration_failed)
        self.auto_stack_calibration_worker.finished.connect(self.on_auto_stack_calibration_finished)
        self.auto_stack_calibration_worker.start()

    def stop_auto_stacking(self):
        if not self.auto_stack_active and self.auto_stack_worker is None:
            return
        self.auto_stack_active = False
        self.auto_stack_timer.stop()
        self.auto_stack_ready_queue.clear()
        if self.auto_stack_calibration_worker is not None and self.auto_stack_calibration_worker.isRunning():
            self.auto_stack_calibration_worker.requestInterruption()
        if self.auto_stack_worker is not None and self.auto_stack_worker.isRunning():
            self.auto_stack_worker.requestInterruption()
        elif self.auto_stack_calibration_worker is None or not self.auto_stack_calibration_worker.isRunning():
            self.auto_stack_start_action.setEnabled(True)
            self.stack_btn.setEnabled(True)
            self.open_action.setEnabled(True)
            self.open_image_action.setEnabled(True)
        self.auto_stack_stop_action.setEnabled(False)
        self.status_label.setText(
            (
                f"Live stacking zastaven. Složeno {self.auto_stack_accepted} snímků, "
                f"vyřazeno {self.auto_stack_rejected}."
            )
            if self.language == "cz"
            else (
                f"Live stacking stopped. Stacked {self.auto_stack_accepted} frames, "
                f"rejected {self.auto_stack_rejected}."
            )
        )

    def on_auto_stack_calibration_progress(self, message: str):
        if self.auto_stack_active:
            self.status_label.setText(message)

    def on_auto_stack_calibration_ready(
        self,
        flat: Optional[np.ndarray],
        bias: Optional[np.ndarray],
        dark: Optional[np.ndarray],
    ):
        if not self.auto_stack_active:
            return
        self.auto_stack_flat = flat
        self.auto_stack_bias = bias
        self.auto_stack_dark = dark
        active = []
        if flat is not None:
            active.append("Flat")
        if bias is not None:
            active.append("Bias")
        if dark is not None:
            active.append("Dark")
        calibration_text = ", ".join(active) if active else (
            "bez kalibrace" if self.language == "cz" else "no calibration"
        )
        self.status_label.setText(
            f"Live stacking sleduje složku; kalibrace: {calibration_text}."
            if self.language == "cz"
            else f"Live stacking is watching the folder; calibration: {calibration_text}."
        )
        self.auto_stack_timer.start()
        self.scan_auto_stack_folder()

    def on_auto_stack_calibration_failed(self, message: str):
        self.auto_stack_active = False
        self.auto_stack_stop_action.setEnabled(False)
        self.auto_stack_start_action.setEnabled(True)
        self.stack_btn.setEnabled(True)
        self.open_action.setEnabled(True)
        self.open_image_action.setEnabled(True)
        QMessageBox.critical(
            self,
            "Chyba kalibrace" if self.language == "cz" else "Calibration error",
            message,
        )

    def on_auto_stack_calibration_finished(self):
        worker = self.auto_stack_calibration_worker
        self.auto_stack_calibration_worker = None
        if worker is not None:
            worker.deleteLater()
        if not self.auto_stack_active:
            self.auto_stack_start_action.setEnabled(True)
            self.stack_btn.setEnabled(True)
            self.open_action.setEnabled(True)
            self.open_image_action.setEnabled(True)

    def scan_auto_stack_folder(self):
        if not self.auto_stack_active or self.folder is None:
            return
        extensions = (
            RAW_STACK_EXTENSIONS
            if hasattr(self, "fit_only_check") and self.fit_only_check.isChecked()
            else IMAGE_EXTENSIONS
        )
        try:
            candidates = [
                path for path in self.folder.iterdir()
                if path.is_file()
                and path.suffix.lower() in extensions
                and not path.name.startswith(".")
                and not path.name.lower().startswith(("master", "stacked_result"))
            ]
            candidates.sort(key=lambda path: (path.stat().st_mtime_ns, path.name.lower()))
        except Exception as exc:
            self.status_label.setText(
                f"Chyba sledování složky: {exc}"
                if self.language == "cz"
                else f"Folder watch error: {exc}"
            )
            return

        now_ns = time.time_ns()
        present = set()
        for path in candidates:
            key = str(path.resolve())
            present.add(key)
            if key in self.auto_stack_queued_paths:
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            old = self.auto_stack_pending.get(key)
            stable_count = old[2] + 1 if old and old[:2] == (stat.st_size, stat.st_mtime_ns) else 0
            self.auto_stack_pending[key] = (stat.st_size, stat.st_mtime_ns, stable_count)
            age_seconds = max(0.0, (now_ns - stat.st_mtime_ns) / 1_000_000_000.0)
            # Two cameras may write large frames concurrently and briefly
            # pause between chunks. Require three unchanged polls plus age.
            if stable_count >= 3 and age_seconds >= 2.5 and stat.st_size > 0:
                self.auto_stack_queued_paths.add(key)
                self.auto_stack_ready_queue.append(path)
                self.auto_stack_pending.pop(key, None)

        for key in list(self.auto_stack_pending):
            if key not in present:
                self.auto_stack_pending.pop(key, None)
        self.start_next_auto_stack_frame()

    def start_next_auto_stack_frame(self):
        if not self.auto_stack_active:
            return
        # Keep strict ownership until the worker's finished signal is handled.
        # isRunning() may already be false while that queued signal is still
        # waiting in the GUI event loop.
        if self.auto_stack_worker is not None:
            return
        if not self.auto_stack_ready_queue:
            return
        path = self.auto_stack_ready_queue.pop(0)
        settings = self.current_stack_settings()
        settings = replace(settings, align_mode="star_affine", stack_mode="mean")
        self.status_label.setText(
            f"Live stacking zpracovává: {path.name}"
            if self.language == "cz"
            else f"Live stacking is processing: {path.name}"
        )
        worker = AutoStackFrameWorker(
            path,
            settings,
            self.auto_stack_reference_gray,
            self.auto_stack_reference_shape,
            self.auto_stack_flat,
            self.auto_stack_bias,
            self.auto_stack_dark,
        )
        self.auto_stack_worker = worker
        worker.frame_ready.connect(self.on_auto_stack_frame_ready)
        worker.failed.connect(self.on_auto_stack_frame_failed)
        worker.finished.connect(
            lambda finished_worker=worker: self.on_auto_stack_worker_finished(
                finished_worker
            )
        )
        worker.start()

    def on_auto_stack_frame_ready(
        self,
        path_text: str,
        aligned: np.ndarray,
        coverage: np.ndarray,
        detail: Dict[str, Any],
    ):
        if not self.auto_stack_active:
            return
        if detail.get("flip_180"):
            self.auto_stack_flip_180_count = getattr(self, "auto_stack_flip_180_count", 0) + 1
        image = np.asarray(aligned, dtype=np.float32)
        mask = np.asarray(coverage, dtype=np.float32)
        if self.should_reject_live_stack_elongated_stars(image):
            return self.reject_auto_stack_frame_for_quality(path_text, image)
        if self.auto_stack_reference_gray is None:
            self.auto_stack_reference_gray = np.ascontiguousarray(to_gray_float(image))
            self.auto_stack_reference_shape = tuple(image.shape)
            self.auto_stack_sum = np.zeros_like(image, dtype=np.float32)
            self.auto_stack_sumsq = np.zeros_like(image, dtype=np.float32)
            self.auto_stack_weight = np.zeros(image.shape[:2], dtype=np.float32)
            self.update_metadata_panel(Path(path_text), image)

        if self.auto_stack_sum is None or self.auto_stack_weight is None:
            return
        if self.auto_stack_sumsq is None:
            self.auto_stack_sumsq = np.zeros_like(image, dtype=np.float32)

        # Online (running) kappa-sigma rejection: porovná nový pixel proti dosavadnímu
        # průběžnému průměru/std. odchylce na dané pozici a vynechá ho ze skládání,
        # pokud je výrazně přejasněný (letadlo, satelit, kosmické záření, hot pixel).
        # Nepotřebuje držet historii snímků v paměti - jen sum/sumsq/weight z minulých snímků.
        weight_before = self.auto_stack_weight
        accept_mask = mask
        rejected_now = 0
        if self.auto_stack_reject_outliers:
            enough_history = weight_before >= max(1, self.auto_stack_reject_min_frames)
            if np.any(enough_history):
                safe_weight_before = np.maximum(weight_before, 1.0)
                weight_b = safe_weight_before[..., None] if image.ndim == 3 else safe_weight_before
                running_mean = self.auto_stack_sum / weight_b
                running_var = self.auto_stack_sumsq / weight_b - running_mean ** 2
                running_std = np.sqrt(np.maximum(running_var, 0.0)) + 1e-3
                # Odmítáme jen kladné odlehlé hodnoty (přejasněné stopy), ne podexponované
                # pixely u okraje pokrytí - ty řeší už samotná coverage maska.
                is_outlier_channel = (image - running_mean) > (self.auto_stack_reject_sigma * running_std)
                outlier_pixel = np.any(is_outlier_channel, axis=2) if image.ndim == 3 else is_outlier_channel
                outlier_pixel = outlier_pixel & enough_history
                if np.any(outlier_pixel):
                    accept_mask = mask * (1.0 - outlier_pixel.astype(np.float32))
                    rejected_now = int(np.sum(outlier_pixel & (mask > 0)))
        self.auto_stack_rejected_pixels_last = rejected_now

        if image.ndim == 3:
            self.auto_stack_sum += image * accept_mask[..., None]
            self.auto_stack_sumsq += (image ** 2) * accept_mask[..., None]
        else:
            self.auto_stack_sum += image * accept_mask
            self.auto_stack_sumsq += (image ** 2) * accept_mask
        self.auto_stack_weight += accept_mask
        safe_weight = np.maximum(self.auto_stack_weight, 1.0)
        if image.ndim == 3:
            result = self.auto_stack_sum / safe_weight[..., None]
            result[self.auto_stack_weight <= 0.0] = 0.0
        else:
            result = self.auto_stack_sum / safe_weight
            result[self.auto_stack_weight <= 0.0] = 0.0

        self.auto_stack_accepted += 1
        self.linear_result = np.ascontiguousarray(result.astype(np.float32))
        self.original_linear_result = self.linear_result
        self.preview_override = None
        self.preview_override_path = None
        self.preview_source_shape = self.linear_result.shape[:2]
        self.preview_auto_display_stretch = True
        self.reset_preview_display_limits()
        first_live_balance = not getattr(self, "auto_stack_balance_applied", False)
        balance_applied = self.apply_balance_to_current_preview(
            record_undo=False,
            refresh_preview=False,
            set_status=False,
            update_sliders=first_live_balance,
        )
        if first_live_balance and balance_applied:
            self.auto_stack_balance_applied = True
        self.update_preview()
        self.progress.setValue(min(99, self.auto_stack_accepted))
        self.status_label.setText(self.auto_stack_status_text(Path(path_text).name))

    def should_reject_live_stack_elongated_stars(self, image: np.ndarray) -> bool:
        if not getattr(self, "auto_stack_reject_elongated_stars", True):
            self.auto_stack_last_quality_metrics = None
            return False
        try:
            metrics = frame_quality_metrics_from_gray(to_gray_float(image))
        except Exception as exc:
            log_debug(f"Live stack elongated-star check failed: {exc}")
            self.auto_stack_last_quality_metrics = None
            return False
        self.auto_stack_last_quality_metrics = metrics
        measured = float(metrics.get("shape_star_count", 0.0))
        if measured < max(1, int(getattr(self, "auto_stack_min_shape_stars", 8))):
            return False
        roundness = float(metrics.get("roundness", 1.0))
        return roundness < float(getattr(self, "auto_stack_min_roundness", 0.50))

    def reject_auto_stack_frame_for_quality(self, path_text: str, image: np.ndarray):
        metrics = dict(getattr(self, "auto_stack_last_quality_metrics", None) or {})
        roundness = float(metrics.get("roundness", 0.0))
        measured = int(round(float(metrics.get("shape_star_count", 0.0))))
        self.auto_stack_rejected += 1
        reason = (
            f"protažené hvězdy (kulatost {roundness:.2f}, hvězd {measured})"
            if self.language == "cz"
            else f"elongated stars (roundness {roundness:.2f}, stars {measured})"
        )
        log_debug(f"Automatic stacking rejected {path_text}: {reason}")
        self.status_label.setText(
            (
                f"{self.auto_stack_status_text(Path(path_text).name)}; důvod: {reason}"
                if self.language == "cz"
                else f"{self.auto_stack_status_text(Path(path_text).name)}; reason: {reason}"
            )
        )

    def on_auto_stack_frame_failed(self, path_text: str, message: str):
        if not self.auto_stack_active:
            return
        self.auto_stack_rejected += 1
        log_debug(f"Automatic stacking rejected {path_text}: {message}")
        self.status_label.setText(
            (
                f"{self.auto_stack_status_text(Path(path_text).name)}; důvod: {message}"
                if self.language == "cz"
                else f"{self.auto_stack_status_text(Path(path_text).name)}; reason: {message}"
            )
        )

    def on_auto_stack_worker_finished(self, worker: AutoStackFrameWorker):
        # A stale completion must never clear a newer worker.
        if self.auto_stack_worker is not worker:
            worker.deleteLater()
            return
        self.auto_stack_worker = None
        worker.deleteLater()
        if self.auto_stack_active and self.auto_stack_worker is None:
            QTimer.singleShot(0, self.start_next_auto_stack_frame)
        else:
            self.auto_stack_start_action.setEnabled(True)
            self.stack_btn.setEnabled(True)
            self.open_action.setEnabled(True)
            self.open_image_action.setEnabled(True)

    def open_image_file(self):
        """Otevře jeden samostatný obrázek nebo FIT/FITS do náhledu.

        Tato funkce nemění vybranou složku ani profil nastavení. Slouží jen
        k rychlému prohlédnutí zdrojového nebo již složeného snímku se stejným
        stretch/zoom ovládáním jako běžný výsledek.
        """
        recent_paths = self.recent_image_paths()
        if recent_paths:
            start_dir = str(recent_paths[0].parent)
        else:
            start_dir = str(self.folder) if self.folder else ""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            self.tr_ui("open_image_title"),
            start_dir,
            f"Images/XISF/FIT/RAW (*.xisf *.fit *.fits *.cr2 *.cr3 *.raw *.nef *.arw *.dng *.orf *.rw2 *.raf *.tif *.tiff *.png *.jpg *.jpeg *.bmp);;XISF (*.xisf);;RAW (*.cr2 *.cr3 *.raw *.nef *.arw *.dng *.orf *.rw2 *.raf);;FITS (*.fit *.fits);;TIFF (*.tif *.tiff);;{self.tr_ui('all_files')} (*)",
        )
        if not filename:
            return

        self.open_image_path(Path(filename), add_to_recent=True)

    def _sequence_paths_for_ui(self) -> List[Path]:
        if not self.folder:
            return []
        extensions = RAW_STACK_EXTENSIONS if hasattr(self, "fit_only_check") and self.fit_only_check.isChecked() else IMAGE_EXTENSIONS
        paths = sorted([p for p in self.folder.iterdir() if p.suffix.lower() in extensions])
        if self.max_images_spin.value() > 0:
            paths = paths[: self.max_images_spin.value()]
        return paths

    def _load_comet_marking_image(self, path: Path, mode: str):
        try:
            set_bayer_pattern_override(self.bayer_combo.currentData() if hasattr(self, "bayer_combo") else "auto")
            img = load_image_as_float(path)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Chyba" if self.language == "cz" else "Error",
                f"Snímek se nepodařilo načíst:\n{exc}" if self.language == "cz" else f"Could not load image:\n{exc}",
            )
            return

        self.reset_preview_render_state_for_new_source()
        self.preview_override = img
        self.preview_override_path = str(path)
        self.preview_source_shape = img.shape[:2]
        self.reset_denoise_for_new_preview_source()
        self.invalidate_preview_cache()
        self.preview_auto_display_stretch = True
        self.update_metadata_panel(path, img)
        self.awaiting_comet_click = True
        self.comet_click_mode = mode
        if hasattr(self, "image_label"):
            self.image_label.set_marking_mode(True)
        self.zoom_mode = "fit"
        self.zoom_factor = 1.0
        self.apply_balance_to_current_preview(
            record_undo=False,
            refresh_preview=False,
            set_status=False,
        )
        label = self.tr_ui("comet_mark_first_label") if mode == "start" else self.tr_ui("comet_mark_last_label")
        self.status_label.setText(self.tr_ui("comet_click_status").format(label=label, name=path.name))
        self.update_preview()

    def select_comet_reference_point(self):
        """Zpětná kompatibilita: staré tlačítko/menu znamená první bod."""
        self.select_comet_start_point()

    def select_comet_start_point(self):
        """Načte první snímek a nechá uživatele kliknout na jádro komety."""
        if not self.folder:
            QMessageBox.warning(self, self.tr_ui("missing_folder_title"), self.tr_ui("missing_folder_message"))
            return
        paths = self._sequence_paths_for_ui()
        if not paths:
            QMessageBox.warning(self, self.tr_ui("missing_frames_title"), self.tr_ui("missing_frames_message"))
            return
        self._load_comet_marking_image(paths[0], "start")

    def select_comet_end_point(self):
        """Načte poslední snímek a nechá uživatele kliknout na jádro komety."""
        if not self.folder:
            QMessageBox.warning(self, self.tr_ui("missing_folder_title"), self.tr_ui("missing_folder_message"))
            return
        paths = self._sequence_paths_for_ui()
        if not paths:
            QMessageBox.warning(self, self.tr_ui("missing_frames_title"), self.tr_ui("missing_frames_message"))
            return
        self._load_comet_marking_image(paths[-1], "end")

    def clear_comet_marks(self):
        self.manual_comet_xy = None
        self.manual_comet_reference_path = None
        self.manual_comet_end_xy = None
        self.manual_comet_end_path = None
        self.awaiting_comet_click = False
        self.comet_click_mode = None
        if hasattr(self, "image_label"):
            self.image_label.set_marking_mode(False)
        self.status_label.setText(self.tr_ui("clear_comet_marks_done"))
        self.update_preview()

    def on_preview_clicked(self, pixmap_x: float, pixmap_y: float):
        if not self.awaiting_comet_click or self.preview_source_shape is None:
            return

        pixmap = self.image_label.pixmap()
        if pixmap is None or pixmap.isNull():
            return

        src_h, src_w = self.preview_source_shape
        x = float(pixmap_x) * float(src_w) / max(1.0, float(pixmap.width()))
        y = float(pixmap_y) * float(src_h) / max(1.0, float(pixmap.height()))
        x = max(0.0, min(float(src_w - 1), x))
        y = max(0.0, min(float(src_h - 1), y))

        mode = self.comet_click_mode or "start"
        if mode == "end":
            self.manual_comet_end_xy = (x, y)
            self.manual_comet_end_path = self.preview_override_path
            msg = self.tr_ui("comet_last_marked").format(x=x, y=y)
        else:
            self.manual_comet_xy = (x, y)
            self.manual_comet_reference_path = self.preview_override_path
            msg = self.tr_ui("comet_first_marked").format(x=x, y=y)

        self.awaiting_comet_click = False
        self.comet_click_mode = None
        if hasattr(self, "image_label"):
            self.image_label.set_marking_mode(False)
        self.align_combo.setCurrentIndex(self.align_combo.findData("comet"))
        self.auto_reference_check.setChecked(False)

        if self.manual_comet_xy is not None and self.manual_comet_end_xy is not None:
            self.status_label.setText(msg + self.tr_ui("comet_two_point_ready"))
        else:
            self.status_label.setText(msg + self.tr_ui("comet_mark_other"))
        self.update_preview()

    def change_language(self, *_args):
        self.language = self.language_combo.currentData() or "en"
        self.apply_language()

    def change_theme(self, *_args):
        theme = self.theme_combo.currentData() if hasattr(self, "theme_combo") else "dark"
        if theme == "light":
            apply_light_theme(QApplication.instance())
            if hasattr(self, "image_label"):
                self.image_label.setStyleSheet("background: #cbd1da; color: #1f2933; font-size: 18px;")
            if hasattr(self, "histogram_label"):
                self.histogram_label.setStyleSheet("background: #ffffff; border: 1px solid #aab4c2; border-radius: 4px;")
            if hasattr(self, "metadata_label"):
                self.metadata_label.setStyleSheet("font-family: Consolas, monospace; font-size: 11px; color: #1f2933; background: #ffffff; border: 1px solid #aab4c2; border-radius: 4px; padding: 6px;")
        else:
            apply_dark_theme(QApplication.instance())
            if hasattr(self, "image_label"):
                self.image_label.setStyleSheet("background: #111; color: #ddd; font-size: 18px;")
            if hasattr(self, "histogram_label"):
                self.histogram_label.setStyleSheet("background: #121212; border: 1px solid #444; border-radius: 4px;")
            if hasattr(self, "metadata_label"):
                self.metadata_label.setStyleSheet("font-family: Consolas, monospace; font-size: 11px; color: #ddd; background: #151515; border: 1px solid #444; border-radius: 4px; padding: 6px;")
        self._update_activity_overlay_theme()

    def apply_language(self):
        self.setWindowTitle(self.tr_ui("window_title"))
        if hasattr(self, "activity_title_label"):
            self.activity_title_label.setText(self.tr_ui("activity_title"))
        if hasattr(self, "title_label"):
            self.title_label.setText(self.tr_ui("settings"))
        if hasattr(self, "folder_label") and self.folder is None:
            self.folder_label.setText(self.tr_ui("folder_none"))
        if hasattr(self, "image_label") and self.linear_result is None and self.preview_override is None:
            self.show_intro_preview()
        if hasattr(self, "max_images_spin"):
            self.max_images_spin.setSuffix(self.tr_ui("max_images_suffix"))
        if hasattr(self, "processes_spin"):
            self.processes_spin.setSuffix(self.tr_ui("processes_suffix"))
        if hasattr(self, "process_mode_combo"):
            current = self.process_mode_combo.currentData() or "auto"
            self.process_mode_combo.blockSignals(True)
            self.process_mode_combo.clear()
            self.process_mode_combo.addItem(self.tr_ui("cpu_auto"), "auto")
            self.process_mode_combo.addItem(self.tr_ui("cpu_manual"), "manual")
            self.process_mode_combo.setCurrentIndex(max(0, self.process_mode_combo.findData(current)))
            self.process_mode_combo.blockSignals(False)
            self.update_process_mode()
        if hasattr(self, "theme_combo"):
            current = self.theme_combo.currentData() or "dark"
            self.theme_combo.blockSignals(True)
            self.theme_combo.clear()
            self.theme_combo.addItem(self.tr_ui("theme_dark"), "dark")
            self.theme_combo.addItem(self.tr_ui("theme_light"), "light")
            self.theme_combo.setCurrentIndex(max(0, self.theme_combo.findData(current)))
            self.theme_combo.blockSignals(False)
        if hasattr(self, "denoise_mode_combo"):
            current = self.denoise_mode_combo.currentData() or "classic"
            self.denoise_mode_combo.blockSignals(True)
            self.denoise_mode_combo.clear()
            self.denoise_mode_combo.addItem(self.tr_ui("denoise_classic"), "classic")
            if not LITE_BUILD:
                self.denoise_mode_combo.addItem(self.tr_ui("denoise_drunet"), "drunet")
            self.denoise_mode_combo.setCurrentIndex(
                max(0, self.denoise_mode_combo.findData(current))
            )
            self.denoise_mode_combo.setToolTip(self.tr_ui("drunet_tooltip"))
            self.denoise_mode_combo.blockSignals(False)
        if hasattr(self, "select_drunet_model_btn"):
            self.select_drunet_model_btn.setText(self.tr_ui("select_drunet_model"))
            self.select_drunet_model_btn.setToolTip(self.tr_ui("drunet_tooltip"))
        if hasattr(self, "ui_mode_combo"):
            current = self.ui_mode_combo.currentData() or "advanced"
            self.ui_mode_combo.blockSignals(True)
            self.ui_mode_combo.clear()
            self.ui_mode_combo.addItem(self.tr_ui("ui_advanced"), "advanced")
            self.ui_mode_combo.addItem(self.tr_ui("ui_simple"), "simple")
            self.ui_mode_combo.setCurrentIndex(max(0, self.ui_mode_combo.findData(current)))
            self.ui_mode_combo.blockSignals(False)
            self.apply_ui_mode()
        if hasattr(self, "flat_label") and self.flat_frame_path is None:
            self.flat_label.setText(self.tr_ui("flat_unused"))
        if hasattr(self, "bias_label") and self.bias_frame_path is None:
            self.bias_label.setText(self.tr_ui("bias_unused"))
        if hasattr(self, "dark_label") and self.dark_frame_path is None:
            self.dark_label.setText(self.tr_ui("dark_unused"))
        if hasattr(self, "fit_only_check"):
            self.fit_only_check.setToolTip(
                "Při skládání použije pouze XISF, FIT/FITS a foto RAW soubory; ignoruje JPG/PNG/BMP/TIFF ve stejné složce."
                if self.language == "cz"
                else "Uses only XISF, FIT/FITS and camera RAW files for stacking; ignores JPG/PNG/BMP/TIFF files in the same folder."
            )

        direct = {
            "choose_btn": "choose_folder",
            "open_image_btn": "open_image",
            "pixinsight_btn": "pixinsight",
            "normalize_check": "normalize_bg",
            "fit_only_check": "fit_only",
            "mp_check": "mp_cpu",
            "gpu_check": "gpu",
            "completion_sound_check": "completion_sound",
            "auto_reference_check": "auto_ref",
            "review_frames_check": "review_frames",
            "manual_reference_btn": "manual_reference",
            "mosaic_mode_check": "mosaic_mode",
            "quality_filter_check": "quality_filter",
            "comet_refine_check": "comet_refine",
            "strict_star_filter_check": "strict_stars",
            "satellite_trail_check": "satellite_trail_filter",
            "comet_select_btn": "mark_comet_start",
            "comet_end_btn": "mark_comet_end",
            "clear_comet_marks_btn": "clear_comet_marks",
            "stack_btn": "start_stack",
            "stop_btn": "stop_stack",
            "clear_calib_btn": "reset_calib",
            "show_stacked_btn": "show_stacked",
            "clear_cache_btn": "clear_cache",
            "undo_btn": "undo",
            "hist_title": "histogram",
            "auto_stretch_btn": "auto_stretch",
            "crop_edges_btn": "crop_edges",
            "crop_select_btn": "crop_select",
            "neutral_bg_btn": "neutralize",
            "clear_neutral_bg_btn": "clear_neutralize",
            "remove_gradient_btn": "remove_gradient",
            "clear_gradient_btn": "clear_gradient",
            "flip_h_btn": "flip_h",
            "flip_v_btn": "flip_v",
            "rotate_left_btn": "rotate_left",
            "rotate_right_btn": "rotate_right",
            "preview_view_title": "preview_view",
            "fit_btn": "fit",
            "reset_btn": "reset",
        }
        for attr, key in direct.items():
            widget = getattr(self, attr, None)
            if widget is not None:
                widget.setText(self.tr_ui(key) if key in self.TRANSLATIONS["cz"] else key)
        if hasattr(self, "pixinsight_btn"):
            self.pixinsight_btn.setToolTip(self.tr_ui("pixinsight_tooltip"))
        if hasattr(self, "show_stacked_btn"):
            self.show_stacked_btn.setToolTip(self.tr_ui("show_stacked_tooltip"))
        if hasattr(self, "undo_btn"):
            self.undo_btn.setToolTip(self.tr_ui("undo_tooltip"))
        if hasattr(self, "clear_cache_btn"):
            self.clear_cache_btn.setToolTip(self.tr_ui("clear_cache_tooltip"))
        if hasattr(self, "left_collapse_btn"):
            self.left_collapse_btn.setToolTip(self.tr_ui("hide_left_panel"))
        if hasattr(self, "left_expand_btn"):
            self.left_expand_btn.setToolTip(self.tr_ui("show_left_panel"))
        if hasattr(self, "right_collapse_btn"):
            self.right_collapse_btn.setToolTip(self.tr_ui("hide_right_panel"))
        if hasattr(self, "right_expand_btn"):
            self.right_expand_btn.setToolTip(self.tr_ui("show_right_panel"))
        if hasattr(self, "left_settings_tabs"):
            for index, key in enumerate(("tab_setting", "tab_stars_alignment", "tab_comet_alignment")):
                if index < self.left_settings_tabs.count():
                    self.left_settings_tabs.setTabText(index, self.tr_ui(key))
        if hasattr(self, "adjustment_tabs"):
            for index, key in enumerate(("tab_color", "tab_correction", "tab_ai_tools")):
                if index < self.adjustment_tabs.count():
                    self.adjustment_tabs.setTabText(index, self.tr_ui(key))
        if hasattr(self, "stack_btn") and getattr(self, "review_ready", False):
            self.stack_btn.setText(self.tr_ui("continue_stack"))
        if hasattr(self, "mosaic_mode_check"):
            self.mosaic_mode_check.setToolTip(self.tr_ui("mosaic_mode_tooltip"))
        if hasattr(self, "satellite_trail_check"):
            self.satellite_trail_check.setToolTip(self.tr_ui("satellite_trail_tooltip"))
        if hasattr(self, "max_comet_shift_spin"):
            self.max_comet_shift_spin.setToolTip(self.tr_ui("max_comet_tooltip"))
        if hasattr(self, "comet_refine_check"):
            self.comet_refine_check.setToolTip(self.tr_ui("comet_refine_tooltip"))
        if hasattr(self, "comet_refine_patch_spin"):
            self.comet_refine_patch_spin.setToolTip(self.tr_ui("comet_template_tooltip"))
        if hasattr(self, "comet_refine_search_spin"):
            self.comet_refine_search_spin.setToolTip(self.tr_ui("comet_search_tooltip"))
        if hasattr(self, "comet_select_btn"):
            self.comet_select_btn.setToolTip(self.tr_ui("comet_first_tooltip"))
        if hasattr(self, "comet_end_btn"):
            self.comet_end_btn.setToolTip(self.tr_ui("comet_last_tooltip"))
        if hasattr(self, "clear_comet_marks_btn"):
            self.clear_comet_marks_btn.setToolTip(self.tr_ui("comet_clear_tooltip"))
        if hasattr(self, "auto_stretch_btn"):
            self.auto_stretch_btn.setText("")
            self.auto_stretch_btn.setToolTip(self.tr_ui("auto_stretch_tooltip"))
        if hasattr(self, "crop_edges_btn"):
            self.crop_edges_btn.setToolTip(self.tr_ui("crop_tooltip"))
        if hasattr(self, "crop_angle_spin"):
            self.crop_angle_spin.setToolTip(self.tr_ui("crop_tooltip"))
        if hasattr(self, "crop_select_btn"):
            self.crop_select_btn.setToolTip(self.tr_ui("crop_select_tooltip"))
        if hasattr(self, "denoise_slider"):
            self.denoise_slider.setToolTip(self.tr_ui("astro_denoise_tooltip"))
        if hasattr(self, "star_deconvolution_slider"):
            self.star_deconvolution_slider.setToolTip(
                self.tr_ui("star_deconvolution_tooltip")
            )
        if hasattr(self, "star_deconvolution_size_slider"):
            self.star_deconvolution_size_slider.setToolTip(
                self.tr_ui("star_deconvolution_tooltip")
            )
        if hasattr(self, "select_stellar_model_btn"):
            self.select_stellar_model_btn.setText(
                self.tr_ui("select_stellar_model")
            )
            self.select_stellar_model_btn.setToolTip(
                self.tr_ui("star_deconvolution_tooltip")
            )
        if hasattr(self, "stf_strength_slider"):
            self.stf_strength_slider.setToolTip(self.tr_ui("stf_strength_tooltip"))
        if hasattr(self, "remove_gradient_btn"):
            self.remove_gradient_btn.setToolTip(self.tr_ui("gradient_tooltip"))
        if hasattr(self, "clear_gradient_btn"):
            self.clear_gradient_btn.setToolTip(self.tr_ui("gradient_tooltip"))
        if hasattr(self, "calib_title"):
            self.calib_title.setText(self.tr_ui("calibration"))
        if hasattr(self, "metadata_title"):
            self.metadata_title.setText(self.tr_ui("metadata"))
        if hasattr(self, "frame_quality_title"):
            self.update_frame_quality_title()
        if hasattr(self, "frame_quality_table"):
            self.frame_quality_table.setHorizontalHeaderLabels(self.tr_ui("frame_quality_headers"))
            self.update_frame_quality_table()

        menu_texts = {
            "cz": {
                "file_menu": "Soubor", "view_menu": "Zobrazení", "auto_stack_menu": "Live stack", "camera_menu": "Camera", "help_menu": "Nápověda",
                "open_action": "Vybrat složku…", "open_image_action": "Otevřít obrázek…",
                "recent_images_menu": "Nedávné obrázky",
                "clear_recent_images_action": "Smazat historii",
                "save_action": "Uložit výsledek jako…", "save_profile_action": "Uložit profil nastavení…",
                "load_profile_action": "Načíst profil nastavení…", "comet_action": "Označit kometu v prvním snímku…",
                "comet_end_action": "Označit kometu v posledním snímku…", "fit_action": "Přizpůsobit",
                "actual_action": "Zobrazit 1:1", "zoom_in_action": "Přiblížit",
                "zoom_out_action": "Oddálit",
                "user_guide_action": "Nápověda k programu…",
                "check_updates_action": "Zkontrolovat aktualizace…",
                "help_about_action": "About...",
                "open_log_action": "Zobrazit / smazat logy…", "quit_action": "Konec",
                "auto_stack_start_action": "Live stacking",
                "auto_stack_stop_action": "Stop Live stacking",
                "auto_stack_reject_action": "Odmítat odlehlé pixely (sigma)",
                "auto_stack_reject_elongated_action": "Odmítat protažené hvězdy",
                "camera_control_action": "ASCOM Alpaca kamery…",
            },
            "en": {
                "file_menu": "File", "view_menu": "View", "auto_stack_menu": "Live stack", "camera_menu": "Camera", "help_menu": "Help",
                "open_action": "Choose folder…", "open_image_action": "Open image…",
                "recent_images_menu": "Recent images",
                "clear_recent_images_action": "Clear history",
                "save_action": "Save result as…", "save_profile_action": "Save settings profile…",
                "load_profile_action": "Load settings profile…", "comet_action": "Mark comet in first frame…",
                "comet_end_action": "Mark comet in last frame…", "fit_action": "Fit",
                "actual_action": "Show 1:1", "zoom_in_action": "Zoom in",
                "zoom_out_action": "Zoom out",
                "user_guide_action": "User guide…",
                "check_updates_action": "Check for updates…",
                "help_about_action": "About...",
                "open_log_action": "View / delete logs…", "quit_action": "Quit",
                "auto_stack_start_action": "Live stacking",
                "auto_stack_stop_action": "Stop Live stacking",
                "auto_stack_reject_action": "Reject outliers (sigma)",
                "auto_stack_reject_elongated_action": "Reject elongated stars",
                "camera_control_action": "ASCOM Alpaca cameras…",
            },
        }
        for attr, text in menu_texts[self.language].items():
            obj = getattr(self, attr, None)
            if obj is not None:
                obj.setTitle(text) if hasattr(obj, "setTitle") else obj.setText(text)
        self.update_recent_images_menu()

        label_map = {}
        for lang_values in self.TRANSLATIONS.values():
            for key, value in lang_values.items():
                if isinstance(value, str):
                    label_map[value] = key
        extra_labels = {
            "Black point": "Black point",
            "White point": "White point",
            "Gamma": "Gamma",
            "SCNR Green": "SCNR Green",
            "1:1": "1:1",
            "+": "+",
            "−": "−",
        }
        for label in self.findChildren(QLabel):
            text = label.text()
            if text in label_map:
                label.setText(self.tr_ui(label_map[text]))
            elif text in extra_labels:
                label.setText(extra_labels[text])

        form_label_translations = {
            "Jazyk": "language", "Language": "language",
            "Motiv": "theme", "Theme": "theme",
            "Režim": "ui_mode", "Mode": "ui_mode",
            "Zarovnání": "align", "Alignment": "align",
            "Skládání": "stacking", "Stacking": "stacking",
            "Max. snímků": "max_images", "Max. frames": "max_images",
            "CPU procesy": "cpu_processes", "CPU processes": "cpu_processes",
            "Ponechat": "keep", "Keep": "keep",
            "Max. drift hvězd": "max_star_drift", "Max. star drift": "max_star_drift",
            "Max. pohyb komety": "max_comet_move", "Max. comet motion": "max_comet_move",
            "Šablona komety": "comet_template", "Comet template": "comet_template",
            "Hledání komety": "comet_search", "Comet search": "comet_search",
            "Ignorovat okraj": "ignore_edge", "Ignore border": "ignore_edge",
            "Bayer FIT": "bayer_fit",
            "Síla STF": "stf_strength", "STF strength": "stf_strength",
            "Odstranění vinětace": "vignette", "Vignette removal": "vignette",
            "Umělý flat": "synthetic_flat", "Synthetic flat": "synthetic_flat",
            "Korekce barevného pozadí": "color_background", "Color background correction": "color_background",
            "Metoda odšumění": "denoise_mode", "Denoise method": "denoise_mode",
            "Astro odšumění": "astro_denoise", "Astro Denoise": "astro_denoise",
            "AI dekonvoluce hvězd": "star_deconvolution", "AI Star Deconvolution": "star_deconvolution",
            "Velikost hvězd": "star_deconvolution_size", "Star size": "star_deconvolution_size",
            "Kontrast": "contrast", "Contrast": "contrast",
            "Saturace": "saturation", "Saturation": "saturation",
            "Červená": "red", "Red": "red",
            "Zelená": "green", "Green": "green",
            "Modrá": "blue", "Blue": "blue",
        }
        for label in self.findChildren(QLabel):
            key = form_label_translations.get(label.text())
            if key:
                label.setText(self.tr_ui(key))

        self.refresh_mode_labels()
        self.refresh_bayer_labels()
        self.update_zoom_status_label()
        self.update_stack_selection_summary()
        if hasattr(self, "preview_sequence_paths") and self.preview_sequence_paths:
            current = None
            idx = self.preview_file_combo.currentIndex()
            if 0 <= idx < len(self.preview_sequence_paths):
                current = self.preview_sequence_paths[idx]
            self.set_preview_sequence(self.preview_sequence_paths, current)

    def show_user_guide_dialog(self):
        if self.language == "en":
            title = "Astro Stacker User Guide"
            text = (
                "<h2>Astro Stacker - User Guide</h2>"
                "<p><b>Purpose:</b> Astro Stacker loads astronomical image sequences, aligns them, calibrates them, stacks them, and lets you tune the visual result for export.</p>"
                "<h3>Quick start</h3>"
                "<ol>"
                "<li>Choose a folder with Light frames. If you have Flat/Bias/Dark folders next to them, the app can use them automatically.</li>"
                "<li>Enable <b>RAW only</b> if the folder also contains JPG/PNG/BMP/TIFF preview files that should not be stacked. XISF, FIT/FITS and camera RAW files are kept.</li>"
                "<li>For a normal deep-sky stack, use <b>Star alignment + RANSAC</b> and <b>Median</b>. Enable <b>Auto reference</b>; use the quality filter only when needed.</li>"
                "<li>Click <b>Start stacking</b>. Progress and any warnings are shown in the status line and diagnostic logs.</li>"
                "<li>The completed stack is shown linearly and may initially look black. Use <b>Balance</b> to stretch the preview only.</li>"
                "<li>Then adjust black/white point, gamma, color, background neutralization, vignette removal, or synthetic flat.</li>"
                "<li>Export FITS/XISF for linear data, or PNG/JPG/TIFF for the current visual result.</li>"
                "</ol>"
                "<h3>Input and browsing</h3>"
                "<ul>"
                "<li><b>Choose folder</b> loads a sequence of light frames and also lets the app find Flat/Bias/Dark subfolders.</li>"
                "<li><b>Open image</b> opens one standalone image for inspection without changing the stack folder.</li>"
                "<li>The preview list shows Light/Flat/Bias/Dark frames. After stacking, <b>*</b> marks the reference frame and <b>x</b> marks frames rejected by the quality filter.</li>"
                "<li>The metadata panel shows camera/FITS/RAW details such as exposure, gain, temperature, filter, binning, Bayer pattern, dimensions, and file information when available.</li>"
                "<li>The left and right panels can be collapsed with the small arrow in the top-left corner. The remaining arrow rail shows the panel again and gives more room to the preview and Frame quality table.</li>"
                "</ul>"
                "<h3>Stacking workflow</h3>"
                "<ul>"
                "<li><b>Alignment</b>: Translation is fastest; ECC affine handles shift/rotation/scale; Star alignment uses detected stars and RANSAC; Comet alignment stacks on a moving comet; Star + Comet saves separate star and comet stacks.</li>"
                "<li><b>Stacking</b>: Mean is fast, Median is robust, Sigma-clipped mean removes outliers in both directions. High rejection mean rejects only unusually bright pixels such as satellite trails.</li>"
                "<li><b>Auto reference</b> chooses the sharpest/best frame. <b>Use only best frames</b> keeps the selected percentage and marks rejected frames in the preview list.</li>"
                "<li>The <b>Frame quality</b> heading shows the total number of input Lights, Darks, Flats, and Biases. Saved masters, cache files, and program outputs are not counted.</li>"
                "<li><b>FWHM px</b> in the Frame quality table is a relative stellar-size measurement in pixels of the reduced quality preview, not an arcsecond value.</li>"
                "<li><b>Satellite trail</b> detects long straight satellite or aircraft trails. Suspect frames are marked in the Frame quality table and the detected trail pixels are masked during stacking, so the unaffected part of each frame is still used.</li>"
                "<li><b>Mosaic - expand canvas</b> preserves aligned image areas outside the reference frame and creates a larger output canvas. It is useful for mosaic sequences from smart telescopes such as Vespera. With GPU enabled, mosaic integration uses VRAM tiles while correctly ignoring partially covered edges. If GPU processing is unavailable, Astro Stacker uses the parallel tiled CPU path.</li>"
                "<li><b>CPU processes</b> can run in Auto mode or Manual mode. Auto keeps the system responsive while using most CPU cores.</li>"
                "<li><b>GPU</b> enables CUDA/CuPy on NVIDIA or Apple Metal/MPS through PyTorch for the final stacking step. Aligned frames are sent to the GPU in row tiles, avoiding a second full-stack RAM copy. If GPU processing fails, the app falls back to CPU.</li>"
                "</ul>"
                "<h3>Calibration</h3>"
                "<ul>"
                "<li>The right-panel Flat/Bias/Dark buttons accept either a finished Master file or any folder with individual calibration frames. A selected folder is stacked and cached automatically.</li>"
                "<li>If manual calibration is not selected, the app tries to use Flat, Bias, and Dark subfolders automatically.</li>"
                "<li>For the simplest setup, keep Light frames directly in the selected folder and create subfolders named <b>Flat</b>, <b>Bias</b>, and <b>Dark</b> inside it. Put each calibration-frame type into its matching subfolder.</li>"
                "<li>Automatic calibration folders are stacked into cached master files named <b>MasterBias_AS.fit</b>, <b>MasterFlat_AS.fit</b>, and <b>MasterDark_AS.fit</b>. The next run reuses the masters if the folder contents and RAW only setting have not changed.</li>"
                "<li><b>RAW only</b> also applies to automatic Flat/Bias/Dark folders, so JPG/PNG/BMP/TIFF previews in calibration folders are ignored.</li>"
                "<li>Calibration frames are applied before alignment and stacking.</li>"
                "</ul>"
                "<h3>Comet workflow</h3>"
                "<ul>"
                "<li>Use <b>Mark comet in first frame</b> and <b>Mark comet in last frame</b>. The cursor changes to a crosshair for precise picking.</li>"
                "<li>The app interpolates comet motion between those points and can refine the comet locally in each frame.</li>"
                "</ul>"
                "<h3>Preview and post-processing</h3>"
                "<ul>"
                "<li>Mouse wheel zooms around the cursor. Dragging pans the image, including a subtle inertial glide.</li>"
                "<li>Double-click opens a full-screen preview with the same zoom and pan behavior. Esc closes it.</li>"
                "<li>Curves, Turn angle, manual Select crop, black/white point, gamma, highlight compression, vignette removal, synthetic flat, contrast, saturation, RGB balance, SCNR Green, AWB, background neutralization, polynomial gradient removal, and flips affect the visual preview and PNG/JPG/TIFF export.</li>"
                "<li><b>Select crop</b> lets you drag a rectangle directly in the preview. The selected area is kept and everything outside it is removed.</li>"
                "<li><b>Remove gradient</b> is intended mainly for smooth light-pollution gradients around galaxies. Use it carefully with large nebulae, where faint real structures can be mistaken for background.</li>"
                "<li>The L/R/G/B histogram includes a subtle 0-100 brightness ruler.</li>"
                "</ul>"
                "<h3>Export</h3>"
                "<ul>"
                "<li><b>FITS/XISF</b> export stays linear and does not bake in the visual stretch.</li>"
                "<li><b>PNG/JPG/TIFF</b> export uses the same visual stretch and color adjustments as the preview. External 16-bit TIFF files are loaded without reducing their tonal depth.</li>"
                "<li>Settings profiles save and restore stack and stretch controls.</li>"
                "</ul>"
            )
        else:
            title = "Nápověda Astro Stacker"
            text = (
                "<h2>Astro Stacker - nápověda</h2>"
                "<p><b>Účel:</b> Astro Stacker načítá astronomické sekvence, zarovnává je, kalibruje, skládá a umožňuje doladit vizuální výsledek pro export.</p>"
                "<h3>Rychlý návod</h3>"
                "<ol>"
                "<li>Vyber složku s Light snímky. Pokud máš vedle ní složky Flat/Bias/Dark, aplikace je umí použít automaticky.</li>"
                "<li>Zapni <b>Pouze RAW</b>, pokud jsou ve složce také JPG/PNG/BMP/TIFF náhledy, které se nemají skládat. XISF, FIT/FITS a foto RAW soubory zůstanou povolené.</li>"
                "<li>Pro běžné deep-sky skládání použij <b>Star alignment + RANSAC</b> a <b>Medián</b>. Zapni <b>Automatickou referenci</b>; filtr kvality používej podle potřeby.</li>"
                "<li>Klikni na <b>Spustit skládání</b>. Průběh a případná varování jsou ve stavovém řádku a v diagnostických lozích.</li>"
                "<li>Hotový stack se zobrazí lineárně a může zpočátku vypadat černě. Tlačítkem <b>Balance</b> roztáhni pouze náhled.</li>"
                "<li>Potom dolaď black/white point, gamma, barvy, neutralizaci pozadí, vinětaci nebo umělý flat.</li>"
                "<li>Exportuj FITS/XISF pro lineární data, nebo PNG/JPG/TIFF pro aktuální vizuální výsledek.</li>"
                "</ol>"
                "<h3>Vstup a prohlížení</h3>"
                "<ul>"
                "<li><b>Vybrat složku</b> načte sekvenci light snímků a umožní automaticky najít podsložky Flat/Bias/Dark.</li>"
                "<li><b>Otevřít obrázek</b> otevře jeden samostatný snímek pro kontrolu bez změny složky stacku.</li>"
                "<li>Seznam náhledů ukazuje Light/Flat/Bias/Dark. Po složení značí <b>*</b> referenční snímek a <b>x</b> snímek vyřazený filtrem kvality.</li>"
                "<li>Panel metadata zobrazuje dostupné informace z FITS/RAW/kamery: expozici, gain, teplotu, filtr, binning, Bayer masku, rozměry a informace o souboru.</li>"
                "<li>Levý i pravý panel lze schovat malou šipkou v levém horním rohu. Zůstane jen úzký proužek se šipkou pro návrat a náhled i tabulka Frame quality získají více místa.</li>"
                "</ul>"
                "<h3>Skládání</h3>"
                "<ul>"
                "<li><b>Zarovnání</b>: Pouze posun je nejrychlejší; ECC affine zvládá posun/rotaci/měřítko; Star alignment používá hvězdy a RANSAC; Comet alignment skládá na kometu; Star + Comet uloží zvlášť hvězdy a kometu.</li>"
                "<li><b>Skládání</b>: Průměr je rychlý, medián robustní, sigma-clipped průměr potlačuje odlehlé hodnoty v obou směrech. Průměr s odmítnutím jasných pixelů odstraňuje pouze neobvykle jasné pixely, například satelitní stopy.</li>"
                "<li><b>Automatická reference</b> vybere nejlepší snímek. <b>Použít jen nejlepší snímky</b> ponechá zvolené procento a vyřazené snímky označí v seznamu.</li>"
                "<li>Nadpis <b>Frame quality</b> ukazuje celkový počet vstupních Lights, Darks, Flats a Biases. Uložené mastery, cache ani výsledné výstupy programu se nezapočítávají.</li>"
                "<li><b>FWHM px</b> v tabulce Frame quality je relativní velikost hvězd v pixelech zmenšeného náhledu pro hodnocení kvality, nikoliv hodnota v úhlových vteřinách.</li>"
                "<li><b>Satelitní stopa</b> detekuje dlouhé rovné stopy družic nebo letadel. Podezřelé snímky označí v tabulce Frame quality a při skládání zamaskuje pouze pixely nalezené stopy, takže nepoškozená část snímku zůstane využita.</li>"
                "<li><b>Mozaika - rozšířit plátno</b> zachová zarovnané části snímků mimo referenční obraz a vytvoří větší výstupní plátno. Hodí se pro mozaikové sekvence chytrých dalekohledů, například Vespera. Mozaika používá CPU skládání, aby byly správně zpracované částečně pokryté okraje.</li>"
                "<li><b>CPU procesy</b> umí režim Auto nebo Ručně. Auto využije většinu CPU, ale nechá systém a GUI dýchat.</li>"
                "<li><b>GPU</b> zapne CUDA/CuPy na NVIDIA nebo Apple Metal/MPS přes PyTorch pro finální skládání. Zarovnané snímky se posílají do GPU po řádkových dlaždicích, takže nevzniká druhá kopie celého stacku v RAM. Při problému program bezpečně spadne zpět na CPU.</li>"
                "</ul>"
                "<h3>Kalibrace</h3>"
                "<ul>"
                "<li>Tlačítka Flat/Bias/Dark v pravém panelu umožňují vybrat hotový Master soubor nebo libovolnou složku s jednotlivými kalibračními snímky. Vybraná složka se automaticky složí a uloží do cache.</li>"
                "<li>Pokud nejsou vybrané ručně, aplikace se pokusí najít podsložky Flat, Bias a Dark automaticky.</li>"
                "<li>Pro nejjednodušší použití nech Light snímky přímo ve vybrané složce a uvnitř vytvoř podsložky <b>Flat</b>, <b>Bias</b> a <b>Dark</b>. Do každé podsložky vlož odpovídající typ kalibračních snímků.</li>"
                "<li>Automatické kalibrační složky se složí do cache master souborů <b>MasterBias_AS.fit</b>, <b>MasterFlat_AS.fit</b> a <b>MasterDark_AS.fit</b>. Další běh mastery znovu použije, pokud se nezměnil obsah složky ani nastavení Pouze RAW.</li>"
                "<li><b>Pouze RAW</b> platí i pro automatické složky Flat/Bias/Dark, takže JPG/PNG/BMP/TIFF náhledy v kalibračních složkách se ignorují.</li>"
                "<li>Kalibrace se aplikuje před zarovnáním a složením.</li>"
                "</ul>"
                "<h3>Komety</h3>"
                "<ul>"
                "<li>Použij <b>Označit kometu v prvním snímku</b> a <b>Označit kometu v posledním snímku</b>. Kurzor se přepne na křížek pro přesné označení.</li>"
                "<li>Aplikace interpoluje pohyb komety mezi body a může jádro v jednotlivých snímcích jemně dohledat.</li>"
                "</ul>"
                "<h3>Náhled a úpravy</h3>"
                "<ul>"
                "<li>Kolečko myši zoomuje od kurzoru. Tažením se posouvá obraz včetně jemné setrvačnosti.</li>"
                "<li>Dvojklik otevře celoobrazovkový náhled se stejným zoomem a posunem. Esc ho zavře.</li>"
                "<li>Křivky, Turn angle, ruční crop výběrem, black/white point, gamma, komprese jasů, odstranění vinětace, umělý flat, kontrast, saturace, RGB balance, SCNR Green, AWB, neutralizace pozadí, polynomické odstranění gradientu a otočení ovlivňují vizuální náhled a PNG/JPG/TIFF export.</li>"
                "<li><b>Výběr / Select crop</b> umožní natáhnout obdélník přímo v náhledu. Vybraná oblast zůstane a vše mimo ni se odstraní.</li>"
                "<li><b>Odstranit gradient</b> je určené hlavně pro hladké světelné gradienty okolo galaxií. Používej opatrně u rozsáhlých mlhovin, kde může za pozadí považovat reálné slabé struktury.</li>"
                "<li>Histogram L/R/G/B obsahuje jemné pravítko jasu 0-100.</li>"
                "</ul>"
                "<h3>Export</h3>"
                "<ul>"
                "<li><b>FITS/XISF</b> export zůstává lineární a neobsahuje vizuální stretch.</li>"
                "<li><b>PNG/JPG/TIFF</b> export používá stejný vizuální stretch a barevné úpravy jako náhled. Externí 16bit TIFF soubory se načítají bez snížení tonální hloubky.</li>"
                "<li>Profily nastavení ukládají a obnovují nastavení stacku a vizuálních úprav.</li>"
                "</ul>"
            )
        self.show_scrollable_text_dialog(title, text)

    def show_scrollable_text_dialog(self, title: str, html: str):
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.resize(760, 620)

        layout = QVBoxLayout(dialog)
        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.setHtml(html)
        browser.setMinimumSize(560, 420)
        layout.addWidget(browser)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(buttons)

        dialog.exec()

    def refresh_mode_labels(self):
        align_data = self.align_combo.currentData()
        align_items = {
            "cz": [
                ("Pouze posun", "translation"),
                ("Kalibrační snímky — bez zarovnání", "calibration"),
                ("Afinní ECC — posun/rotace/měřítko", "ecc_affine"),
                ("Star alignment — hvězdy + RANSAC", "star_affine"),
                ("Comet alignment — skládat na kometu", "comet"),
                ("Star + Comet — uložit zvlášť hvězdy a kometu", "comet_merge"),
            ],
            "en": [
                ("Translation only", "translation"),
                ("Calibration frames — no alignment", "calibration"),
                ("Affine ECC — shift/rotation/scale", "ecc_affine"),
                ("Star alignment — stars + RANSAC", "star_affine"),
                ("Comet alignment — stack on comet", "comet"),
                ("Star + Comet — save stars and comet separately", "comet_merge"),
            ],
        }
        self.align_combo.blockSignals(True)
        self.align_combo.clear()
        for text, data in align_items[self.language]:
            self.align_combo.addItem(text, data)
            self.align_combo.setItemData(self.align_combo.count() - 1, text, Qt.ToolTipRole)
        self.align_combo.setCurrentIndex(max(0, self.align_combo.findData(align_data)))
        self.align_combo.blockSignals(False)

        stack_data = self.stack_combo.currentData()
        stack_items = {
            "cz": [("Sigma-clipped průměr", "sigma"), ("Průměr s odmítnutím jasných pixelů", "high_rejection"), ("Průměr", "mean"), ("Medián", "median")],
            "en": [("Sigma-clipped mean", "sigma"), ("High rejection mean", "high_rejection"), ("Mean", "mean"), ("Median", "median")],
        }
        self.stack_combo.blockSignals(True)
        self.stack_combo.clear()
        for text, data in stack_items[self.language]:
            self.stack_combo.addItem(text, data)
            self.stack_combo.setItemData(self.stack_combo.count() - 1, text, Qt.ToolTipRole)
        self.stack_combo.setCurrentIndex(max(0, self.stack_combo.findData(stack_data)))
        self.stack_combo.blockSignals(False)

    def refresh_bayer_labels(self):
        if not hasattr(self, "bayer_combo"):
            return
        current = self.bayer_combo.currentData()
        items = {
            "cz": [
                ("Auto podle FIT hlavičky", "auto"),
                ("Mono / nedebayerovat", "mono"),
                ("RGGB", "RGGB"),
                ("BGGR", "BGGR"),
                ("GRBG", "GRBG"),
                ("GBRG", "GBRG"),
            ],
            "en": [
                ("Auto from FIT header", "auto"),
                ("Mono / do not debayer", "mono"),
                ("RGGB", "RGGB"),
                ("BGGR", "BGGR"),
                ("GRBG", "GRBG"),
                ("GBRG", "GBRG"),
            ],
        }
        self.bayer_combo.blockSignals(True)
        self.bayer_combo.clear()
        for text, data in items[self.language]:
            self.bayer_combo.addItem(text, data)
            self.bayer_combo.setItemData(self.bayer_combo.count() - 1, text, Qt.ToolTipRole)
        self.bayer_combo.setCurrentIndex(max(0, self.bayer_combo.findData(current)))
        self.bayer_combo.blockSignals(False)

    def show_about_dialog(self):
        title = "About Astro Stacker" if self.language == "en" else "O programu Astro Stacker"
        author_label = "Author" if self.language == "en" else "Autor"
        version_label = "Version" if self.language == "en" else "Verze"
        text = (
            "<h2>Astro Stacker</h2>"
            f"<p><b>{version_label}:</b> {APP_VERSION}</p>"
            f"<p><b>{author_label}:</b> Josef Ladra</p>"
        )
        QMessageBox.about(self, title, text)

    def check_for_updates(self, manual: bool = False):
        worker = getattr(self, "update_check_worker", None)
        if worker is not None and worker.isRunning():
            return
        if manual and hasattr(self, "check_updates_action"):
            self.check_updates_action.setEnabled(False)
        self.update_check_manual = bool(manual)
        self.update_check_worker = UpdateCheckWorker(GITHUB_RELEASES_API_URL, APP_VERSION)
        self.update_check_worker.finished_ok.connect(self.on_update_check_finished)
        self.update_check_worker.failed.connect(self.on_update_check_failed)
        self.update_check_worker.finished.connect(self.on_update_check_thread_finished)
        self.update_check_worker.start()

    def on_update_check_thread_finished(self):
        if hasattr(self, "check_updates_action"):
            self.check_updates_action.setEnabled(True)

    def on_update_check_failed(self, error: str):
        if getattr(self, "update_check_manual", False):
            QMessageBox.warning(
                self,
                self.tr_ui("update_failed_title"),
                self.tr_ui("update_failed_message").format(error=error),
            )

    def on_update_check_finished(self, info: Dict[str, Any]):
        latest = display_version(str(info.get("tag") or ""))
        url = str(info.get("url") or GITHUB_RELEASES_PAGE_URL)
        if info.get("is_newer"):
            reply = QMessageBox.information(
                self,
                self.tr_ui("update_available_title"),
                self.tr_ui("update_available_message").format(version=latest, current=APP_VERSION),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply == QMessageBox.Yes:
                QDesktopServices.openUrl(QUrl(url))
        elif getattr(self, "update_check_manual", False):
            QMessageBox.information(
                self,
                self.tr_ui("update_current_title"),
                self.tr_ui("update_current_message").format(current=APP_VERSION),
            )

    def _html_escape(self, text: str) -> str:
        return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def available_log_files(self) -> List[Tuple[str, Path]]:
        logs: List[Tuple[str, Path]] = []
        logs.append(("Diagnostic log" if self.language == "en" else "Diagnostický log", init_log_path()))

        output_dir = self.folder / "astro_stacker_output" if self.folder else None
        if output_dir is not None:
            candidates = [
                ("Output run log" if self.language == "en" else "Výstupní log běhu", output_dir / "AS_stacker_cli_run.log"),
                ("Output error log" if self.language == "en" else "Výstupní chybový log", output_dir / "AS_stacker_cli_error.log"),
            ]
            for label, path in candidates:
                logs.append((label, path))

        return logs

    def show_logs_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Logs" if self.language == "en" else "Logy")
        dialog.resize(860, 640)

        layout = QVBoxLayout(dialog)

        selector_row = QHBoxLayout()
        selector_label = QLabel("Log:" if self.language == "en" else "Log:")
        selector = QComboBox()
        selector_row.addWidget(selector_label)
        selector_row.addWidget(selector, 1)
        layout.addLayout(selector_row)

        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.setMinimumSize(620, 430)
        layout.addWidget(browser, 1)

        button_row = QHBoxLayout()
        refresh_btn = QPushButton("Refresh" if self.language == "en" else "Obnovit")
        delete_btn = QPushButton("Delete selected log" if self.language == "en" else "Smazat vybraný log")
        close_btn = QPushButton("Close" if self.language == "en" else "Zavřít")
        button_row.addWidget(refresh_btn)
        button_row.addWidget(delete_btn)
        button_row.addStretch()
        button_row.addWidget(close_btn)
        layout.addLayout(button_row)

        state: Dict[str, Any] = {"logs": []}

        def populate_selector(keep_path: Optional[Path] = None):
            state["logs"] = self.available_log_files()
            selector.blockSignals(True)
            selector.clear()
            selected_index = 0
            for idx, (label, path) in enumerate(state["logs"]):
                exists = path.exists()
                size_text = ""
                if exists:
                    try:
                        size_text = f" ({path.stat().st_size / 1024:.1f} KB)"
                    except Exception:
                        size_text = ""
                suffix = size_text if exists else (" - missing" if self.language == "en" else " - chybí")
                selector.addItem(f"{label}{suffix}", str(path))
                if keep_path is not None and path == keep_path:
                    selected_index = idx
            selector.setCurrentIndex(selected_index)
            selector.blockSignals(False)

        def selected_path() -> Optional[Path]:
            data = selector.currentData()
            return Path(data) if data else None

        def load_selected():
            path = selected_path()
            if path is None:
                return
            title = selector.currentText()
            escaped_path = self._html_escape(str(path))
            if path.exists():
                try:
                    content = path.read_text(encoding="utf-8", errors="replace")
                    if len(content) > 60000:
                        note = "Showing last 60000 characters." if self.language == "en" else "Zobrazuji posledních 60000 znaků."
                        content = note + "\n\n" + content[-60000:]
                    escaped_content = self._html_escape(content)
                except Exception as exc:
                    escaped_content = self._html_escape(f"Cannot read log: {exc}" if self.language == "en" else f"Log nelze načíst: {exc}")
            else:
                escaped_content = self._html_escape("Log file does not exist yet." if self.language == "en" else "Log zatím neexistuje.")
            browser.setHtml(
                f"<h2>{self._html_escape(title)}</h2>"
                f"<p><b>{escaped_path}</b></p>"
                "<pre style='white-space: pre-wrap; font-family: Consolas, monospace; font-size: 11px;'>"
                f"{escaped_content}</pre>"
            )

        def delete_selected():
            path = selected_path()
            if path is None:
                return
            if not path.exists():
                load_selected()
                return
            title = "Delete log?" if self.language == "en" else "Smazat log?"
            text = (
                f"Delete this log file?\n\n{path}"
                if self.language == "en"
                else f"Smazat tento log soubor?\n\n{path}"
            )
            if QMessageBox.question(self, title, text, QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
                return
            try:
                path.unlink()
            except Exception as exc:
                QMessageBox.critical(
                    self,
                    "Cannot delete log" if self.language == "en" else "Log nelze smazat",
                    str(exc),
                )
            populate_selector(path)
            load_selected()

        selector.currentIndexChanged.connect(load_selected)
        refresh_btn.clicked.connect(lambda: (populate_selector(selected_path()), load_selected()))
        delete_btn.clicked.connect(delete_selected)
        close_btn.clicked.connect(dialog.accept)

        populate_selector()
        load_selected()
        dialog.exec()

    def show_log_location(self):
        path = init_log_path()
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            content = f"Log nelze načíst: {exc}" if self.language == "cz" else f"Cannot read log: {exc}"
        title = "Diagnostický log" if self.language == "cz" else "Diagnostic log"
        escaped_path = self._html_escape(str(path))
        escaped_content = self._html_escape(content[-20000:])
        html = (
            f"<h2>{title}</h2>"
            f"<p><b>{escaped_path}</b></p>"
            f"<pre style='white-space: pre-wrap; font-family: Consolas, monospace; font-size: 11px;'>"
            f"{escaped_content}</pre>"
        )
        self.show_scrollable_text_dialog(title, html)

    def settings_profile_data(self) -> Dict[str, Any]:
        """Vrátí ručně uložitelný profil nastavení jako JSON-kompatibilní slovník."""
        return {
            "version": 1,
            "stack": {
                "align_mode": self.align_combo.currentData(),
                "stack_mode": self.stack_combo.currentData(),
                "max_images": self.max_images_spin.value(),
                "raw_only": self.fit_only_check.isChecked(),
                "sigma": self.sigma_spin.value(),
                "normalize_background": self.normalize_check.isChecked(),
                "use_multiprocessing": self.mp_check.isChecked(),
                "process_mode": self.process_mode_combo.currentData() if hasattr(self, "process_mode_combo") else "auto",
                "use_gpu": self.gpu_check.isChecked(),
                "completion_sound": self.completion_sound_check.isChecked() if hasattr(self, "completion_sound_check") else True,
                "ui_mode": self.ui_mode_combo.currentData() if hasattr(self, "ui_mode_combo") else "advanced",
                "processes": self.processes_spin.value(),
                "auto_reference": self.auto_reference_check.isChecked(),
                "manual_reference_path": self.manual_reference_path,
                "mosaic_mode": self.mosaic_mode_check.isChecked() if hasattr(self, "mosaic_mode_check") else False,
                "quality_filter": self.quality_filter_check.isChecked(),
                "review_frames": self.review_frames_check.isChecked(),
                "keep_percent": self.keep_percent_spin.value(),
                "max_star_shift": self.max_star_shift_spin.value(),
                "max_comet_shift": self.max_comet_shift_spin.value(),
                "comet_refine": self.comet_refine_check.isChecked(),
                "comet_refine_patch": self.comet_refine_patch_spin.value(),
                "comet_refine_search": self.comet_refine_search_spin.value(),
                "star_border_margin": self.star_border_margin_spin.value(),
                "strict_star_filter": self.strict_star_filter_check.isChecked(),
                "satellite_trail_filter": self.satellite_trail_check.isChecked() if hasattr(self, "satellite_trail_check") else False,
                "bayer_pattern": self.bayer_combo.currentData(),
            },
            "stretch": {
                "black": self.black_slider.value(),
                "white": self.white_slider.value(),
                "gamma": self.gamma_slider.value(),
                "stf_strength": self.stf_strength_slider.value() if hasattr(self, "stf_strength_slider") else 50,
                "vignette_removal": self.vignette_removal_slider.value() if hasattr(self, "vignette_removal_slider") else 0,
                "synthetic_flat": self.synthetic_flat_slider.value() if hasattr(self, "synthetic_flat_slider") else 0,
                "color_background_correction": self.color_background_slider.value() if hasattr(self, "color_background_slider") else 0,
                "denoise_strength": self.denoise_slider.value() if hasattr(self, "denoise_slider") else 0,
                "denoise_mode": self.denoise_mode_combo.currentData() if hasattr(self, "denoise_mode_combo") else "classic",
                "ai_denoise_model_path": getattr(self, "ai_denoise_model_path", None),
                "star_deconvolution_strength": self.star_deconvolution_slider.value() if hasattr(self, "star_deconvolution_slider") else 0,
                "star_deconvolution_size": self.star_deconvolution_size_slider.value() if hasattr(self, "star_deconvolution_size_slider") else 5,
                "star_deconvolution_model_path": getattr(self, "star_deconvolution_model_path", None),
                "contrast": self.contrast_slider.value(),
                "saturation": self.saturation_slider.value(),
                "red": self.red_slider.value(),
                "green": self.green_slider.value(),
                "blue": self.blue_slider.value(),
            },
            "comet_marks": {
                "manual_comet_xy": list(self.manual_comet_xy) if self.manual_comet_xy is not None else None,
                "manual_comet_reference_path": self.manual_comet_reference_path,
                "manual_comet_end_xy": list(self.manual_comet_end_xy) if self.manual_comet_end_xy is not None else None,
                "manual_comet_end_path": self.manual_comet_end_path,
            },
        }

    def apply_settings_profile_data(self, data: Dict[str, Any]):
        """Aplikuje profil nastavení načtený z JSON souboru."""
        stack = data.get("stack", {}) if isinstance(data, dict) else {}
        stretch = data.get("stretch", {}) if isinstance(data, dict) else {}
        comet_marks = data.get("comet_marks", {}) if isinstance(data, dict) else {}

        def set_combo_by_data(combo: QComboBox, value: Any):
            idx = combo.findData(value)
            if idx >= 0:
                combo.setCurrentIndex(idx)

        def set_spin(spin: QSpinBox, key: str, source: Dict[str, Any]):
            if key in source:
                value = int(source[key])
                value = max(spin.minimum(), min(spin.maximum(), value))
                spin.setValue(value)

        def set_slider(slider: QSlider, key: str, source: Dict[str, Any]):
            if key in source:
                value = int(source[key])
                value = max(slider.minimum(), min(slider.maximum(), value))
                slider.setValue(value)

        def set_check(check: QCheckBox, key: str, source: Dict[str, Any]):
            if key in source:
                check.setChecked(bool(source[key]))

        set_combo_by_data(self.align_combo, stack.get("align_mode"))
        set_combo_by_data(self.stack_combo, stack.get("stack_mode"))
        set_spin(self.max_images_spin, "max_images", stack)
        if "raw_only" in stack:
            set_check(self.fit_only_check, "raw_only", stack)
        else:
            set_check(self.fit_only_check, "fit_only", stack)
        set_spin(self.sigma_spin, "sigma", stack)
        set_check(self.normalize_check, "normalize_background", stack)
        set_check(self.mp_check, "use_multiprocessing", stack)
        if hasattr(self, "process_mode_combo"):
            set_combo_by_data(self.process_mode_combo, stack.get("process_mode", "auto"))
        set_check(self.gpu_check, "use_gpu", stack)
        if hasattr(self, "completion_sound_check"):
            set_check(self.completion_sound_check, "completion_sound", stack)
        if hasattr(self, "ui_mode_combo"):
            set_combo_by_data(self.ui_mode_combo, stack.get("ui_mode", "advanced"))
            self.apply_ui_mode()
        set_spin(self.processes_spin, "processes", stack)
        self.update_process_mode()
        set_check(self.auto_reference_check, "auto_reference", stack)
        self.manual_reference_path = stack.get("manual_reference_path")
        if hasattr(self, "mosaic_mode_check"):
            set_check(self.mosaic_mode_check, "mosaic_mode", stack)
        set_check(self.quality_filter_check, "quality_filter", stack)
        if hasattr(self, "review_frames_check"):
            set_check(self.review_frames_check, "review_frames", stack)
        set_spin(self.keep_percent_spin, "keep_percent", stack)
        set_spin(self.max_star_shift_spin, "max_star_shift", stack)
        set_spin(self.max_comet_shift_spin, "max_comet_shift", stack)
        set_check(self.comet_refine_check, "comet_refine", stack)
        set_spin(self.comet_refine_patch_spin, "comet_refine_patch", stack)
        set_spin(self.comet_refine_search_spin, "comet_refine_search", stack)
        set_spin(self.star_border_margin_spin, "star_border_margin", stack)
        set_check(self.strict_star_filter_check, "strict_star_filter", stack)
        if hasattr(self, "satellite_trail_check"):
            set_check(self.satellite_trail_check, "satellite_trail_filter", stack)
        set_combo_by_data(self.bayer_combo, stack.get("bayer_pattern", "auto"))

        set_slider(self.black_slider, "black", stretch)
        set_slider(self.white_slider, "white", stretch)
        set_slider(self.gamma_slider, "gamma", stretch)
        if hasattr(self, "stf_strength_slider"):
            if "stf_strength" in stretch:
                set_slider(self.stf_strength_slider, "stf_strength", stretch)
            elif "highlight_compression" in stretch:
                self.stf_strength_slider.setValue(50)
        if hasattr(self, "vignette_removal_slider"):
            set_slider(self.vignette_removal_slider, "vignette_removal", stretch)
        if hasattr(self, "synthetic_flat_slider"):
            set_slider(self.synthetic_flat_slider, "synthetic_flat", stretch)
        if hasattr(self, "color_background_slider"):
            set_slider(self.color_background_slider, "color_background_correction", stretch)
        if hasattr(self, "denoise_slider"):
            set_slider(self.denoise_slider, "denoise_strength", stretch)
        if hasattr(self, "denoise_mode_combo"):
            self.ai_denoise_model_path = stretch.get(
                "ai_denoise_model_path",
                getattr(self, "ai_denoise_model_path", None),
            )
            self.denoise_mode_combo.blockSignals(True)
            saved_denoise_mode = stretch.get("denoise_mode", "classic")
            if saved_denoise_mode not in {"classic", "drunet"}:
                saved_denoise_mode = "classic"
            set_combo_by_data(
                self.denoise_mode_combo,
                saved_denoise_mode,
            )
            self.denoise_mode_combo.blockSignals(False)
            self.preview_ai_denoise_cache = None
            self.preview_ai_denoise_cache_key = None
            self.ai_denoise_layer_cache.clear()
        if hasattr(self, "star_deconvolution_slider"):
            set_slider(
                self.star_deconvolution_slider,
                "star_deconvolution_strength",
                stretch,
            )
        if hasattr(self, "star_deconvolution_size_slider"):
            set_slider(
                self.star_deconvolution_size_slider,
                "star_deconvolution_size",
                stretch,
            )
        self.star_deconvolution_model_path = stretch.get(
            "star_deconvolution_model_path",
            getattr(self, "star_deconvolution_model_path", None),
        )
        self.star_deconvolution_layer_cache.clear()
        set_slider(self.contrast_slider, "contrast", stretch)
        set_slider(self.saturation_slider, "saturation", stretch)
        set_slider(self.red_slider, "red", stretch)
        set_slider(self.green_slider, "green", stretch)
        set_slider(self.blue_slider, "blue", stretch)

        def xy_or_none(value):
            if isinstance(value, (list, tuple)) and len(value) == 2:
                try:
                    return (float(value[0]), float(value[1]))
                except Exception:
                    return None
            return None

        self.manual_comet_xy = xy_or_none(comet_marks.get("manual_comet_xy"))
        self.manual_comet_reference_path = comet_marks.get("manual_comet_reference_path")
        self.manual_comet_end_xy = xy_or_none(comet_marks.get("manual_comet_end_xy"))
        self.manual_comet_end_path = comet_marks.get("manual_comet_end_path")
        if self.preview_sequence_paths:
            current = self.current_preview_path()
            self.set_preview_sequence(self.preview_sequence_paths, current)
        self.update_preview()

    def save_settings_profile(self):
        default_path = (
            self.folder / "astro_stacker_profile.json"
            if self.folder is not None
            else Path("astro_stacker_profile.json")
        )
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Uložit profil nastavení",
            str(default_path),
            "Astro Stacker profil (*.json)",
        )
        if not filename:
            return
        try:
            path = Path(filename)
            if path.suffix.lower() != ".json":
                path = path.with_suffix(".json")
            path.write_text(json.dumps(self.settings_profile_data(), indent=2, ensure_ascii=False), encoding="utf-8")
            self.status_label.setText(f"Profil nastavení uložen: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Chyba při ukládání profilu", str(exc))

    def load_settings_profile(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Načíst profil nastavení",
            "",
            "Astro Stacker profil (*.json)",
        )
        if not filename:
            return
        try:
            data = json.loads(Path(filename).read_text(encoding="utf-8"))
            self.apply_settings_profile_data(data)
            self.status_label.setText(f"Profil nastavení načten: {filename}")
        except Exception as exc:
            QMessageBox.critical(self, "Chyba při načítání profilu", str(exc))

    def current_stack_settings(self) -> StackSettings:
        preselected_reference = self.stack_reference_path if self.review_ready else None
        if self.review_ready and not self.auto_reference_check.isChecked() and self.manual_reference_path:
            try:
                preselected_reference = str(Path(self.manual_reference_path).resolve())
            except Exception:
                preselected_reference = str(self.manual_reference_path)
        return StackSettings(
            align_mode=self.align_combo.currentData(),
            stack_mode=self.stack_combo.currentData(),
            sigma=float(self.sigma_spin.value()),
            max_images=int(self.max_images_spin.value()),
            raw_only=self.fit_only_check.isChecked(),
            downscale_for_alignment=0.5,
            normalize_background=self.normalize_check.isChecked(),
            auto_reference=self.auto_reference_check.isChecked(),
            manual_reference_path=self.manual_reference_path,
            mosaic_mode=self.mosaic_mode_check.isChecked() if hasattr(self, "mosaic_mode_check") else False,
            quality_filter=self.quality_filter_check.isChecked(),
            keep_percent=self.keep_percent_spin.value(),
            manual_excluded_paths=tuple(sorted(self.manual_excluded_paths)),
            preselected_paths=tuple(self.review_selected_paths_for_stack()) if self.review_ready else (),
            preselected_reference_path=preselected_reference,
            max_star_shift=self.max_star_shift_spin.value(),
            star_border_margin=self.star_border_margin_spin.value(),
            strict_star_filter=self.strict_star_filter_check.isChecked(),
            satellite_trail_filter=self.satellite_trail_check.isChecked() if hasattr(self, "satellite_trail_check") else False,
            bayer_pattern=self.bayer_combo.currentData(),
            max_comet_shift=self.max_comet_shift_spin.value(),
            comet_refine=self.comet_refine_check.isChecked(),
            comet_refine_patch=self.comet_refine_patch_spin.value(),
            comet_refine_search=self.comet_refine_search_spin.value(),
            manual_comet_xy=self.manual_comet_xy,
            manual_comet_reference_path=self.manual_comet_reference_path,
            manual_comet_end_xy=self.manual_comet_end_xy,
            manual_comet_end_path=self.manual_comet_end_path,
            flat_frame_path=str(self.flat_frame_path) if self.flat_frame_path else None,
            bias_frame_path=str(self.bias_frame_path) if self.bias_frame_path else None,
            dark_frame_path=str(self.dark_frame_path) if self.dark_frame_path else None,
            source_folder=str(self.folder) if self.folder else None,
            use_gpu=self.gpu_check.isChecked(),
            language=self.language,
        )

    def current_stretch_settings(self) -> StretchSettings:
        black = self.curve_slider_raw_value(self.black_slider)
        white = max(self.curve_slider_raw_value(self.white_slider), black + 1)
        return StretchSettings(
            black=black,
            white=white,
            gamma=self.gamma_slider.value() / 100.0,
            stf_strength=(100 - self.stf_strength_slider.value()) / 10.0 if hasattr(self, "stf_strength_slider") else 5.0,
            vignette_removal=(self.vignette_removal_slider.value() / 100.0 * 0.30) if hasattr(self, "vignette_removal_slider") else 0.0,
            synthetic_flat=(self.synthetic_flat_slider.value() / 100.0) if hasattr(self, "synthetic_flat_slider") else 0.0,
            color_background_correction=(self.color_background_slider.value() / 100.0) if hasattr(self, "color_background_slider") else 0.0,
            denoise_strength=(self.denoise_slider.value() / 100.0) if hasattr(self, "denoise_slider") else 0.0,
            denoise_mode=self.denoise_mode_combo.currentData() if hasattr(self, "denoise_mode_combo") else "classic",
            ai_denoise_model_path=getattr(self, "ai_denoise_model_path", None),
            star_deconvolution_strength=(self.star_deconvolution_slider.value() / 100.0) if hasattr(self, "star_deconvolution_slider") else 0.0,
            star_deconvolution_size=float(self.star_deconvolution_size_slider.value()) if hasattr(self, "star_deconvolution_size_slider") else 5.0,
            star_deconvolution_model_path=getattr(self, "star_deconvolution_model_path", None),
            contrast=self.contrast_slider.value() / 100.0,
            saturation=self.saturation_slider.value() / 100.0,
            red=self.red_slider.value() / 100.0,
            green=self.green_slider.value() / 100.0,
            blue=self.blue_slider.value() / 100.0,
            scnr_green_strength=self.scnr_green_slider.value() if hasattr(self, "scnr_green_slider") else 0,
        )

    def release_zoomed_preview_before_processing(self):
        """Drop large zoomed preview pixmaps before CPU-heavy processing starts."""
        self.stop_preview_pan_inertia()
        self.preview_slider_zoom_state = None
        self.preview_fast_zoom_scale = 1.0
        self.preview_render_array = None
        self.preview_source_shape = None
        self.zoom_mode = "fit"
        self.zoom_factor = 1.0
        self.update_zoom_status_label()
        if hasattr(self, "image_label"):
            self.image_label.set_overlay_markers([], (1, 1))
            self.image_label.set_marking_mode(False)
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText(self.tr_ui("starting"))
            self.image_label.setMinimumSize(1, 1)
        if hasattr(self, "scroll"):
            self.scroll.setWidgetResizable(True)
            self.scroll.horizontalScrollBar().setValue(0)
            self.scroll.verticalScrollBar().setValue(0)
        QApplication.processEvents()

    def start_stack(self):
        global LAST_STACK_SELECTION
        if not self.folder:
            QMessageBox.warning(self, self.tr_ui("missing_folder_title"), self.tr_ui("missing_folder_message"))
            return

        if self.review_frames_check.isChecked() and not self.review_ready:
            self.start_frame_review()
            return

        continuing_from_review = bool(self.review_frames_check.isChecked() and self.review_ready)
        stack_settings = self.current_stack_settings()

        if self.align_combo.currentData() in {"comet", "comet_merge"} and (self.manual_comet_xy is None or self.manual_comet_end_xy is None):
            QMessageBox.information(
                self,
                self.tr_ui("comet_not_marked_title"),
                self.tr_ui("comet_not_marked_message"),
            )

        self.progress.setValue(0)
        self.status_label.setText(self.tr_ui("starting"))
        self.begin_activity(self.tr_ui("starting"))
        self.stack_start_time = time.perf_counter()
        self.last_total_processing_time = None
        self.last_estimated_used_ram_bytes = None
        self.linear_result = None
        self.original_linear_result = None
        self.preview_override = None
        self.preview_override_path = None
        self.neutralized_preview_layer = None
        self.neutralized_preview_base_source_id = None
        self.gradient_preview_layer = None
        self.gradient_preview_base_source_id = None
        self.reset_preview_display_limits()
        self.release_zoomed_preview_before_processing()
        if not continuing_from_review:
            LAST_STACK_SELECTION = {}
            self.clear_stack_selection_info()

        self.worker = StackWorker(
            self.folder,
            stack_settings,
            use_multiprocessing=self.mp_check.isChecked(),
            processes=self.selected_cpu_processes(),
        )
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)
        self.worker.cancelled.connect(self.on_cancelled)
        self.stack_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.worker.start()

    def start_frame_review(self):
        global LAST_STACK_SELECTION
        if not self.folder:
            QMessageBox.warning(self, self.tr_ui("missing_folder_title"), self.tr_ui("missing_folder_message"))
            return
        self.progress.setValue(0)
        self.status_label.setText(self.tr_ui("starting"))
        self.begin_activity(self.tr_ui("starting"))
        self.manual_excluded_paths = set()
        self.review_ready = False
        LAST_STACK_SELECTION = {}
        self.clear_stack_selection_info()
        self.analysis_worker = FrameAnalysisWorker(self.folder, self.current_stack_settings())
        self.analysis_worker.progress.connect(self.on_progress)
        self.analysis_worker.finished.connect(self.on_frame_review_finished)
        self.analysis_worker.failed.connect(self.on_frame_review_failed)
        self.stack_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.analysis_worker.start()

    def on_frame_review_finished(self, info: Dict[str, Any]):
        self.apply_stack_selection_info(info)
        self.review_ready = True
        self.progress.setValue(100)
        self.status_label.setText(self.tr_ui("review_ready"))
        self.finish_activity(self.tr_ui("review_ready"), value=100)
        self.stack_btn.setText(self.tr_ui("continue_stack"))
        self.stack_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def on_frame_review_failed(self, message: str):
        self.finish_activity(message, delay_ms=1800)
        QMessageBox.critical(self, "Chyba" if self.language == "cz" else "Error", message)
        self.status_label.setText(self.tr_ui("failed"))
        self.stack_btn.setText(self.tr_ui("start_stack"))
        self.stack_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def stop_stack(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.requestInterruption()
            self.stop_btn.setEnabled(False)
            self.status_label.setText(self.tr_ui("cancel_requested"))
            self.update_activity(self.progress.value(), self.tr_ui("cancel_requested"))

    def on_progress(self, value: int, message: str):
        self.progress.setValue(value)
        self.status_label.setText(message)
        self.update_activity(value, self.compact_activity_message(message))

    def compact_activity_message(self, message: str) -> str:
        """Keep the centered activity overlay short and readable."""
        text = str(message or "")
        for prefix in (
            "Loading/calibrating/debayering Bayer frames",
            "Načítám/kalibruji/debayeruji Bayer snímky",
        ):
            if text.startswith(prefix):
                return prefix
        for prefix in (
            "Loading/calibrating/debayering Bayer",
            "Načítám/kalibruji/debayeruji Bayer",
        ):
            if text.startswith(prefix):
                return text.split(":", 1)[0]
        for prefix in (
            "Aligning in parallel",
            "Aligned in parallel",
            "Zarovnávám paralelně",
            "Zarovnáno paralelně",
            "Aligning stars",
            "Star aligned",
            "Zarovnávám hvězdy",
            "Zarovnáno hvězdně",
            "Aligning",
            "Aligned",
            "Zarovnávám",
            "Zarovnáno",
        ):
            if text.startswith(prefix):
                return text.split(":", 1)[0]
        if len(text) > 90:
            return text[:87].rstrip() + "..."
        return text

    def play_completion_sound(self):
        try:
            if hasattr(self, "completion_sound_check") and not self.completion_sound_check.isChecked():
                return
            played = False
            if sys.platform == "darwin":
                sound_path = Path("/System/Library/Sounds/Glass.aiff")
                if sound_path.exists():
                    subprocess.Popen(
                        ["afplay", str(sound_path)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    played = True
            elif sys.platform.startswith("win"):
                try:
                    import winsound

                    winsound.MessageBeep(winsound.MB_ICONASTERISK)
                    played = True
                except Exception:
                    pass
            if not played:
                QApplication.beep()
        except Exception as exc:
            log_debug(f"Completion sound failed: {exc}")

    def on_finished(self, result: Optional[np.ndarray]):
        elapsed = None
        if hasattr(self, "stack_start_time"):
            elapsed = time.perf_counter() - float(self.stack_start_time)
            LAST_STACK_SELECTION["total_processing_time"] = elapsed
        if result is not None:
            used_count = len(LAST_STACK_SELECTION.get("used_paths", [])) or 1
            mode = self.current_stack_settings().stack_mode
            LAST_STACK_SELECTION["estimated_used_ram_bytes"] = int(
                estimate_stack_bytes(tuple(np.asarray(result).shape), used_count) * stack_temp_factor(mode)
            )
        self.apply_stack_selection_info(LAST_STACK_SELECTION)
        self.last_total_processing_time = elapsed
        self.last_estimated_used_ram_bytes = LAST_STACK_SELECTION.get("estimated_used_ram_bytes")
        self.update_stack_selection_summary()
        self.preview_override = None
        self.preview_override_path = None
        self.neutralized_preview_layer = None
        self.neutralized_preview_base_source_id = None
        self.gradient_preview_layer = None
        self.gradient_preview_base_source_id = None
        self.reset_preview_display_limits()
        self.progress.setValue(100)

        if result is None:
            # Režim Star + Comet ukládá pouze dva samostatné lineární FIT soubory a záměrně nevytváří žádný další stack.
            self.linear_result = None
            self.preview_source_shape = None
            self.status_label.setText(self.tr_ui("star_comet_done"))
            self.review_ready = False
            self.stack_btn.setText(self.tr_ui("start_stack"))
            self.stack_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.play_completion_sound()
            self.finish_activity(self.tr_ui("star_comet_done"), value=100)
            return

        self.linear_result = result
        self.original_linear_result = result
        self.preview_source_shape = result.shape[:2]
        self.preview_auto_display_stretch = False
        self.invalidate_preview_cache()
        self.status_label.setText(self.tr_ui("done_edit"))
        self.auto_stretch_initial()
        self.update_preview()
        self.review_ready = False
        self.stack_btn.setText(self.tr_ui("start_stack"))
        self.stack_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.play_completion_sound()
        self.finish_activity(self.tr_ui("done_edit"), value=100)

    def on_failed(self, message: str):
        self.finish_activity(message, delay_ms=1800)
        QMessageBox.critical(self, "Chyba", message)
        self.status_label.setText(self.tr_ui("failed"))
        self.review_ready = False
        self.stack_btn.setText(self.tr_ui("start_stack"))
        self.stack_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def on_cancelled(self):
        self.status_label.setText(self.tr_ui("cancelled"))
        self.progress.setValue(0)
        self.finish_activity(self.tr_ui("cancelled"), value=0)
        self.review_ready = False
        self.stack_btn.setText(self.tr_ui("start_stack"))
        self.stack_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def auto_stretch_initial(self):
        """Nastaví neutrální GUI stretch pro náhled.

        Samotné zviditelnění lineárních FIT/stack dat se dělá pouze v update_preview()
        pomocí make_display_preview_base(). FIT export tím není nijak ovlivněn.
        """
        if self.linear_result is None:
            return

        self.black_slider.blockSignals(True)
        self.white_slider.blockSignals(True)
        self.gamma_slider.blockSignals(True)
        self.contrast_slider.blockSignals(True)
        self.saturation_slider.blockSignals(True)
        self.black_slider.setValue(0)
        self.white_slider.setValue(65535)
        self.gamma_slider.setValue(68)
        if hasattr(self, "stf_strength_slider"):
            self.stf_strength_slider.setValue(50)
        if hasattr(self, "vignette_removal_slider"):
            self.vignette_removal_slider.setValue(0)
        if hasattr(self, "synthetic_flat_slider"):
            self.synthetic_flat_slider.setValue(0)
        if hasattr(self, "color_background_slider"):
            self.color_background_slider.setValue(0)
        if hasattr(self, "denoise_slider"):
            self.denoise_slider.setValue(0)
        if hasattr(self, "star_deconvolution_slider"):
            self.star_deconvolution_slider.setValue(0)
        if hasattr(self, "star_deconvolution_size_slider"):
            self.star_deconvolution_size_slider.setValue(5)
        if hasattr(self, "star_deconvolution_slider"):
            self.star_deconvolution_slider.setValue(0)
        if hasattr(self, "star_deconvolution_size_slider"):
            self.star_deconvolution_size_slider.setValue(5)
        self.contrast_slider.setValue(71)
        self.saturation_slider.setValue(100)
        self.black_slider.blockSignals(False)
        self.white_slider.blockSignals(False)
        self.gamma_slider.blockSignals(False)
        self.contrast_slider.blockSignals(False)
        self.saturation_slider.blockSignals(False)

    def invalidate_preview_cache(self, preserve_ai: bool = False):
        self.preview_display_cache = None
        self.preview_display_cache_source_id = None
        self.preview_display_cache_neutralized = False
        self.preview_display_cache_stretched = True
        self.preview_display_cache_edge = 0
        self.preview_display_scale = 1.0
        self.preview_fast_display_cache = None
        self.preview_fast_display_cache_key = None
        self.preview_heavy_cache = None
        self.preview_heavy_cache_key = None
        self.preview_ai_denoise_cache = None
        self.preview_ai_denoise_cache_key = None
        if not preserve_ai:
            self.ai_denoise_layer_cache.clear()
            self.star_deconvolution_layer_cache.clear()
        self.preview_render_array = None

    def transform_ai_denoise_cache(
        self,
        old_source_id: int,
        new_source_id: int,
        transform,
    ):
        """Apply the same crop/rotation to cached AI reference and residual."""
        transformed: Dict[Tuple[Any, ...], Dict[str, np.ndarray]] = {}
        for key, payload in list(self.ai_denoise_layer_cache.items()):
            if not key or key[0] != old_source_id:
                continue
            try:
                reference = np.ascontiguousarray(
                    transform(np.asarray(payload["reference"], dtype=np.float32))
                )
                residual = np.ascontiguousarray(
                    transform(np.asarray(payload["residual"], dtype=np.float32))
                )
            except Exception:
                log_debug(
                    "Could not transform cached AI denoise layer:\n"
                    + traceback.format_exc()
                )
                continue
            transformed[(new_source_id, *key[1:])] = {
                "reference": reference,
                "residual": residual,
            }

        for key in [key for key in self.ai_denoise_layer_cache if key and key[0] == old_source_id]:
            self.ai_denoise_layer_cache.pop(key, None)
        self.ai_denoise_layer_cache.update(transformed)
        self.preview_ai_denoise_cache = None
        self.preview_ai_denoise_cache_key = None

    def reset_preview_display_limits(self):
        self.preview_display_limits = None
        self.invalidate_preview_cache()

    def preview_display_base(self, source: np.ndarray, max_preview_edge: int = 2200) -> np.ndarray:
        source_h, source_w = source.shape[:2]
        source_scale = min(
            1.0,
            float(max_preview_edge) / float(max(source_h, source_w)),
        )
        target_w = max(1, int(round(source_w * source_scale)))
        target_h = max(1, int(round(source_h * source_scale)))

        def preview_sized_linear(image: np.ndarray) -> np.ndarray:
            """Match every preview layer to one source-derived pixel grid."""
            arr = np.asarray(image, dtype=np.float32)
            if arr.shape[:2] == (target_h, target_w):
                return arr
            interpolation = (
                cv2.INTER_AREA
                if arr.shape[0] >= target_h and arr.shape[1] >= target_w
                else cv2.INTER_LINEAR
            )
            return cv2.resize(
                arr,
                (target_w, target_h),
                interpolation=interpolation,
            )

        gradient_corrected = (
            getattr(self, "gradient_preview_layer", None) is not None
            and getattr(self, "gradient_preview_base_source_id", None) == id(source)
        )
        neutralized = (
            getattr(self, "neutralized_preview_layer", None) is not None
            and getattr(self, "neutralized_preview_base_source_id", None) == id(source)
        )
        corrected = gradient_corrected or neutralized
        cache_source_id = id(source)
        if (
            self.preview_display_cache is not None
            and self.preview_display_cache_source_id == cache_source_id
            and self.preview_display_cache_neutralized == corrected
            and self.preview_display_cache_stretched == self.preview_auto_display_stretch
            and self.preview_display_cache_edge == max_preview_edge
        ):
            return self.preview_display_cache

        if (
            self.preview_display_cache is not None
            and self.preview_display_cache_source_id == cache_source_id
            and self.preview_display_cache_neutralized == corrected
            and self.preview_display_cache_stretched == self.preview_auto_display_stretch
            and self.preview_display_cache_edge >= max_preview_edge
        ):
            base = self.preview_display_cache
            fast_key = (id(base), int(max_preview_edge))
            self.preview_display_scale = source_scale
            if self.preview_fast_display_cache is not None and self.preview_fast_display_cache_key == fast_key:
                return self.preview_fast_display_cache
            if base.shape[:2] != (target_h, target_w):
                base = cv2.resize(
                    base,
                    (target_w, target_h),
                    interpolation=cv2.INTER_AREA,
                )
            fast_base = np.ascontiguousarray(base.astype(np.float32))
            self.preview_fast_display_cache = fast_base
            self.preview_fast_display_cache_key = fast_key
            return fast_base
        else:
            if neutralized:
                base = preview_sized_linear(self.neutralized_preview_layer)
            elif gradient_corrected:
                base = preview_sized_linear(self.gradient_preview_layer)
                if self.preview_auto_display_stretch:
                    base = make_display_preview_base(base)
            else:
                preview_source = preview_sized_linear(source)
                if self.preview_auto_display_stretch:
                    if self.preview_display_limits is None:
                        self.preview_display_limits = display_preview_limits(preview_source)
                    base = apply_display_preview_limits(preview_source, self.preview_display_limits)
                else:
                    base = np.clip(
                        np.nan_to_num(
                            preview_source,
                            nan=0.0,
                            posinf=1.0,
                            neginf=0.0,
                        ),
                        0.0,
                        1.0,
                    )
        if base.shape[:2] != (target_h, target_w):
            base = cv2.resize(
                base,
                (target_w, target_h),
                interpolation=cv2.INTER_AREA,
            )

        self.preview_display_cache = np.ascontiguousarray(base.astype(np.float32))
        self.preview_display_cache_source_id = cache_source_id
        self.preview_display_cache_neutralized = corrected
        self.preview_display_cache_stretched = self.preview_auto_display_stretch
        self.preview_display_cache_edge = max_preview_edge
        self.preview_display_scale = source_scale
        self.preview_fast_display_cache = None
        self.preview_fast_display_cache_key = None
        return self.preview_display_cache

    def preview_heavy_corrected_base(self, display_source: np.ndarray, settings: StretchSettings, fast: bool = False) -> np.ndarray:
        """Cache slow preview-only background corrections before lightweight curves/color."""
        vignette = float(getattr(settings, "vignette_removal", 0.0))
        synthetic = float(getattr(settings, "synthetic_flat", 0.0))
        color_bg = float(getattr(settings, "color_background_correction", 0.0))
        if vignette <= 1e-6 and synthetic <= 1e-6 and color_bg <= 1e-6:
            self.preview_heavy_cache = None
            self.preview_heavy_cache_key = None
            return display_source

        key = (
            id(display_source),
            tuple(display_source.shape),
            round(vignette, 5),
            round(synthetic, 5),
            round(color_bg, 5),
            bool(fast),
        )
        if self.preview_heavy_cache is not None and self.preview_heavy_cache_key == key:
            return self.preview_heavy_cache

        out = np.asarray(display_source, dtype=np.float32)
        out = apply_vignette_removal(out, vignette)
        out = apply_synthetic_flat(out, synthetic)
        out = apply_color_background_correction(out, color_bg)
        self.preview_heavy_cache = np.ascontiguousarray(out.astype(np.float32))
        self.preview_heavy_cache_key = key
        return self.preview_heavy_cache

    def apply_cached_preview_denoise(
        self,
        stretched_preview: np.ndarray,
        settings: StretchSettings,
        layer_key: Tuple[Any, ...],
        fast: bool = False,
    ) -> np.ndarray:
        mode = str(getattr(settings, "denoise_mode", "classic") or "classic")
        strength = float(getattr(settings, "denoise_strength", 0.0))
        if strength <= 1e-6:
            return stretched_preview

        model_path = None
        if mode == "drunet":
            model_path = find_drunet_model(
                getattr(settings, "ai_denoise_model_path", None)
            )

        key = (
            *layer_key,
            mode,
            str(model_path) if model_path is not None else "",
            round(strength, 4),
        )
        cached = self.ai_denoise_layer_cache.get(key)
        if cached is not None:
            reference = np.asarray(cached["reference"], dtype=np.float32)
            residual = np.asarray(cached["residual"], dtype=np.float32)
            target_h, target_w = stretched_preview.shape[:2]
            if reference.shape[:2] != (target_h, target_w):
                reference = cv2.resize(
                    reference,
                    (target_w, target_h),
                    interpolation=cv2.INTER_AREA,
                )
                residual = cv2.resize(
                    residual,
                    (target_w, target_h),
                    interpolation=cv2.INTER_AREA,
                )

            # Gamma/STF/contrast alter the visible amplitude of the same noise.
            # Scale the cached residual by robust high-frequency energy so the
            # denoise remains equally visible without running the model again.
            reference_hp = reference - cv2.GaussianBlur(reference, (0, 0), 1.0)
            current = np.asarray(stretched_preview, dtype=np.float32)
            current_hp = current - cv2.GaussianBlur(current, (0, 0), 1.0)
            ref_energy = np.median(np.abs(reference_hp), axis=(0, 1))
            cur_energy = np.median(np.abs(current_hp), axis=(0, 1))
            energy_ratio = cur_energy / np.maximum(ref_energy, 1e-6)
            residual_scale = np.clip(
                1.0 + (energy_ratio - 1.0) * 1.45,
                0.35,
                3.0,
            ).astype(np.float32)
            result = np.clip(
                current
                - residual
                * residual_scale.reshape(1, 1, -1),
                0.0,
                1.0,
            ).astype(np.float32)
            self.preview_ai_denoise_cache = np.ascontiguousarray(result)
            self.preview_ai_denoise_cache_key = key
            return self.preview_ai_denoise_cache

        # While a slider is moving, use an existing residual if available.
        # A new source/strength is calculated once on the final render.
        if fast:
            return stretched_preview

        source = np.ascontiguousarray(
            np.clip(stretched_preview, 0.0, 1.0).astype(np.float32)
        )
        ai_activity = mode == "drunet" and model_path is not None and ort is not None
        if ai_activity and self.ai_denoise_in_progress:
            return stretched_preview
        if ai_activity:
            self.ai_denoise_in_progress = True
            is_export = "export" in layer_key
            self.begin_activity(
                self.tr_ui("ai_denoise_export" if is_export else "ai_denoise_preview"),
                indeterminate=True,
            )
            self.flush_activity_overlay()
        try:
            if mode == "drunet" and model_path is not None and ort is not None:
                result = apply_drunet_onnx(source, strength, model_path)
            else:
                result = apply_astro_denoise(source, strength)
        except Exception:
            log_debug(f"AI denoise preview failed:\n{traceback.format_exc()}")
            result = apply_astro_denoise(source, strength)
        finally:
            if ai_activity:
                self.finish_activity(
                    self.tr_ui("ai_denoise_done"),
                    value=100,
                    delay_ms=1000,
                )
                if "preview" in layer_key:
                    self.pending_ai_denoise_completion_sound = True
                self.ai_denoise_in_progress = False
        full_result = np.ascontiguousarray(result.astype(np.float32))
        self.preview_ai_denoise_cache = full_result
        self.preview_ai_denoise_cache_key = key
        self.ai_denoise_layer_cache[key] = {
            "reference": source,
            "residual": np.ascontiguousarray(
                (source - full_result).astype(np.float32)
            ),
        }
        while len(self.ai_denoise_layer_cache) > 2:
            self.ai_denoise_layer_cache.pop(next(iter(self.ai_denoise_layer_cache)))
        return self.preview_ai_denoise_cache

    def apply_cached_star_deconvolution(
        self,
        preview: np.ndarray,
        settings: StretchSettings,
        layer_key: Tuple[Any, ...],
        fast: bool = False,
    ) -> np.ndarray:
        strength = float(
            getattr(settings, "star_deconvolution_strength", 0.0)
        )
        if strength <= 1e-6:
            return preview
        model_path = find_stellar_deconvolution_model(
            getattr(settings, "star_deconvolution_model_path", None)
        )
        if model_path is None or ort is None:
            return preview

        key = (*layer_key, str(model_path))
        current = np.ascontiguousarray(
            np.clip(preview, 0.0, 1.0).astype(np.float32)
        )
        cached = self.star_deconvolution_layer_cache.get(key)
        if cached is None:
            if fast or self.star_deconvolution_in_progress:
                return preview
            self.star_deconvolution_in_progress = True
            is_export = "export" in layer_key
            self.begin_activity(
                self.tr_ui(
                    "ai_stellar_export" if is_export else "ai_stellar_preview"
                ),
                indeterminate=True,
            )
            self.flush_activity_overlay()
            try:
                model_result = run_stellar_deconvolution_onnx(
                    current,
                    model_path,
                )
                cached = {
                    "reference": current,
                    "residual": np.ascontiguousarray(
                        (model_result - current).astype(np.float32)
                    ),
                }
                self.star_deconvolution_layer_cache[key] = cached
                while len(self.star_deconvolution_layer_cache) > 2:
                    self.star_deconvolution_layer_cache.pop(
                        next(iter(self.star_deconvolution_layer_cache))
                    )
            except Exception:
                log_debug(
                    "AI stellar deconvolution failed:\n"
                    + traceback.format_exc()
                )
                return preview
            finally:
                self.star_deconvolution_in_progress = False
                self.finish_activity(
                    self.tr_ui("ai_stellar_done"),
                    value=100,
                    delay_ms=1000,
                )

        reference = np.asarray(cached["reference"], dtype=np.float32)
        residual = np.asarray(cached["residual"], dtype=np.float32)
        target_h, target_w = current.shape[:2]
        if reference.shape[:2] != (target_h, target_w):
            reference = cv2.resize(
                reference,
                (target_w, target_h),
                interpolation=cv2.INTER_AREA,
            )
            residual = cv2.resize(
                residual,
                (target_w, target_h),
                interpolation=cv2.INTER_AREA,
            )

        reference_hp = reference - cv2.GaussianBlur(reference, (0, 0), 1.0)
        current_hp = current - cv2.GaussianBlur(current, (0, 0), 1.0)
        reference_energy = float(np.median(np.abs(reference_hp)))
        current_energy = float(np.median(np.abs(current_hp)))
        residual_scale = float(
            np.clip(
                current_energy / max(reference_energy, 1e-6),
                0.4,
                2.5,
            )
        )
        star_mask = stellar_deconvolution_mask(
            current,
            getattr(settings, "star_deconvolution_size", 5.0),
        )
        blend = np.clip(strength * star_mask, 0.0, 1.0)[..., None]
        return np.clip(
            current + residual * residual_scale * blend,
            0.0,
            1.0,
        ).astype(np.float32)

    def processed_display_base(
        self,
        source: np.ndarray,
        settings: StretchSettings,
        max_preview_edge: Optional[int] = None,
        fast: bool = False,
    ) -> Tuple[np.ndarray, StretchSettings]:
        """Build and cache slow display corrections before interactive curves."""
        edge = max_preview_edge or max(
            int(source.shape[0]),
            int(source.shape[1]),
        )
        corrected_display_source = self.preview_display_base(
            source,
            max_preview_edge=edge,
        )
        post_settings = replace(
            settings,
            vignette_removal=0.0,
            synthetic_flat=0.0,
            color_background_correction=0.0,
        )
        display_source = self.preview_heavy_corrected_base(
            corrected_display_source,
            settings,
            fast=fast,
        )
        return np.ascontiguousarray(display_source.astype(np.float32)), post_settings

    def _preview_slider_pressed(self, *_args):
        if not self.preview_interactive:
            self.preview_slider_zoom_state = self.capture_preview_zoom_state()
        self.preview_position_generation += 1
        self.preview_interactive = True
        self.preview_render_pending_final = False
        self.update_zoom_status_label()

    def _preview_slider_released(self, *_args):
        self.preview_interactive = False
        self.preview_render_pending_final = True
        self.preview_render_timer.start(1)

    def schedule_preview_update(self, *_args):
        sender = self.sender()
        if (
            self._is_ai_preview_slider(sender)
            and self.preview_slider_zoom_state is None
            and not self.preview_interactive
        ):
            self.preview_slider_zoom_state = self.capture_preview_zoom_state()
            self.preview_render_pending_final = True
        if self.preview_interactive:
            self.preview_render_pending_final = True
            self.preview_render_timer.start(40)
        elif self.preview_slider_zoom_state is not None:
            self.preview_render_pending_final = True
            self.preview_render_timer.start(1)
        else:
            self.preview_render_pending_final = False
            self.preview_render_timer.start(45)

    def _run_scheduled_preview_update(self):
        if self.preview_interactive:
            self.update_preview(fast=True)
            return

        if self.preview_render_pending_final:
            self.preview_render_pending_final = False
            saved_state = self.preview_slider_zoom_state
            if saved_state:
                self.zoom_mode = str(saved_state.get("mode", "fit"))
                self.zoom_factor = float(saved_state.get("factor", self.zoom_factor))
            self.update_preview(fast=False)
            if saved_state:
                self.restore_preview_zoom_state(saved_state)
                self.preview_slider_zoom_state = None
            return

        self.update_preview(fast=False)

    def capture_preview_zoom_state(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "mode": self.zoom_mode,
            "factor": self.zoom_factor,
            "hbar": 0,
            "vbar": 0,
            "render_shape": (
                tuple(self.preview_render_array.shape[:2])
                if self.preview_render_array is not None
                else None
            ),
        }
        if not hasattr(self, "scroll"):
            return state
        hbar = self.scroll.horizontalScrollBar()
        vbar = self.scroll.verticalScrollBar()
        state["hbar"] = hbar.value()
        state["vbar"] = vbar.value()
        pixmap = self.image_label.pixmap() if hasattr(self, "image_label") else None
        if pixmap is not None and not pixmap.isNull():
            viewport = self.scroll.viewport()
            state["center_rel_x"] = (
                float(hbar.value()) + float(viewport.width()) / 2.0
            ) / max(1.0, float(pixmap.width()))
            state["center_rel_y"] = (
                float(vbar.value()) + float(viewport.height()) / 2.0
            ) / max(1.0, float(pixmap.height()))
        return state

    def restore_preview_zoom_state(self, state: Dict[str, Any]):
        self.preview_position_generation += 1
        position_generation = self.preview_position_generation
        self.zoom_mode = str(state.get("mode", self.zoom_mode))
        self.zoom_factor = float(state.get("factor", self.zoom_factor))
        self.update_zoom_status_label()
        if self.zoom_mode != "fixed" or not hasattr(self, "scroll"):
            return

        hbar = self.scroll.horizontalScrollBar()
        vbar = self.scroll.verticalScrollBar()
        target_h = int(state.get("hbar", hbar.value()))
        target_v = int(state.get("vbar", vbar.value()))
        center_rel_x = state.get("center_rel_x")
        center_rel_y = state.get("center_rel_y")

        def apply_saved_scroll():
            if position_generation != self.preview_position_generation:
                return
            pixmap = self.image_label.pixmap() if hasattr(self, "image_label") else None
            if (
                pixmap is not None
                and not pixmap.isNull()
                and center_rel_x is not None
                and center_rel_y is not None
            ):
                viewport = self.scroll.viewport()
                restored_h = int(
                    round(
                        float(center_rel_x) * float(pixmap.width())
                        - float(viewport.width()) / 2.0
                    )
                )
                restored_v = int(
                    round(
                        float(center_rel_y) * float(pixmap.height())
                        - float(viewport.height()) / 2.0
                    )
                )
                hbar.setValue(restored_h)
                vbar.setValue(restored_v)
            else:
                hbar.setValue(target_h)
                vbar.setValue(target_v)
            self.preview_pan_scroll = (float(hbar.value()), float(vbar.value()))

        apply_saved_scroll()
        QTimer.singleShot(0, apply_saved_scroll)

    def show_intro_preview(self):
        if not hasattr(self, "image_label") or self.linear_result is not None or self.preview_override is not None:
            return
        self.image_label.set_overlay_markers([], (1, 1))

        intro_path = bundled_file_path("AstroStacker_intro.png")
        if intro_path is None:
            self.showing_intro_preview = False
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText(self.tr_ui("preview_prompt"))
            self.image_label.setMinimumSize(1, 1)
            if hasattr(self, "scroll"):
                self.scroll.setWidgetResizable(True)
            return

        pixmap = QPixmap(str(intro_path))
        if pixmap.isNull():
            self.showing_intro_preview = False
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText(self.tr_ui("preview_prompt"))
            self.image_label.setMinimumSize(1, 1)
            if hasattr(self, "scroll"):
                self.scroll.setWidgetResizable(True)
            return

        if hasattr(self, "scroll"):
            self.scroll.setWidgetResizable(True)
            viewport_size = self.scroll.viewport().size()
            max_w = max(300, viewport_size.width() - 24)
            max_h = max(240, viewport_size.height() - 24)
        else:
            max_w, max_h = 1200, 800
        scaled = pixmap.scaled(max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setText("")
        self.image_label.setPixmap(scaled)
        self.image_label.setMinimumSize(1, 1)
        self.showing_intro_preview = True

    def interactive_preview_max_edge(self) -> int:
        """Choose a responsive work resolution for live slider updates."""
        if not hasattr(self, "scroll"):
            return 640
        viewport = self.scroll.viewport().size()
        visible_edge = max(int(viewport.width()), int(viewport.height()))
        return max(480, min(800, int(round(visible_edge * 0.80))))

    def final_preview_max_edge(
        self,
        source: np.ndarray,
        settings: StretchSettings,
    ) -> int:
        """Use a sharper work preview only when AI denoise needs visual accuracy."""
        if (
            getattr(settings, "denoise_mode", "classic") == "drunet"
            and float(getattr(settings, "denoise_strength", 0.0)) > 1e-6
        ):
            source_edge = max(int(source.shape[0]), int(source.shape[1]))
            return min(source_edge, 2500)
        return 2200

    def update_preview(self, fast: bool = False):
        source = self.preview_override if self.preview_override is not None else self.linear_result
        if source is None:
            self.show_intro_preview()
            return
        self.showing_intro_preview = False
        self.preview_source_shape = source.shape[:2]

        interactive_target_shape = None
        if fast and self.zoom_mode == "fixed":
            saved_shape = (
                self.preview_slider_zoom_state.get("render_shape")
                if self.preview_slider_zoom_state is not None
                else None
            )
            if saved_shape is not None:
                interactive_target_shape = tuple(saved_shape)
        self.preview_fast_zoom_scale = 1.0

        stretch_settings = self.current_stretch_settings()
        max_preview_edge = (
            self.interactive_preview_max_edge()
            if fast
            else self.final_preview_max_edge(source, stretch_settings)
        )
        display_source, stretch_settings = self.processed_display_base(
            source,
            stretch_settings,
            max_preview_edge=max_preview_edge,
            fast=fast,
        )
        preview = apply_stretch(
            display_source,
            replace(stretch_settings, denoise_strength=0.0),
        )
        preview = self.apply_cached_preview_denoise(
            preview,
            stretch_settings,
            (id(source), "preview"),
            fast=fast,
        )
        preview = self.apply_cached_star_deconvolution(
            preview,
            stretch_settings,
            (id(source), "preview"),
            fast=fast,
        )
        if not fast and hasattr(self, "histogram_label"):
            hist_w = max(180, self.histogram_label.width())
            self.histogram_label.setPixmap(make_histogram_pixmap(preview, hist_w, 120))

        # Volitelné otočení náhledu.
        if getattr(self, "flip_horizontal", False):
            preview = np.ascontiguousarray(np.fliplr(preview))
        if getattr(self, "flip_vertical", False):
            preview = np.ascontiguousarray(np.flipud(preview))
        rotation = int(getattr(self, "preview_rotation_degrees", 0)) % 360
        if rotation == 90:
            preview = np.ascontiguousarray(np.rot90(preview, k=1))
        elif rotation == 180:
            preview = np.ascontiguousarray(np.rot90(preview, k=2))
        elif rotation == 270:
            preview = np.ascontiguousarray(np.rot90(preview, k=3))

        # Keep the floating-point work image small. Qt scales the resulting
        # pixmap to the previous on-screen dimensions in fixed zoom mode.
        if interactive_target_shape is not None and preview.shape[:2] != interactive_target_shape:
            target_h, target_w = interactive_target_shape
            scale_x = float(target_w) / max(1.0, float(preview.shape[1]))
            scale_y = float(target_h) / max(1.0, float(preview.shape[0]))
            self.preview_fast_zoom_scale = min(scale_x, scale_y)

        # Značky komety se kreslí jako vektorová vrstva nad pixmapou. Při zoomu
        # tak zůstávají hladké a tenké místo zvětšování rastrových pixelů.
        overlay_markers = []
        if self.preview_source_shape is not None:
            h0, w0 = self.preview_source_shape
            preview_path = self.preview_override_path
            markers = []
            if self.manual_comet_xy is not None and (preview_path is None or preview_path == self.manual_comet_reference_path):
                markers.append((self.manual_comet_xy, (1.0, 0.15, 0.15)))
            if self.manual_comet_end_xy is not None and preview_path == self.manual_comet_end_path:
                markers.append((self.manual_comet_end_xy, (0.2, 0.8, 1.0)))
            if markers:
                preview_h, preview_w = preview.shape[:2]
                sx = float(preview_w) / max(1.0, float(w0))
                sy = float(preview_h) / max(1.0, float(h0))
                for (x0, y0), color in markers:
                    if 0 <= x0 < w0 and 0 <= y0 < h0:
                        nx = float(x0) / max(1.0, float(w0 - 1))
                        ny = float(y0) / max(1.0, float(h0 - 1))
                        if getattr(self, "flip_horizontal", False):
                            nx = 1.0 - nx
                        if getattr(self, "flip_vertical", False):
                            ny = 1.0 - ny
                        if rotation == 90:
                            nx, ny = ny, 1.0 - nx
                        elif rotation == 180:
                            nx, ny = 1.0 - nx, 1.0 - ny
                        elif rotation == 270:
                            nx, ny = 1.0 - ny, nx
                        marker_scale = min(sx, sy)
                        outer_radius = max(11, int(round(28 * marker_scale)))
                        inner_radius = max(5, int(round(12 * marker_scale)))
                        overlay_markers.append((nx, ny, color, outer_radius, inner_radius))

        self.preview_render_array = np.ascontiguousarray(preview)
        self.image_label.set_overlay_markers(overlay_markers, (preview.shape[1], preview.shape[0]))
        self.render_preview_pixmap()
        if getattr(self, "pending_ai_denoise_completion_sound", False):
            self.pending_ai_denoise_completion_sound = False
            QTimer.singleShot(0, self.play_completion_sound)

    def render_preview_pixmap(self):
        preview = getattr(self, "preview_render_array", None)
        if preview is None:
            return
        self.preview_position_generation += 1
        position_generation = self.preview_position_generation
        self.showing_intro_preview = False

        if self.zoom_mode == "fit":
            self.scroll.setWidgetResizable(True)
            max_w = max(400, self.scroll.viewport().width() - 20)
            max_h = max(300, self.scroll.viewport().height() - 20)
            pixmap = numpy_to_qpixmap(preview, max_size=(max_w, max_h))
            self.image_label.setPixmap(pixmap)
            self.image_label.setMinimumSize(1, 1)
            self.update_zoom_status_label()
        else:
            self.scroll.setWidgetResizable(False)
            hbar = self.scroll.horizontalScrollBar()
            vbar = self.scroll.verticalScrollBar()
            old_pixmap = self.image_label.pixmap()
            viewport = self.scroll.viewport()
            if old_pixmap is not None and not old_pixmap.isNull():
                center_rel_x = (float(hbar.value()) + float(viewport.width()) / 2.0) / max(1.0, float(old_pixmap.width()))
                center_rel_y = (float(vbar.value()) + float(viewport.height()) / 2.0) / max(1.0, float(old_pixmap.height()))
            else:
                center_rel_x = 0.5
                center_rel_y = 0.5

            effective_zoom = self.zoom_factor * max(1.0, float(self.preview_fast_zoom_scale))
            pixmap = numpy_to_qpixmap(preview, max_size=None, zoom=effective_zoom)
            self.image_label.setPixmap(pixmap)
            self.image_label.resize(pixmap.size())
            self.image_label.setMinimumSize(pixmap.size())
            target_h = int(round(center_rel_x * float(pixmap.width()) - float(viewport.width()) / 2.0))
            target_v = int(round(center_rel_y * float(pixmap.height()) - float(viewport.height()) / 2.0))

            def apply_scroll_position():
                if position_generation != self.preview_position_generation:
                    return
                hbar.setValue(target_h)
                vbar.setValue(target_v)
                self.preview_pan_scroll = (float(hbar.value()), float(vbar.value()))

            apply_scroll_position()
            QTimer.singleShot(0, apply_scroll_position)
            self.update_zoom_status_label()

    def zoom_status_text(self) -> str:
        return "Fit" if self.zoom_mode == "fit" else f"{int(round(self.zoom_factor * 100))} %"

    def update_zoom_status_label(self):
        if hasattr(self, "zoom_label"):
            self.zoom_label.setText(self.zoom_status_text())

    def show_zoom_overlay(self):
        if not hasattr(self, "zoom_overlay"):
            return
        self.zoom_overlay.setText(self.zoom_status_text())
        self.zoom_overlay.adjustSize()
        self.zoom_overlay.show()
        self.zoom_overlay.raise_()
        self.zoom_overlay_timer.start(1800)

    def hide_zoom_overlay(self):
        if hasattr(self, "zoom_overlay"):
            self.zoom_overlay.hide()

    def open_fullscreen_preview(self):
        pixmap = self.image_label.pixmap()
        if pixmap is None or pixmap.isNull():
            return
        self.fullscreen_preview = FullscreenPreviewWindow(pixmap, self)
        self.fullscreen_preview.showFullScreen()

    def zoom_fit(self):
        self.stop_preview_pan_inertia()
        self.zoom_mode = "fit"
        if getattr(self, "preview_render_array", None) is not None:
            self.render_preview_pixmap()
        else:
            self.update_preview()
        self.show_zoom_overlay()

    def zoom_actual_size(self):
        self.stop_preview_pan_inertia()
        self.zoom_mode = "fixed"
        self.zoom_factor = 1.0
        if getattr(self, "preview_render_array", None) is not None:
            self.render_preview_pixmap()
        else:
            self.update_preview()
        self.center_preview_scroll()
        self.show_zoom_overlay()

    def center_preview_scroll(self):
        if not hasattr(self, "scroll"):
            return
        self.preview_position_generation += 1
        position_generation = self.preview_position_generation
        hbar = self.scroll.horizontalScrollBar()
        vbar = self.scroll.verticalScrollBar()

        def apply_center():
            if position_generation != self.preview_position_generation:
                return
            hbar.setValue((hbar.maximum() + hbar.minimum()) // 2)
            vbar.setValue((vbar.maximum() + vbar.minimum()) // 2)
            self.preview_pan_scroll = (float(hbar.value()), float(vbar.value()))

        apply_center()
        QTimer.singleShot(0, apply_center)

    def zoom_in(self):
        self.stop_preview_pan_inertia()
        self.zoom_mode = "fixed"
        self.zoom_factor = min(8.0, self.zoom_factor * 1.25)
        if getattr(self, "preview_render_array", None) is not None:
            self.render_preview_pixmap()
        else:
            self.update_preview()
        self.show_zoom_overlay()

    def zoom_out(self):
        self.stop_preview_pan_inertia()
        self.zoom_mode = "fixed"
        self.zoom_factor = max(0.1, self.zoom_factor / 1.25)
        if getattr(self, "preview_render_array", None) is not None:
            self.render_preview_pixmap()
        else:
            self.update_preview()
        self.show_zoom_overlay()

    def on_preview_wheel_zoom(self, factor: float, widget_x: float, widget_y: float):
        self.stop_preview_pan_inertia()
        old_zoom_mode = self.zoom_mode
        old_zoom = self.zoom_factor if self.zoom_mode == "fixed" else 1.0
        pixmap = self.image_label.pixmap()
        if pixmap is None or pixmap.isNull():
            return

        hbar = self.scroll.horizontalScrollBar()
        vbar = self.scroll.verticalScrollBar()
        label_w = max(1.0, float(self.image_label.width()))
        label_h = max(1.0, float(self.image_label.height()))
        old_ox = max(0.0, (label_w - float(pixmap.width())) / 2.0)
        old_oy = max(0.0, (label_h - float(pixmap.height())) / 2.0)
        pix_x = max(0.0, min(float(pixmap.width()), float(widget_x) - old_ox))
        pix_y = max(0.0, min(float(pixmap.height()), float(widget_y) - old_oy))
        rel_x = pix_x / max(1.0, float(pixmap.width()))
        rel_y = pix_y / max(1.0, float(pixmap.height()))
        viewport_pos = self.image_label.mapTo(self.scroll.viewport(), QPoint(int(round(widget_x)), int(round(widget_y))))
        viewport_x = float(viewport_pos.x())
        viewport_y = float(viewport_pos.y())

        if old_zoom_mode == "fit":
            self.zoom_factor = 1.0
        self.zoom_mode = "fixed"
        self.zoom_factor = max(0.1, min(8.0, self.zoom_factor * float(factor)))
        if abs(self.zoom_factor - old_zoom) < 1e-6 and old_zoom_mode == "fixed":
            return

        if getattr(self, "preview_render_array", None) is not None:
            self.render_preview_pixmap()
        else:
            self.update_preview()
        new_pixmap = self.image_label.pixmap()
        if new_pixmap is None or new_pixmap.isNull():
            return

        new_label_w = max(1.0, float(self.image_label.width()))
        new_label_h = max(1.0, float(self.image_label.height()))
        new_ox = max(0.0, (new_label_w - float(new_pixmap.width())) / 2.0)
        new_oy = max(0.0, (new_label_h - float(new_pixmap.height())) / 2.0)
        target_x = new_ox + rel_x * float(new_pixmap.width()) - viewport_x
        target_y = new_oy + rel_y * float(new_pixmap.height()) - viewport_y
        self.preview_position_generation += 1
        position_generation = self.preview_position_generation

        def apply_anchor():
            if position_generation != self.preview_position_generation:
                return
            hbar.setValue(int(round(target_x)))
            vbar.setValue(int(round(target_y)))
            self.preview_pan_scroll = (float(hbar.value()), float(vbar.value()))

        apply_anchor()
        QTimer.singleShot(0, apply_anchor)
        self.show_zoom_overlay()

    def on_preview_drag_started(self, _x: float, _y: float):
        self.stop_preview_pan_inertia()
        self.preview_pan_velocity = (0.0, 0.0)
        self.preview_drag_start_scroll = (
            self.scroll.horizontalScrollBar().value(),
            self.scroll.verticalScrollBar().value(),
        )
        self.preview_pan_scroll = (
            float(self.scroll.horizontalScrollBar().value()),
            float(self.scroll.verticalScrollBar().value()),
        )

    def on_preview_drag_moved(self, dx: float, dy: float):
        if self.zoom_mode != "fixed":
            return
        pan_x, pan_y = getattr(self, "preview_pan_scroll", (float(self.scroll.horizontalScrollBar().value()), float(self.scroll.verticalScrollBar().value())))
        step_x = -float(dx)
        step_y = -float(dy)
        pan_x += step_x
        pan_y += step_y
        old_vx, old_vy = self.preview_pan_velocity
        self.preview_pan_velocity = (
            old_vx * 0.35 + step_x * 0.65,
            old_vy * 0.35 + step_y * 0.65,
        )
        self.preview_pan_scroll = (pan_x, pan_y)
        self.scroll.horizontalScrollBar().setValue(int(round(pan_x)))
        self.scroll.verticalScrollBar().setValue(int(round(pan_y)))

    def on_preview_drag_finished(self):
        self.preview_drag_start_scroll = (
            self.scroll.horizontalScrollBar().value(),
            self.scroll.verticalScrollBar().value(),
        )
        self.preview_pan_scroll = (
            float(self.scroll.horizontalScrollBar().value()),
            float(self.scroll.verticalScrollBar().value()),
        )
        self.start_preview_pan_inertia()

    def start_preview_pan_inertia(self):
        if self.zoom_mode != "fixed":
            return
        vx, vy = self.preview_pan_velocity
        speed = math.hypot(vx, vy)
        if speed < 1.2:
            self.preview_pan_velocity = (0.0, 0.0)
            return
        max_step = 34.0
        if speed > max_step:
            scale = max_step / speed
            self.preview_pan_velocity = (vx * scale, vy * scale)
        self.preview_pan_inertia_timer.start()

    def stop_preview_pan_inertia(self):
        if hasattr(self, "preview_pan_inertia_timer"):
            self.preview_pan_inertia_timer.stop()
        self.preview_pan_velocity = (0.0, 0.0)

    def run_preview_pan_inertia(self):
        if self.zoom_mode != "fixed":
            self.stop_preview_pan_inertia()
            return

        vx, vy = self.preview_pan_velocity
        if math.hypot(vx, vy) < 0.35:
            self.stop_preview_pan_inertia()
            return

        hbar = self.scroll.horizontalScrollBar()
        vbar = self.scroll.verticalScrollBar()
        old_h = hbar.value()
        old_v = vbar.value()
        hbar.setValue(int(round(old_h + vx)))
        vbar.setValue(int(round(old_v + vy)))
        new_h = hbar.value()
        new_v = vbar.value()
        self.preview_pan_scroll = (float(new_h), float(new_v))

        hit_x_edge = new_h == old_h and abs(vx) > 0.01
        hit_y_edge = new_v == old_v and abs(vy) > 0.01
        vx = 0.0 if hit_x_edge else vx * 0.88
        vy = 0.0 if hit_y_edge else vy * 0.88
        self.preview_pan_velocity = (vx, vy)

    def start_manual_crop_selection(self):
        source = self.preview_override if self.preview_override is not None else self.linear_result
        if source is None:
            QMessageBox.information(self, self.tr_ui("missing_image_title"), self.tr_ui("missing_image_message"))
            return
        self.awaiting_comet_click = False
        self.comet_click_mode = None
        if hasattr(self, "image_label"):
            self.image_label.set_marking_mode(False)
            self.image_label.set_crop_selection_mode(True)
        self.stop_preview_pan_inertia()
        self.status_label.setText(self.tr_ui("crop_select_status"))

    def preview_point_to_source_point(self, x: float, y: float, final_w: int, final_h: int, source_w: int, source_h: int) -> Tuple[float, float]:
        rotation = int(getattr(self, "preview_rotation_degrees", 0)) % 360
        if rotation in (90, 270):
            base_h, base_w = final_w, final_h
        else:
            base_h, base_w = final_h, final_w

        bx = float(x)
        by = float(y)
        if rotation == 90:
            bx, by = float(base_w - 1) - by, bx
        elif rotation == 180:
            bx, by = float(base_w - 1) - bx, float(base_h - 1) - by
        elif rotation == 270:
            bx, by = by, float(base_h - 1) - bx

        if getattr(self, "flip_vertical", False):
            by = float(base_h - 1) - by
        if getattr(self, "flip_horizontal", False):
            bx = float(base_w - 1) - bx

        # Derive the mapping from the preview that is actually on screen.
        # preview_display_scale can briefly describe a different fast-preview
        # resolution while sliders are being adjusted.
        if base_w > 1 and source_w > 1:
            sx = bx * float(source_w - 1) / float(base_w - 1)
        else:
            sx = 0.0
        if base_h > 1 and source_h > 1:
            sy = by * float(source_h - 1) / float(base_h - 1)
        else:
            sy = 0.0
        sx = max(0.0, min(float(source_w - 1), sx))
        sy = max(0.0, min(float(source_h - 1), sy))
        return sx, sy

    def on_preview_crop_selected(self, pixmap_x0: float, pixmap_y0: float, pixmap_x1: float, pixmap_y1: float):
        source = self.preview_override if self.preview_override is not None else self.linear_result
        pixmap = self.image_label.pixmap() if hasattr(self, "image_label") else None
        preview = getattr(self, "preview_render_array", None)
        if source is None or pixmap is None or pixmap.isNull() or preview is None:
            return

        final_h, final_w = preview.shape[:2]
        source_h, source_w = source.shape[:2]
        px_scale_x = float(final_w) / max(1.0, float(pixmap.width()))
        px_scale_y = float(final_h) / max(1.0, float(pixmap.height()))
        x0 = float(pixmap_x0) * px_scale_x
        x1 = float(pixmap_x1) * px_scale_x
        y0 = float(pixmap_y0) * px_scale_y
        y1 = float(pixmap_y1) * px_scale_y

        corners = [
            self.preview_point_to_source_point(x0, y0, final_w, final_h, source_w, source_h),
            self.preview_point_to_source_point(x1, y0, final_w, final_h, source_w, source_h),
            self.preview_point_to_source_point(x0, y1, final_w, final_h, source_w, source_h),
            self.preview_point_to_source_point(x1, y1, final_w, final_h, source_w, source_h),
        ]
        xs = [p[0] for p in corners]
        ys = [p[1] for p in corners]
        self.crop_current_image_rect(min(xs), min(ys), max(xs), max(ys), manual=True)

    def crop_current_image_rect(self, x0: float, y0: float, x1: float, y1: float, manual: bool = False):
        source = self.preview_override if self.preview_override is not None else self.linear_result
        if source is None:
            QMessageBox.information(self, self.tr_ui("missing_image_title"), self.tr_ui("missing_image_message"))
            return

        arr = np.asarray(source, dtype=np.float32)
        h, w = arr.shape[:2]
        left = int(math.floor(min(x0, x1)))
        right = int(math.ceil(max(x0, x1)))
        top = int(math.floor(min(y0, y1)))
        bottom = int(math.ceil(max(y0, y1)))
        left = max(0, min(w - 1, left))
        right = max(left + 1, min(w, right))
        top = max(0, min(h - 1, top))
        bottom = max(top + 1, min(h, bottom))

        if right - left < 16 or bottom - top < 16:
            QMessageBox.warning(self, self.tr_ui("missing_image_title"), self.tr_ui("crop_too_small"))
            return

        self.push_preview_undo_state(copy_pixels=True)
        old_source_id = id(source)
        cropped = np.ascontiguousarray(arr[top:bottom, left:right, ...].astype(np.float32))
        cropped_neutralized = None
        if (
            getattr(self, "neutralized_preview_layer", None) is not None
            and getattr(self, "neutralized_preview_base_source_id", None) == old_source_id
        ):
            layer = np.asarray(self.neutralized_preview_layer, dtype=np.float32)
            if layer.shape[:2] == (h, w):
                cropped_neutralized = np.ascontiguousarray(layer[top:bottom, left:right, ...].astype(np.float32))
        cropped_gradient = None
        if (
            getattr(self, "gradient_preview_layer", None) is not None
            and getattr(self, "gradient_preview_base_source_id", None) == old_source_id
        ):
            layer = np.asarray(self.gradient_preview_layer, dtype=np.float32)
            if layer.shape[:2] == (h, w):
                cropped_gradient = np.ascontiguousarray(layer[top:bottom, left:right, ...].astype(np.float32))

        if self.preview_override is not None:
            self.preview_override = cropped
            new_source = self.preview_override
        else:
            self.linear_result = cropped
            new_source = self.linear_result
        self.preview_source_shape = cropped.shape[:2]
        self.neutralized_preview_layer = cropped_neutralized
        self.neutralized_preview_base_source_id = id(new_source) if cropped_neutralized is not None else None
        self.gradient_preview_layer = cropped_gradient
        self.gradient_preview_base_source_id = id(new_source) if cropped_gradient is not None else None
        def crop_cached_layer(layer: np.ndarray) -> np.ndarray:
            layer_h, layer_w = layer.shape[:2]
            cache_left = int(round(left * layer_w / max(1, w)))
            cache_right = int(round(right * layer_w / max(1, w)))
            cache_top = int(round(top * layer_h / max(1, h)))
            cache_bottom = int(round(bottom * layer_h / max(1, h)))
            cache_left = max(0, min(layer_w - 1, cache_left))
            cache_right = max(cache_left + 1, min(layer_w, cache_right))
            cache_top = max(0, min(layer_h - 1, cache_top))
            cache_bottom = max(cache_top + 1, min(layer_h, cache_bottom))
            return layer[cache_top:cache_bottom, cache_left:cache_right, ...]

        self.transform_ai_denoise_cache(old_source_id, id(new_source), crop_cached_layer)
        self.invalidate_preview_cache(preserve_ai=True)
        if manual:
            self.status_label.setText(self.tr_ui("crop_selection_applied").format(w=cropped.shape[1], h=cropped.shape[0]))
        else:
            self.status_label.setText(self.tr_ui("crop_selection_applied").format(w=cropped.shape[1], h=cropped.shape[0]))
        self.update_preview()

    def rotate_current_image_by_crop_angle(self):
        source = self.preview_override if self.preview_override is not None else self.linear_result
        if source is None:
            QMessageBox.information(self, self.tr_ui("missing_image_title"), self.tr_ui("missing_image_message"))
            return

        arr = np.asarray(source, dtype=np.float32)
        angle = float(self.crop_angle_spin.value()) if hasattr(self, "crop_angle_spin") else 0.0
        if abs(angle) < 1e-6:
            return

        self.push_preview_undo_state(copy_pixels=True)
        old_source_id = id(source)
        rotated = rotate_image_expand(arr, angle)
        rotated_neutralized = None
        if (
            getattr(self, "neutralized_preview_layer", None) is not None
            and getattr(self, "neutralized_preview_base_source_id", None) == old_source_id
        ):
            layer = np.asarray(self.neutralized_preview_layer, dtype=np.float32)
            if layer.shape[:2] == arr.shape[:2]:
                rotated_neutralized = rotate_image_expand(layer, angle)
        rotated_gradient = None
        if (
            getattr(self, "gradient_preview_layer", None) is not None
            and getattr(self, "gradient_preview_base_source_id", None) == old_source_id
        ):
            layer = np.asarray(self.gradient_preview_layer, dtype=np.float32)
            if layer.shape[:2] == arr.shape[:2]:
                rotated_gradient = rotate_image_expand(layer, angle)

        if self.preview_override is not None:
            self.preview_override = rotated
            new_source = self.preview_override
        else:
            self.linear_result = rotated
            new_source = self.linear_result
        self.preview_source_shape = rotated.shape[:2]
        self.neutralized_preview_layer = rotated_neutralized
        self.neutralized_preview_base_source_id = id(new_source) if rotated_neutralized is not None else None
        self.gradient_preview_layer = rotated_gradient
        self.gradient_preview_base_source_id = id(new_source) if rotated_gradient is not None else None
        self.transform_ai_denoise_cache(
            old_source_id,
            id(new_source),
            lambda layer: rotate_image_expand(layer, angle),
        )
        self.invalidate_preview_cache(preserve_ai=True)
        self.status_label.setText(self.tr_ui("crop_applied").format(angle=angle, w=rotated.shape[1], h=rotated.shape[0]))
        self.update_preview()


    def remove_background_gradient(self):
        """Create a preview/export-only background-gradient corrected layer."""
        source = self.preview_override if self.preview_override is not None else self.linear_result
        if source is None:
            QMessageBox.information(self, self.tr_ui("missing_image_title"), self.tr_ui("missing_image_message"))
            return
        img = np.asarray(source, dtype=np.float32)
        if img.ndim != 3 or img.shape[2] < 3:
            QMessageBox.information(self, self.tr_ui("neutralize_not_possible_title"), self.tr_ui("neutralize_rgb_required"))
            return
        self.push_preview_undo_state(copy_pixels=False)
        # Keep the corrected layer linear, exactly like the standalone utility.
        # Display stretching belongs exclusively to preview_display_base();
        # otherwise Balance would stretch the same data twice and clip colors.
        corrected = apply_polynomial_gradient_removal(img)
        self.gradient_preview_layer = np.ascontiguousarray(corrected.astype(np.float32))
        self.gradient_preview_base_source_id = id(source)
        # Any previous neutralized layer and black point were calculated from
        # the pre-correction tonal scale. Reusing them can clip an entire RGB
        # channel after a strong gradient correction.
        # Keep the current display stretch when a DRUNet layer already exists.
        # The gradient is applied afterwards as a cached correction, so it does
        # not need to invalidate the expensive AI inference.
        if not self.ai_denoise_layer_cache:
            self.preview_auto_display_stretch = True
        neutralized = self.build_neutralized_preview_layer(corrected, show_errors=False)
        if neutralized is not None:
            display, _status = neutralized
            self.neutralized_preview_layer = display
            self.neutralized_preview_base_source_id = id(source)
        else:
            self.neutralized_preview_layer = None
            self.neutralized_preview_base_source_id = None
            display = make_display_preview_base(corrected)
        self._set_auto_stretch_sliders(display)
        self.invalidate_preview_cache(preserve_ai=True)
        self.status_label.setText(self.tr_ui("gradient_removed"))
        self.update_preview()

    def clear_background_gradient(self):
        """Clear the preview/export-only background-gradient correction."""
        if self.gradient_preview_layer is not None:
            self.push_preview_undo_state(copy_pixels=False)
        self.gradient_preview_layer = None
        self.gradient_preview_base_source_id = None
        self.neutralized_preview_layer = None
        self.neutralized_preview_base_source_id = None
        self.invalidate_preview_cache(preserve_ai=True)
        self.status_label.setText(self.tr_ui("gradient_cleared"))
        self.update_preview()


    def neutralize_background(self):
        """Astro neutralizace pozadí přímou korekcí obrazu.

        Na rozdíl od původní verze už jen nenastavuje RGB slidery.
        Vytvoří dočasnou neutralizovanou vrstvu obrazu, která se použije pro
        náhled a PNG/JPG/TIFF export. Původní lineární FIT data v self.linear_result
        zůstávají beze změny.
        """
        source = self.preview_override if self.preview_override is not None else self.linear_result
        if source is None:
            QMessageBox.information(self, self.tr_ui("missing_image_title"), self.tr_ui("missing_image_message"))
            return

        result = self.build_neutralized_preview_layer(source, show_errors=True)
        if result is None:
            return
        self.push_preview_undo_state(copy_pixels=False)
        corrected, status = result

        # Ulož jako dočasnou display vrstvu. update_preview ji použije místo
        # make_display_preview_base(source). Lineární data se nemění.
        self.neutralized_preview_layer = corrected
        self.neutralized_preview_base_source_id = id(source)
        self.invalidate_preview_cache(preserve_ai=True)

        self.status_label.setText(status)
        self.update_preview()
        QApplication.processEvents()

    def build_neutralized_preview_layer(self, source: np.ndarray, show_errors: bool = False) -> Optional[Tuple[np.ndarray, str]]:
        img = np.asarray(source, dtype=np.float32)
        if img.ndim != 3 or img.shape[2] != 3:
            if show_errors:
                QMessageBox.information(self, self.tr_ui("neutralize_not_possible_title"), self.tr_ui("neutralize_rgb_required"))
            return None

        # Pozadí stačí analyzovat na menší kopii. Výsledné multiplikátory se
        # níže stále aplikují na plné rozlišení pro náhled i PNG/JPG/TIFF export.
        # Tím zůstane barevný výsledek stejný, ale Balance výrazně zrychlí.
        analysis_img = img
        source_h, source_w = img.shape[:2]
        max_analysis_edge = 1400
        analysis_scale = min(1.0, float(max_analysis_edge) / float(max(source_h, source_w)))
        if analysis_scale < 0.999:
            analysis_img = cv2.resize(
                img,
                (
                    max(1, int(round(source_w * analysis_scale))),
                    max(1, int(round(source_h * analysis_scale))),
                ),
                interpolation=cv2.INTER_AREA,
            ).astype(np.float32)

        # Find the background independently of channel balance. A severely weak
        # blue channel is still valid image data and must not force the mask
        # onto brighter galaxy structures. Geometry and luminance reject black
        # warp borders; brightness percentiles and the high-pass mask below
        # reject stars and extended objects.
        h, w = analysis_img.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        finite_rgb = np.all(np.isfinite(analysis_img[..., :3]), axis=2)
        raw_lum = 0.2126 * analysis_img[..., 0] + 0.7152 * analysis_img[..., 1] + 0.0722 * analysis_img[..., 2]
        positive_lum = raw_lum[finite_rgb & np.isfinite(raw_lum) & (raw_lum > 0)]
        if positive_lum.size < 100:
            if show_errors:
                QMessageBox.warning(self, self.tr_ui("not_enough_background_title"), self.tr_ui("not_enough_background_message"))
            return None

        signal_floor = max(1e-8, float(np.percentile(positive_lum, 0.5)) * 0.25)
        signal_pixel = finite_rgb & np.isfinite(raw_lum) & (raw_lum > signal_floor)
        row_good = np.mean(signal_pixel, axis=1) > 0.50
        col_good = np.mean(signal_pixel, axis=0) > 0.50
        if np.count_nonzero(row_good) >= 10 and np.count_nonzero(col_good) >= 10:
            y0, y1 = np.where(row_good)[0][[0, -1]]
            x0, x1 = np.where(col_good)[0][[0, -1]]
            shrink = max(2, int(min(h, w) * 0.01))
            y0 = min(h - 1, max(0, int(y0) + shrink))
            y1 = max(0, min(h - 1, int(y1) - shrink))
            x0 = min(w - 1, max(0, int(x0) + shrink))
            x1 = max(0, min(w - 1, int(x1) - shrink))
            valid_area = (xx >= x0) & (yy >= y0) & (xx <= x1) & (yy <= y1)
        else:
            margin_x = int(w * 0.15)
            margin_y = int(h * 0.15)
            valid_area = (xx >= margin_x) & (yy >= margin_y) & (xx < w - margin_x) & (yy < h - margin_y)

        valid_area &= signal_pixel
        display_for_detection = make_display_preview_base(analysis_img)

        valid_lum = raw_lum[valid_area]
        valid_lum = valid_lum[np.isfinite(valid_lum)]
        if valid_lum.size < 100:
            valid_area = finite_rgb
            valid_lum = raw_lum[valid_area]
            valid_lum = valid_lum[np.isfinite(valid_lum)]
        if valid_lum.size < 100:
            if show_errors:
                QMessageBox.warning(self, self.tr_ui("not_enough_background_title"), self.tr_ui("not_enough_background_message"))
            return None

        # Tmavé, ale ne úplně černé pozadí. Horní hranice 45 % typicky vynechá
        # galaxii/mlhovinu a jasnější hvězdy.
        lo = np.percentile(valid_lum, 8)
        hi = np.percentile(valid_lum, 45)
        mask = (raw_lum >= lo) & (raw_lum <= hi)

        mask &= valid_area

        # Vyřazení hvězd a drobných struktur.
        gray = to_gray_float(display_for_detection)
        blurred = cv2.GaussianBlur(gray, (0, 0), 2.5)
        highpass = gray - blurred
        star_thr = np.percentile(highpass[valid_area], 97.0) if np.count_nonzero(valid_area) > 100 else np.percentile(highpass, 97.0)
        mask &= highpass < star_thr

        if np.count_nonzero(mask) < 100:
            # Jemnější fallback: pořád jen platná RGB oblast, ale bez omezení
            # úzkého jasového pásma.
            lo2 = np.percentile(valid_lum, 5)
            hi2 = np.percentile(valid_lum, 60)
            mask = (raw_lum >= lo2) & (raw_lum <= hi2) & valid_area & (highpass < star_thr)

        if np.count_nonzero(mask) < 100:
            if show_errors:
                QMessageBox.warning(self, self.tr_ui("not_enough_background_title"), self.tr_ui("not_enough_background_message"))
            return None

        # Korekci počítáme z lineárních dat, ne ze stretchovaného náhledu.
        r_med = float(np.median(analysis_img[..., 0][mask]))
        g_med = float(np.median(analysis_img[..., 1][mask]))
        b_med = float(np.median(analysis_img[..., 2][mask]))

        eps = 1e-8
        if max(r_med, g_med, b_med) <= eps:
            if show_errors:
                QMessageBox.warning(self, self.tr_ui("background_too_dark_title"), self.tr_ui("background_too_dark_message"))
            return None

        old_lum = 0.2126 * r_med + 0.7152 * g_med + 0.0722 * b_med
        if not np.isfinite(old_lum) or old_lum <= eps:
            if show_errors:
                QMessageBox.warning(self, self.tr_ui("background_too_dark_title"), self.tr_ui("background_too_dark_message"))
            return None

        # Neutralization moves the channel backgrounds by offsets. Multiplying
        # a very weak channel would also amplify its noise.
        offsets = np.array(
            [r_med - old_lum, g_med - old_lum, b_med - old_lum],
            dtype=np.float32,
        )
        if not np.all(np.isfinite(offsets)):
            if show_errors:
                QMessageBox.warning(self, self.tr_ui("not_enough_background_title"), self.tr_ui("not_enough_background_message"))
            return None

        corrected_linear = img.copy()
        corrected_linear[..., :3] -= offsets[None, None, :]
        corrected = make_display_preview_base(np.clip(corrected_linear, 0, 1).astype(np.float32))

        applied_offsets = -offsets
        status = self.tr_ui("neutralization_applied").format(
            r=float(applied_offsets[0]),
            g=float(applied_offsets[1]),
            b=float(applied_offsets[2]),
            r_med=r_med,
            g_med=g_med,
            b_med=b_med,
        )
        return corrected, status

    def clear_background_neutralization(self):
        """Zruší dočasnou neutralizaci pozadí."""
        if self.neutralized_preview_layer is not None:
            self.push_preview_undo_state(copy_pixels=False)
        self.neutralized_preview_layer = None
        self.neutralized_preview_base_source_id = None
        self.invalidate_preview_cache(preserve_ai=True)
        self.status_label.setText(self.tr_ui("neutralization_cleared"))
        self.update_preview()



    def toggle_flip_horizontal(self):
        self.push_preview_undo_state(copy_pixels=False)
        self.flip_horizontal = not bool(getattr(self, "flip_horizontal", False))
        self.update_preview()

    def toggle_flip_vertical(self):
        self.push_preview_undo_state(copy_pixels=False)
        self.flip_vertical = not bool(getattr(self, "flip_vertical", False))
        self.update_preview()

    def rotate_preview_left(self):
        self.push_preview_undo_state(copy_pixels=False)
        self.preview_rotation_degrees = (int(getattr(self, "preview_rotation_degrees", 0)) + 90) % 360
        self.update_preview()

    def rotate_preview_right(self):
        self.push_preview_undo_state(copy_pixels=False)
        self.preview_rotation_degrees = (int(getattr(self, "preview_rotation_degrees", 0)) - 90) % 360
        self.update_preview()

    def auto_white_balance(self):
        """Automatické vyvážení bílé podle robustního odhadu neutrálního pozadí.

        Princip:
        - použije aktuální zdroj obrazu: otevřený obrázek/FIT, nebo lineární složený obraz,
        - vybere střední jasové pixely, aby ignoroval černé okraje, hot pixely a jasné hvězdy,
        - spočítá medián R/G/B a nastaví multiplikátory tak, aby mediány kanálů byly stejné.
        """
        source = self.preview_override if self.preview_override is not None else self.linear_result
        if source is None:
            QMessageBox.information(self, "AWB", self.tr_ui("missing_image_message"))
            return

        img = np.clip(source.astype(np.float32), 0, 1)
        if img.ndim != 3 or img.shape[2] != 3:
            QMessageBox.warning(self, "AWB", self.tr_ui("awb_rgb_required"))
            return

        # Hrubý luminanční obraz bez aktuálního RGB vyvážení.
        lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]

        # Vynecháme úplně tmavé okraje a jasné hvězdy/jádra. Percentily fungují dobře pro EAA i DSO data.
        lo = np.percentile(lum, 10)
        hi = np.percentile(lum, 80)
        mask = (lum > lo) & (lum < hi)

        # Fallback, kdyby byl obraz hodně specifický nebo maska příliš malá.
        if np.count_nonzero(mask) < 1000:
            lo = np.percentile(lum, 5)
            hi = np.percentile(lum, 95)
            mask = (lum > lo) & (lum < hi)

        if np.count_nonzero(mask) < 100:
            QMessageBox.warning(self, "AWB", self.tr_ui("awb_not_enough_neutral"))
            return

        self.push_preview_undo_state(copy_pixels=False)
        med = np.median(img[mask], axis=0).astype(np.float32)
        med = np.maximum(med, 1e-6)
        target = float(np.mean(med))
        gains = target / med

        # Zachovej celkový jas: normalizace vůči zelenému kanálu je pro astro intuitivní.
        if gains[1] > 1e-6:
            gains = gains / gains[1]

        # Omez extrémy, aby jeden špatný odhad neposlal slidery mimo rozumný rozsah.
        gains = np.clip(gains, 0.25, 3.0)

        self.red_slider.blockSignals(True)
        self.green_slider.blockSignals(True)
        self.blue_slider.blockSignals(True)
        self.red_slider.setValue(int(round(gains[0] * 100)))
        self.green_slider.setValue(int(round(gains[1] * 100)))
        self.blue_slider.setValue(int(round(gains[2] * 100)))
        self.red_slider.blockSignals(False)
        self.green_slider.blockSignals(False)
        self.blue_slider.blockSignals(False)

        # RGB gains are applied after the cached denoise residual. Keep the
        # expensive model result and only rebuild the lightweight preview.
        self.invalidate_preview_cache(preserve_ai=True)
        source_label = self.tr_ui("awb_source_open") if self.preview_override is not None else self.tr_ui("awb_source_stack")
        self.status_label.setText(self.tr_ui("awb_status").format(source=source_label, r=gains[0], g=gains[1], b=gains[2]))
        self.update_preview()

    def reset_stretch(self, push_undo: bool = True):
        if push_undo:
            self.push_preview_undo_state(copy_pixels=False)
        self.preview_auto_display_stretch = False
        self.reset_preview_display_limits()
        self.black_slider.setValue(0)
        self.white_slider.setValue(65535)
        self.gamma_slider.setValue(68)
        if hasattr(self, "stf_strength_slider"):
            self.stf_strength_slider.setValue(50)
        if hasattr(self, "vignette_removal_slider"):
            self.vignette_removal_slider.setValue(0)
        if hasattr(self, "synthetic_flat_slider"):
            self.synthetic_flat_slider.setValue(0)
        if hasattr(self, "color_background_slider"):
            self.color_background_slider.setValue(0)
        if hasattr(self, "denoise_slider"):
            self.denoise_slider.setValue(0)
        self.contrast_slider.setValue(71)
        self.saturation_slider.setValue(100)
        self.red_slider.setValue(100)
        self.green_slider.setValue(100)
        self.blue_slider.setValue(100)
        self.update_preview()

    def _set_auto_stretch_sliders(self, display: np.ndarray) -> bool:
        """Calculate preview black point and gamma for an already prepared display base."""
        # The linked STF has already placed the background at the target
        # brightness while preserving the true white point. Keep the ordinary
        # sliders neutral to avoid applying a second automatic stretch.
        if np.asarray(display).size < 32:
            return False
        for slider in (
            self.black_slider,
            self.white_slider,
            self.gamma_slider,
            self.contrast_slider,
            self.saturation_slider,
        ):
            slider.blockSignals(True)
        self.black_slider.setValue(0)
        self.white_slider.setValue(65535)
        self.gamma_slider.setValue(68)
        self.contrast_slider.setValue(71)
        self.saturation_slider.setValue(100)
        for slider in (
            self.black_slider,
            self.white_slider,
            self.gamma_slider,
            self.contrast_slider,
            self.saturation_slider,
        ):
            slider.blockSignals(False)
        return True

    def apply_balance_to_current_preview(
        self,
        record_undo: bool = True,
        refresh_preview: bool = True,
        set_status: bool = True,
        update_sliders: bool = True,
    ) -> bool:
        source = self.preview_override if self.preview_override is not None else self.linear_result
        if source is None:
            return False

        if record_undo:
            self.push_preview_undo_state(copy_pixels=False)
        self.preview_auto_display_stretch = True
        balance_source = source
        if (
            self.gradient_preview_layer is not None
            and self.gradient_preview_base_source_id == id(source)
        ):
            balance_source = self.gradient_preview_layer
        neutralized = self.build_neutralized_preview_layer(balance_source, show_errors=False)
        if neutralized is not None:
            display, _status = neutralized
            self.neutralized_preview_layer = display
            self.neutralized_preview_base_source_id = id(source)
            self.invalidate_preview_cache(preserve_ai=True)
        else:
            display = make_display_preview_base(source)

        if update_sliders and not self._set_auto_stretch_sliders(display):
            return False

        if set_status:
            self.status_label.setText(self.tr_ui("auto_stretch_done"))
        if refresh_preview:
            self.update_preview()
        return True

    def auto_stretch_preview(self):
        source = self.preview_override if self.preview_override is not None else self.linear_result
        if source is None:
            QMessageBox.information(
                self,
                self.tr_ui("missing_image_title"),
                self.tr_ui("missing_image_message"),
            )
            return
        self.apply_balance_to_current_preview()

    def export_display_base(self) -> np.ndarray:
        """Return the full-resolution visual base exactly as used by the preview."""
        if (
            getattr(self, "neutralized_preview_layer", None) is not None
            and getattr(self, "neutralized_preview_base_source_id", None) == id(self.linear_result)
        ):
            return self.neutralized_preview_layer
        if (
            getattr(self, "gradient_preview_layer", None) is not None
            and getattr(self, "gradient_preview_base_source_id", None) == id(self.linear_result)
        ):
            if self.preview_auto_display_stretch:
                return make_display_preview_base(self.gradient_preview_layer)
            return self.gradient_preview_layer
        if self.preview_auto_display_stretch:
            return make_display_preview_base(self.linear_result)
        return np.clip(
            np.nan_to_num(
                np.asarray(self.linear_result, dtype=np.float32),
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ),
            0.0,
            1.0,
        )

    def save_preview(self):
        source = self.preview_override if self.preview_override is not None else self.linear_result
        if source is None:
            QMessageBox.information(
                self,
                self.tr_ui("missing_image_title"),
                self.tr_ui("missing_image_message"),
            )
            return

        opened_path = (
            Path(self.preview_override_path)
            if self.preview_override is not None and self.preview_override_path
            else None
        )
        if opened_path is not None:
            default_path = opened_path.with_name(f"{opened_path.stem}_edited.tif")
        elif self.folder is not None:
            default_path = self.folder / "stacked_result.tif"
        else:
            default_path = Path("stacked_result.tif")

        filename, selected_filter = QFileDialog.getSaveFileName(
            self,
            self.tr_ui("save_image_title"),
            str(default_path),
            "TIFF 16-bit (*.tif *.tiff);;PNG 8-bit (*.png);;JPEG 8-bit (*.jpg *.jpeg);;FITS 32-bit linear (*.fits *.fit);;XISF 32-bit linear (*.xisf)",
        )
        if not filename:
            return

        path = Path(filename)
        suffix = path.suffix.lower()

        if suffix not in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".fits", ".fit", ".xisf"}:
            if "PNG" in selected_filter:
                path = path.with_suffix(".png")
            elif "JPEG" in selected_filter or "JPG" in selected_filter:
                path = path.with_suffix(".jpg")
            elif "XISF" in selected_filter:
                path = path.with_suffix(".xisf")
            elif "FITS" in selected_filter:
                path = path.with_suffix(".fits")
            else:
                path = path.with_suffix(".tif")
            filename = str(path)
            suffix = path.suffix.lower()

        try:
            if suffix in {".fits", ".fit"}:
                # FITS ukládáme vždy lineárně, bez apply_stretch(), black/white pointu, gammy a RGB sliderů.
                settings = self.current_stack_settings()
                if opened_path is not None and opened_path.suffix.lower() in {".fit", ".fits"}:
                    source_header = get_first_fits_header_safely(opened_path)
                else:
                    source_header = find_reference_fits_header(self.folder, settings)
                stack_info = {
                    "align_mode": settings.align_mode,
                    "stack_mode": settings.stack_mode,
                    "sigma": float(settings.sigma),
                    "bayer_pattern": getattr(settings, "bayer_pattern", "auto"),
                }
                save_stack_fits(Path(filename), source, source_header=source_header, stack_info=stack_info)
            elif suffix in XISF_EXTENSIONS:
                settings = self.current_stack_settings()
                stack_info = {
                    "align_mode": settings.align_mode,
                    "stack_mode": settings.stack_mode,
                    "sigma": float(settings.sigma),
                    "num_images": len(LAST_STACK_SELECTION.get("used_paths", [])),
                    "bayer_pattern": getattr(settings, "bayer_pattern", "auto"),
                }
                save_stack_xisf(Path(filename), source, stack_info=stack_info)
            elif suffix in {".tif", ".tiff"}:
                # TIFF ukládáme jako vizuální 16bit náhled:
                # nejdřív stejný display stretch jako v GUI, potom uživatelské křivky/slidery.
                # Lineární master zůstává ve FIT exportu.
                stretch_settings = self.current_stretch_settings()
                display_base, stretch_settings = self.processed_display_base(
                    source,
                    stretch_settings,
                )
                preview = apply_stretch(
                    display_base,
                    replace(stretch_settings, denoise_strength=0.0),
                )
                preview = self.apply_cached_preview_denoise(
                    preview,
                    stretch_settings,
                    (
                        id(source),
                        "export",
                        bool(self.preview_auto_display_stretch),
                        tuple(self.preview_display_limits or ()),
                    ),
                )
                preview = self.apply_cached_star_deconvolution(
                    preview,
                    stretch_settings,
                    (
                        id(source),
                        "export",
                        bool(self.preview_auto_display_stretch),
                        tuple(self.preview_display_limits or ()),
                    ),
                )
                img16_rgb = visual_float_to_uint16(preview)
                img16_bgr = cv2.cvtColor(img16_rgb, cv2.COLOR_RGB2BGR)
                ok = cv2.imwrite(filename, img16_bgr)
                if not ok:
                    raise RuntimeError(self.tr_ui("save_tiff_error"))
            elif suffix == ".png":
                # PNG ukládáme jako vizuální 8bit náhled:
                # používá stejný display stretch jako zobrazení v programu.
                stretch_settings = self.current_stretch_settings()
                display_base, stretch_settings = self.processed_display_base(
                    source,
                    stretch_settings,
                )
                preview = apply_stretch(
                    display_base,
                    replace(stretch_settings, denoise_strength=0.0),
                )
                preview = self.apply_cached_preview_denoise(
                    preview,
                    stretch_settings,
                    (
                        id(source),
                        "export",
                        bool(self.preview_auto_display_stretch),
                        tuple(self.preview_display_limits or ()),
                    ),
                )
                preview = self.apply_cached_star_deconvolution(
                    preview,
                    stretch_settings,
                    (
                        id(source),
                        "export",
                        bool(self.preview_auto_display_stretch),
                        tuple(self.preview_display_limits or ()),
                    ),
                )
                Image.fromarray(visual_float_to_uint8(preview)).save(filename)
            else:
                # JPEG ukládáme jako vizuální 8bit náhled se stejnými úpravami jako PNG.
                stretch_settings = self.current_stretch_settings()
                display_base, stretch_settings = self.processed_display_base(
                    source,
                    stretch_settings,
                )
                preview = apply_stretch(
                    display_base,
                    replace(stretch_settings, denoise_strength=0.0),
                )
                preview = self.apply_cached_preview_denoise(
                    preview,
                    stretch_settings,
                    (
                        id(source),
                        "export",
                        bool(self.preview_auto_display_stretch),
                        tuple(self.preview_display_limits or ()),
                    ),
                )
                preview = self.apply_cached_star_deconvolution(
                    preview,
                    stretch_settings,
                    (
                        id(source),
                        "export",
                        bool(self.preview_auto_display_stretch),
                        tuple(self.preview_display_limits or ()),
                    ),
                )
                Image.fromarray(
                    visual_float_to_uint8(preview)
                ).save(filename, quality=95, subsampling=0, optimize=True)

            self.status_label.setText(
                self.tr_ui("save_image_done").format(path=filename)
            )
        except Exception as exc:
            QMessageBox.critical(
                self,
                self.tr_ui("save_image_error_title"),
                str(exc),
            )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_activity_overlay()
        self.update_preview()

    def closeEvent(self, event):
        if (
            self.alpaca_camera_dialog is not None
            and self.alpaca_camera_dialog.capture_workers
        ):
            camera_workers = list(
                self.alpaca_camera_dialog.capture_workers.values()
            )
            for camera_worker in camera_workers:
                camera_worker.requestInterruption()
            for camera_worker in camera_workers:
                camera_worker.wait(5000)
        self.auto_stack_timer.stop()
        self.auto_stack_active = False
        if self.auto_stack_calibration_worker is not None and self.auto_stack_calibration_worker.isRunning():
            self.auto_stack_calibration_worker.requestInterruption()
            self.auto_stack_calibration_worker.wait(3000)
        if self.auto_stack_worker is not None and self.auto_stack_worker.isRunning():
            self.auto_stack_worker.requestInterruption()
            self.auto_stack_worker.wait(3000)
        super().closeEvent(event)


def create_startup_splash(app: QApplication) -> Optional[QSplashScreen]:
    intro_path = bundled_file_path("AstroStacker_intro.png")
    content = QPixmap(str(intro_path)) if intro_path is not None else QPixmap()
    if content.isNull():
        content = QPixmap(520, 320)
        content.fill(QColor("#10151d"))
    else:
        content = content.scaled(560, 360, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    border = 10
    pixmap = QPixmap(content.width() + border * 2, content.height() + border * 2)
    pixmap.fill(QColor("#05070b"))

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.fillRect(0, 0, pixmap.width(), pixmap.height(), QColor("#05070b"))
    painter.setPen(QPen(QColor("#8fb8ff"), 2))
    painter.drawRoundedRect(1, 1, pixmap.width() - 2, pixmap.height() - 2, 8, 8)
    painter.setPen(QPen(QColor("#1f2937"), 1))
    painter.drawRoundedRect(border - 1, border - 1, content.width() + 1, content.height() + 1, 4, 4)
    painter.drawPixmap(border, border, content)
    overlay_h = 68
    overlay_y = border
    painter.fillRect(border, overlay_y, content.width(), overlay_h, QColor(0, 0, 0, 170))

    title_font = painter.font()
    title_font.setPointSize(21)
    title_font.setBold(True)
    painter.setFont(title_font)
    painter.setPen(QColor("#ffffff"))
    painter.drawText(
        QRectF(border + 18, overlay_y + 8, content.width() - 36, 28),
        Qt.AlignLeft | Qt.AlignVCenter,
        "Astro Stacker",
    )

    detail_font = painter.font()
    detail_font.setPointSize(12)
    detail_font.setBold(False)
    painter.setFont(detail_font)
    painter.setPen(QColor("#d8e6ff"))
    painter.drawText(
        QRectF(border + 20, overlay_y + 38, content.width() - 40, 22),
        Qt.AlignLeft | Qt.AlignVCenter,
        f"Version {APP_DISPLAY_VERSION}  -  Loading...",
    )
    painter.end()

    splash = QSplashScreen(pixmap, Qt.WindowStaysOnTopHint)
    splash.setWindowFlag(Qt.FramelessWindowHint, True)
    splash.show()
    screen = app.primaryScreen()
    if screen is not None:
        screen_geo = screen.availableGeometry()
        splash.move(screen_geo.center() - splash.rect().center())
    app.processEvents()
    return splash


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Astro Stacker")
    if hasattr(app, "setApplicationDisplayName"):
        app.setApplicationDisplayName("Astro Stacker")
    app.setOrganizationName("Josef Ladra")
    apply_dark_theme(app)
    splash = create_startup_splash(app)
    win = AstroStackerWindow()
    if splash is not None:
        splash.showMessage("Opening main window...", Qt.AlignBottom | Qt.AlignHCenter, QColor("#ffffff"))
        app.processEvents()
    win.showMaximized()
    if splash is not None:
        QTimer.singleShot(3000, lambda: splash.finish(win))
    sys.exit(app.exec())


if __name__ == "__main__":
    # Nutné pro PyInstaller / macOS / Windows při použití multiprocessing.
    # Bez toho se při spuštění worker procesů může aplikace spouštět rekurzivně
    # v dalších instancích.
    mp.freeze_support()
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
