from __future__ import annotations

import argparse
import multiprocessing as mp
import dataclasses
import os
import sys
import traceback
import unicodedata
from pathlib import Path

CLI_VERSION = "2.7"


def ascii_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.encode("ascii", errors="replace").decode("ascii").replace("?", "-")


def english_progress_message(message: str) -> str:
    replacements = [
        ("Kvalita z cache", "Quality from cache"),
        ("Hodnotím kvalitu paralelně", "Scoring quality in parallel"),
        ("Hodnotim kvalitu paralelne", "Scoring quality in parallel"),
        ("Hodnotím kvalitu", "Scoring quality"),
        ("Hodnotim kvalitu", "Scoring quality"),
        ("Vybírám referenční snímek", "Selecting reference frame"),
        ("Zarovnávám paralelně", "Aligning in parallel"),
        ("Zarovnavam paralelne", "Aligning in parallel"),
        ("Dokoncuji paralelni alignment", "Finishing parallel alignment"),
        ("Zarovnáno z cache", "Aligned from cache"),
        ("Zarovnano z cache", "Aligned from cache"),
        ("Zarovnávám hvězdy", "Aligning stars"),
        ("Zarovnávám", "Aligning"),
        ("Zarovnavam", "Aligning"),
        ("CPU procesy omezeny kvuli RAM", "CPU processes limited by RAM"),
        ("CPU alignment procesy", "CPU alignment processes"),
        ("Skládám snímky na GPU", "Stacking frames on GPU"),
        ("Skladam snimky na GPU", "Stacking frames on GPU"),
        ("Posilam stack po blocich primo do GPU", "Sending stack tiles directly to GPU"),
        ("Skladam na GPU po blocich VRAM", "Stacking on GPU in VRAM tiles"),
        ("Skladam na GPU po blocich sdilene pameti", "Stacking on GPU in shared-memory tiles"),
        ("Mozaika: posilam stack po blocich primo do GPU", "Mosaic: sending stack tiles directly to GPU"),
        ("Mozaika: skládám paralelně na CPU po blocích RAM", "Mosaic: stacking in parallel on CPU in RAM tiles"),
        ("Mozaika: skladam paralelne na CPU po blocich RAM", "Mosaic: stacking in parallel on CPU in RAM tiles"),
        ("Mozaika: převádím snímky na plátno", "Mosaic: warping frames to canvas"),
        ("Mozaika: prevadim snimky na platno", "Mosaic: warping frames to canvas"),
        ("Skladam prumer na CPU po blocich RAM", "Stacking mean on CPU in RAM tiles"),
        ("Skladam prumer prubezne", "Stacking mean incrementally"),
        ("Skladam na CPU po blocich RAM", "Stacking on CPU in RAM tiles"),
        ("Skladam Bias master na CPU po blocich RAM", "Stacking Bias master on CPU in RAM tiles"),
        ("Skladam Flat master na CPU po blocich RAM", "Stacking Flat master on CPU in RAM tiles"),
        ("Skladam Dark master na CPU po blocich RAM", "Stacking Dark master on CPU in RAM tiles"),
        ("Skládám ručně vybranou složku", "Stacking manually selected folder"),
        ("Nacitam MasterBias z cache", "Loading MasterBias from cache"),
        ("Nacitam MasterFlat z cache", "Loading MasterFlat from cache"),
        ("Nacitam MasterDark z cache", "Loading MasterDark from cache"),
        ("Pripravuji stack v RAM", "Preparing stack in RAM"),
        ("RAM ochrana", "RAM protection"),
        ("nedostatek pameti pro cely stack", "not enough memory for the full stack"),
        ("skladam po castech", "stacking in tiles"),
        ("Skládám snímky", "Stacking frames"),
        ("Skladam snimky", "Stacking frames"),
        ("Skládám hvězdy", "Stacking stars"),
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
        ("GPU vypocet selhal", "GPU computation failed"),
        ("pokracuji na CPU", "continuing on CPU"),
        ("skladam na CPU", "stacking on CPU"),
        ("GPU neni dostupne", "GPU is not available"),
        ("CuPy nelze nacist", "CuPy cannot be loaded"),
        ("PyTorch/MPS nelze nacist", "PyTorch/MPS cannot be loaded"),
        ("Vyřazuji bez platného star alignmentu", "Rejecting frame without valid star alignment"),
        ("Vyřazuji černý snímek bez hvězd", "Rejecting black frame without stars"),
        ("vlaken", "threads"),
        ("radku", "rows"),
        ("Hotovo", "Done"),
        ("Žádný snímek neprošel zarovnáním. Zkontroluj, zda složka Light neobsahuje Dark/Bias snímky nebo zda jsou ve snímcích detekovatelné hvězdy.", "No frame passed alignment. Check whether the Light folder contains Dark/Bias frames or whether detectable stars are present."),
        ("Star + Comet výstupy vyžadují označení komety v prvním i posledním snímku.", "Star + Comet outputs require marking the comet in both the first and last frame."),
        ("Ve složce nejsou žádné FIT/FITS ani RAW snímky. Vypni volbu Pouze RAW, pokud chceš skládat i PNG/JPG/TIFF/BMP.", "The folder contains no FIT/FITS or RAW frames. Disable RAW only if you also want to stack PNG/JPG/TIFF/BMP files."),
        ("Ve složce nejsou žádné podporované obrázky. Podporované formáty zahrnují FIT/FITS, CR2/CR3/RAW, TIFF, PNG, JPG a BMP.", "The folder contains no supported images. Supported formats include FIT/FITS, CR2/CR3/RAW, TIFF, PNG, JPG and BMP."),
        ("Ve složce nezbyly žádné light snímky. Zkontroluj, zda nejsou soubory označené jako Dark/Bias/Flat.", "No light frames remain in the folder. Check whether files are marked as Dark/Bias/Flat."),
        ("Pro FITS podporu nainstaluj", "Install for FITS support"),
        ("RAW podpora vyžaduje rawpy. Nainstaluj", "RAW support requires rawpy. Install"),
        ("Prázdný obrazový soubor.", "Empty image file."),
    ]
    for src, dst in replacements:
        message = message.replace(src, dst)
    return message


def progress(value: int, message: str) -> None:
    message = english_progress_message(message)
    print(f"[{int(value):3d}%] {ascii_text(message)}", flush=True)


def error_log_path(args=None) -> Path:
    if args is not None:
        try:
            output_dir = args.output_dir or (args.input / "astro_stacker_output")
            output_dir.mkdir(parents=True, exist_ok=True)
            return output_dir / "AS_stacker_cli_error.log"
        except Exception:
            pass
    return Path(__file__).resolve().with_name("AS_stacker_cli_error.log")


def run_log_path(args=None) -> Path:
    if args is not None:
        output_dir = args.output_dir or (args.input / "astro_stacker_output")
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / "AS_stacker_cli_run.log"
    return Path(__file__).resolve().with_name("AS_stacker_cli_run.log")


def write_error_log(path: Path, text: str) -> None:
    try:
        path.write_text(text, encoding="utf-8")
    except Exception:
        pass


def build_settings(args: argparse.Namespace, stack_settings_cls):
    kwargs = {
        "align_mode": args.align,
        "stack_mode": args.stack,
        "sigma": args.sigma,
        "max_images": args.max_images,
        "raw_only": args.raw_only,
        "downscale_for_alignment": 0.5,
        "normalize_background": not args.no_normalize_background,
        "auto_reference": not args.no_auto_reference,
        "quality_filter": bool(args.quality_filter) and not args.no_quality_filter,
        "keep_percent": args.keep_percent,
        "max_star_shift": args.max_star_shift,
        "star_border_margin": args.star_border_margin,
        "strict_star_filter": not args.no_strict_star_filter,
        "satellite_trail_filter": args.satellite_trail,
        "bayer_pattern": args.bayer,
        "flat_frame_path": str(args.flat) if args.flat else None,
        "bias_frame_path": str(args.bias) if args.bias else None,
        "dark_frame_path": str(args.dark) if args.dark else None,
        "source_folder": str(args.input),
        "use_gpu": args.gpu,
        "use_aligned_cache": args.aligned_cache,
        "mosaic_mode": args.mosaic,
        "language": "en",
    }
    if dataclasses.is_dataclass(stack_settings_cls):
        allowed = {field.name for field in dataclasses.fields(stack_settings_cls)}
        kwargs = {key: value for key, value in kwargs.items() if key in allowed}
    return stack_settings_cls(**kwargs)


def auto_processes() -> int:
    cpu_count = max(1, os.cpu_count() or 1)
    if cpu_count <= 2:
        return 1
    return max(1, min(cpu_count - 1, int(round(cpu_count * 0.75))))


def output_path_for(args: argparse.Namespace) -> Path:
    output_dir = args.output_dir or (args.input / "astro_stacker_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    name = args.output_name
    if not name.lower().endswith((".fit", ".fits")):
        name += ".fit"
    if not name.startswith("AS_"):
        name = "AS_" + name
    return output_dir / name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Astro Stacker command line engine for PixInsight wrappers.")
    parser.add_argument("--version", action="version", version=f"Astro Stacker CLI {CLI_VERSION}")
    parser.add_argument("input", type=Path, help="Folder with light frames.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output folder. Default: input/astro_stacker_output.")
    parser.add_argument("--output-name", default="AS_stack.fit", help="Output FIT/FITS name. AS_ prefix is added automatically.")
    parser.add_argument("--align", choices=["translation", "ecc_affine", "star_affine", "calibration"], default="star_affine")
    parser.add_argument("--stack", choices=["mean", "median", "sigma", "high_rejection"], default="median")
    parser.add_argument("--sigma", type=float, default=2.5)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--raw-only", action="store_true", help="Use only FIT/FITS and camera RAW files; ignore JPG/PNG/BMP/TIFF previews.")
    parser.add_argument("--keep-percent", type=int, default=80)
    parser.add_argument("--max-star-shift", type=int, default=180)
    parser.add_argument("--star-border-margin", type=int, default=120)
    parser.add_argument("--bayer", default="auto", choices=["auto", "mono", "RGGB", "BGGR", "GRBG", "GBRG"])
    parser.add_argument("--flat", type=Path, default=None, help="Master Flat file or folder with individual Flat frames.")
    parser.add_argument("--bias", type=Path, default=None, help="Master Bias file or folder with individual Bias frames.")
    parser.add_argument("--dark", type=Path, default=None, help="Master Dark file or folder with individual Dark frames.")
    parser.add_argument("--processes", type=int, default=0, help="CPU processes. 0 = auto.")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--aligned-cache", action="store_true", help="Enable aligned-frame cache for repeated runs.")
    parser.add_argument("--mosaic", action="store_true", help="Expand the output canvas to include all aligned frames.")
    parser.add_argument("--no-normalize-background", action="store_true")
    parser.add_argument("--no-auto-reference", action="store_true")
    parser.add_argument("--quality-filter", action="store_true", help="Use only the best frames. Default: off.")
    parser.add_argument("--no-quality-filter", action="store_true", help="Compatibility option; quality filter is already off by default.")
    parser.add_argument("--no-strict-star-filter", action="store_true")
    parser.add_argument("--satellite-trail", action="store_true", help="Detect satellite trails and mask their pixels during stacking.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.exists() or not args.input.is_dir():
        raise SystemExit(f"Input folder does not exist: {args.input}")

    from astro_stacker_app import (
        StackSettings,
        find_reference_fits_header,
        save_stack_fits,
        stack_folder,
        stack_folder_multiprocessing,
    )
    import astro_stacker_app as engine

    settings = build_settings(args, StackSettings)
    processes = args.processes if args.processes and args.processes > 1 else auto_processes()
    progress(0, f"Settings: align={settings.align_mode}, stack={settings.stack_mode}, sigma={settings.sigma}, "
                f"auto_ref={settings.auto_reference}, quality_filter={settings.quality_filter}, "
                f"raw_only={getattr(settings, 'raw_only', False)}, keep={settings.keep_percent}, max_star_shift={settings.max_star_shift}, "
                f"border={settings.star_border_margin}, strict={settings.strict_star_filter}, "
                f"satellite_trail={getattr(settings, 'satellite_trail_filter', False)}, "
                f"mosaic={getattr(settings, 'mosaic_mode', False)}, "
                f"bayer={settings.bayer_pattern}, normalize={settings.normalize_background}, "
                f"processes={processes}")
    if processes > 1:
        result = stack_folder_multiprocessing(args.input, settings, processes, progress)
    else:
        result = stack_folder(args.input, settings, progress)

    if result is None:
        raise SystemExit("No single output was produced by this mode.")

    output_path = output_path_for(args)
    source_header = find_reference_fits_header(args.input, settings)
    stack_info = {
        "align_mode": settings.align_mode,
        "stack_mode": settings.stack_mode,
        "sigma": float(settings.sigma),
        "bayer_pattern": getattr(settings, "bayer_pattern", "auto"),
    }
    save_stack_fits(output_path, result, source_header=source_header, stack_info=stack_info)
    selection = dict(getattr(engine, "LAST_STACK_SELECTION", {}) or {})
    get_alignment_stats = getattr(engine, "get_alignment_stats", None)
    stats = dict(get_alignment_stats() if callable(get_alignment_stats) else {})
    run_log = [
        f"Astro Stacker CLI {CLI_VERSION} run",
        f"Python: {sys.executable}",
        f"Input: {args.input}",
        f"Output: {output_path}",
        f"align={settings.align_mode}",
        f"stack={settings.stack_mode}",
        f"sigma={settings.sigma}",
        f"auto_reference={settings.auto_reference}",
        f"quality_filter={settings.quality_filter}",
        f"raw_only={getattr(settings, 'raw_only', False)}",
        f"keep_percent={settings.keep_percent}",
        f"max_star_shift={settings.max_star_shift}",
        f"star_border_margin={settings.star_border_margin}",
        f"strict_star_filter={settings.strict_star_filter}",
        f"satellite_trail_filter={getattr(settings, 'satellite_trail_filter', False)}",
        f"bayer={settings.bayer_pattern}",
        f"normalize_background={settings.normalize_background}",
        f"processes={processes}",
        "",
        f"reference_path={selection.get('reference_path', '')}",
        f"used_count={len(selection.get('used_paths', []))}",
        f"excluded_count={len(selection.get('excluded_paths', []))}",
    ]
    if stats:
        run_log.extend(["", "alignment_stats:"])
        for key in sorted(stats):
            run_log.append(f"  {key}: {stats[key]}")
    try:
        run_log_path(args).write_text("\n".join(run_log) + "\n", encoding="utf-8")
    except Exception:
        pass
    progress(100, f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    parsed_args = None
    try:
        parsed_args = parse_args()
        # Reuse parsed arguments without parsing twice.
        original_parse_args = parse_args
        parse_args = lambda: parsed_args  # type: ignore[assignment]
        raise SystemExit(main())
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        if code:
            log_path = error_log_path(parsed_args)
            message = str(exc)
            if not message or message == str(code):
                message = "Astro Stacker CLI exited with an error."
            text = (
                "Astro Stacker CLI error\n"
                f"Python: {sys.executable}\n"
                f"Exit code: {code}\n\n"
                f"{message}\n"
            )
            write_error_log(log_path, text)
            print(text, file=sys.stderr, flush=True)
            print(f"Error log: {log_path}", file=sys.stderr, flush=True)
        raise
    except Exception:
        log_path = error_log_path(parsed_args)
        text = (
            "Astro Stacker CLI unhandled exception\n"
            f"Python: {sys.executable}\n\n"
            f"{traceback.format_exc()}"
        )
        write_error_log(log_path, text)
        print(text, file=sys.stderr, flush=True)
        print(f"Error log: {log_path}", file=sys.stderr, flush=True)
        raise SystemExit(1)
