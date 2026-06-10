from pathlib import Path
import sys

from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "Ikon AS16.png"
OUT = Path(__file__).resolve().parent / "icons"


def make_windows_icon() -> None:
    image = Image.open(SOURCE).convert("RGBA")
    image.save(
        OUT / "AstroStacker.ico",
        format="ICO",
        sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )


def make_macos_icon() -> None:
    if sys.platform != "darwin":
        return
    image = Image.open(SOURCE).convert("RGBA")
    if image.size != (1024, 1024):
        image.thumbnail((900, 900), Image.Resampling.LANCZOS)
        canvas = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
        canvas.alpha_composite(
            image,
            ((1024 - image.width) // 2, (1024 - image.height) // 2),
        )
        image = canvas
    image.save(OUT / "AstroStacker.icns", format="ICNS")


def main() -> None:
    if not SOURCE.exists():
        raise SystemExit(f"Missing icon source: {SOURCE}")
    OUT.mkdir(parents=True, exist_ok=True)
    make_windows_icon()
    make_macos_icon()
    print(f"Icons prepared in: {OUT}")


if __name__ == "__main__":
    main()
