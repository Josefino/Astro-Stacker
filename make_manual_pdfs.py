from __future__ import annotations

import html
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def inline_md(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    return escaped


def markdown_to_html(markdown: str, title: str) -> str:
    lines = markdown.splitlines()
    out: list[str] = []
    list_stack: list[str] = []

    def close_lists(target_indent: int = -1) -> None:
        nonlocal list_stack
        while list_stack and (target_indent < 0 or len(list_stack) > target_indent):
            out.append(f"</{list_stack.pop()}>")

    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            close_lists()
            continue

        heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading:
            close_lists()
            level = len(heading.group(1))
            out.append(f"<h{level}>{inline_md(heading.group(2))}</h{level}>")
            continue

        ordered = re.match(r"^(\d+)\.\s+(.*)$", stripped)
        bullet = re.match(r"^[-*]\s+(.*)$", stripped)
        if ordered or bullet:
            tag = "ol" if ordered else "ul"
            text = ordered.group(2) if ordered else bullet.group(1)
            if not list_stack or list_stack[-1] != tag:
                close_lists()
                out.append(f"<{tag}>")
                list_stack.append(tag)
            out.append(f"<li>{inline_md(text)}</li>")
            continue

        close_lists()
        if stripped.endswith("  "):
            stripped = stripped[:-2]
        out.append(f"<p>{inline_md(stripped)}</p>")

    close_lists()
    body = "\n".join(out)
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
@page {{ margin: 18mm 16mm; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial, sans-serif;
  font-size: 10.5pt;
  line-height: 1.42;
  color: #1f2933;
}}
h1 {{
  font-size: 24pt;
  margin: 0 0 16pt;
  padding-bottom: 8pt;
  border-bottom: 1px solid #cbd5e1;
}}
h2 {{
  font-size: 16pt;
  margin: 20pt 0 8pt;
  color: #0f172a;
}}
h3 {{
  font-size: 12.5pt;
  margin: 14pt 0 5pt;
  color: #1e3a5f;
}}
p {{ margin: 0 0 7pt; }}
ul, ol {{ margin: 0 0 8pt 20pt; padding: 0; }}
li {{ margin: 2pt 0; }}
code {{
  font-family: Menlo, Consolas, monospace;
  font-size: 9.5pt;
  background: #eef2f7;
  padding: 1pt 3pt;
  border-radius: 3pt;
}}
strong {{ color: #111827; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def convert_one(md_name: str, title: str) -> None:
    md_path = ROOT / md_name
    html_path = md_path.with_suffix(".html")
    pdf_path = md_path.with_suffix(".pdf")
    html_path.write_text(markdown_to_html(md_path.read_text(encoding="utf-8"), title), encoding="utf-8")
    with pdf_path.open("wb") as out:
        html_result = subprocess.run(
            ["cupsfilter", "-i", "text/html", "-m", "application/pdf", str(html_path)],
            cwd=str(ROOT),
            stdout=out,
            stderr=subprocess.PIPE,
            check=False,
        )
    if html_result.returncode == 0 and pdf_path.stat().st_size > 0:
        return

    # macOS installations without the HTML CUPS filter can still create a
    # reliable PDF from plain UTF-8 text. This keeps the manual self-contained
    # without adding external dependencies such as pandoc or reportlab.
    with pdf_path.open("wb") as out:
        subprocess.run(
            ["cupsfilter", "-i", "text/plain", "-m", "application/pdf", str(md_path)],
            cwd=str(ROOT),
            stdout=out,
            stderr=subprocess.PIPE,
            check=True,
        )


def main() -> None:
    convert_one("MANUAL_CZ.md", "Astro Stacker 2.2 - uživatelský manuál")
    convert_one("MANUAL_EN.md", "Astro Stacker 2.2 - User Manual")


if __name__ == "__main__":
    main()
