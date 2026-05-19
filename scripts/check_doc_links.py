"""Verify relative markdown links inside the sglang repo all resolve to existing files.

Checks all ``[text](path)`` references in ``test/registered/*/README.md`` and similar
in-repo markdown notes. Skips anchor-only references (``#section``), external URLs
(``http://`` / ``https://`` / ``mailto:``), and reference-link footers. Anchor
correctness inside an existing file is not verified — that needs a real markdown
parser.

Usage:

    uv run python scripts/check_doc_links.py
    uv run python scripts/check_doc_links.py path/to/docs/

Exits non-zero with a per-broken-link report when any link target is missing.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Annotated

import typer

_MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
_DEFAULT_ROOTS = (
    "test/registered",
    "docs",
)


def find_broken_links(
    repo_root: Path, search_roots: list[Path]
) -> list[tuple[Path, str, str]]:
    broken: list[tuple[Path, str, str]] = []

    md_files: list[Path] = []
    for root in search_roots:
        md_files.extend(sorted(root.rglob("*.md")))

    for md_path in md_files:
        text = md_path.read_text(encoding="utf-8")
        for raw_target in _MARKDOWN_LINK.findall(text):
            target = raw_target.strip()
            if not target:
                continue
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            if target.startswith("#"):
                continue
            path_part = target.split("#", 1)[0].split("?", 1)[0]
            if not path_part:
                continue
            resolved = (md_path.parent / path_part).resolve()
            if not resolved.exists():
                rel_source = md_path.relative_to(repo_root)
                broken.append((rel_source, target, str(resolved)))

    return broken


app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    roots: Annotated[
        list[Path],
        typer.Argument(
            help="Directories to scan recursively for .md files (relative to repo root). "
            "Defaults to test/registered and docs.",
        ),
    ] = None,
) -> None:
    repo_root = Path(__file__).resolve().parent.parent

    if roots:
        search_roots = [repo_root / r if not r.is_absolute() else r for r in roots]
    else:
        search_roots = [repo_root / r for r in _DEFAULT_ROOTS]

    missing_roots = [r for r in search_roots if not r.exists()]
    if missing_roots:
        for r in missing_roots:
            print(f"warning: search root missing, skipping: {r}", file=sys.stderr)
        search_roots = [r for r in search_roots if r.exists()]

    broken = find_broken_links(repo_root, search_roots)

    if broken:
        print(f"found {len(broken)} broken markdown link(s):", file=sys.stderr)
        for source, target, resolved in broken:
            print(f"  {source}: '{target}' -> {resolved}", file=sys.stderr)
        raise typer.Exit(code=1)

    print(f"OK: scanned {len(search_roots)} root(s), no broken markdown links")


if __name__ == "__main__":
    app()
