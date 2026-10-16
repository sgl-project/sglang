"""Check that every docs page is reachable from the site navigation.

`docs.json` is the source of truth for site structure, so a page that is not
listed there ships but never appears in the sidebar. Pages that are deliberately
unlisted belong in UNLISTED below, with a reason.

Usage:
    python docs/scripts/check_nav_coverage.py
"""

import json
import sys
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent.parent
CONFIG = DOCS_DIR / "docs.json"

UNLISTED = {
    # Mintlify serves the site root from this page; it has no sidebar entry.
    "index",
    # Repository README for the cookbook directory, not a published page.
    "cookbook/diffusion/README",
    # No Diffusion group exists under Ascend NPUs yet; placing it needs a call
    # on whether Ascend diffusion docs live here or under SGLang Diffusion.
    "docs/hardware-platforms/ascend-npus/diffusion/disaggregation",
}


def navigation_pages(config: dict) -> set[str]:
    """Every page path referenced anywhere under `navigation`."""
    found: set[str] = set()

    def walk(node) -> None:
        if isinstance(node, str):
            found.add(node)
        elif isinstance(node, list):
            for item in node:
                walk(item)
        elif isinstance(node, dict):
            for value in node.values():
                walk(value)

    walk(config.get("navigation", {}))
    return {
        page
        for page in found
        if "/" in page and not page.startswith(("http://", "https://", "/", "#"))
    }


def main() -> int:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    nav = navigation_pages(config)
    pages = {
        str(path.relative_to(DOCS_DIR)).removesuffix(".mdx")
        for path in DOCS_DIR.rglob("*.mdx")
    }

    missing = sorted(nav - pages)
    unlisted = sorted(pages - nav - UNLISTED)
    stale_allowlist = sorted(UNLISTED - pages)

    for page in missing:
        print(f"navigation points at a page that does not exist: {page}")
    for page in unlisted:
        print(f"page is not reachable from the navigation: {page}")
    for page in stale_allowlist:
        print(f"UNLISTED entry no longer matches a page: {page}")

    if missing or unlisted or stale_allowlist:
        print(
            f"\n{len(missing) + len(unlisted) + len(stale_allowlist)} problem(s) found."
        )
        return 1
    print(f"{len(pages)} pages, all accounted for.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
