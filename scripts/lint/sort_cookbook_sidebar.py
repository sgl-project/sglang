#!/usr/bin/env python3
"""Keep the vendor groups in the Cookbook sidebar sorted alphabetically.

In docs/docs.json, the Cookbook tab has one section per model family
(Autoregressive, Diffusion, VLA, ...). Each section lists its plain pages
(e.g. the intro) first, followed by one nested group per vendor. This script
sorts those vendor groups case-insensitively by name, leaving plain pages and
the pages inside each vendor group untouched.

By default it rewrites docs/docs.json in place (pre-commit then fails if the
file changed). Pass --check to only report, without writing.
"""

import argparse
import json
import sys
from pathlib import Path

DOCS_JSON = Path("docs/docs.json")
COOKBOOK_TAB = "Cookbook"


def sort_key(group: dict) -> str:
    return group["group"].casefold()


def iter_family_sections(docs: dict):
    """Yield each section under the Cookbook tab whose pages hold vendor groups."""
    for tab in docs["navigation"]["tabs"]:
        if tab.get("tab") != COOKBOOK_TAB:
            continue
        for top_group in tab.get("groups", []):
            for section in top_group.get("pages", []):
                if isinstance(section, dict) and any(
                    isinstance(p, dict) for p in section.get("pages", [])
                ):
                    yield section


def sort_sections(docs: dict) -> list[str]:
    """Sort vendor groups in place; return messages for sections that changed."""
    errors = []
    for section in iter_family_sections(docs):
        pages = section["pages"]
        plain = [p for p in pages if isinstance(p, str)]
        groups = [p for p in pages if isinstance(p, dict)]
        if pages[: len(plain)] != plain:
            errors.append(
                f"{section['group']}: plain pages must come before vendor groups"
            )
        expected = plain + sorted(groups, key=sort_key)
        if pages != expected:
            if pages[: len(plain)] == plain:
                errors.append(
                    f"{section['group']}: vendor groups are not in alphabetical order"
                )
            section["pages"] = expected
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true", help="report only, do not rewrite the file"
    )
    args = parser.parse_args()

    text = DOCS_JSON.read_text(encoding="utf-8")
    docs = json.loads(text)
    errors = sort_sections(docs)
    if not errors:
        return 0

    for msg in errors:
        print(f"{DOCS_JSON}: {msg}", file=sys.stderr)
    if not args.check:
        DOCS_JSON.write_text(
            json.dumps(docs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"Sorted Cookbook vendor groups in {DOCS_JSON}.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
