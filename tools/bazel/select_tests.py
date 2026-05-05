#!/usr/bin/env python3
"""Build Bazel query expressions for changed-file test selection."""

from __future__ import annotations

import argparse
import fnmatch
import json
from pathlib import Path

from gen_source_graph import REPO_ROOT, source_target_for_file

IMPACT_RULES = REPO_ROOT / "tools" / "bazel" / "impact_rules.json"
REGISTERED_TESTS = "tests(//test/registered/...)"


def _impact_rules() -> list[dict]:
    return json.loads(IMPACT_RULES.read_text())


def _reverse_dep_query(label: str) -> str:
    return f'kind(".*_test rule", rdeps({REGISTERED_TESTS}, {label}))'


def _impact_query(tag_regex: str) -> str:
    return f'attr("tags", "{tag_regex}", {REGISTERED_TESTS})'


def _queries_for(changed: list[str]) -> dict[str, list[dict[str, str]]]:
    reverse_dep_queries: list[dict[str, str]] = []
    impact_queries: list[dict[str, str]] = []

    seen_source_labels: set[str] = set()
    for raw in changed:
        path = (REPO_ROOT / raw).resolve()
        label = source_target_for_file(path)
        if label and label not in seen_source_labels:
            seen_source_labels.add(label)
            reverse_dep_queries.append({"source": raw, "query": _reverse_dep_query(label)})

    for raw in changed:
        normalized = Path(raw).as_posix()
        for rule in _impact_rules():
            if not any(fnmatch.fnmatch(normalized, pattern) for pattern in rule["paths"]):
                continue
            for tag_regex in rule["suite_tag_regexes"]:
                impact_queries.append(
                    {
                        "source": raw,
                        "rule": rule["name"],
                        "reason": rule["reason"],
                        "query": _impact_query(tag_regex),
                    }
                )

    return {
        "reverse_dep_queries": reverse_dep_queries,
        "impact_queries": impact_queries,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("changed_files", nargs="+")
    args = parser.parse_args()
    print(json.dumps(_queries_for(args.changed_files), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
