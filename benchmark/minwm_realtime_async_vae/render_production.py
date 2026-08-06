#!/usr/bin/env python3
"""Render the benchmark base as a long-lived production manifest."""

from __future__ import annotations

import sys
from typing import Any

import yaml


_DISPOSABLE_LABELS = {
    "seedleap.ai/test-run",
    "seedleap.ai/ttl-after-test",
}


def _rewrite_metadata(value: Any) -> None:
    if isinstance(value, dict):
        metadata = value.get("metadata")
        if isinstance(metadata, dict):
            labels = metadata.setdefault("labels", {})
            if isinstance(labels, dict):
                for label in _DISPOSABLE_LABELS:
                    labels.pop(label, None)
                labels["seedleap.ai/environment"] = "production"
        for child in value.values():
            _rewrite_metadata(child)
    elif isinstance(value, list):
        for child in value:
            _rewrite_metadata(child)


def main() -> None:
    documents = [document for document in yaml.safe_load_all(sys.stdin) if document]
    for document in documents:
        _rewrite_metadata(document)
    yaml.safe_dump_all(documents, sys.stdout, sort_keys=False)


if __name__ == "__main__":
    main()
