#!/usr/bin/env python3
"""Validate the durable Coordinator table contract before a production rollout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


EXPECTED_PRIMARY_KEY = [
    {"AttributeName": "pk", "KeyType": "HASH"},
    {"AttributeName": "sk", "KeyType": "RANGE"},
]
EXPECTED_ALLOCATION_KEY = [
    {"AttributeName": "allocation_key", "KeyType": "HASH"},
    {"AttributeName": "allocation_sort", "KeyType": "RANGE"},
]
EXPECTED_ATTRIBUTES = {
    "pk": "S",
    "sk": "S",
    "allocation_key": "S",
    "allocation_sort": "S",
}


def _load(path: Path | None):
    if path is None:
        return yaml.safe_load(sys.stdin)
    return json.loads(path.read_text(encoding="utf-8"))


def validate(payload: dict) -> None:
    table = payload.get("Table", payload)
    ttl = payload.get("TimeToLiveDescription", {})
    if table.get("TableStatus") != "ACTIVE":
        raise ValueError("Coordinator table must be ACTIVE")
    if table.get("KeySchema") != EXPECTED_PRIMARY_KEY:
        raise ValueError("Coordinator primary key schema must be pk HASH + sk RANGE")
    attributes = {
        item.get("AttributeName"): item.get("AttributeType")
        for item in table.get("AttributeDefinitions", [])
    }
    if attributes != EXPECTED_ATTRIBUTES:
        raise ValueError("Coordinator key attribute definitions must all use String")
    indexes = {
        item.get("IndexName"): item
        for item in table.get("GlobalSecondaryIndexes", [])
    }
    allocation = indexes.get("allocation-index")
    if allocation is None or allocation.get("IndexStatus") != "ACTIVE":
        raise ValueError("allocation-index must exist and be ACTIVE")
    if allocation.get("KeySchema") != EXPECTED_ALLOCATION_KEY:
        raise ValueError("allocation-index key schema is incompatible")
    if allocation.get("Projection", {}).get("ProjectionType") != "ALL":
        raise ValueError("allocation-index projection must be ALL")
    if ttl.get("TimeToLiveStatus") != "ENABLED" or ttl.get("AttributeName") != "ttl":
        raise ValueError("Coordinator TTL must be ENABLED on the ttl attribute")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table-file", type=Path)
    parser.add_argument("--ttl-file", type=Path)
    args = parser.parse_args()
    if args.table_file or args.ttl_file:
        if not args.table_file or not args.ttl_file:
            parser.error("--table-file and --ttl-file must be provided together")
        payload = _load(args.table_file)
        payload.update(_load(args.ttl_file))
    else:
        payload = _load(None)
    try:
        validate(payload)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
