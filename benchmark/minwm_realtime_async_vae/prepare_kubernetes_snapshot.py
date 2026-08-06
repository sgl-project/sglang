#!/usr/bin/env python3
"""Remove server-owned fields from a Kubernetes object before rollback apply."""

from __future__ import annotations

import json
import sys


_SERVER_METADATA = {
    "creationTimestamp",
    "generation",
    "managedFields",
    "resourceVersion",
    "selfLink",
    "uid",
}


def main() -> None:
    resource = json.load(sys.stdin)
    resource.pop("status", None)
    metadata = resource.get("metadata", {})
    for field in _SERVER_METADATA:
        metadata.pop(field, None)
    annotations = metadata.get("annotations")
    if isinstance(annotations, dict):
        annotations.pop("kubectl.kubernetes.io/last-applied-configuration", None)
        if not annotations:
            metadata.pop("annotations", None)
    json.dump(resource, sys.stdout, separators=(",", ":"))


if __name__ == "__main__":
    main()
