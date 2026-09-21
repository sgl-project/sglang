"""Opt-in, single-operation validation injection. No default fault behavior."""

import json
import os
import threading

from sglang.srt.kv_compression.types import BufferDrainError, KVVerificationError


class TestFault:
    POINTS = frozenset(
        {"after_scatter", "before_publish", "before_restore", "after_restore_copy"}
    )
    KINDS = frozenset({"error", "checksum", "undrained"})

    def __init__(self):
        raw = os.environ.get("SGLANG_KV_COMPRESSION_TEST_FAULT", "")
        self.spec = json.loads(raw) if raw else None
        self.lock = threading.Lock()
        self.fired = False
        if self.spec is not None:
            if set(self.spec) != {"operation", "point", "kind"}:
                raise ValueError(
                    "Test fault requires exactly operation, point and kind"
                )
            op = self.spec["operation"]
            if (
                not isinstance(op, str)
                or not op.startswith(("write:", "restore:"))
                or not op.split(":")[1].isdigit()
            ):
                raise ValueError(
                    "Test fault requires one exact write/restore handle; wildcards forbidden"
                )
            if (
                self.spec["point"] not in self.POINTS
                or self.spec["kind"] not in self.KINDS
            ):
                raise ValueError("Invalid test fault point or kind")

    def check(self, operation, point):
        if self.spec is None:
            return
        with self.lock:
            if (
                self.fired
                or self.spec["operation"] != operation
                or self.spec["point"] != point
            ):
                return
            self.fired = True
            kind = self.spec["kind"]
        error = {
            "error": RuntimeError,
            "checksum": KVVerificationError,
            "undrained": BufferDrainError,
        }[kind]
        raise error(f"Test fault at {operation}/{point}: {kind}")
