"""Check the injection tool before starting GPU inference workers."""

import importlib.util
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class TestFaultInjection(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        path = Path(__file__).parent / "cpu_buffer_fault_injection/sitecustomize.py"
        spec = importlib.util.spec_from_file_location("cpu_fault_test_hook", path)
        self.hook = importlib.util.module_from_spec(spec)
        with patch.dict(os.environ, {"SGLANG_CPU_FAULT_DIR": ""}):
            spec.loader.exec_module(self.hook)
        self.hook.ROOT = str(self.root)

        class Pool:
            def __init__(self):
                self._lock = threading.Lock()
                self._ranges = {}
                self._pending = []

            def allocate(self, size):
                return size

        class Mixin:
            def _send_cpu_chunk(self):
                self._cpu_pool._ranges[0] = 256
                self._cpu_pool._pending.append("handle")
                return self.agent.transfer("handle")

        self.module = SimpleNamespace(CpuBufferPool=Pool, CpuBufferMixin=Mixin)
        self.hook.install(self.module)
        with patch.object(threading.Thread, "start"):
            self.pool = Pool()

    def events(self):
        return [
            json.loads(line)
            for line in (self.root / f"{os.getpid()}.jsonl").read_text().splitlines()
        ]

    def test_unarmed_allocator_is_unchanged(self):
        self.assertEqual(self.pool.allocate(256), 256)
        self.assertFalse(any(row["event"] == "injected" for row in self.events()))

    def test_allocation_fault_is_buffer_error_and_fires_once(self):
        (self.root / "arm").write_text("allocate")
        with self.assertRaisesRegex(BufferError, "Injected"):
            self.pool.allocate(256)
        self.assertEqual(self.pool.allocate(256), 256)
        self.assertEqual(
            [row["kind"] for row in self.events() if row["event"] == "injected"],
            ["allocate"],
        )

    def test_post_fault_occurs_after_transfer_and_retains_ownership(self):
        (self.root / "arm").write_text("post")
        submitted = []
        manager = self.module.CpuBufferMixin()
        manager._cpu_pool = self.pool
        manager.agent = SimpleNamespace(
            transfer=lambda handle: submitted.append(handle)
        )
        with self.assertRaisesRegex(RuntimeError, "after NIXL write"):
            manager._send_cpu_chunk()
        self.assertEqual(submitted, ["handle"])
        events = self.events()
        self.assertEqual(
            [row["kind"] for row in events if row["event"] == "injected"], ["post"]
        )
        failure = next(row for row in events if row["event"] == "send_exception")
        self.assertEqual(failure["allocations"], 1)
        self.assertEqual(failure["pending"], 1)


if __name__ == "__main__":
    unittest.main()
