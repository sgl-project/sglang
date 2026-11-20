"""HiCache file backend classifies get/put outcomes instead of collapsing them.

A truncated object and a missing object both make the caller see "no page", but
only the backend can tell them apart. These tests pin that classification and
the reporting boundary (metrics enabled plus an attached collector).
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheFile,
    HiCacheStorageConfig,
    HiCacheStorageFailureInjector,
    StorageTransferResult,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _RecordingCollector:
    def __init__(self):
        self.events = []

    def log_storage_transfer(self, *, op, result, count=1):
        self.events.append((op, result, count))


def _config(enable_metrics=True):
    return HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=enable_metrics,
        is_page_first_layout=False,
        model_name="test/model",
        extra_config={
            "enable_metadata_cache": False,
            "max_size": 0,
            "min_free_space": 0,
        },
    )


class TestStorageTransferResults(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        context = envs.SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR.override(
            self.directory.name
        )
        context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        self.page = torch.arange(16, dtype=torch.uint8)

    def _backend(self, enable_metrics=True):
        backend = HiCacheFile(_config(enable_metrics))
        self.collector = _RecordingCollector()
        backend.set_metrics_collector(self.collector)
        return backend

    def test_missing_object_is_a_miss(self):
        backend = self._backend()
        target = torch.zeros_like(self.page)
        self.assertIsNone(backend.get("absent", target))
        self.assertEqual(self.collector.events, [("get", "miss", 1)])

    def test_truncated_object_is_short_read_not_a_hit(self):
        backend = self._backend()
        self.assertTrue(backend.set("truncated", self.page))
        self.collector.events.clear()

        path = Path(backend.file_path) / f"{backend._get_suffixed_key('truncated')}.bin"
        path.write_bytes(b"short")

        target = torch.full_like(self.page, 0xFF)
        self.assertIsNone(backend.get("truncated", target))
        self.assertEqual(self.collector.events, [("get", "short_read", 1)])
        self.assertTrue(torch.all(target == 0xFF))

    def test_read_oserror_is_an_error(self):
        backend = self._backend()
        with patch("builtins.open", side_effect=OSError("disk gone")):
            self.assertIsNone(backend.get("unreadable", torch.zeros_like(self.page)))
        self.assertEqual(self.collector.events, [("get", "error", 1)])

    def test_put_records_success_then_skipped(self):
        backend = self._backend()
        self.assertTrue(backend.set("page", self.page))
        self.assertTrue(backend.set("page", self.page))
        self.assertEqual(
            self.collector.events,
            [("put", "success", 1), ("put", "skipped", 1)],
        )

    def test_put_failure_is_an_error(self):
        backend = self._backend()
        with patch.object(
            backend._evictor, "reserve", side_effect=RuntimeError("no space")
        ):
            self.assertFalse(backend.set("new", self.page))
        self.assertEqual(self.collector.events, [("put", "error", 1)])

    def test_disabled_metrics_records_nothing(self):
        backend = self._backend(enable_metrics=False)
        self.assertIsNone(backend.get("absent", torch.zeros_like(self.page)))
        self.assertTrue(backend.set("page", self.page))
        self.assertEqual(self.collector.events, [])

    def test_failure_injector_forwards_the_collector(self):
        backend = self._backend()
        injector = HiCacheStorageFailureInjector(backend, 0.0, 0.0)
        injector.set_metrics_collector(self.collector)
        self.assertIs(backend.metrics_collector, self.collector)

    def test_labels_match_the_public_metric(self):
        self.assertEqual(
            {result.value for result in StorageTransferResult},
            {"success", "miss", "short_read", "error", "skipped"},
        )


if __name__ == "__main__":
    unittest.main()
