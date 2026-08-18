"""FlashInfer autotune must reach the same tactics on every TP rank.

Without a cross-rank reduction each rank's ``argmin`` follows local timing noise
(observed: 20/20 tuned MoE shapes diverged across 4 ranks on gpt-oss-120b).
Covers the group handed to FlashInfer and the cache digest gating cache reuse.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from sglang.srt.model_executor.runner.flashinfer_autotune import (
    _autotune_cache_digest,
    autotune_tactic_sync_group,
)
from sglang.test.test_utils import CustomTestCase

CPU_GROUP = object()


def _tp_group(world_size: int) -> SimpleNamespace:
    return SimpleNamespace(world_size=world_size, cpu_group=CPU_GROUP)


class TestAutotuneTacticSyncGroup(CustomTestCase):
    def test_multi_rank_tp_syncs_on_the_cpu_group(self):
        self.assertIs(autotune_tactic_sync_group(_tp_group(4)), CPU_GROUP)

    def test_single_rank_has_nobody_to_agree_with(self):
        self.assertIsNone(autotune_tactic_sync_group(_tp_group(1)))


class TestAutotuneCacheDigest(CustomTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def _write(self, name: str, configs: dict) -> Path:
        path = self.dir / name
        path.write_text(json.dumps(configs))
        return path

    def test_missing_and_corrupt_caches_read_as_empty(self):
        self.assertEqual(_autotune_cache_digest(self.dir / "absent.json"), "")
        corrupt = self.dir / "corrupt.json"
        corrupt.write_text("{not json")
        self.assertEqual(_autotune_cache_digest(corrupt), "")

    def test_metadata_is_not_part_of_the_tuning_result(self):
        # Ranks that agree on every tactic must compare equal.
        rank0 = self._write("rank0.json", {"_metadata": {"cublas": "12.8"}, "op": 7})
        rank1 = self._write("rank1.json", {"_metadata": {"cublas": "12.9"}, "op": 7})
        self.assertEqual(_autotune_cache_digest(rank0), _autotune_cache_digest(rank1))

    def test_differing_tactics_are_detected(self):
        rank0 = self._write("rank0.json", {"op": 7})
        rank1 = self._write("rank1.json", {"op": 8})
        self.assertNotEqual(
            _autotune_cache_digest(rank0), _autotune_cache_digest(rank1)
        )

    def test_key_order_does_not_matter(self):
        rank0 = self._write("rank0.json", {"a": 1, "b": 2})
        rank1 = self._write("rank1.json", {"b": 2, "a": 1})
        self.assertEqual(_autotune_cache_digest(rank0), _autotune_cache_digest(rank1))


if __name__ == "__main__":
    unittest.main()
