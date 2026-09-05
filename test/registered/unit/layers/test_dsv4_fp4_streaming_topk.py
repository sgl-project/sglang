import sys
import types
import unittest
from unittest import mock

from sglang.kernels.ops.attention.dsv4 import fp4_indexer_hip
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _fake_aiter_modules(entrypoint=None):
    aiter = types.ModuleType("aiter")
    ops = types.ModuleType("aiter.ops")
    flydsl = types.ModuleType("aiter.ops.flydsl")
    if entrypoint is not None:
        flydsl.flydsl_pa_mqa_topk_fp4_prefill = entrypoint
    aiter.ops = ops
    ops.flydsl = flydsl
    return {
        "aiter": aiter,
        "aiter.ops": ops,
        "aiter.ops.flydsl": flydsl,
    }


class TestAiterFP4StreamingTopKCapability(unittest.TestCase):
    def tearDown(self):
        fp4_indexer_hip.get_aiter_fp4_streaming_topk.cache_clear()

    def test_detects_exact_entrypoint(self):
        entrypoint = lambda: None
        with mock.patch.dict(sys.modules, _fake_aiter_modules(entrypoint)):
            self.assertIs(
                fp4_indexer_hip.get_aiter_fp4_streaming_topk(),
                entrypoint,
            )
            self.assertTrue(fp4_indexer_hip.aiter_fp4_streaming_topk_available())

    def test_missing_entrypoint_uses_legacy_path(self):
        with mock.patch.dict(sys.modules, _fake_aiter_modules()):
            self.assertIsNone(fp4_indexer_hip.get_aiter_fp4_streaming_topk())
            self.assertFalse(fp4_indexer_hip.aiter_fp4_streaming_topk_available())

    def test_capability_is_frozen_after_first_probe(self):
        entrypoint = lambda: None
        with mock.patch.dict(sys.modules, _fake_aiter_modules()):
            self.assertIsNone(fp4_indexer_hip.get_aiter_fp4_streaming_topk())
        with mock.patch.dict(sys.modules, _fake_aiter_modules(entrypoint)):
            self.assertIsNone(fp4_indexer_hip.get_aiter_fp4_streaming_topk())


if __name__ == "__main__":
    unittest.main()
