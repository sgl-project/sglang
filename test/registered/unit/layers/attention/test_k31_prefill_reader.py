import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.kernels.ops.attention.dsa import k31_prefill_reader as reader
from sglang.srt.layers.attention.dsa_backend import (
    _k31_sparse_prefill_reader_decision,
    _resolve_k31_sparse_prefill_reader,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_COVERED = dict(requested="packed", deterministic=False, is_cuda=True, is_sm100=True)
_COVERED.update(shards_index_k=True, prefill_cp=False, index_kpool=1, hisparse=False)
_COVERED.update(kv_cache_store_fp8=True, kv_format="V32", physical_page_size=64)
_COVERED.update(local_num_q_heads=12, prefill_impl="flashmla_kv")


def _decide(available=lambda: True, **changes):
    args = {**_COVERED, "available": available, **changes}
    mode, why = _k31_sparse_prefill_reader_decision(**args)
    assert mode == _resolve_k31_sparse_prefill_reader(**args)
    return mode, why


class TestEligibility(CustomTestCase):
    def test_active_decisions(self):
        self.assertEqual(_decide(), ("packed", "all requirements met"))
        self.assertEqual(_decide(requested="unpacked")[0], "unpacked")
        mode, why = _decide(deterministic=True)
        self.assertEqual(mode, "unpacked")
        self.assertIn("deterministic", why)

    def test_uncovered_deployments_stay_off_and_say_why(self):
        available = MagicMock(return_value=True)
        cases = (
            (dict(is_cuda=False), "CUDA"),
            (dict(is_sm100=False), "SM100"),
            (dict(prefill_impl="prims_ts"), "the reader needs 'flashmla_kv'"),
            (dict(prefill_impl=None), "--dsa-prefill-backend is None"),
            (dict(shards_index_k=False), "index-K"),
            (dict(prefill_cp=True), "prefill CP"),
            (dict(index_kpool=2), "K-pool is 2"),
            (dict(kv_cache_store_fp8=False), "fp8_ds_mla"),
            (dict(kv_format="V32_NO_ROPE"), "V32_NO_ROPE"),
            (dict(physical_page_size=32), "page size is 32"),
            (dict(local_num_q_heads=48), "48 heads"),
            (dict(hisparse=True), "HiSparse"),
        )
        for change, text in cases:
            with self.subTest(change=change):
                mode, why = _decide(available=available, **change)
                self.assertEqual(mode, "off")
                self.assertIn(text, why)
        available.assert_not_called()
        mode, why = _decide(available=lambda: False)
        self.assertEqual(mode, "off")
        self.assertIn("sgl-flash-mla", why)
        _, why = _decide(prefill_impl="trtllm", kv_cache_store_fp8=False)
        self.assertIn("--dsa-prefill-backend", why)
        self.assertIn("fp8_ds_mla", why)


class TestFallback(CustomTestCase):
    def setUp(self):
        reader._reset_for_tests()
        self.addCleanup(reader._reset_for_tests)

    def test_missing_operators_or_wheel_keep_stock_with_one_warning(self):
        old_wheel = SimpleNamespace(supports_sparse_prefill_token_packing=lambda: False)
        cases = (
            (dict(return_value=old_wheel), "flash_mla_sparse_fwd_packed"),
            (dict(side_effect=ImportError("No module named 'flash_mla'")), "No module"),
        )
        for how, text in cases:
            reader._reset_for_tests()
            with self.subTest(text=text), patch.object(reader, "_flash_mla", **how):
                with self.assertLogs(reader.logger, level="WARNING") as logs:
                    self.assertFalse(reader.k31_prefill_reader_available())
                    self.assertFalse(reader.k31_prefill_reader_available())
                self.assertEqual(len(logs.output), 1)
                self.assertIn("unavailable", logs.output[0])
                self.assertIn(text, logs.output[0])

    def test_runtime_error_disables_the_reader(self):
        dequant = MagicMock(side_effect=RuntimeError("CUDA out of memory"))
        wheel = SimpleNamespace(
            supports_sparse_prefill_token_packing=lambda: True,
            flash_mla_sparse_fwd_packed=MagicMock(),
            flash_mla_sparse_pack_indices=MagicMock(),
            flash_mla_dequant_v32_kv=dequant,
        )
        q = torch.zeros(4, 12, 512, dtype=torch.bfloat16)
        q_rope = torch.zeros(4, 12, 64, dtype=torch.bfloat16)
        kv_cache = torch.zeros(1, 64, 1, 656, dtype=torch.uint8)
        indices = torch.zeros(4, 128, dtype=torch.int32)
        with patch.object(reader, "_flash_mla", return_value=wheel):
            with self.assertLogs(reader.logger, level="ERROR"):
                for _ in range(2):
                    out = reader.k31_prefill_reader(q, q_rope, kv_cache, indices, 0.05)
                    self.assertIsNone(out)
        self.assertEqual(dequant.call_count, 1)
        self.assertIn("out of memory", reader.reader_stats()["disabled"])


if __name__ == "__main__":
    unittest.main()
