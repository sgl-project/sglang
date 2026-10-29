"""CPU tests for the Unlimited-OCR v2 KV pool and its request-row sizing."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.memory_pool import UnlimitedOCRV2KVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestUnlimitedOCRV2KVPool(CustomTestCase):
    def test_move_copies_main_and_index_k(self):
        """Document KV reuse relocates tokens; the index K must move with K/V."""
        layer_ids = [0, 1]
        pool = UnlimitedOCRV2KVPool(
            size=8,
            page_size=4,
            dtype=torch.float32,
            head_num=2,
            head_dim=8,
            idx_head_dim=16,
            dense_layer_ids=[],
            sparse_layer_ids=layer_ids,
            disable_value_sparse_layer_ids=layer_ids,
            device="cpu",
            start_layer=0,
            end_layer=2,
        )
        src = torch.tensor([4, 5], dtype=torch.int64)
        dst = torch.tensor([6, 7], dtype=torch.int64)
        buffers = (
            pool.main_pool.k_buffer
            + pool.main_pool.v_buffer
            + pool.index_k_pool.k_buffer
        )
        for i, buf in enumerate(buffers):
            buf[src] = i + 1
            buf[dst] = 0
        pool.move_kv_cache(dst, src)
        for buf in buffers:
            torch.testing.assert_close(buf[dst], buf[src])


class TestRequestPoolRows(CustomTestCase):
    """A retained window and its document's running window each need a row.

    Regression: with rows == max_running_requests, 64 in-flight documents left
    retained windows and running requests competing for the same rows, so
    windows were evicted (and recomputed without their predecessor's KV) on
    every admission once the in-flight count neared the running cap.
    """

    def _rows(self, *, v2, disable_split=False):
        from sglang.srt.mem_cache import kv_cache_configurator as kcc
        from sglang.srt.runtime_context import get_context

        hf_config = {
            "architectures": ["UnlimitedOCRForCausalLM"],
            "sparse_attention_config": {"training_stage": "dash" if v2 else "kl"},
        }
        configurator = kcc.KVCacheConfigurator.__new__(kcc.KVCacheConfigurator)
        configurator.model_config = SimpleNamespace(hf_config=hf_config)
        with get_context().override_server_args(disable_uocr_doc_split=disable_split):
            return configurator._req_pool_rows(64)

    def test_split_v2_gets_a_retained_row_per_running_row(self):
        self.assertEqual(self._rows(v2=True), 128)

    def test_other_models_and_unsplit_v2_keep_one_row_per_request(self):
        self.assertEqual(self._rows(v2=False), 64)
        self.assertEqual(self._rows(v2=True, disable_split=True), 64)


if __name__ == "__main__":
    unittest.main()
