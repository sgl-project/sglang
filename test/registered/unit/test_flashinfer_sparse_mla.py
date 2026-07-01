import sys
import unittest
from types import ModuleType
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.dsa.flashinfer_sparse_mla import (
    flashinfer_sparse_mla_forward,
    get_flashinfer_sparse_mla_op,
    validate_flashinfer_sparse_mla_skip_softmax,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestFlashInferSparseMLAAdapter(unittest.TestCase):
    def setUp(self):
        get_flashinfer_sparse_mla_op.cache_clear()

    def tearDown(self):
        get_flashinfer_sparse_mla_op.cache_clear()

    def _mock_flashinfer(self, op):
        flashinfer = ModuleType("flashinfer")
        flashinfer.__path__ = []
        mla = ModuleType("flashinfer.mla")
        mla.trtllm_batch_decode_with_kv_cache_mla = op
        flashinfer.mla = mla
        return patch.dict(
            sys.modules,
            {"flashinfer": flashinfer, "flashinfer.mla": mla},
        )

    def _run_adapter(self):
        return flashinfer_sparse_mla_forward(
            q=torch.zeros((2, 8, 576), dtype=torch.bfloat16),
            kv_cache=torch.zeros((128, 1, 656), dtype=torch.uint8),
            indices=torch.tensor([[7, 9, -1, -1], [4, 6, 8, -1]], dtype=torch.int32),
            seq_lens=torch.tensor([2, 3], dtype=torch.int32),
            workspace_buffer=torch.zeros(1024, dtype=torch.uint8),
            page_size=64,
            kv_cache_dim=656,
            qk_nope_head_dim=192,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            v_head_dim=512,
            sm_scale=0.125,
        )

    def test_maps_sglang_layout_to_public_flashinfer_api(self):
        captured = {}

        def fake_op(
            *,
            query,
            kv_cache,
            workspace_buffer,
            qk_nope_head_dim,
            kv_lora_rank,
            qk_rope_head_dim,
            block_tables,
            seq_lens,
            max_seq_len,
            sparse_mla_top_k,
            out,
            bmm1_scale,
            bmm2_scale,
            backend,
            kv_scale_format,
        ):
            captured.update(locals())
            out.fill_(2)
            return out

        with self._mock_flashinfer(fake_op):
            output = self._run_adapter()

        self.assertEqual(tuple(captured["query"].shape), (2, 1, 8, 576))
        self.assertEqual(tuple(captured["kv_cache"].shape), (2, 1, 64, 656))
        self.assertEqual(tuple(captured["block_tables"].shape), (2, 1, 4))
        self.assertEqual(
            captured["block_tables"].tolist(),
            [[[7, 9, -1, -1]], [[4, 6, 8, -1]]],
        )
        self.assertEqual(captured["seq_lens"].tolist(), [2, 3])
        self.assertEqual(captured["max_seq_len"], 4)
        self.assertEqual(captured["sparse_mla_top_k"], 4)
        self.assertEqual(captured["qk_nope_head_dim"], 192)
        self.assertEqual(captured["bmm1_scale"], 0.125)
        self.assertEqual(captured["bmm2_scale"], 1.0)
        self.assertEqual(captured["backend"], "sparse")
        self.assertEqual(captured["kv_scale_format"], "arbitrary_fp32")
        self.assertEqual(tuple(captured["out"].shape), (2, 1, 8, 512))
        self.assertEqual(tuple(output.shape), (2, 8, 512))
        self.assertTrue(torch.all(output == 2))

    def test_rejects_flashinfer_api_before_sm120_merge(self):
        def old_op(query, kv_cache, workspace_buffer):
            return query

        with self._mock_flashinfer(old_op):
            with self.assertRaisesRegex(RuntimeError, "missing arguments"):
                self._run_adapter()

    def test_rejects_raw_mla_kv_layout(self):
        with self.assertRaisesRegex(ValueError, "656-byte packed"):
            flashinfer_sparse_mla_forward(
                q=torch.zeros((1, 8, 576), dtype=torch.bfloat16),
                kv_cache=torch.zeros((64, 1, 576), dtype=torch.uint8),
                indices=torch.zeros((1, 4), dtype=torch.int32),
                seq_lens=torch.ones(1, dtype=torch.int32),
                workspace_buffer=torch.zeros(1024, dtype=torch.uint8),
                page_size=64,
                kv_cache_dim=576,
                qk_nope_head_dim=192,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                v_head_dim=512,
                sm_scale=0.125,
            )

    def test_rejects_skip_softmax_configuration(self):
        with self.assertRaisesRegex(
            ValueError, "SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR"
        ):
            validate_flashinfer_sparse_mla_skip_softmax("decode", 1e-6)


if __name__ == "__main__":
    unittest.main()
