import sys
import unittest
from types import ModuleType
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention import flash_mla_sm120 as flash_mla_sm120_module
from sglang.kernels.ops.attention.flash_mla_sm120 import (
    _flash_mla_sm120_prefill,
    _validate_flashinfer_sparse_mla_backend,
    flashinfer_sparse_mla_forward,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


class TestFlashInferSparseMLAAdapter(unittest.TestCase):
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

    def test_maps_sglang_layout_to_public_flashinfer_api(self):
        captured = {}

        def fake_op(**kwargs):
            captured.update(kwargs)
            query = kwargs["query"]
            return query.new_full((*query.shape[:-1], kwargs["kv_lora_rank"]), 2)

        with self._mock_flashinfer(fake_op):
            output = flashinfer_sparse_mla_forward(
                q=torch.zeros((2, 8, 576), dtype=torch.bfloat16),
                kv_cache=torch.zeros((128, 1, 656), dtype=torch.uint8),
                indices=torch.tensor(
                    [[7, 9, -1, -1], [4, 6, 8, -1]], dtype=torch.int32
                ),
                seq_lens=torch.tensor([2, 3], dtype=torch.int32),
                workspace_buffer=torch.zeros(1024, dtype=torch.uint8),
                page_size=64,
                kv_cache_dim=656,
                qk_nope_head_dim=192,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                sm_scale=0.125,
                skip_softmax_threshold_scale_factor=0.25,
            )

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
        self.assertEqual(captured["kv_scale_format"], "arbitrary_fp32")
        self.assertEqual(captured["skip_softmax_threshold_scale_factor"], 0.25)
        self.assertNotIn("backend", captured)
        self.assertEqual(tuple(output.shape), (2, 8, 512))
        self.assertTrue(torch.all(output == 2))

    def test_dsv41_prefill_splits_main_and_extra_pages_independently(self):
        captured = {}

        def fake_paged_attention(*args, **kwargs):
            captured.update(kwargs)

        flashinfer = ModuleType("flashinfer")
        flashinfer.__path__ = []
        mla = ModuleType("flashinfer.mla")
        mla.__path__ = []
        sparse = ModuleType("flashinfer.mla._sparse_mla_sm120")
        sparse._sparse_mla_sm120_paged_attention = fake_paged_attention
        flashinfer.mla = mla

        main_split = torch.empty((8, 64, 1, 584), dtype=torch.uint8)
        extra_split = torch.empty((4, 64, 1, 584), dtype=torch.uint8)
        with (
            patch.dict(
                sys.modules,
                {
                    "flashinfer": flashinfer,
                    "flashinfer.mla": mla,
                    "flashinfer.mla._sparse_mla_sm120": sparse,
                },
            ),
            patch.object(
                flash_mla_sm120_module,
                "_split_kv_pages_to_64",
                side_effect=(main_split, extra_split),
            ) as split,
        ):
            output, _ = _flash_mla_sm120_prefill(
                q=torch.zeros((2, 1, 4, 576), dtype=torch.bfloat16),
                k_cache=torch.zeros((2, 256, 1, 584), dtype=torch.uint8),
                indices=torch.zeros((2, 1, 8), dtype=torch.int32),
                topk_length=torch.full((2,), 8, dtype=torch.int32),
                attn_sink=None,
                head_dim_v=512,
                softmax_scale=0.125,
                extra_k_cache=torch.zeros((2, 128, 1, 584), dtype=torch.uint8),
                extra_indices=torch.zeros((2, 1, 4), dtype=torch.int32),
                extra_topk_length=torch.full((2,), 4, dtype=torch.int32),
            )

        self.assertEqual(tuple(output.shape), (2, 1, 4, 512))
        self.assertIs(captured["extra_kv_cache"], extra_split)
        self.assertEqual(split.call_count, 2)
        self.assertNotIn("buf_key_suffix", split.call_args_list[0].kwargs)
        self.assertEqual(split.call_args_list[1].kwargs["buf_key_suffix"], ":extra")


class TestFlashInferSparseMLABackendGate(unittest.TestCase):
    def _validate(self, prefill, decode, model_arch="GlmMoeDsaForCausalLM"):
        return _validate_flashinfer_sparse_mla_backend(
            model_arch=model_arch,
            device_sm_major=12,
            kv_cache_dtype=torch.float8_e4m3fn,
            prefill_impl=prefill,
            decode_impl=decode,
        )

    def test_accepts_flashinfer_for_both_phases(self):
        for model_arch in (
            "GlmMoeDsaForCausalLM",
            "GlmMoeDsaForCausalLMNextN",
        ):
            with self.subTest(model_arch=model_arch):
                self.assertTrue(
                    self._validate(
                        "flashinfer_sparse_mla",
                        "flashinfer_sparse_mla",
                        model_arch,
                    )
                )

    def test_rejects_other_or_mixed_backends(self):
        for prefill, decode in (
            ("trtllm", "trtllm"),
            ("flashinfer_sparse_mla", "trtllm"),
        ):
            with self.subTest(prefill=prefill, decode=decode):
                with self.assertRaisesRegex(ValueError, "only flashinfer_sparse_mla"):
                    self._validate(prefill, decode)

    def test_reports_unsupported_configuration(self):
        with self.assertRaises(ValueError) as error:
            self._validate(
                "flashinfer_sparse_mla",
                "flashinfer_sparse_mla",
                "DeepseekV3ForCausalLM",
            )

        message = str(error.exception)
        self.assertIn("model_arch='DeepseekV3ForCausalLM'", message)
        self.assertIn("sm_major=12", message)
        self.assertIn("kv_cache_dtype=torch.float8_e4m3fn", message)


if __name__ == "__main__":
    unittest.main()
