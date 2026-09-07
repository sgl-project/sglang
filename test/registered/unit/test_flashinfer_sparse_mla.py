import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.flash_mla_sm120 import (
    _validate_flashinfer_sparse_mla_backend,
    flashinfer_sparse_mla_forward,
)
from sglang.srt.mem_cache import kv_cache_configurator
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


class TestFlashInferSparseMLAAdapter(unittest.TestCase):
    def test_maps_glm53_nope_layout_to_persistent_flashinfer_runner(self):
        captured = {}

        class FakeRunner:
            def run(self, q, kv_cache, indices, output, sm_scale, **kwargs):
                captured.update(
                    q=q,
                    kv_cache=kv_cache,
                    indices=indices,
                    output=output,
                    sm_scale=sm_scale,
                    **kwargs,
                )
                output.fill_(2)

        indices = torch.full((2, 2051), -1, dtype=torch.int32)
        indices[0, :2] = torch.tensor([7, 9], dtype=torch.int32)
        indices[1, :3] = torch.tensor([4, 6, 8], dtype=torch.int32)
        output = flashinfer_sparse_mla_forward(
            q=torch.zeros((2, 8, 512), dtype=torch.bfloat16),
            kv_cache=torch.zeros((128, 1, 528), dtype=torch.uint8),
            indices=indices,
            seq_lens=torch.tensor([2, 3], dtype=torch.int32),
            workspace_buffer=torch.zeros(2 * 1024 * 1024, dtype=torch.uint8),
            runner=FakeRunner(),
            page_size=64,
            kv_cache_dim=528,
            # This is the checkpoint's pre-absorption dimension. The query
            # reaching the native sparse-MLA kernel is 512 wide.
            qk_nope_head_dim=256,
            kv_lora_rank=512,
            qk_rope_head_dim=0,
            sm_scale=0.125,
            skip_softmax_threshold_scale_factor=None,
        )

        self.assertEqual(tuple(captured["q"].shape), (2, 8, 512))
        self.assertEqual(tuple(captured["kv_cache"].shape), (2, 64, 528))
        self.assertEqual(tuple(captured["indices"].shape), (2, 2176))
        self.assertEqual(captured["indices"][0, :4].tolist(), [7, 9, -1, -1])
        self.assertTrue(torch.all(captured["indices"][:, 2051:] == -1))
        self.assertEqual(captured["topk_length"].tolist(), [2, 3])
        self.assertEqual(captured["sm_scale"], 0.125)
        self.assertEqual(tuple(captured["mid_out"].shape), (2, 8, 34, 512))
        self.assertEqual(tuple(captured["mid_lse"].shape), (2, 8, 34))
        self.assertEqual(tuple(output.shape), (2, 8, 512))
        self.assertTrue(torch.all(output == 2))


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
            "Glm5NextForConditionalGeneration",
            "Glm5NextForConditionalGenerationNextN",
        ):
            with self.subTest(model_arch=model_arch):
                self.assertTrue(
                    self._validate(
                        "flashinfer_sparse_mla",
                        "flashinfer_sparse_mla",
                        model_arch,
                    )
                )

    def test_ignores_other_backends_when_flashinfer_is_not_selected(self):
        for prefill, decode in (("tilelang", "tilelang"), ("trtllm", "trtllm")):
            with self.subTest(prefill=prefill, decode=decode):
                self.assertFalse(self._validate(prefill, decode))

    def test_rejects_mixed_flashinfer_backend(self):
        with self.assertRaisesRegex(ValueError, "only flashinfer_sparse_mla"):
            self._validate("flashinfer_sparse_mla", "trtllm")

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


class TestFlashInferSparseMLAKVLayout(unittest.TestCase):
    def test_glm53_nope_layout_follows_kernel_capability(self):
        model_config = SimpleNamespace(
            hf_config=SimpleNamespace(), kv_lora_rank=512, qk_rope_head_dim=0,
        )
        execution = SimpleNamespace(kernel=SimpleNamespace(
            dsa_prefill_backend="flashinfer_sparse_mla",
            dsa_decode_backend="flashinfer_sparse_mla",
        ))
        for capability, expected in [
            (SimpleNamespace(), 656),
            (SimpleNamespace(compact_bytes_per_token=None), 656),
            (SimpleNamespace(compact_bytes_per_token=528), 528),
        ]:
            with (
                self.subTest(expected=expected),
                patch.object(kv_cache_configurator, "is_deepseek_dsa", return_value=True),
                patch.object(kv_cache_configurator, "get_exec", return_value=execution),
                patch.object(kv_cache_configurator, "get_disagg", return_value=SimpleNamespace(disaggregation_mode="null")),
                patch.object(kv_cache_configurator, "_is_hip", False),
                patch("flashinfer.mla.supported_sparse_mla_sm120_configs", return_value={"glm53_nope": capability}),
            ):
                self.assertEqual(kv_cache_configurator.calculate_mla_kv_cache_dim(
                    model_config=model_config, kv_cache_dtype=torch.float8_e4m3fn,
                ), expected)


if __name__ == "__main__":
    unittest.main()
