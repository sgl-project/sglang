import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.flash_mla_sm120 import (
    _flashinfer_sparse_mla_max_tokens,
    _validate_flashinfer_sparse_mla_backend,
    create_flashinfer_sparse_mla_runner,
    flashinfer_sparse_mla_forward,
)
from sglang.srt.layers.attention.dsa.dsa_backend_kpool import (
    DeepseekSparseAttnBackendKPoolMixin,
)
from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
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
        backend = SimpleNamespace(
            workspace_buffer=torch.zeros(2 * 1024 * 1024, dtype=torch.uint8),
            flashinfer_sparse_mla_runner=FakeRunner(),
            real_page_size=64,
            kv_cache_dim=528,
            qk_nope_head_dim=256,
            kv_lora_rank=512,
            qk_rope_head_dim=0,
            dsa_index_kpool=4,
        )
        DeepseekSparseAttnBackendKPoolMixin._check_kpool_tail_backend(
            backend, indices, "flashinfer_sparse_mla", "prefill"
        )
        output = DeepseekSparseAttnBackend._forward_flashinfer_sparse_mla(
            backend,
            q_all=torch.zeros((2, 8, 512), dtype=torch.bfloat16),
            kv_cache=torch.zeros((128, 1, 528), dtype=torch.uint8),
            page_table_1=indices,
            seq_lens=torch.tensor([4096, 8192], dtype=torch.int32),
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


class TestFlashInferSparseMLARunnerCompatibility(unittest.TestCase):
    def _create(self, rope=0):
        return create_flashinfer_sparse_mla_runner(
            qk_rope_head_dim=rope,
            kv_lora_rank=512,
            max_num_tokens=4096,
            max_num_heads=32,
            device="cpu",
        )

    def test_existing_rope_path_does_not_require_native_api(self):
        with patch("flashinfer.mla.SparseMLASm120Wrapper", None, create=True):
            self.assertIsNone(self._create(rope=64))

    def test_nope_requires_advertised_native_support(self):
        with patch("flashinfer.mla.SparseMLASm120Wrapper", None, create=True):
            with self.assertRaisesRegex(RuntimeError, "glm53_nope"):
                self._create()

    def test_native_wrapper_is_sized_before_capture(self):
        marker = object()
        with (
            patch(
                "flashinfer.mla.SparseMLASm120Wrapper", return_value=marker, create=True
            ) as wrapper,
            patch(
                "flashinfer.mla.supported_sparse_mla_sm120_configs",
                return_value={"glm53_nope": object()},
                create=True,
            ),
        ):
            self.assertIs(self._create(), marker)
            wrapper.assert_called_once_with(
                max_num_tokens=4096,
                max_num_heads=32,
                d_v=512,
                kv_scale_format="arbitrary_fp32",
                device="cpu",
            )

    def test_legacy_rope_call_keeps_pinned_flashinfer_api(self):
        expected = torch.ones(2, 1, 8, 512, dtype=torch.bfloat16)
        with patch(
            "flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla",
            return_value=expected,
        ) as legacy:
            output = flashinfer_sparse_mla_forward(
                q=torch.zeros(2, 8, 576, dtype=torch.bfloat16),
                kv_cache=torch.zeros(128, 1, 656, dtype=torch.uint8),
                indices=torch.zeros(2, 2048, dtype=torch.int32),
                seq_lens=torch.tensor([2, 3], dtype=torch.int32),
                workspace_buffer=torch.zeros(1, dtype=torch.uint8),
                page_size=64,
                kv_cache_dim=656,
                qk_nope_head_dim=512,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                sm_scale=0.125,
                skip_softmax_threshold_scale_factor=None,
            )
        torch.testing.assert_close(output, expected.squeeze(1))
        self.assertEqual(legacy.call_args.kwargs["query"].shape, (2, 1, 8, 576))
        self.assertEqual(legacy.call_args.kwargs["kv_cache"].shape, (2, 1, 64, 656))


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
            hf_config=SimpleNamespace(),
            kv_lora_rank=512,
            qk_rope_head_dim=0,
        )
        execution = SimpleNamespace(
            kernel=SimpleNamespace(
                dsa_prefill_backend="flashinfer_sparse_mla",
                dsa_decode_backend="flashinfer_sparse_mla",
            )
        )
        for capability, expected in [
            ("missing_api", 656),
            (None, 656),
            (SimpleNamespace(), 656),
            (SimpleNamespace(compact_bytes_per_token=None), 656),
            (SimpleNamespace(compact_bytes_per_token=528), 528),
        ]:
            with (
                self.subTest(expected=expected),
                patch.object(
                    kv_cache_configurator, "is_deepseek_dsa", return_value=True
                ),
                patch.object(kv_cache_configurator, "get_exec", return_value=execution),
                patch.object(
                    kv_cache_configurator,
                    "get_disagg",
                    return_value=SimpleNamespace(disaggregation_mode="null"),
                ),
                patch.object(kv_cache_configurator, "_is_hip", False),
                patch(
                    "flashinfer.mla.supported_sparse_mla_sm120_configs",
                    new=(
                        None
                        if capability == "missing_api"
                        else lambda: {"glm53_nope": capability}
                    ),
                    create=True,
                ),
            ):
                self.assertEqual(
                    kv_cache_configurator.calculate_mla_kv_cache_dim(
                        model_config=model_config,
                        kv_cache_dtype=torch.float8_e4m3fn,
                    ),
                    expected,
                )


class TestFlashInferSparseMLARunnerCapacity(unittest.TestCase):
    def test_unchunked_first_prompt_can_exceed_prefill_budget(self):
        for chunk in (None, 0, -1):
            with self.subTest(chunk=chunk):
                capacity = _flashinfer_sparse_mla_max_tokens(
                    chunked_prefill_size=chunk,
                    max_prefill_tokens=16384,
                    context_len=32768,
                    max_running_requests=4,
                    speculative_num_draft_tokens=6,
                )
                self.assertGreaterEqual(capacity, 32768 + 4 * 6)

    def test_chunked_mixed_batch_does_not_reserve_entire_context(self):
        capacity = _flashinfer_sparse_mla_max_tokens(
            chunked_prefill_size=4096,
            max_prefill_tokens=4096,
            context_len=524288,
            max_running_requests=4,
            speculative_num_draft_tokens=6,
        )
        self.assertEqual(capacity, 4120)

    def test_unchunked_batch_budget_can_exceed_one_context(self):
        capacity = _flashinfer_sparse_mla_max_tokens(
            chunked_prefill_size=None,
            max_prefill_tokens=65536,
            context_len=32768,
            max_running_requests=4,
            speculative_num_draft_tokens=None,
        )
        self.assertGreaterEqual(capacity, 65536 + 4)


class TestFlashInferSparseMLAIndexAndWorkspaceBounds(unittest.TestCase):
    def _run(self, indices, heads=32, workspace_bytes=None):
        captured = {}

        class FakeRunner:
            def run(self, q, kv_cache, indices, output, sm_scale, **kwargs):
                captured.update(indices=indices, **kwargs)
                output.zero_()

        tokens = indices.shape[0]
        scratch_heads = 8 if heads == 8 else ((heads + 15) // 16) * 16
        required = tokens * scratch_heads * 34 * (512 * 2 + 4)
        flashinfer_sparse_mla_forward(
            q=torch.zeros(tokens, heads, 512, dtype=torch.bfloat16),
            kv_cache=torch.zeros(1, 64, 528, dtype=torch.uint8),
            indices=indices,
            seq_lens=torch.full((tokens,), 8192, dtype=torch.int32),
            workspace_buffer=torch.zeros(
                required if workspace_bytes is None else workspace_bytes,
                dtype=torch.uint8,
            ),
            runner=FakeRunner(),
            page_size=64,
            kv_cache_dim=528,
            qk_nope_head_dim=256,
            kv_lora_rank=512,
            qk_rope_head_dim=0,
            sm_scale=0.125,
            skip_softmax_threshold_scale_factor=None,
        )
        return captured, required

    def test_holes_do_not_exclude_kpool_tail(self):
        indices = torch.full((1, 2051), -1, dtype=torch.int32)
        indices[0, :2044] = torch.arange(1, 2045, dtype=torch.int32)
        indices[0, 2048:] = torch.tensor([9000, 9001, 9002], dtype=torch.int32)
        captured, _ = self._run(indices)
        self.assertEqual(captured["topk_length"].tolist(), [2051])
        torch.testing.assert_close(captured["indices"][:, :2051], indices)

    def test_leading_holes_and_empty_rows(self):
        indices = torch.tensor([[-1, 2, 3, -1], [-1, -1, -1, -1]], dtype=torch.int32)
        captured, _ = self._run(indices)
        self.assertEqual(captured["topk_length"].tolist(), [3, 0])

    def test_exact_and_one_byte_short_workspace(self):
        indices = torch.tensor([[2, -1]], dtype=torch.int32)
        for heads in (8, 24, 32):
            with self.subTest(heads=heads):
                captured, required = self._run(indices, heads=heads)
                expected_heads = 8 if heads == 8 else 32
                self.assertEqual(
                    captured["mid_out"].shape, (1, expected_heads, 34, 512)
                )
                self.assertEqual(captured["mid_lse"].shape, (1, expected_heads, 34))
                with self.assertRaisesRegex(ValueError, "workspace is too small"):
                    self._run(indices, heads=heads, workspace_bytes=required - 1)


if __name__ == "__main__":
    unittest.main()
