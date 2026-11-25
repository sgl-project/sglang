import sys
import unittest
from types import ModuleType
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.flash_mla_sm120 import (
    _GLM_DSA_MODEL_ARCHS,
    _validate_flashinfer_sparse_mla_backend,
    flashinfer_sparse_mla_forward,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


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
        for model_arch in _GLM_DSA_MODEL_ARCHS:
            with self.subTest(model_arch=model_arch):
                self.assertTrue(
                    self._validate(
                        "flashinfer_sparse_mla",
                        "flashinfer_sparse_mla",
                        model_arch,
                    )
                )

    def test_accepts_glm53_flash_architectures(self):
        # GLM-5.3-Flash (glm5_next) shares the DSA + flash_mla_sm120 path with
        # the GLM-MoE-DSA checkpoints. Without its architectures the SM120
        # auto-selection raises from the backend constructor at startup.
        for model_arch in (
            "Glm5NextForConditionalGeneration",
            "Glm5NextForConditionalGenerationNextN",
        ):
            with self.subTest(model_arch=model_arch):
                self.assertIn(model_arch, _GLM_DSA_MODEL_ARCHS)
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

    def test_accepts_triton_sparse_mla_for_prefill(self):
        # triton_sparse_mla is selectable for prefill on this platform while
        # flashinfer stays the decode kernel; the allowlist widened for exactly
        # that pair and no other, so a third backend must still be refused.
        self.assertTrue(self._validate("triton_sparse_mla", "flashinfer_sparse_mla"))
        with self.assertRaisesRegex(ValueError, "only flashinfer_sparse_mla"):
            self._validate("triton_sparse_mla", "trtllm")

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


class TestFlashInferSparseMLAPrefillWidthPadding(unittest.TestCase):
    """With index_kpool > 1, kpool widens top-k to index_topk + index_kpool - 1
    (2051 for GLM-5.3-Flash). Prefill requires an aligned width (padded to 128
    with -1 sentinels), whereas decode must pass indices through untouched so
    decode CUDA graphs never alias the shared grow-only pad buffer."""

    def _adapter(self):
        from sglang.srt.layers.attention.dsa_backend import (
            DeepseekSparseAttnBackend,
        )

        backend = DeepseekSparseAttnBackend.__new__(DeepseekSparseAttnBackend)
        backend._q8kv8_topk_pad_buf = None
        backend._q8kv8_topk_pad_width = None
        backend.workspace_buffer = torch.zeros(16, dtype=torch.uint8)
        backend.physical_page_size = 64
        backend.kv_cache_dim = 528
        backend.qk_nope_head_dim = 256
        backend.kv_lora_rank = 512
        backend.qk_rope_head_dim = 0
        return backend

    def _dispatch(
        self,
        indices: torch.Tensor,
        *,
        pad_topk_width: bool,
        capturing: bool = False,
    ):
        captured = {}

        def fake_forward(**kwargs):
            captured.update(kwargs)
            return torch.zeros(1)

        backend = self._adapter()
        with (
            patch(
                "sglang.kernels.ops.attention.flash_mla_sm120."
                "flashinfer_sparse_mla_forward",
                fake_forward,
            ),
            patch(
                "torch.cuda.is_current_stream_capturing",
                return_value=capturing,
            ),
        ):
            backend._forward_flashinfer_sparse_mla(
                q_all=torch.zeros((4, 16, 512), dtype=torch.bfloat16),
                kv_cache=torch.zeros((8, 1, 64, 528), dtype=torch.uint8),
                page_table_1=indices,
                seq_lens=torch.full((4,), indices.shape[-1], dtype=torch.int32),
                sm_scale=0.0625,
                skip_softmax_threshold_scale_factor=None,
                pad_topk_width=pad_topk_width,
            )
        return backend, indices, captured["indices"]

    def test_prefill_pads_unaligned_kpool_width_to_128(self):
        indices = torch.arange(2051, dtype=torch.int32).repeat(4, 1)
        backend, original, sent = self._dispatch(indices, pad_topk_width=True)

        self.assertEqual(sent.shape[-1], 2176)
        self.assertEqual(sent.shape[-1] % 64, 0)
        self.assertTrue(torch.equal(sent[:, :2051], original))
        self.assertTrue((sent[:, 2051:] == -1).all())
        self.assertIsNotNone(backend._q8kv8_topk_pad_buf)

    def test_prefill_leaves_aligned_width_untouched(self):
        indices = torch.arange(2048, dtype=torch.int32).repeat(4, 1)
        backend, original, sent = self._dispatch(indices, pad_topk_width=True)

        self.assertIs(sent, original)
        self.assertIsNone(backend._q8kv8_topk_pad_buf)

    def test_decode_leaves_unaligned_width_untouched(self):
        indices = torch.arange(2051, dtype=torch.int32).repeat(4, 1)
        backend, original, sent = self._dispatch(indices, pad_topk_width=False)

        self.assertIs(sent, original)
        self.assertIsNone(backend._q8kv8_topk_pad_buf)

    @unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
    def test_prefill_capture_avoids_shared_grow_only_buffer(self):
        # The capture branch is only taken for CUDA inputs (page_table_1.is_cuda),
        # so this case needs a real device tensor even though the forward is faked.
        indices = torch.arange(2051, dtype=torch.int32, device="cuda").repeat(4, 1)
        backend, original, sent = self._dispatch(
            indices, pad_topk_width=True, capturing=True
        )

        self.assertEqual(sent.shape[-1], 2176)
        self.assertTrue(torch.equal(sent[:, :2051], original))
        self.assertTrue((sent[:, 2051:] == -1).all())
        self.assertIsNone(backend._q8kv8_topk_pad_buf)


@unittest.skipUnless(
    torch.cuda.is_available()
    and torch.version.hip is None
    and torch.cuda.get_device_capability()[0] == 12,
    "Requires NVIDIA SM120/SM121 GPU",
)
class TestFlashInferSparseMLAKPoolPaddingNumerics(unittest.TestCase):
    """Verify 2051 -> 2176 -1 padding against non-zero packed FP8 KV on SM120,
    including NaN-poisoned unused slot 0 as a negative control."""

    @staticmethod
    def _make_packed_fp8_kv(
        num_pages: int, page_size: int, device: torch.device, seed: int
    ):
        gen = torch.Generator(device=device).manual_seed(seed)
        total_slots = num_pages * page_size
        fp8_vals = (
            torch.randn((total_slots, 512), generator=gen, device=device) * 0.5
        ).to(torch.float8_e4m3fn)
        scales = (
            torch.rand((total_slots, 4), generator=gen, device=device) * 0.5 + 0.25
        ).to(torch.float32)
        # Poison slot 0 (unused by valid indices) with NaN to catch unmasked
        # sentinel reads.
        scales[0].fill_(float("nan"))

        packed = torch.empty((total_slots, 528), dtype=torch.uint8, device=device)
        packed[:, :512] = fp8_vals.view(torch.uint8)
        packed[:, 512:528] = scales.view(torch.uint8)
        return packed.view(num_pages, 1, page_size, 528), fp8_vals, scales

    @staticmethod
    def _ref_attention_fp64(
        q: torch.Tensor,
        fp8_vals: torch.Tensor,
        scales: torch.Tensor,
        indices: torch.Tensor,
        sm_scale: float,
    ) -> torch.Tensor:
        kv = (
            fp8_vals.to(torch.float64).view(-1, 4, 128)
            * scales.to(torch.float64).unsqueeze(-1)
        ).reshape(-1, 512)
        valid = indices >= 0
        safe_idx = indices.clamp(min=1).long()
        k = kv[safe_idx]  # [T, topk, 512]
        scores = torch.einsum("thd,tkd->thk", q.to(torch.float64), k) * sm_scale
        scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        return torch.einsum("thk,tkd->thd", probs, k)

    def test_prefill_2051_padded_to_2176_matches_fp64_ref(self):
        # Dense case: all 2051 columns are real indices in [1, 4095]. The last
        # three (2048..2050) are exactly the kpool tail columns, so a kernel
        # that silently truncates the input back to 2048 changes the result.
        self._check_padding_case(sparse_mask=False)
        # Sparse case: -1 entries must be skipped, never resolved to slot 0.
        self._check_padding_case(sparse_mask=True)

    def _check_padding_case(self, *, sparse_mask: bool):
        device = torch.device("cuda")
        page_size = 64
        num_pages = 64  # 4096 slots
        num_tokens = 128
        num_heads = 16
        sm_scale = 512.0**-0.5

        kv_cache, fp8_vals, scales = self._make_packed_fp8_kv(
            num_pages, page_size, device, seed=42
        )
        gen = torch.Generator(device=device).manual_seed(43)
        q = torch.randn(
            (num_tokens, num_heads, 512),
            generator=gen,
            device=device,
            dtype=torch.bfloat16,
        )
        # Valid indices in [1, 4095]: slot 0 is NaN-poisoned, so any -1 that is
        # resolved to slot 0 instead of being masked out poisons the output.
        idx_2051 = torch.randint(
            1,
            num_pages * page_size,
            (num_tokens, 2051),
            generator=gen,
            device=device,
            dtype=torch.int32,
        )
        if sparse_mask:
            idx_2051[:, 1536:] = -1
        idx_2176 = torch.nn.functional.pad(idx_2051, (0, 2176 - 2051), value=-1)

        workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)
        out = flashinfer_sparse_mla_forward(
            q=q,
            kv_cache=kv_cache,
            indices=idx_2176,
            seq_lens=torch.full(
                (num_tokens,),
                2051 if sparse_mask else num_pages * page_size,
                dtype=torch.int32,
                device=device,
            ),
            workspace_buffer=workspace,
            page_size=page_size,
            kv_cache_dim=528,
            qk_nope_head_dim=256,
            kv_lora_rank=512,
            qk_rope_head_dim=0,
            sm_scale=sm_scale,
            skip_softmax_threshold_scale_factor=None,
        )
        ref = self._ref_attention_fp64(q, fp8_vals, scales, idx_2051, sm_scale)

        self.assertTrue(torch.isfinite(out).all(), f"sparse_mask={sparse_mask}")
        rel_err = (out.to(torch.float64) - ref).norm() / ref.norm()
        self.assertLess(rel_err.item(), 0.05, f"sparse_mask={sparse_mask}")


if __name__ == "__main__":
    unittest.main()
