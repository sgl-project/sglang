"""Qwen3-Next full-attention decode through aiter's fused
``paged_attention_output_gate_group_fp8_quant``.

Two things are checked, and they fail for different reasons:

1. **The kernel computes the right thing.** A single-GPU comparison against an
   fp32 eager reference (dequantize the fp8 KV, softmax, multiply by
   ``sigmoid(gate)``) across both kernel bodies -- the short-context one and
   the long-context one, which ``max_context > 32768`` selects.

2. **The in-tree caller declines instead of raising.** Every ineligible case
   (prefill, no gate, a non-aiter backend, an aiter build without the op) must
   return ``None`` so ``Qwen3HybridAttentionDecoderLayer.self_attention`` runs
   its existing three-launch path. A crash here would take down any
   non-gfx950 deployment, so it is tested with stubs and runs on any machine,
   GPU or not.

Requires one gfx950 GPU for (1); skipped elsewhere.

    python -m unittest test.registered.amd.test_paged_attention_output_gate
"""

import unittest

import torch

from sglang.srt.layers.attention.aiter_paged_attention_output_gate import (
    fused_gated_decode_attention,
)
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=120, suite="stage-c-test-large-8-gpu-amd-mi35x")

HEAD_DIM = 256
# The op's own crossover between its two bodies, in context tokens.
SHORT_CONTEXT_MAX = 32768


def _is_gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        arch = torch.cuda.get_device_properties(0).gcnArchName
    except Exception:
        return False
    return arch.split(":", 1)[0] == "gfx950"


def _have_op() -> bool:
    try:
        import aiter.ops.triton.attention.paged_attention_output_gate  # noqa: F401
    except ImportError:
        return False
    return True


def _reference(
    query, key_cache, value_cache, kv_indptr, kv_indices, gate, k_scale, v_scale
):
    """fp32 eager: dequantized paged attention, then the sigmoid gate."""
    rows, heads, head_dim = query.shape
    out = torch.empty(rows, heads * head_dim, device=query.device, dtype=torch.float32)
    for t in range(rows):
        slots = kv_indices[kv_indptr[t] : kv_indptr[t + 1]].long()
        k = key_cache[slots, 0, :].to(torch.float32) * k_scale
        v = value_cache[slots, 0, :].to(torch.float32) * v_scale
        logits = (query[t].to(torch.float32) @ k.T) * (head_dim**-0.5)
        out[t] = (torch.softmax(logits, dim=-1) @ v).reshape(-1)
    return out * torch.sigmoid(gate.to(torch.float32))


class TestPagedAttentionOutputGateKernel(CustomTestCase):
    @unittest.skipUnless(
        _is_gfx950() and _have_op(), "needs a gfx950 GPU and a recent aiter"
    )
    def test_matches_fp32_reference(self):
        from aiter.ops.triton.attention.paged_attention_output_gate import (
            paged_attention_output_gate_group_fp8_quant,
            paged_attention_output_gate_supported,
        )

        torch.manual_seed(1234)
        device = torch.device("cuda", 0)
        # (heads, decode rows, context, max_context). The last column picks the
        # kernel body; the final case is above SHORT_CONTEXT_MAX so the
        # long-context body is exercised with a real long context, not just a
        # large capacity hint.
        cases = [
            (4, 1, 1024, 8192),
            (4, 8, 4096, 8192),
            (16, 3, 2048, 8192),
            (4, 5, 8192, 65536),
            (16, 2, 40000, 65536),
        ]
        for heads, rows, context, max_context in cases:
            with self.subTest(heads=heads, rows=rows, context=context):
                lengths = [max(1, context - 7 * i) for i in range(rows)]
                total = sum(lengths)
                pages = total + 64
                query = (torch.randn(rows, heads, HEAD_DIM, device=device) * 0.3).to(
                    torch.bfloat16
                )
                gate = torch.randn(rows, heads * HEAD_DIM, device=device).to(
                    torch.bfloat16
                )
                key_cache = (torch.randn(pages, 1, HEAD_DIM, device=device) * 0.4).to(
                    torch.float8_e4m3fn
                )
                value_cache = (torch.randn(pages, 1, HEAD_DIM, device=device) * 0.4).to(
                    torch.float8_e4m3fn
                )
                kv_indptr = torch.zeros(rows + 1, dtype=torch.int32, device=device)
                kv_indptr[1:] = torch.tensor(lengths, device=device).cumsum(0)
                # Scattered slots, as a real paged cache hands them over.
                kv_indices = torch.randperm(pages, device=device)[:total].to(
                    torch.int32
                )
                k_scale = torch.tensor(0.7, device=device)
                v_scale = torch.tensor(1.3, device=device)

                ok, reason = paged_attention_output_gate_supported(
                    query, key_cache, value_cache, gate, None
                )
                self.assertTrue(ok, reason)

                out, quantized, scales = paged_attention_output_gate_group_fp8_quant(
                    query,
                    key_cache,
                    value_cache,
                    kv_indptr,
                    kv_indices,
                    gate,
                    scale=HEAD_DIM**-0.5,
                    max_context=max_context,
                    k_scale=k_scale,
                    v_scale=v_scale,
                    quant_dtype=None,
                )
                # SGLang asks for no FP8 epilogue: o_proj takes the BF16 tensor.
                self.assertEqual(out.dtype, torch.bfloat16)
                self.assertIsNone(quantized)
                self.assertIsNone(scales)

                ref = _reference(
                    query,
                    key_cache,
                    value_cache,
                    kv_indptr,
                    kv_indices,
                    gate,
                    k_scale,
                    v_scale,
                )
                err = (out.to(torch.float32) - ref).abs().max().item()
                # A few bf16 ULP of the largest output element. The kernel
                # accumulates in fp32 and rounds once, so this is tight.
                tol = 1.0e-2 * max(ref.abs().max().item(), 1.0e-3)
                self.assertLess(err, tol, f"max abs error {err} >= {tol}")

    @unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
    def test_backend_exposes_save_kv_cache(self):
        """The fused caller writes the KV cache through the backend, not a copy
        of its dispatch chain. Import is ROCm-only."""
        from sglang.srt.utils import is_hip

        if not is_hip():
            self.skipTest("aiter backend is ROCm only")
        from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend

        self.assertTrue(callable(getattr(AiterAttnBackend, "save_kv_cache", None)))


class _ForwardMode:
    def __init__(self, decode: bool):
        self._decode = decode

    def is_decode(self) -> bool:
        return self._decode


class _ForwardBatch:
    def __init__(self, decode: bool = True):
        self.forward_mode = _ForwardMode(decode)


class TestPagedAttentionOutputGateFallback(CustomTestCase):
    """The ineligible cases must return None, never raise. No GPU needed."""

    def test_no_gate(self):
        self.assertIsNone(
            fused_gated_decode_attention(None, None, None, None, None, _ForwardBatch())
        )

    def test_not_decode(self):
        self.assertIsNone(
            fused_gated_decode_attention(
                None,
                None,
                None,
                None,
                torch.zeros(1, 1024),
                _ForwardBatch(decode=False),
            )
        )

    def test_no_forward_context(self):
        """A decode batch with no published forward context, i.e. nowhere to
        read the attention backend from."""
        from sglang.srt.model_executor.forward_context import set_forward_context

        previous = set_forward_context(None)
        try:
            self.assertIsNone(
                fused_gated_decode_attention(
                    None, None, None, None, torch.zeros(1, 1024), _ForwardBatch()
                )
            )
        finally:
            set_forward_context(previous)

    def test_non_aiter_backend(self):
        """A published backend that is not AiterAttnBackend -- every CUDA
        deployment, and every ROCm one not using --attention-backend aiter."""
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )

        with forward_context(ForwardContext(attn_backend=object())):
            self.assertIsNone(
                fused_gated_decode_attention(
                    None, None, None, None, torch.zeros(1, 1024), _ForwardBatch()
                )
            )


if __name__ == "__main__":
    unittest.main()
