"""Unit tests for the moonmath MLA attention backend.

Two layers of testing:
  1. Eligibility: _decode_eligible() gate routing (no GPU, mocked kernel).
  2. Correctness: real A16W8 kernel output vs fp32 reference on dequantized
     fp8 KV (requires ROCm GPU + moonmath_attention installed).

Requires: moonmath_attention installed (the backend imports it at __init__).
Requires: AMD ROCm (torch.float8_e4m3fnuz is a ROCm-only dtype).
"""

import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")


def _make_layer(
    q_head_num=16,
    qk_head_dim=576,
    v_head_dim=512,
    tp_k_head_num=1,
    logit_cap=0,
    scaling=0.0884,
    k_scale=None,
):
    """Create a mock RadixAttention layer with MLA defaults."""
    layer = MagicMock()
    layer.tp_q_head_num = q_head_num
    layer.qk_head_dim = qk_head_dim
    layer.v_head_dim = v_head_dim
    layer.tp_k_head_num = tp_k_head_num
    layer.logit_cap = logit_cap
    layer.scaling = scaling
    layer.k_scale = k_scale
    return layer


def _make_fb(batch_size=4, forward_mode="decode", spec_info=None):
    """Create a mock ForwardBatch."""
    fb = MagicMock()
    fb.batch_size = batch_size
    fb.spec_info = spec_info
    mode = MagicMock()
    mode.is_decode.return_value = forward_mode == "decode"
    mode.is_target_verify.return_value = forward_mode == "target_verify"
    mode.is_extend.return_value = forward_mode == "extend"
    fb.forward_mode = mode
    return fb


def _make_backend(kv_cache_dtype=None, use_mla=True):
    """Create a MoonmathMLABackend with mocked dependencies."""
    with patch(
        "sglang.srt.layers.attention.aiter_backend.AiterAttnBackend.__init__"
    ), patch("moonmath_attention.mla") as mock_mla:
        mock_mla.mla_decode_a16w8_plan_parts_capped.return_value = 1
        from sglang.srt.layers.attention.moonmath_mla_backend import (
            MoonmathMLABackend,
        )

        runner = MagicMock()
        runner.use_mla = use_mla
        runner.device = torch.device("cuda")
        runner.model_config.context_len = 131072

        backend = MoonmathMLABackend.__new__(MoonmathMLABackend)
        backend._mla = mock_mla
        backend._mla_ok = use_mla
        backend._disabled = False
        backend._fp8_dtype = torch.float8_e4m3fnuz
        backend._fp8_kv = use_mla and (kv_cache_dtype == torch.float8_e4m3fnuz)
        backend._dec_parts = {}
        backend._mla_max_ctx = 131072
        backend._mla_seqlen_i32 = torch.zeros(8192, dtype=torch.int32, device="cpu")
        backend.kv_cache_dtype = kv_cache_dtype
        backend.forward_metadata = MagicMock()
        backend.forward_metadata.kv_indices = torch.zeros(1, dtype=torch.int32)
        backend.forward_metadata.kv_indptr = torch.zeros(1, dtype=torch.int32)
        return backend


class TestMoonmathMLAEligibility(unittest.TestCase):
    """Test _decode_eligible gates correctly for all input combinations."""

    def test_eligible_h16_fp8_decode(self):
        """The supported case: H=16, fp8 KV, pure decode, MLA dims."""
        backend = _make_backend(kv_cache_dtype=torch.float8_e4m3fnuz)
        layer = _make_layer(q_head_num=16)
        fb = _make_fb(batch_size=4, forward_mode="decode")
        q = torch.zeros(1, dtype=torch.bfloat16)
        self.assertTrue(backend._decode_eligible(q, layer, fb))

    def test_reject_h128(self):
        """H=128 (DSV3) should fall back to aiter."""
        backend = _make_backend(kv_cache_dtype=torch.float8_e4m3fnuz)
        layer = _make_layer(q_head_num=128)
        fb = _make_fb(batch_size=4, forward_mode="decode")
        q = torch.zeros(1, dtype=torch.bfloat16)
        self.assertFalse(backend._decode_eligible(q, layer, fb))

    def test_reject_bf16_kv(self):
        """bf16 KV should fall back to aiter (A16W8 requires fp8 KV)."""
        backend = _make_backend(kv_cache_dtype=torch.bfloat16)
        layer = _make_layer(q_head_num=16)
        fb = _make_fb(batch_size=4, forward_mode="decode")
        q = torch.zeros(1, dtype=torch.bfloat16)
        self.assertFalse(backend._decode_eligible(q, layer, fb))

    def test_reject_extend(self):
        """Prefill/extend should fall back to aiter."""
        backend = _make_backend(kv_cache_dtype=torch.float8_e4m3fnuz)
        layer = _make_layer(q_head_num=16)
        fb = _make_fb(batch_size=4, forward_mode="extend")
        q = torch.zeros(1, dtype=torch.bfloat16)
        self.assertFalse(backend._decode_eligible(q, layer, fb))

    def test_reject_spec_verify(self):
        """Spec-verify should fall back to aiter."""
        backend = _make_backend(kv_cache_dtype=torch.float8_e4m3fnuz)
        layer = _make_layer(q_head_num=16)
        fb = _make_fb(batch_size=4, spec_info=MagicMock())
        q = torch.zeros(1, dtype=torch.bfloat16)
        self.assertFalse(backend._decode_eligible(q, layer, fb))

    def test_reject_disabled(self):
        """SGLANG_MOONMATH_MLA_DISABLE=1 should fall back to aiter."""
        backend = _make_backend(kv_cache_dtype=torch.float8_e4m3fnuz)
        backend._disabled = True
        layer = _make_layer(q_head_num=16)
        fb = _make_fb(batch_size=4, forward_mode="decode")
        q = torch.zeros(1, dtype=torch.bfloat16)
        self.assertFalse(backend._decode_eligible(q, layer, fb))

    def test_reject_wrong_dims(self):
        """Non-MLA dims (e.g. head_dim=128) should fall back to aiter."""
        backend = _make_backend(kv_cache_dtype=torch.float8_e4m3fnuz)
        layer = _make_layer(q_head_num=16, qk_head_dim=128, v_head_dim=128)
        fb = _make_fb(batch_size=4, forward_mode="decode")
        q = torch.zeros(1, dtype=torch.bfloat16)
        self.assertFalse(backend._decode_eligible(q, layer, fb))

    def test_reject_non_mla(self):
        """Non-MLA model should fall back to aiter."""
        backend = _make_backend(kv_cache_dtype=torch.float8_e4m3fnuz, use_mla=False)
        layer = _make_layer(q_head_num=16)
        fb = _make_fb(batch_size=4, forward_mode="decode")
        q = torch.zeros(1, dtype=torch.bfloat16)
        self.assertFalse(backend._decode_eligible(q, layer, fb))


def _has_moonmath():
    try:
        import moonmath_attention.mla  # noqa: F401

        return True
    except Exception:
        return False


class TestMoonmathMLAKernelCorrectness(unittest.TestCase):
    """Correctness of the A16W8 decode kernel vs fp32 reference.

    The kernel carries Q in bf16 and computes softmax in fp32.  The fp32
    reference dequantizes the same fp8 KV bytes and runs the absorbed-MLA
    decode math in full fp32.  Relative error must stay below 1e-2.
    """

    @unittest.skipUnless(
        torch.cuda.is_available() and _has_moonmath(),
        "Requires ROCm GPU and moonmath_attention",
    )
    def test_decode_matches_fp32_reference(self):
        import math

        import moonmath_attention.mla as mla

        DEV = "cuda"
        FP8 = torch.float8_e4m3fnuz
        KV_LAT = 512
        ROPE = 64
        KV_DIM = KV_LAT + ROPE
        H = 16
        SCALE = 1.0 / math.sqrt(KV_DIM)

        for B, S in [(1, 128), (2, 256)]:
            with self.subTest(B=B, S=S):
                torch.manual_seed(42 + B * 1000 + S)

                q_lat = torch.randn(B, H, KV_LAT, dtype=torch.bfloat16, device=DEV)
                q_pe = torch.randn(B, H, ROPE, dtype=torch.bfloat16, device=DEV)

                num_slots = S * B + 1
                kv_pool = torch.zeros(num_slots, 1, KV_DIM, dtype=FP8, device=DEV)
                kv_indices = torch.zeros(S * B, dtype=torch.int32, device=DEV)
                kv_indptr = torch.zeros(B + 1, dtype=torch.int32, device=DEV)
                seq_lens = torch.full((B,), S, dtype=torch.int32, device=DEV)

                c_refs, k_refs = [], []
                for b in range(B):
                    off = b * S
                    slots = torch.arange(
                        off + 1, off + S + 1, device=DEV, dtype=torch.int32
                    )
                    kv_indices[off : off + S] = slots
                    kv_indptr[b + 1] = off + S
                    c = torch.randn(S, KV_LAT, device=DEV)
                    k = torch.randn(S, ROPE, device=DEV)
                    kv_pool[slots.long(), 0, :KV_LAT] = c.to(FP8)
                    kv_pool[slots.long(), 0, KV_LAT:] = k.to(FP8)
                    c_refs.append(kv_pool[slots.long(), 0, :KV_LAT].float())
                    k_refs.append(kv_pool[slots.long(), 0, KV_LAT:].float())

                out = torch.empty(B, H, KV_LAT, dtype=torch.bfloat16, device=DEV)
                parts = mla.mla_decode_a16w8_plan_parts_capped(B, H, S, KV_LAT)
                mla.mla_decode_a16w8_paged_dev(
                    q_lat,
                    q_pe,
                    kv_pool,
                    out,
                    seq_lens,
                    None,
                    kv_indices,
                    kv_indptr,
                    parts,
                    SCALE,
                    1.0,
                    1.0,
                )
                torch.cuda.synchronize()

                ref = torch.empty(B, H, KV_LAT, dtype=torch.float32, device=DEV)
                for b in range(B):
                    c, k = c_refs[b], k_refs[b]
                    ql, qp = q_lat[b].float(), q_pe[b].float()
                    scores = (ql @ c.t() + qp @ k.t()) * SCALE
                    ref[b] = torch.softmax(scores, dim=-1) @ c

                relerr = (out.float() - ref).abs().max().item() / ref.abs().max().item()
                self.assertLess(relerr, 1e-2, f"B={B} S={S}: relerr={relerr:.3e}")


if __name__ == "__main__":
    unittest.main()
