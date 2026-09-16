"""Unit tests for the moonmath MLA attention backend.

Two layers of testing:
  1. Eligibility: _decode_eligible() / _verify_eligible() gate routing. Pure
     Python -- head counts, KV dtype, forward mode, draft-window bounds -- so it
     runs anywhere, including where moonmath_amd is absent: the kernel
     package is stubbed into sys.modules for the construction. No GPU.
  2. Correctness: real A16W8 kernel output vs fp32 reference on dequantized
     fp8 KV. Skipped unless a ROCm GPU and moonmath_amd are both present.

Only (2) needs moonmath_amd, which is not on PyPI. (1) must stay runnable
without it, so a CI lane that cannot install the kernel package still covers the
routing decisions -- which is where a regression would silently change which
kernel serves a shape.
"""

import contextlib
import sys
import types
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


def _fake_aiter_init(self, model_runner):
    """Stand-in for AiterAttnBackend.__init__: set only what the subclass reads."""
    self.use_mla = model_runner.use_mla
    self.kv_cache_dtype = model_runner.kv_cache_dtype
    self.num_head = model_runner.num_head
    self.token_to_kv_pool = model_runner.token_to_kv_pool


def _has_moonmath():
    try:
        import moonmath_amd.mla  # noqa: F401

        return True
    except Exception:
        return False


@contextlib.contextmanager
def _mocked_mla_module():
    """Yield a mock standing in for ``moonmath_amd.mla``.

    ``patch`` cannot patch a module that does not exist, so where the kernel
    package is absent (it is not on PyPI) a stub package is installed into
    ``sys.modules`` for the duration instead. The backend imports it inside
    ``__init__`` -- "fail fast if not installed" -- so that is the only thing
    the stub has to satisfy for the gating tests.
    """
    if _has_moonmath():
        with patch("moonmath_amd.mla") as mock_mla:
            yield mock_mla
        return
    pkg = types.ModuleType("moonmath_amd")
    mock_mla = MagicMock()
    pkg.mla = mock_mla
    with patch.dict(
        sys.modules,
        {"moonmath_amd": pkg, "moonmath_amd.mla": mock_mla},
    ):
        yield mock_mla


def _make_backend(kv_cache_dtype=None, use_mla=True):
    """Construct a real MoonmathMLABackend with the aiter base __init__ stubbed.

    The kernel selection under test is `__init__`'s own, not a copy of it: only
    the base class and the moonmath_amd import are replaced.
    """
    from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend
    from sglang.srt.layers.attention.moonmath_mla_backend import MoonmathMLABackend

    runner = MagicMock()
    runner.use_mla = use_mla
    runner.kv_cache_dtype = kv_cache_dtype
    runner.num_head = 16
    runner.device = "cpu"

    with (
        patch.object(AiterAttnBackend, "__init__", _fake_aiter_init),
        _mocked_mla_module(),
    ):
        backend = MoonmathMLABackend(runner)

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

    def test_head_count_bound(self):
        """The kernel takes H up to 128 (DSV3 at TP1); past it falls back to aiter."""
        backend = _make_backend(kv_cache_dtype=torch.float8_e4m3fnuz)
        fb = _make_fb(batch_size=4, forward_mode="decode")
        q = torch.zeros(1, dtype=torch.bfloat16)
        self.assertTrue(backend._decode_eligible(q, _make_layer(q_head_num=128), fb))
        self.assertFalse(backend._decode_eligible(q, _make_layer(q_head_num=144), fb))

    def test_reject_bf16_kv(self):
        """The A16W8 kernels read an fp8 pool; a bf16 KV cache falls back."""
        backend = _make_backend(kv_cache_dtype=torch.bfloat16)
        layer = _make_layer(q_head_num=16)
        fb = _make_fb(batch_size=4, forward_mode="decode")
        q = torch.zeros(1, dtype=torch.bfloat16)
        self.assertFalse(backend._enabled)
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

    def test_reject_batch_past_kernel_row_slices(self):
        """The kernel launches B * ceil(q_len * H / 96) row slices, at most 304.

        A larger batch must fall back to aiter instead of raising in the kernel.
        """
        backend = _make_backend(kv_cache_dtype=torch.float8_e4m3fnuz)
        layer = _make_layer(q_head_num=16)
        q = torch.zeros(1, dtype=torch.bfloat16)
        self.assertTrue(backend._decode_eligible(q, layer, _make_fb(batch_size=304)))
        self.assertFalse(backend._decode_eligible(q, layer, _make_fb(batch_size=305)))


class TestMoonmathMLAVerifyGate(unittest.TestCase):
    """Any draft window within the kernel's row-slice budget; past it falls back."""

    def _fb(self, q_len):
        spec = MagicMock()
        spec.num_tokens_per_req = q_len
        return _make_fb(batch_size=4, forward_mode="target_verify", spec_info=spec)

    def test_window_accepted(self):
        backend = _make_backend(kv_cache_dtype=torch.float8_e4m3fnuz)
        layer = _make_layer(q_head_num=12)  # Kimi-K3 at TP8
        q = torch.zeros(1, dtype=torch.bfloat16)
        for q_len in (2, 3, 4, 8, 16):
            self.assertTrue(backend._verify_eligible(q, layer, self._fb(q_len)))

    def test_window_past_row_slices_rejected(self):
        """4 * ceil(q_len * 12 / 96) row slices: 304 at q_len 608, 308 at 609."""
        backend = _make_backend(kv_cache_dtype=torch.float8_e4m3fnuz)
        layer = _make_layer(q_head_num=12)
        q = torch.zeros(1, dtype=torch.bfloat16)
        self.assertTrue(backend._verify_eligible(q, layer, self._fb(608)))
        self.assertFalse(backend._verify_eligible(q, layer, self._fb(609)))

    def test_rejects_package_without_unified_op(self):
        """An older moonmath_amd has a same-named op with another signature."""
        from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend
        from sglang.srt.layers.attention.moonmath_mla_backend import (
            MoonmathMLABackend,
        )

        runner = MagicMock()
        runner.kv_cache_dtype = torch.float8_e4m3fnuz
        old_mla = MagicMock(spec=["mla_decode_a16w8", "mla_decode_a16w8_paged_dev"])
        pkg = types.ModuleType("moonmath_amd")
        pkg.mla = old_mla
        with (
            patch.object(AiterAttnBackend, "__init__", _fake_aiter_init),
            patch.dict(sys.modules, {"moonmath_amd": pkg, "moonmath_amd.mla": old_mla}),
        ):
            with self.assertRaises(ImportError):
                MoonmathMLABackend(runner)

    def test_decode_gate_rejects_verify(self):
        """The two arms are disjoint: verify never reaches forward_decode."""
        backend = _make_backend(kv_cache_dtype=torch.float8_e4m3fnuz)
        layer = _make_layer(q_head_num=12)
        q = torch.zeros(1, dtype=torch.bfloat16)
        self.assertFalse(backend._decode_eligible(q, layer, self._fb(8)))


class TestMoonmathMLAKernelCorrectness(unittest.TestCase):
    """Correctness of the A16W8 decode kernel vs fp32 reference.

    The kernel carries Q in bf16 and computes softmax in fp32.  The fp32
    reference dequantizes the same fp8 KV bytes and runs the absorbed-MLA
    decode math in full fp32.  Relative error must stay below 1e-2.
    """

    @unittest.skipUnless(
        torch.cuda.is_available() and _has_moonmath(),
        "Requires ROCm GPU and moonmath_amd",
    )
    def test_decode_matches_fp32_reference(self):
        import math

        import moonmath_amd.mla as mla

        DEV = "cuda"
        FP8 = torch.float8_e4m3fnuz
        KV_LAT = 512
        ROPE = 64
        KV_DIM = KV_LAT + ROPE
        SCALE = 1.0 / math.sqrt(KV_DIM)

        # (B, S, q_len, H): plain decode, then Kimi-K3's TP8 verify window.
        for B, S, q_len, H in [(1, 128, 1, 16), (2, 256, 1, 16), (2, 256, 4, 12)]:
            with self.subTest(B=B, S=S, q_len=q_len, H=H):
                torch.manual_seed(42 + B * 1000 + S + q_len)
                T = B * q_len

                q_lat = torch.randn(T, H, KV_LAT, dtype=torch.bfloat16, device=DEV)
                q_pe = torch.randn(T, H, ROPE, dtype=torch.bfloat16, device=DEV)

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

                out = torch.empty(T, H, KV_LAT, dtype=torch.bfloat16, device=DEV)
                mla.mla_decode_a16w8(
                    q_lat,
                    q_pe,
                    kv_pool,
                    out,
                    seq_lens,
                    kv_indices,
                    kv_indptr,
                    SCALE,
                    1.0,
                )
                torch.cuda.synchronize()

                # Draft position t attends KV [0, S - (q_len - 1 - t)).
                ref = torch.empty(T, H, KV_LAT, dtype=torch.float32, device=DEV)
                for b in range(B):
                    for t in range(q_len):
                        n = S - (q_len - 1 - t)
                        c, k = c_refs[b][:n], k_refs[b][:n]
                        row = b * q_len + t
                        ql, qp = q_lat[row].float(), q_pe[row].float()
                        scores = (ql @ c.t() + qp @ k.t()) * SCALE
                        ref[row] = torch.softmax(scores, dim=-1) @ c

                relerr = (out.float() - ref).abs().max().item() / ref.abs().max().item()
                self.assertLess(relerr, 1e-2, f"relerr={relerr:.3e}")


if __name__ == "__main__":
    unittest.main()
