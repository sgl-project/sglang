"""
Unit tests for the QKNormRope compilable region.

Validates that the compiled-region path (RMSNorm module + forward_native with
KV scatter) produces the same results as the CUDA-kernel path (RMSNorm module
+ forward_cuda with KV scatter).  This is the numerical correctness check for
the `_qk_norm_rope_kv` method added to Qwen3MoeAttention.
"""

import unittest

import torch

from sglang.jit_kernel.rope import FusedSetKVBufferArg
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase
from sgl_kernel.testing.rotary_embedding import (
    FlashInferRotaryEmbedding,
    FusedSetKVBufferArg as TestingFusedSetKVBufferArg,
    MHATokenToKVPool,
    RotaryEmbedding as TestingRotaryEmbedding,
    create_inputs,
)

register_cuda_ci(est_time=15, suite="stage-b-test-large-1-gpu")

# Qwen3 MoE 235B-like attention config
HEAD_SIZE = 128
NUM_Q_HEADS = 16
NUM_KV_HEADS = 4
ROPE_BASE = 1_000_000.0
MAX_POS = 40960
RMS_EPS = 1e-6

POOL_SIZE = MHATokenToKVPool.KV_POOL_SIZE

# BF16 tolerances — CUDA kernels compute cos/sin in FP32 internally while
# forward_native casts to BF16 before the multiply, giving up to 1 BF16 ULP
# of error.
ATOL = 4e-2
RTOL = 4e-2


def _qk_norm_rope(q, k, positions, q_norm, k_norm, head_dim, rope_fn, fused_kv_arg):
    """Reimplementation of _qk_norm_rope_kv for testing.

    Mirrors the production code: calls q_norm/k_norm modules directly,
    then *rope_fn* (forward_cuda or forward_native) for the rope + kv step.
    """
    q = q_norm(q.reshape(-1, head_dim)).view(q.shape)
    k = k_norm(k.reshape(-1, head_dim)).view(k.shape)
    q, k = rope_fn(positions, q, k, fused_set_kv_buffer_arg=fused_kv_arg)
    return q, k


class TestQKNormRopeVsCudaKernel(CustomTestCase):
    """Numerics: compare the native path (RMSNorm + forward_native + KV scatter)
    against the CUDA kernel path (RMSNorm + forward_cuda + KV scatter).

    Both paths use the same RMSNorm modules (which dispatch via
    MultiPlatformOp), so the norm results are identical.  The difference
    is in the RoPE + KV-scatter implementation: forward_native (pure
    PyTorch, used by torch.compile) vs forward_cuda (JIT CUDA kernel).
    """

    def setUp(self):
        self.device = get_device()

        self.q_norm = RMSNorm(HEAD_SIZE, eps=RMS_EPS).to(self.device)
        self.k_norm = RMSNorm(HEAD_SIZE, eps=RMS_EPS).to(self.device)

        rope_cfg = dict(
            head_size=HEAD_SIZE,
            rotary_dim=HEAD_SIZE,
            max_position_embeddings=MAX_POS,
            base=ROPE_BASE,
            is_neox_style=True,
            dtype=torch.bfloat16,
        )
        self.rope_cuda = FlashInferRotaryEmbedding(**rope_cfg).to(self.device)
        self.rope_native = TestingRotaryEmbedding(**rope_cfg).to(self.device)
        self.rope_native.cos_sin_cache = self.rope_cuda.cos_sin_cache

    def _run_comparison(self, batch_size, seq_len, with_kv_scatter=True):
        inputs = create_inputs(
            head_size=HEAD_SIZE,
            batch_size=batch_size,
            seq_len=seq_len,
            device=self.device,
            dtype=torch.bfloat16,
            num_q_heads=NUM_Q_HEADS,
            num_kv_heads=NUM_KV_HEADS,
        )
        positions = inputs["pos_ids"]
        query = inputs["query"]
        key = inputs["key"]
        value = inputs["value"]
        cache_loc = inputs["out_cache_loc"]

        # --- Reference: RMSNorm + forward_cuda (JIT RoPE + KV scatter) ---
        if with_kv_scatter:
            k_buf_ref = torch.zeros(
                POOL_SIZE, NUM_KV_HEADS, HEAD_SIZE,
                dtype=torch.bfloat16, device=self.device,
            )
            v_buf_ref = torch.zeros_like(k_buf_ref)
            cuda_arg = TestingFusedSetKVBufferArg(
                value=value.clone(),
                k_buffer=k_buf_ref.view(POOL_SIZE, -1),
                v_buffer=v_buf_ref.view(POOL_SIZE, -1),
                cache_loc=cache_loc,
            )
        else:
            cuda_arg = None

        q_ref, k_ref = _qk_norm_rope(
            query.clone(), key.clone(), positions,
            self.q_norm, self.k_norm, HEAD_SIZE,
            self.rope_cuda.forward_cuda, cuda_arg,
        )

        # --- Test: RMSNorm + forward_native (compiled-region path) ---
        if with_kv_scatter:
            k_buf_test = torch.zeros(
                POOL_SIZE, NUM_KV_HEADS, HEAD_SIZE,
                dtype=torch.bfloat16, device=self.device,
            )
            v_buf_test = torch.zeros_like(k_buf_test)
            native_arg = FusedSetKVBufferArg(
                value=value.clone(),
                k_buffer=k_buf_test.view(POOL_SIZE, -1),
                v_buffer=v_buf_test.view(POOL_SIZE, -1),
                cache_loc=cache_loc,
            )
        else:
            native_arg = None

        q_test, k_test = _qk_norm_rope(
            query.clone(), key.clone(), positions,
            self.q_norm, self.k_norm, HEAD_SIZE,
            self.rope_native.forward_native, native_arg,
        )

        # Compare q and k (normed + rotated)
        torch.testing.assert_close(
            q_test, q_ref, atol=ATOL, rtol=RTOL,
            msg="query mismatch: native path vs CUDA kernel",
        )
        torch.testing.assert_close(
            k_test, k_ref, atol=ATOL, rtol=RTOL,
            msg="key mismatch: native path vs CUDA kernel",
        )

        if with_kv_scatter:
            torch.testing.assert_close(
                k_buf_test.view(POOL_SIZE, -1)[cache_loc],
                k_buf_ref.view(POOL_SIZE, -1)[cache_loc],
                atol=ATOL, rtol=RTOL,
                msg="k_buffer mismatch: native path vs CUDA kernel",
            )
            self.assertTrue(
                torch.equal(
                    v_buf_test.view(POOL_SIZE, -1)[cache_loc],
                    v_buf_ref.view(POOL_SIZE, -1)[cache_loc],
                ),
                "v_buffer mismatch: scatter must be bitwise exact",
            )

    # -- decode (single-token) tests ------------------------------------

    def test_decode_bs1(self):
        """Single-token decode — most common CUDA graph bucket."""
        self._run_comparison(batch_size=1, seq_len=1)

    def test_decode_bs32(self):
        """32-token decode batch."""
        self._run_comparison(batch_size=32, seq_len=1)

    def test_decode_bs128(self):
        """128-token decode batch."""
        self._run_comparison(batch_size=128, seq_len=1)

    def test_decode_bs512(self):
        """512-token decode batch — large CUDA graph bucket."""
        self._run_comparison(batch_size=512, seq_len=1)

    # -- decode without KV scatter (norm + rope only) -------------------

    def test_decode_bs1_no_kv_scatter(self):
        self._run_comparison(batch_size=1, seq_len=1, with_kv_scatter=False)

    def test_decode_bs32_no_kv_scatter(self):
        self._run_comparison(batch_size=32, seq_len=1, with_kv_scatter=False)

    # -- prefill tests --------------------------------------------------

    def test_prefill_short(self):
        """Short prefill (2 seqs x 512 tokens)."""
        self._run_comparison(batch_size=2, seq_len=512)

    def test_prefill_short_no_kv_scatter(self):
        self._run_comparison(batch_size=2, seq_len=512, with_kv_scatter=False)


class TestQKNormRopeHeadDim64(CustomTestCase):
    """Same numerics test but with head_dim=64 (common in smaller MoE models)."""

    HD = 64
    Q_HEADS = 8
    KV_HEADS = 8

    def setUp(self):
        self.device = get_device()

        self.q_norm = RMSNorm(self.HD, eps=RMS_EPS).to(self.device)
        self.k_norm = RMSNorm(self.HD, eps=RMS_EPS).to(self.device)

        rope_cfg = dict(
            head_size=self.HD,
            rotary_dim=self.HD,
            max_position_embeddings=4096,
            base=8000.0,
            is_neox_style=True,
            dtype=torch.bfloat16,
        )
        self.rope_cuda = FlashInferRotaryEmbedding(**rope_cfg).to(self.device)
        self.rope_native = TestingRotaryEmbedding(**rope_cfg).to(self.device)
        self.rope_native.cos_sin_cache = self.rope_cuda.cos_sin_cache

    def _run(self, batch_size, seq_len):
        inputs = create_inputs(
            head_size=self.HD,
            batch_size=batch_size,
            seq_len=seq_len,
            device=self.device,
            dtype=torch.bfloat16,
            num_q_heads=self.Q_HEADS,
            num_kv_heads=self.KV_HEADS,
        )
        positions = inputs["pos_ids"]
        query = inputs["query"]
        key = inputs["key"]
        value = inputs["value"]
        cache_loc = inputs["out_cache_loc"]

        k_buf_ref = torch.zeros(
            POOL_SIZE, self.KV_HEADS, self.HD,
            dtype=torch.bfloat16, device=self.device,
        )
        v_buf_ref = torch.zeros_like(k_buf_ref)
        cuda_arg = TestingFusedSetKVBufferArg(
            value=value.clone(),
            k_buffer=k_buf_ref.view(POOL_SIZE, -1),
            v_buffer=v_buf_ref.view(POOL_SIZE, -1),
            cache_loc=cache_loc,
        )
        q_ref, k_ref = _qk_norm_rope(
            query.clone(), key.clone(), positions,
            self.q_norm, self.k_norm, self.HD,
            self.rope_cuda.forward_cuda, cuda_arg,
        )

        k_buf_test = torch.zeros(
            POOL_SIZE, self.KV_HEADS, self.HD,
            dtype=torch.bfloat16, device=self.device,
        )
        v_buf_test = torch.zeros_like(k_buf_test)
        native_arg = FusedSetKVBufferArg(
            value=value.clone(),
            k_buffer=k_buf_test.view(POOL_SIZE, -1),
            v_buffer=v_buf_test.view(POOL_SIZE, -1),
            cache_loc=cache_loc,
        )
        q_test, k_test = _qk_norm_rope(
            query.clone(), key.clone(), positions,
            self.q_norm, self.k_norm, self.HD,
            self.rope_native.forward_native, native_arg,
        )

        torch.testing.assert_close(q_test, q_ref, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(k_test, k_ref, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(
            k_buf_test.view(POOL_SIZE, -1)[cache_loc],
            k_buf_ref.view(POOL_SIZE, -1)[cache_loc],
            atol=ATOL, rtol=RTOL,
        )
        self.assertTrue(
            torch.equal(
                v_buf_test.view(POOL_SIZE, -1)[cache_loc],
                v_buf_ref.view(POOL_SIZE, -1)[cache_loc],
            ),
        )

    def test_decode_bs1(self):
        self._run(batch_size=1, seq_len=1)

    def test_decode_bs32(self):
        self._run(batch_size=32, seq_len=1)

    def test_prefill(self):
        self._run(batch_size=2, seq_len=512)


if __name__ == "__main__":
    unittest.main(verbosity=3)
