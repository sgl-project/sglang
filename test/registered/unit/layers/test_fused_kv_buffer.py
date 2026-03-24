"""
Unit tests for fused KV buffer helpers and forward_native KV scatter.

Validates the torch.compile-path changes that enable RoPE + KV cache scatter
fusion for SWA models (e.g. gpt-oss-20b).
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.jit_kernel.rope import FusedSetKVBufferArg
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.models.utils import (
    create_fused_set_kv_buffer_arg,
    enable_fused_set_kv_buffer,
)
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

register_cuda_ci(est_time=10, suite="stage-b-test-large-1-gpu")

# gpt-oss-20b-like layout: 24 layers, every 4th is full attention, rest SWA
NUM_LAYERS = 24
GLOBAL_INTERVAL = 4
FULL_LAYER_IDS = list(range(0, NUM_LAYERS, GLOBAL_INTERVAL))
SWA_LAYER_IDS = [i for i in range(NUM_LAYERS) if i not in FULL_LAYER_IDS]

HEAD_NUM = 8
HEAD_DIM = 128
KV_SIZE = 256
KV_SIZE_SWA = 128
BATCH_TOKENS = 16


def _make_swa_pool(device):
    return SWAKVPool(
        size=KV_SIZE,
        size_swa=KV_SIZE_SWA,
        page_size=1,
        dtype=torch.bfloat16,
        head_num=HEAD_NUM,
        head_dim=HEAD_DIM,
        swa_attention_layer_ids=SWA_LAYER_IDS,
        full_attention_layer_ids=FULL_LAYER_IDS,
        enable_kvcache_transpose=False,
        device=device,
    )


def _make_forward_batch(kv_pool, device):
    """Minimal ForwardBatch with fields needed by the fused KV helpers."""
    out_cache_loc = torch.randperm(KV_SIZE, dtype=torch.int64, device=device)[
        :BATCH_TOKENS
    ]
    out_cache_loc_swa = torch.randperm(KV_SIZE_SWA, dtype=torch.int64, device=device)[
        :BATCH_TOKENS
    ]
    return ForwardBatch(
        forward_mode=ForwardMode.DECODE,
        batch_size=BATCH_TOKENS,
        input_ids=torch.zeros(BATCH_TOKENS, dtype=torch.long, device=device),
        req_pool_indices=torch.zeros(BATCH_TOKENS, dtype=torch.long, device=device),
        seq_lens=torch.ones(BATCH_TOKENS, dtype=torch.int32, device=device),
        out_cache_loc=out_cache_loc,
        out_cache_loc_swa=out_cache_loc_swa,
        seq_lens_sum=BATCH_TOKENS,
        token_to_kv_pool=kv_pool,
    )


def _make_layer_mock(layer_id):
    """Lightweight stand-in for RadixAttention."""
    return SimpleNamespace(layer_id=layer_id, k_scale=None, v_scale=None)


class TestEnableFusedSetKvBuffer(CustomTestCase):
    """Test the is_compiled gate on enable_fused_set_kv_buffer."""

    def setUp(self):
        self.device = get_device()
        self.swa_pool = _make_swa_pool(self.device)
        self.fb = _make_forward_batch(self.swa_pool, self.device)

    def test_swa_pool_blocked_by_default(self):
        self.assertFalse(enable_fused_set_kv_buffer(self.fb, is_compiled=False))

    def test_swa_pool_allowed_when_compiled(self):
        self.assertTrue(enable_fused_set_kv_buffer(self.fb, is_compiled=True))

    def test_non_swa_pool_always_allowed(self):
        non_swa = SimpleNamespace(dtype=torch.bfloat16)
        fb = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=1,
            input_ids=torch.zeros(1, dtype=torch.long, device=self.device),
            req_pool_indices=torch.zeros(1, dtype=torch.long, device=self.device),
            seq_lens=torch.ones(1, dtype=torch.int32, device=self.device),
            out_cache_loc=torch.zeros(1, dtype=torch.long, device=self.device),
            seq_lens_sum=1,
            token_to_kv_pool=non_swa,
        )
        self.assertTrue(enable_fused_set_kv_buffer(fb, is_compiled=False))
        self.assertTrue(enable_fused_set_kv_buffer(fb, is_compiled=True))


class TestCreateFusedSetKvBufferArg(CustomTestCase):
    """Test that create_fused_set_kv_buffer_arg picks correct cache_loc."""

    def setUp(self):
        self.device = get_device()
        self.swa_pool = _make_swa_pool(self.device)
        self.fb = _make_forward_batch(self.swa_pool, self.device)

    def test_full_layer_uses_out_cache_loc(self):
        full_layer_id = FULL_LAYER_IDS[0]
        layer = _make_layer_mock(full_layer_id)
        value = torch.randn(
            BATCH_TOKENS, HEAD_NUM * HEAD_DIM, dtype=torch.bfloat16, device=self.device
        )
        arg = create_fused_set_kv_buffer_arg(value, layer, self.fb)

        self.assertIsInstance(arg, FusedSetKVBufferArg)
        self.assertTrue(torch.equal(arg.cache_loc, self.fb.out_cache_loc))

    def test_swa_layer_uses_out_cache_loc_swa(self):
        swa_layer_id = SWA_LAYER_IDS[0]
        layer = _make_layer_mock(swa_layer_id)
        value = torch.randn(
            BATCH_TOKENS, HEAD_NUM * HEAD_DIM, dtype=torch.bfloat16, device=self.device
        )
        arg = create_fused_set_kv_buffer_arg(value, layer, self.fb)

        self.assertIsInstance(arg, FusedSetKVBufferArg)
        self.assertTrue(torch.equal(arg.cache_loc, self.fb.out_cache_loc_swa))

    def test_buffers_have_correct_shape(self):
        layer = _make_layer_mock(FULL_LAYER_IDS[0])
        value = torch.randn(
            BATCH_TOKENS, HEAD_NUM * HEAD_DIM, dtype=torch.bfloat16, device=self.device
        )
        arg = create_fused_set_kv_buffer_arg(value, layer, self.fb)

        self.assertEqual(arg.k_buffer.ndim, 2)
        self.assertEqual(arg.v_buffer.ndim, 2)
        self.assertEqual(arg.k_buffer.shape[-1], HEAD_NUM * HEAD_DIM)
        self.assertEqual(arg.v_buffer.shape[-1], HEAD_NUM * HEAD_DIM)


class TestForwardNativeVsCudaKernel(CustomTestCase):
    """Numerics: compare forward_native KV scatter against the JIT CUDA kernel.

    Uses gpt-oss-20b-like config: head_size=64, 8 KV heads, neox-style RoPE.
    The JIT CUDA kernel (FlashInferRotaryEmbedding.forward_cuda) is the
    ground-truth reference that the hand-written fused_rope_kernel implements.
    forward_native must produce matching q, k, k_buffer, v_buffer."""

    # gpt-oss-20b attention config
    GPT_OSS_HEAD_SIZE = 64
    GPT_OSS_NUM_Q_HEADS = 8
    GPT_OSS_NUM_KV_HEADS = 8
    GPT_OSS_ROPE_BASE = 8000
    GPT_OSS_MAX_POS = 4096

    POOL_SIZE = MHATokenToKVPool.KV_POOL_SIZE
    # The CUDA kernel reads cos/sin in FP32 internally; forward_native casts
    # them to BF16 before the multiply.  The worst-case per-element delta is
    # 1 BF16 ULP ≈ 3.125e-2 at the relevant magnitudes, so we need > 1e-2.
    ATOL = 4e-2
    RTOL = 4e-2

    def setUp(self):
        self.device = get_device()
        rope_cfg = dict(
            head_size=self.GPT_OSS_HEAD_SIZE,
            rotary_dim=self.GPT_OSS_HEAD_SIZE,
            max_position_embeddings=self.GPT_OSS_MAX_POS,
            base=self.GPT_OSS_ROPE_BASE,
            is_neox_style=True,
            dtype=torch.bfloat16,
        )
        self.rope_cuda = FlashInferRotaryEmbedding(**rope_cfg).to(self.device)
        self.rope_native = TestingRotaryEmbedding(**rope_cfg).to(self.device)
        self.rope_native.cos_sin_cache = self.rope_cuda.cos_sin_cache

    def _run_comparison(self, batch_size, seq_len):
        """Run both paths and compare all outputs."""
        inputs = create_inputs(
            head_size=self.GPT_OSS_HEAD_SIZE,
            batch_size=batch_size,
            seq_len=seq_len,
            device=self.device,
            dtype=torch.bfloat16,
            num_q_heads=self.GPT_OSS_NUM_Q_HEADS,
            num_kv_heads=self.GPT_OSS_NUM_KV_HEADS,
        )
        positions = inputs["pos_ids"]
        query = inputs["query"]
        key = inputs["key"]
        value = inputs["value"]
        cache_loc = inputs["out_cache_loc"]

        flat_kv = self.GPT_OSS_NUM_KV_HEADS * self.GPT_OSS_HEAD_SIZE

        # --- Reference: JIT CUDA kernel (fused RoPE + KV scatter) ---
        k_buf_ref = torch.zeros(
            self.POOL_SIZE, self.GPT_OSS_NUM_KV_HEADS, self.GPT_OSS_HEAD_SIZE,
            dtype=torch.bfloat16, device=self.device,
        )
        v_buf_ref = torch.zeros_like(k_buf_ref)
        cuda_arg = TestingFusedSetKVBufferArg(
            value=value.clone(),
            k_buffer=k_buf_ref.view(self.POOL_SIZE, -1),
            v_buffer=v_buf_ref.view(self.POOL_SIZE, -1),
            cache_loc=cache_loc,
        )
        q_cuda, k_cuda = self.rope_cuda.forward_cuda(
            positions, query.clone(), key.clone(),
            fused_set_kv_buffer_arg=cuda_arg,
        )

        # --- Test: forward_native (pure PyTorch scatter) ---
        k_buf_native = torch.zeros(
            self.POOL_SIZE, self.GPT_OSS_NUM_KV_HEADS, self.GPT_OSS_HEAD_SIZE,
            dtype=torch.bfloat16, device=self.device,
        )
        v_buf_native = torch.zeros_like(k_buf_native)
        native_arg = FusedSetKVBufferArg(
            value=value.clone(),
            k_buffer=k_buf_native.view(self.POOL_SIZE, -1),
            v_buffer=v_buf_native.view(self.POOL_SIZE, -1),
            cache_loc=cache_loc,
        )
        q_native, k_native = self.rope_native.forward_native(
            positions, query.clone(), key.clone(),
            fused_set_kv_buffer_arg=native_arg,
        )

        # Compare q and k (RoPE outputs)
        torch.testing.assert_close(
            q_native, q_cuda, atol=self.ATOL, rtol=self.RTOL,
            msg="query mismatch: forward_native vs CUDA kernel",
        )
        torch.testing.assert_close(
            k_native, k_cuda, atol=self.ATOL, rtol=self.RTOL,
            msg="key mismatch: forward_native vs CUDA kernel",
        )

        # Compare k_buffer (rotated key scattered into cache)
        torch.testing.assert_close(
            k_buf_native.view(self.POOL_SIZE, -1)[cache_loc],
            k_buf_ref.view(self.POOL_SIZE, -1)[cache_loc],
            atol=self.ATOL, rtol=self.RTOL,
            msg="k_buffer mismatch: forward_native vs CUDA kernel",
        )

        # v_buffer scatter is pure index_put (no RoPE math), so it must be exact
        self.assertTrue(
            torch.equal(
                v_buf_native.view(self.POOL_SIZE, -1)[cache_loc],
                v_buf_ref.view(self.POOL_SIZE, -1)[cache_loc],
            ),
            "v_buffer mismatch: scatter must be bitwise exact",
        )

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

    def test_prefill_short(self):
        """Short prefill (2 seqs × 512 tokens)."""
        self._run_comparison(batch_size=2, seq_len=512)

    def test_prefill_long(self):
        """Long prefill (4 seqs × 4096 tokens)."""
        self._run_comparison(batch_size=4, seq_len=4096)

    def test_none_arg_rope_matches_cuda(self):
        """RoPE-only (no scatter) must also match the CUDA kernel."""
        inputs = create_inputs(
            head_size=self.GPT_OSS_HEAD_SIZE,
            batch_size=32, seq_len=1,
            device=self.device, dtype=torch.bfloat16,
            num_q_heads=self.GPT_OSS_NUM_Q_HEADS,
            num_kv_heads=self.GPT_OSS_NUM_KV_HEADS,
        )
        q_cuda, k_cuda = self.rope_cuda.forward_cuda(
            inputs["pos_ids"], inputs["query"].clone(), inputs["key"].clone(),
            fused_set_kv_buffer_arg=None,
        )
        q_native, k_native = self.rope_native.forward_native(
            inputs["pos_ids"], inputs["query"].clone(), inputs["key"].clone(),
            fused_set_kv_buffer_arg=None,
        )
        torch.testing.assert_close(q_native, q_cuda, atol=self.ATOL, rtol=self.RTOL)
        torch.testing.assert_close(k_native, k_cuda, atol=self.ATOL, rtol=self.RTOL)


if __name__ == "__main__":
    unittest.main(verbosity=3)
