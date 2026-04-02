"""
Unit tests for fused KV buffer helpers.

Validates the torch.compile-path gate logic and cache_loc selection
for SWA models (e.g. gpt-oss-20b).
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


if __name__ == "__main__":
    unittest.main(verbosity=3)
