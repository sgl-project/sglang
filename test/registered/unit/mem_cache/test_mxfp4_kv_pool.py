"""CUDA pool-level tests for the MXFP4 KV cache: real MHATokenToKVPool.

Covers the production write path (`set_kv_buffer` -> `quantize_and_store`),
the reserved padding slot, the PLAIN BF16 read (`get_kv_buffer`), and slot
moves that must carry packed data AND E8M0 scales together.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.quantization.kvfp4_tensor import MXFP4KVQuantizeUtil
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_HAS_CUDA = torch.cuda.is_available()


def _make_pool(num_slots, heads, head_dim, layers=1):
    from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
        MXFP4KVCacheMethod,
    )

    return MHATokenToKVPool(
        size=num_slots,
        page_size=1,
        dtype=torch.float4_e2m1fn_x2,
        head_num=heads,
        head_dim=head_dim,
        layer_num=layers,
        device="cuda",
        enable_memory_saver=False,
        enable_alt_stream=False,
        enable_kv_cache_copy=True,
        quant_method=MXFP4KVCacheMethod(),
    )


@unittest.skipUnless(_HAS_CUDA, "CUDA is required")
class TestMxfp4PoolWriteReadMove(CustomTestCase):
    HEADS = 4
    HEAD_DIM = 64
    SLOTS = 64

    def _pool(self):
        return _make_pool(self.SLOTS, self.HEADS, self.HEAD_DIM)

    def _locs(self, n, seed):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        return torch.randperm(self.SLOTS - 1, device="cuda", generator=gen)[:n] + 1

    def test_set_kv_buffer_bit_exact_and_slot0_reserved(self):
        torch.manual_seed(20260903)
        pool = self._pool()
        layer = SimpleNamespace(layer_id=0)
        k = torch.randn(
            6, self.HEADS, self.HEAD_DIM, dtype=torch.bfloat16, device="cuda"
        )
        v = torch.randn_like(k)
        loc = torch.cat(
            [torch.zeros(1, dtype=torch.int64, device="cuda"), self._locs(5, 7)]
        )
        k0, v0 = k.clone(), v.clone()
        # Checkpoint FP8 scales must be ignored by the MXFP4 recipe.
        pool.set_kv_buffer(layer, loc, k, v, 0.0275, 0.0245)

        exp_k, exp_ks = MXFP4KVQuantizeUtil.batched_quantize(k0[1:])
        exp_v, exp_vs = MXFP4KVQuantizeUtil.batched_quantize(v0[1:])
        self.assertTrue(torch.equal(pool.k_buffer[0][loc[1:]], exp_k))
        self.assertTrue(torch.equal(pool.k_scale_buffer[0][loc[1:]], exp_ks))
        self.assertTrue(torch.equal(pool.v_buffer[0][loc[1:]], exp_v))
        self.assertTrue(torch.equal(pool.v_scale_buffer[0][loc[1:]], exp_vs))
        # Slot 0 stays untouched (reserved CUDA-graph padding slot).
        self.assertEqual(pool.k_buffer[0][0].sum().item(), 0)
        self.assertEqual(pool.k_scale_buffer[0][0].sum().item(), 0)
        self.assertEqual(pool.v_buffer[0][0].sum().item(), 0)
        self.assertEqual(pool.v_scale_buffer[0][0].sum().item(), 0)
        self.assertTrue(torch.equal(k, k0), "MXFP4 write must not mutate inputs")
        self.assertTrue(torch.equal(v, v0))

    def test_plain_read_matches_codec(self):
        torch.manual_seed(20260904)
        pool = self._pool()
        layer = SimpleNamespace(layer_id=0)
        k = torch.randn(
            8, self.HEADS, self.HEAD_DIM, dtype=torch.bfloat16, device="cuda"
        )
        v = torch.randn_like(k)
        loc = self._locs(8, 11)
        k0, v0 = k.clone(), v.clone()
        pool.set_kv_buffer(layer, loc, k, v)

        k_plain, v_plain = pool.get_kv_buffer(0)
        self.assertEqual(k_plain.dtype, torch.bfloat16)
        self.assertEqual(k_plain.shape, (self.SLOTS + 1, self.HEADS, self.HEAD_DIM))
        exp_k = MXFP4KVQuantizeUtil.batched_dequantize(
            pool.k_buffer[0][loc],
            pool.k_scale_buffer[0][loc],
            logical_dim=self.HEAD_DIM,
        )
        exp_v = MXFP4KVQuantizeUtil.batched_dequantize(
            pool.v_buffer[0][loc],
            pool.v_scale_buffer[0][loc],
            logical_dim=self.HEAD_DIM,
        )
        torch.testing.assert_close(k_plain[loc], exp_k, rtol=0, atol=0)
        torch.testing.assert_close(v_plain[loc], exp_v, rtol=0, atol=0)
        self.assertTrue(torch.equal(k, k0))
        self.assertTrue(torch.equal(v, v0))

    def test_slot_move_carries_data_and_scales(self):
        torch.manual_seed(20260905)
        pool = self._pool()
        layer = SimpleNamespace(layer_id=0)
        k = torch.randn(
            8, self.HEADS, self.HEAD_DIM, dtype=torch.bfloat16, device="cuda"
        )
        v = torch.randn_like(k)
        src = self._locs(8, 13)
        pool.set_kv_buffer(layer, src, k, v)

        dst = self._locs(8, 17)
        self.assertFalse(torch.equal(src, dst))
        pool.move_kv_cache(dst, src)

        exp_k, exp_ks = MXFP4KVQuantizeUtil.batched_quantize(k)
        exp_v, exp_vs = MXFP4KVQuantizeUtil.batched_quantize(v)
        self.assertTrue(torch.equal(pool.k_buffer[0][dst], exp_k))
        self.assertTrue(torch.equal(pool.k_scale_buffer[0][dst], exp_ks))
        self.assertTrue(torch.equal(pool.v_buffer[0][dst], exp_v))
        self.assertTrue(torch.equal(pool.v_scale_buffer[0][dst], exp_vs))
        # The tiled copy duplicates rows; the source slots keep their payload
        # (same semantics as the block16/mxfp8 pools).
        self.assertTrue(torch.equal(pool.k_buffer[0][src], exp_k))
        self.assertTrue(torch.equal(pool.k_scale_buffer[0][src], exp_ks))

    def test_v_head_dim_mismatch_fails_fast(self):
        from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
            MXFP4KVCacheMethod,
        )

        with self.assertRaisesRegex(ValueError, "v_head_dim"):
            MHATokenToKVPool(
                size=16,
                page_size=1,
                dtype=torch.float4_e2m1fn_x2,
                head_num=2,
                head_dim=64,
                layer_num=1,
                device="cuda",
                enable_memory_saver=False,
                enable_alt_stream=False,
                v_head_dim=32,
                quant_method=MXFP4KVCacheMethod(),
            )


if __name__ == "__main__":
    unittest.main()
