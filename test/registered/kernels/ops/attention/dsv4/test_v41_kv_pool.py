"""The fused SM100 writers of a V4.1-layout KV pool round-trip through the layout-aware dequant; the pool's buffer contract is pinned on CPU in test/registered/unit/mem_cache/test_dsv4_compressed_pools.py."""

import unittest

import torch

from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DeepSeekV4TokenToKVPool,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test import dsv41_kv_quant_reference as tq
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

HEAD_DIM = 512
ROPE_DIM = 64
PAGE_SIZE = 256
FULL_SIZE = 4 * PAGE_SIZE


def _sm100():
    return (
        torch.cuda.is_available()
        and torch.version.cuda is not None
        and torch.cuda.get_device_capability()[0] >= 10
    )


def _make_pool(ratios, kv_source_layers, kv_layout, compressed_kv_layout=None, **sizes):
    return DeepSeekV4TokenToKVPool(
        max_num_reqs=16,
        swa_size=FULL_SIZE,
        c4_size=sizes.get("c4_size", 0),
        c128_size=sizes.get("c128_size", 0),
        c4_state_pool_size=sizes.get("c4_state_pool_size", 0),
        c128_state_pool_size=sizes.get("c128_state_pool_size", 0),
        page_size=PAGE_SIZE,
        swa_page_size=PAGE_SIZE,
        dtype=torch.float8_e4m3fn,
        c4_state_dtype=torch.float32,
        c128_state_dtype=torch.float32,
        qk_nope_head_dim=HEAD_DIM - ROPE_DIM,
        qk_rope_head_dim=ROPE_DIM,
        indexer_head_dim=128,
        layer_num=len(ratios),
        device="cuda",
        enable_memory_saver=False,
        compression_ratios=ratios,
        kv_source_layers=kv_source_layers,
        full_size=FULL_SIZE,
        kv_layout=kv_layout,
        compressed_kv_layout=compressed_kv_layout,
    )


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestV41KVPool(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        set_global_server_args_for_scheduler(
            ServerArgs(model_path="dummy", page_size=PAGE_SIZE)
        )

    @unittest.skipUnless(_sm100(), "the V4.1 store kernels are SM100 kernels")
    def test_fused_writers_round_trip(self):
        """SWA write (fp8) and compressed write with in-kernel RoPE (fp4) read back
        through the layout-aware dequant as the reference values."""
        from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (
            dequantize_k_cache_paged,
        )
        from sglang.srt.layers.attention.dsv4.dsv41_sparse import rope_tail

        pool = _make_pool([0, 0, 2, 1], [2, 3], KVLayout.V41)
        g = torch.Generator(device="cuda").manual_seed(3)
        n = 100
        # SWA: finished (normed, rotated) bf16 rows.
        x = torch.randn(n, HEAD_DIM, generator=g, device="cuda", dtype=torch.bfloat16)
        swa_loc = torch.randperm(FULL_SIZE, generator=g, device="cuda")[:n].to(
            torch.int32
        )
        pool.set_swa_key_buffer_radix_fused(layer_id=0, swa_loc=swa_loc, cache_k=x)
        got = dequantize_k_cache_paged(
            pool.get_swa_key_buffer_radix(0),
            swa_loc,
            pool.swa_page_size,
            layout=pool.get_swa_key_layout(),
        )
        ref = tq.dequantize_k_cache_v41(
            tq.quantize_k_cache_v41(x.view(1, n, HEAD_DIM)), n
        ).view(n, 1, HEAD_DIM)
        self.assertTrue(torch.equal(got, ref))
        # Compressed (fp4): the un-rotated latent plus its freqs; the cache holds
        # exactly fake_quant_compressed_kv(rope_tail(latent)).
        layer_id = pool.sources_by_ratio[1][0]
        latent = torch.randn(
            n, HEAD_DIM, generator=g, device="cuda", dtype=torch.bfloat16
        )
        angles = torch.randn(n, ROPE_DIM // 2, generator=g, device="cuda")
        freqs = torch.polar(torch.ones_like(angles), angles)
        loc = torch.randperm(FULL_SIZE, generator=g, device="cuda")[:n].to(torch.int64)
        pool.set_extra_key_buffer_fused(
            layer_id=layer_id, loc=loc, cache_k=latent, freqs_cis=freqs
        )
        got = dequantize_k_cache_paged(
            pool.get_extra_key_buffer(layer_id),
            loc,
            pool.get_extra_key_page_size(layer_id),
            layout=pool.get_extra_key_layout(layer_id),
        )
        self.assertTrue(
            torch.equal(
                got.squeeze(1),
                tq.fake_quant_compressed_kv(rope_tail(latent, freqs, ROPE_DIM)),
            )
        )
        # The (fp8 nope, bf16 rope) pack writer is the V4 layout only.
        with self.assertRaises(AssertionError):
            pool.set_swa_key_buffer(0, swa_loc, None)


if __name__ == "__main__":
    unittest.main()
