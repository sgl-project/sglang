"""Low-ratio indexer-K fp4 storage vs the reference quantization (per-32 ue8m0,
e2m1, max 6.0): store_fp4_index_k_cache must reproduce fake_quant_fp4, the
pure-torch mirror of the reference, bit-exactly. GPU only (triton + packed store).
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-large")


def dev_dequantize(k_fp4: torch.Tensor, k_sf: torch.Tensor) -> torch.Tensor:
    """Invert dev's packed fp4 (payload [n,64] int8 + one packed-uint32 scale)."""
    from sglang.srt.layers.quantization.fp8 import DSV4_DEQUANT_FP4_TABLE

    u = k_fp4.view(torch.uint8)
    codes = torch.stack([u & 0x0F, (u >> 4) & 0x0F], dim=-1).flatten(1)  # [n, 128]
    vals = DSV4_DEQUANT_FP4_TABLE.to(k_fp4.device)[codes.long()]
    exps = torch.stack([(k_sf >> (8 * i)) & 0xFF for i in range(4)], dim=-1)
    scales = torch.exp2(exps.float() - 127).repeat_interleave(32, dim=-1)
    return vals * scales


class TestIndexerFp4Storage(CustomTestCase):
    def test_quantize_matches_reference(self):
        from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
            quantize_fp4_indexer_tensor,
        )
        from sglang.srt.layers.attention.dsv4.torch_quant import fake_quant_fp4

        torch.manual_seed(0)
        x = (torch.randn(64, 128, device="cuda", dtype=torch.bfloat16) * 3).contiguous()

        k_fp4, k_sf = quantize_fp4_indexer_tensor(x, rne=True)
        dev_vals = dev_dequantize(k_fp4, k_sf)
        ref_vals = fake_quant_fp4(x)

        self.assertTrue(
            torch.equal(dev_vals.float(), ref_vals.float()),
            msg=(
                "dev fp4 quantization diverged from the reference: max abs diff "
                f"{(dev_vals.float() - ref_vals.float()).abs().max()}"
            ),
        )

    def test_pool_store_roundtrip(self):
        from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
            store_fp4_index_k_cache,
        )
        from sglang.srt.layers.attention.dsv4.torch_quant import fake_quant_fp4

        torch.manual_seed(1)
        n, page_size = 8, 128
        x = torch.randn(n, 128, device="cuda", dtype=torch.bfloat16).contiguous()
        # Packed fp4 pool page: page_size * (64 payload + 4 scale) bytes.
        cache = torch.zeros(4, page_size * (64 + 4), dtype=torch.uint8, device="cuda")
        loc = torch.tensor(
            [3, 130, 260, 5, 129, 384, 200, 7], dtype=torch.int32, device="cuda"
        )

        store_fp4_index_k_cache(
            input=x, cache=cache, loc=loc, page_size=page_size, rne=True
        )
        ref_vals = fake_quant_fp4(x)

        for i in range(n):
            L = int(loc[i])
            page, off = L // page_size, L % page_size
            payload = cache[page, off * 64 : (off + 1) * 64].view(torch.int8)
            sfb = cache[page, page_size * 64 + off * 4 : page_size * 64 + off * 4 + 4]
            sf = (
                (
                    sfb[0].int()
                    | (sfb[1].int() << 8)
                    | (sfb[2].int() << 16)
                    | (sfb[3].int() << 24)
                )
                .to(torch.int32)
                .reshape(1)
            )
            vals = dev_dequantize(payload.unsqueeze(0), sf)[0]
            self.assertTrue(
                torch.equal(vals.float(), ref_vals[i].float()),
                msg=f"slot {L}: roundtrip diverged from reference",
            )

    def test_pool_dequant_readback(self):
        """get_index_k_dequant must invert the packed 64-slot page layout exactly;
        the prefill indexer reads K through it."""
        from sglang.srt.layers.attention.dsv4.torch_quant import fake_quant_fp4
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DSV41_INDEX_PAGE_SIZE,
            DeepSeekV4IndexerPool,
        )

        torch.manual_seed(2)
        size = 5 * DSV41_INDEX_PAGE_SIZE + 17  # not a whole number of pages
        pool = DeepSeekV4IndexerPool(
            size=size,
            page_size=DSV41_INDEX_PAGE_SIZE,
            dtype=torch.bfloat16,
            index_head_dim=128,
            layer_num=2,
            device="cuda",
            enable_memory_saver=False,
            use_fp4_indexer=True,
        )
        pool.index_k_rne = True

        n = 40
        loc = torch.randperm(size, device="cuda")[:n].to(torch.int32)
        x = (torch.randn(n, 128, device="cuda", dtype=torch.bfloat16) * 2).contiguous()
        pool.set_index_fp4(layer_id=1, loc=loc, cache_k=x)

        table = pool.get_index_k_dequant(1)
        self.assertEqual(tuple(table.shape), (size, 128))
        self.assertTrue(
            torch.equal(table[loc.long()].float(), fake_quant_fp4(x).float()),
            msg="dequant readback diverged from the reference quantization",
        )
        # Slot-selective readback (what the prefill indexer uses) matches too,
        # in the caller's slot order.
        perm = torch.randperm(n, device="cuda")
        rows = pool.get_index_k_dequant(1, loc[perm])
        self.assertTrue(
            torch.equal(rows.float(), fake_quant_fp4(x)[perm].float()),
            msg="slot-selective dequant readback diverged",
        )
        untouched = torch.ones(size, dtype=torch.bool, device="cuda")
        untouched[loc.long()] = False
        self.assertTrue(bool((table[untouched] == 0).all()))
        # The other layer was never written.
        self.assertTrue(bool((pool.get_index_k_dequant(0) == 0).all()))


if __name__ == "__main__":
    unittest.main()
