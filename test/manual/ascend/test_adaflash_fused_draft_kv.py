"""Check the fused masked store against the standalone Ascend norm and RoPE path."""

import unittest

import torch
import torch_npu
from sgl_kernel_npu.norm.fused_rope_qk_mqa import fused_rope_qk_mqa

from sglang.srt.hardware_backend.npu.adaflash_kv import store_packed_adaflash_kv


class TestAdaFlashFusedDraftKV(unittest.TestCase):
    def test_bf16_cache_matches_unfused_for_irregular_and_masked_slots(self):
        torch.manual_seed(17)
        heads, dim, capacity = 8, 128, 1024
        dtype = torch.bfloat16
        angles = torch.randn((1024, dim // 2), device="npu")
        table = torch.cat((angles.cos(), angles.sin()), dim=-1)
        weight = torch.randn(dim, device="npu", dtype=dtype)
        for rows in (1, 17, 401):
            for table_dtype in (torch.float32, dtype):
                with self.subTest(rows=rows, table_dtype=table_dtype):
                    kv = torch.randn((rows, 2 * heads * dim), device="npu", dtype=dtype)
                    k, v = kv.split(heads * dim, dim=1)
                    positions = torch.randint(0, len(table), (rows,)).npu()
                    slots = torch.randperm(capacity - 1)[:rows].npu() + 1
                    # Mark every third row as uncommitted (-1) to verify slot 0
                    # and existing cache entries are untouched.
                    keep = (torch.arange(rows, device="npu") % 3) != 1
                    masked_slots = slots.masked_fill(~keep, -1)
                    cos_sin = table.to(table_dtype)
                    norm, _ = torch_npu.npu_rms_norm(
                        k.reshape(-1, dim), weight, 1e-6
                    )
                    norm = norm.reshape_as(k)
                    dummy_q = torch.zeros_like(norm)
                    if table_dtype == torch.float32:
                        _, expected_k = fused_rope_qk_mqa(
                            dummy_q.view(rows, heads, dim),
                            norm.view(rows, heads, dim),
                            cos_sin.index_select(0, positions),
                            dim,
                            True,
                        )
                    else:
                        _, expected_k = torch_npu.npu_mrope(
                            positions,
                            dummy_q,
                            norm,
                            cos_sin,
                            dim,
                            mrope_section=[0, 0, 0],
                            rotary_mode="half",
                        )
                    # Seed sentinel values (including slot 0) to confirm masked
                    # rows never clobber untouched slots.
                    k_cache = torch.full(
                        (capacity, heads, dim), 3.25, device="npu", dtype=dtype
                    )
                    v_cache = torch.full(
                        (capacity, heads, dim), -2.5, device="npu", dtype=dtype
                    )
                    ref_k = k_cache.clone()
                    ref_v = v_cache.clone()
                    store_packed_adaflash_kv(
                        k,
                        v,
                        weight,
                        cos_sin,
                        positions,
                        masked_slots,
                        k_cache,
                        v_cache,
                        num_heads=heads,
                        head_dim=dim,
                        eps=1e-6,
                    )
                    live = torch.nonzero(keep, as_tuple=True)[0]
                    if live.numel() > 0:
                        ref_k.index_copy_(
                            0,
                            slots.index_select(0, live),
                            expected_k.reshape(rows, heads, dim).index_select(0, live),
                        )
                        ref_v.index_copy_(
                            0,
                            slots.index_select(0, live),
                            v.reshape(rows, heads, dim).index_select(0, live),
                        )
                    torch.testing.assert_close(k_cache, ref_k, atol=0.016, rtol=0.01)
                    torch.testing.assert_close(v_cache, ref_v, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
