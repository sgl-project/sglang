"""Per-rank trtllm-gen DSA partials under sharded DCP merge to the single-rank result."""

from __future__ import annotations

import unittest

import torch

from sglang.srt.layers.attention.dsa.dsa_dcp import dcp_trtllm_sparse_attention
from sglang.srt.layers.attention.dsa.dsa_dcp_sharded import _localize_compact_kernel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

FP8 = torch.float8_e4m3fn
KV_LORA, ROPE, QK_NOPE, PAGE, TOPK = 512, 64, 128, 64, 2048
DIM = KV_LORA + ROPE
SM_SCALE = (QK_NOPE + ROPE) ** -0.5


def _sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


def _localize(slots: torch.Tensor, dcp: int, rank: int):
    out = torch.empty_like(slots)
    lens = torch.empty(slots.shape[0], dtype=torch.int32, device=slots.device)
    _localize_compact_kernel[(slots.shape[0],)](
        slots, slots.stride(0), out, out.stride(0), lens, dcp, rank, TOPK=TOPK
    )
    return out, lens


class TestShardedDcpPartials(CustomTestCase):
    def _attend(self, kv, q8, table, seq_lens, workspace, counter):
        rows, heads, _ = q8.shape
        return dcp_trtllm_sparse_attention(
            query=q8.view(rows, 1, heads, DIM),
            block_tables=table.unsqueeze(1),
            seq_lens=seq_lens,
            cu_seqlens_q=torch.arange(rows + 1, dtype=torch.int32, device="cuda"),
            kv_cache=kv.view(-1, 1, PAGE, DIM),
            workspace_buffer=workspace,
            qk_nope_head_dim=QK_NOPE,
            kv_lora_rank=KV_LORA,
            qk_rope_head_dim=ROPE,
            max_seq_len=kv.shape[0],
            sparse_mla_top_k=TOPK,
            bmm1_scale=SM_SCALE,
            backend="trtllm-gen",
            multi_ctas_kv_counter_buffer=counter,
        )

    def test_dcp8_partials_merge_to_full(self):
        if not _sm100():
            self.skipTest("trtllm-gen sparse MLA needs SM100")
        from sglang.srt.layers.attention.trtllm_mla_backend import (
            make_persistent_multi_ctas_kv_counter_buffer,
        )

        dcp, heads, padded, rows, ctx = 8, 96, 128, 24, 16384
        gen = torch.Generator(device="cuda").manual_seed(1)
        kv = (torch.randn(ctx, DIM, generator=gen, device="cuda") * 0.5).to(FP8)
        q = (torch.randn(rows, heads, DIM, generator=gen, device="cuda")).to(
            torch.bfloat16
        )
        q8 = torch.zeros(rows, padded, DIM, dtype=FP8, device="cuda")
        q8[:, :heads] = q.to(FP8)

        # Row 0 has keys on rank 0 only, so the other ranks see an empty row.
        slots = torch.full((rows, TOPK), -1, dtype=torch.int32, device="cuda")
        counts = torch.randint(1, TOPK + 1, (rows,), generator=gen, device="cuda")
        counts[1] = TOPK
        for r in range(rows):
            n = int(counts[r])
            pick = torch.randperm(ctx, generator=gen, device="cuda")[:n]
            if r == 0:
                pick = torch.arange(0, ctx, dcp, device="cuda")[:n]
            slots[r, :n] = pick.to(torch.int32)
        counts = (slots >= 0).sum(1).to(torch.int32)

        workspace = torch.zeros(384 << 20, dtype=torch.uint8, device="cuda")
        counter = make_persistent_multi_ctas_kv_counter_buffer(
            torch.device("cuda"), padded, max_batch_size=rows
        )
        full, full_lse = self._attend(kv, q8, slots, counts, workspace, counter)
        full = full.view(rows, padded, KV_LORA)[:, :heads].float()

        outs, lses = [], []
        for rank in range(dcp):
            local_kv = kv[rank::dcp].contiguous()
            table, lens = _localize(slots, dcp, rank)
            o, lse = self._attend(local_kv, q8, table, lens, workspace, counter)
            outs.append(o.view(rows, padded, KV_LORA)[:, :heads].float())
            lses.append(lse[:, :heads])
        lse = torch.stack(lses)  # base 2
        w = torch.exp2(lse - lse.max(0).values)
        merged = (w.unsqueeze(-1) * torch.stack(outs)).sum(0) / w.sum(0).unsqueeze(-1)
        merged_lse = torch.log2(w.sum(0)) + lse.max(0).values

        # trtllm-gen's own fp8 error is ~2%, so compare both paths to fp32.
        qf, kvf = q8[:, :heads].float(), kv.float()
        ref = torch.empty_like(full)
        for r in range(rows):
            k = kvf[slots[r, : counts[r]].long()]
            p = torch.softmax((qf[r] @ k.T) * SM_SCALE, dim=-1)
            ref[r] = p @ k[:, :KV_LORA]

        def rel(a):
            return ((a - ref).norm() / ref.norm()).item()

        self.assertLess(rel(full), 0.03)
        self.assertLess(rel(merged), rel(full) * 1.2)
        torch.testing.assert_close(
            merged_lse, full_lse[:, :heads], atol=1e-3, rtol=1e-3
        )


if __name__ == "__main__":
    unittest.main()
