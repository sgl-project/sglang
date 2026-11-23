"""prims_ts DSA backend: FlashInfer Prims-TS sparse MLA over raw FP8 MLA rows (E4M3
q, natural-log LSE). Under sharded DCP each rank attends over its owned top-k rows
(compacted, lengths = owned count); the partials merge to the single-rank result,
and the DCP combine must merge them base e."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.attention.dsa.dsa_dcp_sharded import _localize_compact_kernel
from sglang.srt.layers.attention.dsa.prims_ts import (
    _EXACT_PLAN_MAX_ROWS,
    PrimsTSSparseMLA,
    check_prims_ts_sparse_mla_support,
)
from sglang.srt.layers.attention.dsa_backend import _validate_prims_ts_dcp
from sglang.srt.layers.dcp import layout
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

FP8 = torch.float8_e4m3fn
KV_LORA, ROPE, QK_NOPE, TOPK = 512, 64, 128, 2048
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


class TestShardedDcpPrimsTs(CustomTestCase):
    def _require_prims_ts(self):
        if not _sm100():
            self.skipTest("Prims-TS sparse MLA needs SM100/SM103/SM107")
        try:
            check_prims_ts_sparse_mla_support(
                device=torch.device("cuda"),
                num_heads=96,
                kv_lora_rank=KV_LORA,
                qk_rope_head_dim=ROPE,
            )
        except ValueError as exc:
            self.skipTest(str(exc))

    def _check_dcp8_partials_merge_to_full(self, rows: int):
        """Per-rank Prims-TS partials (natural-log LSE; a rank owning none of a
        row's keys returns zero output and -inf LSE) merge by exp weights, the base
        TestPrimsTsDcpLseBase pins for the combine, to the single-rank result."""
        self._require_prims_ts()
        dcp, heads, ctx = 8, 96, 16384
        gen = torch.Generator(device="cuda").manual_seed(3)
        kv = (torch.randn(ctx, DIM, generator=gen, device="cuda") * 0.5).to(FP8)
        q8 = torch.randn(rows, heads, DIM, generator=gen, device="cuda").to(FP8)
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

        runner = PrimsTSSparseMLA(torch.device("cuda"))

        def attend(cache, table, lens):
            return runner.forward(
                q=q8,
                kv_cache=cache,
                indices=table,
                lengths=lens,
                sm_scale=SM_SCALE,
                kv_scale=1.0,
            )

        full, full_lse = attend(kv, slots, counts)
        outs, lses = [], []
        for rank in range(dcp):
            table, lens = _localize(slots, dcp, rank)
            o, lse = attend(kv[rank::dcp].contiguous(), table, lens)
            if rank:
                self.assertTrue(torch.isneginf(lse[0]).all().item())
                self.assertFalse(o[0].any().item())
            outs.append(o.float())
            lses.append(lse)
        # (rows, heads, width, packed) plan keys: rows past the exact limit pack
        packed = rows > _EXACT_PLAN_MAX_ROWS
        self.assertEqual({key[3] for key in runner._plans}, {packed})
        lse = torch.stack(lses)  # natural log
        w = torch.exp(lse - lse.max(0).values)
        merged = (w.unsqueeze(-1) * torch.stack(outs)).sum(0) / w.sum(0).unsqueeze(-1)
        merged_lse = torch.log(w.sum(0)) + lse.max(0).values

        qf, kvf = q8.float(), kv.float()
        ref = torch.empty_like(full, dtype=torch.float32)
        ref_lse = torch.empty_like(full_lse)
        for r in range(rows):
            k = kvf[slots[r, : counts[r]].long()]
            s = (qf[r] @ k.T) * SM_SCALE
            ref_lse[r] = torch.logsumexp(s, dim=-1)
            ref[r] = torch.softmax(s, dim=-1) @ k[:, :KV_LORA]

        def rel(a):
            return ((a.float() - ref).norm() / ref.norm()).item()

        self.assertLess(rel(full), 0.03)
        self.assertLess(rel(merged), rel(full) * 1.2)
        torch.testing.assert_close(full_lse, ref_lse, atol=2e-2, rtol=2e-3)
        torch.testing.assert_close(merged_lse, full_lse, atol=1e-3, rtol=1e-3)

    def test_dcp8_partials_merge_to_full(self):
        self._check_dcp8_partials_merge_to_full(rows=24)

    def test_packed_rows_dcp8_partials_merge_to_full(self):
        # Past the exact-plan limit: power-of-two packed capacity with qo_indptr.
        self._check_dcp8_partials_merge_to_full(rows=_EXACT_PLAN_MAX_ROWS + 1)


class TestPrimsTsDcpLseBase(CustomTestCase):
    """Prims-TS partials carry a natural-log LSE, and the DCP combine picks the LSE
    base per DSA DCP mode; each mode must merge them base e or reject prims_ts."""

    def _combine_base_e(self, *, shards_index_k: bool, decode_backend: str) -> bool:
        exec_ = SimpleNamespace(
            kernel=SimpleNamespace(dsa_decode_backend=decode_backend)
        )
        with (
            mock.patch.object(
                layout, "dsa_shards_index_k", return_value=shards_index_k
            ),
            mock.patch.object(layout, "get_exec", return_value=exec_),
        ):
            return layout.is_mla_dcp_lse_base_on_e("dsa")

    def test_sharded_index_k_merges_base_e(self):
        _validate_prims_ts_dcp(dcp_size=8, shards_index_k=True)
        self.assertTrue(
            self._combine_base_e(shards_index_k=True, decode_backend="prims_ts")
        )
        # trtllm-gen partials are base 2
        self.assertFalse(
            self._combine_base_e(shards_index_k=True, decode_backend="trtllm")
        )

    def test_replicated_index_k_rejects_prims_ts(self):
        # Replicated index-K (non-K3.1 DSA models) merges base 2 for every kernel.
        self.assertFalse(
            self._combine_base_e(shards_index_k=False, decode_backend="prims_ts")
        )
        with self.assertRaisesRegex(ValueError, "sharded index-K"):
            _validate_prims_ts_dcp(dcp_size=2, shards_index_k=False)
        # Without DCP there is no merge.
        _validate_prims_ts_dcp(dcp_size=1, shards_index_k=False)


if __name__ == "__main__":
    unittest.main()
