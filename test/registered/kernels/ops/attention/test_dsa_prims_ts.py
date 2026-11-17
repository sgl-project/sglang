"""prims_ts DSA backend: FlashInfer Prims-TS sparse MLA over raw FP8 MLA rows and
over 352 B/token NVFP4 rows (E4M3 q, natural-log LSE). Under sharded DCP each rank
attends over its owned top-k rows (compacted, lengths = owned count); the partials
merge to the single-rank result, and the DCP combine must merge them base e. The
NVFP4 rows are checked against FP32 attention over the values the loader expands."""

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


_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _expand_nvfp4_rows(kv: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    """FP32 [n, 576]: what the Prims-TS loader expands each 352 B row to. NoPE is
    E2M1 (even feature in the low nibble) x its block's E4M3 scale (block b's at
    byte 320 + 8*(b % 4) + b//4), rounded to E4M3; RoPE is E4M3 at bytes 256:320."""
    rows = kv.view(-1, 352)[slots.long()]
    packed = rows[:, :256]
    codes = torch.stack([packed & 15, packed >> 4], dim=-1).view(-1, 512)
    magnitude = torch.tensor(_E2M1, device=kv.device)[(codes & 7).long()]
    values = torch.where((codes & 8) != 0, -magnitude, magnitude)
    block = torch.arange(32, device=kv.device)
    scales = rows[:, 320 + 8 * (block % 4) + block // 4].view(FP8).float()
    nope = (values.view(-1, 32, 16) * scales[:, :, None]).view(-1, 512)
    nope = nope.clamp(-448, 448).to(FP8).float()
    rope = rows[:, 256:320].contiguous().view(FP8).float()
    return torch.cat([nope, rope], dim=-1)


def _require_prims_ts(case, heads: int = 96, kv_cache_format: str = "native"):
    if not _sm100():
        case.skipTest("Prims-TS sparse MLA needs SM100/SM103/SM107")
    try:
        check_prims_ts_sparse_mla_support(
            device=torch.device("cuda"),
            num_heads=heads,
            kv_lora_rank=KV_LORA,
            qk_rope_head_dim=ROPE,
            kv_cache_format=kv_cache_format,
        )
    except ValueError as exc:
        case.skipTest(str(exc))


def _localize(slots: torch.Tensor, dcp: int, rank: int):
    out = torch.empty_like(slots)
    lens = torch.empty(slots.shape[0], dtype=torch.int32, device=slots.device)
    _localize_compact_kernel[(slots.shape[0],)](
        slots, slots.stride(0), out, out.stride(0), lens, dcp, rank, TOPK=TOPK
    )
    return out, lens


class TestShardedDcpPrimsTs(CustomTestCase):
    def _check_dcp8_partials_merge_to_full(self, rows: int):
        """Per-rank Prims-TS partials (natural-log LSE; a rank owning none of a
        row's keys returns zero output and -inf LSE) merge by exp weights, the base
        TestPrimsTsDcpLseBase pins for the combine, to the single-rank result."""
        _require_prims_ts(self)
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


class TestPrimsTsNvfp4(CustomTestCase):
    def _check_nvfp4_matches_reference(self, rows: int):
        """The same check over the 352 B/token NVFP4 cache that the prims_ts NVFP4
        route writes, against FP32 attention over the values its loader expands
        (E2M1 x block scale rounded to E4M3, E4M3 RoPE). Rows past their count are
        disabled with a zero length, as DP padding rows are."""
        from sglang.kernels.ops.attention.mla_qk_prep import (
            concat_and_cast_q_fp8_heads,
        )
        from sglang.srt.layers.attention.dsa.prims_k_cache import (
            quantize_prims_k_cache_into,
        )

        heads, ctx = 64, 16384
        _require_prims_ts(self, heads, "nvfp4.fp8rope")
        gen = torch.Generator(device="cuda").manual_seed(5)
        kv = torch.zeros((ctx, 1, 352), dtype=torch.uint8, device="cuda")
        quantize_prims_k_cache_into(
            torch.randn(ctx, KV_LORA, generator=gen, device="cuda").bfloat16(),
            torch.randn(ctx, ROPE, generator=gen, device="cuda").bfloat16(),
            kv,
            torch.arange(ctx, device="cuda"),
        )
        q8 = torch.empty(rows, heads, DIM, dtype=FP8, device="cuda")
        concat_and_cast_q_fp8_heads(
            q8,
            torch.randn(rows, heads, KV_LORA, generator=gen, device="cuda").bfloat16(),
            torch.randn(rows, heads, ROPE, generator=gen, device="cuda").bfloat16(),
        )
        live = rows - 1
        slots = torch.full((rows, TOPK), -1, dtype=torch.int32, device="cuda")
        for r in range(rows):
            pick = torch.randperm(ctx, generator=gen, device="cuda")[:TOPK]
            # interior -1 holes, as the indexer leaves past a short context
            holes = torch.rand(TOPK, generator=gen, device="cuda") < 0.25
            slots[r] = torch.where(holes, -1, pick.to(torch.int32))
        lengths = torch.full((rows,), TOPK, dtype=torch.int32, device="cuda")
        lengths[live:] = 0

        runner = PrimsTSSparseMLA(torch.device("cuda"), "nvfp4.fp8rope")
        out, lse = runner.forward(
            q=q8,
            kv_cache=kv,
            indices=slots,
            lengths=lengths,
            sm_scale=SM_SCALE,
            kv_scale=1.0,
        )
        packed = rows > _EXACT_PLAN_MAX_ROWS
        self.assertEqual({key[3] for key in runner._plans}, {packed})

        qf = q8.float()
        ref = torch.empty((live, heads, KV_LORA), dtype=torch.float32, device="cuda")
        ref_lse = torch.empty((live, heads), dtype=torch.float32, device="cuda")
        for r in range(live):
            valid = slots[r][slots[r] >= 0]
            k = _expand_nvfp4_rows(kv, valid)
            s = (qf[r] @ k.T) * SM_SCALE
            ref_lse[r] = torch.logsumexp(s, dim=-1)
            ref[r] = torch.softmax(s, dim=-1) @ k[:, :KV_LORA]

        got = out[:live].float()
        self.assertLess(((got - ref).norm() / ref.norm()).item(), 0.03)
        torch.testing.assert_close(lse[:live], ref_lse, atol=2e-2, rtol=2e-3)
        # The zero-length row reads none of its indices.
        self.assertTrue(torch.all(out[live:] == 0))
        self.assertTrue(torch.all(torch.isneginf(lse[live:])))

    def test_writer_layout_and_rounding(self):
        """Blocks of E2M1 multiples of a power-of-two scale round-trip exactly;
        other values land within one E2M1 step; -1 locations and the reserved
        slot 0 are never written."""
        from sglang.srt.layers.attention.dsa.prims_k_cache import (
            quantize_prims_k_cache_into,
        )

        if not torch.cuda.is_available():
            self.skipTest("needs CUDA")
        gen = torch.Generator(device="cuda").manual_seed(7)
        n = 64
        grid = torch.tensor(_E2M1, device="cuda")
        codes = torch.randint(0, 8, (n, 32, 16), generator=gen, device="cuda")
        codes[:, :, 0] = 7  # amax = 6 x scale, so the stored scale is exact
        sign = torch.where(
            torch.rand(n, 32, 16, generator=gen, device="cuda") < 0.5, -1.0, 1.0
        )
        scale = 2.0 ** torch.randint(-6, 4, (n, 32, 1), generator=gen, device="cuda")
        exact = (grid[codes] * sign * scale).view(n, KV_LORA)
        rope = torch.randn(n, ROPE, generator=gen, device="cuda").bfloat16()
        kv = torch.full((n + 2, 1, 352), 0xAB, dtype=torch.uint8, device="cuda")
        loc = torch.arange(1, n + 1, device="cuda")
        quantize_prims_k_cache_into(exact.bfloat16(), rope, kv, loc)
        # padding tokens point at slot 0, which the writer skips
        quantize_prims_k_cache_into(
            exact[:2].bfloat16(), rope[:2], kv, loc.new_zeros(2)
        )
        self.assertTrue(torch.all(kv[0] == 0xAB))
        got = _expand_nvfp4_rows(kv, loc)
        torch.testing.assert_close(got[:, :KV_LORA], exact, atol=0, rtol=0)
        torch.testing.assert_close(got[:, KV_LORA:], rope.float().to(FP8).float())
        self.assertTrue(torch.all(kv[n + 1] == 0xAB))

        noisy = torch.randn(n, KV_LORA, generator=gen, device="cuda").bfloat16()
        dropped = torch.cat([loc[:-1], loc.new_tensor([-1])])
        quantize_prims_k_cache_into(noisy, rope, kv, dropped)
        got = _expand_nvfp4_rows(kv, loc[:-1])[:, :KV_LORA].view(-1, 32, 16)
        x = noisy[:-1].float().view(-1, 32, 16)
        # E2M1 steps are at most 2 x scale = amax / 3; rounding is within one step.
        step = x.abs().amax(-1, keepdim=True) / 3
        self.assertTrue(torch.all((got - x).abs() <= step * 1.01 + 1e-6))
        self.assertTrue(torch.all(kv[n + 1] == 0xAB))

    def test_nvfp4_exact_plan_matches_reference(self):
        self._check_nvfp4_matches_reference(rows=8)

    def test_nvfp4_packed_plan_matches_reference(self):
        self._check_nvfp4_matches_reference(rows=_EXACT_PLAN_MAX_ROWS + 1)


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
