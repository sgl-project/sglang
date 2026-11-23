"""prims_ts DSA backend: FlashInfer Prims-TS sparse MLA over raw FP8 MLA rows (E4M3
q, natural-log LSE). Under sharded DCP each rank attends over its owned top-k rows
(compacted, lengths = owned count); the partials merge to the single-rank result,
and the DCP combine must merge them base e."""

from __future__ import annotations

import sys
import unittest
from types import ModuleType, SimpleNamespace
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

    def test_prepare_compiles_every_plan_before_serving(self):
        """After prepare() serving compiles nothing, and each plan runs the same
        compiled kernel, with the same output, as an unprepared lazy plan."""
        self._require_prims_ts()
        from flashinfer.attention.prims_ts import mla_decode

        heads = 96
        prepared = PrimsTSSparseMLA(torch.device("cuda"))
        prepared.prepare(heads=[heads], width=TOPK)
        misses = mla_decode._get_compiled_mla_decode.cache_info().misses
        lazy = PrimsTSSparseMLA(torch.device("cuda"))
        gen = torch.Generator(device="cuda").manual_seed(5)
        kv = (torch.randn(4096, DIM, generator=gen, device="cuda") * 0.5).to(FP8)
        for rows in (1, 5, 24, 153, _EXACT_PLAN_MAX_ROWS, _EXACT_PLAN_MAX_ROWS + 1):
            q = torch.randn(rows, heads, DIM, generator=gen, device="cuda").to(FP8)
            slots = torch.randint(
                0, 4096, (rows, TOPK), generator=gen, device="cuda", dtype=torch.int32
            )
            lens = torch.randint(
                1, TOPK + 1, (rows,), generator=gen, device="cuda", dtype=torch.int32
            )
            kwargs = dict(
                q=q, kv_cache=kv, indices=slots, lengths=lens, sm_scale=SM_SCALE
            )
            out, lse = prepared.forward(kv_scale=1.0, **kwargs)
            ref, ref_lse = lazy.forward(kv_scale=1.0, **kwargs)
            self.assertTrue(torch.equal(out, ref) and torch.equal(lse, ref_lse))
        for key, wrapper in lazy._plans.items():
            self.assertIs(
                prepared._plans[key]._state["compiled"], wrapper._state["compiled"]
            )
        self.assertEqual(
            mla_decode._get_compiled_mla_decode.cache_info().misses, misses
        )


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


# CPU torch: no stream is capturing.
@mock.patch.object(torch.cuda, "is_current_stream_capturing", new=lambda: False)
class TestPrimsTsPrepare(CustomTestCase):
    """prepare() plans once per kernel FlashInfer compiles over every row count a
    launch can plan, so serving compiles nothing. A stand-in FlashInfer resolves
    plans from their arguments and, like 0.7.4 plan(), compiles the resolved spec
    with the profile's split count (2CTA fuses the epilogue only unsplit)."""

    def _flashinfer(self, *, resolve: bool = True):
        compiled, plans = [], []

        def resolve_plan(device, batch_size, heads, **kwargs):
            rows = batch_size * kwargs["max_q_len"]
            # Schedules change with rows: one only between powers of two, one
            # far past the exact plans.
            schedule = (min(rows.bit_length(), 6), 600 < rows <= 1000, rows >= 2**20)
            spec = (heads, schedule, *sorted(kwargs.items()))
            # The split count is not part of the spec.
            profile = ("2cta", 1 + rows % 2)
            return device, profile, spec, None, None, None, None

        class Wrapper:
            def plan(self, *args, **kwargs):
                _, profile, spec, *_ = resolve_plan(*args, **kwargs)
                self.kernel = (spec, profile[1] == 1)
                plans.append(self.kernel)
                if self.kernel not in compiled:
                    compiled.append(self.kernel)

            def run(self, *args, **kwargs):
                pass

        modules = {
            name: ModuleType(name)
            for name in (
                "flashinfer",
                "flashinfer.attention",
                "flashinfer.attention.prims_ts",
                "flashinfer.attention.prims_ts.mla_decode",
            )
        }
        modules[
            "flashinfer.attention.prims_ts"
        ].BatchSparseMLADecodePagedTSWrapper = Wrapper
        if resolve:
            modules[
                "flashinfer.attention.prims_ts.mla_decode"
            ]._resolve_sparse_mla_plan = resolve_plan
        return mock.patch.dict(sys.modules, modules), compiled, plans

    @staticmethod
    def _forward(runner, rows: int, heads: int):
        runner.forward(
            q=torch.empty(1, heads, DIM, dtype=FP8).expand(rows, -1, -1),
            kv_cache=torch.empty(1, DIM, dtype=FP8),
            indices=torch.empty(1, TOPK, dtype=torch.int32).expand(rows, -1),
            lengths=torch.empty(1, dtype=torch.int32).expand(rows),
            sm_scale=SM_SCALE,
            kv_scale=1.0,
        )

    def test_prepare_plans_each_compile_spec_once_and_serving_compiles_none(self):
        modules, compiled, plans = self._flashinfer()
        runner = PrimsTSSparseMLA(torch.device("cpu", 0))
        with modules:
            # A head count repeated across phases (no DCP) prepares once.
            runner.prepare(heads=[12, 96, 12], width=TOPK)
            prepared = list(plans)
            self.assertEqual(len(prepared), len(set(prepared)))
            self.assertEqual(compiled, prepared)
            # Equal specs with different split counts plan both kernels.
            self.assertGreater(len(prepared), len({spec for spec, _ in prepared}))
            for heads in (12, 96):
                for rows in range(1, 2 * _EXACT_PLAN_MAX_ROWS + 2):
                    self._forward(runner, rows, heads)
                # Larger launches plan per power-of-two capacity, up to FlashInfer's
                # int32 rows x heads; forward plans each through _wrapper.
                capacity = 4 * _EXACT_PLAN_MAX_ROWS
                while capacity * heads < 2**31:
                    for rows in (capacity // 2 + 1, capacity):
                        runner._wrapper(rows=rows, heads=heads, width=TOPK)
                    capacity *= 2
        self.assertEqual(compiled, prepared)
        self.assertEqual({w.kernel for w in runner._plans.values()}, set(prepared))

    def test_prepare_without_flashinfer_spec_resolver_compiles_lazily(self):
        modules, compiled, plans = self._flashinfer(resolve=False)
        runner = PrimsTSSparseMLA(torch.device("cpu", 0))
        with modules:
            with self.assertLogs(
                "sglang.srt.layers.attention.dsa.prims_ts", "WARNING"
            ) as logs:
                runner.prepare(heads=[12, 96], width=TOPK)
            self.assertEqual(len(logs.output), 1)
            self.assertEqual(plans, [])
            self._forward(runner, 3, 12)
        self.assertEqual(len(compiled), 1)


if __name__ == "__main__":
    unittest.main()
