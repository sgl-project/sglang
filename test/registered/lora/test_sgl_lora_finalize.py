"""Independent-reference tests for the Step-6 finalizer candidates.

What these cases guard (plan §65.2 correctness obligations):

* semantic agreement with a plain-torch FP32 oracle — ``token_out_ref =
  base + sum over valid pairs of w[t, k] * (bridge[pair] @
  b_down[veid]^T)`` — for every family at both ownerships, with
  NON-UNIT combine weights so an exactly-once violation (dropped,
  doubled, or unweighted application) errs at the full signal scale;
* the NONZERO-BASE contract: the destination arrives carrying the base
  model output and every family ACCUMULATES into it; a token with no
  valid pairs, or whose combine weights are all zero, keeps its base
  row BITWISE intact;
* caller-selected FP32 destination alongside the BF16 default;
* invalid pairs contribute exactly zero AND never read their inputs:
  NaN-poisoned bridge rows and combine weights at invalid pairs must
  not leak (the masked-load guard against ``0 * inf``);
* deterministic families' replay stability: fixed-order FP32
  accumulation with one writing program per destination cell must be
  bitwise stable across eager replays and identical under CUDA graph
  capture/replay;
* the fail-closed vocabulary: impossible specs, route/ownership
  mismatches, wrong dtypes, and missing or silently-ignored family
  arguments all raise instead of degrading.
"""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")

from benchmark.kernels.lora_moe.finalize_candidates import (
    MATERIALIZED_FINALIZE_DEFAULT_CONFIG,
    SHARED_RANK_REDUCE_DEFAULT_CONFIG,
    TOKEN_FINALIZE_DEFAULT_CONFIG,
    FinalizeExecutionSpec,
    build_shared_finalize_info,
    run_finalize,
)
from benchmark.kernels.lora_moe.signal_gates import require_delta_close, require_finite
from sglang.srt.lora.sgl_lora.routing import ROUTE_RAW, build_virtual_expert_routing

CONFIGS = {
    "materialized": MATERIALIZED_FINALIZE_DEFAULT_CONFIG,
    "token_owned": TOKEN_FINALIZE_DEFAULT_CONFIG,
    "shared_rank_reduce": SHARED_RANK_REDUCE_DEFAULT_CONFIG,
}
OWNERSHIPS = ("per_expert", "shared")


def _families(ownership: str) -> tuple[str, ...]:
    if ownership == "shared":
        return ("materialized", "token_owned", "shared_rank_reduce")
    return ("materialized", "token_owned")


class _Fixture:
    """Mixed-validity finalize workload with an independent validity mask.

    Guaranteed structure regardless of the random draw: token 0 is a BASE
    token (slot -1), token 1 has a slot but only invalid expert ids, and
    tokens 2 and 3 are fully valid — so base-preservation, invalid-pair,
    and zero-weight rows all exist deterministically.
    """

    def __init__(
        self,
        device,
        *,
        ownership="per_expert",
        num_tokens=12,
        top_k=4,
        lora_experts_per_adapter=6,
        max_loras=3,
        rank=32,
        hidden=96,
        dest_dtype=torch.bfloat16,
        seed=31,
    ):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.ownership = ownership
        self.top_k = top_k
        self.rank = rank
        self.hidden = hidden
        self.max_loras = max_loras
        self.experts = lora_experts_per_adapter
        self.num_pairs = num_tokens * top_k
        # -1 sentinels and out-of-range ids exercise every invalidity kind.
        topk_ids = torch.randint(
            -1,
            lora_experts_per_adapter + 2,
            (num_tokens, top_k),
            generator=generator,
            dtype=torch.int32,
        )
        token_slots = torch.randint(
            -1, max_loras, (num_tokens,), generator=generator, dtype=torch.int32
        )
        token_slots[0] = -1
        topk_ids[1, :] = -1
        token_slots[2], token_slots[3] = 0, 1 % max_loras
        topk_ids[2] = torch.arange(top_k, dtype=torch.int32) % lora_experts_per_adapter
        topk_ids[3] = (
            torch.arange(top_k, dtype=torch.int32) + 1
        ) % lora_experts_per_adapter
        self.topk_ids = topk_ids.to(device)
        self.token_slots = token_slots.to(device)
        if ownership == "shared":
            groups = max_loras
            self.route_kwargs = dict(
                lora_experts_per_adapter=1,
                max_loras=max_loras,
                block_size=16,
                shared_outer_local_expert_count=lora_experts_per_adapter,
            )
        else:
            groups = max_loras * lora_experts_per_adapter
            self.route_kwargs = dict(
                lora_experts_per_adapter=lora_experts_per_adapter,
                max_loras=max_loras,
                block_size=16,
            )
        self.b_down = (
            (
                torch.randn(
                    groups, hidden, rank, generator=generator, dtype=torch.float32
                )
                * rank**-0.5
            )
            .to(torch.bfloat16)
            .to(device)
        )
        self.bridge = (
            (
                torch.randn(
                    self.num_pairs, rank, generator=generator, dtype=torch.float32
                )
                * 0.5
            )
            .to(torch.bfloat16)
            .to(device)
        )
        # Non-unit weights everywhere: an exactly-once violation cannot hide.
        self.combine_weights = (
            torch.rand((num_tokens, top_k), generator=generator, dtype=torch.float32)
            * 1.75
            + 0.25
        ).to(device)
        self.base = (
            torch.randn(num_tokens, hidden, generator=generator, dtype=torch.float32)
            .to(dest_dtype)
            .to(device)
        )
        # Host validity + veid: the same key definition, computed independently.
        ids = topk_ids.to(torch.int64).reshape(-1)
        slots = token_slots.to(torch.int64)[:, None].expand_as(topk_ids).reshape(-1)
        valid = (
            (slots >= 0)
            & (slots < max_loras)
            & (ids >= 0)
            & (ids < lora_experts_per_adapter)
        )
        raw_veid = slots if ownership == "shared" else slots * self.experts + ids
        self.veid = raw_veid.clamp(0, groups - 1).to(device)
        self.valid = valid.to(device)
        self.pair_delta = self.fp32_pair_delta().to(torch.bfloat16)
        self.finalize_info = (
            build_shared_finalize_info(self.token_slots, max_loras=max_loras, rank=rank)
            if ownership == "shared"
            else None
        )

    def route(self):
        return build_virtual_expert_routing(
            self.topk_ids, self.token_slots, view=ROUTE_RAW, **self.route_kwargs
        )

    def fp32_pair_delta(self) -> torch.Tensor:
        """[P, H] FP32 per-pair down-B delta, exact zeros at invalid pairs."""
        gathered = self.b_down.float()[self.veid]  # [P, H, R]
        delta = torch.einsum("pr,phr->ph", self.bridge.float(), gathered)
        delta[~self.valid] = 0
        return delta

    def fp32_oracle(self, combine_weights=None) -> torch.Tensor:
        """Independent FP32 reference: plain torch over the same bf16 inputs."""
        weights = self.combine_weights if combine_weights is None else combine_weights
        weighted = self.fp32_pair_delta() * weights.reshape(-1)[:, None]
        token_delta = weighted.reshape(weights.shape[0], self.top_k, self.hidden).sum(
            dim=1
        )
        return self.base.float() + token_delta


class TestFinalizeCandidates(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        cls.device = torch.device("cuda")

    def _fixture(self, ownership, **kwargs):
        # Shared fixtures run a non-power-of-two rank so the BLOCK_K,
        # BLOCK_R, and SGMV tail masks are all exercised; per-expert keeps
        # rank == BLOCK_SIZE_K to cover the exact-fit path.
        if ownership == "shared":
            kwargs.setdefault("rank", 24)
        return _Fixture(self.device, ownership=ownership, **kwargs)

    def _apply(
        self,
        fixture,
        family,
        out,
        *,
        combine_weights=None,
        tok_bridge=None,
    ):
        spec = FinalizeExecutionSpec(family=family, ownership=fixture.ownership)
        kwargs = {}
        if family == "materialized":
            kwargs["pair_delta"] = fixture.pair_delta
        elif family == "token_owned":
            kwargs.update(bridge=fixture.bridge, b_down=fixture.b_down)
        else:
            kwargs.update(
                bridge=fixture.bridge,
                b_down=fixture.b_down,
                finalize_info=fixture.finalize_info,
                tok_bridge=tok_bridge,
            )
        run_finalize(
            spec,
            routing=fixture.route(),
            combine_weights=(
                fixture.combine_weights if combine_weights is None else combine_weights
            ),
            token_out=out,
            config=CONFIGS[family],
            **kwargs,
        )

    def test_families_match_fp32_oracle_with_nonunit_weights(self):
        # Exactly-once application: weights are drawn in [0.25, 2.0), so a
        # dropped, doubled, or unweighted pair contribution errs at the
        # signal scale and the S/10 gate turns red.
        for ownership in OWNERSHIPS:
            fixture = self._fixture(ownership)
            self.assertTrue(bool(fixture.valid.any()) and not bool(fixture.valid.all()))
            oracle_delta = fixture.fp32_oracle() - fixture.base.float()
            for family in _families(ownership):
                with self.subTest(ownership=ownership, family=family):
                    out = fixture.base.clone()
                    self._apply(fixture, family, out)
                    require_delta_close(
                        out.float() - fixture.base.float(),
                        oracle_delta,
                        gate_dtype=torch.bfloat16,
                        label=f"{family}/{ownership} vs fp32 oracle",
                    )

    def test_fp32_destination_variant(self):
        # The caller-selected FP32 destination: same oracle, exact base
        # extraction (no bf16 storage rounding on the accumulate).
        for ownership in OWNERSHIPS:
            fixture = self._fixture(ownership, dest_dtype=torch.float32)
            oracle_delta = fixture.fp32_oracle() - fixture.base.float()
            for family in _families(ownership):
                with self.subTest(ownership=ownership, family=family):
                    out = fixture.base.clone()
                    self._apply(fixture, family, out)
                    self.assertEqual(out.dtype, torch.float32)
                    require_delta_close(
                        out - fixture.base,
                        oracle_delta,
                        gate_dtype=torch.bfloat16,
                        label=f"{family}/{ownership} fp32 destination",
                    )

    def test_no_valid_pair_rows_preserve_base_bitwise(self):
        # Token 0 (base token) and token 1 (all pairs invalid) own no valid
        # pair; their nonzero base rows must survive BITWISE — a finalizer
        # that writes anything else corrupts the base model output.
        for ownership in OWNERSHIPS:
            fixture = self._fixture(ownership)
            untouched = ~fixture.valid.reshape(-1, fixture.top_k).any(dim=1)
            self.assertGreaterEqual(int(untouched.sum()), 2)
            for family in _families(ownership):
                with self.subTest(ownership=ownership, family=family):
                    out = fixture.base.clone()
                    self._apply(fixture, family, out)
                    self.assertTrue(
                        torch.equal(out[untouched], fixture.base[untouched]),
                        f"{family}: a no-valid-pair base row changed",
                    )

    def test_zero_weight_rows_preserve_base_bitwise(self):
        # Zero routed weight must contribute EXACT zero even against a
        # nonzero GEMM result (0 * part) — token 3 is fully valid by
        # construction, so its preservation can only come from the weight.
        for ownership in OWNERSHIPS:
            fixture = self._fixture(ownership)
            weights = fixture.combine_weights.clone()
            weights[3, :] = 0.0
            self.assertTrue(bool(fixture.valid.reshape(-1, fixture.top_k)[3].all()))
            oracle_delta = fixture.fp32_oracle(weights) - fixture.base.float()
            for family in _families(ownership):
                with self.subTest(ownership=ownership, family=family):
                    out = fixture.base.clone()
                    self._apply(fixture, family, out, combine_weights=weights)
                    self.assertTrue(
                        torch.equal(out[3], fixture.base[3]),
                        f"{family}: zero-weight token row must stay bitwise base",
                    )
                    require_delta_close(
                        out.float() - fixture.base.float(),
                        oracle_delta,
                        gate_dtype=torch.bfloat16,
                        label=f"{family}/{ownership} with a zero-weight token",
                    )

    def test_invalid_pairs_never_read_bridge_or_weights(self):
        # NaN-poison the bridge rows AND combine weights of every invalid
        # pair: the route-deriving families must mask those loads (the
        # ``0 * inf`` guard), leaving the output finite and oracle-close.
        # The materialized family is excluded by design — its invalid-pair
        # guarantee is inherited from B's zero-fill contract, and the
        # production combine reads every (finite) routed weight.
        for ownership in OWNERSHIPS:
            fixture = self._fixture(ownership)
            oracle_delta = fixture.fp32_oracle() - fixture.base.float()
            fixture.bridge[~fixture.valid] = float("nan")
            poisoned_weights = fixture.combine_weights.clone()
            poisoned_weights[~fixture.valid.reshape(-1, fixture.top_k)] = float("nan")
            families = tuple(f for f in _families(ownership) if f != "materialized")
            for family in families:
                with self.subTest(ownership=ownership, family=family):
                    out = fixture.base.clone()
                    self._apply(fixture, family, out, combine_weights=poisoned_weights)
                    require_finite(out, label=f"{family}/{ownership} poison")
                    require_delta_close(
                        out.float() - fixture.base.float(),
                        oracle_delta,
                        gate_dtype=torch.bfloat16,
                        label=f"{family}/{ownership} with poisoned invalid pairs",
                    )

    def test_bitwise_repeatability_across_replays_and_graph(self):
        # Every family claims determinism: FP32 accumulation in a fixed
        # serial order with one writing program per destination cell — no
        # atomics anywhere. 32 observed eager replays, then CUDA graph
        # capture with every replay observed (sync + compare each time;
        # checking once would inspect only the final overwrite).
        for ownership in OWNERSHIPS:
            fixture = self._fixture(ownership)
            tok_bridge = torch.empty(
                (fixture.topk_ids.shape[0], fixture.rank),
                dtype=fixture.b_down.dtype,
                device=self.device,
            )
            for family in _families(ownership):
                with self.subTest(ownership=ownership, family=family):
                    out = fixture.base.clone()
                    self._apply(fixture, family, out, tok_bridge=tok_bridge)
                    first = out.clone()
                    for _ in range(32):
                        out.copy_(fixture.base)
                        self._apply(fixture, family, out, tok_bridge=tok_bridge)
                        torch.cuda.synchronize()
                        self.assertTrue(torch.equal(first, out))
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        self._apply(fixture, family, out, tok_bridge=tok_bridge)
                    for _ in range(100):
                        out.copy_(fixture.base)
                        graph.replay()
                        torch.cuda.synchronize()
                        self.assertTrue(torch.equal(first, out))

    def test_spec_vocabulary_is_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "precondition"):
            FinalizeExecutionSpec(family="shared_rank_reduce", ownership="per_expert")
        with self.assertRaisesRegex(ValueError, "not one of"):
            FinalizeExecutionSpec(family="atomic", ownership="per_expert")
        with self.assertRaisesRegex(ValueError, "not one of"):
            FinalizeExecutionSpec(family="token_owned", ownership="global")

    def test_argument_and_route_coupling_is_fail_closed(self):
        per_expert = self._fixture("per_expert")
        shared = self._fixture("shared")
        base_kwargs = dict(
            routing=per_expert.route(),
            combine_weights=per_expert.combine_weights,
            token_out=per_expert.base.clone(),
        )
        with self.assertRaisesRegex(ValueError, "does not take"):
            run_finalize(
                FinalizeExecutionSpec(family="materialized", ownership="per_expert"),
                config=CONFIGS["materialized"],
                pair_delta=per_expert.pair_delta,
                bridge=per_expert.bridge,
                **base_kwargs,
            )
        with self.assertRaisesRegex(ValueError, "requires b_down"):
            run_finalize(
                FinalizeExecutionSpec(family="token_owned", ownership="per_expert"),
                config=CONFIGS["token_owned"],
                bridge=per_expert.bridge,
                **base_kwargs,
            )
        with self.assertRaisesRegex(ValueError, "requires finalize_info"):
            run_finalize(
                FinalizeExecutionSpec(family="shared_rank_reduce", ownership="shared"),
                routing=shared.route(),
                combine_weights=shared.combine_weights,
                token_out=shared.base.clone(),
                config=CONFIGS["shared_rank_reduce"],
                bridge=shared.bridge,
                b_down=shared.b_down,
            )
        # Ownership x route form mismatches, both directions.
        with self.assertRaisesRegex(ValueError, "shared-outer route form"):
            run_finalize(
                FinalizeExecutionSpec(family="token_owned", ownership="shared"),
                config=CONFIGS["token_owned"],
                bridge=per_expert.bridge,
                b_down=per_expert.b_down,
                **base_kwargs,
            )
        with self.assertRaisesRegex(ValueError, "contradictory"):
            run_finalize(
                FinalizeExecutionSpec(family="token_owned", ownership="per_expert"),
                routing=shared.route(),
                combine_weights=shared.combine_weights,
                token_out=shared.base.clone(),
                config=CONFIGS["token_owned"],
                bridge=shared.bridge,
                b_down=shared.b_down,
            )

    def test_dtype_contracts_are_fail_closed(self):
        fixture = self._fixture("per_expert")
        with self.assertRaisesRegex(ValueError, "float32"):
            run_finalize(
                FinalizeExecutionSpec(family="token_owned", ownership="per_expert"),
                routing=fixture.route(),
                combine_weights=fixture.combine_weights.to(torch.bfloat16),
                token_out=fixture.base.clone(),
                config=CONFIGS["token_owned"],
                bridge=fixture.bridge,
                b_down=fixture.b_down,
            )
        with self.assertRaisesRegex(ValueError, "dtype"):
            run_finalize(
                FinalizeExecutionSpec(family="token_owned", ownership="per_expert"),
                routing=fixture.route(),
                combine_weights=fixture.combine_weights,
                token_out=fixture.base.to(torch.float16),
                config=CONFIGS["token_owned"],
                bridge=fixture.bridge,
                b_down=fixture.b_down,
            )
        shared = self._fixture("shared")
        with self.assertRaisesRegex(ValueError, "matching operands"):
            run_finalize(
                FinalizeExecutionSpec(family="shared_rank_reduce", ownership="shared"),
                routing=shared.route(),
                combine_weights=shared.combine_weights,
                token_out=shared.base.clone(),
                config=CONFIGS["shared_rank_reduce"],
                bridge=shared.bridge,
                b_down=shared.b_down,
                tok_bridge=torch.zeros(
                    (shared.topk_ids.shape[0], shared.rank),
                    dtype=torch.float32,
                    device=self.device,
                ),
                finalize_info=shared.finalize_info,
            )
        with self.assertRaisesRegex(ValueError, "out of bounds"):
            build_shared_finalize_info(
                torch.full((4,), 99, dtype=torch.int32, device=self.device),
                max_loras=3,
                rank=16,
            )


if __name__ == "__main__":
    unittest.main()


class TestBenchConfigIdentity(CustomTestCase):
    """CPU-only identity/selection guards for bench_finalize (15th/18th
    S5/6 reviews). Moved here from the fused-middle test file so the
    Step-5 commit is self-consistent without Step 6."""

    def test_fshared_phase_competition_cannot_forfeit_the_default(self):
        """18th review: phase 2 excluded the default reduce and then
        unconditionally overwrote phase 1 — a default faster than every
        alternative lost. The phases must COMPETE on median."""
        from benchmark.kernels.lora_moe import bench_finalize as bench

        default = ({"reduce": "d", "gemm": "g"}, 10.0)
        slower_alt = ({"reduce": "a", "gemm": "g"}, 12.0)
        faster_alt = ({"reduce": "a", "gemm": "g"}, 8.0)
        self.assertEqual(bench.select_between_phases(default, slower_alt), default)
        self.assertEqual(bench.select_between_phases(default, faster_alt), faster_alt)
        # all alternatives skipped -> the phase-1 winner stands
        self.assertEqual(
            bench.select_between_phases(default, (None, float("inf"))), default
        )
        # and production must route through THIS function (the sweep
        # loop lives in _run_sweeps, not main)
        import inspect

        source = inspect.getsource(bench._run_sweeps)
        self.assertIn("select_between_phases(", source)

    def test_bf16_admission_uses_the_accumulate_form_gate(self):
        """Bug regression (first GB300 finalize bench run): every arm is
        accumulate-form (``token_out += delta`` onto a REAL bf16 base), so
        the extracted delta carries up to one bf16 ulp of the BASE per
        element — yet admission gated it with the delta-pure fixed 1e-2
        rel-L2 gate, which a PERFECT kernel cannot satisfy once
        L2(base)/L2(delta) exceeds ~2.6. The bench aborted on its first
        cell (dense/per_expert r16 decode_tiny: rel_l2 1.685e-2, max|err|
        exactly one ulp of the ~2.0 base). Reproduces the physics at gate
        level with the observed norm ratio, and pins that bf16 admission
        routes through the accumulate-form gate."""
        import inspect

        from benchmark.kernels.lora_moe import bench_finalize as bench
        from benchmark.kernels.lora_moe.bench_common import (
            require_delta_close_chunked,
        )
        from benchmark.kernels.lora_moe.signal_gates import require_delta_close

        torch.manual_seed(29)
        base = torch.empty(4, 4096, dtype=torch.float64).uniform_(0.5, 1.5)
        base *= torch.where(torch.rand_like(base) < 0.5, -1.0, 1.0)
        delta = torch.randn(4, 4096, dtype=torch.float64) / 13.0
        perfect = (base + delta).to(torch.bfloat16)
        # The pre-fix admission call rejects the PERFECT write...
        with self.assertRaises(AssertionError):
            require_delta_close(
                perfect.float() - base.to(torch.bfloat16).float(),
                delta.float(),
                gate_dtype=torch.bfloat16,
                label="pre-fix delta-pure admission",
            )
        # ...while the accumulate-form gate admits it.
        require_delta_close_chunked(
            perfect,
            delta.float(),
            gate_dtype=torch.bfloat16,
            label="accumulate-form admission",
            observed_base=base.to(torch.bfloat16),
        )
        # Production bf16 admission must route through the chunked
        # accumulate-form gate with the matched base.
        source = inspect.getsource(bench._admit)
        self.assertIn("require_delta_close_chunked(", source)
        self.assertIn("observed_base=base", source)

    def test_accumulate_cell_probe_skips_undecidable_cells(self):
        """The accumulate-form allowance is bounded (ceiling 0.125): a
        cell whose base drowns the delta cannot certify ANY bf16 arm, and
        that is a fixture property, not a config property. The cell probe
        must convert the gate's DegenerateSignalError into a recorded
        skip reason — and stay silent on certifiable cells — so one
        undecidable cell no longer aborts a multi-hour suite."""
        from types import SimpleNamespace

        from benchmark.kernels.lora_moe import bench_finalize as bench

        torch.manual_seed(29)
        base = torch.empty(4, 4096, dtype=torch.float64).uniform_(0.5, 1.5)
        base *= torch.where(torch.rand_like(base) < 0.5, -1.0, 1.0)

        def stub(ratio):
            delta = torch.randn(4, 4096, dtype=torch.float64) / ratio
            return SimpleNamespace(
                base={torch.bfloat16: base.to(torch.bfloat16)},
                oracle_delta=delta.float(),
            )

        self.assertIsNone(bench._accumulate_cell_skip(stub(13.0)))
        reason = bench._accumulate_cell_skip(stub(40.0))
        self.assertIsNotNone(reason)
        self.assertTrue(reason.startswith("degenerate:"), reason)
        # Both cell loops must consult the probe before tuning/deciding.
        import inspect

        for producer in (bench._run_sweeps, bench._run_decided):
            self.assertIn("_accumulate_cell_skip(", inspect.getsource(producer))

    def test_fshared_sections_are_independently_keyed(self):
        from benchmark.kernels.lora_moe import bench_finalize as bench

        base = {
            "reduce": {"BLOCK_SIZE_T": 32, "num_warps": 4, "num_stages": 2},
            "gemm": {
                "BLOCK_S": 16,
                "BLOCK_N": 256,
                "BLOCK_K": 16,
                "num_warps": 4,
                "num_stages": 3,
            },
        }
        other_reduce = {**base, "reduce": {**base["reduce"], "num_warps": 2}}
        other_gemm = {**base, "gemm": {**base["gemm"], "num_warps": 8}}
        keys = {
            bench._config_key("fshared", cfg)
            for cfg in (base, other_reduce, other_gemm)
        }
        self.assertEqual(len(keys), 3, "warps must be keyed PER SECTION")
