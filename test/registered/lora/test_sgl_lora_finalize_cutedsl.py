"""Independent-reference tests for the CuTeDSL Step-6 finalize arms.

What these cases guard (plan §65.2 correctness obligations):

* semantic agreement with a plain-torch FP32 oracle for BOTH arms — the
  routed weight applied EXACTLY ONCE (token 0 forces duplicate experts in
  one row, so double-application cannot hide behind distinct ids), on both
  the BF16 and the caller-selected FP32 destination;
* the NONZERO-base contract: rows of ``token_out`` owned by no valid pair
  must survive BITWISE, even when the plan's scratch (staged/C) and the
  contractually-undefined invalid bridge rows are NaN-poisoned — a leak
  here means the scatter read an unwritten C row or an undefined bridge
  row through a zero weight;
* bitwise repeatability at the prepared boundary: eager replays and
  OBSERVED CUDA-graph replays produce identical sums, and a metadata
  rebuild (which re-runs the nondeterministic dispatch atomics) leaves the
  output bitwise unchanged — placement invariance, the same claim the
  LoRA-A plan pins;
* fail-closed IO and binding: non-fp32 combine weights, non-bf16/fp32
  destinations, a per-expert route handed to the BY-ADAPTER shared
  builder, missing/foreign/aliased bindings, and same-capacity route
  content mutation at rebuild must all raise instead of silently
  mis-finalizing.
"""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-small")

from benchmark.kernels.lora_moe.signal_gates import (
    nan_poison_,
    require_delta_close,
    require_finite,
)
from sglang.srt.lora.sgl_lora.routing import (
    ROUTE_FUSED_IDS,
    build_virtual_expert_routing,
)

ARMS = ("shared", "token")


class _Fixture:
    """Small mixed-validity finalize workload with independent validity.

    Deterministic coverage regardless of seed: token 0 repeats ONE expert
    across every k (exactly-once weighting cannot hide behind distinct
    ids), token 1 has no valid pair (bitwise base preservation).
    """

    def __init__(
        self,
        device,
        *,
        num_tokens=12,
        top_k=4,
        experts=6,
        max_loras=3,
        rank=32,
        hidden=256,
        seed=29,
    ):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.num_tokens, self.top_k = num_tokens, top_k
        self.experts, self.max_loras = experts, max_loras
        self.rank, self.hidden = rank, hidden
        self.num_pairs = num_tokens * top_k
        topk_ids = torch.randint(
            -1,
            experts + 2,
            (num_tokens, top_k),
            generator=generator,
            dtype=torch.int32,
        )
        token_slots = torch.randint(
            -1, max_loras, (num_tokens,), generator=generator, dtype=torch.int32
        )
        topk_ids[0] = 2
        token_slots[0] = 1
        topk_ids[1] = -1
        self.topk_ids = topk_ids.to(device)
        self.token_slots = token_slots.to(device)
        self.weight_token = (
            (
                torch.randn(
                    max_loras * experts, hidden, rank, generator=generator
                )
                * rank**-0.5
            )
            .to(torch.bfloat16)
            .to(device)
        )
        self.weight_shared = (
            (torch.randn(max_loras, hidden, rank, generator=generator) * rank**-0.5)
            .to(torch.bfloat16)
            .to(device)
        )
        self.bridge = (
            (torch.randn(self.num_pairs, rank, generator=generator) * 0.5)
            .to(torch.bfloat16)
            .to(device)
        )
        self.combine_weights = torch.rand(
            num_tokens, top_k, generator=generator, dtype=torch.float32
        ).to(device)
        self.base = (
            (torch.randn(num_tokens, hidden, generator=generator) * 0.25)
            .to(torch.bfloat16)
            .to(device)
        )
        # Host validity: the same key definition, computed independently.
        ids = topk_ids.to(torch.int64)
        slots = token_slots.to(torch.int64)[:, None].expand_as(ids)
        self.pair_valid = (
            ((slots >= 0) & (slots < max_loras) & (ids >= 0) & (ids < experts))
            .reshape(-1)
            .to(device)
        )
        self.token_has_valid = self.pair_valid.view(num_tokens, top_k).any(dim=1)
        assert bool(self.pair_valid.any()) and not bool(self.pair_valid.all())
        assert not bool(self.token_has_valid[1])

    def route(self, arm: str):
        if arm == "shared":
            return build_virtual_expert_routing(
                self.topk_ids,
                self.token_slots,
                lora_experts_per_adapter=1,
                max_loras=self.max_loras,
                block_size=16,
                shared_outer_local_expert_count=self.experts,
                view=ROUTE_FUSED_IDS,
            )
        return build_virtual_expert_routing(
            self.topk_ids,
            self.token_slots,
            lora_experts_per_adapter=self.experts,
            max_loras=self.max_loras,
            block_size=16,
            view=ROUTE_FUSED_IDS,
        )

    def weight(self, arm: str) -> torch.Tensor:
        return self.weight_shared if arm == "shared" else self.weight_token

    def oracle_delta(self, arm: str) -> torch.Tensor:
        """Plain-torch FP32 reference of the finalize contribution [T, H]:
        every valid pair contributes ``w[t,k] * bridge[p] @ B[group]^T``
        exactly once; invalid pairs contribute exactly zero."""
        ids = self.topk_ids.to(torch.int64).reshape(-1)
        slots = (
            self.token_slots.to(torch.int64)[:, None]
            .expand_as(self.topk_ids)
            .reshape(-1)
        )
        if arm == "shared":
            group = slots.clamp(min=0, max=self.max_loras - 1)
            weight = self.weight_shared.float()[group]  # [P, H, R]
        else:
            veid = (slots * self.experts + ids).clamp(
                min=0, max=self.max_loras * self.experts - 1
            )
            weight = self.weight_token.float()[veid]
        pair_delta = torch.einsum("pr,phr->ph", self.bridge.float(), weight)
        pair_delta = pair_delta * self.combine_weights.reshape(-1)[:, None]
        pair_delta[~self.pair_valid] = 0
        return pair_delta.view(self.num_tokens, self.top_k, self.hidden).sum(dim=1)


class TestCutedslFinalizeArms(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        if torch.cuda.get_device_capability() < (9, 0):
            raise unittest.SkipTest("the masked grouped GEMM needs SM90+")
        try:
            import cutlass  # noqa: F401

            from benchmark.kernels.lora_moe.finalize_cutedsl import (  # noqa: F401
                build_cutedsl_shared_finalize_plan,
            )
        except ImportError as error:
            raise unittest.SkipTest(f"CuTeDSL unavailable: {error}")
        cls.device = torch.device("cuda")

    def _config(self):
        from benchmark.kernels.lora_moe.lora_a_cutedsl import (
            CutedslAConfig,
            supported_token_widths,
        )

        return CutedslAConfig(token_width=supported_token_widths(self.device)[0])

    def _plan(self, fixture, arm):
        from benchmark.kernels.lora_moe.finalize_cutedsl import (
            build_cutedsl_shared_finalize_plan,
            build_cutedsl_token_finalize_plan,
        )

        if arm == "shared":
            return build_cutedsl_shared_finalize_plan(
                shared_route=fixture.route(arm),
                down_weight=fixture.weight_shared,
                config=self._config(),
            )
        return build_cutedsl_token_finalize_plan(
            fused_route=fixture.route(arm),
            down_weight=fixture.weight_token,
            config=self._config(),
        )

    def _run(self, fixture, arm, plan, token_out):
        from benchmark.kernels.lora_moe.finalize_cutedsl import (
            cutedsl_shared_finalize,
            cutedsl_token_finalize,
        )

        entry = cutedsl_shared_finalize if arm == "shared" else cutedsl_token_finalize
        entry(
            plan=plan,
            bridge=fixture.bridge,
            combine_weights=fixture.combine_weights,
            token_out=token_out,
            weight=fixture.weight(arm),
            routing=fixture.route(arm),
        )
        return token_out

    def test_arms_match_fp32_oracle_on_bf16_and_fp32_destinations(self):
        fixture = _Fixture(self.device)
        for arm in ARMS:
            plan = self._plan(fixture, arm)
            oracle = fixture.oracle_delta(arm)
            for out_dtype in (torch.bfloat16, torch.float32):
                base = fixture.base.to(out_dtype)
                out = self._run(fixture, arm, plan, base.clone())
                observed = out.float() - base.float()
                with self.subTest(arm=arm, dtype=str(out_dtype)):
                    require_delta_close(
                        observed,
                        oracle,
                        gate_dtype=torch.bfloat16,
                        label=f"{arm} finalize vs fp32 oracle [{out_dtype}]",
                    )

    def test_foreign_rows_stay_bitwise_base_under_poison(self):
        # NaN-poison what the arms must NOT read: the contractually
        # undefined invalid bridge rows and the plans' scratch buffers.
        # A poisoned value reaching token_out means the scatter read an
        # unwritten C row (a src2dst sentinel bug) or the reduce multiplied
        # an undefined bridge row by a zero weight instead of masking it.
        fixture = _Fixture(self.device)
        nan_poison_(fixture.bridge, (~fixture.pair_valid)[:, None])
        for arm in ARMS:
            plan = self._plan(fixture, arm)
            nan_poison_(plan.staged)
            nan_poison_(plan.c)
            base = fixture.base.clone()
            out = self._run(fixture, arm, plan, base.clone())
            with self.subTest(arm=arm):
                require_finite(out, label=f"{arm} finalize output")
                self.assertTrue(
                    torch.equal(
                        out[~fixture.token_has_valid],
                        base[~fixture.token_has_valid],
                    ),
                    f"{arm}: a token with no valid pair must keep its base "
                    "row bitwise",
                )
                require_delta_close(
                    out.float() - base.float(),
                    fixture.oracle_delta(arm),
                    gate_dtype=torch.bfloat16,
                    label=f"{arm} finalize under poison",
                )

    def test_replays_and_observed_graph_replays_are_bitwise(self):
        fixture = _Fixture(self.device)
        for arm in ARMS:
            plan = self._plan(fixture, arm)
            base = fixture.base.clone()
            first = self._run(fixture, arm, plan, base.clone()).clone()
            for _ in range(16):
                again = self._run(fixture, arm, plan, base.clone())
                self.assertTrue(torch.equal(first, again), f"{arm} eager replay")
            out = base.clone()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out.copy_(base)
                plan.run(
                    bridge=fixture.bridge,
                    combine_weights=fixture.combine_weights,
                    token_out=out,
                )
            # Every replay must be OBSERVED — replaying N times and
            # checking once inspects only the final overwrite.
            for _ in range(50):
                graph.replay()
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(first, out), f"{arm} graph replay")

    def test_metadata_rebuild_is_output_invariant(self):
        # The dispatch atomics assign a fresh within-group row order every
        # rebuild; a stale inverse map (staging or scatter) silently
        # permutes or misroutes token contributions.
        fixture = _Fixture(self.device)
        for arm in ARMS:
            plan = self._plan(fixture, arm)
            base = fixture.base.clone()
            first = self._run(fixture, arm, plan, base.clone()).clone()
            plan.build_metadata()
            again = self._run(fixture, arm, plan, base.clone())
            self.assertTrue(
                torch.equal(first, again),
                f"{arm}: metadata rebuild changed the finalized output",
            )

    def test_run_io_contract_is_fail_closed(self):
        fixture = _Fixture(self.device)
        plan = self._plan(fixture, "token")
        base = fixture.base.clone()
        with self.assertRaisesRegex(ValueError, "float32"):
            plan.run(
                bridge=fixture.bridge,
                combine_weights=fixture.combine_weights.to(torch.bfloat16),
                token_out=base.clone(),
            )
        with self.assertRaisesRegex(ValueError, "caller-selected"):
            plan.run(
                bridge=fixture.bridge,
                combine_weights=fixture.combine_weights,
                token_out=base.clone().to(torch.float16),
            )
        with self.assertRaisesRegex(ValueError, "bridge_down"):
            plan.run(
                bridge=fixture.bridge[:, : fixture.rank // 2],
                combine_weights=fixture.combine_weights,
                token_out=base.clone(),
            )

    def test_shared_builder_refuses_per_expert_route(self):
        from benchmark.kernels.lora_moe.finalize_cutedsl import (
            build_cutedsl_shared_finalize_plan,
        )

        fixture = _Fixture(self.device)
        with self.assertRaisesRegex(ValueError, "BY ADAPTER"):
            build_cutedsl_shared_finalize_plan(
                shared_route=fixture.route("token"),  # per-expert factors
                down_weight=fixture.weight_shared,
                config=self._config(),
            )

    def test_binding_refusals(self):
        fixture = _Fixture(self.device)
        plan = self._plan(fixture, "shared")
        with self.assertRaisesRegex(ValueError, "cannot be skipped"):
            plan.require_binding(fixture.weight_shared, None)
        alias = fixture.weight_shared.view(-1, fixture.rank)
        with self.assertRaisesRegex(ValueError, "bind per fixture"):
            plan.require_binding(alias, fixture.route("shared"))
        other = _Fixture(self.device, seed=137)
        with self.assertRaisesRegex(ValueError, "different route"):
            plan.require_binding(fixture.weight_shared, other.route("shared"))

    def test_rebuild_after_route_content_change_is_refused(self):
        # Whether the mutation widens a group past the sized m_max or only
        # flips sentinel validity within capacity, the rebuild must refuse
        # rather than spill rows or index stale maps.
        fixture = _Fixture(self.device)
        for arm in ARMS:
            skew = self._plan(fixture, arm)
            skew.virtual_topk_ids.fill_(0)  # every pair/token in one group
            with self.subTest(arm=arm, mutation="skew"):
                with self.assertRaisesRegex(ValueError, "route contents changed"):
                    skew.build_metadata()
            flip = self._plan(fixture, arm)
            if arm == "token":
                # Pair-domain plan: ONE flipped pair changes the valid set.
                flat = flip.virtual_topk_ids.view(-1)
                first_valid = int((flat >= 0).nonzero(as_tuple=False).view(-1)[0])
                flat[first_valid] = -1
            else:
                # Token-domain plan: the dispatch never saw pairs, so the
                # refusable mutation is a token LOSING its group — kill
                # every pair of token 0 (valid by construction). A single
                # pair flip that keeps the token's group is tolerated by
                # design (the rank reduce reads the live ids).
                flip.virtual_topk_ids[0] = -1
            with self.subTest(arm=arm, mutation="sentinel_flip"):
                with self.assertRaisesRegex(ValueError, "route contents changed"):
                    flip.build_metadata()


if __name__ == "__main__":
    unittest.main()
