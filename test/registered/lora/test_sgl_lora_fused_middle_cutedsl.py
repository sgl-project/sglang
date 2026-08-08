"""Independent-reference tests for the Step-5 CuTeDSL fused-middle arm.

What these cases guard (plan §65.1 correctness obligations):

* semantic agreement with a TEST-LOCAL fp32 torch oracle (gather-einsum,
  written independently of the module's group-loop reference) for both
  guardrail activations and the token-dedup bridge, under POISONED
  output buffers;
* the §65.1 universal-act contract: sentinel pairs produce
  ``act = activation(base)`` and EXACT-ZERO ``down_rank_out`` — the
  fused middle's outputs are consumed additively downstream, so a
  preserved-garbage cell is silent corruption under buffer reuse;
* metadata-rebuild output invariance: the dispatch atomics assign a
  fresh within-group row order per rebuild, and a stale inverse map or
  stale stage-row map silently permutes (the A-plan's seventh-review
  bug class, now with TWO derived index tensors to go stale);
* binding fail-closure: cross-site weight, foreign same-shape route,
  and missing routing are refused (the A-plan's eighth/ninth-review
  fail-open classes);
* fixture-only rebuild: route-content changes (wider group, sentinel
  flip within capacity) are refused instead of spilling or
  mis-scattering;
* graph capture/replay bitwise stability (Step-5 evidence requirement);
* fail-closed build geometry, including the slice-interleaved GEMM's
  DOUBLED group domain against the direct schedule's packing cap;
* the module's admission reference helper agrees with the test-local
  oracle (turns red if the helper drifts while kernels follow it).
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
    require_delta_close,
    require_finite,
)
from sglang.srt.lora.sgl_lora.routing import (
    ROUTE_FUSED_IDS,
    build_virtual_expert_routing,
)


class _MiddleFixture:
    """Small mixed-validity fused-middle workload with an independent
    validity mask and a test-local fp32 oracle."""

    def __init__(
        self,
        device,
        *,
        num_tokens=40,
        top_k=4,
        lora_experts_per_adapter=6,
        max_loras=3,
        rank=32,
        inter=64,
        activation="silu_mul",
        seed=29,
    ):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.activation = activation
        self.num_slices = 2 if activation == "silu_mul" else 1
        self.top_k = top_k
        self.rank = rank
        self.inter = inter
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
        self.topk_ids = topk_ids.to(device)
        self.token_slots = token_slots.to(device)
        self.kwargs = dict(
            lora_experts_per_adapter=lora_experts_per_adapter,
            max_loras=max_loras,
            block_size=16,
        )
        groups = max_loras * lora_experts_per_adapter
        slices = self.num_slices

        def draw(*shape, scale):
            values = torch.randn(*shape, generator=generator, dtype=torch.float32)
            return (values * scale).to(torch.bfloat16).to(device)

        self.b_gate_up = draw(groups, slices * inter, rank, scale=rank**-0.5)
        self.a_down = draw(groups, rank, inter, scale=inter**-0.5)
        self.bridge = draw(self.num_pairs, slices * rank, scale=0.5)
        self.base_gu = draw(self.num_pairs, slices * inter, scale=1.0)
        # Host validity: the same key definition, computed independently.
        ids = topk_ids.to(torch.int64)
        adapters = token_slots.to(torch.int64)[:, None].expand_as(ids)
        valid = (
            (adapters >= 0)
            & (adapters < max_loras)
            & (ids >= 0)
            & (ids < lora_experts_per_adapter)
        )
        self.valid = valid.reshape(-1).to(device)
        self.veid = torch.where(
            valid, adapters * lora_experts_per_adapter + ids, torch.tensor(-1)
        ).reshape(-1)

    def route(self, view=ROUTE_FUSED_IDS):
        return build_virtual_expert_routing(
            self.topk_ids, self.token_slots, view=view, **self.kwargs
        )

    def buffers(self, fill):
        act = torch.full(
            (self.num_pairs, self.inter),
            fill,
            dtype=torch.bfloat16,
            device=self.topk_ids.device,
        )
        down = torch.full(
            (self.num_pairs, self.rank),
            fill,
            dtype=torch.bfloat16,
            device=self.topk_ids.device,
        )
        return act, down

    def oracle(self, bridge=None, intermediate_top_k=1):
        """Test-local fp32 reference: gather-einsum, no group loop, no
        Triton — independent of both the kernels and the module's own
        group-loop admission helper."""
        bridge = self.bridge if bridge is None else bridge
        device = bridge.device
        groups = self.b_gate_up.shape[0]
        safe = self.veid.clamp(min=0, max=groups - 1).to(device)
        rows = (
            torch.arange(self.num_pairs, device=device, dtype=torch.int64)
            // intermediate_top_k
        )
        x = bridge.float()[rows]  # [P, S*R]
        weight = self.b_gate_up.float()[safe]  # [P, S*W, R]
        delta = torch.zeros(
            self.num_pairs, self.num_slices * self.inter, device=device
        )
        for s in range(self.num_slices):
            delta[:, s * self.inter : (s + 1) * self.inter] = torch.einsum(
                "pr,pnr->pn",
                x[:, s * self.rank : (s + 1) * self.rank],
                weight[:, s * self.inter : (s + 1) * self.inter, :],
            )
        delta[~self.valid] = 0
        pre = self.base_gu.float() + delta
        if self.activation == "silu_mul":
            act = torch.nn.functional.silu(pre[:, : self.inter]) * pre[:, self.inter :]
        else:
            act = torch.relu(pre) ** 2
        down = torch.einsum("pw,prw->pr", act, self.a_down.float()[safe])
        down[~self.valid] = 0
        return act, down


class TestCutedslFusedMiddle(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        if torch.cuda.get_device_capability() < (9, 0):
            raise unittest.SkipTest("the masked grouped GEMM needs SM90+")
        try:
            import cutlass  # noqa: F401

            from benchmark.kernels.lora_moe import fused_middle_cutedsl
        except ImportError as error:
            raise unittest.SkipTest(f"CuTeDSL unavailable: {error}")
        cls.fm = fused_middle_cutedsl
        cls.device = torch.device("cuda")

    def _plan(self, fixture, *, intermediate_top_k=1):
        return self.fm.build_cutedsl_fused_middle_plan(
            fused_route=fixture.route(),
            b_gate_up=fixture.b_gate_up,
            a_down=fixture.a_down,
            config=self.fm.CutedslAConfig(
                token_width=self.fm.supported_token_widths(self.device)[0]
            ),
            activation=fixture.activation,
            intermediate_top_k=intermediate_top_k,
        )

    def _invoke(self, fixture, plan, *, bridge=None, fill=float("nan")):
        act, down = fixture.buffers(fill)
        self.fm.invoke_cutedsl_fused_middle(
            bridge_gu=fixture.bridge if bridge is None else bridge,
            b_gate_up=fixture.b_gate_up,
            base_gu=fixture.base_gu,
            a_down=fixture.a_down,
            routing=fixture.route(),
            act=act,
            down_rank_out=down,
            plan=plan,
        )
        return act, down

    def test_matches_oracle_and_zero_fills_under_poison(self):
        for activation in ("silu_mul", "relu2"):
            with self.subTest(activation=activation):
                fixture = _MiddleFixture(self.device, activation=activation)
                self.assertTrue(
                    bool(fixture.valid.any()) and not bool(fixture.valid.all())
                )
                plan = self._plan(fixture)
                act, down = self._invoke(fixture, plan, fill=float("nan"))
                oracle_act, oracle_down = fixture.oracle()
                # act is UNIVERSAL: every pair (sentinels included, whose
                # oracle value is activation(base)) must be written.
                require_finite(act, label=f"{activation} act poison hygiene")
                require_finite(down, label=f"{activation} down poison hygiene")
                require_delta_close(
                    act.float(),
                    oracle_act,
                    gate_dtype=torch.bfloat16,
                    label=f"{activation} act vs fp32 oracle (all pairs)",
                )
                require_delta_close(
                    down[fixture.valid].float(),
                    oracle_down[fixture.valid],
                    gate_dtype=torch.bfloat16,
                    label=f"{activation} down_rank_out vs fp32 oracle",
                )
                self.assertTrue(
                    bool((down[~fixture.valid] == 0).all()),
                    "a poisoned sentinel down_rank_out cell survived — the "
                    "zero-fill contract is what makes buffer reuse safe",
                )

    def test_token_dedup_bridge_consumption(self):
        # intermediate_top_k = top_k: the bridge is token-major and every
        # pair of a token reads the same row.
        fixture = _MiddleFixture(self.device)
        generator = torch.Generator(device="cpu").manual_seed(97)
        token_bridge = (
            (
                torch.randn(
                    fixture.topk_ids.shape[0],
                    fixture.bridge.shape[1],
                    generator=generator,
                    dtype=torch.float32,
                )
                * 0.5
            )
            .to(torch.bfloat16)
            .to(self.device)
        )
        plan = self._plan(fixture, intermediate_top_k=fixture.top_k)
        act, down = self._invoke(fixture, plan, bridge=token_bridge, fill=-9.0)
        oracle_act, oracle_down = fixture.oracle(
            bridge=token_bridge, intermediate_top_k=fixture.top_k
        )
        require_delta_close(
            act.float(),
            oracle_act,
            gate_dtype=torch.bfloat16,
            label="token-dedup act vs fp32 oracle",
        )
        require_delta_close(
            down[fixture.valid].float(),
            oracle_down[fixture.valid],
            gate_dtype=torch.bfloat16,
            label="token-dedup down_rank_out vs fp32 oracle",
        )
        self.assertTrue(bool((down[~fixture.valid] == 0).all()))

    def test_metadata_rebuild_is_output_invariant(self):
        # The rebuild refreshes THREE placement-coupled tensors (src2dst,
        # src2dst_valid, stage_rows_first); a stale one silently permutes
        # pair outputs — must be bitwise invariant, not merely close.
        fixture = _MiddleFixture(self.device)
        plan = self._plan(fixture)
        first_act, first_down = self._invoke(fixture, plan, fill=0.0)
        plan.build_metadata()
        again_act, again_down = self._invoke(fixture, plan, fill=1.0)
        self.assertTrue(
            torch.equal(first_act, again_act),
            "rebuild permuted the pair-domain activation",
        )
        self.assertTrue(
            torch.equal(first_down, again_down),
            "rebuild permuted down_rank_out",
        )

    def test_binding_contracts_fail_closed(self):
        fixture = _MiddleFixture(self.device)
        plan = self._plan(fixture)
        with self.assertRaisesRegex(ValueError, "per fixture and per site"):
            plan.require_binding(
                gate_up_weight=fixture.a_down,  # the DOWN weight on gate/up
                down_weight=fixture.a_down,
                routing=fixture.route(),
            )
        with self.assertRaisesRegex(ValueError, "cannot be skipped"):
            plan.require_binding(
                gate_up_weight=fixture.b_gate_up,
                down_weight=fixture.a_down,
                routing=None,
            )
        foreign = _MiddleFixture(self.device, seed=137)
        with self.assertRaisesRegex(ValueError, "different route"):
            plan.require_binding(
                gate_up_weight=fixture.b_gate_up,
                down_weight=fixture.a_down,
                routing=foreign.route(),  # same [T, K] shape, other tensors
            )

    def test_route_content_changes_are_refused(self):
        fixture = _MiddleFixture(self.device)
        plan = self._plan(fixture)
        first_valid = int(plan.valid_pair_ids[0])
        plan.virtual_topk_ids.view(-1)[first_valid] = -1
        with self.assertRaisesRegex(ValueError, "route contents changed"):
            plan.build_metadata()
        skew = self._plan(_MiddleFixture(self.device, seed=53))
        # Every pair into one group exceeds the m_max sized for the
        # balanced route (P=160 > any supported token width's rounding).
        skew.virtual_topk_ids.fill_(0)
        with self.assertRaisesRegex(ValueError, "exceeds the sized m_max"):
            skew.build_metadata()

    def test_graph_replay_is_bitwise(self):
        fixture = _MiddleFixture(self.device)
        plan = self._plan(fixture)
        expected_act, expected_down = self._invoke(fixture, plan, fill=0.0)
        expected_act = expected_act.clone()
        expected_down = expected_down.clone()
        act, down = fixture.buffers(0.0)
        routing = fixture.route()  # built OUTSIDE capture: route build is
        # per-forward metadata, not part of the prepared-boundary thunk
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.fm.invoke_cutedsl_fused_middle(
                bridge_gu=fixture.bridge,
                b_gate_up=fixture.b_gate_up,
                base_gu=fixture.base_gu,
                a_down=fixture.a_down,
                routing=routing,
                act=act,
                down_rank_out=down,
                plan=plan,
            )
        # Every replay OBSERVED: replaying N times and checking once
        # inspects only the final overwrite.
        for _ in range(30):
            graph.replay()
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(expected_act, act))
            self.assertTrue(torch.equal(expected_down, down))

    def test_build_geometry_fails_closed(self):
        fixture = _MiddleFixture(self.device)
        route = fixture.route()
        config = self.fm.CutedslAConfig(
            token_width=self.fm.supported_token_widths(self.device)[0]
        )
        with self.assertRaisesRegex(ValueError, "activation must be one of"):
            self.fm.build_cutedsl_fused_middle_plan(
                fused_route=route,
                b_gate_up=fixture.b_gate_up,
                a_down=fixture.a_down,
                config=config,
                activation="gelu",
            )
        with self.assertRaisesRegex(ValueError, "needs b_gate_up rows"):
            self.fm.build_cutedsl_fused_middle_plan(
                fused_route=route,
                # One-slice-shaped weight declared as gated.
                b_gate_up=fixture.b_gate_up[:, : fixture.inter, :].contiguous(),
                a_down=fixture.a_down,
                config=config,
                activation="silu_mul",
            )
        with self.assertRaisesRegex(ValueError, "intermediate_top_k"):
            self.fm.build_cutedsl_fused_middle_plan(
                fused_route=route,
                b_gate_up=fixture.b_gate_up,
                a_down=fixture.a_down,
                config=config,
                intermediate_top_k=3,
            )
        # The slice-interleaved GEMM1 runs over 2V groups: a V within the
        # schedule packing cap can still overflow it once doubled. This
        # must be refused BEFORE any compile (V=540 -> 1080 > 1024).
        wide = _MiddleFixture(
            self.device,
            num_tokens=2,
            top_k=2,
            lora_experts_per_adapter=6,
            max_loras=90,
            rank=8,
            inter=8,
        )
        with self.assertRaisesRegex(ValueError, "group packing"):
            self.fm.build_cutedsl_fused_middle_plan(
                fused_route=wide.route(),
                b_gate_up=wide.b_gate_up,
                a_down=wide.a_down,
                config=config,
                activation="silu_mul",
            )

    def test_module_reference_matches_local_oracle(self):
        # The module's group-loop admission helper vs this file's
        # gather-einsum oracle: both pure fp32 torch, independently
        # written — red if the shipped reference drifts.
        for activation in ("silu_mul", "relu2"):
            fixture = _MiddleFixture(self.device, activation=activation)
            module_act, module_down = self.fm.reference_fused_middle(
                bridge_gu=fixture.bridge,
                b_gate_up=fixture.b_gate_up,
                base_gu=fixture.base_gu,
                a_down=fixture.a_down,
                virtual_topk_ids=fixture.route().virtual_topk_ids,
                activation=activation,
            )
            oracle_act, oracle_down = fixture.oracle()
            with self.subTest(activation=activation):
                require_delta_close(
                    module_act,
                    oracle_act,
                    gate_dtype=torch.float32,
                    label=f"{activation} module reference act",
                )
                require_delta_close(
                    module_down[fixture.valid],
                    oracle_down[fixture.valid],
                    gate_dtype=torch.float32,
                    label=f"{activation} module reference down",
                )
                self.assertTrue(bool((module_down[~fixture.valid] == 0).all()))


if __name__ == "__main__":
    unittest.main()
