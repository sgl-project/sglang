"""Independent-reference tests for the Step-5 fused-middle candidates.

What these cases guard (plan §65.1 correctness obligations):

* semantic agreement of every ladder arm (b_act / act_down_a / full)
  with a plain-torch FP32 oracle for BOTH outputs — the universal
  ``act`` [P, W] and the ``down_rank_out`` [P, R2] rank bridge — on a
  mixed-validity fixture, for the gated SwiGLU form and the non-gated
  ReLU^2 guardrail;
* the ACT-UNIVERSAL contract under NaN-poisoned destinations: every act
  cell is written on every call, and sentinel rows (base tokens, invalid
  pairs) get ``activation(base)`` with delta = 0 — act is the base W2
  input, so a skipped row is silent base-model corruption;
* the sentinel INPUT discipline: bridge rows and materialized-delta rows
  of invalid pairs are NaN-poisoned in the fixture, so any arm that
  reads them (instead of skipping the dots for ``virtual_expert_id ==
  -1`` blocks) turns the whole run non-finite;
* the down_rank_out EXACT-ZERO contract for sentinel rows under a
  poisoned destination — the finalizer consumes this buffer additively;
* bitwise determinism of the accumulating arms (serial W-tile loop, no
  atomics) across eager replays and under CUDA graph capture/replay;
* the fail-closed spec vocabulary, the dispatcher's required/forbidden
  tensor surface, and the geometry guards (register-resident down-rank
  bound, down-dot dtype unity, tl.dot tile minimum).
"""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")

from benchmark.kernels.lora_moe.fused_middle_candidates import (
    FUSED_B_ACT_DEFAULT_CONFIG,
    FUSED_MIDDLE_SERIAL_DEFAULT_CONFIG,
    FUSIONS,
    MAX_DOWN_RANK,
    FusedMiddleSpec,
    run_fused_middle,
)
from benchmark.kernels.lora_moe.signal_gates import require_delta_close, require_finite
from sglang.srt.lora.sgl_lora.routing import ROUTE_ALIGNED, build_virtual_expert_routing

CONFIGS = {
    "b_act": FUSED_B_ACT_DEFAULT_CONFIG,
    "act_down_a": FUSED_MIDDLE_SERIAL_DEFAULT_CONFIG,
    "full": FUSED_MIDDLE_SERIAL_DEFAULT_CONFIG,
}


class _Fixture:
    """Mixed-validity middle workload with an independent validity mask.

    Geometry is chosen to exercise every mask path: rank 48 needs two
    BLOCK_SIZE_K=32 iterations with a partial tail, width 80 needs a
    partial W tile (and two serial-loop iterations at BLOCK_SIZE_W=64),
    down rank 24 is padded to 32 lanes with a masked tail. Bridge and
    materialized-delta rows of invalid pairs are NaN-poisoned so any
    kernel read of a sentinel row is detectable in the outputs.
    """

    def __init__(
        self,
        device,
        *,
        activation="silu_mul",
        num_tokens=12,
        top_k=4,
        lora_experts_per_adapter=6,
        max_loras=3,
        rank=48,
        width=80,
        down_rank=24,
        seed=31,
    ):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.device = device
        self.activation = activation
        self.num_slices = 2 if activation == "silu_mul" else 1
        self.rank = rank
        self.width = width
        self.down_rank = down_rank
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

        def _draw(*shape, scale):
            values = torch.randn(*shape, generator=generator, dtype=torch.float32)
            return (values * scale).to(torch.bfloat16).to(device)

        self.b_gate_up = _draw(groups, slices * width, rank, scale=rank**-0.5)
        self.a_down = _draw(groups, down_rank, width, scale=width**-0.5)
        self.base_gu = _draw(self.num_pairs, slices * width, scale=0.5)
        self.bridge = _draw(self.num_pairs, slices * rank, scale=0.5)
        # Host validity: the same key definition, computed independently.
        ids = topk_ids.to(torch.int64)
        adapters = token_slots.to(torch.int64)[:, None].expand_as(ids)
        self.valid = (
            (
                (adapters >= 0)
                & (adapters < max_loras)
                & (ids >= 0)
                & (ids < lora_experts_per_adapter)
            )
            .reshape(-1)
            .to(device)
        )
        # Invalid pairs must not index the weight gathers; any in-range
        # stand-in group works because their rows are zeroed post-einsum.
        flat_ids = self.topk_ids.to(torch.int64).reshape(-1)
        flat_slots = (
            self.token_slots.to(torch.int64)[:, None]
            .expand_as(self.topk_ids)
            .reshape(-1)
        )
        self.gather_veid = (flat_slots * lora_experts_per_adapter + flat_ids).clamp(
            min=0, max=groups - 1
        )
        # Sentinel-input discipline: an arm that reads a sentinel pair's
        # bridge or delta row (instead of skipping its dots) goes NaN.
        self.bridge[~self.valid] = float("nan")
        delta_clean = self.fp32_delta().to(torch.bfloat16)
        self.delta_clean = delta_clean
        self.gate_up_delta = delta_clean.clone()
        self.gate_up_delta[~self.valid] = float("nan")

    def route(self):
        return build_virtual_expert_routing(
            self.topk_ids, self.token_slots, view=ROUTE_ALIGNED, **self.kwargs
        )

    def fp32_delta(self):
        """Independent FP32 gate/up-B reference; invalid rows exact zero."""
        weight = self.b_gate_up.float()[self.gather_veid]  # [P, S*W, R]
        x = self.bridge.float()
        out = torch.zeros(
            self.num_pairs,
            self.num_slices * self.width,
            dtype=torch.float32,
            device=self.device,
        )
        rank, width = self.rank, self.width
        for s in range(self.num_slices):
            out[:, s * width : (s + 1) * width] = torch.einsum(
                "pr,pnr->pn",
                x[:, s * rank : (s + 1) * rank],
                weight[:, s * width : (s + 1) * width, :],
            )
        out[~self.valid] = 0
        return out

    def fp32_act(self, delta_fp32):
        """activation(base + delta) in FP32 — the universal act contract.

        Sentinel rows come out as activation(base) because the delta
        oracle zeroes them, matching the delta = 0 contract exactly.
        """
        joined = self.base_gu.float() + delta_fp32
        if self.num_slices == 2:
            gate = joined[:, : self.width]
            up = joined[:, self.width :]
            return torch.nn.functional.silu(gate) * up
        clamped = torch.relu(joined)
        return clamped * clamped

    def fp32_down(self, act_bf16):
        """FP32 down-A over the STORED (bf16) act; invalid rows exact zero."""
        weight = self.a_down.float()[self.gather_veid]  # [P, R2, W]
        out = torch.einsum("pw,prw->pr", act_bf16.float(), weight)
        out[~self.valid] = 0
        return out

    def act_destination(self, fill):
        return torch.full(
            (self.num_pairs, self.width),
            fill,
            dtype=torch.bfloat16,
            device=self.device,
        )

    def down_destination(self, fill):
        return torch.full(
            (self.num_pairs, self.down_rank),
            fill,
            dtype=torch.bfloat16,
            device=self.device,
        )


class TestFusedMiddleCandidates(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        cls.device = torch.device("cuda")

    def _dispatch_kwargs(self, fixture, fusion, down):
        kwargs = {}
        if fusion in ("b_act", "full"):
            kwargs["bridge_gu"] = fixture.bridge
            kwargs["b_gate_up"] = fixture.b_gate_up
        if fusion == "act_down_a":
            kwargs["gate_up_delta"] = fixture.gate_up_delta
        if fusion != "b_act":
            kwargs["a_down"] = fixture.a_down
            kwargs["down_rank_out"] = down
        return kwargs

    def _run(self, fixture, fusion, *, fill):
        spec = FusedMiddleSpec(fusion=fusion, activation=fixture.activation)
        act = fixture.act_destination(fill)
        down = None if fusion == "b_act" else fixture.down_destination(fill)
        run_fused_middle(
            spec,
            base_gu=fixture.base_gu,
            act=act,
            routing=fixture.route(),
            config=CONFIGS[fusion],
            **self._dispatch_kwargs(fixture, fusion, down),
        )
        return act, down

    def _check_all_arms(self, fixture):
        """Every arm vs the FP32 oracle under NaN-poisoned destinations.

        Turns red if: the delta dots, activation join, or down-A layout
        drift from the reference math; a sentinel block reads its
        (NaN-poisoned) bridge/delta rows; any act cell goes unwritten
        (poison survives); or a sentinel down_rank_out row is not exact
        zero. This is the case that makes graph buffer reuse safe.
        """
        valid = fixture.valid
        self.assertTrue(bool(valid.any()) and not bool(valid.all()))
        # b_act/full compute the delta in FP32 registers; act_down_a
        # consumes the MATERIALIZED bf16 delta — one extra rounding, so
        # its honest reference is the bf16-rounded delta.
        oracle_computed = fixture.fp32_act(fixture.fp32_delta())
        oracle_material = fixture.fp32_act(fixture.delta_clean.float())
        for fusion in FUSIONS:
            with self.subTest(fusion=fusion):
                act, down = self._run(fixture, fusion, fill=float("nan"))
                oracle = oracle_material if fusion == "act_down_a" else oracle_computed
                require_finite(act, label=f"{fusion} act fully written")
                require_delta_close(
                    act.float(),
                    oracle,
                    gate_dtype=torch.bfloat16,
                    label=f"{fusion} act vs fp32 oracle",
                )
                if down is None:
                    continue
                self.assertTrue(
                    bool((down[~valid] == 0).all()),
                    f"{fusion}: a poisoned sentinel down_rank_out cell "
                    "survived — the finalizer consumes this buffer "
                    "additively, so exact zero is the contract",
                )
                require_delta_close(
                    down.float(),
                    fixture.fp32_down(oracle.to(torch.bfloat16)),
                    gate_dtype=torch.bfloat16,
                    label=f"{fusion} down_rank_out vs fp32 oracle",
                )

    def test_every_arm_matches_fp32_oracle_silu_mul(self):
        self._check_all_arms(_Fixture(self.device))

    def test_every_arm_matches_fp32_oracle_relu2(self):
        # The non-gated guardrail: one slice, W == total width, act =
        # relu(base + delta)^2. Red if the NUM_SLICES/ACT constexpr
        # specialization breaks either form.
        self._check_all_arms(
            _Fixture(self.device, activation="relu2", rank=32, width=48, down_rank=16)
        )

    def test_accumulating_arms_are_bitwise_across_replays_and_graph(self):
        # The serial W-tile loop is the determinism claim: fixed tile
        # order, FP32 register accumulation, no atomics. Red if a
        # rewrite introduces order-dependent accumulation or a
        # graph-replay-unsafe launch (host-side plan capacity drift).
        fixture = _Fixture(self.device)
        for fusion in ("full", "act_down_a"):
            with self.subTest(fusion=fusion):
                first_act, first_down = self._run(fixture, fusion, fill=7.0)
                for _ in range(32):
                    act, down = self._run(fixture, fusion, fill=7.0)
                    self.assertTrue(torch.equal(first_act, act))
                    self.assertTrue(torch.equal(first_down, down))
                spec = FusedMiddleSpec(fusion=fusion, activation=fixture.activation)
                route = fixture.route()
                act = fixture.act_destination(7.0)
                down = fixture.down_destination(7.0)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run_fused_middle(
                        spec,
                        base_gu=fixture.base_gu,
                        act=act,
                        routing=route,
                        config=CONFIGS[fusion],
                        **self._dispatch_kwargs(fixture, fusion, down),
                    )
                # Every replay must be OBSERVED — sync + compare each
                # time, not once after the last overwrite.
                for _ in range(16):
                    graph.replay()
                    torch.cuda.synchronize()
                    self.assertTrue(torch.equal(first_act, act))
                    self.assertTrue(torch.equal(first_down, down))

    def test_spec_and_dispatch_fail_closed(self):
        # The dispatcher's required/forbidden surface is the arm
        # identity: handing b_act a delta buffer (or expecting an
        # unwritten down_rank_out) is wired-wrong-arm corruption.
        with self.assertRaisesRegex(ValueError, "fusion"):
            FusedMiddleSpec(fusion="everything", activation="silu_mul")
        with self.assertRaisesRegex(ValueError, "activation"):
            FusedMiddleSpec(fusion="full", activation="gelu")
        fixture = _Fixture(self.device)
        route = fixture.route()
        act = fixture.act_destination(0.0)
        down = fixture.down_destination(0.0)
        with self.assertRaisesRegex(ValueError, "does not consume"):
            run_fused_middle(
                FusedMiddleSpec(fusion="b_act", activation="silu_mul"),
                base_gu=fixture.base_gu,
                act=act,
                routing=route,
                config=CONFIGS["b_act"],
                bridge_gu=fixture.bridge,
                b_gate_up=fixture.b_gate_up,
                gate_up_delta=fixture.gate_up_delta,
            )
        with self.assertRaisesRegex(ValueError, "requires gate_up_delta"):
            run_fused_middle(
                FusedMiddleSpec(fusion="act_down_a", activation="silu_mul"),
                base_gu=fixture.base_gu,
                act=act,
                routing=route,
                config=CONFIGS["act_down_a"],
                a_down=fixture.a_down,
                down_rank_out=down,
            )

    def test_geometry_guards_fail_closed(self):
        # Red if a rewrite drops the register-resident down-rank bound
        # (silent spills), the down-dot dtype unity (Triton compile
        # error deep in a launch instead of a contract error), or the
        # tl.dot tile minimum (cryptic codegen failure).
        fixture = _Fixture(self.device)
        route = fixture.route()
        spec = FusedMiddleSpec(fusion="full", activation="silu_mul")
        groups = fixture.b_gate_up.shape[0]
        big_rank = MAX_DOWN_RANK + 32
        with self.assertRaisesRegex(ValueError, "register-resident"):
            run_fused_middle(
                spec,
                base_gu=fixture.base_gu,
                act=fixture.act_destination(0.0),
                routing=route,
                config=CONFIGS["full"],
                bridge_gu=fixture.bridge,
                b_gate_up=fixture.b_gate_up,
                a_down=torch.zeros(
                    groups,
                    big_rank,
                    fixture.width,
                    dtype=torch.bfloat16,
                    device=self.device,
                ),
                down_rank_out=torch.zeros(
                    fixture.num_pairs,
                    big_rank,
                    dtype=torch.bfloat16,
                    device=self.device,
                ),
            )
        with self.assertRaisesRegex(ValueError, "STORED act tiles"):
            run_fused_middle(
                spec,
                base_gu=fixture.base_gu,
                act=torch.zeros(
                    fixture.num_pairs,
                    fixture.width,
                    dtype=torch.float32,
                    device=self.device,
                ),
                routing=route,
                config=CONFIGS["full"],
                bridge_gu=fixture.bridge,
                b_gate_up=fixture.b_gate_up,
                a_down=fixture.a_down,
                down_rank_out=fixture.down_destination(0.0),
            )
        narrow = dict(CONFIGS["full"])
        narrow["BLOCK_SIZE_W"] = 8
        with self.assertRaisesRegex(ValueError, "tl.dot minimum"):
            run_fused_middle(
                spec,
                base_gu=fixture.base_gu,
                act=fixture.act_destination(0.0),
                routing=route,
                config=narrow,
                bridge_gu=fixture.bridge,
                b_gate_up=fixture.b_gate_up,
                a_down=fixture.a_down,
                down_rank_out=fixture.down_destination(0.0),
            )


class TestBenchConfigIdentity(CustomTestCase):
    """15th S5/6 review regression: the tuned join was silently DISCARDED
    between sweep and decided (tuned.get defaulted), and join widths were
    invisible to the config key. CPU-only — no kernels launched."""

    def test_join_winner_is_carried_from_sweep_to_decided(self):
        import ast
        import inspect

        from benchmark.kernels.lora_moe import bench_fused_middle as bench

        self.assertIn("m_join", bench.DECIDED_TUNED_FAMILIES)
        # 16th review: exercise the REAL loader production calls, so
        # _decided_cell cannot revert to a literal family tuple while
        # this test stays green.
        cell_key = ("gated", "dense", 16, "decode")
        best = {
            (*cell_key, family): {"num_warps": 4}
            for family in bench.DECIDED_TUNED_FAMILIES
        }
        loaded = bench.load_decided_tuned(best, cell_key)
        self.assertEqual(set(loaded), set(bench.DECIDED_TUNED_FAMILIES))
        incomplete = dict(best)
        del incomplete[(*cell_key, "m_join")]
        with self.assertRaises(KeyError):
            bench.load_decided_tuned(incomplete, cell_key)

        decided_source = inspect.getsource(bench._decided_cell)
        self.assertIn("load_decided_tuned(", decided_source)
        # every tuned["..."] subscript inside run_arm must be a family the
        # decided loader actually loads — the drift that hid this bug
        source = inspect.getsource(bench._MiddleFixture.run_arm)
        read_keys = {
            node.slice.value
            for node in ast.walk(ast.parse(source.lstrip()))
            if isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "tuned"
            and isinstance(node.slice, ast.Constant)
        }
        self.assertTrue(read_keys)
        self.assertLessEqual(read_keys, set(bench.DECIDED_TUNED_FAMILIES))
        # and the M arm must read the join HARD, never .get with a default
        self.assertIn('tuned["m_join"]', source)
        self.assertNotIn('tuned.get("m_join")', source)

    def test_distinct_join_widths_have_distinct_keys(self):
        from benchmark.kernels.lora_moe import bench_fused_middle as bench

        keys = {
            bench._cfg_key({"BLOCK_W": bw, "num_warps": 4})
            for bw in (128, 256, 512, 1024)
        }
        self.assertEqual(len(keys), 4)
        tuned = {
            "m_b": {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 16, "num_warps": 4},
            "m_join": {"BLOCK_W": 256, "num_warps": 4},
            "m_down": {"BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64, "num_warps": 4},
        }
        self.assertIn("join:jw256", bench._arm_config_key("m", tuned, None))


if __name__ == "__main__":
    unittest.main()
