"""Independent-reference tests for the Step-4 LoRA-B candidates.

What these cases guard (plan §64.1 correctness obligations):

* semantic agreement with the STOCK kernel on valid pairs for every
  family, at both sites, including the token-dedup bridge
  (``intermediate_top_k = top_k``);
* the ZERO-FILL contract under a POISONED destination: every targeted
  cell not owned by a valid pair must be exact zero after the call —
  base tokens, sentinel experts, and non-owned experts. B's output is
  consumed additively by the activation join and the combine, so a
  preserved-garbage cell is silent output corruption under buffer reuse;
* one-slice non-gated execution (``destination_offsets=(0,)``);
* partial-target adapters: an adapter whose B factor is ZERO contributes
  exactly zero without disturbing other adapters' rows;
* the deterministic families' replay stability: indexed (serial K) and
  rank-split (FP32 workspace + fixed-order reduce) must be bitwise
  stable across replays and identical under CUDA graph capture.
"""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")

from benchmark.kernels.lora_moe.lora_b_candidates import (
    INDEXED_B_DEFAULT_CONFIG,
    ONE_LAUNCH_DEFAULT_CONFIG,
    RANK_SPLIT_DEFAULT_CONFIG,
    run_lora_b,
)
from benchmark.kernels.lora_moe.lora_b_execution import LoraBExecutionSpec
from benchmark.kernels.lora_moe.signal_gates import require_delta_close
from sglang.srt.lora.sgl_lora.bf16 import stock_grouped_lora_b
from sglang.srt.lora.sgl_lora.moe_lora_runner import PROVISIONAL_LAUNCH_CONFIG
from sglang.srt.lora.sgl_lora.routing import (
    ROUTE_ALIGNED,
    ROUTE_RAW,
    build_virtual_expert_routing,
)

SPEC_ONE_LAUNCH = LoraBExecutionSpec(
    site="gate_up", ownership="grouped", slicing="one_launch_sliced"
)
SPEC_LEAN = LoraBExecutionSpec(
    site="gate_up", ownership="grouped", slicing="lean_per_slice"
)
SPEC_INDEXED = LoraBExecutionSpec(
    site="gate_up", ownership="indexed", slicing="one_launch_sliced"
)
SPEC_RANK_SPLIT = LoraBExecutionSpec(
    site="gate_up",
    ownership="grouped",
    slicing="one_launch_sliced",
    reduction="deterministic_rank_split",
)
CHALLENGERS = {
    "one_launch": (SPEC_ONE_LAUNCH, ONE_LAUNCH_DEFAULT_CONFIG),
    "lean_two_launch": (SPEC_LEAN, ONE_LAUNCH_DEFAULT_CONFIG),
    "indexed": (SPEC_INDEXED, INDEXED_B_DEFAULT_CONFIG),
    "rank_split": (SPEC_RANK_SPLIT, RANK_SPLIT_DEFAULT_CONFIG),
}


class _Fixture:
    """Small mixed-validity B workload with an independent validity mask."""

    def __init__(
        self,
        device,
        *,
        num_tokens=12,
        top_k=4,
        lora_experts_per_adapter=6,
        max_loras=3,
        rank=32,
        slice_width=48,
        num_slices=2,
        seed=29,
    ):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.num_slices = num_slices
        self.top_k = top_k
        self.rank = rank
        self.slice_width = slice_width
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
        self.weight = (
            (
                torch.randn(
                    groups,
                    num_slices * slice_width,
                    rank,
                    generator=generator,
                    dtype=torch.float32,
                )
                * rank**-0.5
            )
            .to(torch.bfloat16)
            .to(device)
        )
        self.bridge = (
            (
                torch.randn(
                    self.num_pairs,
                    num_slices * rank,
                    generator=generator,
                    dtype=torch.float32,
                )
                * 0.5
            )
            .to(torch.bfloat16)
            .to(device)
        )
        self.destination_columns = num_slices * slice_width
        self.offsets = (0, slice_width) if num_slices == 2 else (0,)
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

    def route(self, view):
        return build_virtual_expert_routing(
            self.topk_ids, self.token_slots, view=view, **self.kwargs
        )

    def fp32_oracle(self, bridge=None, intermediate_top_k=1):
        """Independent FP32 reference: the same bf16 inputs, plain torch
        einsum in fp32, no Triton kernel anywhere — closes the gate-4
        gap where the numerical oracle was itself the stock kernel."""
        bridge = self.bridge if bridge is None else bridge
        device = bridge.device
        experts = self.kwargs["lora_experts_per_adapter"]
        ids = self.topk_ids.to(torch.int64).reshape(-1)
        slots = (
            self.token_slots.to(torch.int64)[:, None]
            .expand_as(self.topk_ids)
            .reshape(-1)
        )
        # Invalid pairs (negative OR out-of-range ids — the fixture draws
        # both on purpose) must not index the weight gather; their rows are
        # zeroed after the einsum, so any in-range stand-in group works.
        groups = self.weight.shape[0]
        veid = (slots * experts + ids).clamp(min=0, max=groups - 1)
        weight = self.weight.float()[veid]  # [P, num_slices*W, R]
        rows = (
            torch.arange(self.num_pairs, device=device, dtype=torch.int64)
            // intermediate_top_k
        )
        x = bridge.float()[rows]  # [P, num_slices*R]
        out = torch.zeros(
            self.num_pairs,
            self.destination_columns,
            dtype=torch.float32,
            device=device,
        )
        rank, width = self.rank, self.slice_width
        for s in range(self.num_slices):
            out[:, self.offsets[s] : self.offsets[s] + width] = torch.einsum(
                "pr,pnr->pn",
                x[:, s * rank : (s + 1) * rank],
                weight[:, s * width : (s + 1) * width, :],
            )
        out[~self.valid] = 0
        return out

    def destination(self, device, fill):
        out = torch.full(
            (self.num_pairs, self.destination_columns),
            fill,
            dtype=torch.bfloat16,
            device=device,
        )
        return out


class TestLoraBCandidates(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        cls.device = torch.device("cuda")

    def _stock(self, fixture, aligned):
        reference = fixture.destination(self.device, 71.0)
        stock_grouped_lora_b(
            fixture.bridge,
            fixture.weight,
            reference,
            aligned,
            destination_offsets=fixture.offsets,
            config=PROVISIONAL_LAUNCH_CONFIG.lora_b,
        )
        return reference

    def _run(self, name, fixture, *, fill):
        spec, config = CHALLENGERS[name]
        route = fixture.route(ROUTE_RAW if name == "indexed" else ROUTE_ALIGNED)
        out = fixture.destination(self.device, fill)
        run_lora_b(
            spec,
            bridge=fixture.bridge,
            weight=fixture.weight,
            destination=out,
            routing=route,
            destination_offsets=fixture.offsets,
            config=config,
        )
        return out

    def test_every_family_matches_stock_and_zero_fills_under_poison(self):
        fixture = _Fixture(self.device)
        aligned = fixture.route(ROUTE_ALIGNED)
        reference = self._stock(fixture, aligned)
        valid = fixture.valid
        self.assertTrue(bool(valid.any()) and not bool(valid.all()))
        # The stock kernel itself must satisfy the zero-fill contract —
        # it is the contract's source, so pin it before comparing to it.
        self.assertTrue(bool((reference[~valid] == 0).all()))
        for name in CHALLENGERS:
            with self.subTest(family=name):
                out = self._run(name, fixture, fill=71.0)
                require_delta_close(
                    out[valid].float(),
                    reference[valid].float(),
                    gate_dtype=torch.bfloat16,
                    label=f"{name} vs stock on valid pairs",
                )
                self.assertTrue(
                    bool((out[~valid] == 0).all()),
                    f"{name}: a poisoned invalid-pair cell survived — the "
                    "zero-fill contract is what makes graph buffer reuse "
                    "safe",
                )

    def test_one_slice_non_gated(self):
        fixture = _Fixture(self.device, num_slices=1, slice_width=64)
        aligned = fixture.route(ROUTE_ALIGNED)
        reference = self._stock(fixture, aligned)
        for name in CHALLENGERS:
            with self.subTest(family=name):
                out = self._run(name, fixture, fill=-3.0)
                require_delta_close(
                    out[fixture.valid].float(),
                    reference[fixture.valid].float(),
                    gate_dtype=torch.bfloat16,
                    label=f"{name} one-slice vs stock",
                )
                self.assertTrue(bool((out[~fixture.valid] == 0).all()))

    def test_zero_factor_adapter_contributes_exactly_zero(self):
        # Partial-target semantics: an adapter with no B factor at this
        # site carries a ZERO weight; its rows must be exactly zero while
        # other adapters' rows are untouched by that choice.
        fixture = _Fixture(self.device)
        zeroed_adapter = 1
        groups_per_adapter = fixture.kwargs["lora_experts_per_adapter"]
        start = zeroed_adapter * groups_per_adapter
        fixture.weight[start : start + groups_per_adapter] = 0
        aligned = fixture.route(ROUTE_ALIGNED)
        reference = self._stock(fixture, aligned)
        adapter_rows = (
            (fixture.token_slots == zeroed_adapter)[:, None]
            .expand(-1, fixture.top_k)
            .reshape(-1)
        )
        zero_rows = adapter_rows & fixture.valid
        self.assertTrue(bool(zero_rows.any()))
        for name in CHALLENGERS:
            with self.subTest(family=name):
                out = self._run(name, fixture, fill=17.0)
                self.assertTrue(
                    bool((out[zero_rows] == 0).all()),
                    f"{name}: zero-factor adapter rows must be exact zero",
                )
                require_delta_close(
                    out[fixture.valid].float(),
                    reference[fixture.valid].float(),
                    gate_dtype=torch.bfloat16,
                    label=f"{name} with a zero-factor adapter",
                )

    def test_token_dedup_bridge_consumption(self):
        # intermediate_top_k = top_k: the bridge is token-major and every
        # pair of a token reads the same row. Compare against the stock
        # kernel consuming the SAME token-major bridge.
        fixture = _Fixture(self.device)
        aligned = fixture.route(ROUTE_ALIGNED)
        token_bridge = (
            torch.randn(
                fixture.topk_ids.shape[0],
                fixture.bridge.shape[1],
                dtype=torch.float32,
                device=self.device,
            )
            * 0.5
        ).to(torch.bfloat16)
        reference = fixture.destination(self.device, 5.0)
        stock_grouped_lora_b(
            token_bridge,
            fixture.weight,
            reference,
            aligned,
            destination_offsets=fixture.offsets,
            config=PROVISIONAL_LAUNCH_CONFIG.lora_b,
            intermediate_top_k=fixture.top_k,
        )
        for name in CHALLENGERS:
            spec, config = CHALLENGERS[name]
            route = fixture.route(ROUTE_RAW if name == "indexed" else ROUTE_ALIGNED)
            out = fixture.destination(self.device, -9.0)
            run_lora_b(
                spec,
                bridge=token_bridge,
                weight=fixture.weight,
                destination=out,
                routing=route,
                destination_offsets=fixture.offsets,
                config=config,
                intermediate_top_k=fixture.top_k,
            )
            with self.subTest(family=name):
                require_delta_close(
                    out[fixture.valid].float(),
                    reference[fixture.valid].float(),
                    gate_dtype=torch.bfloat16,
                    label=f"{name} token-dedup bridge",
                )
                self.assertTrue(bool((out[~fixture.valid] == 0).all()))

    def test_deterministic_families_are_bitwise_across_replays_and_graph(self):
        fixture = _Fixture(self.device)
        for name in ("indexed", "rank_split"):
            with self.subTest(family=name):
                first = self._run(name, fixture, fill=0.0).clone()
                for _ in range(32):
                    again = self._run(name, fixture, fill=0.0)
                    self.assertTrue(torch.equal(first, again))
                spec, config = CHALLENGERS[name]
                route = fixture.route(ROUTE_RAW if name == "indexed" else ROUTE_ALIGNED)
                out = fixture.destination(self.device, 0.0)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run_lora_b(
                        spec,
                        bridge=fixture.bridge,
                        weight=fixture.weight,
                        destination=out,
                        routing=route,
                        destination_offsets=fixture.offsets,
                        config=config,
                    )
                # 3rd review: every replay must be OBSERVED — replaying
                # 100x and checking once inspects only the final
                # overwrite. Sync + compare after each replay.
                for _ in range(100):
                    graph.replay()
                    torch.cuda.synchronize()
                    self.assertTrue(torch.equal(first, out))

    def test_spec_vocabulary_is_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "only honest description"):
            LoraBExecutionSpec(site="gate_up", ownership="indexed", slicing="per_slice")
        with self.assertRaisesRegex(ValueError, "does not exist"):
            LoraBExecutionSpec(
                site="down",
                ownership="indexed",
                slicing="one_launch_sliced",
                reduction="deterministic_rank_split",
            )
        for slicing in ("per_slice", "lean_per_slice"):
            with self.assertRaisesRegex(ValueError, "one-launch sliced body"):
                LoraBExecutionSpec(
                    site="down",
                    ownership="grouped",
                    slicing=slicing,
                    reduction="deterministic_rank_split",
                )

    def test_every_family_and_stock_match_fp32_oracle(self):
        # Independent-reference oracle (gate-4 review): plain torch fp32
        # einsum over the SAME bf16 inputs. This is the only case whose
        # reference is not itself a Triton kernel — it turns red if the
        # stock kernel and a challenger drift together. Second review:
        # cover one-slice and the token-dedup bridge too, not just the
        # default two-slice geometry.
        for label, kwargs, itk in (
            ("two_slice", {}, 1),
            ("one_slice", {"num_slices": 1, "slice_width": 64}, 1),
            ("token_dedup", {}, None),  # itk resolved to top_k below
        ):
            fixture = _Fixture(self.device, **kwargs)
            itk = fixture.top_k if itk is None else itk
            bridge = fixture.bridge
            if itk != 1:
                bridge = (
                    torch.randn(
                        fixture.topk_ids.shape[0],
                        fixture.bridge.shape[1],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    * 0.5
                ).to(torch.bfloat16)
            oracle = fixture.fp32_oracle(bridge=bridge, intermediate_top_k=itk)
            aligned = fixture.route(ROUTE_ALIGNED)
            stock = fixture.destination(self.device, 71.0)
            stock_grouped_lora_b(
                bridge,
                fixture.weight,
                stock,
                aligned,
                destination_offsets=fixture.offsets,
                config=PROVISIONAL_LAUNCH_CONFIG.lora_b,
                intermediate_top_k=itk,
            )
            with self.subTest(geometry=label, family="stock"):
                require_delta_close(
                    stock.float(),
                    oracle,
                    gate_dtype=torch.bfloat16,
                    label=f"stock vs fp32 oracle [{label}]",
                )
            for name in CHALLENGERS:
                spec, config = CHALLENGERS[name]
                route = fixture.route(ROUTE_RAW if name == "indexed" else ROUTE_ALIGNED)
                out = fixture.destination(self.device, 13.0)
                run_lora_b(
                    spec,
                    bridge=bridge,
                    weight=fixture.weight,
                    destination=out,
                    routing=route,
                    destination_offsets=fixture.offsets,
                    config=config,
                    intermediate_top_k=itk,
                )
                with self.subTest(geometry=label, family=name):
                    require_delta_close(
                        out.float(),
                        oracle,
                        gate_dtype=torch.bfloat16,
                        label=f"{name} vs fp32 oracle [{label}]",
                    )

    def test_destination_offsets_fail_closed(self):
        # Negative offsets under-run the buffer via int64 pointer math;
        # overlapping slices double-write cells and break the zero-fill
        # and determinism contracts. Both must be rejected up front.
        fixture = _Fixture(self.device)
        route = fixture.route(ROUTE_ALIGNED)
        out = fixture.destination(self.device, 0.0)
        stock_spec = LoraBExecutionSpec(site="gate_up", ownership="grouped")
        arms = dict(CHALLENGERS)
        arms["stock"] = (stock_spec, PROVISIONAL_LAUNCH_CONFIG.lora_b)
        for name in ("one_launch", "stock"):  # stock: dispatch-level check
            spec, config = arms[name]
            for offsets, pattern in (
                ((-16, fixture.slice_width), "must be >= 0"),
                ((0, fixture.slice_width // 2), "overlap"),
            ):
                with self.subTest(family=name, offsets=offsets):
                    with self.assertRaisesRegex(ValueError, pattern):
                        run_lora_b(
                            spec,
                            bridge=fixture.bridge,
                            weight=fixture.weight,
                            destination=out,
                            routing=route,
                            destination_offsets=offsets,
                            config=config,
                        )

    def test_rank_split_workspace_fail_closed(self):
        # A non-FP32 caller workspace silently degrades the deterministic
        # fixed-order reduction to bf16 partials; must be rejected.
        fixture = _Fixture(self.device)
        route = fixture.route(ROUTE_ALIGNED)
        out = fixture.destination(self.device, 0.0)
        spec, config = CHALLENGERS["rank_split"]
        split_k = int(config["SPLIT_K"])
        bad = torch.zeros(
            (split_k, out.shape[0], out.shape[1]),
            dtype=torch.bfloat16,
            device=self.device,
        )
        with self.assertRaisesRegex(ValueError, "float32"):
            run_lora_b(
                spec,
                bridge=fixture.bridge,
                weight=fixture.weight,
                destination=out,
                routing=route,
                destination_offsets=fixture.offsets,
                config=config,
                workspace=bad,
            )
        wrong_device = torch.zeros(
            (split_k, out.shape[0], out.shape[1]), dtype=torch.float32
        )  # CPU tensor vs CUDA destination
        with self.assertRaisesRegex(ValueError, "device"):
            run_lora_b(
                spec,
                bridge=fixture.bridge,
                weight=fixture.weight,
                destination=out,
                routing=route,
                destination_offsets=fixture.offsets,
                config=config,
                workspace=wrong_device,
            )


if __name__ == "__main__":
    unittest.main()
