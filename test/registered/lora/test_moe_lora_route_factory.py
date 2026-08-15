"""Host-only guards for the Step-8 route-factory split-M candidate."""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import msgspec
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-c-test-cpu")

ROOT = Path(__file__).resolve().parents[3]
LORA_MOE = ROOT / "python/sglang/srt/lora/moe"


def _load_execution_plan():
    module_name = "_route_factory_execution_plan"
    spec = importlib.util.spec_from_file_location(
        module_name, LORA_MOE / "execution_plan.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


PLAN = _load_execution_plan()


def _load_launch_config():
    package_names = (
        "sglang",
        "sglang.srt",
        "sglang.srt.lora",
        "sglang.srt.lora.moe",
        "sglang.srt.lora.moe.base_gemm_provider",
    )
    packages = {}
    for name in package_names:
        package = types.ModuleType(name)
        package.__path__ = []
        packages[name] = package
    finalize = types.ModuleType(
        "sglang.srt.lora.moe.base_gemm_provider.masked_finalize"
    )
    finalize.SHARED_RANK_DEFAULT_CONFIG = {
        "reduce": {"BLOCK_SIZE_T": 16},
        "tail": {"BLOCK_SIZE_H": 16},
    }
    middle = types.ModuleType(
        "sglang.srt.lora.moe.base_gemm_provider.masked_fused_middle"
    )
    middle.FUSED_B_ACT_DEFAULT_CONFIG = {"BLOCK_SIZE_W": 16}
    module_name = "_host_launch_config"
    spec = importlib.util.spec_from_file_location(
        module_name, LORA_MOE / "launch_config.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(
        sys.modules,
        {
            **packages,
            "sglang.srt.lora.moe.execution_plan": PLAN,
            finalize.__name__: finalize,
            middle.__name__: middle,
            module_name: module,
        },
    ):
        spec.loader.exec_module(module)
    return module


LAUNCH = _load_launch_config()


class _HostRouteView(msgspec.Struct, frozen=True, kw_only=True):
    view: str
    num_virtual_experts: int
    block_size: int
    topk_ids: torch.Tensor
    token_slots: torch.Tensor
    lora_experts_per_adapter: int
    max_loras: int
    shared_outer_local_expert_count: int | None = None
    maybe_virtual_topk_ids: torch.Tensor | None = None
    maybe_sorted_pair_ids: torch.Tensor | None = None
    maybe_block_virtual_expert_ids: torch.Tensor | None = None
    maybe_num_pairs_post_padded: torch.Tensor | None = None


def _load_route_factory():
    package_names = (
        "sglang",
        "sglang.srt",
        "sglang.srt.lora",
        "sglang.srt.lora.moe",
    )
    packages = {}
    for name in package_names:
        package = types.ModuleType(name)
        package.__path__ = []
        packages[name] = package

    routing = types.ModuleType("sglang.srt.lora.moe.routing")
    routing.ROUTE_RAW = "raw"
    routing.ROUTE_FUSED_IDS = "fused_ids"
    routing.ROUTE_ALIGNED = "aligned"
    routing.RouteView = _HostRouteView
    routing.FusedAlignScratch = types.SimpleNamespace
    routing.build_virtual_expert_routing = lambda *args, **kwargs: None
    routing.uses_fused_align = lambda *_args, **_kwargs: True

    def unexpected_dual_granularity_route(*_args, **_kwargs):
        raise AssertionError(
            "the dual-granularity fused builder must be reached only through "
            "an explicit test mock"
        )

    routing.build_dual_granularity_aligned_routes = unexpected_dual_granularity_route

    joint_routing = types.ModuleType("sglang.srt.lora.moe.joint_routing")

    def unexpected_joint_route(*_args, **_kwargs):
        raise AssertionError("the standard route plan must not use R10")

    joint_routing.build_joint_shared_routes = unexpected_joint_route

    workspace = types.ModuleType("sglang.srt.lora.moe.workspace")
    workspace.MoeLoraWorkspace = object

    module_name = "_host_route_factory"
    spec = importlib.util.spec_from_file_location(
        module_name, LORA_MOE / "route_factory.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    stubs = {
        **packages,
        "sglang.srt.lora.moe.execution_plan": PLAN,
        routing.__name__: routing,
        joint_routing.__name__: joint_routing,
        workspace.__name__: workspace,
        module_name: module,
    }
    with mock.patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


ROUTE_FACTORY = _load_route_factory()


def _load_joint_routing():
    package_names = (
        "sglang",
        "sglang.srt",
        "sglang.srt.lora",
        "sglang.srt.lora.moe",
    )
    packages = {}
    for name in package_names:
        package = types.ModuleType(name)
        package.__path__ = []
        packages[name] = package

    fake_triton = types.ModuleType("triton")
    fake_triton.jit = lambda function: function
    fake_triton.cdiv = lambda value, divisor: (value + divisor - 1) // divisor
    fake_tl = types.ModuleType("triton.language")
    fake_triton.language = fake_tl

    routing = types.ModuleType("sglang.srt.lora.moe.routing")
    routing.ROUTE_ALIGNED = "aligned"
    routing.RouteView = _HostRouteView
    routing._routing_capacity = lambda num_pairs, block_size, num_virtual: (
        block_size * ((num_pairs + block_size - 1) // block_size + num_virtual)
    )
    routing.virtual_expert_ids_inline = lambda *_args, **_kwargs: None

    workspace = types.ModuleType("sglang.srt.lora.moe.workspace")
    workspace.MoeLoraWorkspace = object

    module_name = "_host_joint_routing"
    spec = importlib.util.spec_from_file_location(
        module_name, LORA_MOE / "joint_routing.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(
        sys.modules,
        {
            **packages,
            "triton": fake_triton,
            "triton.language": fake_tl,
            routing.__name__: routing,
            workspace.__name__: workspace,
            module_name: module,
        },
    ):
        spec.loader.exec_module(module)
    return module


JOINT_ROUTING = _load_joint_routing()


def _load_routing():
    package_names = (
        "sglang",
        "sglang.kernels",
        "sglang.kernels.ops",
        "sglang.kernels.ops.moe",
        "sglang.srt",
        "sglang.srt.lora",
        "sglang.srt.lora.moe",
    )
    packages = {}
    for name in package_names:
        package = types.ModuleType(name)
        package.__path__ = []
        packages[name] = package

    fake_triton = types.ModuleType("triton")
    fake_triton.jit = lambda function: function
    fake_triton.cdiv = lambda value, divisor: (value + divisor - 1) // divisor
    fake_tl = types.ModuleType("triton.language")
    fake_triton.language = fake_tl

    virtual_experts = types.ModuleType("sglang.kernels.ops.moe.virtual_experts")
    virtual_experts._align_block_size_jit = lambda *_args, **_kwargs: None
    virtual_experts._align_block_size_torch = lambda *_args, **_kwargs: None

    # The policy module is pure shape math; load the real one so the host
    # under test cannot drift from production dispatch.
    policy_spec = importlib.util.spec_from_file_location(
        "_host_routing_shape", LORA_MOE / "routing_shape.py"
    )
    assert policy_spec is not None and policy_spec.loader is not None
    policy = importlib.util.module_from_spec(policy_spec)
    policy_spec.loader.exec_module(policy)

    module_name = "_host_routing"
    spec = importlib.util.spec_from_file_location(module_name, LORA_MOE / "routing.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(
        sys.modules,
        {
            **packages,
            "triton": fake_triton,
            "triton.language": fake_tl,
            virtual_experts.__name__: virtual_experts,
            "sglang.srt.lora.moe.routing_shape": policy,
            module_name: module,
        },
    ):
        spec.loader.exec_module(module)
    return module


ROUTING = _load_routing()


class _Workspace:
    def __init__(self):
        self.tensors: dict[str, torch.Tensor] = {}

    def tensor(self, name, shape, *, dtype, device, **_kwargs):
        factory = (
            torch.zeros if _kwargs.get("zero_on_first_allocation") else torch.empty
        )
        value = factory(shape, dtype=dtype, device=device)
        self.tensors[name] = value
        return value


class _KernelRecorder:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid, args, kwargs))

        return launch


def _route(
    topk_ids: torch.Tensor,
    token_slots: torch.Tensor,
    *,
    block_size: int,
    padded_count: torch.Tensor,
    lora_experts_per_adapter: int = 2,
    max_loras: int = 2,
    shared_outer_local_expert_count: int | None = None,
    virtual_topk_ids: torch.Tensor | None = None,
) -> _HostRouteView:
    return _HostRouteView(
        view="aligned",
        num_virtual_experts=lora_experts_per_adapter * max_loras,
        block_size=block_size,
        topk_ids=topk_ids,
        token_slots=token_slots,
        lora_experts_per_adapter=lora_experts_per_adapter,
        max_loras=max_loras,
        shared_outer_local_expert_count=shared_outer_local_expert_count,
        maybe_virtual_topk_ids=virtual_topk_ids,
        maybe_sorted_pair_ids=torch.arange(2, dtype=torch.int32),
        maybe_block_virtual_expert_ids=torch.arange(1, dtype=torch.int32),
        maybe_num_pairs_post_padded=padded_count,
    )


class TestDualGateARoute(unittest.TestCase):
    def setUp(self):
        self.topk_ids = torch.tensor([[0, 1]], dtype=torch.int32)
        self.token_slots = torch.tensor([0], dtype=torch.int32)
        self.workspace = _Workspace()

    def _build(self, *, gate_a_block_size: int, fused_shapes: bool = True):
        """Build the reference plan with the standalone (non-dual) route path.

        ``fused_shapes`` keeps the stub's always-fused dispatch; ``False``
        pins the JIT small-shape regime, where the standalone builds carry no
        caller-owned fused metadata.  Dual-granularity dispatch is asserted
        separately: this helper's ``_pair_route`` mock would hide it.
        """
        cached_padded_count = torch.zeros(1, dtype=torch.int32)
        calls: list[int] = []

        def fake_pair_route(
            topk_ids,
            token_slots,
            *,
            is_shared_outer,
            num_local_experts,
            max_loras,
            block_size,
            view,
            use_pdl,
            num_pairs_post_padded_out=None,
            fused_align_scratch=None,
        ):
            self.assertIs(is_shared_outer, False)
            self.assertEqual(num_local_experts, 2)
            self.assertEqual(max_loras, 2)
            self.assertEqual(view, "aligned")
            self.assertIsNone(use_pdl)
            calls.append(block_size)
            if fused_shapes:
                self.assertIsNotNone(num_pairs_post_padded_out)
                self.assertIsNotNone(fused_align_scratch)
                # Model both the fused builder's shared cache and its direct
                # write into the route factory's caller-owned output.
                cached_padded_count.fill_(block_size)
                num_pairs_post_padded_out.fill_(block_size)
                padded_count = num_pairs_post_padded_out
            else:
                self.assertIsNone(num_pairs_post_padded_out)
                self.assertIsNone(fused_align_scratch)
                padded_count = torch.tensor([block_size], dtype=torch.int32)
            return _route(
                topk_ids,
                token_slots,
                block_size=block_size,
                padded_count=padded_count,
            )

        with contextlib.ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(
                    ROUTE_FACTORY, "_pair_route", side_effect=fake_pair_route
                )
            )
            if not fused_shapes:
                stack.enter_context(
                    mock.patch.object(
                        ROUTE_FACTORY, "uses_fused_align", return_value=False
                    )
                )
            routes = ROUTE_FACTORY.build_routes(
                PLAN.SERIAL_MATERIALIZED_REFERENCE,
                topk_ids=self.topk_ids,
                token_slots=self.token_slots,
                num_local_experts=2,
                max_loras=2,
                block_size=16,
                gate_a_block_size=gate_a_block_size,
                workspace=self.workspace,
            )
        return routes, calls, cached_padded_count

    def test_m16_m64_builds_both_views_in_one_dual_granularity_pass(self):
        """At fused shapes the two per-expert granularities share ONE pass.

        The dual builder must receive the SAME workspace-owned metadata the
        standalone builds would have used (so alternating paths across
        forwards reuses storage), and no standalone per-expert build may run.
        """
        dual_calls = []

        def fake_dual(
            topk_ids,
            token_slots,
            *,
            lora_experts_per_adapter,
            max_loras,
            block_sizes,
            num_pairs_post_padded_outs,
            scratches,
            use_pdl,
        ):
            self.assertIs(topk_ids, self.topk_ids)
            self.assertIs(token_slots, self.token_slots)
            self.assertEqual(lora_experts_per_adapter, 2)
            self.assertEqual(max_loras, 2)
            self.assertIsNone(use_pdl)
            dual_calls.append({"block_sizes": block_sizes, "scratches": scratches})
            views = []
            for block_size, padded_count in zip(
                block_sizes, num_pairs_post_padded_outs
            ):
                padded_count.fill_(block_size)
                views.append(
                    _route(
                        topk_ids,
                        token_slots,
                        block_size=block_size,
                        padded_count=padded_count,
                    )
                )
            return views[0], views[1]

        def unexpected_pair_route(*_args, **_kwargs):
            raise AssertionError(
                "the dual pass must replace every standalone per-expert build"
            )

        with (
            mock.patch.object(
                ROUTE_FACTORY,
                "build_dual_granularity_aligned_routes",
                side_effect=fake_dual,
            ),
            mock.patch.object(
                ROUTE_FACTORY, "_pair_route", side_effect=unexpected_pair_route
            ),
        ):
            routes = ROUTE_FACTORY.build_routes(
                PLAN.SERIAL_MATERIALIZED_REFERENCE,
                topk_ids=self.topk_ids,
                token_slots=self.token_slots,
                num_local_experts=2,
                max_loras=2,
                block_size=16,
                gate_a_block_size=64,
                workspace=self.workspace,
            )

        self.assertEqual(len(dual_calls), 1)
        self.assertEqual(dual_calls[0]["block_sizes"], (16, 64))
        self.assertEqual(routes.aligned_per_expert.block_size, 16)
        self.assertEqual(routes.gate_a_aligned_per_expert.block_size, 64)
        self.assertEqual(
            routes.aligned_per_expert.maybe_num_pairs_post_padded.item(), 16
        )
        self.assertEqual(
            routes.gate_a_aligned_per_expert.maybe_num_pairs_post_padded.item(), 64
        )

        prefixes = ("route:aligned_per_expert", "route:gate_a_aligned_per_expert")
        fields = ("counts", "block_cumulative", "cursor", "bucket_end")
        for prefix in prefixes:
            for field in fields + ("padded_pairs",):
                self.assertIn(f"{prefix}:{field}", self.workspace.tensors)
        for field in fields + ("padded_pairs",):
            self.assertNotEqual(
                self.workspace.tensors[f"{prefixes[0]}:{field}"].data_ptr(),
                self.workspace.tensors[f"{prefixes[1]}:{field}"].data_ptr(),
            )
        for route, prefix in (
            (routes.aligned_per_expert, prefixes[0]),
            (routes.gate_a_aligned_per_expert, prefixes[1]),
        ):
            self.assertEqual(
                route.maybe_num_pairs_post_padded.data_ptr(),
                self.workspace.tensors[f"{prefix}:padded_pairs"].data_ptr(),
            )
        for scratch, prefix in zip(dual_calls[0]["scratches"], prefixes):
            for field in fields:
                self.assertEqual(
                    getattr(scratch, field).data_ptr(),
                    self.workspace.tensors[f"{prefix}:{field}"].data_ptr(),
                )

    def test_m16_m64_small_shapes_keep_two_standalone_builds(self):
        """Below the fused dispatch edge the measured JIT paths run unchanged."""
        routes, calls, _ = self._build(gate_a_block_size=64, fused_shapes=False)

        self.assertEqual(calls, [16, 64])
        self.assertEqual(routes.aligned_per_expert.block_size, 16)
        self.assertEqual(routes.gate_a_aligned_per_expert.block_size, 64)
        # The JIT regime owns its metadata; no fused scratch is allocated.
        self.assertEqual(self.workspace.tensors, {})

    def test_equal_m_tile_reuses_the_canonical_route(self):
        routes, calls, _ = self._build(gate_a_block_size=16)
        self.assertEqual(calls, [16])
        self.assertIsNone(routes.gate_a_aligned_per_expert)
        self.assertEqual(routes.aligned_per_expert.block_size, 16)
        self.assertEqual(
            set(self.workspace.tensors),
            {
                f"route:aligned_per_expert:{field}"
                for field in (
                    "counts",
                    "block_cumulative",
                    "cursor",
                    "bucket_end",
                    "padded_pairs",
                )
            },
        )

    def test_second_route_is_rejected_for_non_grouped_gate_a(self):
        reference = PLAN.SERIAL_MATERIALIZED_REFERENCE
        token_dedup_plan = dataclasses.replace(
            reference,
            gate_a=PLAN.LoraASpec(
                PLAN.FactorSite.GATE_UP,
                PLAN.LoraAFamily.TOKEN_DEDUP_GROUPED,
                True,
                PLAN.FactorLayout.TOKEN_MAJOR,
            ),
            gate_b=dataclasses.replace(
                reference.gate_b,
                input_layout=PLAN.FactorLayout.TOKEN_MAJOR,
            ),
        )
        with self.assertRaisesRegex(
            ValueError, "qualified only for grouped.*gate/up-A"
        ):
            ROUTE_FACTORY.build_routes(
                token_dedup_plan,
                topk_ids=self.topk_ids,
                token_slots=self.token_slots,
                num_local_experts=2,
                max_loras=2,
                block_size=16,
                gate_a_block_size=64,
                workspace=self.workspace,
            )


class TestRoutePdlWiring(unittest.TestCase):
    def test_standard_plan_threads_explicit_pdl_to_every_aligned_route(self):
        reference = PLAN.SERIAL_MATERIALIZED_REFERENCE
        shared_down = dataclasses.replace(
            reference,
            down_b=dataclasses.replace(
                reference.down_b,
                is_shared_outer=True,
            ),
        )
        shared_token = dataclasses.replace(
            reference,
            gate_a=PLAN.LoraASpec(
                PLAN.FactorSite.GATE_UP,
                PLAN.LoraAFamily.TOKEN_DEDUP_GROUPED,
                True,
                PLAN.FactorLayout.TOKEN_MAJOR,
            ),
            gate_b=dataclasses.replace(
                reference.gate_b,
                input_layout=PLAN.FactorLayout.TOKEN_MAJOR,
            ),
        )

        for enabled in (False, True):
            calls = []
            dual_calls = []

            def fake_aligned(
                topk_ids,
                token_slots,
                *,
                is_shared_outer,
                num_local_experts,
                max_loras,
                block_size,
                use_pdl,
                workspace,
                scratch_prefix,
            ):
                calls.append((scratch_prefix, use_pdl))
                return _route(
                    topk_ids,
                    token_slots,
                    block_size=block_size,
                    padded_count=torch.tensor([block_size], dtype=torch.int32),
                )

            def fake_dual(
                topk_ids,
                token_slots,
                *,
                block_sizes,
                num_pairs_post_padded_outs,
                use_pdl,
                **_kwargs,
            ):
                dual_calls.append(use_pdl)
                views = []
                for block_size, padded_count in zip(
                    block_sizes, num_pairs_post_padded_outs
                ):
                    padded_count.fill_(block_size)
                    views.append(
                        _route(
                            topk_ids,
                            token_slots,
                            block_size=block_size,
                            padded_count=padded_count,
                        )
                    )
                return views[0], views[1]

            with (
                mock.patch.object(
                    ROUTE_FACTORY,
                    "_aligned_pair_route",
                    side_effect=fake_aligned,
                ),
                mock.patch.object(
                    ROUTE_FACTORY,
                    "build_dual_granularity_aligned_routes",
                    side_effect=fake_dual,
                ),
            ):
                for plan in (reference, shared_down, shared_token):
                    ROUTE_FACTORY.build_routes(
                        dataclasses.replace(plan, route_pdl=enabled),
                        topk_ids=torch.tensor([[0, 1]], dtype=torch.int32),
                        token_slots=torch.tensor([0], dtype=torch.int32),
                        num_local_experts=2,
                        max_loras=2,
                        block_size=16,
                        gate_a_block_size=(16 if plan is shared_token else 64),
                        workspace=_Workspace(),
                    )

            by_prefix = {}
            for prefix, value in calls:
                by_prefix.setdefault(prefix, set()).add(value)
            # reference and shared_down retain both per-expert granularities,
            # so their per-expert + gate-A routes ride the dual pass; the
            # token-dedup plan (equal tiles) keeps every standalone build.
            self.assertEqual(
                set(by_prefix),
                {
                    "route:aligned_per_expert",
                    "route:aligned_shared_outer",
                    "route:shared_token",
                },
            )
            self.assertTrue(all(values == {enabled} for values in by_prefix.values()))
            self.assertEqual(dual_calls, [enabled, enabled])

    def test_joint_plan_threads_explicit_pdl_control(self):
        reference = PLAN.SERIAL_MATERIALIZED_REFERENCE
        shared_down_b = dataclasses.replace(
            reference.down_b,
            is_shared_outer=True,
        )
        joint_calls = []
        standard_calls = []

        def fake_joint(
            topk_ids,
            token_slots,
            *,
            num_local_experts,
            max_loras,
            block_size,
            workspace,
            use_pdl,
        ):
            joint_calls.append(use_pdl)
            route = _route(
                topk_ids,
                token_slots,
                block_size=block_size,
                padded_count=torch.tensor([block_size], dtype=torch.int32),
            )
            return route, route

        def fake_pair_route(
            topk_ids,
            token_slots,
            *,
            is_shared_outer,
            num_local_experts,
            max_loras,
            block_size,
            view,
            use_pdl,
            num_pairs_post_padded_out=None,
            fused_align_scratch=None,
        ):
            standard_calls.append(use_pdl)
            num_pairs_post_padded_out.fill_(block_size)
            return _route(
                topk_ids,
                token_slots,
                block_size=block_size,
                padded_count=num_pairs_post_padded_out,
            )

        for enabled in (False, True):
            plan = dataclasses.replace(
                reference,
                down_b=shared_down_b,
                route_builder=PLAN.RouteBuilderFamily.JOINT_SHARED_OUTER,
                route_pdl=enabled,
            )
            with (
                mock.patch.object(
                    ROUTE_FACTORY,
                    "build_joint_shared_routes",
                    side_effect=fake_joint,
                ),
                mock.patch.object(
                    ROUTE_FACTORY,
                    "_pair_route",
                    side_effect=fake_pair_route,
                ),
            ):
                ROUTE_FACTORY.build_routes(
                    plan,
                    topk_ids=torch.tensor([[0, 1]], dtype=torch.int32),
                    token_slots=torch.tensor([0], dtype=torch.int32),
                    num_local_experts=2,
                    max_loras=2,
                    block_size=16,
                    gate_a_block_size=64,
                    workspace=_Workspace(),
                )

        self.assertEqual(joint_calls, [False, True])
        self.assertEqual(standard_calls, [False, False])

    def _run_joint(self, *, use_pdl):
        recorders = [_KernelRecorder() for _ in range(3)]
        with (
            mock.patch.object(JOINT_ROUTING, "_joint_hist_kernel", recorders[0]),
            mock.patch.object(JOINT_ROUTING, "_dual_scan_kernel", recorders[1]),
            mock.patch.object(
                JOINT_ROUTING, "_joint_expand_scatter_kernel", recorders[2]
            ),
        ):
            routes = JOINT_ROUTING.build_joint_shared_routes(
                torch.tensor([[0, 1]], dtype=torch.int32),
                torch.tensor([0], dtype=torch.int32),
                num_local_experts=2,
                max_loras=2,
                block_size=16,
                workspace=_Workspace(),
                use_pdl=use_pdl,
            )
        return routes, recorders

    def test_joint_route_launches_real_pdl_chain(self):
        routes, (hist, scan, expand) = self._run_joint(use_pdl=True)

        self.assertEqual(len(routes), 2)
        self.assertTrue(hist.calls[0][2]["USE_PDL"])
        self.assertNotIn("launch_pdl", hist.calls[0][2])
        for consumer in (scan, expand):
            self.assertTrue(consumer.calls[0][2]["USE_PDL"])
            self.assertTrue(consumer.calls[0][2]["launch_pdl"])

    def test_joint_route_delegates_default_to_architecture_support(self):
        arch = types.ModuleType("sglang.kernels.jit.utils")
        arch.is_arch_support_pdl = lambda: True
        with mock.patch.dict(sys.modules, {arch.__name__: arch}):
            _, recorders = self._run_joint(use_pdl=None)

        for recorder in recorders:
            self.assertTrue(recorder.calls[0][2]["USE_PDL"])

    def test_joint_kernel_dependency_operations_are_complete(self):
        source = (LORA_MOE / "joint_routing.py").read_text()
        tree = ast.parse(source)
        function_nodes = {
            node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
        }
        functions = {name: ast.unparse(node) for name, node in function_nodes.items()}

        hist = functions["_joint_hist_kernel"]
        scan = functions["_dual_scan_kernel"]
        expand = functions["_joint_expand_scatter_kernel"]
        self.assertIn("gdc_launch_dependents", hist)
        self.assertNotIn("gdc_wait", hist)
        self.assertIn("gdc_wait", scan)
        self.assertIn("gdc_launch_dependents", scan)
        self.assertIn("gdc_wait", expand)
        self.assertNotIn("gdc_launch_dependents", expand)

        # K3 must be released before either scan body, so its pair path can
        # overlap scan work while still waiting before cursor consumption.
        scan_ifs = [
            node
            for node in function_nodes["_dual_scan_kernel"].body
            if isinstance(node, ast.If)
        ]
        self.assertEqual(len(scan_ifs), 2)
        self.assertEqual(ast.unparse(scan_ifs[0].test), "USE_PDL")
        pdl_body = ast.unparse(scan_ifs[0])
        self.assertLess(
            pdl_body.index("gdc_wait"),
            pdl_body.index("gdc_launch_dependents"),
        )
        self.assertIn("tl.program_id", ast.unparse(scan_ifs[1].test))
        self.assertTrue(scan_ifs[1].orelse)


class TestDualGranularityRoutingHost(unittest.TestCase):
    """Host-side guards for the dual-granularity fused route builder."""

    def _scratch(self, num_buckets: int) -> object:
        return ROUTING.FusedAlignScratch(
            counts=torch.zeros(num_buckets, dtype=torch.int32),
            block_cumulative=torch.empty(num_buckets + 1, dtype=torch.int32),
            cursor=torch.empty(num_buckets, dtype=torch.int32),
            bucket_end=torch.empty(num_buckets, dtype=torch.int32),
        )

    def _run(
        self,
        *,
        use_pdl=None,
        block_sizes=(16, 64),
        num_tokens=4,
        top_k=2,
        scratches=None,
        num_pairs_post_padded_outs=None,
    ):
        recorders = [_KernelRecorder() for _ in range(3)]
        num_buckets = 2 * 2 + 1
        if scratches is None:
            scratches = (self._scratch(num_buckets), self._scratch(num_buckets))
        if num_pairs_post_padded_outs is None:
            num_pairs_post_padded_outs = (
                torch.empty(1, dtype=torch.int32),
                torch.empty(1, dtype=torch.int32),
            )
        with (
            mock.patch.object(ROUTING, "_dual_granularity_hist_kernel", recorders[0]),
            mock.patch.object(ROUTING, "_dual_granularity_scan_kernel", recorders[1]),
            mock.patch.object(
                ROUTING, "_dual_granularity_expand_scatter_kernel", recorders[2]
            ),
        ):
            views = ROUTING.build_dual_granularity_aligned_routes(
                torch.zeros((num_tokens, top_k), dtype=torch.int32),
                torch.zeros(num_tokens, dtype=torch.int32),
                lora_experts_per_adapter=2,
                max_loras=2,
                block_sizes=block_sizes,
                num_pairs_post_padded_outs=num_pairs_post_padded_outs,
                scratches=scratches,
                use_pdl=use_pdl,
            )
        return views, recorders, scratches, num_pairs_post_padded_outs

    def test_one_kernel_triple_covers_both_granularities(self):
        views, (hist, scan, expand), scratches, padded = self._run()

        # One histogram over pairs feeding BOTH counter arrays.
        self.assertEqual(len(hist.calls), 1)
        grid, args, kwargs = hist.calls[0]
        self.assertEqual(grid, (1,))
        self.assertEqual(args[2].data_ptr(), scratches[0].counts.data_ptr())
        self.assertEqual(args[3].data_ptr(), scratches[1].counts.data_ptr())
        self.assertEqual(kwargs["NUM_BUCKETS"], 5)

        # One two-CTA scan carrying one M tile per program.
        self.assertEqual(len(scan.calls), 1)
        grid, args, kwargs = scan.calls[0]
        self.assertEqual(grid, (2,))
        self.assertEqual(kwargs["BLOCK_SIZE_M_FIRST"], 16)
        self.assertEqual(kwargs["BLOCK_SIZE_M_SECOND"], 64)
        self.assertEqual(args[4].data_ptr(), padded[0].data_ptr())
        self.assertEqual(args[9].data_ptr(), padded[1].data_ptr())

        # One expand/scatter over two label halves plus the pair half.
        self.assertEqual(len(expand.calls), 1)
        grid, args, kwargs = expand.calls[0]
        self.assertEqual(grid, (3,))
        # capacity(P=8, bs, V=4): 96 -> 6 blocks at M=16, 384 -> 6 at M=64.
        self.assertEqual(args[7], 6)
        self.assertEqual(args[8], 1)
        self.assertEqual(args[14], 6)
        self.assertEqual(args[15], 1)
        self.assertEqual(args[16], 8)
        # [0, NUM_BUCKETS] holds NUM_BUCKETS + 1 states -> bit_length, not
        # bit_length of (NUM_BUCKETS - 1).
        self.assertEqual(kwargs["SEARCH_STEPS"], 3)
        self.assertEqual(kwargs["BLOCK_SIZE_M_FIRST"], 16)
        self.assertEqual(kwargs["BLOCK_SIZE_M_SECOND"], 64)

        self.assertEqual(views[0].block_size, 16)
        self.assertEqual(views[1].block_size, 64)
        self.assertEqual(views[0].view, "aligned")
        self.assertEqual(views[0].num_virtual_experts, 4)
        self.assertEqual(views[0].maybe_sorted_pair_ids.numel(), 96)
        self.assertEqual(views[1].maybe_sorted_pair_ids.numel(), 384)
        self.assertEqual(views[0].maybe_block_virtual_expert_ids.numel(), 6)
        self.assertEqual(views[1].maybe_block_virtual_expert_ids.numel(), 6)
        for view, count in zip(views, padded):
            self.assertEqual(
                view.maybe_num_pairs_post_padded.data_ptr(), count.data_ptr()
            )

    def test_pdl_launch_wiring_matches_the_incumbent_chain(self):
        for enabled in (False, True):
            with self.subTest(use_pdl=enabled):
                _, recorders, _, _ = self._run(use_pdl=enabled)
                hist, scan, expand = recorders
                for recorder in recorders:
                    self.assertEqual(recorder.calls[0][2]["USE_PDL"], enabled)
                # Edges are consecutive-launch only: the producer never
                # carries launch_pdl, both consumers do (when enabled).
                self.assertNotIn("launch_pdl", hist.calls[0][2])
                for consumer in (scan, expand):
                    self.assertEqual(
                        consumer.calls[0][2].get("launch_pdl", False), enabled
                    )

    def test_none_pdl_stays_off_for_the_standard_route(self):
        _, recorders, _, _ = self._run(use_pdl=None)
        for recorder in recorders:
            self.assertFalse(recorder.calls[0][2]["USE_PDL"])
            self.assertNotIn("launch_pdl", recorder.calls[0][2])

    def test_empty_pair_domain_keeps_host_static_launches(self):
        views, (hist, scan, expand), _, _ = self._run(num_tokens=0)
        self.assertEqual(hist.calls[0][0], (1,))
        self.assertEqual(scan.calls[0][0], (2,))
        self.assertEqual(expand.calls[0][0], (3,))
        self.assertEqual(views[0].maybe_sorted_pair_ids.numel(), 0)
        self.assertEqual(views[1].maybe_block_virtual_expert_ids.numel(), 0)

    def test_kernel_dependency_operations_are_complete(self):
        source = (LORA_MOE / "routing.py").read_text()
        tree = ast.parse(source)
        function_nodes = {
            node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
        }
        functions = {name: ast.unparse(node) for name, node in function_nodes.items()}

        hist = functions["_dual_granularity_hist_kernel"]
        scan = functions["_dual_granularity_scan_kernel"]
        expand = functions["_dual_granularity_expand_scatter_kernel"]
        self.assertIn("gdc_launch_dependents", hist)
        self.assertNotIn("gdc_wait", hist)
        self.assertIn("gdc_wait", scan)
        self.assertIn("gdc_launch_dependents", scan)
        self.assertIn("gdc_wait", expand)
        self.assertNotIn("gdc_launch_dependents", expand)

        # K3 must be released before either scan body, so its pair path can
        # overlap scan work while still waiting before cursor consumption.
        scan_ifs = [
            node
            for node in function_nodes["_dual_granularity_scan_kernel"].body
            if isinstance(node, ast.If)
        ]
        self.assertEqual(len(scan_ifs), 2)
        self.assertEqual(ast.unparse(scan_ifs[0].test), "USE_PDL")
        pdl_body = ast.unparse(scan_ifs[0])
        self.assertLess(
            pdl_body.index("gdc_wait"),
            pdl_body.index("gdc_launch_dependents"),
        )
        self.assertIn("tl.program_id", ast.unparse(scan_ifs[1].test))
        self.assertTrue(scan_ifs[1].orelse)

    def test_aliased_scratch_is_rejected(self):
        shared = self._scratch(5)
        with self.assertRaisesRegex(ValueError, "disjoint scratch"):
            self._run(scratches=(shared, shared))

    def test_scratch_contract_is_checked_per_granularity(self):
        bad = ROUTING.FusedAlignScratch(
            counts=torch.zeros(4, dtype=torch.int32),
            block_cumulative=torch.empty(6, dtype=torch.int32),
            cursor=torch.empty(5, dtype=torch.int32),
            bucket_end=torch.empty(5, dtype=torch.int32),
        )
        with self.assertRaisesRegex(ValueError, "route 1 counts"):
            self._run(scratches=(self._scratch(5), bad))

    def test_geometry_contract_violations_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "must all be positive"):
            self._run(block_sizes=(16, 0))
        with self.assertRaisesRegex(ValueError, "exactly two"):
            self._run(block_sizes=(16,))
        with self.assertRaisesRegex(ValueError, "topk_ids \\[T,K\\]"):
            ROUTING.build_dual_granularity_aligned_routes(
                torch.zeros(4, dtype=torch.int32),
                torch.zeros(4, dtype=torch.int32),
                lora_experts_per_adapter=2,
                max_loras=2,
                block_sizes=(16, 64),
                num_pairs_post_padded_outs=(
                    torch.empty(1, dtype=torch.int32),
                    torch.empty(1, dtype=torch.int32),
                ),
                scratches=(self._scratch(5), self._scratch(5)),
                use_pdl=None,
            )
        with self.assertRaisesRegex(ValueError, "int32 plan math"):
            ROUTING.build_dual_granularity_aligned_routes(
                torch.zeros((4, 2), dtype=torch.int32),
                torch.zeros(4, dtype=torch.int32),
                lora_experts_per_adapter=2**30,
                max_loras=4,
                block_sizes=(16, 64),
                num_pairs_post_padded_outs=(
                    torch.empty(1, dtype=torch.int32),
                    torch.empty(1, dtype=torch.int32),
                ),
                scratches=(self._scratch(5), self._scratch(5)),
                use_pdl=None,
            )


class TestSharedTokenRoute(unittest.TestCase):
    def test_token_dedup_skips_tokens_without_a_local_pair(self):
        reference = PLAN.SERIAL_MATERIALIZED_REFERENCE
        shared_plan = dataclasses.replace(
            reference,
            gate_a=PLAN.LoraASpec(
                PLAN.FactorSite.GATE_UP,
                PLAN.LoraAFamily.TOKEN_DEDUP_GROUPED,
                True,
                PLAN.FactorLayout.TOKEN_MAJOR,
            ),
            gate_b=dataclasses.replace(
                reference.gate_b,
                input_layout=PLAN.FactorLayout.TOKEN_MAJOR,
            ),
        )
        topk_ids = torch.tensor([[-1, -1], [-1, 1], [0, -1]], dtype=torch.int32)
        token_slots = torch.tensor([0, 1, 0], dtype=torch.int32)
        workspace = _Workspace()
        calls = []

        def fake_pair_route(
            route_topk_ids,
            route_token_slots,
            *,
            is_shared_outer,
            num_local_experts,
            max_loras,
            block_size,
            view,
            use_pdl,
            num_pairs_post_padded_out=None,
            fused_align_scratch=None,
        ):
            self.assertIsNone(use_pdl)
            calls.append(
                (
                    is_shared_outer,
                    route_topk_ids.clone(),
                    route_token_slots.clone(),
                )
            )
            num_pairs_post_padded_out.fill_(block_size)
            return _route(
                route_topk_ids,
                route_token_slots,
                block_size=block_size,
                padded_count=num_pairs_post_padded_out,
            )

        with mock.patch.object(
            ROUTE_FACTORY, "_pair_route", side_effect=fake_pair_route
        ):
            routes = ROUTE_FACTORY.build_routes(
                shared_plan,
                topk_ids=topk_ids,
                token_slots=token_slots,
                num_local_experts=2,
                max_loras=2,
                block_size=16,
                workspace=workspace,
            )

        shared_call = next(call for call in calls if call[0] is True)
        torch.testing.assert_close(
            shared_call[1], torch.zeros((3, 1), dtype=torch.int32)
        )
        torch.testing.assert_close(
            shared_call[2], torch.tensor([-1, 1, 0], dtype=torch.int32)
        )
        self.assertIsNotNone(routes.shared_token)
        self.assertIn("route:shared_token_slots", workspace.tensors)

    def test_large_shared_token_route_cannot_overwrite_retained_pair_counts(self):
        """Every retained fused route owns its scalar, even at shared bucket V."""
        reference = PLAN.SERIAL_MATERIALIZED_REFERENCE
        shared_plan = dataclasses.replace(
            reference,
            gate_a=PLAN.LoraASpec(
                PLAN.FactorSite.GATE_UP,
                PLAN.LoraAFamily.TOKEN_DEDUP_GROUPED,
                True,
                PLAN.FactorLayout.TOKEN_MAJOR,
            ),
            gate_b=dataclasses.replace(
                reference.gate_b,
                input_layout=PLAN.FactorLayout.TOKEN_MAJOR,
            ),
            down_b=dataclasses.replace(
                reference.down_b,
                is_shared_outer=True,
            ),
        )
        num_tokens = 16384
        top_k = 8
        topk_ids = torch.zeros((num_tokens, top_k), dtype=torch.int32)
        token_slots = torch.zeros(num_tokens, dtype=torch.int32)

        for num_local_experts in (1, 2):
            with self.subTest(num_local_experts=num_local_experts):
                workspace = _Workspace()
                cached_counts: dict[int, torch.Tensor] = {}

                def fake_pair_route(
                    route_topk_ids,
                    route_token_slots,
                    *,
                    is_shared_outer,
                    num_local_experts,
                    max_loras,
                    block_size,
                    view,
                    use_pdl,
                    num_pairs_post_padded_out=None,
                    fused_align_scratch=None,
                ):
                    self.assertEqual(view, "aligned")
                    self.assertIsNone(use_pdl)
                    per_adapter = num_local_experts if is_shared_outer is False else 1
                    num_virtual = per_adapter * max_loras
                    cached_count = cached_counts.setdefault(
                        num_virtual, torch.zeros(1, dtype=torch.int32)
                    )
                    # Model the fused builder's cache entry keyed only by
                    # bucket count. The pair plans have T*K rows; the later
                    # shared-token plan has T rows and overwrites the same
                    # shared-outer entry (and the per-expert entry at E=1).
                    cached_count.fill_(route_topk_ids.numel())
                    self.assertIsNotNone(num_pairs_post_padded_out)
                    self.assertIsNotNone(fused_align_scratch)
                    num_pairs_post_padded_out.fill_(route_topk_ids.numel())
                    return _route(
                        route_topk_ids,
                        route_token_slots,
                        block_size=block_size,
                        padded_count=num_pairs_post_padded_out,
                        lora_experts_per_adapter=per_adapter,
                        max_loras=max_loras,
                        shared_outer_local_expert_count=(
                            num_local_experts if is_shared_outer is True else None
                        ),
                    )

                with mock.patch.object(
                    ROUTE_FACTORY, "_pair_route", side_effect=fake_pair_route
                ):
                    routes = ROUTE_FACTORY.build_routes(
                        shared_plan,
                        topk_ids=topk_ids,
                        token_slots=token_slots,
                        num_local_experts=num_local_experts,
                        max_loras=2,
                        block_size=16,
                        workspace=workspace,
                    )

                pair_count = num_tokens * top_k
                token_count = num_tokens
                self.assertEqual(
                    routes.aligned_per_expert.maybe_num_pairs_post_padded.item(),
                    pair_count,
                )
                self.assertEqual(
                    routes.aligned_shared_outer.maybe_num_pairs_post_padded.item(),
                    pair_count,
                )
                self.assertEqual(
                    routes.shared_token.maybe_num_pairs_post_padded.item(),
                    token_count,
                )

                shared_cached_count = cached_counts[2]
                self.assertEqual(shared_cached_count.item(), token_count)
                for route in (
                    routes.aligned_per_expert,
                    routes.aligned_shared_outer,
                    routes.shared_token,
                ):
                    self.assertNotEqual(
                        route.maybe_num_pairs_post_padded.data_ptr(),
                        shared_cached_count.data_ptr(),
                    )
                self.assertTrue(
                    {
                        "route:aligned_per_expert:padded_pairs",
                        "route:aligned_shared_outer:padded_pairs",
                        "route:shared_token:padded_pairs",
                    }.issubset(workspace.tensors)
                )
                scratch_fields = (
                    "counts",
                    "block_cumulative",
                    "cursor",
                    "bucket_end",
                )
                route_scratch = [
                    {
                        workspace.tensors[f"{prefix}:{field}"].data_ptr()
                        for field in scratch_fields
                    }
                    for prefix in (
                        "route:aligned_per_expert",
                        "route:aligned_shared_outer",
                        "route:shared_token",
                    )
                ]
                for index, pointers in enumerate(route_scratch):
                    self.assertEqual(len(pointers), len(scratch_fields))
                    for other in route_scratch[index + 1 :]:
                        self.assertTrue(pointers.isdisjoint(other))


class TestRunnerRouteSelectionSource(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = (LORA_MOE / "moe_lora_runner.py").read_text()
        cls.tree = ast.parse(source)

    def _method(self, name: str) -> ast.FunctionDef:
        runner = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.ClassDef) and node.name == "MoeLoraRunner"
        )
        return next(
            node
            for node in runner.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )

    def test_only_gate_a_can_select_the_dedicated_aligned_route(self):
        route_for_a = ast.unparse(self._method("_route_for_a"))
        route_for_b = ast.unparse(self._method("_route_for_b"))
        self.assertIn("routes.gate_a_aligned_per_expert", route_for_a)
        self.assertIn("spec.site is FactorSite.GATE_UP", route_for_a)
        self.assertIn("routes.aligned(spec.is_shared_outer)", route_for_a)
        self.assertNotIn("gate_a_aligned_per_expert", route_for_b)
        self.assertIn("routes.aligned(spec.is_shared_outer)", route_for_b)


class TestLaunchConfigRoutePreflight(unittest.TestCase):
    def test_aligned_tensor_core_plan_rejects_subwarp_route_tile(self):
        config = dataclasses.replace(
            LAUNCH.PROVISIONAL_LAUNCH_CONFIG,
            routing_block_size=8,
            gate_a_routing_block_size=8,
        )
        with self.assertRaisesRegex(ValueError, "at least 16"):
            config.validate_for_plan(PLAN.SERIAL_MATERIALIZED_REFERENCE)

    def test_distinct_gate_tile_rejects_non_grouped_gate_family(self):
        reference = PLAN.SERIAL_MATERIALIZED_REFERENCE
        token_dedup = dataclasses.replace(
            reference,
            gate_a=PLAN.LoraASpec(
                PLAN.FactorSite.GATE_UP,
                PLAN.LoraAFamily.TOKEN_DEDUP_GROUPED,
                True,
                PLAN.FactorLayout.TOKEN_MAJOR,
            ),
            gate_b=dataclasses.replace(
                reference.gate_b,
                input_layout=PLAN.FactorLayout.TOKEN_MAJOR,
            ),
        )
        config = dataclasses.replace(
            LAUNCH.PROVISIONAL_LAUNCH_CONFIG,
            gate_a_routing_block_size=64,
        )
        with self.assertRaisesRegex(ValueError, "grouped per-expert"):
            config.validate_for_plan(token_dedup)


if __name__ == "__main__":
    unittest.main()
