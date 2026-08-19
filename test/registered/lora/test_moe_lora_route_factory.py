"""Host-only guards for the Step-8 route-factory split-M candidate."""

from __future__ import annotations

import ast
import dataclasses
import importlib.util
import sys
import types
import unittest
from enum import Enum
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


def _serial_materialized_reference():
    """The simplest correct pipeline: every stage standalone, nothing fused,
    no overlap window.  Every plan below is a departure from this one."""
    return PLAN.MoeLoraExecutionPlan(
        gate_up_a=PLAN.LoraASpec(
            PLAN.Site.GATE_UP,
            PLAN.LoraAFamily.GROUPED,
            False,
            PLAN.BridgeLayout.PAIR_MAJOR,
        ),
        gate_up_b=PLAN.LoraBSpec(
            PLAN.Site.GATE_UP,
            PLAN.LoraBFamily.ONE_LAUNCH_SLICED,
            False,
            PLAN.BridgeLayout.PAIR_MAJOR,
        ),
        middle=PLAN.MiddleSpec(PLAN.MiddleFamily.MATERIALIZED, PLAN.ActivationFn.SILU),
        down_a=PLAN.LoraASpec(
            PLAN.Site.DOWN,
            PLAN.LoraAFamily.GROUPED,
            False,
            PLAN.BridgeLayout.PAIR_MAJOR,
        ),
        down_b=PLAN.LoraBSpec(
            PLAN.Site.DOWN,
            PLAN.LoraBFamily.ONE_LAUNCH_SLICED,
            False,
            PLAN.BridgeLayout.PAIR_MAJOR,
        ),
        finalize=PLAN.FinalizeSpec(PLAN.FinalizeFamily.MATERIALIZED),
    )


SERIAL_MATERIALIZED_REFERENCE = _serial_materialized_reference()


def _arch_pdl(enabled: bool):
    """Pin build_routes' architecture probe: route PDL is arch-keyed now."""
    arch = types.ModuleType("sglang.kernels.jit.utils")
    arch.is_arch_support_pdl = lambda: enabled
    return mock.patch.dict(sys.modules, {arch.__name__: arch})


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


class _RouteViewKind(str, Enum):
    """Stand-in for routing.RouteViewKind: these tests stub that module."""

    RAW = "raw"
    FUSED_IDS = "fused_ids"
    ALIGNED = "aligned"


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
    routing.RouteViewKind = _RouteViewKind
    routing.RouteView = _HostRouteView
    routing.FusedAlignScratch = types.SimpleNamespace
    routing.build_virtual_expert_routing = lambda *args, **kwargs: None
    routing.uses_fused_align = lambda *_args, **_kwargs: True

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
    routing.RouteViewKind = _RouteViewKind
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


class TestRoutePdlWiring(unittest.TestCase):
    def test_plans_build_exactly_their_aligned_routes(self):
        reference = SERIAL_MATERIALIZED_REFERENCE
        shared_down = dataclasses.replace(
            reference,
            down_b=dataclasses.replace(reference.down_b, is_shared_outer=True),
        )
        shared_token = dataclasses.replace(
            reference,
            gate_up_a=PLAN.LoraASpec(
                PLAN.Site.GATE_UP,
                PLAN.LoraAFamily.TOKEN_DEDUP_GROUPED,
                True,
                PLAN.BridgeLayout.TOKEN_MAJOR,
            ),
            gate_up_b=dataclasses.replace(
                reference.gate_up_b,
                input_layout=PLAN.BridgeLayout.TOKEN_MAJOR,
            ),
        )

        calls = []

        def fake_aligned(
            topk_ids,
            token_slots,
            *,
            is_shared_outer,
            num_local_experts,
            max_loras,
            block_size,
            workspace,
            scratch_prefix,
        ):
            calls.append(scratch_prefix)
            return _route(
                topk_ids,
                token_slots,
                block_size=block_size,
                padded_count=torch.tensor([block_size], dtype=torch.int32),
            )

        with mock.patch.object(
            ROUTE_FACTORY, "_aligned_pair_route", side_effect=fake_aligned
        ):
            for plan in (reference, shared_down, shared_token):
                ROUTE_FACTORY.build_routes(
                    plan,
                    topk_ids=torch.tensor([[0, 1]], dtype=torch.int32),
                    token_slots=torch.tensor([0], dtype=torch.int32),
                    num_local_experts=2,
                    max_loras=2,
                    block_size=16,
                    workspace=_Workspace(),
                )

        self.assertEqual(
            set(calls),
            {
                "route:aligned_per_expert",
                "route:aligned_shared_outer",
                "route:shared_token",
            },
        )

    def test_joint_plan_builds_joint_routes_plus_the_shared_token_follow_on(self):
        reference = SERIAL_MATERIALIZED_REFERENCE
        plan = dataclasses.replace(
            reference,
            gate_up_a=PLAN.LoraASpec(
                PLAN.Site.GATE_UP,
                PLAN.LoraAFamily.TOKEN_DEDUP_GROUPED,
                True,
                PLAN.BridgeLayout.TOKEN_MAJOR,
            ),
            gate_up_b=dataclasses.replace(
                reference.gate_up_b,
                input_layout=PLAN.BridgeLayout.TOKEN_MAJOR,
            ),
            down_b=dataclasses.replace(reference.down_b, is_shared_outer=True),
            route_builder=PLAN.RouteBuilderFamily.JOINT_SHARED_OUTER,
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
        ):
            joint_calls.append(block_size)
            route = _route(
                topk_ids,
                token_slots,
                block_size=block_size,
                padded_count=torch.tensor([block_size], dtype=torch.int32),
            )
            return route, route

        def fake_route(
            topk_ids,
            token_slots,
            *,
            lora_experts_per_adapter,
            max_loras,
            block_size,
            view,
            shared_outer_local_expert_count=None,
            lora_expert_map=None,
            num_pairs_post_padded_out=None,
            fused_align_scratch=None,
        ):
            standard_calls.append(view)
            num_pairs_post_padded_out.fill_(block_size)
            return _route(
                topk_ids,
                token_slots,
                block_size=block_size,
                padded_count=num_pairs_post_padded_out,
            )

        with (
            mock.patch.object(
                ROUTE_FACTORY, "build_joint_shared_routes", side_effect=fake_joint
            ),
            mock.patch.object(
                ROUTE_FACTORY, "build_virtual_expert_routing", side_effect=fake_route
            ),
        ):
            ROUTE_FACTORY.build_routes(
                plan,
                topk_ids=torch.tensor([[0, 1]], dtype=torch.int32),
                token_slots=torch.tensor([0], dtype=torch.int32),
                num_local_experts=2,
                max_loras=2,
                block_size=16,
                workspace=_Workspace(),
            )

        self.assertEqual(joint_calls, [16])
        self.assertEqual(standard_calls, ["aligned"])

    def _run_joint(self, *, use_pdl):
        recorders = [_KernelRecorder() for _ in range(3)]
        with (
            _arch_pdl(bool(use_pdl)),
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

    def test_joint_route_pdl_off_leaves_launches_unarmed(self):
        _, recorders = self._run_joint(use_pdl=False)
        for recorder in recorders:
            self.assertFalse(recorder.calls[0][2]["USE_PDL"])
            self.assertNotIn("launch_pdl", recorder.calls[0][2])

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


class TestSharedTokenRoute(unittest.TestCase):
    def test_token_dedup_skips_tokens_without_a_local_pair(self):
        reference = SERIAL_MATERIALIZED_REFERENCE
        shared_plan = dataclasses.replace(
            reference,
            gate_up_a=PLAN.LoraASpec(
                PLAN.Site.GATE_UP,
                PLAN.LoraAFamily.TOKEN_DEDUP_GROUPED,
                True,
                PLAN.BridgeLayout.TOKEN_MAJOR,
            ),
            gate_up_b=dataclasses.replace(
                reference.gate_up_b,
                input_layout=PLAN.BridgeLayout.TOKEN_MAJOR,
            ),
        )
        topk_ids = torch.tensor([[-1, -1], [-1, 1], [0, -1]], dtype=torch.int32)
        token_slots = torch.tensor([0, 1, 0], dtype=torch.int32)
        workspace = _Workspace()
        calls = []

        def fake_route(
            route_topk_ids,
            route_token_slots,
            *,
            lora_experts_per_adapter,
            max_loras,
            block_size,
            view,
            shared_outer_local_expert_count=None,
            lora_expert_map=None,
            num_pairs_post_padded_out=None,
            fused_align_scratch=None,
        ):
            calls.append(
                (
                    shared_outer_local_expert_count is not None,
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

        with (
            _arch_pdl(False),
            mock.patch.object(
                ROUTE_FACTORY, "build_virtual_expert_routing", side_effect=fake_route
            ),
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
        reference = SERIAL_MATERIALIZED_REFERENCE
        shared_plan = dataclasses.replace(
            reference,
            gate_up_a=PLAN.LoraASpec(
                PLAN.Site.GATE_UP,
                PLAN.LoraAFamily.TOKEN_DEDUP_GROUPED,
                True,
                PLAN.BridgeLayout.TOKEN_MAJOR,
            ),
            gate_up_b=dataclasses.replace(
                reference.gate_up_b,
                input_layout=PLAN.BridgeLayout.TOKEN_MAJOR,
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

                def fake_route(
                    route_topk_ids,
                    route_token_slots,
                    *,
                    lora_experts_per_adapter,
                    max_loras,
                    block_size,
                    view,
                    shared_outer_local_expert_count=None,
                    lora_expert_map=None,
                    num_pairs_post_padded_out=None,
                    fused_align_scratch=None,
                ):
                    self.assertEqual(view, "aligned")
                    num_virtual = lora_experts_per_adapter * max_loras
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
                        lora_experts_per_adapter=lora_experts_per_adapter,
                        max_loras=max_loras,
                        shared_outer_local_expert_count=shared_outer_local_expert_count,
                    )

                with (
                    _arch_pdl(False),
                    mock.patch.object(
                        ROUTE_FACTORY,
                        "build_virtual_expert_routing",
                        side_effect=fake_route,
                    ),
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


class TestLaunchConfigRoutePreflight(unittest.TestCase):
    def test_aligned_tensor_core_plan_rejects_subwarp_route_tile(self):
        config = LAUNCH.MoeLoraLaunchConfig(routing_block_size=8)
        with self.assertRaisesRegex(ValueError, "at least 16"):
            config.validate_for_plan(SERIAL_MATERIALIZED_REFERENCE)


if __name__ == "__main__":
    unittest.main()
