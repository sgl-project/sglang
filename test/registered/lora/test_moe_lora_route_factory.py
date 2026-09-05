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


def _segments_of(mapping: torch.Tensor) -> torch.Tensor:
    """Request boundaries for a test batch: one request per token, so the
    segment route is exercised with many short requests and the helper stays
    free of host syncs (CUDA-graph capture forbids a device-to-host copy)."""
    return torch.arange(mapping.numel() + 1, dtype=torch.int32, device=mapping.device)


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
            PLAN.LoraBFamily.GROUPED,
            False,
            PLAN.BridgeLayout.PAIR_MAJOR,
        ),
        act=PLAN.ActSpec(PLAN.ActFamily.MATERIALIZED, PLAN.ActivationFn.SILU),
        down_a=PLAN.LoraASpec(
            PLAN.Site.DOWN,
            PLAN.LoraAFamily.GROUPED,
            False,
            PLAN.BridgeLayout.PAIR_MAJOR,
        ),
        down_b=PLAN.LoraBSpec(
            PLAN.Site.DOWN,
            PLAN.LoraBFamily.GROUPED,
            False,
            PLAN.BridgeLayout.PAIR_MAJOR,
        ),
        finalize=PLAN.FinalizeSpec(PLAN.FinalizeFamily.MATERIALIZED),
    )


SERIAL_MATERIALIZED_REFERENCE = _serial_materialized_reference()


def _fake_segment_route(calls, prefixes=None):
    """Stand in for build_routes' segment token-route builder: record it the
    way the aligned-route fakes record theirs and hand back a host route."""

    def build(
        *,
        seg_indptr,
        token_lora_mapping,
        num_tokens,
        num_local_experts,
        max_loras,
        block_size,
        workspace,
    ):
        assert seg_indptr[-1].item() == num_tokens
        calls.append(("route:shared_token", True))
        if prefixes is not None:
            prefixes.append("route:shared_token")
        # Its padded count lives under its own workspace name, like the
        # aligned routes' do, and reads the token count back.
        padded = workspace.tensor(
            "route:shared_token:padded_pairs",
            (1,),
            dtype=torch.int32,
            device=token_lora_mapping.device,
        )
        padded.fill_(num_tokens)
        return _route(
            torch.zeros((num_tokens, 1), dtype=torch.int32),
            token_lora_mapping,
            block_size=block_size,
            padded_count=padded,
            num_local_experts=num_local_experts,
            max_loras=max_loras,
            is_shared_outer=True,
        )

    return build


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
    finalize = types.ModuleType("sglang.srt.lora.moe.kernels.finalize")
    finalize.SHARED_RANK_DEFAULT_CONFIG = {
        "reduce": {"BLOCK_SIZE_T": 16},
        "tail": {"BLOCK_SIZE_H": 16},
    }
    act = types.ModuleType("sglang.srt.lora.moe.kernels.fused_act")
    act.FUSED_B_ACT_DEFAULT_CONFIG = {"BLOCK_SIZE_W": 16}
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
            act.__name__: act,
            module_name: module,
        },
    ):
        spec.loader.exec_module(module)
    return module


LAUNCH = _load_launch_config()


class _RouteViewKind(str, Enum):
    """Stand-in for routing.RouteViewKind: these tests stub that module."""

    RAW = "raw"
    ALIGNED = "aligned"


class _HostRouteView(msgspec.Struct, frozen=True, kw_only=True):
    view: str
    block_size: int
    topk_ids: torch.Tensor
    token_lora_mapping: torch.Tensor
    num_local_experts: int
    max_loras: int
    is_shared_outer: bool = False
    maybe_sorted_pair_ids: torch.Tensor | None = None
    maybe_block_virtual_expert_ids: torch.Tensor | None = None
    maybe_num_pairs_post_padded: torch.Tensor | None = None


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

    route_view = types.ModuleType("sglang.srt.lora.moe.route_view")
    route_view.RouteViewKind = _RouteViewKind
    route_view.RouteView = _HostRouteView

    class _Unlaunched:
        """Every kernel a host-level test reaches has to be patched first."""

        def __getitem__(self, _grid):
            def launch(*_args, **_kwargs):
                raise AssertionError("a kernel launched in the host sandbox")

            return launch

    route_kernels = types.ModuleType("sglang.srt.lora.moe.kernels.routing")
    for kernel in (
        "_build_virtual_topk_ids_kernel",
        "_hist_kernel",
        "_place_kernel",
        "_scan_kernel",
        "_segment_token_route_kernel",
    ):
        setattr(route_kernels, kernel, _Unlaunched())

    # routing.py imports this for the fused builder's metadata; stub it so the
    # loader does not depend on another test module having imported the real
    # one first.
    workspace = types.ModuleType("sglang.srt.lora.moe.workspace")
    workspace.MoeLoraWorkspace = object

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
            "sglang.srt.lora.moe.execution_plan": PLAN,
            virtual_experts.__name__: virtual_experts,
            route_view.__name__: route_view,
            route_kernels.__name__: route_kernels,
            workspace.__name__: workspace,
            module_name: module,
        },
    ):
        spec.loader.exec_module(module)
    return module


ROUTING = _load_routing()


class _Workspace:
    def __init__(self):
        self.tensors: dict[str, torch.Tensor] = {}

    def run_parallel(self, *, name, device, compute, side):
        # The real CPU fallback: side first, then compute, fully joined.
        side()
        return compute()

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
    token_lora_mapping: torch.Tensor,
    *,
    block_size: int,
    padded_count: torch.Tensor,
    num_local_experts: int = 2,
    max_loras: int = 2,
    is_shared_outer: bool = False,
) -> _HostRouteView:
    return _HostRouteView(
        view="aligned",
        block_size=block_size,
        topk_ids=topk_ids,
        token_lora_mapping=token_lora_mapping,
        num_local_experts=num_local_experts,
        max_loras=max_loras,
        is_shared_outer=is_shared_outer,
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
                PLAN.LoraAFamily.TOKEN_GROUPED,
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
            token_lora_mapping,
            *,
            num_local_experts,
            max_loras,
            block_size,
            view,
            is_shared_outer=False,
            workspace=None,
            tensor_prefix=None,
        ):
            # Every route gets the SAME local expert count; only the flag
            # differs, and it has to match the route being built.
            self.assertEqual(num_local_experts, 2)
            calls.append((tensor_prefix, is_shared_outer))
            built = _route(
                topk_ids,
                token_lora_mapping,
                block_size=block_size,
                padded_count=torch.tensor([block_size], dtype=torch.int32),
            )
            return built

        with (
            mock.patch.object(
                ROUTING, "build_virtual_expert_routing", side_effect=fake_aligned
            ),
            mock.patch.object(
                ROUTING,
                "_build_segment_token_route",
                side_effect=_fake_segment_route(calls),
            ),
        ):
            for plan in (reference, shared_down, shared_token):
                ROUTING.build_routes(
                    plan,
                    topk_ids=torch.tensor([[0, 1]], dtype=torch.int32),
                    token_lora_mapping=torch.tensor([0], dtype=torch.int32),
                    seg_indptr=_segments_of(torch.tensor([0], dtype=torch.int32)),
                    num_local_experts=2,
                    max_loras=2,
                    block_size=16,
                    workspace=_Workspace(),
                )

        self.assertEqual(
            set(calls),
            {
                ("route:aligned_per_expert", False),
                ("route:aligned_shared_outer", True),
                ("route:shared_token", True),
            },
        )

    def test_parallel_plan_builds_both_routes_plus_the_shared_token_follow_on(self):
        reference = SERIAL_MATERIALIZED_REFERENCE
        plan = dataclasses.replace(
            reference,
            gate_up_a=PLAN.LoraASpec(
                PLAN.Site.GATE_UP,
                PLAN.LoraAFamily.TOKEN_GROUPED,
                True,
                PLAN.BridgeLayout.TOKEN_MAJOR,
            ),
            gate_up_b=dataclasses.replace(
                reference.gate_up_b,
                input_layout=PLAN.BridgeLayout.TOKEN_MAJOR,
            ),
            down_b=dataclasses.replace(reference.down_b, is_shared_outer=True),
            route_builder=PLAN.RouteBuilderFamily.PARALLEL_SHARED_OUTER,
        )
        calls = []

        def fake_route(
            topk_ids,
            token_lora_mapping,
            *,
            num_local_experts,
            max_loras,
            block_size,
            view=None,
            is_shared_outer=False,
            workspace=None,
            tensor_prefix=None,
        ):
            padded = torch.zeros(1, dtype=torch.int32)
            padded.fill_(block_size)
            built = _route(
                topk_ids,
                token_lora_mapping,
                block_size=block_size,
                padded_count=padded,
            )
            calls.append((tensor_prefix, is_shared_outer))
            return built

        with (
            mock.patch.object(
                ROUTING, "build_virtual_expert_routing", side_effect=fake_route
            ),
            mock.patch.object(
                ROUTING,
                "_build_segment_token_route",
                side_effect=_fake_segment_route(calls),
            ),
        ):
            ROUTING.build_routes(
                plan,
                topk_ids=torch.tensor([[0, 1]], dtype=torch.int32),
                token_lora_mapping=torch.tensor([0], dtype=torch.int32),
                seg_indptr=_segments_of(torch.tensor([0], dtype=torch.int32)),
                num_local_experts=2,
                max_loras=2,
                block_size=16,
                workspace=_Workspace(),
            )

        # The fork builds both aligned routes; the shared-token follow-on is a
        # third standard call.
        self.assertEqual(
            set(calls),
            {
                ("route:aligned_per_expert", False),
                ("route:aligned_shared_outer", True),
                ("route:shared_token", True),
            },
        )

    def _run_route(self, *, use_pdl, is_shared_outer=False):
        recorders = [_KernelRecorder() for _ in range(3)]
        with (
            _arch_pdl(bool(use_pdl)),
            mock.patch.object(ROUTING, "_hist_kernel", recorders[0]),
            mock.patch.object(ROUTING, "_scan_kernel", recorders[1]),
            mock.patch.object(ROUTING, "_place_kernel", recorders[2]),
        ):
            route = ROUTING._build_aligned(
                torch.tensor([[0, 1]], dtype=torch.int32),
                torch.tensor([0], dtype=torch.int32),
                num_local_experts=2,
                max_loras=2,
                block_size=16,
                workspace=_Workspace(),
                tensor_prefix="test:route",
                is_shared_outer=is_shared_outer,
            )
        return route, recorders

    def test_route_launches_real_pdl_chain(self):
        route, (hist, scan, expand) = self._run_route(use_pdl=True)

        self.assertIsNotNone(route)
        self.assertTrue(hist.calls[0][2]["USE_PDL"])
        self.assertNotIn("launch_pdl", hist.calls[0][2])
        for consumer in (scan, expand):
            self.assertTrue(consumer.calls[0][2]["USE_PDL"])
            self.assertTrue(consumer.calls[0][2]["launch_pdl"])

    def test_route_pdl_off_leaves_launches_unarmed(self):
        _, recorders = self._run_route(use_pdl=False)
        for recorder in recorders:
            self.assertFalse(recorder.calls[0][2]["USE_PDL"])
            self.assertNotIn("launch_pdl", recorder.calls[0][2])

    def test_kernel_dependency_operations_are_complete(self):
        source = (LORA_MOE / "kernels" / "routing.py").read_text()
        tree = ast.parse(source)
        function_nodes = {
            node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
        }
        functions = {name: ast.unparse(node) for name, node in function_nodes.items()}

        hist = functions["_hist_kernel"]
        scan = functions["_scan_kernel"]
        expand = functions["_place_kernel"]
        self.assertIn("gdc_launch_dependents", hist)
        self.assertNotIn("gdc_wait", hist)
        self.assertIn("gdc_wait", scan)
        self.assertIn("gdc_launch_dependents", scan)
        self.assertIn("gdc_wait", expand)
        self.assertNotIn("gdc_launch_dependents", expand)

        # K3 must be released before the scan body, so the place kernel's pair
        # path can overlap scan work while still waiting before cursor
        # consumption.
        scan_ifs = [
            node
            for node in function_nodes["_scan_kernel"].body
            if isinstance(node, ast.If)
        ]
        self.assertEqual(len(scan_ifs), 1)
        self.assertEqual(ast.unparse(scan_ifs[0].test), "USE_PDL")
        pdl_body = ast.unparse(scan_ifs[0])
        self.assertLess(
            pdl_body.index("gdc_wait"),
            pdl_body.index("gdc_launch_dependents"),
        )
        self.assertIn("tl.program_id", expand)


class TestSharedTokenRoute(unittest.TestCase):
    def test_token_route_is_built_from_the_request_segments(self):
        # The token route needs no histogram or sort: a request's tokens are
        # contiguous and share one slot, so build_routes hands the request
        # boundaries and the token mapping, unmasked, to the segment builder.
        reference = SERIAL_MATERIALIZED_REFERENCE
        shared_plan = dataclasses.replace(
            reference,
            gate_up_a=PLAN.LoraASpec(
                PLAN.Site.GATE_UP,
                PLAN.LoraAFamily.TOKEN_GROUPED,
                True,
                PLAN.BridgeLayout.TOKEN_MAJOR,
            ),
            gate_up_b=dataclasses.replace(
                reference.gate_up_b,
                input_layout=PLAN.BridgeLayout.TOKEN_MAJOR,
            ),
        )
        topk_ids = torch.tensor([[-1, -1], [-1, 1], [0, -1]], dtype=torch.int32)
        token_lora_mapping = torch.tensor([0, 1, 0], dtype=torch.int32)
        seg_indptr = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        workspace = _Workspace()
        seen = {}

        def fake_segment(**kwargs):
            seen.update(kwargs)
            return _route(
                torch.zeros((3, 1), dtype=torch.int32),
                kwargs["token_lora_mapping"],
                block_size=kwargs["block_size"],
                padded_count=torch.tensor([16], dtype=torch.int32),
                num_local_experts=kwargs["num_local_experts"],
                max_loras=kwargs["max_loras"],
                is_shared_outer=True,
            )

        def fake_aligned(
            topk_ids, token_lora_mapping, *, is_shared_outer=False, **kwargs
        ):
            built = _route(
                topk_ids,
                token_lora_mapping,
                block_size=kwargs["block_size"],
                padded_count=torch.tensor([16], dtype=torch.int32),
            )
            return built

        with (
            _arch_pdl(False),
            mock.patch.object(
                ROUTING, "build_virtual_expert_routing", side_effect=fake_aligned
            ),
            mock.patch.object(
                ROUTING, "_build_segment_token_route", side_effect=fake_segment
            ),
        ):
            routes = ROUTING.build_routes(
                shared_plan,
                topk_ids=topk_ids,
                token_lora_mapping=token_lora_mapping,
                seg_indptr=seg_indptr,
                num_local_experts=2,
                max_loras=2,
                block_size=16,
                workspace=workspace,
            )
        self.assertIs(seen["seg_indptr"], seg_indptr)
        self.assertIs(seen["token_lora_mapping"], token_lora_mapping)
        self.assertEqual(seen["num_tokens"], 3)
        self.assertIsNotNone(routes.shared_token)

    def test_large_shared_token_route_cannot_overwrite_retained_pair_counts(self):
        """Every retained fused route gets its OWN workspace keys, even when the
        per-expert and shared-outer bucket counts collide (num_local_experts=1).

        The hazard is the fused builder's process-global scratch cache, which is
        keyed by bucket count alone: two routes at the same V would share one
        scalar, and the T-row shared-token plan would clobber the T*K-row pair
        plan. build_virtual_expert_routing allocates per ``tensor_prefix``
        instead, so what this has to pin is that build_routes hands every route
        a DISTINCT prefix -- storage isolation per name is the workspace's own
        contract.
        """
        reference = SERIAL_MATERIALIZED_REFERENCE
        shared_plan = dataclasses.replace(
            reference,
            gate_up_a=PLAN.LoraASpec(
                PLAN.Site.GATE_UP,
                PLAN.LoraAFamily.TOKEN_GROUPED,
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
        token_lora_mapping = torch.zeros(num_tokens, dtype=torch.int32)

        for num_local_experts in (1, 2):
            with self.subTest(num_local_experts=num_local_experts):
                workspace = _Workspace()
                prefixes = []

                def fake_route(
                    route_topk_ids,
                    route_token_lora_mapping,
                    *,
                    num_local_experts,
                    max_loras,
                    block_size,
                    view,
                    is_shared_outer=False,
                    workspace=None,
                    tensor_prefix=None,
                ):
                    self.assertEqual(view, "aligned")
                    self.assertIsNotNone(workspace)
                    prefixes.append(tensor_prefix)
                    # Model the real allocation: one scalar per route, named.
                    padded = workspace.tensor(
                        f"{tensor_prefix}:padded_pairs",
                        (1,),
                        dtype=torch.int32,
                        device=route_topk_ids.device,
                    )
                    padded.fill_(route_topk_ids.numel())
                    built = _route(
                        route_topk_ids,
                        route_token_lora_mapping,
                        block_size=block_size,
                        padded_count=padded,
                        num_local_experts=num_local_experts,
                        max_loras=max_loras,
                        is_shared_outer=is_shared_outer,
                    )
                    return built

                with (
                    mock.patch.object(
                        ROUTING,
                        "build_virtual_expert_routing",
                        side_effect=fake_route,
                    ),
                    mock.patch.object(
                        ROUTING,
                        "_build_segment_token_route",
                        side_effect=_fake_segment_route([], prefixes),
                    ),
                ):
                    routes = ROUTING.build_routes(
                        shared_plan,
                        topk_ids=topk_ids,
                        token_lora_mapping=token_lora_mapping,
                        seg_indptr=_segments_of(token_lora_mapping),
                        num_local_experts=num_local_experts,
                        max_loras=2,
                        block_size=16,
                        workspace=workspace,
                    )

                # Distinct prefix per route is the whole guarantee.
                self.assertEqual(
                    set(prefixes),
                    {
                        "route:aligned_per_expert",
                        "route:aligned_shared_outer",
                        "route:shared_token",
                    },
                )
                self.assertEqual(len(prefixes), len(set(prefixes)))

                pair_count = num_tokens * top_k
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
                    num_tokens,
                )
                # ... and the scalars are separate storage, so the T-row plan
                # cannot have overwritten either T*K-row one.
                pointers = {
                    route.maybe_num_pairs_post_padded.data_ptr()
                    for route in (
                        routes.aligned_per_expert,
                        routes.aligned_shared_outer,
                        routes.shared_token,
                    )
                }
                self.assertEqual(len(pointers), 3)


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
    def test_subwarp_route_tile_is_rejected_at_construction(self):
        # The grouped LoRA kernels take this value as their tl.dot row tile,
        # so the config constructor itself rejects anything below 16.
        with self.assertRaisesRegex(ValueError, ">= 16"):
            LAUNCH.MoeLoraLaunchConfig(routing_block_size=8)


if __name__ == "__main__":
    unittest.main()
