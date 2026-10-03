"""Host contracts for route construction, dispatch, and workspace isolation."""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[4]
LORA_MOE = ROOT / "python/sglang/srt/lora/moe"
ROUTE_KERNELS = ROOT / "python/sglang/srt/lora/kernels/routing.py"


def _load_plan():
    module_name = "_route_factory_plan"
    spec = importlib.util.spec_from_file_location(module_name, LORA_MOE / "plan.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {module_name: module}):
        spec.loader.exec_module(module)
    return module


PLAN = _load_plan()


def _serial_materialized_reference():
    """Use standalone stages with no fusion or overlap."""
    return PLAN.MoePlan(
        gate_up_a=PLAN.ASpec(
            PLAN.Site.GATE_UP,
            PLAN.AFamily.GROUPED,
            False,
            PLAN.BridgeLayout.PAIR_MAJOR,
        ),
        gate_up_b=PLAN.BSpec(
            PLAN.Site.GATE_UP,
            PLAN.BFamily.GROUPED,
            False,
            PLAN.BridgeLayout.PAIR_MAJOR,
        ),
        act=PLAN.ActSpec(PLAN.ActFamily.MATERIALIZED, PLAN.ActivationFn.SILU),
        down_a=PLAN.ASpec(
            PLAN.Site.DOWN,
            PLAN.AFamily.GROUPED,
            False,
            PLAN.BridgeLayout.PAIR_MAJOR,
        ),
        down_b=PLAN.BSpec(
            PLAN.Site.DOWN,
            PLAN.BFamily.GROUPED,
            False,
            PLAN.BridgeLayout.PAIR_MAJOR,
        ),
        finalize=PLAN.FinalizeSpec(PLAN.FinalizeFamily.MATERIALIZED),
    )


SERIAL_MATERIALIZED_REFERENCE = _serial_materialized_reference()


def _arch_pdl(enabled: bool):
    """Pin build_moe_routes' architecture probe: route PDL is arch-keyed now."""
    arch = types.ModuleType("sglang.kernels.jit.utils")
    arch.is_arch_support_pdl = lambda: enabled
    return mock.patch.dict(sys.modules, {arch.__name__: arch})


def _load_route_view():
    module_name = "_route_factory_route_view"
    spec = importlib.util.spec_from_file_location(
        module_name, ROOT / "python/sglang/srt/lora/route_view.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {module_name: module}):
        spec.loader.exec_module(module)
    return module


ROUTE_VIEW = _load_route_view()
RouteView = ROUTE_VIEW.RouteView
RouteViewKind = ROUTE_VIEW.RouteViewKind


def _load_routing():
    package_names = (
        "sglang",
        "sglang.kernels",
        "sglang.kernels.ops",
        "sglang.kernels.ops.moe",
        "sglang.srt",
        "sglang.srt.lora",
        "sglang.srt.lora.kernels",
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
    fake_triton.next_power_of_2 = lambda value: 1 << (value - 1).bit_length()
    fake_tl = types.ModuleType("triton.language")
    fake_triton.language = fake_tl

    virtual_experts = types.ModuleType("sglang.kernels.ops.moe.virtual_experts")
    virtual_experts._align_block_size_jit = lambda *_args, **_kwargs: None

    class _Unlaunched:
        """Every kernel a host-level test reaches has to be patched first."""

        def __getitem__(self, _grid):
            def launch(*_args, **_kwargs):
                raise AssertionError("a kernel launched in the host sandbox")

            return launch

    # Isolate the fused builder's metadata import from test collection order.
    workspace = types.ModuleType("sglang.srt.lora.workspace")
    workspace.LoraWorkspace = object

    # Load the real shared routing module before the MoE bundle,
    # with both module aliases confined to the sandbox.
    shared_spec = importlib.util.spec_from_file_location(
        "sglang.srt.lora.kernels.routing", ROUTE_KERNELS
    )
    assert shared_spec is not None and shared_spec.loader is not None
    shared = importlib.util.module_from_spec(shared_spec)
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
            "sglang.srt.lora.moe.plan": PLAN,
            virtual_experts.__name__: virtual_experts,
            "sglang.srt.lora.route_view": ROUTE_VIEW,
            workspace.__name__: workspace,
            shared.__name__: shared,
            module_name: module,
        },
    ):
        shared_spec.loader.exec_module(shared)
        spec.loader.exec_module(module)
    # The fake triton.jit leaves the route kernels plain functions.
    for kernel in (
        "_build_route_bucket_ids_kernel",
        "_build_small_route_kernel",
        "_route_histogram_kernel",
        "_route_place_kernel",
        "_route_scan_kernel",
    ):
        setattr(shared, kernel, _Unlaunched())
    return module, shared


ROUTING, SHARED_ROUTING = _load_routing()


class _Workspace:
    def __init__(self):
        self.tensors: dict[str, torch.Tensor] = {}

    def route(self, mapping, key, build):
        return build()

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
    token_slots: torch.Tensor,
    *,
    group_ids: torch.Tensor | None,
    block_size: int,
    padded_count: torch.Tensor,
    groups_per_slot: int = 2,
    max_loras: int = 2,
) -> RouteView:
    return RouteView(
        view=RouteViewKind.ALIGNED,
        block_size=block_size,
        token_slots=token_slots,
        group_ids=group_ids,
        groups_per_slot=groups_per_slot,
        max_loras=max_loras,
        maybe_sorted_pair_ids=torch.arange(2, dtype=torch.int32),
        maybe_block_bucket_ids=torch.arange(1, dtype=torch.int32),
        maybe_num_pairs_post_padded=padded_count,
    )


class TestRouteViewContract(CustomTestCase):
    def test_metadata_keeps_groups_separate_from_combined_buckets(self):
        token_slots = torch.tensor([0, 1], dtype=torch.int32)
        groups = torch.tensor([[0, 2, 1], [2, -1, 0]], dtype=torch.int32)
        for group_ids, groups_per_slot, rows, width in (
            (None, 1, 2, 1),
            (groups, 3, 6, 3),
            (groups, 1, 6, 3),
        ):
            with self.subTest(groups_per_slot=groups_per_slot, width=width):
                route = SHARED_ROUTING.build_route(
                    token_slots,
                    group_ids=group_ids,
                    groups_per_slot=groups_per_slot,
                    max_loras=2,
                    block_size=16,
                    view=RouteViewKind.RAW,
                )
                self.assertIsInstance(route, RouteView)
                self.assertIs(route.view, RouteViewKind.RAW)
                self.assertIs(route.token_slots, token_slots)
                self.assertIs(route.group_ids, group_ids)
                self.assertIs(
                    route.kernel_groups,
                    token_slots if group_ids is None else group_ids,
                )
                self.assertEqual(route.groups_per_slot, groups_per_slot)
                self.assertEqual(route.num_tokens, 2)
                self.assertEqual(route.num_rows, rows)
                self.assertEqual(route.width, width)
                # RouteView counts only route buckets, not the sentinel.
                self.assertEqual(route.num_buckets, 2 * groups_per_slot)
                for field in (
                    "sorted_pair_ids",
                    "block_bucket_ids",
                    "num_pairs_post_padded",
                ):
                    with self.assertRaisesRegex(
                        ValueError,
                        f"route view 'raw' did not build {field};.*'aligned'",
                    ):
                        getattr(route, field)

    def test_aligned_properties_return_the_built_tensors(self):
        route = _route(
            torch.tensor([1], dtype=torch.int32),
            group_ids=torch.tensor([[0, 1]], dtype=torch.int32),
            block_size=16,
            padded_count=torch.tensor([16], dtype=torch.int32),
        )
        self.assertIs(route.view, RouteViewKind.ALIGNED)
        self.assertEqual(route.num_buckets, 4)
        self.assertIs(route.sorted_pair_ids, route.maybe_sorted_pair_ids)
        self.assertIs(route.block_bucket_ids, route.maybe_block_bucket_ids)
        self.assertIs(route.num_pairs_post_padded, route.maybe_num_pairs_post_padded)

    def test_invalid_view_or_missing_groups_is_rejected_before_launch(self):
        for kwargs, message in (
            ({"view": "unknown"}, "unknown route view"),
            ({"groups_per_slot": 2}, "needs group_ids"),
        ):
            with (
                self.subTest(kwargs=kwargs),
                self.assertRaisesRegex(ValueError, message),
            ):
                SHARED_ROUTING.build_route(
                    torch.tensor([0], dtype=torch.int32),
                    max_loras=2,
                    block_size=16,
                    **kwargs,
                )


class TestRouteBucketHelper(CustomTestCase):
    def test_bucket_ids_preserve_liveness_and_shared_group_folding(self):
        # Interpret only this elementwise helper with CPU tensors. This is not
        # a Triton compilation or GPU-kernel correctness test.
        class Pointer:
            def __init__(self, tensor):
                self.tensor = tensor.flatten()

            def __add__(self, offsets):
                return self.tensor, offsets

        def load(pointer, *, mask, other):
            tensor, offsets = pointer
            result = torch.full_like(offsets, other, dtype=tensor.dtype)
            result[mask] = tensor[offsets[mask]]
            return result

        slots = torch.tensor([-1, 0, 1, 2], dtype=torch.int32)
        groups = torch.tensor(
            [[0, 1, 2], [2, -1, 3], [0, 2, 1], [0, 1, 2]], dtype=torch.int32
        )
        for group_ids, groups_per_slot, expected in (
            (None, 1, [-1, 0, 1, -1]),
            (groups, 3, [-1, -1, -1, 2, -1, -1, 3, 5, 4, -1, -1, -1]),
            (groups, 1, [-1, -1, -1, 0, -1, 0, 1, 1, 1, -1, -1, -1]),
        ):
            with self.subTest(
                groups_per_slot=groups_per_slot, has_groups=group_ids is not None
            ):
                rows = len(expected)
                pair_ids = torch.arange(rows + 2)
                with mock.patch.multiple(
                    SHARED_ROUTING.tl,
                    load=load,
                    where=torch.where,
                    int32=torch.int32,
                    create=True,
                ):
                    bucket_ids = SHARED_ROUTING.route_bucket_ids(
                        Pointer(slots if group_ids is None else group_ids),
                        Pointer(slots),
                        pair_ids,
                        pair_ids < rows,
                        GROUPS_PER_SLOT=groups_per_slot,
                        MAX_LORAS=2,
                        WIDTH=1 if group_ids is None else group_ids.shape[1],
                        HAS_GROUPS=group_ids is not None,
                    )
                # Tail lanes must also be dead without reading beyond inputs.
                self.assertEqual(bucket_ids.tolist(), expected + [-1, -1])

    def test_bucket_builder_skips_empty_output_and_passes_group_metadata(self):
        for rows in (0, 2):
            with self.subTest(rows=rows):
                recorder = _KernelRecorder()
                slots = torch.zeros(rows, dtype=torch.int32)
                groups = torch.zeros((rows, 3), dtype=torch.int32)
                out = torch.empty_like(groups)
                with mock.patch.object(
                    SHARED_ROUTING, "_build_route_bucket_ids_kernel", recorder
                ):
                    result = SHARED_ROUTING._build_route_bucket_ids(
                        slots, groups, groups_per_slot=3, max_loras=2, out=out
                    )
                self.assertIs(result, out)
                self.assertEqual(len(recorder.calls), int(rows > 0))
                if rows:
                    grid, args, kwargs = recorder.calls[0]
                    self.assertEqual(grid, (1,))
                    self.assertIs(args[0], groups)
                    self.assertIs(args[1], slots)
                    self.assertIs(args[2], out)
                    self.assertEqual(args[3], 6)
                    self.assertEqual(kwargs["GROUPS_PER_SLOT"], 3)
                    self.assertEqual(kwargs["MAX_LORAS"], 2)
                    self.assertEqual(kwargs["WIDTH"], 3)
                    self.assertTrue(kwargs["HAS_GROUPS"])


class TestRouteDispatchContract(CustomTestCase):
    def test_dispatch_boundaries_and_sentinel_counts(self):
        cases = (
            (0, 2, None, "jit"),
            (1, 2, None, "small"),
            (512, 2, None, "small"),
            (513, 2, None, "small"),
            (768, 2, None, "small"),
            (769, 2, None, "jit"),
            (16383, 2, None, "jit"),
            (16384, 2, None, "aligned"),
            (1, 8191, None, "small"),
            (1, 8192, None, "aligned"),
            (1, 2, 0, "aligned"),
            (1, 2, 64, "aligned"),
        )
        for rows, num_route_buckets, capacity, expected in cases:
            with self.subTest(rows=rows, buckets=num_route_buckets, capacity=capacity):
                recorders = {
                    name: _KernelRecorder()
                    for name in (
                        "_build_route_bucket_ids_kernel",
                        "_build_small_route_kernel",
                        "_route_histogram_kernel",
                        "_route_scan_kernel",
                        "_route_place_kernel",
                    )
                }
                aligned_tensors = (
                    torch.empty(0, dtype=torch.int32),
                    torch.empty(0, dtype=torch.int32),
                    torch.zeros(1, dtype=torch.int32),
                )
                workspace = _Workspace()
                with (
                    _arch_pdl(False),
                    mock.patch.multiple(SHARED_ROUTING, **recorders),
                    mock.patch.object(
                        SHARED_ROUTING,
                        "_align_block_size_jit",
                        return_value=aligned_tensors,
                    ) as jit_align,
                ):
                    route = SHARED_ROUTING.build_route(
                        torch.zeros(rows, dtype=torch.int32),
                        max_loras=num_route_buckets,
                        block_size=16,
                        workspace=workspace,
                        tensor_prefix="test:dispatch",
                        capacity=capacity,
                    )
                self.assertIsInstance(route, RouteView)
                self.assertEqual(route.num_rows, rows)
                self.assertEqual(route.num_buckets, num_route_buckets)
                self.assertEqual(jit_align.call_count, int(expected == "jit"))
                for name, recorder in recorders.items():
                    launched = (
                        expected == "small"
                        if name == "_build_small_route_kernel"
                        else (
                            expected == "jit" and rows > 0
                            if name == "_build_route_bucket_ids_kernel"
                            else expected == "aligned"
                        )
                    )
                    self.assertEqual(len(recorder.calls), int(launched), name)
                if expected == "jit":
                    args, _ = jit_align.call_args
                    self.assertIs(
                        args[0], workspace.tensors["test:dispatch:jit:bucket_ids"]
                    )
                    self.assertEqual(args[1:], (16, num_route_buckets))
                    self.assertIs(route.block_bucket_ids, aligned_tensors[1])
                elif expected == "small":
                    args = recorders["_build_small_route_kernel"].calls[0][1]
                    self.assertEqual(args[7], num_route_buckets)
                    self.assertIs(route.block_bucket_ids, args[3])
                else:
                    hist_kwargs = recorders["_route_histogram_kernel"].calls[0][2]
                    place_args = recorders["_route_place_kernel"].calls[0][1]
                    place_kwargs = recorders["_route_place_kernel"].calls[0][2]
                    self.assertEqual(hist_kwargs["NUM_BUCKETS"], num_route_buckets + 1)
                    self.assertEqual(place_kwargs["NUM_BUCKETS"], num_route_buckets + 1)
                    self.assertEqual(
                        place_kwargs["NUM_ROUTE_BUCKETS"], num_route_buckets
                    )
                    self.assertIs(route.block_bucket_ids, place_args[6])
                    self.assertGreaterEqual(
                        route.sorted_pair_ids.numel(), capacity or 0
                    )

    def test_low_bucket_small_extension_preserves_other_dispatch(self):
        blocks = (1, 2, 4, 8, 16, 24, 32, 48, 63, 64, 96, 128, 256, 1024, 4096)
        cases = [
            (rows, buckets, block, None, "small" if rows <= 768 else "jit")
            for rows in (512, 513, 640, 768, 769)
            for buckets in (1, 2, 3, 4)
            for block in blocks
        ]
        cases += [
            (rows, 5, block, None, "small" if rows <= 512 else "jit")
            for rows in (512, 513, 768)
            for block in blocks
        ]
        cases += [
            (768, 4, block, capacity, "large")
            for block in blocks
            for capacity in (0, 2048)
        ]
        for rows, buckets, block, capacity, expected in cases:
            with self.subTest(
                rows=rows, buckets=buckets, block=block, capacity=capacity
            ):
                with (
                    mock.patch.object(SHARED_ROUTING, "_build_small_route") as small,
                    mock.patch.object(SHARED_ROUTING, "_build_large_route") as large,
                    mock.patch.object(SHARED_ROUTING, "_build_route_bucket_ids"),
                    mock.patch.object(
                        SHARED_ROUTING,
                        "_align_block_size_jit",
                        return_value=(None, None, None),
                    ) as jit,
                ):
                    SHARED_ROUTING.build_route(
                        torch.zeros(rows, dtype=torch.int32),
                        max_loras=buckets,
                        block_size=block,
                        capacity=capacity,
                    )
                for name, builder in (("small", small), ("large", large), ("jit", jit)):
                    self.assertEqual(builder.call_count, int(expected == name))

    def test_small_extension_counts_pairs_and_keeps_native_warps(self):
        for rows, width, warps in ((512, 1, 4), (513, 1, 8), (768, 1, 8), (768, 8, 8)):
            with self.subTest(rows=rows, width=width):
                slots = torch.zeros(rows // width, dtype=torch.int32)
                groups = (
                    torch.zeros((rows // width, width), dtype=torch.int32)
                    if width > 1
                    else None
                )
                recorder = _KernelRecorder()
                with mock.patch.object(
                    SHARED_ROUTING, "_build_small_route_kernel", recorder
                ):
                    route = SHARED_ROUTING.build_route(
                        slots,
                        group_ids=groups,
                        groups_per_slot=1,
                        max_loras=4,
                        block_size=32,
                        workspace=_Workspace(),
                        tensor_prefix="test:small",
                    )
                self.assertEqual(route.num_rows, rows)
                self.assertEqual(len(recorder.calls), 1)
                grid, args, kwargs = recorder.calls[0]
                self.assertEqual(grid, (1,))
                self.assertEqual(args[6], rows)
                self.assertEqual(kwargs["WIDTH"], width)
                self.assertEqual(kwargs["num_warps"], warps)

    def test_raw_view_never_allocates_or_launches_even_above_cutoffs(self):
        workspace = _Workspace()
        with (
            mock.patch.object(SHARED_ROUTING, "_build_small_route") as small,
            mock.patch.object(SHARED_ROUTING, "_build_large_route") as aligned,
            mock.patch.object(SHARED_ROUTING, "_build_route_bucket_ids") as buckets,
            mock.patch.object(SHARED_ROUTING, "_align_block_size_jit") as jit_align,
        ):
            route = SHARED_ROUTING.build_route(
                torch.zeros(16384, dtype=torch.int32),
                max_loras=8192,
                block_size=16,
                view=RouteViewKind.RAW,
                workspace=workspace,
                tensor_prefix="test:raw",
                capacity=65536,
            )
        for builder in (small, aligned, buckets, jit_align):
            builder.assert_not_called()
        self.assertIs(route.view, RouteViewKind.RAW)
        self.assertEqual(workspace.tensors, {})


class TestRoutePdlWiring(CustomTestCase):
    def test_plans_build_exactly_their_aligned_routes(self):
        reference = SERIAL_MATERIALIZED_REFERENCE
        shared_down = dataclasses.replace(
            reference,
            down_b=dataclasses.replace(reference.down_b, is_shared_outer=True),
        )
        shared_token = dataclasses.replace(
            reference,
            gate_up_a=PLAN.ASpec(
                PLAN.Site.GATE_UP,
                PLAN.AFamily.TOKEN_GROUPED,
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
            token_slots,
            *,
            group_ids=None,
            groups_per_slot=1,
            max_loras,
            block_size,
            view,
            workspace=None,
            tensor_prefix=None,
        ):
            # Per-expert routes get one bucket per local expert; the shared-outer
            # route folds the pairs' experts into one bucket per slot; the
            # shared-token route has one row per token and no groups at all.
            if tensor_prefix.startswith("route:shared_token"):
                self.assertEqual((groups_per_slot, group_ids), (1, None))
            elif tensor_prefix.endswith("shared_outer"):
                self.assertEqual(groups_per_slot, 1)
                self.assertIsNotNone(group_ids)
            else:
                self.assertEqual(groups_per_slot, 2)
            calls.append((tensor_prefix, groups_per_slot == 1))
            built = _route(
                token_slots,
                group_ids=group_ids,
                block_size=block_size,
                padded_count=torch.tensor([block_size], dtype=torch.int32),
            )
            return built

        with mock.patch.object(ROUTING, "build_route", side_effect=fake_aligned):
            for plan in (reference, shared_down, shared_token):
                ROUTING.build_moe_routes(
                    plan,
                    topk_ids=torch.tensor([[0, 1]], dtype=torch.int32),
                    token_lora_mapping=torch.tensor([0], dtype=torch.int32),
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
                ("route:shared_token:sorted:16", True),
            },
        )

    def test_parallel_plan_builds_both_routes_plus_the_shared_token_follow_on(self):
        reference = SERIAL_MATERIALIZED_REFERENCE
        plan = dataclasses.replace(
            reference,
            gate_up_a=PLAN.ASpec(
                PLAN.Site.GATE_UP,
                PLAN.AFamily.TOKEN_GROUPED,
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
            token_slots,
            *,
            group_ids=None,
            groups_per_slot=1,
            max_loras,
            block_size,
            view=None,
            workspace=None,
            tensor_prefix=None,
        ):
            padded = torch.zeros(1, dtype=torch.int32)
            padded.fill_(block_size)
            built = _route(
                token_slots,
                group_ids=group_ids,
                block_size=block_size,
                padded_count=padded,
            )
            calls.append((tensor_prefix, groups_per_slot == 1))
            return built

        with mock.patch.object(ROUTING, "build_route", side_effect=fake_route):
            ROUTING.build_moe_routes(
                plan,
                topk_ids=torch.tensor([[0, 1]], dtype=torch.int32),
                token_lora_mapping=torch.tensor([0], dtype=torch.int32),
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
                ("route:shared_token:sorted:16", True),
            },
        )

    def _run_route(self, *, use_pdl, is_shared_outer=False):
        recorders = [_KernelRecorder() for _ in range(3)]
        with (
            _arch_pdl(bool(use_pdl)),
            mock.patch.object(SHARED_ROUTING, "_route_histogram_kernel", recorders[0]),
            mock.patch.object(SHARED_ROUTING, "_route_scan_kernel", recorders[1]),
            mock.patch.object(SHARED_ROUTING, "_route_place_kernel", recorders[2]),
        ):
            route = SHARED_ROUTING._build_large_route(
                torch.tensor([0], dtype=torch.int32),
                torch.tensor([[0, 1]], dtype=torch.int32),
                groups_per_slot=1 if is_shared_outer else 2,
                max_loras=2,
                block_size=16,
                workspace=_Workspace(),
                tensor_prefix="test:route",
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


class TestSharedTokenRoute(CustomTestCase):
    def test_shared_token_route_groups_the_tokens_by_slot_without_groups(self):
        # Shared-token routes contain one row per token, grouped only by adapter slot.
        reference = SERIAL_MATERIALIZED_REFERENCE
        shared_plan = dataclasses.replace(
            reference,
            gate_up_a=PLAN.ASpec(
                PLAN.Site.GATE_UP,
                PLAN.AFamily.TOKEN_GROUPED,
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
        workspace = _Workspace()
        seen = {}

        def fake_aligned(route_token_slots, **kwargs):
            if kwargs["tensor_prefix"].startswith("route:shared_token"):
                seen.update(kwargs, token_slots=route_token_slots)
            return _route(
                route_token_slots,
                group_ids=kwargs.get("group_ids"),
                block_size=kwargs["block_size"],
                padded_count=torch.tensor([16], dtype=torch.int32),
                groups_per_slot=kwargs.get("groups_per_slot", 1),
                max_loras=kwargs["max_loras"],
            )

        with (
            _arch_pdl(False),
            mock.patch.object(ROUTING, "build_route", side_effect=fake_aligned),
        ):
            routes = ROUTING.build_moe_routes(
                shared_plan,
                topk_ids=topk_ids,
                token_lora_mapping=token_lora_mapping,
                num_local_experts=2,
                max_loras=2,
                block_size=16,
                workspace=workspace,
            )
        self.assertIs(seen["token_slots"], token_lora_mapping)
        self.assertIsNone(seen.get("group_ids"))
        self.assertEqual(seen.get("groups_per_slot", 1), 1)
        self.assertEqual(seen["tensor_prefix"], "route:shared_token:sorted:16")
        self.assertEqual(routes.shared_token.groups_per_slot, 1)

    def test_large_shared_token_route_cannot_overwrite_retained_pair_counts(self):
        """Retained routes need distinct workspace prefixes even when bucket counts match.
        With one expert, shared-token and pair routes collide at G but retain different
        counts (T versus T*K); constructing one must not overwrite the other.
        """
        reference = SERIAL_MATERIALIZED_REFERENCE
        shared_plan = dataclasses.replace(
            reference,
            gate_up_a=PLAN.ASpec(
                PLAN.Site.GATE_UP,
                PLAN.AFamily.TOKEN_GROUPED,
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
                    route_token_slots,
                    *,
                    group_ids=None,
                    groups_per_slot=1,
                    max_loras,
                    block_size,
                    view,
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
                        device=route_token_slots.device,
                    )
                    rows = route_token_slots if group_ids is None else group_ids
                    padded.fill_(rows.numel())
                    built = _route(
                        route_token_slots,
                        group_ids=group_ids,
                        block_size=block_size,
                        padded_count=padded,
                        groups_per_slot=groups_per_slot,
                        max_loras=max_loras,
                    )
                    return built

                with mock.patch.object(ROUTING, "build_route", side_effect=fake_route):
                    routes = ROUTING.build_moe_routes(
                        shared_plan,
                        topk_ids=topk_ids,
                        token_lora_mapping=token_lora_mapping,
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
                        "route:shared_token:sorted:16",
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


class TestLaunchConfigRoutePreflight(CustomTestCase):
    def test_subwarp_route_tile_is_rejected_at_construction(self):
        # The grouped LoRA kernels take this value as their tl.dot row tile,
        # so the config constructor itself rejects anything below 16.
        with self.assertRaisesRegex(ValueError, ">= 16"):
            PLAN.MoeLoraLaunchConfig(routing_block_size=8)


if __name__ == "__main__":
    unittest.main()
