"""CPU-only guards for the production masked fusion/finalizer ABI.

GPU numerical tests run with the Step-8 sweep.  These checks intentionally
exercise source-level properties that must be true before a kernel can be
compiled: provider capability names, exact config vocabularies, opaque
workspace routing, optional pair activation stores, and exactly-once scaling
ownership.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import msgspec
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-c-test-cpu")


ROOT = Path(__file__).resolve().parents[4]
LORA_MOE = ROOT / "python/sglang/srt/lora/moe"
PROVIDER = LORA_MOE / "base_gemm_provider"
EP_MOE = ROOT / "python/sglang/kernels/ops/moe/ep_moe_kernels.py"


def _source(name: str) -> str:
    return (PROVIDER / name).read_text()


def _function(source: str, name: str) -> str:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ):
            return ast.get_source_segment(source, node) or ""
    raise AssertionError(f"function {name!r} not found")


class TestMaskedFusionSource(unittest.TestCase):
    def test_lora_a_to_b_pdl_has_a_complete_signal_wait_launch_chain(self):
        # The kernel-level A->B PDL ABI stays complete (benchmarks drive it);
        # the serving runner no longer threads it (plan-level PDL knobs were
        # retired: route PDL is arch-keyed, A->B edges measured no better
        # than launch-order overlap, 2026-08 twins).
        a = (LORA_MOE / "bf16.py").read_text()
        b = (LORA_MOE / "lora_b.py").read_text()
        runner = (LORA_MOE / "moe_lora_runner.py").read_text()

        grouped_a = _function(a, "_grouped_lora_a_kernel")
        one_launch_b = _function(b, "_one_launch_sliced_lora_b_kernel")
        b_launcher = _function(b, "one_launch_sliced_lora_b")
        self.assertIn("gdc_launch_dependents", grouped_a)
        self.assertIn("gdc_wait", one_launch_b)
        self.assertLess(
            grouped_a.index("gdc_launch_dependents"),
            grouped_a.index("accumulator ="),
        )
        self.assertLess(
            one_launch_b.index("if group == -1"),
            one_launch_b.index("gdc_wait"),
        )
        self.assertIn('"launch_pdl": True', b_launcher)
        self.assertNotIn("produce_pdl", runner)
        self.assertNotIn("consume_pdl", runner)

    def test_cutedsl_base_pdl_separates_producer_and_consumer_roles(self):
        api = _source("cutedsl_masked/api.py")
        sm100 = _source("cutedsl_masked/kernel.py")
        sm90 = _source("cutedsl_masked/kernel_sm90.py")
        activation = _source("masked_activation.py")
        middle = _source("masked_fused_middle.py")
        finalize = _source("masked_finalize.py")
        post_reorder = EP_MOE.read_text()

        self.assertIn("produce_pdl: bool = False", api)
        self.assertNotIn("use_pdl: bool = False", api)
        for source in (sm100, sm90):
            kernel = _function(source, "kernel")
            self.assertIn("griddepcontrol_launch_dependents", kernel)
            self.assertIn("self.produce_pdl", kernel)
        # The base GEMM is the primary producer. It must not request the
        # dependent-launch attribute against its own S1/S3 predecessor.
        self.assertNotIn("use_pdl=self.produce_pdl", sm100)
        self.assertNotIn("use_pdl=self.produce_pdl", sm90)

        for source, kernel_name, launcher_name in (
            (
                activation,
                "_activation_delta_masked_kernel",
                "act_delta_masked",
            ),
            (middle, "_b_act_kernel", "run_masked_fused_middle"),
        ):
            self.assertIn("gdc_wait", _function(source, kernel_name))
            self.assertIn('"launch_pdl": True', _function(source, launcher_name))

        # The shared upstream combine deliberately takes NO PDL edge: no
        # shipped config row enables a base down -> finalize handoff, so the
        # wait would be dead weight in a file the whole MoE stack shares.
        combine = _function(post_reorder, "post_reorder_deepgemm_triton_kernel")
        self.assertNotIn("gdc_wait", combine)
        self.assertNotIn(
            '"launch_pdl": True', _function(post_reorder, "post_reorder_deepgemm")
        )

    def test_contiguous_cutedsl_launch_passes_no_pdl_argument(self):
        # The contiguous provider's _launch has no produce_pdl parameter
        # (contiguous compiles no producer twins); passing one crashes at
        # GB300 prefill-graph capture. SM90 CI cannot reach this class, so
        # pin it at the source level: no call-site kwarg inside the class.
        src = _source("cutedsl_bf16.py")
        contiguous = src[src.index("class CuteDslBf16ContiguousProvider") :]
        self.assertNotIn("produce_pdl=", contiguous)

    def test_deepgemm_sm90_geometry_guard_checks_both_contractions(self):
        class StubProvider:
            def __init__(self, quant_info):
                self.quant_info = quant_info

        base_module = types.ModuleType("sglang.srt.lora.moe.base_gemm_provider.base")
        base_module.MoeBaseProviderContract = lambda **kwargs: SimpleNamespace(**kwargs)
        row_module = types.ModuleType(
            "sglang.srt.lora.moe.base_gemm_provider.masked_row_domain"
        )
        row_module.MaskedRowDomainProvider = StubProvider
        row_module.MaskedRowWorkspace = object
        quant_module = types.ModuleType("sglang.srt.lora.moe.quant_info")
        quant_module.MoeLoraBf16QuantInfo = object
        packages = {}
        for name in (
            "sglang",
            "sglang.srt",
            "sglang.srt.lora",
            "sglang.srt.lora.moe",
            "sglang.srt.lora.moe.base_gemm_provider",
        ):
            package = types.ModuleType(name)
            package.__path__ = []
            packages[name] = package
        spec = importlib.util.spec_from_file_location(
            "_cpu_deepgemm_bf16", PROVIDER / "deep_gemm_bf16.py"
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(
            sys.modules,
            {
                **packages,
                base_module.__name__: base_module,
                row_module.__name__: row_module,
                quant_module.__name__: quant_module,
            },
        ):
            spec.loader.exec_module(module)

        def quant_info(hidden_size, intermediate_size):
            return SimpleNamespace(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                w2_weight=SimpleNamespace(device=torch.device("cpu")),
            )

        guard = module.DeepGemmBf16Provider._require_supported_geometry
        with mock.patch.object(
            module.torch.cuda, "get_device_capability", return_value=(9, 0)
        ):
            guard(quant_info(512, 384))
            for hidden_size, intermediate_size, expected in (
                (510, 384, "hidden_size=510"),
                (512, 352, "intermediate_size=352"),
            ):
                with self.subTest(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                ):
                    with self.assertRaisesRegex(ValueError, expected):
                        guard(quant_info(hidden_size, intermediate_size))
            with self.assertRaisesRegex(ValueError, "hidden_size=510"):
                module.DeepGemmBf16Provider(quant_info(510, 384))

        with mock.patch.object(
            module.torch.cuda, "get_device_capability", return_value=(10, 0)
        ):
            guard(quant_info(510, 352))

    def test_provider_surface_is_explicit_and_workspace_stays_opaque(self):
        base = _source("base.py")
        row = _source("masked_row_domain.py")
        runner = (LORA_MOE / "moe_lora_runner.py").read_text()
        for method in (
            "fused_middle_implementations",
            "supports_fused_middle",
            "run_fused_middle",
            "fused_finalize_implementations",
            "supports_fused_finalize",
            "run_shared_rank_finalize",
            "run_shared_rank_reduce",
            "finish_shared_rank_finalize",
            "mapped_down_lora_a_input",
        ):
            self.assertIn(f"def {method}(", base)
        for method in (
            "fused_middle_implementations",
            "run_fused_middle",
            "fused_finalize_implementations",
            "run_shared_rank_finalize",
            "run_shared_rank_reduce",
            "finish_shared_rank_finalize",
            "mapped_down_lora_a_input",
        ):
            self.assertIn(f"def {method}(", row)

        body = _function(row, "run_fused_middle")
        self.assertIn("ws.src2dst", body)
        self.assertNotIn("ws.hidden_permuted", body)
        self.assertNotIn("ws.masked_m", body)
        shared = _function(row, "run_shared_rank_finalize")
        self.assertIn("self.run_shared_rank_reduce(", shared)
        self.assertIn("self.finish_shared_rank_finalize(", shared)
        self.assertNotIn("ws.hidden_permuted", shared)
        self.assertNotIn("ws.masked_m", shared)
        finish = _function(row, "finish_shared_rank_finalize")
        self.assertNotIn("self.finalize(", finish)
        self.assertIn("invoke(", finish)
        self.assertIn("down_masked=down_masked", finish)
        self.assertIn("src2dst=ws.src2dst", finish)
        mapped = _function(row, "mapped_down_lora_a_input")
        self.assertIn("pair_to_row=ws.src2dst", mapped)
        self.assertNotIn("ws.src2dst", runner)

        # The contiguous domain exposes the same shared-rank finalize ABI
        # through the same opaque-workspace routing: the reduce is pure
        # pair-domain and the tail reaches physical rows only via src2dst.
        contiguous = _source("contiguous_row_domain.py")
        for method in (
            "fused_finalize_implementations",
            "install_fused_finalize_implementation",
            "run_shared_rank_finalize",
            "run_shared_rank_reduce",
            "finish_shared_rank_finalize",
        ):
            self.assertIn(f"def {method}(", contiguous)
        contiguous_finish = _function(contiguous, "finish_shared_rank_finalize")
        self.assertIn("invoke(", contiguous_finish)
        self.assertIn("down_masked=down_masked", contiguous_finish)
        self.assertIn("src2dst=ws.src2dst", contiguous_finish)
        contiguous_reduce = _function(contiguous, "run_shared_rank_reduce")
        self.assertIn("del ws", contiguous_reduce)

    def test_middle_pair_store_is_optional_and_masked_store_is_unconditional(self):
        source = _source("masked_fused_middle.py")
        self.assertIn('("b_activation",)', source)
        self.assertIn('("silu", "relu2")', source)
        base_columns = _function(source, "_base_columns")
        self.assertIn("interleaved", base_columns)
        self.assertIn("gate_first", base_columns)
        body = _function(source, "_b_act_kernel")
        self.assertIn("act_masked_ptr", body)
        self.assertIn("act_pairs_ptr", body)
        self.assertIn("src2dst_ptr", body)
        self.assertIn("tl.where(base_valid", body)
        self.assertIn("mask=base_valid", body)
        b_act = _function(source, "_b_act_kernel")
        self.assertIn("store_pair_act: tl.constexpr", b_act)
        self.assertIn("if store_pair_act:", b_act)
        self.assertLess(
            b_act.index("act_masked_ptr +"),
            b_act.index("if store_pair_act:"),
        )
        launcher = _function(source, "run_masked_fused_middle")
        self.assertIn("store_pair_act=act_pairs is not None", launcher)

    def test_materialized_activation_supports_swiglu_and_relu2(self):
        source = _source("masked_activation.py")
        kernel = _function(source, "_activation_delta_masked_kernel")
        wrapper = _function(source, "act_delta_masked")
        self.assertIn("NUM_SLICES", kernel)
        self.assertIn("ACTIVATION_TYPE", kernel)
        self.assertIn("tl.maximum", _function(source, "apply_activation"))
        self.assertIn('("silu", "relu2")', source)
        self.assertIn("num_slices * inter", wrapper)

    def test_shared_rank_finalize_is_fail_closed_and_two_stage(self):
        source = _source("masked_finalize.py")
        provider = _source("masked_row_domain.py")
        wrapper = _function(provider, "run_shared_rank_finalize")
        validator = _function(source, "_validate_shared_route")
        self.assertIn("shared_outer_local_expert_count is None", validator)
        self.assertIn("lora_experts_per_adapter != 1", validator)
        self.assertIn("self.run_shared_rank_reduce(", wrapper)
        self.assertIn("self.finish_shared_rank_finalize(", wrapper)
        reduce_body = _function(source, "_shared_rank_reduce_kernel")
        self.assertNotIn("routed_scaling", reduce_body)
        from_scratch = _function(source, "_shared_from_scratch_finalize_kernel")
        self.assertEqual(from_scratch.count("routed_scaling"), 2)
        self.assertIn(
            "routed_scaling * (base_acc +",
            " ".join(from_scratch.split()),
        )

    def test_config_names_match_the_execution_contract(self):
        middle = _source("masked_fused_middle.py")
        finalize = _source("masked_finalize.py")
        for key in (
            "BLOCK_SIZE_W",
            "BLOCK_SIZE_K",
            "GROUP_SIZE_M",
            "num_warps",
            "num_stages",
        ):
            self.assertIn(f'"{key}"', middle)
        for key in ("BLOCK_SIZE_T", "BLOCK_SIZE_H", "BLOCK_SIZE_K"):
            self.assertIn(f'"{key}"', finalize)
        self.assertIn('"reduce"', finalize)
        self.assertIn('"tail"', finalize)

    def test_provider_can_force_injected_implementation(self):
        row = _source("masked_row_domain.py")
        self.assertIn("def install_fused_middle_implementation(", row)
        self.assertIn("def install_fused_finalize_implementation(", row)
        self.assertIn(
            "self._fused_middle_impls[(family, activation, implementation)]",
            row,
        )
        self.assertIn("self._shared_reduce_impls[implementation]", row)
        self.assertIn("self._shared_tail_impls[implementation]", row)

    def test_provider_constructs_every_builtin_capability_from_shared_constants(self):
        """Catch missing runtime imports in the attach-time registry builder."""
        packages = {}
        for name in (
            "sglang",
            "sglang.kernels",
            "sglang.kernels.ops",
            "sglang.kernels.ops.moe",
            "sglang.srt",
            "sglang.srt.lora",
            "sglang.srt.lora.moe",
            "sglang.srt.lora.moe.base_gemm_provider",
        ):
            package = types.ModuleType(name)
            package.__path__ = []
            packages[name] = package

        base = types.ModuleType("sglang.srt.lora.moe.base_gemm_provider.base")

        class MappedInput(msgspec.Struct, frozen=True, kw_only=True):
            rows: torch.Tensor
            pair_to_row: torch.Tensor

        class StubBaseProvider:
            def act_out_shape(self, ws):
                return (
                    self.quant_info.num_local_experts,
                    ws.m_max,
                    self.quant_info.intermediate_size,
                )

        base.MappedLoraAInput = MappedInput
        base.MoeBaseProvider = StubBaseProvider
        quant = types.ModuleType("sglang.srt.lora.moe.quant_info")
        quant.MoeLoraBf16QuantInfo = object
        ep = types.ModuleType("sglang.kernels.ops.moe.ep_moe_kernels")
        ep.post_reorder_deepgemm = lambda *_args, **_kwargs: None
        activation = types.ModuleType(
            "sglang.srt.lora.moe.base_gemm_provider.masked_activation"
        )
        activation.act_delta_masked = lambda *_args, **_kwargs: None
        dispatch = types.ModuleType(
            "sglang.srt.lora.moe.base_gemm_provider.masked_dispatch"
        )
        dispatch.fused_masked_preprocess = lambda *_args, **_kwargs: None
        finalize = types.ModuleType(
            "sglang.srt.lora.moe.base_gemm_provider.masked_finalize"
        )
        finalize.MASKED_FINALIZE_TRITON = "triton"
        finalize.invoke_shared_from_scratch_finalize = lambda **_kwargs: None
        finalize.invoke_shared_rank_reduce = lambda **_kwargs: None
        middle = types.ModuleType(
            "sglang.srt.lora.moe.base_gemm_provider.masked_fused_middle"
        )
        middle.MASKED_MIDDLE_ACTIVATIONS = ("silu", "relu2")
        middle.MASKED_MIDDLE_FAMILIES = ("b_activation",)
        middle.MASKED_MIDDLE_TRITON = "triton"
        middle.run_masked_fused_middle = lambda *_args, **_kwargs: None

        spec = importlib.util.spec_from_file_location(
            "_cpu_masked_row_domain", PROVIDER / "masked_row_domain.py"
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        injected = {
            **packages,
            base.__name__: base,
            quant.__name__: quant,
            ep.__name__: ep,
            activation.__name__: activation,
            dispatch.__name__: dispatch,
            finalize.__name__: finalize,
            middle.__name__: middle,
        }
        with mock.patch.dict(sys.modules, injected):
            spec.loader.exec_module(module)
            provider = module.MaskedRowDomainProvider(
                SimpleNamespace(
                    intermediate_size=8,
                    num_local_experts=2,
                    hidden_size=4,
                    w2_weight=torch.empty((2, 4, 8)),
                    w13_weight=torch.empty((2, 16, 4)),
                )
            )

        for family in middle.MASKED_MIDDLE_FAMILIES:
            for candidate_activation in middle.MASKED_MIDDLE_ACTIVATIONS:
                self.assertTrue(
                    provider.supports_fused_middle(
                        family,
                        activation=candidate_activation,
                    )
                )
        self.assertIn(
            "triton",
            provider.fused_finalize_implementations("shared_rank_reduce", "shared"),
        )

        provider.contract = SimpleNamespace(lora_activation_dtype=torch.bfloat16)
        src2dst = torch.tensor([2, 0, -1, 3], dtype=torch.int32)
        workspace = module.MaskedRowWorkspace(
            hidden_permuted=torch.empty((2, 4, 4), dtype=torch.bfloat16),
            masked_m=torch.tensor([2, 2], dtype=torch.int32),
            expected_m=2,
            src2dst=src2dst,
            m_max=4,
            retained_inputs=True,
        )
        activation_rows = torch.randn((2, 4, 8), dtype=torch.bfloat16)
        mapped = provider.mapped_down_lora_a_input(workspace, activation_rows)
        self.assertEqual(tuple(mapped.rows.shape), (8, 8))
        self.assertIs(mapped.pair_to_row, src2dst)
        self.assertEqual(mapped.rows.data_ptr(), activation_rows.data_ptr())
        with self.assertRaisesRegex(ValueError, "must be"):
            provider.mapped_down_lora_a_input(workspace, activation_rows[:, :3])
        workspace.src2dst = torch.arange(8, dtype=torch.int32)[::2]
        with self.assertRaisesRegex(ValueError, "contiguous"):
            provider.mapped_down_lora_a_input(workspace, activation_rows)

    def test_cutedsl_relu2_schedule_uses_the_resident_slice_count(self):
        """A one-slice ReLU2 provider must not schedule a two-slice GEMM1."""

        class StubWorkspace(msgspec.Struct, kw_only=True):
            hidden_permuted: torch.Tensor
            masked_m: torch.Tensor
            expected_m: int
            src2dst: torch.Tensor
            m_max: int
            retained_inputs: bool

        class StubProvider:
            @property
            def gate_up_slices(self):
                return self._gate_up_slices

            def prepare(self, *_args):
                return self._base_workspace

        base_module = types.ModuleType("sglang.srt.lora.moe.base_gemm_provider.base")
        base_module.MoeBaseProviderContract = lambda **kwargs: SimpleNamespace(**kwargs)
        row_module = types.ModuleType(
            "sglang.srt.lora.moe.base_gemm_provider.masked_row_domain"
        )
        row_module.MaskedRowDomainProvider = StubProvider
        row_module.MaskedRowWorkspace = StubWorkspace
        quant_module = types.ModuleType("sglang.srt.lora.moe.quant_info")
        quant_module.MoeLoraBf16QuantInfo = object
        packages = {}
        for name in (
            "sglang",
            "sglang.srt",
            "sglang.srt.lora",
            "sglang.srt.lora.moe",
            "sglang.srt.lora.moe.base_gemm_provider",
        ):
            package = types.ModuleType(name)
            package.__path__ = []
            packages[name] = package
        spec = importlib.util.spec_from_file_location(
            "_cpu_cutedsl_bf16", PROVIDER / "cutedsl_bf16.py"
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(
            sys.modules,
            {
                **packages,
                base_module.__name__: base_module,
                row_module.__name__: row_module,
                quant_module.__name__: quant_module,
            },
        ):
            spec.loader.exec_module(module)

        base = StubWorkspace(
            hidden_permuted=torch.empty((2, 4, 8)),
            masked_m=torch.tensor([2, 2], dtype=torch.int32),
            expected_m=2,
            src2dst=torch.arange(4, dtype=torch.int32),
            m_max=4,
            retained_inputs=False,
        )
        provider = object.__new__(module.CuteDslBf16Provider)
        provider._base_workspace = base
        provider.quant_info = SimpleNamespace(intermediate_size=16, hidden_size=8)
        provider._gate_up_slices = 1
        provider._max_token_clusters = 1024
        provider._compiled = {8: object()}
        provider._config_table = None
        observed = {}

        def build(masked_m, **kwargs):
            observed.update(kwargs)
            return (masked_m, masked_m, masked_m, masked_m)

        provider._build_schedules = build
        provider.prepare(torch.empty((4, 8)), torch.zeros((4, 1)), 1)
        self.assertEqual(observed["n_gemm1"], 16)
        self.assertEqual(observed["n_gemm2"], 8)


if __name__ == "__main__":
    unittest.main()
