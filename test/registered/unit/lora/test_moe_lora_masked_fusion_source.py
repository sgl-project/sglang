"""Source-level checks on the masked fusion and finalize files.

These tests read the source text, so they report a broken call contract
before anyone compiles a kernel. The Step-8 sweep checks the GPU numbers.
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


class _ContiguousRowStateStub(msgspec.Struct, kw_only=True):
    """Providers subclass the real row state, so the stub must be a Struct too."""


class TestMaskedFusionSource(unittest.TestCase):
    def test_lora_a_to_b_pdl_has_a_complete_signal_wait_launch_chain(self):
        # The benchmarks still use the A-to-B chain of Programmatic Dependent
        # Launch (PDL), so the kernels must keep it. The serving runner does
        # not use PDL. An A-to-B edge measured no better than launch order.
        a = (LORA_MOE / "lora_a.py").read_text()
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
        act = _source("masked_fused_act.py")
        finalize = _source("masked_finalize.py")
        post_reorder = EP_MOE.read_text()

        self.assertIn("produce_pdl: bool = False", api)
        self.assertNotIn("use_pdl: bool = False", api)
        for source in (sm100, sm90):
            kernel = _function(source, "kernel")
            self.assertIn("griddepcontrol_launch_dependents", kernel)
            self.assertIn("self.produce_pdl", kernel)
        # The base GEMM signals the kernels after it. It must not also wait
        # on the kernel before it, so use_pdl must not read produce_pdl.
        self.assertNotIn("use_pdl=self.produce_pdl", sm100)
        self.assertNotIn("use_pdl=self.produce_pdl", sm90)

        for source, kernel_name, launcher_name in (
            (
                activation,
                "_activation_delta_masked_kernel",
                "act_delta_masked",
            ),
            (act, "_b_act_kernel", "run_masked_fused_act"),
        ):
            self.assertIn("gdc_wait", _function(source, kernel_name))
            self.assertIn('"launch_pdl": True', _function(source, launcher_name))

        # The combine kernel takes no PDL edge. No shipped plan row turns on
        # the base-down to finalize handoff. The whole MoE stack shares this
        # file, and an unused wait slows every other caller.
        combine = _function(post_reorder, "post_reorder_deepgemm_triton_kernel")
        self.assertNotIn("gdc_wait", combine)
        self.assertNotIn(
            '"launch_pdl": True', _function(post_reorder, "post_reorder_deepgemm")
        )

    def test_contiguous_cutedsl_launch_passes_no_pdl_argument(self):
        # The contiguous provider compiles no producer kernel, so its _launch
        # has no produce_pdl parameter. A call that passes one crashes when
        # the prefill graph captures.
        src = _source("cutedsl_bf16.py")
        contiguous = src[src.index("class CuteDslBf16ContiguousProvider") :]
        self.assertNotIn("produce_pdl=", contiguous)

    def test_both_device_kernels_carry_the_contiguous_admission_and_fold(self):
        # prepare_contiguous picks a kernel by device capability, so the SM90
        # kernel must check the same rules as the SM100 kernel. Both device
        # loops must add the segment base. A loop that misses the add reads
        # the rows of the wrong expert.
        for name in ("cutedsl_masked/kernel.py", "cutedsl_masked/kernel_sm90.py"):
            source = _source(name)
            self.assertIn(
                "contiguous_segments requires swap_ab and use_direct_schedule",
                source,
            )
            self.assertIn("contiguous_segments requires a (1, 1) cluster", source)
            self.assertEqual(source.count("seg_base_tile + "), 2, name)
        api = _source("cutedsl_masked/api.py")
        contiguous_fn = _function(api, "prepare_contiguous")
        self.assertIn("_kernel_class_for(a.device)", contiguous_fn)

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
        row_module.MaskedRowState = object
        quant_module = types.ModuleType("sglang.srt.lora.moe.quant_info")
        quant_module.MoeLoraBf16QuantInfo = object
        contiguous_module = types.ModuleType(
            "sglang.srt.lora.moe.base_gemm_provider.contiguous_row_domain"
        )
        contiguous_module.ContiguousRowDomainProvider = object
        contiguous_module.ContiguousRowState = object
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
                contiguous_module.__name__: contiguous_module,
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
        # run_fused_act stays per row domain: its body differs.
        self.assertIn("def run_fused_act(", base)
        self.assertIn("def run_fused_act(", row)
        # These have one body, on the base.
        contiguous_src = _source("contiguous_row_domain.py")
        for method in (
            "run_shared_rank_finalize",
            "run_shared_rank_reduce",
            "finish_shared_rank_finalize",
            "mapped_down_lora_a_input",
            "finalize",
            "num_local_experts",
            "intermediate_size",
            "hidden_size",
            "gate_up_slices",
        ):
            self.assertIn(f"def {method}(", base)
            self.assertNotIn(f"    def {method}(", row)
            self.assertNotIn(f"    def {method}(", contiguous_src)
        # Every fused stage has exactly one implementation, so no stage takes
        # an implementation name and no provider enumerates or installs one.
        for gone in (
            "implementation: str",
            "def supports_fused_act(",
            "def supports_fused_finalize(",
            "def fused_act_implementations(",
            "def install_fused_act_implementation(",
        ):
            self.assertNotIn(gone, base)
            self.assertNotIn(gone, row)

        body = _function(row, "run_fused_act")
        self.assertIn("row_state.src2dst", body)
        self.assertNotIn("row_state.hidden_permuted", body)
        self.assertNotIn("row_state.masked_m", body)
        # The shared-rank path is one body on the base. It reaches a physical
        # row only through src2dst, and never through a masked-only field.
        shared = _function(base, "run_shared_rank_finalize")
        self.assertIn("self.run_shared_rank_reduce(", shared)
        self.assertIn("self.finish_shared_rank_finalize(", shared)
        finish = _function(base, "finish_shared_rank_finalize")
        self.assertNotIn("self.finalize(", finish)
        self.assertIn("invoke_shared_from_scratch_finalize(", finish)
        self.assertIn("down_masked=down_masked", finish)
        self.assertIn("src2dst=row_state.src2dst", finish)
        reduce_body = _function(base, "run_shared_rank_reduce")
        self.assertIn("del row_state", reduce_body)
        mapped = _function(base, "mapped_down_lora_a_input")
        self.assertIn("pair_to_row=row_state.src2dst", mapped)
        for body in (shared, finish, reduce_body, mapped):
            self.assertNotIn("row_state.hidden_permuted", body)
            self.assertNotIn("row_state.masked_m", body)
        self.assertNotIn("base_gemm_state.src2dst", runner)

    def test_act_pair_store_is_optional_and_masked_store_is_unconditional(self):
        source = _source("masked_fused_act.py")
        self.assertIn('("b_activation",)', source)
        self.assertIn("ActivationFn.parse", source)
        self.assertNotIn('("silu", "relu2")', source)
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
        launcher = _function(source, "run_masked_fused_act")
        self.assertIn("store_pair_act=act_pairs is not None", launcher)

    def test_materialized_activation_keeps_the_two_axes_independent(self):
        """Gating and the activation function are two separate constants.

        NUM_SLICES sets the gating and ACTIVATION_TYPE sets the function, so
        all four combinations compile. One "swiglu-or-relu2" constant cannot
        express a non-gated silu.
        """
        source = _source("masked_activation.py")
        kernel = _function(source, "_activation_delta_masked_kernel")
        wrapper = _function(source, "act_delta_masked")
        self.assertIn("NUM_SLICES", kernel)
        self.assertIn("ACTIVATION_TYPE", kernel)
        self.assertIn("tl.maximum", _function(source, "apply_activation"))
        self.assertIn("ActivationFn.parse", source)
        self.assertNotIn('("silu", "relu2")', source)
        self.assertIn("num_slices * inter", wrapper)

    def test_no_module_restates_the_activation_set(self):
        """Eight modules once held their own copy of the activation names.

        A new activation then needed an edit in all eight modules. A pair of
        names in a literal under moe/ is that copy starting again.
        """
        import re

        from sglang.srt.lora.moe.activation import ActivationFn

        moe_root = PROVIDER.parent
        owner = moe_root / "activation.py"
        # The pattern finds a literal that lists two of the names.
        names = "|".join(re.escape(fn.value) for fn in ActivationFn)
        pattern = re.compile(
            rf"""[\(\[{{]\s*["']({names})["']\s*,\s*["']({names})["']"""
        )
        offenders = []
        for path in sorted(moe_root.rglob("*.py")):
            if path == owner:
                continue
            for number, line in enumerate(path.read_text().splitlines(), 1):
                if pattern.search(line):
                    offenders.append(
                        f"{path.relative_to(moe_root)}:{number}: {line.strip()}"
                    )
        self.assertEqual(offenders, [], "\n".join(offenders))
        self.assertEqual({fn.value for fn in ActivationFn}, {"silu", "relu2"})

    def test_shared_rank_finalize_is_fail_closed_and_two_stage(self):
        source = _source("masked_finalize.py")
        # The wrapper is on the base: both row domains ran it identically.
        wrapper = _function(_source("base.py"), "run_shared_rank_finalize")
        validator = _function(source, "_validate_shared_route")
        self.assertIn("not routing.is_shared_outer", validator)
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
        act = _source("masked_fused_act.py")
        finalize = _source("masked_finalize.py")
        for key in (
            "BLOCK_SIZE_W",
            "BLOCK_SIZE_K",
            "GROUP_SIZE_M",
            "num_warps",
            "num_stages",
        ):
            self.assertIn(f'"{key}"', act)
        for key in ("BLOCK_SIZE_T", "BLOCK_SIZE_H", "BLOCK_SIZE_K"):
            self.assertIn(f'"{key}"', finalize)
        self.assertIn('"reduce"', finalize)
        self.assertIn('"tail"', finalize)

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

        # Load the real base. It imports only msgspec and torch, and the
        # row domains now inherit mapped_down_lora_a_input from it, so a stub
        # would hide the very method this test exercises.
        base_spec = importlib.util.spec_from_file_location(
            "sglang.srt.lora.moe.base_gemm_provider.base", PROVIDER / "base.py"
        )
        base = importlib.util.module_from_spec(base_spec)
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
        act = types.ModuleType(
            "sglang.srt.lora.moe.base_gemm_provider.masked_fused_act"
        )
        act.MASKED_ACT_FAMILIES = ("b_activation",)
        act.MASKED_ACT_TRITON = "triton"
        act.run_masked_fused_act = lambda *_args, **_kwargs: None
        into_base = types.ModuleType("sglang.srt.lora.moe.lora_b")
        into_base.invoke_down_b_into_base = lambda *_args, **_kwargs: None

        base_spec.loader.exec_module(base)
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
            act.__name__: act,
            into_base.__name__: into_base,
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

        # One callable per fused stage, bound at construction.
        self.assertIs(provider._fused_act, act.run_masked_fused_act)

        provider.contract = SimpleNamespace(lora_activation_dtype=torch.bfloat16)
        src2dst = torch.tensor([2, 0, -1, 3], dtype=torch.int32)
        workspace = module.MaskedRowState(
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

    def test_cutedsl_schedule_uses_the_resident_slice_count(self):
        """A non-gated provider must not schedule a two-slice GEMM1.

        The slice count comes from the width of the resident weight. A
        non-gated silu is therefore the same case as relu2.
        """

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
        row_module.MaskedRowState = StubWorkspace
        quant_module = types.ModuleType("sglang.srt.lora.moe.quant_info")
        quant_module.MoeLoraBf16QuantInfo = object
        contiguous_module = types.ModuleType(
            "sglang.srt.lora.moe.base_gemm_provider.contiguous_row_domain"
        )
        contiguous_module.ContiguousRowDomainProvider = object
        contiguous_module.ContiguousRowState = _ContiguousRowStateStub

        def _ceiling_not_exercised(*_args, **_kwargs):
            raise AssertionError("contiguous_m_pad_ceiling is not exercised here")

        contiguous_module.contiguous_m_pad_ceiling = _ceiling_not_exercised
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
                contiguous_module.__name__: contiguous_module,
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
