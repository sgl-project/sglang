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
KERNELS = LORA_MOE / "kernels"
EP_MOE = ROOT / "python/sglang/kernels/ops/moe/ep_moe_kernels.py"


def _source(name: str) -> str:
    """Read a module by file name from either package.

    The Triton kernels moved to moe/kernels while the providers stayed in
    base_gemm_provider, so look in both rather than making every caller say
    which one it means.
    """
    for base in (PROVIDER, KERNELS):
        path = base / name
        if path.exists():
            return path.read_text()
    raise AssertionError(f"{name!r} is in neither {PROVIDER} nor {KERNELS}")


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
    def test_no_pdl_edges_outside_routing(self):
        # PDL edges between stages tied in every e2e cell on all three
        # architectures, so the A-to-B and base-to-act edges were removed
        # outright for simplicity. Only the routing
        # builder keeps its adjudicated PDL edge. This contract keeps the
        # dead edges from creeping back without a fresh e2e win.
        for name in (
            "lora_a.py",
            "lora_b.py",
            "activation_delta.py",
            "fused_act.py",
            "cutedsl/api.py",
            "cutedsl/kernel_sm100_bf16.py",
            "cutedsl/kernel_sm100_fp8.py",
            "cutedsl/kernel_sm90_bf16.py",
            "cutedsl/kernel_sm90_fp8.py",
        ):
            source = _source(name)
            self.assertNotIn("gdc_launch_dependents", source, name)
            self.assertNotIn("gdc_wait", source, name)
            self.assertNotIn("launch_pdl", source, name)
            self.assertNotIn("produce_pdl", source, name)
            self.assertNotIn("consume_base_pdl", source, name)
        self.assertIn("USE_PDL", _source("routing.py"))

    def test_contiguous_cutedsl_launch_passes_no_pdl_argument(self):
        # The contiguous provider compiles no producer kernel, so its _launch
        # has no produce_pdl parameter. A call that passes one crashes when
        # the prefill graph captures.
        src = _source("cutedsl_bf16.py")
        contiguous = src[src.index("class CuteDslBf16ContiguousProvider") :]
        self.assertNotIn("produce_pdl=", contiguous)

    def test_both_device_kernels_carry_the_contiguous_admission_and_fold(self):
        # prepare_contiguous_bf16 picks a kernel by device capability, so the SM90
        # kernel must check the same rules as the SM100 kernel. Both device
        # loops must add the segment base. A loop that misses the add reads
        # the rows of the wrong expert.
        for name in ("cutedsl/kernel_sm100_bf16.py", "cutedsl/kernel_sm90_bf16.py"):
            source = _source(name)
            self.assertIn(
                "contiguous_segments requires swap_ab",
                source,
            )
            self.assertIn("contiguous_segments requires a (1, 1) cluster", source)
            self.assertEqual(source.count("seg_base_tile + "), 2, name)
        api = _source("cutedsl/api.py")
        contiguous_fn = _function(api, "prepare_contiguous_bf16")
        self.assertIn("contiguous_segments=True", contiguous_fn)
        self.assertIn("_bf16_kernel_class_for(a.device)", contiguous_fn)
        compile_fn = _function(api, "_compile_prepared")
        self.assertIn("contiguous_segments=contiguous_segments", compile_fn)

    def test_provider_surface_is_explicit_and_workspace_stays_opaque(self):
        base = _source("base.py")
        row = _source("masked_row_domain.py")
        runner = (LORA_MOE / "moe_lora_runner.py").read_text()
        # fused_act stays per row domain: its body differs.
        self.assertIn("def fused_act(", base)
        self.assertIn("def fused_act(", row)
        # These have one body, on the base.
        contiguous_src = _source("contiguous_row_domain.py")
        for method in (
            "shared_token_delta_finalize",
            "shared_one_pass_finalize",
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

        body = _function(row, "fused_act")
        self.assertIn("row_state.pair_to_row", body)
        self.assertNotIn("row_state.hidden_permuted", body)
        self.assertNotIn("row_state.masked_m", body)
        # The shared finalizes are one body each on the base. They reach a
        # physical row only through pair_to_row, never a masked-only field.
        shared_bodies = [
            _function(base, method)
            for method in (
                "shared_token_delta_finalize",
                "shared_one_pass_finalize",
            )
        ]
        for shared in shared_bodies:
            self.assertNotIn("self.finalize(", shared)
            self.assertIn("down_rows=down_rows", shared)
            self.assertIn("pair_to_row=row_state.pair_to_row", shared)
        mapped = _function(base, "mapped_down_lora_a_input")
        # It hands back the flat row view paired with the route's pair_to_row.
        # Assert the return itself: a punctuation-level match breaks whenever
        # black rewraps the line.
        self.assertIn(
            "return activation.view(-1, activation.shape[-1]), row_state.pair_to_row",
            " ".join(mapped.split()),
        )
        for body in (*shared_bodies, mapped):
            self.assertNotIn("row_state.hidden_permuted", body)
            self.assertNotIn("row_state.masked_m", body)
        self.assertNotIn("base_gemm_state.pair_to_row", runner)

    def test_act_pair_store_is_optional_and_masked_store_is_unconditional(self):
        source = _source("fused_act.py")
        self.assertIn("ActivationFn.parse", source)
        self.assertNotIn('("silu", "relu2")', source)
        base_columns = _function(source, "_base_columns")
        self.assertIn("interleaved", base_columns)
        self.assertIn("gate_first", base_columns)
        body = _function(source, "_b_act_kernel")
        self.assertIn("act_rows_ptr", body)
        self.assertIn("act_pairs_ptr", body)
        self.assertIn("pair_to_row_ptr", body)
        self.assertIn("tl.where(base_valid", body)
        self.assertIn("mask=base_valid", body)
        b_act = _function(source, "_b_act_kernel")
        self.assertIn("store_pair_act: tl.constexpr", b_act)
        self.assertIn("if store_pair_act:", b_act)
        self.assertLess(
            b_act.index("act_rows_ptr +"),
            b_act.index("if store_pair_act:"),
        )
        launcher = _function(source, "_launch_b_act")
        self.assertIn("store_pair_act=act_pairs is not None", launcher)

    def test_materialized_activation_keeps_the_two_axes_independent(self):
        """Gating and the activation function are two separate constants.

        NUM_SLICES sets the gating and ACTIVATION_TYPE sets the function, so
        all four combinations compile. One "swiglu-or-relu2" constant cannot
        express a non-gated silu.
        """
        source = _source("activation_delta.py")
        kernel = _function(source, "_act_delta_kernel")
        launcher = _function(source, "_launch_act_delta")
        self.assertIn("NUM_SLICES", kernel)
        self.assertIn("ACTIVATION_TYPE", kernel)
        self.assertIn("tl.maximum", _function(source, "apply_activation"))
        self.assertIn("ActivationFn.parse", source)
        self.assertNotIn('("silu", "relu2")', source)
        # The slice count comes from the row widths, never from the activation.
        self.assertIn("gateup_rows.shape[-1] // inter", launcher)

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

    def test_shared_token_delta_finalize_is_three_stage(self):
        # The fail-closed guard lives in FinalizeSpec.validate: a shared
        # finalize plan row without shared-outer down-B ownership cannot
        # construct.
        source = _source("finalize.py")
        # The wrapper is on the base: both row domains run it identically.
        wrapper = _function(_source("base.py"), "shared_token_delta_finalize")
        self.assertIn("invoke_shared_token_delta_reduce(", wrapper)
        self.assertIn("grouped_lora_b(", wrapper)
        self.assertIn("invoke_shared_token_delta_tail(", wrapper)
        reduce_body = _function(source, "_shared_token_delta_reduce_kernel")
        self.assertNotIn("routed_scaling", reduce_body)
        tail_body = _function(source, "_shared_token_delta_tail_kernel")
        self.assertEqual(tail_body.count("routed_scaling"), 2)
        self.assertIn(
            "routed_scaling * (base_acc +",
            " ".join(tail_body.split()),
        )

    def test_every_shipped_plan_family_has_an_executor(self):
        """A plan row may only name LoRA-A/B families that run_lora_a /
        run_lora_b execute. The 2026-09-03 kernel retirement dropped the
        token_dense A executor while gb300's decode.shared.nvfp4 row kept
        naming it; every shared-outer nvfp4 decode on SM100 then failed at
        launch and no unit test noticed.
        """
        import glob
        import json

        lora_a = _source("lora_a.py")
        lora_b = _source("lora_b.py")
        rows = []
        for path in sorted(glob.glob(str(LORA_MOE / "configs" / "*.plans.json"))):
            table = json.load(open(path))
            rows.extend(table.get("scenarios", []) + table.get("fallback", []))
        self.assertGreater(len(rows), 0)
        for row in rows:
            plan = row["plan"]
            for key in ("gate_up_a_family", "down_a_family"):
                fam = plan.get(key)
                if fam is not None:
                    self.assertIn(f'case "{fam}":', lora_a, (row["name"], key))
            for key in ("gate_up_b_family", "down_b_family"):
                fam = plan.get(key)
                if fam is not None:
                    self.assertIn(f'case "{fam}":', lora_b, (row["name"], key))

    def test_config_names_match_the_execution_contract(self):
        act = _source("fused_act.py")
        finalize = _source("finalize.py")
        for key in (
            "BLOCK_SIZE_W",
            "BLOCK_SIZE_K",
            "GROUP_SIZE_M",
            "num_warps",
            "num_stages",
        ):
            self.assertIn(f'"{key}"', act)
        for key in ("BLOCK_SIZE_T", "BLOCK_SIZE_H"):
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
            "sglang.srt.lora.moe.kernels",
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
        quant.StandardLayoutQuantInfo = object
        ep = types.ModuleType("sglang.kernels.ops.moe.ep_moe_kernels")
        ep.post_reorder_deepgemm = lambda *_args, **_kwargs: None
        activation = types.ModuleType("sglang.srt.lora.moe.kernels.activation_delta")
        activation.act_delta_masked = lambda *_args, **_kwargs: None
        dispatch = types.ModuleType("sglang.srt.lora.moe.kernels.dispatch_masked")
        dispatch.dispatch_fill_masked_bf16 = lambda *_args, **_kwargs: None
        finalize = types.ModuleType("sglang.srt.lora.moe.kernels.finalize")
        finalize.invoke_shared_token_delta_tail = lambda **_kwargs: None
        finalize.invoke_shared_token_delta_reduce = lambda **_kwargs: None
        act = types.ModuleType("sglang.srt.lora.moe.kernels.fused_act")
        act.fused_b_act_masked = lambda *_args, **_kwargs: None
        into_base = types.ModuleType("sglang.srt.lora.moe.kernels.lora_b")
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
        self.assertIs(provider._fused_act, act.fused_b_act_masked)

        provider.contract = SimpleNamespace(lora_activation_dtype=torch.bfloat16)
        pair_to_row = torch.tensor([2, 0, -1, 3], dtype=torch.int32)
        workspace = module.MaskedRowState(
            hidden_permuted=torch.empty((2, 4, 4), dtype=torch.bfloat16),
            masked_m=torch.tensor([2, 2], dtype=torch.int32),
            expected_m=2,
            pair_to_row=pair_to_row,
            m_max=4,
            retained_inputs=True,
        )
        activation_rows = torch.randn((2, 4, 8), dtype=torch.bfloat16)
        rows, pair_to_row = provider.mapped_down_lora_a_input(
            workspace, activation_rows
        )
        self.assertEqual(tuple(rows.shape), (8, 8))
        self.assertIs(pair_to_row, pair_to_row)
        self.assertEqual(rows.data_ptr(), activation_rows.data_ptr())

    def test_cutedsl_schedule_uses_the_resident_slice_count(self):
        """A non-gated provider must not schedule a two-slice GEMM1.

        The slice count comes from the width of the resident weight. A
        non-gated silu is therefore the same case as relu2.
        """

        class StubWorkspace(msgspec.Struct, kw_only=True):
            hidden_permuted: torch.Tensor
            masked_m: torch.Tensor
            expected_m: int
            pair_to_row: torch.Tensor
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
        base_module.expected_rows_per_expert = lambda num_pairs, num_experts: 1
        row_module = types.ModuleType(
            "sglang.srt.lora.moe.base_gemm_provider.masked_row_domain"
        )
        row_module.MaskedRowDomainProvider = StubProvider
        row_module.MaskedRowState = StubWorkspace
        quant_module = types.ModuleType("sglang.srt.lora.moe.quant_info")
        quant_module.MoeLoraBf16QuantInfo = object
        quant_module.StandardLayoutQuantInfo = object
        contiguous_module = types.ModuleType(
            "sglang.srt.lora.moe.base_gemm_provider.contiguous_row_domain"
        )
        contiguous_module.ContiguousRowDomainProvider = object
        contiguous_module.ContiguousRowState = _ContiguousRowStateStub

        # cutedsl_bf16 builds on the shared tile mixin. Load the real module:
        # it imports only msgspec and torch.
        common_spec = importlib.util.spec_from_file_location(
            "sglang.srt.lora.moe.base_gemm_provider.cutedsl_common",
            PROVIDER / "cutedsl_common.py",
        )
        common = importlib.util.module_from_spec(common_spec)
        common_spec.loader.exec_module(common)
        packages = {}
        for name in (
            "sglang",
            "sglang.srt",
            "sglang.srt.lora",
            "sglang.srt.lora.moe",
            "sglang.srt.lora.moe.base_gemm_provider",
            "sglang.srt.lora.moe.kernels",
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
                common.__name__: common,
            },
        ):
            spec.loader.exec_module(module)

        base = StubWorkspace(
            hidden_permuted=torch.empty((2, 4, 8)),
            masked_m=torch.tensor([2, 2], dtype=torch.int32),
            expected_m=2,
            pair_to_row=torch.arange(4, dtype=torch.int32),
            m_max=4,
            retained_inputs=False,
        )
        provider = object.__new__(module.CuteDslBf16MaskedProvider)
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
