"""Source-level pins on the PDL wait/signal wiring.

PDL misordering is an intermittent race that GPU numerics cannot catch
reliably, and the produce_pdl ABI stays until the fp4/fp8 provider sweeps.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-c-test-cpu")


ROOT = Path(__file__).resolve().parents[4]
LORA_MOE = ROOT / "python/sglang/srt/lora/moe"
PROVIDER = LORA_MOE / "base_gemm_provider"
KERNELS = LORA_MOE / "kernels"
EP_MOE = ROOT / "python/sglang/kernels/ops/moe/ep_moe_kernels.py"


def _source(name: str) -> str:
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


class TestPdlWiringSource(unittest.TestCase):
    def test_lora_a_to_b_pdl_has_a_complete_signal_wait_launch_chain(self):
        # The benchmarks still use the A-to-B chain of Programmatic Dependent
        # Launch (PDL), so the kernels must keep it. The serving runner does
        # not use PDL. An A-to-B edge measured no better than launch order.
        a = (KERNELS / "lora_a.py").read_text()
        b = (KERNELS / "lora_b.py").read_text()
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
        api = _source("cutedsl/api.py")
        sm100 = _source("cutedsl/kernel_sm100.py")
        sm90 = _source("cutedsl/kernel_sm90.py")
        activation = _source("activation_delta.py")
        act = _source("fused_act.py")
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
                "_act_delta_kernel",
                "_launch_act_delta",
            ),
            (act, "_b_act_kernel", "_launch_b_act"),
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


if __name__ == "__main__":
    unittest.main()
