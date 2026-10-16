"""Regress SM80 stream device ownership and reduction architecture isolation.

Run detached repository function ASTs so these policy regressions need no CUDA
installation. Kernels are not replaced in numerical tests; this suite only checks
resource ownership and dispatch contracts.
"""

import ast
import runpy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
register_cpu_ci = runpy.run_path(
    str(ROOT / "python/sglang/test/ci/ci_register.py")
)["register_cpu_ci"]
register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def load_function(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class TestSM80ResourcePolicy(unittest.TestCase):
    def test_streams_follow_device_capture_and_context_reset(self):
        """A second device must not inherit the first device's CUDA stream."""
        resources = NS(streams={})
        state = NS(capturing=False)
        cuda = NS(
            current_device=lambda: 0,
            is_current_stream_capturing=lambda: state.capturing,
            Stream=lambda device: NS(device=device),
        )
        torch = NS(cuda=cuda, device=lambda value: NS(index=int(value.split(":")[1])))
        path = ROOT / "python/sglang/kernels/ops/attention/dsv4/sparse_attention_sm80_tiled.py"
        get_stream = load_function(
            path, "_partial_side_stream", {"torch": torch, "_PARTIAL_SIDE_STREAM": None}
        )
        runtime = NS(get_resources=lambda: resources)
        with patch.dict(sys.modules, {"sglang.srt.runtime_context": runtime}):
            first = get_stream("cuda:0")
            second = get_stream("cuda:1")
            self.assertIsNot(first, second)
            self.assertEqual(first.device.index, 0)
            self.assertEqual(second.device.index, 1)
            state.capturing = True
            self.assertIs(get_stream("cuda:0"), first)
            self.assertIsNone(get_stream("cuda:2"))
            resources.streams.clear()  # reset_context installs a fresh resource tier.
            self.assertIsNone(get_stream("cuda:0"))
            state.capturing = False
            self.assertIsNot(get_stream("cuda:0"), first)

    def test_tiled_decode_stays_opt_in_and_rejects_junk_thresholds(self):
        """Batched decode needs a real positive threshold; prefill must stay serial."""
        path = ROOT / "python/sglang/kernels/ops/attention/dsv4/flash_mla_sm80.py"
        use_tiled = load_function(path, "_use_tiled_decode", {})
        # Single-row decode is unconditional, exactly as before the batched change.
        for raw in (None, "", "0", "abc", "-4", 0):
            with self.subTest(raw=raw):
                self.assertTrue(use_tiled(1, raw))
        # An unset or malformed threshold must not widen the capture set.
        for raw in (None, "", "0", "abc", "true", "-1", " "):
            with self.subTest(raw=raw):
                self.assertFalse(use_tiled(2, raw))
                self.assertFalse(use_tiled(4096, raw))
        # Official batches and prefill chunks sit on opposite sides of the opt-in.
        self.assertTrue(use_tiled(8, "8"))
        self.assertFalse(use_tiled(9, "8"))
        self.assertFalse(use_tiled(2048, "8"))
        self.assertTrue(use_tiled(8, 8))
        self.assertFalse(use_tiled(0, "8"))

    def test_reduce_tile_stays_on_qualified_sm80_non_invariant_shape(self):
        """SM80 flags on Hopper must not switch its reduction implementation."""
        path = ROOT / "python/sglang/kernels/ops/layernorm/mhc.py"
        tree = ast.parse(path.read_text())
        # Evaluate the production dispatch condition, not a second implementation.
        gate = next(
            node.test for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and any(
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Subscript)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "_hc_mix_reduce_sinkhorn_kernel_tile"
                for stmt in node.body for call in ast.walk(stmt)
            )
        )
        expression = compile(ast.Expression(body=gate), str(path), "eval")
        flags = NS(**{
            name: NS(get=lambda: "1") for name in (
                "SGLANG_DSV41_SM80_EXPERIMENTAL",
                "SGLANG_DSV41_SM80_MHC_REDUCE_TILE",
            )
        })
        for capability, invariant, m, expected in (
            ((8, 0), False, 1, True),
            ((9, 0), False, 1, False),
            ((10, 0), False, 1, False),
            ((8, 6), False, 1, False),
            ((8, 0), True, 1, False),
            ((8, 0), False, 2, False),
        ):
            with self.subTest(capability=capability, invariant=invariant, m=m):
                namespace = dict(
                    envs=flags,
                    os=NS(environ={
                        "SGLANG_DSV41_SM80_EXPERIMENTAL": "1",
                        "SGLANG_DSV41_SM80_MHC_REDUCE_TILE": "1",
                    }),
                    torch=NS(
                        version=NS(hip=None), bfloat16="bf16",
                        cuda=NS(get_device_capability=lambda device: capability),
                    ),
                    x_flat=NS(is_cuda=True, dtype="bf16", device="cuda:0"),
                    is_batch_invariant_mode_enabled=lambda: invariant,
                    m=m, k=20480, mix=24, hc_mult=4, num_slices=80,
                )
                self.assertEqual(eval(expression, namespace), expected)


if __name__ == "__main__":
    unittest.main()
