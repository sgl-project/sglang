"""Declared boolean switches are read through `envs` in the quantization and
rotary-embedding layers.

`EnvBool` accepts yes/y; `get_bool_env_var` does not. A switch read through both
turns a backend on in one module and leaves it off in the next.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import sglang
from sglang.srt.environ import envs
from sglang.srt.layers.quantization import fp8, fp8_activation
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
from sglang.test.test_utils import CustomTestCase

_LAYERS = Path(next(iter(sglang.__path__))) / "srt" / "layers"
_SCANNED = (_LAYERS / "quantization", _LAYERS / "rotary_embedding")
_SWITCHES = ("SGLANG_USE_AITER", "SGLANG_FORCE_FP8_MARLIN")


def _raw_reads(path: Path) -> list[str]:
    found = []
    for node in ast.walk(ast.parse(path.read_text())):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "get_bool_env_var"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value in _SWITCHES
        ):
            continue
        found.append(f"{path.relative_to(_LAYERS)}:{node.lineno}: {node.args[0].value}")
    return found


class TestDeclaredEnvReads(CustomTestCase):
    def test_switches_are_not_read_with_get_bool_env_var(self):
        raw = [
            read
            for root in _SCANNED
            for path in sorted(root.rglob("*.py"))
            for read in _raw_reads(path)
        ]
        self.assertEqual(raw, [])

    def test_yes_enables_aiter_for_the_dense_silu_fusion(self):
        down_proj = SimpleNamespace(quant_method=None)
        with (
            envs.SGLANG_USE_AITER.override("yes"),
            patch.object(fp8_activation, "is_hip", return_value=True),
            # gfx950 and SM100 hosts go on to probe down_proj unless tp_size != 1.
            patch.object(
                fp8_activation, "get_parallel", return_value=SimpleNamespace(tp_size=2)
            ),
        ):
            self.assertTrue(fp8_activation.DenseSiluAndMulQuant(down_proj).aiter)

    def test_yes_forces_fp8_marlin(self):
        with (
            envs.SGLANG_FORCE_FP8_MARLIN.override("yes"),
            patch.object(fp8, "_is_cuda", True),
            patch.object(fp8, "can_auto_enable_marlin_fp8", return_value=False),
        ):
            method = Fp8LinearMethod(Fp8Config(is_checkpoint_fp8_serialized=True))
        self.assertTrue(method.use_marlin)


if __name__ == "__main__":
    unittest.main()
