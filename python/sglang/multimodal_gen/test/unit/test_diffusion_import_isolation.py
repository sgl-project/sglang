"""Import smoke tests for the diffusion Torch path."""

import os
import subprocess
import sys
import unittest

from sglang.test.test_utils import CustomTestCase


class TestDiffusionImportIsolation(CustomTestCase):
    def test_linear_and_model_loader_import_independently(self):
        script = """
import importlib
import sys
importlib.import_module(sys.argv[1])
from sglang.multimodal_gen.runtime.layers.quantization import get_quantization_config
from sglang.multimodal_gen.runtime.layers.quantization.auto_round import AutoRoundConfig
assert get_quantization_config("auto-round") is AutoRoundConfig
"""
        for module in (
            "sglang.multimodal_gen.runtime.layers.linear",
            "sglang.multimodal_gen.runtime.utils.hf_diffusers_utils",
        ):
            with self.subTest(module=module):
                completed = subprocess.run(
                    [sys.executable, "-c", script, module],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                self.assertEqual(
                    completed.returncode,
                    0,
                    msg=f"stdout={completed.stdout}\nstderr={completed.stderr}",
                )

    def test_disabled_backend_does_not_import_mlx(self):
        """Diffusion modules must remain usable without the optional MLX path."""
        script = """
import sys
from sglang.kernels.ops.diffusion import norm_infer
from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm
assert norm_infer is not None and RMSNorm is not None
assert not any(name == "mlx" or name.startswith("mlx.") for name in sys.modules)
"""
        env = os.environ.copy()
        env.pop("SGLANG_USE_MLX", None)
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            env=env,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout={completed.stdout}\nstderr={completed.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
