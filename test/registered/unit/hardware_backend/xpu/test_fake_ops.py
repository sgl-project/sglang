"""XPU graphs must initialize when optional MLA kernels are not built."""

import subprocess
import sys
import unittest
from pathlib import Path
from textwrap import dedent

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


_REGISTER_SCRIPT = dedent(
    """
    import importlib.util
    import sys
    import types
    from unittest.mock import patch

    import torch

    library = torch.library.Library("sgl_kernel", "DEF")
    library.define("fwd(Tensor output) -> ()")
    mla_enabled = sys.argv[2] == "enabled"
    if mla_enabled:
        library.define("flash_mla_decode(Tensor output) -> ()")

    runner_dependency = types.ModuleType("sglang.srt.model_executor.runner")
    runner_dependency.DecodeCudaGraphRunner = object
    runtime_dependency = types.ModuleType("sglang.srt.runtime_context")
    runtime_dependency.get_exec = lambda: None
    utils_dependency = types.ModuleType("sglang.srt.utils")
    utils_dependency.register_xpu_device_properties_for_dynamo = lambda: None
    dependencies = {
        "sglang.srt.model_executor.runner": runner_dependency,
        "sglang.srt.runtime_context": runtime_dependency,
        "sglang.srt.utils": utils_dependency,
    }
    spec = importlib.util.spec_from_file_location("xpu_graph_runner_test", sys.argv[1])
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, dependencies):
        spec.loader.exec_module(module)

    module.register_fake_ops()
    module.register_fake_ops()
    assert torch._C._dispatch_has_kernel_for_dispatch_key("sgl_kernel::fwd", "Meta")
    if mla_enabled:
        assert torch._C._dispatch_has_kernel_for_dispatch_key(
            "sgl_kernel::flash_mla_decode", "Meta"
        )
    else:
        assert "sgl_kernel::flash_mla_decode" not in torch._C._dispatch_get_all_op_names()
    """
)


class TestXpuFakeOps(CustomTestCase):
    def _check_registration(self, mla_mode):
        source = (
            Path(__file__).resolve().parents[5]
            / "python/sglang/srt/hardware_backend/xpu/graph_runner/xpu_graph_runner.py"
        )
        result = subprocess.run(
            [sys.executable, "-c", _REGISTER_SCRIPT, str(source), mla_mode],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_mla_disabled(self):
        self._check_registration("disabled")

    def test_mla_enabled(self):
        self._check_registration("enabled")


if __name__ == "__main__":
    unittest.main()
