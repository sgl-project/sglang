"""Unit tests for bugfix #26269: tensor dump produces no .pt files when
submodules are called via .forward() instead of __call__().

The fix changes TensorDumper to monkey-patch module.forward so hooks
fire regardless of how the module is invoked.
"""

import os
import tempfile
import unittest

import torch
import torch.nn as nn

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.debug_utils.tensor_dump_forward_hook import (  # noqa: E402
    register_forward_hook_for_model,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class SimpleModel(nn.Module):
    """Minimal model matching the XxxxForCausalLM -> model -> layers layout."""

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(2)])
        self.lm_head = nn.Linear(4, 8)

    def forward(self, x):
        for layer in self.model.layers:
            x = layer.forward(x)
        return self.lm_head.forward(x)


class TestTensorDumpDirectForwardCall(CustomTestCase):
    """Test that tensor dump hooks fire when modules use .forward() directly."""

    def test_dump_with_direct_forward_call(self):
        """Hooks should fire and produce .pt files even when .forward() is called directly."""
        model = SimpleModel()
        with tempfile.TemporaryDirectory() as tmpdir:
            dumper = register_forward_hook_for_model(
                model,
                dump_dir=tmpdir,
                dump_layers=None,
                tp_size=1,
                tp_rank=0,
                pp_rank=0,
            )
            x = torch.randn(2, 4)
            model.forward(x)
            dumper.dump_current_tensors()

            pt_files = [
                f for f in os.listdir(dumper.get_dump_dir()) if f.endswith(".pt")
            ]
            self.assertGreater(len(pt_files), 0, "No .pt files produced")

    def test_dump_with_call_operator(self):
        """Hooks should also work when using __call__ (the normal path)."""
        model = SimpleModel()
        with tempfile.TemporaryDirectory() as tmpdir:
            dumper = register_forward_hook_for_model(
                model,
                dump_dir=tmpdir,
                dump_layers=None,
                tp_size=1,
                tp_rank=0,
                pp_rank=0,
            )
            x = torch.randn(2, 4)
            model(x)
            dumper.dump_current_tensors()

            pt_files = [
                f for f in os.listdir(dumper.get_dump_dir()) if f.endswith(".pt")
            ]
            self.assertGreater(len(pt_files), 0, "No .pt files produced")

    def test_forward_method_is_patched(self):
        """After hook registration, module.forward should be wrapped."""
        model = SimpleModel()
        original_forward = model.model.layers[0].forward

        with tempfile.TemporaryDirectory() as tmpdir:
            register_forward_hook_for_model(
                model,
                dump_dir=tmpdir,
                dump_layers=None,
                tp_size=1,
                tp_rank=0,
                pp_rank=0,
            )

        patched_forward = model.model.layers[0].forward
        self.assertIsNot(patched_forward, original_forward)

    def test_output_correctness_with_hooks(self):
        """Model output should be unchanged after hook registration."""
        model = SimpleModel()
        x = torch.randn(2, 4)
        expected = model(x).detach().clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            register_forward_hook_for_model(
                model,
                dump_dir=tmpdir,
                dump_layers=None,
                tp_size=1,
                tp_rank=0,
                pp_rank=0,
            )
            actual = model.forward(x).detach()

        self.assertTrue(
            torch.allclose(expected, actual, atol=1e-6),
            "Model output changed after hook registration",
        )


if __name__ == "__main__":
    unittest.main()
