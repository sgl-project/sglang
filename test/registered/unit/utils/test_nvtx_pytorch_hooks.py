"""CPU-only tests for the SRT NVTX forward hooks."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.utils import nvtx_pytorch_hooks
from sglang.srt.utils.nvtx_pytorch_hooks import PytHooks
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _ExplodingModule(torch.nn.Module):
    def forward(self, value):
        raise RuntimeError("forward failed")


class TestPytHooks(CustomTestCase):
    def test_forward_pushes_and_pops_each_registered_module(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.ReLU())
        PytHooks().register_hooks(model)

        with (
            patch.object(nvtx_pytorch_hooks.nvtx, "range_push") as range_push,
            patch.object(nvtx_pytorch_hooks.nvtx, "range_pop") as range_pop,
        ):
            model(torch.ones(1, 2))

        self.assertEqual(range_push.call_count, 3)
        self.assertEqual(range_pop.call_count, 3)

    def test_forward_exception_still_pops_nvtx_range(self):
        """A failed forward must not leave a half-open NVTX range."""
        model = _ExplodingModule()
        PytHooks().register_hooks(model)

        with (
            patch.object(nvtx_pytorch_hooks.nvtx, "range_push") as range_push,
            patch.object(nvtx_pytorch_hooks.nvtx, "range_pop") as range_pop,
            self.assertRaisesRegex(RuntimeError, "forward failed"),
        ):
            model(torch.ones(1))

        range_push.assert_called_once()
        range_pop.assert_called_once()

    def test_dropout_modules_remain_unregistered(self):
        dropout = torch.nn.Dropout()
        model = torch.nn.Sequential(dropout)
        hooks = PytHooks()

        hooks.register_hooks(model)

        self.assertIn(model, hooks.module_to_name_map)
        self.assertNotIn(dropout, hooks.module_to_name_map)


if __name__ == "__main__":
    unittest.main()
