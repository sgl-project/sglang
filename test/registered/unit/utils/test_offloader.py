"""Unit tests for OffloaderV2 parameter offloaders — no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from unittest import mock

import torch
from torch import nn

from sglang.srt.utils import offloader
from sglang.test.test_utils import CustomTestCase

_real_tensor_to = torch.Tensor.to
_real_move_param_to_cpu = offloader._move_param_to_cpu


def _fake_h2d(self, *args, **kwargs):
    # Stands in for the host-to-device copy on CPU-only runners: like the real
    # copy, it returns a new tensor that carries none of the source's attributes.
    if args and args[0] == "cuda":
        return _real_tensor_to(self.detach(), "cpu", copy=True)
    return _real_tensor_to(self, *args, **kwargs)


class TestCpuParamOffloader(CustomTestCase):
    def _make_offloader(self, is_shuffled):
        module = nn.Module()
        module.w13_weight = nn.Parameter(
            torch.arange(8, dtype=torch.float32).reshape(2, 4), requires_grad=False
        )
        if is_shuffled is not None:
            module.w13_weight.is_shuffled = is_shuffled
        # Pinned memory needs an accelerator.
        with mock.patch.object(
            offloader,
            "_move_param_to_cpu",
            lambda param, pin_memory: _real_move_param_to_cpu(param, pin_memory=False),
        ):
            param_offloader = offloader._CpuParamOffloader(module, "w13_weight")
        return module, param_offloader

    def _create_device_tensor(self, param_offloader):
        with mock.patch.object(torch.Tensor, "to", _fake_h2d):
            return param_offloader.create_device_tensor()

    def test_device_tensor_keeps_is_shuffled(self):
        module, param_offloader = self._make_offloader(is_shuffled=True)

        output = self._create_device_tensor(param_offloader)

        self.assertIsNot(output, module.w13_weight)
        self.assertTrue(getattr(output, "is_shuffled", False))
        torch.testing.assert_close(output, module.w13_weight.data)

    def test_device_tensor_without_is_shuffled(self):
        for is_shuffled in (None, False):
            with self.subTest(is_shuffled=is_shuffled):
                _, param_offloader = self._make_offloader(is_shuffled=is_shuffled)

                output = self._create_device_tensor(param_offloader)

                self.assertFalse(getattr(output, "is_shuffled", False))


if __name__ == "__main__":
    unittest.main()
