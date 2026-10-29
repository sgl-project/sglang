"""CPU regressions for shared state passed to the offloader's functional_call."""

import unittest

import torch
from torch.func import functional_call

from sglang.srt.utils.offloader import _get_offloaded_device_state
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _AliasedProjection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.tensor([[2.0, 0.0], [0.0, 3.0]]))
        self.cached_proj = torch.nn.Linear(2, 2, bias=False)
        self.cached_proj.weight = self.proj.weight

    def forward(self, x):
        return self.proj(x) + self.cached_proj(x)


class _AliasedBuffer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("scale", torch.tensor([2.0, 3.0]))
        self.child = torch.nn.Module()
        self.child.register_buffer("scale", self.scale)

    def forward(self, x):
        return x * self.scale + self.child.scale


class _IndependentParameters(torch.nn.Module):
    def __init__(self, share_storage):
        super().__init__()
        values = torch.tensor([2.0, 3.0])
        self.weight = torch.nn.Parameter(values)
        column = values if share_storage else values.clone()
        self.column = torch.nn.Parameter(column.view(2, 1))

    def forward(self, x):
        return x * self.weight + self.column.sum(dim=0)


class TestOffloadedDeviceState(CustomTestCase):
    def test_shared_parameter_aliases(self):
        """Aliased projections must survive functional_call's tied-weight check."""
        module = _AliasedProjection()
        original = module.proj.weight
        state = _get_offloaded_device_state(module, torch.device("cpu"))

        self.assertEqual(set(state), {"proj.weight", "cached_proj.weight"})
        output = functional_call(
            module, state, (torch.tensor([5.0, 7.0]),), strict=True
        )

        torch.testing.assert_close(output, torch.tensor([20.0, 42.0]))
        self.assertIs(state["proj.weight"], state["cached_proj.weight"])
        self.assertIs(module.proj.weight, original)
        self.assertIs(module.cached_proj.weight, original)

    def test_shared_buffer_aliases(self):
        """Shared registered buffers also participate in the tied-state check."""
        module = _AliasedBuffer()
        original = module.scale
        state = _get_offloaded_device_state(module, torch.device("cpu"))

        self.assertEqual(set(state), {"scale", "child.scale"})
        output = functional_call(
            module, state, (torch.tensor([5.0, 7.0]),), strict=True
        )

        torch.testing.assert_close(output, torch.tensor([12.0, 24.0]))
        self.assertIs(state["scale"], state["child.scale"])
        self.assertIs(module.scale, original)
        self.assertIs(module.child.scale, original)

    def test_distinct_parameters_keep_their_shapes(self):
        """Equal values or a shared storage pointer must preserve distinct views."""
        for share_storage in (False, True):
            with self.subTest(share_storage=share_storage):
                module = _IndependentParameters(share_storage)
                state = _get_offloaded_device_state(module, torch.device("cpu"))

                self.assertEqual(set(state), {"weight", "column"})
                self.assertIsNot(state["weight"], state["column"])
                self.assertEqual(state["weight"].shape, (2,))
                self.assertEqual(state["column"].shape, (2, 1))
                output = functional_call(
                    module, state, (torch.tensor([5.0, 7.0]),), strict=True
                )

                torch.testing.assert_close(output, torch.tensor([15.0, 26.0]))


if __name__ == "__main__":
    unittest.main()
