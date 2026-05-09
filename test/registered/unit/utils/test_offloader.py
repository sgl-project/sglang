"""Unit tests for srt/utils/offloader.py -- no server, no model loading."""

import unittest

import torch
from torch import nn
from torch.func import functional_call

from sglang.srt.utils.offloader import _make_functional_call_state
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class _ChildWithTiedParameter(nn.Module):
    def __init__(self, shared_weight: nn.Parameter):
        super().__init__()
        self.A_log = shared_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.A_log


class _ModuleWithTiedParameter(nn.Module):
    def __init__(self):
        super().__init__()
        self.A_log = nn.Parameter(torch.tensor(2.0))
        self.attn = _ChildWithTiedParameter(self.A_log)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(x) + self.A_log


class TestOffloaderFunctionalCallState(CustomTestCase):
    def test_tied_parameter_aliases_are_deduplicated(self):
        module = _ModuleWithTiedParameter()
        raw_state = {k: v.to("cpu") for k, v in module.state_dict().items()}

        self.assertEqual(["A_log", "attn.A_log"], list(raw_state.keys()))
        with self.assertRaisesRegex(ValueError, "multiple values.*tied"):
            functional_call(module, raw_state, args=(torch.tensor(3.0),))

        device_state = _make_functional_call_state(module, torch.device("cpu"))

        self.assertEqual(["A_log"], list(device_state.keys()))
        self.assertEqual(
            torch.tensor(8.0),
            functional_call(module, device_state, args=(torch.tensor(3.0),)),
        )


if __name__ == "__main__":
    unittest.main()
