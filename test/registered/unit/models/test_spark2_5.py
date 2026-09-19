"""CPU tests for Spark2.5 attention output-gate activation selection."""

import pytest
import torch

from sglang.srt.configs.spark2_5 import Spark2_5Config
from sglang.srt.models.spark2_5 import _apply_gate_activation
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_config_preserves_gate_activation_mode():
    assert Spark2_5Config(gate_attn_act_mode="silu").gate_attn_act_mode == "silu"


def test_gate_activation_sigmoid_matches_reference():
    gate = torch.tensor([-2.0, 0.0, 2.0], dtype=torch.float16)

    actual = _apply_gate_activation(gate, "sigmoid")

    torch.testing.assert_close(actual, torch.sigmoid(gate.float()))


def test_gate_activation_silu_matches_reference():
    gate = torch.tensor([-2.0, 0.0, 2.0], dtype=torch.float16)

    actual = _apply_gate_activation(gate, "silu")

    torch.testing.assert_close(actual, torch.nn.functional.silu(gate.float()))


def test_gate_activation_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unsupported gate_attn_act_mode: gelu"):
        _apply_gate_activation(torch.ones(1), "gelu")
