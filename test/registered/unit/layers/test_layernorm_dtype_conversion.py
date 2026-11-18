"""LayerNorm follows its parameters after a module dtype conversion."""

import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn

from sglang.kernels.spec import KernelBackend
from sglang.srt.layers import layernorm
from sglang.srt.layers.layernorm import LayerNorm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def make_norm(dtype=torch.float32, *, bias=True):
    norm = LayerNorm(8, eps=1e-5, bias=bias, dtype=dtype)
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(0.5, 1.4, 8, dtype=dtype))
        norm.bias.copy_(torch.linspace(-0.2, 0.3, 8, dtype=dtype))
    return norm


def reference_norm(norm):
    reference = nn.LayerNorm(
        8, eps=norm.variance_epsilon, bias=norm.use_bias, dtype=norm.weight.dtype
    )
    with torch.no_grad():
        reference.weight.copy_(norm.weight)
        if norm.use_bias:
            reference.bias.copy_(norm.bias)
    return reference


def data(dtype):
    return torch.tensor(
        [
            [-1.2, 0.3, 0.8, 1.7, -0.4, 0.6, -0.9, 1.1],
            [0.7, -0.2, 1.8, 0.4, -1.1, 0.2, 0.9, -0.6],
        ],
        dtype=dtype,
    )


class Projection(nn.Module):
    def __init__(self, norm):
        super().__init__()
        self.input = nn.Linear(8, 8)
        self.norm = norm
        self.output = nn.Linear(8, 3)

    def forward(self, x):
        x = self.input(x)
        if isinstance(self.norm, LayerNorm):
            x = self.norm(x, backend=KernelBackend.TORCH)
        else:
            x = self.norm(x)
        return self.output(x)


class TestLayerNormDtypeConversion(CustomTestCase):
    def test_standard_module_conversions(self):
        for conversion in ("half", "bfloat16", "double", "float"):
            with self.subTest(conversion=conversion):
                norm = make_norm(
                    torch.float64 if conversion == "float" else torch.float32
                )
                getattr(norm, conversion)()
                inputs = data(norm.weight.dtype)
                reference = reference_norm(norm)
                actual = norm(inputs, backend=KernelBackend.TORCH)
                torch.testing.assert_close(actual, reference(inputs), rtol=0, atol=0)
                self.assertEqual(actual.dtype, inputs.dtype)

    def test_mixed_input_dtype_after_conversion(self):
        for parameter_dtype, input_dtype in (
            (torch.float16, torch.float32),
            (torch.bfloat16, torch.float32),
            (torch.float64, torch.float32),
            (torch.float32, torch.bfloat16),
        ):
            with self.subTest(parameters=parameter_dtype, inputs=input_dtype):
                norm = make_norm(torch.float64).to(parameter_dtype)
                inputs = data(input_dtype)
                before = inputs.clone()
                expected = reference_norm(norm)(inputs.to(parameter_dtype)).to(
                    input_dtype
                )
                actual = norm.forward_native(inputs)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(inputs, before, rtol=0, atol=0)
                self.assertEqual(actual.dtype, input_dtype)

    def test_unconverted_float32_norm_keeps_mixed_precision_behavior(self):
        norm = make_norm()
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                inputs = data(dtype)
                expected = reference_norm(norm)(inputs.float()).to(dtype)
                torch.testing.assert_close(
                    norm.forward_native(inputs), expected, rtol=0, atol=0
                )

    def test_converted_biasless_norm(self):
        norm = make_norm(bias=False).double()
        inputs = data(torch.float64)
        torch.testing.assert_close(
            norm.forward_native(inputs), reference_norm(norm)(inputs), rtol=0, atol=0
        )

    def test_checkpoint_assignment_uses_assigned_parameter_dtype(self):
        saved = make_norm().double().state_dict()
        norm = LayerNorm(8, eps=1e-5)
        norm.load_state_dict(saved, assign=True)
        self.assertEqual(norm.weight.dtype, torch.float64)
        inputs = data(torch.float64)
        torch.testing.assert_close(
            norm.forward_native(inputs), reference_norm(norm)(inputs), rtol=0, atol=0
        )

    def test_three_native_training_updates_match_pytorch(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float64):
            with self.subTest(dtype=dtype):
                torch.manual_seed(19)
                model = Projection(make_norm()).to(dtype)
                reference = copy.deepcopy(model)
                reference.norm = reference_norm(model.norm)
                optimizers = [
                    torch.optim.SGD(model.parameters(), lr=0.02),
                    torch.optim.SGD(reference.parameters(), lr=0.02),
                ]
                inputs = data(dtype)
                target = torch.tensor([[0.2, -0.4, 0.6], [-0.1, 0.5, 0.3]])
                before = model.norm.weight.detach().clone()
                for _ in range(3):
                    actual = model(inputs)
                    expected = reference(inputs)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    for output, optimizer in zip((actual, expected), optimizers):
                        optimizer.zero_grad()
                        (output.float() - target).square().mean().backward()
                    expected_parameters = dict(reference.named_parameters())
                    for name, parameter in model.named_parameters():
                        torch.testing.assert_close(
                            parameter.grad,
                            expected_parameters[name].grad,
                            rtol=0,
                            atol=0,
                        )
                    for optimizer in optimizers:
                        optimizer.step()
                    for name, parameter in model.named_parameters():
                        torch.testing.assert_close(
                            parameter, expected_parameters[name], rtol=0, atol=0
                        )
                self.assertFalse(torch.equal(model.norm.weight, before))

    def test_cuda_gate_rejects_converted_non_float32_parameters(self):
        # Only dispatch eligibility is exercised on CPU; no fused GPU op runs.
        with patch.object(layernorm, "_flashinfer_layernorm_available", True):
            for dtype in (torch.float16, torch.bfloat16, torch.float64):
                with self.subTest(dtype=dtype):
                    norm = make_norm().to(dtype)
                    inputs = data(torch.bfloat16)
                    expected = reference_norm(norm)(inputs.to(dtype)).to(inputs.dtype)
                    torch.testing.assert_close(
                        norm.forward_cuda(inputs), expected, rtol=0, atol=0
                    )

    def test_hip_gate_rejects_parameters_converted_away_from_input_dtype(self):
        # An old constructor dtype must not make incompatible operands eligible.
        with patch.object(layernorm, "_has_aiter_layer_norm", True):
            norm = make_norm(torch.float16).float()
            inputs = data(torch.float16)
            expected = reference_norm(norm)(inputs.float()).half()
            torch.testing.assert_close(
                norm.forward_hip(inputs), expected, rtol=0, atol=0
            )


if __name__ == "__main__":
    unittest.main()
