"""Nemotron Shortcut's soft-clamped squared-ReLU experts: runner selection and
routing."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.moe.moe_runner.flashinfer_cutedsl import (  # noqa: E402
    _cutedsl_wrapper_activation_type,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend  # noqa: E402
from sglang.srt.models.nemotron_shortcut import (  # noqa: E402
    _check_clamped_relu2_backend,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_ACTIVATIONS = SimpleNamespace(
    Swiglu="Swiglu", Relu2="Relu2", ClampedRelu2="ClampedRelu2"
)


def _check_backend(backend: MoeRunnerBackend, cutedsl_supported: bool = True):
    with (
        patch(
            "sglang.srt.models.nemotron_shortcut.get_moe_runner_backend",
            return_value=backend,
        ),
        patch(
            "sglang.srt.layers.moe.moe_runner.flashinfer_cutedsl."
            "cutedsl_supports_clamped_relu2",
            return_value=cutedsl_supported,
        ),
    ):
        _check_clamped_relu2_backend()


class TestExpertActivation(CustomTestCase):
    def test_clamp_limit_selects_clamped_relu2(self):
        pick = _cutedsl_wrapper_activation_type
        self.assertEqual(pick("relu2", _ACTIVATIONS, 16.0), "ClampedRelu2")
        # Without a limit the plain squared ReLU stays plain.
        self.assertEqual(pick("relu2", _ACTIVATIONS, None), "Relu2")
        self.assertEqual(pick("silu", _ACTIVATIONS, None), "Swiglu")
        with self.assertRaises(ValueError):
            pick("gelu", _ACTIVATIONS, None)

    def test_only_runners_with_the_soft_clamp_are_accepted(self):
        _check_backend(MoeRunnerBackend.MARLIN)
        _check_backend(MoeRunnerBackend.FLASHINFER_CUTEDSL)
        with self.assertRaises(ValueError):
            _check_backend(MoeRunnerBackend.FLASHINFER_CUTEDSL, cutedsl_supported=False)
        with self.assertRaises(ValueError):
            _check_backend(MoeRunnerBackend.FLASHINFER_TRTLLM)

    def test_triton_serves_the_bf16_mtp_experts(self):
        _check_backend(MoeRunnerBackend.TRITON)


if __name__ == "__main__":
    unittest.main()
