"""CPU regression test for the Marlin weight-shape check.

Both `CompressedTensorsWNA16.process_weights_after_loading` and
`GPTQMarlinLinearKernel.process_weights_after_loading` called
`check_marlin_supports_shape`, which returns `(ok, err_msg)` and never raises,
and dropped the result. An unsupported shape therefore sailed past the check
and died later inside `gptq_marlin_repack` with a bare device-side assert
instead of the actionable "not divisible by min_thread_n = 64" message. The AWQ
Marlin scheme already uses the raising `verify_marlin_supports_shape`.

These tests drive both checks with the Gemma-4 vision MLP shape from #28018
(out_features = 2 * 4304 = 8608) and assert they now raise. They run on CPU:
the call under test happens before any device work, which is stubbed out.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from unittest import mock

import torch

from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    compressed_tensors_wNa16 as wna16,
)
from sglang.srt.layers.quantization.marlin_utils import MarlinLinearLayerConfig
from sglang.test.test_utils import CustomTestCase

IN_FEATURES = 1152
GEMMA4_VISION_GATE_UP_OUT = 2 * 4304  # 8608, 8608 % 64 == 32
SUPPORTED_OUT = 8640  # the next multiple of 64
GROUP_SIZE = 32


def _gptq_kernels():
    # Imported lazily: `hardware_backend...gptq_kernels` and the
    # compressed-tensors schemes package import each other, so this has to come
    # after the module-level import above.
    from sglang.srt.hardware_backend.gpu.quantization import gptq_kernels

    return gptq_kernels


class _ReachedDeviceWork(Exception):
    """Raised by the stub that stands in for the first post-check device call."""


def _kernel_config(out_features: int) -> MarlinLinearLayerConfig:
    return MarlinLinearLayerConfig(
        full_weight_shape=(IN_FEATURES, out_features),
        partition_weight_shape=(IN_FEATURES, out_features),
        weight_type=wna16.WNA16_SUPPORTED_TYPES_MAP[4],
        act_type=torch.bfloat16,
        group_size=GROUP_SIZE,
        zero_points=False,
        has_g_idx=False,
    )


def _fake_layer(weight_name: str) -> torch.nn.Module:
    layer = torch.nn.Module()
    layer.register_parameter(
        weight_name,
        torch.nn.Parameter(torch.empty(0, dtype=torch.int32), requires_grad=False),
    )
    return layer


def _process_wna16(out_features: int) -> None:
    """Run the WNA16 path up to the first device-side call."""
    scheme = wna16.CompressedTensorsWNA16.__new__(wna16.CompressedTensorsWNA16)
    scheme.kernel_config = _kernel_config(out_features)
    with mock.patch.object(
        wna16, "marlin_make_workspace", side_effect=_ReachedDeviceWork
    ):
        scheme.process_weights_after_loading(_fake_layer("weight_packed"))


def _process_gptq_marlin(out_features: int) -> None:
    """Run the GPTQ-Marlin path up to the first device-side call."""
    gptq_kernels = _gptq_kernels()
    kernel = gptq_kernels.GPTQMarlinLinearKernel(quant_config=None)
    kernel.kernel_config = _kernel_config(out_features)
    with mock.patch.object(
        gptq_kernels, "marlin_make_workspace", side_effect=_ReachedDeviceWork
    ):
        kernel.process_weights_after_loading(_fake_layer("qweight"))


class TestMarlinShapeCheck(CustomTestCase):
    def _assert_rejects(self, process):
        with self.assertRaises(ValueError) as ctx:
            process(GEMMA4_VISION_GATE_UP_OUT)
        self.assertIn(str(GEMMA4_VISION_GATE_UP_OUT), str(ctx.exception))
        self.assertIn("min_thread_n = 64", str(ctx.exception))

    def _assert_accepts(self, process):
        # Positive control: a divisible shape must get past the check, so the
        # rejection tests are not just rejecting everything.
        with self.assertRaises(_ReachedDeviceWork):
            process(SUPPORTED_OUT)

    def test_wna16_rejects_unsupported_out_features(self):
        self._assert_rejects(_process_wna16)

    def test_wna16_accepts_supported_out_features(self):
        self._assert_accepts(_process_wna16)

    def test_gptq_marlin_rejects_unsupported_out_features(self):
        self._assert_rejects(_process_gptq_marlin)

    def test_gptq_marlin_accepts_supported_out_features(self):
        self._assert_accepts(_process_gptq_marlin)


if __name__ == "__main__":
    unittest.main()
