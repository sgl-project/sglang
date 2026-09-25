"""Below Hopper the MXFP4 weight swizzle pins the triton_kernels warp count,
since no scale layout carries it there."""

import sys
import unittest
from types import ModuleType
from unittest.mock import patch

import torch

from sglang.srt.platforms.device_mixin import DeviceCapability
from sglang.srt.runtime_context import override_platform
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _fake_triton_kernels(captured: dict) -> dict:
    """The pieces of triton_kernels that _swizzle_mxfp4 imports, with the
    constraint sink replaced by `captured`."""
    root = ModuleType("triton_kernels")
    matmul_details = ModuleType("triton_kernels.matmul_details")
    opt_flags = ModuleType("triton_kernels.matmul_details.opt_flags")
    opt_flags.update_opt_flags_constraints = captured.update
    numerics = ModuleType("triton_kernels.numerics")
    numerics.InFlexData = lambda: None
    tensor = ModuleType("triton_kernels.tensor")
    tensor.FP4 = object()
    tensor.convert_layout = lambda t, layout, **_: t
    tensor.wrap_torch_tensor = lambda t, dtype=None: t
    tensor_details = ModuleType("triton_kernels.tensor_details")
    layout = ModuleType("triton_kernels.tensor_details.layout")
    layout.make_default_matmul_mxfp4_w_layout = lambda mx_axis: None
    layout.make_default_matmul_mxfp4_w_scale_layout = lambda mx_axis, num_warps: None
    root.matmul_details = matmul_details
    matmul_details.opt_flags = opt_flags
    root.numerics = numerics
    root.tensor = tensor
    root.tensor_details = tensor_details
    tensor_details.layout = layout
    modules = (
        root,
        matmul_details,
        opt_flags,
        numerics,
        tensor,
        tensor_details,
        layout,
    )
    return {m.__name__: m for m in modules}


def _constraints_under(capability, **facts) -> dict:
    from sglang.srt.layers.quantization.mxfp4 import _swizzle_mxfp4

    captured = {}
    quant = torch.zeros((8, 64, 32), dtype=torch.uint8)
    scale = torch.zeros((8, 64, 2), dtype=torch.uint8)
    platform = dict(
        is_cuda=True,
        is_sm100=False,
        is_sm90=False,
        device_capability=DeviceCapability(*capability),
    )
    platform.update(facts)
    with (
        patch.dict(sys.modules, _fake_triton_kernels(captured)),
        override_platform(**platform),
    ):
        _swizzle_mxfp4(quant, scale, num_warps=8)
    return captured


class TestMxfp4SwizzleNumWarps(CustomTestCase):
    def test_pre_hopper_pins_the_warp_count(self):
        self.assertEqual(_constraints_under((8, 9)), {"num_warps": 8})
        self.assertEqual(_constraints_under((8, 0)), {"num_warps": 8})

    def test_hopper_keeps_its_own_constraints(self):
        self.assertEqual(_constraints_under((9, 0), is_sm90=True), {"split_k": 1})


if __name__ == "__main__":
    unittest.main()
