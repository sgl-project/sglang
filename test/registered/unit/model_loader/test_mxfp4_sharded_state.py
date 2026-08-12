"""Regression test for MXFP4 Triton-kernel sharded state exports."""

import sys
import tempfile
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import safe_open

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

maybe_stub_sgl_kernel()

import sglang.srt.layers.quantization.mxfp4 as mxfp4_module  # noqa: E402
import sglang.srt.model_loader.loader as loader_module  # noqa: E402
from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod  # noqa: E402
from sglang.srt.model_loader.loader import ShardedStateLoader  # noqa: E402

_RUNTIME_STATE_NAMES = (
    "w13_weight",
    "w13_weight_scale",
    "w2_weight",
    "w2_weight_scale",
)


def _new_method():
    method = Mxfp4MoEMethod.__new__(Mxfp4MoEMethod)
    method.use_marlin = False
    method.use_deep_gemm = False
    method._fi_kernel = None
    method.use_flashinfer = False
    method.use_triton_kernels = True
    return method


class _TinyMxfp4Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        for name, shape, value in (
            ("w13_weight", (1, 4, 3), 17),
            ("w13_weight_scale", (1, 4, 2), 113),
            ("w2_weight", (1, 3, 2), 18),
            ("w2_weight_scale", (1, 3, 2), 114),
        ):
            parameter = torch.nn.Parameter(
                torch.full(shape, value, dtype=torch.uint8), requires_grad=False
            )
            parameter.test_marker = name
            self.register_parameter(name, parameter)
        self.w13_weight_bias = torch.nn.Parameter(
            torch.full((1, 4), 3, dtype=torch.bfloat16), requires_grad=False
        )
        self.w2_weight_bias = torch.nn.Parameter(
            torch.full((1, 3), 4, dtype=torch.bfloat16), requires_grad=False
        )


class _WrappedTensor:
    def __init__(self, data):
        self.storage = SimpleNamespace(data=data)

    @property
    def data(self):
        return self.storage.data


def _fake_swizzle(weight, scale, _num_warps):
    def wrap(tensor):
        data = tensor.detach().clone().transpose(-2, -1)
        return _WrappedTensor(data)

    return wrap(weight), object(), wrap(scale)


class TestMxfp4ShardedState(CustomTestCase):
    def test_runtime_weight_and_scale_round_trip(self):
        triton_kernels = ModuleType("triton_kernels")
        triton_kernels.__path__ = []
        triton_matmul = ModuleType("triton_kernels.matmul")
        triton_matmul.FlexCtx = SimpleNamespace
        triton_matmul.PrecisionConfig = SimpleNamespace

        layer = _TinyMxfp4Layer()
        method = _new_method()
        original_parameters = {
            name: getattr(layer, name) for name in _RUNTIME_STATE_NAMES
        }

        with patch.dict(
            sys.modules,
            {
                "triton_kernels": triton_kernels,
                "triton_kernels.matmul": triton_matmul,
            },
        ), patch.object(
            mxfp4_module, "_swizzle_mxfp4", side_effect=_fake_swizzle
        ), patch.object(
            mxfp4_module, "_use_aiter", False
        ), patch.object(
            torch.cuda, "empty_cache"
        ):
            method.process_weights_after_loading(layer)

        runtime_tensors = {
            "w13_weight": method.w13_weight_triton_tensor,
            "w13_weight_scale": method.w13_precision_config.b_mx_scale,
            "w2_weight": method.w2_weight_triton_tensor,
            "w2_weight_scale": method.w2_precision_config.b_mx_scale,
        }
        state = layer.state_dict()
        for value, name in enumerate(_RUNTIME_STATE_NAMES, start=1):
            parameter = getattr(layer, name)
            runtime_tensor = runtime_tensors[name]
            self.assertIs(parameter, original_parameters[name])
            self.assertEqual(parameter.test_marker, name)
            self.assertIn(name, state)
            self.assertFalse(parameter.is_contiguous())
            self.assertIs(runtime_tensor.storage.data, parameter)
            self.assertEqual(
                runtime_tensor.storage.data.data_ptr(), parameter.data_ptr()
            )

            # This is the copy performed by ShardedStateLoader.load_model.
            incoming = torch.full_like(state[name], value)
            state[name].copy_(incoming)
            self.assertTrue(torch.equal(runtime_tensor.storage.data, incoming))

        with tempfile.TemporaryDirectory() as output_dir, patch.object(
            loader_module, "get_parallel", return_value=SimpleNamespace(tp_rank=0)
        ):
            ShardedStateLoader.save_model(layer, output_dir)
            checkpoint = f"{output_dir}/model-rank-0-part-0.safetensors"
            with safe_open(checkpoint, framework="pt") as handle:
                self.assertEqual(set(handle.keys()), set(state))
                for name, expected in state.items():
                    self.assertTrue(
                        torch.equal(handle.get_tensor(name), expected), name
                    )


if __name__ == "__main__":
    unittest.main()
