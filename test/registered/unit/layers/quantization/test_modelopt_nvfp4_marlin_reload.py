"""Weight updates of ModelOpt NVFP4 layers repacked for Marlin (SM80 to SM90).

A weight update restores every layer to the checkpoint layout, loads through
each parameter's weight_loader, and repacks. These tests run the real Marlin
repack and check that restore gives back the checkpoint bytes in the same
storage, keeps the loaders, and that a layer reloaded with new weights serves
what a fresh layer with those weights serves.
"""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from sglang.srt.layers.moe import MoeRunnerBackend
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptFp4LinearMethod,
    ModelOptNvFp4A16LinearMethod,
    ModelOptNvFp4FusedMoEMethod,
)
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils.device import is_sm80_supported, is_sm90_supported
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-large")

MODELOPT = "sglang.srt.layers.quantization.modelopt_quant"
MARLIN = SimpleNamespace(is_marlin=lambda: True)


def _fp8_scales(*shape):
    return (torch.rand(shape, device="cuda") * 400).to(torch.float8_e4m3fn)


def _checkpoint_linear(n, k):
    return {
        "weight": torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda"),
        "weight_scale": _fp8_scales(n, k // 16),
        "weight_scale_2": torch.rand((), device="cuda") / 1000,
        "input_scale": torch.rand((), device="cuda"),
    }


def _ptrs(layer, names):
    return {name: getattr(layer, name).data_ptr() for name in names}


@unittest.skipIf(
    not (is_sm80_supported() or is_sm90_supported()),
    "NVFP4 Marlin runs on SM80 to SM90",
)
@patch(f"{MODELOPT}.get_fp4_gemm_runner_backend", return_value=MARLIN)
class TestModelOptNvFp4MarlinLinearReload(CustomTestCase):
    method_cls = ModelOptFp4LinearMethod

    def _make(self, n, k):
        config = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            use_per_token_activation=False,
        )
        method = self.method_cls(config)
        layer = nn.Module()
        with torch.device("cuda"):
            method.create_weights(
                layer,
                input_size_per_partition=k,
                output_partition_sizes=[n],
                input_size=k,
                output_size=n,
                params_dtype=torch.bfloat16,
                weight_loader=default_weight_loader,
            )
        return method, layer

    def _load(self, layer, checkpoint):
        for name, tensor in checkpoint.items():
            param = getattr(layer, name)
            param.weight_loader(param, tensor)

    def _check_reload(self, n, k):
        torch.manual_seed(0)
        x = torch.randn(7, k, dtype=torch.bfloat16, device="cuda") / 10
        first, second = _checkpoint_linear(n, k), _checkpoint_linear(n, k)

        method, layer = self._make(n, k)
        self._load(layer, first)
        method.process_weights_after_loading(layer)
        out_first = method.apply(layer, x)
        served = ("weight", "weight_scale", "weight_global_scale", "workspace")
        params = {name: getattr(layer, name) for name in served}
        ptrs = _ptrs(layer, served)

        # Restore gives back the checkpoint bytes in the same parameters.
        method.restore_weights_before_loading(layer)
        self.assertTrue(torch.equal(layer.weight, first["weight"]))
        self.assertTrue(
            torch.equal(
                layer.weight_scale.view(torch.uint8),
                first["weight_scale"].view(torch.uint8),
            )
        )
        self.assertIs(layer.weight.weight_loader, default_weight_loader)
        self.assertIs(layer.weight_scale.weight_loader, default_weight_loader)

        # Restore then repack is the identity on what the kernel reads.
        method.process_weights_after_loading(layer)
        self.assertTrue(torch.equal(method.apply(layer, x), out_first))
        self.assertEqual(_ptrs(layer, served), ptrs)

        # A reload with new weights serves what a fresh layer serves, from
        # the same parameters and addresses, so captured graphs stay valid.
        method.restore_weights_before_loading(layer)
        self._load(layer, second)
        method.process_weights_after_loading(layer)
        fresh_method, fresh = self._make(n, k)
        self._load(fresh, second)
        fresh_method.process_weights_after_loading(fresh)
        self.assertTrue(
            torch.equal(method.apply(layer, x), fresh_method.apply(fresh, x))
        )
        self.assertFalse(torch.equal(method.apply(layer, x), out_first))
        self.assertEqual(_ptrs(layer, served), ptrs)
        for name, param in params.items():
            self.assertIs(getattr(layer, name), param)

    def test_reload(self, _):
        self._check_reload(256, 512)

    def test_reload_padded_shape(self, _):
        # N=96 and K=160 are padded to Marlin's tile sizes.
        self._check_reload(96, 160)

    def test_reload_w4a16(self, _):
        self.method_cls = ModelOptNvFp4A16LinearMethod
        self._check_reload(256, 512)
        self._check_reload(96, 160)


@unittest.skipIf(
    not (is_sm80_supported() or is_sm90_supported()),
    "NVFP4 Marlin runs on SM80 to SM90",
)
@patch(f"{MODELOPT}._use_nvfp4_dispatch", return_value=False)
@patch(f"{MODELOPT}.get_moe_a2a_backend")
class TestModelOptNvFp4MarlinMoEReload(CustomTestCase):
    E, H = 4, 256

    def _make(self):
        method = object.__new__(ModelOptNvFp4FusedMoEMethod)
        method._moe_runner_backend = MoeRunnerBackend.MARLIN
        E, H, I, S = self.E, self.H, self.I, self.shards
        layer = nn.Module()
        layer.quant_config = SimpleNamespace(group_size=16)
        layer.moe_runner_config = SimpleNamespace(is_gated=S == 2)
        layer.params_dtype = torch.bfloat16
        layer.hidden_size = H
        layer.intermediate_size_per_partition = I
        layer.nvfp4_marlin_repacked = False
        shapes = {
            "w13_weight": ((E, S * I, H // 2), torch.uint8),
            "w2_weight": ((E, H, I // 2), torch.uint8),
            "w13_weight_scale": ((E, S * I, H // 16), torch.float8_e4m3fn),
            "w2_weight_scale": ((E, H, I // 16), torch.float8_e4m3fn),
            "w13_weight_scale_2": ((E, S) if S == 2 else (E,), torch.float32),
            "w2_weight_scale_2": ((E,), torch.float32),
        }
        for name, (shape, dtype) in shapes.items():
            param = nn.Parameter(
                torch.empty(shape, dtype=dtype, device="cuda"), requires_grad=False
            )
            param.weight_loader = default_weight_loader
            layer.register_parameter(name, param)
        return method, layer

    def _checkpoint(self):
        E, H, I, S = self.E, self.H, self.I, self.shards
        gate_scale = torch.rand(E, device="cuda") / 1000
        return {
            "w13_weight": torch.randint(0, 256, (E, S * I, H // 2), device="cuda").to(
                torch.uint8
            ),
            "w2_weight": torch.randint(0, 256, (E, H, I // 2), device="cuda").to(
                torch.uint8
            ),
            "w13_weight_scale": _fp8_scales(E, S * I, H // 16),
            "w2_weight_scale": _fp8_scales(E, H, I // 16),
            # Marlin serves the gate scale only; restore must still return the
            # up scale the checkpoint holds.
            "w13_weight_scale_2": (
                torch.stack([gate_scale, 2 * gate_scale], dim=1)
                if S == 2
                else gate_scale
            ),
            "w2_weight_scale_2": torch.rand(E, device="cuda") / 1000,
        }

    def _load(self, layer, checkpoint):
        for name, tensor in checkpoint.items():
            param = getattr(layer, name)
            param.weight_loader(param, tensor)

    def _served(self, layer):
        names = [name for name, _ in layer.named_parameters()]
        return {name: getattr(layer, name).detach().clone() for name in names}

    def test_reload(self, a2a_backend, _):
        self.I, self.shards = 128, 2
        self._check_reload(a2a_backend)

    def test_reload_non_gated_padded(self, a2a_backend, _):
        # A non-gated intermediate of 64 is padded to Marlin's 128.
        self.I, self.shards = 64, 1
        self._check_reload(a2a_backend)

    def _check_reload(self, a2a_backend):
        a2a_backend.return_value.is_megamoe.return_value = False
        torch.manual_seed(0)
        first, second = self._checkpoint(), self._checkpoint()
        method, layer = self._make()
        self._load(layer, first)
        method.process_weights_after_loading(layer)
        served_first = self._served(layer)
        params = dict(layer.named_parameters())
        ptrs = _ptrs(layer, list(params) + ["workspace"])

        method.restore_weights_before_loading(layer)
        for name in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"):
            self.assertTrue(
                torch.equal(
                    getattr(layer, name).view(torch.uint8),
                    first[name].view(torch.uint8),
                ),
                name,
            )
        if self.shards == 2:
            self.assertTrue(
                torch.equal(layer.w13_weight_scale_2, first["w13_weight_scale_2"])
            )
        for param in layer.parameters():
            self.assertIs(param.weight_loader, default_weight_loader)

        # Restore then repack gives the kernel the same bytes.
        method.process_weights_after_loading(layer)
        for name, value in self._served(layer).items():
            self.assertTrue(
                torch.equal(
                    value.view(torch.uint8), served_first[name].view(torch.uint8)
                ),
                name,
            )
        self.assertEqual(_ptrs(layer, list(params) + ["workspace"]), ptrs)

        # A reload with new weights matches a fresh layer, in place.
        method.restore_weights_before_loading(layer)
        self._load(layer, second)
        method.process_weights_after_loading(layer)
        fresh_method, fresh = self._make()
        self._load(fresh, second)
        fresh_method.process_weights_after_loading(fresh)
        for name, value in self._served(fresh).items():
            self.assertTrue(
                torch.equal(
                    getattr(layer, name).view(torch.uint8), value.view(torch.uint8)
                ),
                name,
            )
        self.assertEqual(_ptrs(layer, list(params) + ["workspace"]), ptrs)
        for name, param in params.items():
            self.assertIs(getattr(layer, name), param)


if __name__ == "__main__":
    sys.exit(unittest.main())
