"""Weight updates of per-tensor FP8 layers (ModelOpt FP8 linear and MoE,
compressed-tensors W8A8 FP8 with per-tensor or per-channel weight scales) and
compressed-tensors W8A8 INT8 with per-channel weight scales.

Processing requantizes fused shards to their max scale, transposes the linear
weight and collapses the scales. A weight update restores every layer to the
checkpoint form, loads through each parameter's weight_loader, and processes
again. These tests check that restore keeps the Parameters, their loaders and
their storage, that an untouched layer is served unchanged, and that a layer
partly or fully reloaded serves what a fresh layer with those weights serves.
"""

import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn
from compressed_tensors.quantization import QuantizationArgs

from sglang.srt.layers.moe import MoeRunnerBackend
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW8A8Fp8,
    CompressedTensorsW8A8Int8,
)
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp8Config,
    ModelOptFp8LinearMethod,
    ModelOptFp8MoEMethod,
)
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils.device import is_sm90_supported
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-large")

CONFIG = ModelOptFp8Config(is_checkpoint_fp8_serialized=True)


def _fp8(*shape):
    values = torch.randn(shape, device="cuda") * 100
    return values.clamp(-448, 448).to(torch.float8_e4m3fn)


def _bytes(t):
    return t.detach().contiguous().reshape(-1).view(torch.uint8).clone()


def _served(layer, names):
    return {name: _bytes(getattr(layer, name)) for name in names}


def _ptrs(layer, names):
    return {name: getattr(layer, name).data_ptr() for name in names}


@unittest.skipIf(not is_sm90_supported(), "ModelOpt FP8 linear runs on SM90+")
class TestModelOptFp8LinearReload(CustomTestCase):
    widths = [256, 64, 64]
    k = 128
    # Identical served bytes at identical addresses already imply identical
    # outputs; apply is also run where it needs no server config.
    check_apply = True
    served = ("weight", "weight_scale", "input_scale")

    def _checkpoint(self):
        n = len(self.widths)
        return {
            "weight": _fp8(sum(self.widths), self.k),
            # Different scales per shard, so processing really requantizes.
            "weight_scale": torch.tensor([0.02, 0.05, 0.01], device="cuda")[:n],
            "input_scale": torch.tensor([0.3, 0.1, 0.2], device="cuda")[:n],
        }

    def _method(self):
        return ModelOptFp8LinearMethod(CONFIG)

    def _make(self, checkpoint):
        method = self._method()
        layer = nn.Module()
        with torch.device("cuda"):
            method.create_weights(
                layer,
                input_size_per_partition=self.k,
                output_partition_sizes=self.widths,
                input_size=self.k,
                output_size=sum(self.widths),
                params_dtype=torch.bfloat16,
                weight_loader=default_weight_loader,
            )
        for name, tensor in checkpoint.items():
            param = getattr(layer, name)
            param.weight_loader(param, tensor)
        method.process_weights_after_loading(layer)
        return method, layer

    def _restored_scale_shape(self):
        return (len(self.widths),)

    def _load_shard(self, layer, checkpoint, shard):
        # What the fused QKV / gate-up loaders write for one shard.
        start = sum(self.widths[:shard])
        rows = slice(start, start + self.widths[shard])
        layer.weight.data[rows].copy_(checkpoint["weight"][rows])
        layer.weight_scale.data[shard] = checkpoint["weight_scale"][shard]
        layer.input_scale.data[shard] = checkpoint["input_scale"][shard]

    def test_restore_keeps_parameters(self):
        torch.manual_seed(0)
        method, layer = self._make(self._checkpoint())
        params = {name: getattr(layer, name) for name in self.served}
        ptrs = _ptrs(layer, self.served)
        served = _served(layer, self.served)
        x = torch.randn(5, self.k, dtype=torch.bfloat16, device="cuda")
        out = method.apply(layer, x) if self.check_apply else None

        method.restore_weights_before_loading(layer)
        n = len(self.widths)
        self.assertEqual(layer.weight.shape, (sum(self.widths), self.k))
        self.assertEqual(layer.weight_scale.shape, self._restored_scale_shape())
        if "input_scale" in self.served:
            self.assertEqual(layer.input_scale.shape, (n,))
        for name in self.served:
            self.assertIs(getattr(layer, name), params[name])
            self.assertIs(getattr(layer, name).weight_loader, default_weight_loader)

        # Restore then process serves the same bytes from the same storage.
        method.process_weights_after_loading(layer)
        for name, value in _served(layer, self.served).items():
            self.assertTrue(torch.equal(value, served[name]), name)
        self.assertEqual(_ptrs(layer, self.served), ptrs)
        if self.check_apply:
            self.assertTrue(torch.equal(method.apply(layer, x), out))

        # A second processing without a restore changes nothing.
        method.process_weights_after_loading(layer)
        for name, value in _served(layer, self.served).items():
            self.assertTrue(torch.equal(value, served[name]), name)

    def test_partial_and_full_reload_match_fresh(self):
        torch.manual_seed(1)
        first, second = self._checkpoint(), self._checkpoint()
        second["weight_scale"] = second["weight_scale"].flip(0)
        method, layer = self._make(first)
        ptrs = _ptrs(layer, self.served)

        # A session rewriting one shard with the bytes it was loaded from.
        for shard in range(len(self.widths)):
            method.restore_weights_before_loading(layer)
            self._load_shard(layer, first, shard)
            method.process_weights_after_loading(layer)
            fresh = _served(self._make(first)[1], self.served)
            for name, value in _served(layer, self.served).items():
                self.assertTrue(torch.equal(value, fresh[name]), (shard, name))

        # A full reload with new weights serves what a fresh layer serves.
        method.restore_weights_before_loading(layer)
        for name, tensor in second.items():
            param = getattr(layer, name)
            param.weight_loader(param, tensor)
        method.process_weights_after_loading(layer)
        fresh = _served(self._make(second)[1], self.served)
        for name, value in _served(layer, self.served).items():
            self.assertTrue(torch.equal(value, fresh[name]), name)
        self.assertEqual(_ptrs(layer, self.served), ptrs)


class TestModelOptFp8LinearReloadUnfused(TestModelOptFp8LinearReload):
    widths = [192]


class _CompressedTensorsMethod:
    """The scheme behind CompressedTensorsLinearMethod, with its apply."""

    def __init__(self, strategy, scheme=None):
        weights = QuantizationArgs(
            num_bits=8, type="float", strategy=strategy, symmetric=True, dynamic=False
        )
        self.scheme = scheme or CompressedTensorsW8A8Fp8(
            weights, is_static_input_scheme=True
        )

    def create_weights(self, layer, **kwargs):
        self.scheme.create_weights(layer, **kwargs)

    def process_weights_after_loading(self, layer):
        self.scheme.process_weights_after_loading(layer)

    def restore_weights_before_loading(self, layer):
        self.scheme.restore_weights_before_loading(layer)

    def apply(self, layer, x):
        return self.scheme.apply_weights(layer, x, None)


class TestCompressedTensorsFp8TensorReload(TestModelOptFp8LinearReload):
    check_apply = False

    def _method(self):
        return _CompressedTensorsMethod("tensor")


class TestCompressedTensorsFp8ChannelReload(TestCompressedTensorsFp8TensorReload):
    def _method(self):
        return _CompressedTensorsMethod("channel")

    def _checkpoint(self):
        checkpoint = super()._checkpoint()
        checkpoint["weight_scale"] = (
            torch.rand(sum(self.widths), 1, device="cuda") / 50 + 1e-3
        )
        return checkpoint

    def _load_shard(self, layer, checkpoint, shard):
        start = sum(self.widths[:shard])
        rows = slice(start, start + self.widths[shard])
        layer.weight.data[rows].copy_(checkpoint["weight"][rows])
        layer.weight_scale.data[rows] = checkpoint["weight_scale"][rows]
        layer.input_scale.data[shard] = checkpoint["input_scale"][shard]

    def _restored_scale_shape(self):
        return (sum(self.widths), 1)


class TestCompressedTensorsInt8ChannelReload(TestCompressedTensorsFp8ChannelReload):
    """W8A8 INT8 with per-channel weights and dynamic per-token activations."""

    served = ("weight", "weight_scale")

    def _method(self):
        return _CompressedTensorsMethod(
            None,
            CompressedTensorsW8A8Int8(
                "channel", is_static_input_scheme=False, input_symmetric=True
            ),
        )

    def _checkpoint(self):
        checkpoint = super()._checkpoint()
        del checkpoint["input_scale"]
        checkpoint["weight"] = torch.randint(
            -128, 128, checkpoint["weight"].shape, dtype=torch.int8, device="cuda"
        )
        return checkpoint

    def _load_shard(self, layer, checkpoint, shard):
        start = sum(self.widths[:shard])
        rows = slice(start, start + self.widths[shard])
        layer.weight.data[rows].copy_(checkpoint["weight"][rows])
        layer.weight_scale.data[rows] = checkpoint["weight_scale"][rows]


@unittest.skipIf(not is_sm90_supported(), "ModelOpt FP8 MoE runs on SM90+")
class TestModelOptFp8MoEReload(CustomTestCase):
    backend = MoeRunnerBackend.TRITON
    gated = True
    experts, hidden, inter = 4, 128, 64
    served = (
        "w13_weight",
        "w2_weight",
        "w13_weight_scale",
        "w2_weight_scale",
        "w13_input_scale",
        "w2_input_scale",
    )

    def _checkpoint(self):
        e, shards = self.experts, 2 if self.gated else 1
        return {
            "w13_weight": _fp8(e, shards * self.inter, self.hidden),
            "w2_weight": _fp8(e, self.hidden, self.inter),
            "w13_weight_scale": torch.rand(e, shards, device="cuda") / 50 + 1e-3,
            "w2_weight_scale": torch.rand(e, device="cuda") / 50 + 1e-3,
            "w13_input_scale": torch.rand(e, device="cuda") + 0.1,
            "w2_input_scale": torch.rand(e, device="cuda") + 0.1,
        }

    def _make(self, checkpoint):
        method = ModelOptFp8MoEMethod(CONFIG)
        method._moe_runner_backend = self.backend
        layer = nn.Module()
        layer.moe_runner_config = SimpleNamespace(
            is_gated=self.gated,
            activation="silu" if self.gated else "relu2",
            gemm1_clamp_limit=None,
            swiglu_limit=None,
        )
        layer.num_local_experts = self.experts
        with torch.device("cuda"):
            method.create_weights(
                layer,
                num_experts=self.experts,
                hidden_size=self.hidden,
                intermediate_size_per_partition=self.inter,
                params_dtype=torch.bfloat16,
                weight_loader=default_weight_loader,
            )
        for name, tensor in checkpoint.items():
            param = getattr(layer, name)
            param.weight_loader(param, tensor)
        method.process_weights_after_loading(layer)
        return method, layer

    def test_restore_and_reload(self):
        torch.manual_seed(2)
        first, second = self._checkpoint(), self._checkpoint()
        method, layer = self._make(first)
        params = {name: getattr(layer, name) for name in self.served}
        ptrs = _ptrs(layer, self.served)
        served = _served(layer, self.served)

        method.restore_weights_before_loading(layer)
        for name in self.served:
            self.assertIs(getattr(layer, name), params[name])
        for name in first:
            self.assertIs(getattr(layer, name).weight_loader, default_weight_loader)
            self.assertEqual(getattr(layer, name).shape, first[name].shape, name)
        method.process_weights_after_loading(layer)
        for name, value in _served(layer, self.served).items():
            self.assertTrue(torch.equal(value, served[name]), name)
        self.assertEqual(_ptrs(layer, self.served), ptrs)

        # One expert's w13 rewritten with its own bytes and scales.
        method.restore_weights_before_loading(layer)
        layer.w13_weight.data[1].copy_(first["w13_weight"][1])
        layer.w13_weight_scale.data[1] = first["w13_weight_scale"][1]
        method.process_weights_after_loading(layer)
        for name, value in _served(layer, self.served).items():
            self.assertTrue(torch.equal(value, served[name]), name)

        method.restore_weights_before_loading(layer)
        for name, tensor in second.items():
            param = getattr(layer, name)
            param.weight_loader(param, tensor)
        method.process_weights_after_loading(layer)
        fresh = _served(self._make(second)[1], self.served)
        for name, value in _served(layer, self.served).items():
            self.assertTrue(torch.equal(value, fresh[name]), name)
        self.assertEqual(_ptrs(layer, self.served), ptrs)


class TestModelOptFp8MoEReloadNonGated(TestModelOptFp8MoEReload):
    gated = False


class TestModelOptFp8MoEReloadCutlass(TestModelOptFp8MoEReload):
    backend = MoeRunnerBackend.FLASHINFER_CUTLASS
    served = TestModelOptFp8MoEReload.served + (
        "fc1_dequant",
        "fc2_quant",
        "fc2_dequant",
        "fc1_input_dequant",
    )


class TestModelOptFp8MoEReloadCutlassNonGated(TestModelOptFp8MoEReloadCutlass):
    gated = False


if __name__ == "__main__":
    unittest.main()
