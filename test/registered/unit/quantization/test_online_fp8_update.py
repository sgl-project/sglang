"""CPU storage/lifecycle checks; actual kernel parity needs GPU qualification."""

import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn

# Load the torch-only helper without importing every optional quantization backend.
_source = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/srt/layers/quantization/online_fp8_update.py"
)
_spec = importlib.util.spec_from_file_location("online_fp8_update", _source)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
OnlineFP8Update = _module.OnlineFP8Update


class LoadParameter(nn.Parameter):
    def __new__(cls, data, **kwargs):
        return super().__new__(cls, data, requires_grad=False)

    def load_shard(self, value, index):
        self.data[index].copy_(value)


def quantize(layer):
    for name in ("w13_weight", "w2_weight"):
        source = getattr(layer, name).float()
        scale = source.abs().flatten(1).amax(1).clamp_min(1e-10) / 448
        packed = (source / scale[:, None, None]).to(torch.float8_e4m3fn)
        setattr(layer, name, nn.Parameter(packed, requires_grad=False))
        setattr(layer, name + "_scale", nn.Parameter(scale, requires_grad=False))


def make_layer():
    layer = nn.Module()
    for name, shape in (("w13_weight", (2, 6, 4)), ("w2_weight", (2, 4, 3))):
        param = LoadParameter(torch.randn(shape, dtype=torch.bfloat16))
        param.shard_metadata = ("native", 2)
        setattr(layer, name, param)
    update = OnlineFP8Update(layer, ("w13_weight", "w2_weight"))
    sources = {name: param.detach().clone() for name, param in layer.named_parameters()}
    quantize(layer)
    return layer, update, sources


def test_split_experts_repeated_updates_preserve_storage_and_native_metadata():
    layer, update, sources = make_layer()
    serving = dict(layer.named_parameters())
    pointers = {name: value.data_ptr() for name, value in serving.items()}
    for factor in (1, -3, 1):
        update.begin(layer)
        with pytest.raises(RuntimeError, match="already open"):
            update.begin(layer)
        # Gate and up are separate buckets and separate experts, in mixed order.
        for name, source in sources.items():
            param = getattr(layer, name)
            assert isinstance(param, LoadParameter)
            assert param.shard_metadata == ("native", 2)
            for expert in (1, 0):
                for row in reversed(range(source.shape[1])):
                    param.load_shard(source[expert, row] * factor, (expert, row))
        update.finish(layer, quantize)
        expected = nn.Module()
        for name, source in sources.items():
            setattr(expected, name, nn.Parameter(source * factor, requires_grad=False))
        quantize(expected)
        for name, param in layer.named_parameters():
            assert param is serving[name]
            assert param.data_ptr() == pointers[name]
            torch.testing.assert_close(
                param.float(), getattr(expected, name).float(), rtol=0, atol=0
            )
        assert not update.active


@pytest.mark.parametrize(
    "failure", ["missing_up", "empty", "nan", "infinity", "layout", "quantizer"]
)
def test_failed_update_preserves_serving_weights_and_allows_retry(failure):
    layer, update, sources = make_layer()
    serving = dict(layer.named_parameters())
    baseline = {name: value.float().clone() for name, value in serving.items()}
    update.begin(layer)
    if failure != "empty":
        for name, source in sources.items():
            getattr(layer, name).data.copy_(source)
        if failure == "missing_up":
            layer.w13_weight.data[:, 3:].fill_(float("nan"))
        if failure in ("nan", "infinity"):
            layer.w2_weight.data.fill_(
                float("nan") if failure == "nan" else float("inf")
            )

    def broken_quantize(layer):
        quantize(layer)
        if failure == "layout":
            layer.w2_weight = nn.Parameter(
                layer.w2_weight.transpose(1, 2), requires_grad=False
            )
        if failure == "quantizer":
            raise RuntimeError("kernel failed")

    with pytest.raises((ValueError, RuntimeError)):
        update.finish(layer, broken_quantize)
    assert not update.active
    for name, value in layer.named_parameters():
        assert value is serving[name]
        torch.testing.assert_close(value.float(), baseline[name], rtol=0, atol=0)
    update.begin(layer)
    for name, source in sources.items():
        getattr(layer, name).data.copy_(source)
    update.finish(layer, quantize)
