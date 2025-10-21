import pytest

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - environment without torch
    torch = None
    nn = None

from sglang.srt.model_loader.auto_weights_loader import AutoWeightsLoader


class _LinearBlock(nn.Module):

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_features, out_features, bias=False)


class _ToyModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.block = _LinearBlock(4, 2)
        self.register_buffer("unused_stat", torch.zeros(1))


@pytest.mark.skipif(torch is None, reason="torch not available")
def test_auto_weights_loader_loads_nested_parameters():
    model = _ToyModel()
    weight_tensor = torch.arange(8, dtype=model.block.proj.weight.dtype).view(2, 4)

    loader = AutoWeightsLoader(model)
    loaded = loader.load_weights([("block.proj.weight", weight_tensor)])

    assert "block.proj.weight" in loaded
    assert torch.allclose(model.block.proj.weight, weight_tensor)


@pytest.mark.skipif(torch is None, reason="torch not available")
def test_auto_weights_loader_skips_ignored_prefix():
    model = _ToyModel()
    weight_tensor = torch.ones_like(model.block.proj.weight)

    loader = AutoWeightsLoader(
        model,
        ignore_unexpected_prefixes=["ignore."],
    )

    loaded = loader.load_weights(
        [
            ("block.proj.weight", weight_tensor),
            ("ignore.extra", torch.zeros(1)),
        ]
    )

    assert "block.proj.weight" in loaded
    assert torch.allclose(model.block.proj.weight, weight_tensor)


def test_auto_weights_loader_placeholder_when_torch_missing():
    if torch is not None:
        pytest.skip("torch available")
    assert True
