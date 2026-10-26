import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sglang.srt.layers.quantization.nvfp4_online import (
    ModelOptNvFp4OnlineFusedMoEMethod,
)
from sglang.srt.model_loader.weight_utils import (
    RUNAI_STREAMER_TENSOR_ATTR,
    runai_safetensors_weights_iterator,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _borrowed_tensor(monkeypatch, tensor, tensor_form):
    if tensor_form == "direct":
        setattr(tensor, RUNAI_STREAMER_TENSOR_ATTR, True)
        return tensor
    # RunAI yields a shaped view, and packed-expert loaders slice it again.
    packed = torch.stack([tensor, tensor]).reshape(2, *tensor.shape)
    streamer = MagicMock()
    streamer.__enter__.return_value = streamer
    streamer.files_to_tensors_metadata = {"checkpoint": ["weight"]}
    streamer.get_tensors.return_value = iter([("weight", packed)])
    monkeypatch.setitem(
        sys.modules,
        "runai_model_streamer",
        SimpleNamespace(
            SafetensorsStreamer=lambda: streamer,
        ),
    )
    _, streamed = next(runai_safetensors_weights_iterator(["checkpoint"]))
    view = streamed.chunk(2, dim=0)[0][0]
    if tensor_form == "detached":
        view = view.detach()
    assert not getattr(view, RUNAI_STREAMER_TENSOR_ATTR, False)
    assert not getattr(
        view.clone().untyped_storage(), RUNAI_STREAMER_TENSOR_ATTR, False
    )
    return view


def _loader(monkeypatch: pytest.MonkeyPatch, *, fp8: bool = False):
    converted = []

    def quantize(weight: torch.Tensor):
        converted.append(weight.clone())
        return weight.clone(), torch.ones(weight.shape[0], 1), torch.tensor(1.0)

    monkeypatch.setattr(
        ModelOptNvFp4OnlineFusedMoEMethod,
        "_quantize_weight_nvfp4",
        staticmethod(quantize),
    )
    monkeypatch.setattr(
        ModelOptNvFp4OnlineFusedMoEMethod,
        "_should_skip_loaded_expert",
        staticmethod(lambda *args: False),
    )
    layer = SimpleNamespace(
        moe_runner_config=SimpleNamespace(is_gated=True),
        w13_weight_scale=None,
        w2_weight_scale=None,
        w13_weight_scale_2=None,
        w2_weight_scale_2=None,
    )
    loader = ModelOptNvFp4OnlineFusedMoEMethod.get_online_weight_loader(
        layer,
        lambda *args, **kwargs: None,
        layer_log_name="test.experts",
        fp8_dequantizer=(
            (lambda weight, scale, device: (weight.float() * scale).to(torch.bfloat16))
            if fp8
            else None
        ),
    )
    return loader, converted


@pytest.mark.parametrize("tensor_form", ["direct", "view", "detached"])
@pytest.mark.parametrize("first_shard", ["w1", "w3"])
def test_streamed_gated_pair_owns_pending_weight(
    monkeypatch: pytest.MonkeyPatch, first_shard: str, tensor_form: str
):
    loader, converted = _loader(monkeypatch)
    param = torch.nn.Parameter(torch.empty(2, 16))
    borrowed = _borrowed_tensor(
        monkeypatch, torch.ones(2, 16, dtype=torch.bfloat16), tensor_form
    )
    loader(param, borrowed, "expert.weight", first_shard, 0)
    borrowed.fill_(float("inf"))
    second_shard = "w3" if first_shard == "w1" else "w1"
    loader(param, torch.full_like(borrowed, 2), "expert.weight", second_shard, 0)
    torch.testing.assert_close(
        converted[0],
        torch.cat([torch.ones_like(borrowed), torch.full_like(borrowed, 2)]),
    )


@pytest.mark.parametrize("tensor_form", ["direct", "view", "detached"])
@pytest.mark.parametrize("scale_first", [False, True])
def test_streamed_fp8_pair_owns_pending_tensor(
    monkeypatch: pytest.MonkeyPatch, scale_first: bool, tensor_form: str
):
    loader, converted = _loader(monkeypatch, fp8=True)
    param = torch.nn.Parameter(torch.empty(2, 16))
    weight = torch.ones(2, 16).to(torch.float8_e4m3fn)
    scale = torch.tensor(2.0)
    if scale_first:
        scale = _borrowed_tensor(monkeypatch, scale, tensor_form)
    else:
        weight = _borrowed_tensor(monkeypatch, weight, tensor_form)
    if scale_first:
        loader(param, scale, "expert.weight_scale_inv", "w2", 0)
        scale.fill_(float("nan"))
        loader(param, weight, "expert.weight", "w2", 0)
    else:
        loader(param, weight, "expert.weight", "w2", 0)
        weight.view(torch.uint8).fill_(127)
        loader(param, scale, "expert.weight_scale_inv", "w2", 0)
    torch.testing.assert_close(
        converted[0], torch.full((2, 16), 2.0, dtype=torch.bfloat16)
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
