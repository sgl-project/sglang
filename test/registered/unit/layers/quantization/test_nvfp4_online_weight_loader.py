from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.quantization.nvfp4_online import (
    ModelOptNvFp4OnlineFusedMoEMethod,
)
from sglang.srt.model_loader.weight_utils import RUNAI_STREAMER_TENSOR_ATTR
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


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


@pytest.mark.parametrize("first_shard", ["w1", "w3"])
def test_streamed_gated_pair_owns_pending_weight(
    monkeypatch: pytest.MonkeyPatch, first_shard: str
):
    loader, converted = _loader(monkeypatch)
    param = torch.nn.Parameter(torch.empty(2, 16))
    borrowed = torch.ones(2, 16, dtype=torch.bfloat16)
    setattr(borrowed, RUNAI_STREAMER_TENSOR_ATTR, True)
    loader(param, borrowed, "expert.weight", first_shard, 0)
    borrowed.fill_(float("inf"))
    second_shard = "w3" if first_shard == "w1" else "w1"
    loader(param, torch.full_like(borrowed, 2), "expert.weight", second_shard, 0)
    torch.testing.assert_close(
        converted[0],
        torch.cat([torch.ones_like(borrowed), torch.full_like(borrowed, 2)]),
    )


@pytest.mark.parametrize("scale_first", [False, True])
def test_streamed_fp8_pair_owns_pending_tensor(
    monkeypatch: pytest.MonkeyPatch, scale_first: bool
):
    loader, converted = _loader(monkeypatch, fp8=True)
    param = torch.nn.Parameter(torch.empty(2, 16))
    weight = torch.ones(2, 16).to(torch.float8_e4m3fn)
    scale = torch.tensor(2.0)
    borrowed = scale if scale_first else weight
    setattr(borrowed, RUNAI_STREAMER_TENSOR_ATTR, True)
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
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
