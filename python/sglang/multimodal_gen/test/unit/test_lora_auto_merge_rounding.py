"""merge_mode "auto" keeps a LoRA unmerged when merging would round its update away.

Distilled adapters (MiniMax-H3 Turbo) carry updates below a BF16 ulp of the base
weights, so a merge into BF16 drops most of them. These check the estimate against
an actual merge on CPU, and the decision the pipeline takes from it.
"""

from types import SimpleNamespace

import torch
from torch import nn

from sglang.multimodal_gen.runtime.layers.lora.linear import LinearWithLoRA
from sglang.multimodal_gen.runtime.pipelines_core.lora.pipeline import (
    AUTO_MERGE_MAX_ROUNDING_LOSS,
    LoRAPipeline,
    merge_rounding_loss,
)


def _layer(base: torch.Tensor, a: torch.Tensor, b: torch.Tensor, strength: float):
    linear = nn.Linear(base.shape[1], base.shape[0], bias=False, dtype=base.dtype)
    with torch.no_grad():
        linear.weight.copy_(base)
    layer = LinearWithLoRA(linear, lora_rank=a.shape[0], lora_alpha=a.shape[0])
    layer.set_lora_weights(
        a, b, strength=strength, clear_existing=True, merge_weights=False
    )
    return layer


def _inputs(update_scale: float, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(96, 64, generator=g).to(torch.bfloat16)
    a = torch.randn(8, 64, generator=g)
    b = update_scale * torch.randn(96, 8, generator=g)
    return base, a, b


def test_estimate_matches_the_rounding_of_an_actual_merge() -> None:
    base, a, b = _inputs(1e-3)
    layer = _layer(base, a, b, strength=0.5)
    lost_sq, delta_sq = layer.merge_rounding_norms(max_rows=10**6)
    exact = base.float() + 0.5 * (b @ a)
    layer.merge_lora_weights()
    merged = layer.base_layer.weight.detach().float()
    torch.testing.assert_close(
        torch.tensor(lost_sq), (merged - exact).square().sum(), rtol=1e-3, atol=0
    )
    torch.testing.assert_close(
        torch.tensor(delta_sq), (0.5 * (b @ a)).square().sum(), rtol=1e-4, atol=0
    )


def test_update_below_a_bf16_ulp_is_mostly_rounded_away() -> None:
    base, a, b = _inputs(1e-5)
    loss = merge_rounding_loss({"l": _layer(base, a, b, strength=1.0)})
    assert loss is not None and loss > 0.9


def test_update_above_a_bf16_ulp_survives() -> None:
    base, a, b = _inputs(0.1)
    loss = merge_rounding_loss({"l": _layer(base, a, b, strength=1.0)})
    assert loss is not None and loss < 0.05


def test_fp32_base_has_nothing_to_round() -> None:
    base, a, b = _inputs(1e-5)
    layer = _layer(base.float(), a, b, strength=1.0)
    assert layer.merge_rounding_norms() is None
    assert merge_rounding_loss({"l": layer}) is None


def test_auto_decision_measures_once_and_follows_the_threshold() -> None:
    calls = []
    norms = {"lossy": (0.81, 1.0), "fine": (1e-4, 1.0)}

    def run(kind: str) -> bool:
        layers = {"l": SimpleNamespace(merge_rounding_norms=lambda: norms[kind])}
        stub = SimpleNamespace(
            auto_merge_rounding_loss={},
            loaded_adapter_alphas={},
            _apply_lora_to_layers=lambda *args, **kwargs: calls.append(kwargs),
        )
        first = LoRAPipeline._auto_merge_rounds_away(
            stub, "transformer", layers, ["a"], [kind], 0, [1.0]
        )
        second = LoRAPipeline._auto_merge_rounds_away(
            stub, "transformer", layers, ["a"], [kind], 0, [1.0]
        )
        assert first == second
        return first

    assert run("lossy") is True  # 90% of the update's norm lost
    assert run("fine") is False
    # measured once per configuration, attached unmerged to measure
    assert len(calls) == 2 and all(c["merge_weights"] is False for c in calls)
    assert AUTO_MERGE_MAX_ROUNDING_LOSS == 0.1
