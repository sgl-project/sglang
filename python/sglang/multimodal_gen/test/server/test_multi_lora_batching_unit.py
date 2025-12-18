# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang.multimodal_gen.runtime.layers.lora.linear import BaseLayerWithLoRA
from sglang.multimodal_gen.runtime.lora.lora_manager import DiffusionLoRAManager, LoRAAdapter


class _DummyBaseLayer(torch.nn.Module):
    """A minimal layer that matches BaseLayerWithLoRA expectations.

    - Exposes `.weight` for device/shape inference
    - Callable returns (out, output_bias) like SGLang diffusion linear layers
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.zeros((out_dim, in_dim), dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor):
        out = x @ self.weight.T
        return out, None


def test_base_layer_with_lora_uses_per_adapter_alpha_rank_for_multi_lora_scaling():
    in_dim = 3
    out_dim = 2

    base = _DummyBaseLayer(in_dim=in_dim, out_dim=out_dim)
    layer = BaseLayerWithLoRA(base_layer=base, lora_rank=16, lora_alpha=16)
    layer.disable_lora = False

    # Two adapters with different ranks and alphas
    # For ones weights: delta before scaling = in_dim * rank, after scaling = in_dim * alpha
    A1 = torch.ones((2, in_dim), dtype=torch.float32)  # rank=2
    B1 = torch.ones((out_dim, 2), dtype=torch.float32)
    A2 = torch.ones((1, in_dim), dtype=torch.float32)  # rank=1
    B2 = torch.ones((out_dim, 1), dtype=torch.float32)

    layer.prepare_multi_lora_batch(
        lora_weights_pool={"a": (A1, B1), "b": (A2, B2)},
        lora_nickname_to_index={"a": 0, "b": 1},
        lora_adapter_configs={
            "a": {"alpha": 4.0, "rank": 2},
            "b": {"alpha": 2.0, "rank": 1},
        },
    )
    layer.set_multi_lora_indices([0, 1, -1])

    x = torch.ones((3, in_dim), dtype=torch.float32)
    base_out, output_bias = base(x)
    out, _ = layer._apply_multi_lora(x, base_out, output_bias)

    assert torch.allclose(out[0], torch.full((out_dim,), 12.0))
    assert torch.allclose(out[1], torch.full((out_dim,), 6.0))
    assert torch.allclose(out[2], torch.zeros((out_dim,)))


@dataclass
class _Req:
    request_id: str
    lora_nickname: str | None = None
    lora_path: str | None = None


def test_diffusion_lora_manager_prepare_lora_batch_returns_per_adapter_configs():
    device = torch.device("cpu")
    manager = DiffusionLoRAManager(device=device, server_args=None, modules=None)

    layer_name = "layer0"
    in_dim = 3
    out_dim = 2

    adapter_a = LoRAAdapter(
        nickname="a",
        path="/tmp/a.safetensors",
        weights={
            f"{layer_name}.lora_A": torch.ones((2, in_dim), dtype=torch.float32),
            f"{layer_name}.lora_B": torch.ones((out_dim, 2), dtype=torch.float32),
        },
        rank=2,
        alpha=4.0,
    )
    adapter_b = LoRAAdapter(
        nickname="b",
        path="/tmp/b.safetensors",
        weights={
            f"{layer_name}.lora_A": torch.ones((1, in_dim), dtype=torch.float32),
            f"{layer_name}.lora_B": torch.ones((out_dim, 1), dtype=torch.float32),
        },
        rank=1,
        alpha=2.0,
    )
    manager.lora_adapters["a"] = adapter_a
    manager.lora_adapters["b"] = adapter_b

    reqs = [
        _Req(request_id="r0", lora_nickname="a"),
        _Req(request_id="r1", lora_nickname="b"),
    ]

    batch_weights, nickname_to_index, adapter_configs = manager.prepare_lora_batch(
        requests=reqs,
        lora_layers={layer_name: object()},
    )

    assert set(batch_weights.keys()) == {layer_name}
    assert set(batch_weights[layer_name].keys()) == {"a", "b"}
    assert nickname_to_index == {"a": 0, "b": 1}

    assert adapter_configs["a"]["rank"] == 2
    assert adapter_configs["a"]["alpha"] == 4.0
    assert adapter_configs["b"]["rank"] == 1
    assert adapter_configs["b"]["alpha"] == 2.0


def test_diffusion_lora_manager_prepare_lora_batch_raises_when_too_many_loras():
    device = torch.device("cpu")
    manager = DiffusionLoRAManager(
        max_loras_per_batch=1, device=device, server_args=None, modules=None
    )

    manager.lora_adapters["a"] = LoRAAdapter(
        nickname="a", path="/tmp/a", weights={}, rank=1, alpha=1.0
    )
    manager.lora_adapters["b"] = LoRAAdapter(
        nickname="b", path="/tmp/b", weights={}, rank=1, alpha=1.0
    )

    reqs = [
        _Req(request_id="r0", lora_nickname="a"),
        _Req(request_id="r1", lora_nickname="b"),
    ]

    try:
        manager.prepare_lora_batch(requests=reqs, lora_layers={})
        assert False, "Expected ValueError for too many LoRAs in batch"
    except ValueError as e:
        assert "Too many LoRAs in batch" in str(e)


