# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers.models.gemma.configuration_gemma import GemmaConfig

from sglang.multimodal_gen.configs.pipeline_configs.pi05 import Pi05PipelineConfig
from sglang.multimodal_gen.runtime.models.vlas.pi05_core import (
    Pi05CoreModel,
    PiGemmaDecoderLayer,
)
from sglang.multimodal_gen.runtime.vla.prefix_cache import VLADensePrefixCache
from sglang.test.ci.ci_register import register_cuda_ci

# The diffusion unit lane provides the model's runtime dependencies.
register_cuda_ci(est_time=90, stage="base-b", runner_config="diffusion-unit-1-gpu-h100")


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA required"
            ),
        ),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("masked", [False, True])
def test_action_layer_replay_and_backward(device, dtype, masked):
    """No-grad sampling and differentiable replay use the same action network."""
    torch.manual_seed(7)
    config = GemmaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=8,
        num_hidden_layers=1,
        hidden_activation="gelu_pytorch_tanh",
        use_adarms=True,
        use_training_ops=True,
        adarms_cond_dim=32,
        use_bidirectional_attention=True,
        dtype=str(dtype).removeprefix("torch."),
    )
    layer = (
        PiGemmaDecoderLayer(config, layer_idx=0)
        .to(device=device, dtype=dtype)
        .requires_grad_(True)
    )
    for parameter in layer.parameters():
        torch.nn.init.normal_(parameter, std=0.05)
    hidden = torch.randn(2, 3, 32, device=device, dtype=dtype)
    condition = torch.randn(2, 32, device=device, dtype=dtype, requires_grad=True)
    cache = VLADensePrefixCache(
        [
            (
                torch.randn(2, 1, 5, 8, device=device, dtype=dtype),
                torch.randn(2, 1, 5, 8, device=device, dtype=dtype),
                None,
            )
        ],
        read_only=True,
    )
    positions = (
        torch.ones(2, 3, 8, device=device),
        torch.zeros(2, 3, 8, device=device),
    )
    mask = torch.zeros(2, 1, 3, 8, dtype=torch.float32, device=device)
    mask[:, :, :, 1] = -2.3819763e38
    kwargs = dict(
        position_embeddings=positions,
        attention_mask=mask if masked else None,
        past_key_values=cache,
        adarms_cond=condition,
    )
    with torch.no_grad():
        sampled = layer(hidden, **kwargs)
    replayed = layer(hidden, **kwargs)
    torch.testing.assert_close(replayed, sampled, rtol=0, atol=0)
    replayed.square().mean().backward()
    for parameter in (
        layer.self_attn.qkv_proj.weight,
        layer.mlp.gate_up_proj.weight,
        layer.input_layernorm.dense.weight,
        condition,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert cache.get_seq_length() == 5


@pytest.mark.parametrize(
    "kwargs", [{"runtime_role": "action"}, {"prefix_tensor_parallel": True}]
)
def test_action_training_requires_resident_unsharded_model(kwargs):
    with pytest.raises(ValueError, match="resident, unsharded"):
        Pi05CoreModel(Pi05PipelineConfig(), action_training=True, **kwargs)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
