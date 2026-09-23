# SPDX-License-Identifier: Apache-2.0
"""Run with torchrun --standalone --nproc-per-node=2 -m pytest -q <this file>."""

import os

import pytest
import torch

from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_sp_group,
    get_tp_group,
    maybe_init_distributed_environment_and_model_parallel,
    use_tensor_parallel_group,
)
from sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify.configuration_neo_chat import (
    NEOLLMConfig,
)
from sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify.modeling_qwen3 import (
    Qwen3ForCausalLM,
    set_attn_backend,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or int(os.environ.get("WORLD_SIZE", "1")) != 2,
    reason="requires two CUDA ranks launched by torchrun",
)


@pytest.fixture(scope="module", autouse=True)
def distributed():
    maybe_init_distributed_environment_and_model_parallel(tp_size=2, sp_size=1)
    set_attn_backend("sdpa")
    yield


def _config() -> NEOLLMConfig:
    config = NEOLLMConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=128,
        pad_token_id=0,
        attention_bias=False,
        attention_dropout=0.0,
        use_sglang_tp=True,
    )
    config._attn_implementation = "eager"
    return config


def _load_full_weights(module: torch.nn.Module, full_state: dict[str, torch.Tensor]):
    with torch.no_grad():
        for name, param in module.named_parameters():
            loaded_weight = full_state[name]
            weight_loader = getattr(param, "weight_loader", None)
            if weight_loader is None:
                param.copy_(loaded_weight)
            else:
                weight_loader(param, loaded_weight)


@torch.no_grad()
def test_sensenova_u1_tp_components_match_tp1():
    torch.manual_seed(23)
    config = _config()
    with use_tensor_parallel_group(get_sp_group()):
        reference = Qwen3ForCausalLM(config).cuda().eval()
        for param in reference.parameters():
            torch.nn.init.normal_(param, std=0.02)

    with use_tensor_parallel_group(get_tp_group()):
        sharded = Qwen3ForCausalLM(config).cuda().eval()
    _load_full_weights(sharded, reference.state_dict())

    reference_layer = reference.model.layers[0]
    sharded_layer = sharded.model.layers[0]
    reference_attention = reference_layer.self_attn
    attention = sharded_layer.self_attn
    reference_mlp = reference_layer.mlp
    mlp = sharded_layer.mlp

    assert attention.q_proj.weight.shape == (32, 64)
    assert attention.k_proj.weight.shape == (16, 64)
    assert attention.o_proj.weight.shape == (64, 32)
    assert mlp.gate_proj.weight.shape == (64, 64)
    assert mlp.down_proj.weight.shape == (64, 64)

    hidden_states = torch.randn(2, 5, 64, device="cuda")
    indexes = torch.arange(5, device="cuda").view(1, 1, 5).expand(2, 3, 5)
    input_ids = torch.tensor([[0, 63, 64, 127]], device="cuda")

    expected_und, _ = reference_attention.forward_und(
        hidden_states, indexes, attention_mask=None
    )
    actual_und, _ = attention.forward_und(hidden_states, indexes, attention_mask=None)
    expected_gen, _ = reference_attention.forward_gen(
        hidden_states, indexes, attention_mask=None
    )
    actual_gen, _ = attention.forward_gen(hidden_states, indexes, attention_mask=None)

    torch.testing.assert_close(actual_und, expected_und, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(actual_gen, expected_gen, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(
        mlp(hidden_states), reference_mlp(hidden_states), atol=2e-5, rtol=2e-5
    )
    torch.testing.assert_close(
        sharded.model.embed_tokens(input_ids),
        reference.model.embed_tokens(input_ids),
        atol=2e-5,
        rtol=2e-5,
    )
    torch.testing.assert_close(
        sharded.lm_head(hidden_states)[0],
        reference.lm_head(hidden_states)[0],
        atol=2e-5,
        rtol=2e-5,
    )
