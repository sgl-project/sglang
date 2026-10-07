# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify.configuration_neo_chat import (
    NEOLLMConfig,
    NEOMoELLMConfig,
)
from sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify.modeling_qwen3 import (
    Qwen3ForCausalLM,
)
from sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify.modeling_qwen3_moe import (
    Qwen3MoeForCausalLM,
    Qwen3MoeMLP,
    Qwen3MoeSparseMoeBlock,
)


@pytest.mark.parametrize("moe", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_sensenova_backbone_prefix_cache_and_generation(moe, dtype):
    config = (NEOMoELLMConfig if moe else NEOLLMConfig)(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        max_position_embeddings=32,
        max_position_embeddings_hw=32,
        num_experts=3,
        num_experts_per_tok=2,
        moe_intermediate_size=24,
        gen_num_experts=2,
        gen_num_experts_per_tok=1,
        gen_moe_intermediate_size=16,
        mlp_only_layers=[0],
        decoder_sparse_step=1,
    )
    config._attn_implementation = "eager"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = (
        (Qwen3MoeForCausalLM if moe else Qwen3ForCausalLM)(config)
        .to(device=device, dtype=dtype)
        .eval()
    )
    assert model._can_compile_fullgraph is (not moe)
    assert model._can_record_outputs["hidden_states"] is type(model.model.layers[0])
    if moe:
        assert isinstance(model.model.layers[0].mlp, Qwen3MoeMLP)
        assert isinstance(model.model.layers[1].mlp, Qwen3MoeSparseMoeBlock)
        assert all(layer.mlp_mot_gen.num_experts == 2 for layer in model.model.layers)
    tokens = torch.tensor([[1, 2, 3]], device=device)
    prefix = model(
        input_ids=tokens, labels=tokens, use_cache=True, output_hidden_states=True
    )
    assert prefix.logits.shape == (1, 3, 32) and prefix.loss.isfinite()
    assert len(prefix.hidden_states) == config.num_hidden_layers + 1
    cache = prefix.past_key_values
    snapshots = [(layer.keys.clone(), layer.values.clone()) for layer in cache.layers]
    image = torch.randn(1, 4, 32, device=device, dtype=dtype)
    indexes = torch.tensor([[[3, 3, 3, 3], [0, 0, 1, 1], [0, 1, 0, 1]]], device=device)
    kwargs = dict(
        inputs_embeds=image,
        indexes=indexes,
        past_key_values=cache,
        use_cache=False,
        update_cache=False,
        image_gen_indicators=torch.ones(1, 4, dtype=torch.bool, device=device),
    )
    first = model.model(**kwargs).last_hidden_state
    second = model.model(**kwargs).last_hidden_state
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    for layer, expected in zip(cache.layers, snapshots):
        torch.testing.assert_close((layer.keys, layer.values), expected, rtol=0, atol=0)
    kwargs["image_gen_indicators"][0, 0] = False
    with pytest.raises(NotImplementedError, match="Mixed und/gen"):
        model.model(**kwargs)
