import pytest
import torch
import torch.nn.functional as F
from transformers import Ministral3Config
from transformers.models.ministral3.modeling_ministral3 import (
    Ministral3ForCausalLM as HFMinistral3ForCausalLM,
)

from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import SDPAImpl
from sglang.multimodal_gen.runtime.layers.attention.selector import (
    global_force_attn_backend_context_manager,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.pe_loader import (
    PEModelWrapper,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.encoders.mistral_3 import (
    Ministral3ForCausalLM,
    _get_llama_4_attn_scale,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


def _config() -> Ministral3Config:
    return Ministral3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        rope_parameters={
            "rope_type": "yarn",
            "rope_theta": 1_000_000.0,
            "factor": 4.0,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "llama_4_scaling_beta": 0.1,
            "original_max_position_embeddings": 16,
        },
        tie_word_embeddings=True,
        use_cache=True,
    )


def test_ministral3_query_scale_matches_llama4_rule():
    position_ids = torch.tensor([[0, 15, 16, 32]])

    scale = _get_llama_4_attn_scale(position_ids, 0.1, 16)

    expected = 1 + 0.1 * torch.log(1 + torch.floor(position_ids.float() / 16))
    torch.testing.assert_close(scale[:, 0, :, 0], expected)


@pytest.mark.parametrize("query_length", [1, 2])
def test_causal_sdpa_uses_lower_right_alignment_for_cached_keys(query_length):
    torch.manual_seed(11)
    key_length = 5
    query = torch.randn(1, query_length, 2, 8)
    key = torch.randn(1, key_length, 2, 8)
    value = torch.randn(1, key_length, 2, 8)
    attention = SDPAImpl(
        num_heads=2,
        head_size=8,
        causal=True,
        softmax_scale=8**-0.5,
    )

    actual = attention.forward(query, key, value, attn_metadata=None)

    mask = None
    if query_length > 1:
        mask = torch.ones(query_length, key_length, dtype=torch.bool).tril(
            diagonal=key_length - query_length
        )
    expected = F.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        attn_mask=mask,
        is_causal=False,
        scale=8**-0.5,
    ).transpose(1, 2)
    torch.testing.assert_close(actual, expected)


def test_native_ministral3_matches_hf_prefill_and_generation():
    torch.manual_seed(7)
    with global_force_attn_backend_context_manager(AttentionBackendEnum.TORCH_SDPA):
        config = _config()
        reference = HFMinistral3ForCausalLM(config).eval()
        native = Ministral3ForCausalLM(config).eval()
        native.load_state_dict(reference.state_dict(), strict=True)
        input_ids = (torch.arange(20).unsqueeze(0) + 1) % config.vocab_size

        context = set_forward_context(current_timestep=0, attn_metadata=None)
        with torch.no_grad(), context:
            reference_output = reference(input_ids=input_ids, use_cache=True)
            native_output = native(input_ids=input_ids, use_cache=True)

        torch.testing.assert_close(native_output.logits, reference_output.logits)
        assert len(native_output.past_key_values.layers) == config.num_hidden_layers

        with torch.no_grad(), set_forward_context(
            current_timestep=0, attn_metadata=None
        ):
            native_ids = native.generate(input_ids, max_new_tokens=2, do_sample=False)
        with torch.no_grad():
            reference_ids = reference.generate(
                input_ids, max_new_tokens=2, do_sample=False
            )
        torch.testing.assert_close(native_ids, reference_ids)


def test_native_ministral3_exposes_decoder_layers_for_offload():
    assert Ministral3ForCausalLM.layer_names == ["model.layers"]


def test_pe_wrapper_exposes_native_decoder_layers_for_offload():
    with global_force_attn_backend_context_manager(AttentionBackendEnum.TORCH_SDPA):
        model = Ministral3ForCausalLM(_config())
    wrapper = PEModelWrapper(
        model=model,
        tokenizer=None,
        device=torch.device("cpu"),
        model_max_length=128,
    )

    assert PEModelWrapper.layer_names == ["model.model.layers"]
    decoder_layers = dict(wrapper.named_modules())["model.model.layers"]
    assert isinstance(decoder_layers, torch.nn.ModuleList)
