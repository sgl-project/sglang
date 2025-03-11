from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig

from sglang.srt.layers.vocab_parallel_embedding import pad_vocab_size

DEFAULT_MOE_PADDING_SIZE = 32


def update_config(model_config: ModelConfig, tp_size: int) -> ModelConfig:
    # Support the case where the num_attention_heads is not divisible by the TP size.
    if model_config.num_attention_heads % tp_size != 0:
        query_heads_per_kv = (
            model_config.num_attention_heads // model_config.get_total_num_kv_heads()
        )
        total_kv_heads = model_config.get_total_num_kv_heads()
        num_key_value_heads = pad_vocab_size(total_kv_heads, tp_size)
        model_config.num_key_value_heads = num_key_value_heads
        model_config.hf_config.num_key_value_heads = num_key_value_heads
        model_config.hf_text_config.num_key_value_heads = num_key_value_heads

        num_attention_heads = num_key_value_heads * query_heads_per_kv
        model_config.num_attention_heads = num_attention_heads
        model_config.hf_config.num_attention_heads = num_attention_heads
        model_config.hf_text_config.num_attention_heads = num_attention_heads

        moe_intermediate_size = model_config.hf_config.moe_intermediate_size
        moe_intermediate_size = pad_vocab_size(
            moe_intermediate_size, tp_size * DEFAULT_MOE_PADDING_SIZE
        )
        model_config.hf_config.moe_intermediate_size = moe_intermediate_size
        model_config.hf_text_config.moe_intermediate_size = moe_intermediate_size

    return model_config


def get_actual_shard_size(shard_size, weight_start, weight_end):
    return min(shard_size, weight_end - weight_start)


def reset_param_data_if_needed(param_data, dim, start, length):
    if length == 0:
        return

    assert length > 0, f"Length should be positive, but got {length}"

    param_data.narrow(dim, start, length).zero_()
    return
