from __future__ import annotations

from typing import TYPE_CHECKING

DEFAULT_MOE_PADDING_SIZE = 32


if TYPE_CHECKING:
    from sglang.srt.configs.load_config import LoadConfig
    from sglang.srt.configs.model_config import ModelConfig


def may_get_weight_block_size(model_config, load_config):
    from sglang.srt.model_loader.loader import _get_quantization_config
    from sglang.srt.model_loader.utils import get_model_architecture

    model_class, _ = get_model_architecture(model_config)
    packed_modules_mapping = getattr(model_class, "packed_modules_mapping", {})

    quant_config = _get_quantization_config(
        model_config, load_config, packed_modules_mapping
    )

    if quant_config is not None and hasattr(quant_config, "weight_block_size"):
        return getattr(quant_config, "weight_block_size")
    return None


def get_moe_padding_size(weight_block_size):
    if weight_block_size is not None:
        # See NOTE(HandH1998): To ensure proper alignment of the block-wise quantization scales, the output_size of the weights for both the gate and up layers must be divisible by block_n.
        assert (
            len(weight_block_size) == 2
        ), "Only len(weight_block_size) == 2 is supported"
        assert (
            weight_block_size[0] == weight_block_size[1]
        ), "Only weight_block_size[0] == weight_block_size[1] is supported"

        return weight_block_size[0]

    return DEFAULT_MOE_PADDING_SIZE


def get_num_heads_padding_size(tp_size, weight_block_size):
    pad_size = (
        tp_size * 2 if tp_size % 2 == 1 and weight_block_size is not None else tp_size
    )
    return pad_size


def update_intermediate_size(model_config, attr_name, intermediate_padding_size):
    if hasattr(model_config.hf_config, attr_name):
        attr_value = getattr(model_config.hf_config, attr_name)
        if attr_value % intermediate_padding_size != 0:
            from sglang.srt.layers.vocab_parallel_embedding import pad_vocab_size

            attr_value = pad_vocab_size(attr_value, intermediate_padding_size)
            setattr(model_config.hf_config, attr_name, attr_value)
            setattr(model_config.hf_text_config, attr_name, attr_value)
    return model_config


def adjust_config_with_unaligned_cpu_tp(
    model_config: ModelConfig, load_config: LoadConfig, tp_size: int
) -> ModelConfig:
    # Support the case where the num_attention_heads is not divisible by the TP size.
    weight_block_size = may_get_weight_block_size(model_config, load_config)

    model_config.hf_config.original_num_attention_heads = (
        model_config.num_attention_heads
    )
    model_config.hf_text_config.original_num_attention_heads = (
        model_config.num_attention_heads
    )

    model_config.hf_config.original_total_num_kv_heads = (
        model_config.get_total_num_kv_heads()
    )
    model_config.hf_text_config.original_total_num_kv_heads = (
        model_config.get_total_num_kv_heads()
    )

    if (
        model_config.num_attention_heads % tp_size != 0
        or model_config.get_total_num_kv_heads() % tp_size != 0
    ):
        # Compute the head_dim using the model_config.num_attention_heads before padding
        if not hasattr(model_config.hf_config, "head_dim"):
            model_config.hf_config.head_dim = (
                model_config.hidden_size // model_config.num_attention_heads
            )

        query_heads_per_kv = (
            model_config.num_attention_heads // model_config.get_total_num_kv_heads()
        )
        total_kv_heads = model_config.get_total_num_kv_heads()
        from sglang.srt.layers.vocab_parallel_embedding import pad_vocab_size

        pad_size = get_num_heads_padding_size(tp_size, weight_block_size)
        num_key_value_heads = pad_vocab_size(total_kv_heads, pad_size)

        model_config.num_key_value_heads = num_key_value_heads
        model_config.hf_config.num_key_value_heads = num_key_value_heads
        model_config.hf_text_config.num_key_value_heads = num_key_value_heads

        num_attention_heads = num_key_value_heads * query_heads_per_kv
        model_config.num_attention_heads = num_attention_heads
        model_config.hf_config.num_attention_heads = num_attention_heads
        model_config.hf_text_config.num_attention_heads = num_attention_heads

    intermediate_padding_size = tp_size * get_moe_padding_size(weight_block_size)
    model_config = update_intermediate_size(
        model_config, "moe_intermediate_size", intermediate_padding_size
    )
    model_config = update_intermediate_size(
        model_config, "intermediate_size", intermediate_padding_size
    )

    return model_config
