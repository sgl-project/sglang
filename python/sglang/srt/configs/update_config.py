from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sglang.srt.utils import (
    log_debug_on_rank0,
)

logger = logging.getLogger(__name__)
DEFAULT_MOE_PADDING_SIZE = 32


if TYPE_CHECKING:
    from sglang.srt.configs.load_config import LoadConfig
    from sglang.srt.configs.model_config import ModelConfig


def may_get_weight_block_size(model_config, load_config):
    from sglang.srt.model_loader.loader import _get_quantization_config

    quant_config = _get_quantization_config(model_config, load_config)

    if quant_config is not None and hasattr(quant_config, "weight_block_size"):
        return getattr(quant_config, "weight_block_size")

    if quant_config is not None and hasattr(quant_config, "group_size"):
        return [getattr(quant_config, "group_size")]

    return None


def get_moe_padding_size(weight_block_size):
    if weight_block_size is not None:
        # See NOTE(HandH1998): To ensure proper alignment of the block-wise quantization scales, the output_size of the weights for both the gate and up layers must be divisible by block_n.
        assert len(weight_block_size) in [
            1,
            2,
        ], "Only len(weight_block_size) in [1, 2] is supported"
        if len(weight_block_size) == 2:
            assert (
                weight_block_size[0] == weight_block_size[1]
            ), "Only weight_block_size[0] == weight_block_size[1] is supported"
        return weight_block_size[0]

    return DEFAULT_MOE_PADDING_SIZE


def get_num_heads_padding_size(tp_size, weight_block_size, head_dim=None):
    if head_dim is None:
        pad_size = (
            tp_size * 2
            if tp_size % 2 == 1 and weight_block_size is not None
            else tp_size
        )
        return pad_size
    pad_size = tp_size

    if weight_block_size is not None and head_dim % weight_block_size[0] != 0:
        import math

        pad_size = tp_size * (
            math.lcm(head_dim, weight_block_size[0]) // weight_block_size[0]
        )

    return pad_size


def resolve_head_dim(cfg, num_heads, is_text_config):
    # default getting head_dim by hidden_size and num_heads
    hidden_size = getattr(cfg, "hidden_size", getattr(cfg, "d_model", None))
    head_dim = hidden_size // num_heads if hidden_size else None
    # update head_dim if specified in model config
    if is_text_config:
        if hasattr(cfg.hf_config, "qk_head_dim"):
            head_dim = cfg.hf_config.qk_head_dim
        elif hasattr(cfg.hf_text_config, "head_dim"):
            head_dim = cfg.hf_text_config.head_dim
        elif hasattr(cfg.hf_config, "head_dim"):
            head_dim = cfg.hf_config.head_dim
    else:
        if hasattr(cfg, "head_dim"):
            head_dim = cfg.head_dim

    return head_dim


def adjust_tp_num_heads_if_necessary(model_config, tp_size, is_post_update):
    # is_post_update: whether to update an existing config
    from sglang.srt.layers.vocab_parallel_embedding import pad_vocab_size

    # Linear attn check logic
    if hasattr(model_config, "linear_num_key_heads") and hasattr(
        model_config, "linear_num_value_heads"
    ):
        if (
            model_config.linear_num_key_heads % tp_size != 0
            or model_config.linear_num_value_heads % tp_size != 0
        ):
            pad_size = tp_size
            linear_num_key_heads_cpu = pad_vocab_size(
                model_config.linear_num_key_heads, pad_size
            )
            linear_num_value_heads_cpu = (
                linear_num_key_heads_cpu
                * model_config.linear_num_value_heads
                // model_config.linear_num_key_heads
            )
            if is_post_update:
                update_config(
                    model_config, "linear_num_key_heads_cpu", linear_num_key_heads_cpu
                )
                update_config(
                    model_config,
                    "linear_num_value_heads_cpu",
                    linear_num_value_heads_cpu,
                )
            else:
                update_config(
                    model_config, "linear_num_key_heads", linear_num_key_heads_cpu
                )
                update_config(
                    model_config, "linear_num_value_heads", linear_num_value_heads_cpu
                )

        else:
            if is_post_update:
                update_config(
                    model_config,
                    "linear_num_key_heads_cpu",
                    model_config.linear_num_key_heads,
                )
                update_config(
                    model_config,
                    "linear_num_value_heads_cpu",
                    model_config.linear_num_value_heads,
                )


def update_intermediate_size(model_config, attr_name, intermediate_padding_size):
    attr_value = intermediate_padding_size
    if (
        hasattr(model_config, "hf_config")
        and hasattr(model_config.hf_config, "text_config")
        and hasattr(model_config.hf_config.text_config, attr_name)
    ):
        attr_value = getattr(model_config.hf_config.text_config, attr_name)
    elif hasattr(model_config, "hf_config") and hasattr(
        model_config.hf_config, attr_name
    ):
        attr_value = getattr(model_config.hf_config, attr_name)
    elif hasattr(model_config, attr_name):
        attr_value = getattr(model_config, attr_name)

    if attr_value % intermediate_padding_size != 0:
        from sglang.srt.layers.vocab_parallel_embedding import pad_vocab_size

        attr_value = pad_vocab_size(attr_value, intermediate_padding_size)
        if hasattr(model_config, "hf_config"):
            update_config(model_config.hf_config, attr_name, attr_value)
            if hasattr(model_config, "hf_text_config"):
                update_config(model_config.hf_text_config, attr_name, attr_value)
            if hasattr(model_config.hf_config, "text_config"):
                update_config(model_config.hf_config.text_config, attr_name, attr_value)
        else:
            update_config(model_config, attr_name, attr_value)

    return model_config


def update_config(model_config, attr_name, new_value):
    config_name = model_config.__class__.__name__
    if hasattr(model_config, attr_name):
        old_value = getattr(model_config, attr_name)
        if old_value != new_value:
            log_debug_on_rank0(
                logger,
                f"Updating {config_name}.{attr_name} from {old_value} to {new_value}",
            )
    else:
        log_debug_on_rank0(logger, f"Setting {config_name}.{attr_name} to {new_value}")
    setattr(model_config, attr_name, new_value)


def adjust_config_with_unaligned_cpu_tp(
    model_config: ModelConfig, load_config: LoadConfig, tp_size: int
) -> ModelConfig:
    # Support the case where the num_attention_heads is not divisible by the TP size.
    weight_block_size = may_get_weight_block_size(model_config, load_config)

    for config in [model_config.hf_config, model_config.hf_text_config]:
        update_config(
            config,
            "original_num_attention_heads",
            model_config.num_attention_heads,
        )
        update_config(
            config,
            "original_total_num_kv_heads",
            model_config.get_total_num_kv_heads(),
        )

    if (
        model_config.num_attention_heads % tp_size != 0
        or model_config.get_total_num_kv_heads() % tp_size != 0
    ):

        if hasattr(model_config.hf_config, "qk_nope_head_dim") and hasattr(
            model_config.hf_config, "qk_rope_head_dim"
        ):
            update_config(
                model_config.hf_config,
                "qk_head_dim",
                model_config.hf_config.qk_nope_head_dim
                + model_config.hf_config.qk_rope_head_dim,
            )

        query_heads_per_kv = (
            model_config.num_attention_heads // model_config.get_total_num_kv_heads()
        )
        total_kv_heads = model_config.get_total_num_kv_heads()
        from sglang.srt.layers.vocab_parallel_embedding import pad_vocab_size

        head_dim = resolve_head_dim(
            model_config, model_config.num_attention_heads, True
        )

        pad_size = get_num_heads_padding_size(tp_size, weight_block_size, head_dim)
        num_key_value_heads = pad_vocab_size(total_kv_heads, pad_size)

        num_attention_heads = num_key_value_heads * query_heads_per_kv
        for config in [
            model_config,
            model_config.hf_config,
            model_config.hf_text_config,
        ]:
            update_config(config, "num_key_value_heads", num_key_value_heads)
            update_config(config, "num_attention_heads", num_attention_heads)

    adjust_tp_num_heads_if_necessary(model_config.hf_config, tp_size, True)
    if hasattr(model_config.hf_config, "text_config"):
        adjust_tp_num_heads_if_necessary(
            model_config.hf_config.text_config, tp_size, True
        )

    intermediate_padding_size = tp_size * get_moe_padding_size(weight_block_size)
    for moe_intermediate_attr in [
        "moe_intermediate_size",
        "intermediate_size",
        "intermediate_size_mlp",
        "shared_expert_intermediate_size",
    ]:
        model_config = update_intermediate_size(
            model_config, moe_intermediate_attr, intermediate_padding_size
        )

    multimodal_config = [
        [
            model_config.hf_config,
            "vision_config",
            "siglip_vision_model",
            "num_attention_heads",
        ],
        [model_config.hf_config, "vision_config", "qwen3_vl_moe", "num_heads"],
        [model_config.hf_config, "vision_config", "qwen3_vl", "num_heads"],
        [model_config.hf_config, "vision_config", "qwen3_5_moe", "num_heads"],
        [model_config.hf_config, "vision_config", "qwen3_5", "num_heads"],
    ]
    if hasattr(model_config.hf_config, "thinker_config"):
        multimodal_config.append(
            [
                model_config.hf_config.thinker_config,
                "vision_config",
                "qwen3_omni_moe_vision_encoder",
                "num_heads",
            ]
        )
        multimodal_config.append(
            [
                model_config.hf_config.thinker_config,
                "audio_config",
                "qwen3_omni_moe_audio_encoder",
                "encoder_attention_heads",
            ]
        )

    for m_config, config_name, model_type, num_head_str in multimodal_config:
        if (
            hasattr(m_config, config_name)
            and getattr(m_config, config_name).model_type == model_type
        ):
            num_heads = getattr(getattr(m_config, config_name), num_head_str)
            update_config(
                getattr(m_config, config_name), "original_" + num_head_str, num_heads
            )
            if num_heads % tp_size != 0:
                from sglang.srt.layers.vocab_parallel_embedding import pad_vocab_size

                multimodal_head_dim = resolve_head_dim(
                    getattr(m_config, config_name), num_heads, False
                )
                pad_size = get_num_heads_padding_size(
                    tp_size, weight_block_size, multimodal_head_dim
                )
                new_num_heads = pad_vocab_size(num_heads, pad_size)
                update_config(
                    getattr(m_config, config_name), num_head_str, new_num_heads
                )
            setattr(
                m_config,
                config_name,
                update_intermediate_size(
                    getattr(m_config, config_name),
                    "intermediate_size",
                    intermediate_padding_size,
                ),
            )

    return model_config
