"""Model shape the all-reduce tuner sizes its messages from."""

from typing import Dict

import torch

from sglang.srt.hf_transformers import get_config


def get_model_config(model_path: str, tp_size: int, ep_size: int = 1) -> Dict:
    config = get_config(model_path, trust_remote_code=True)
    architecture = config.architectures[0]
    block_shape = None
    if (
        hasattr(config, "quantization_config")
        and "weight_block_size" in config.quantization_config
    ):
        block_shape = config.quantization_config["weight_block_size"]
    if (
        hasattr(config, "quantization_config")
        and "config_groups" in config.quantization_config
    ):
        config_groups = config.quantization_config["config_groups"]
        first_group = next(iter(config_groups.values()), {})
        weights_config = first_group.get("weights", {})
        group_size = weights_config.get("group_size")
        block_shape = [0, group_size]

    if hasattr(config, "text_config"):
        from types import SimpleNamespace
        text_config = config.get_text_config()
        if isinstance(text_config, dict):
            config = SimpleNamespace(**text_config)
        else:
            config = text_config

    hidden_size = config.hidden_size
    architecture = getattr(config, "architectures", [architecture])[0]

    moe_attrs = ["num_experts", "num_local_experts", "n_routed_experts", "moe_intermediate_size"]
    has_moe = any(hasattr(config, attr) for attr in moe_attrs)

    if not has_moe:
        return {
            "num_experts": 0, "topk": 0, "hidden_size": hidden_size,
            "shard_intermediate_size": 0,
            "dtype": getattr(config, "torch_dtype", None) or torch.bfloat16,
            "block_shape": block_shape, "architecture": architecture, "is_moe": False,
        }

    if architecture == "DbrxForCausalLM":
        E = config.ffn_config.moe_num_experts // ep_size
        topk = config.ffn_config.moe_top_k
        intermediate_size = config.ffn_config.ffn_hidden_size
    elif architecture == "JambaForCausalLM":
        E = config.num_experts // ep_size
        topk = config.num_experts_per_tok
        intermediate_size = config.intermediate_size
    elif architecture in [
        "Qwen2MoeForCausalLM", "Qwen3MoeForCausalLM", "Qwen3NextForCausalLM",
        "Qwen3VLMoeForConditionalGeneration", "Qwen3_5MoeForConditionalGeneration",
        "InternS2PreviewForConditionalGeneration", "MellumForCausalLM",
    ]:
        E = config.num_experts // ep_size
        topk = config.num_experts_per_tok
        intermediate_size = config.moe_intermediate_size
    elif architecture in [
        "DeepseekV2ForCausalLM", "DeepseekV3ForCausalLM", "DeepseekV32ForCausalLM",
        "DeepseekV4ForCausalLM", "Glm4MoeForCausalLM", "GlmMoeDsaForCausalLM",
        "KimiVLForConditionalGeneration", "MistralLarge3ForCausalLM",
    ]:
        E = config.n_routed_experts // ep_size
        topk = config.num_experts_per_tok
        intermediate_size = config.moe_intermediate_size
    elif architecture in ["Llama4ForConditionalGeneration", "Grok1ForCausalLM", "Grok1ImgGen", "Grok1AForCausalLM"]:
        E = config.num_local_experts // ep_size
        topk = config.num_experts_per_tok
        intermediate_size = config.moe_intermediate_size if architecture != "Llama4ForConditionalGeneration" else config.intermediate_size
    elif architecture in ["BailingMoEForCausalLM", "BailingMoeForCausalLM", "BailingMoeV2ForCausalLM"]:
        E = config.num_experts // ep_size
        topk = config.num_experts_per_tok
        intermediate_size = config.moe_intermediate_size
    elif architecture == "HYV3ForCausalLM":
        E = config.num_experts // ep_size
        topk = config.num_experts_per_tok
        intermediate_size = config.expert_hidden_dim
    elif architecture == "NemotronHForCausalLM":
        E = config.n_routed_experts // ep_size
        topk = config.num_experts_per_tok
        intermediate_size = config.moe_intermediate_size
        hidden_size = getattr(config, "moe_latent_size", None) or hidden_size
    elif architecture == "Gemma4ForConditionalGeneration":
        E = config.num_experts // ep_size
        topk = config.top_k_experts
        intermediate_size = config.moe_intermediate_size
    elif architecture == "Lfm2MoeForCausalLM":
        E = config.num_experts // ep_size
        topk = config.num_experts_per_tok
        intermediate_size = config.moe_intermediate_size
    elif architecture == "MiniMaxM3SparseForConditionalGeneration":
        E = config.num_local_experts // ep_size
        topk = config.num_experts_per_tok
        intermediate_size = config.intermediate_size
    elif architecture == "UnlimitedOCRForCausalLM":
        E = config.n_routed_experts // ep_size
        topk = config.num_experts_per_tok
        intermediate_size = config.moe_intermediate_size
    else:
        if hasattr(config, "num_local_experts"):
            E = config.num_local_experts // ep_size
        elif hasattr(config, "num_experts"):
            E = config.num_experts // ep_size
        else:
            E = 0
        topk = getattr(config, "num_experts_per_tok", 0)
        intermediate_size = getattr(config, "moe_intermediate_size", getattr(config, "intermediate_size", 0))

    shard_intermediate_size = calculate_shard_intermediate_size(intermediate_size, tp_size, ep_size)
    torch_dtype = getattr(config, "torch_dtype", None) or torch.bfloat16

    return {
        "num_experts": E, "topk": topk, "hidden_size": hidden_size,
        "shard_intermediate_size": shard_intermediate_size,
        "dtype": torch_dtype, "block_shape": block_shape,
        "architecture": architecture, "is_moe": True,
    }


def calculate_shard_intermediate_size(intermediate_size: int, tp_size: int, ep_size: int = 1) -> int:
    assert tp_size % ep_size == 0
    moe_tp_size = tp_size // ep_size
    assert intermediate_size % moe_tp_size == 0
    return 2 * intermediate_size // moe_tp_size


