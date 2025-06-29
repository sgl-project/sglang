"""Shared utilities for Llama-based models."""

# Mapping of checkpoint weight names to stacked parameter names for Llama models.
# Each tuple is (param_name, shard_name, shard_id).
LLAMA_STACKED_PARAMS_MAPPING = [
    (".qkv_proj", ".q_proj", "q"),
    (".qkv_proj", ".k_proj", "k"),
    (".qkv_proj", ".v_proj", "v"),
    (".gate_up_proj", ".gate_proj", 0),
    (".gate_up_proj", ".up_proj", 1),
]

# Extended mapping used by Llama4-based architectures with explicit module names.
# Each tuple is (param_name, shard_name, shard_id).
LLAMA4_STACKED_PARAMS_MAPPING = [
    (".self_attn.qkv_proj", ".self_attn.q_proj", "q"),
    (".self_attn.qkv_proj", ".self_attn.k_proj", "k"),
    (".self_attn.qkv_proj", ".self_attn.v_proj", "v"),
    (".shared_expert.gate_up_proj", ".shared_expert.gate_proj", 0),
    (".shared_expert.gate_up_proj", ".shared_expert.up_proj", 1),
    (".feed_forward.gate_up_proj", ".feed_forward.gate_proj", 0),
    (".feed_forward.gate_up_proj", ".feed_forward.up_proj", 1),
]
