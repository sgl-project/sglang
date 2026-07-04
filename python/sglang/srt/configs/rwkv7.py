# coding=utf-8
"""RWKV-7 (Goose) model configuration for sglang.

Mirrors the fla-format checkpoint config.json (model_type="rwkv7"). RWKV-7 is an
all-linear-attention recurrent model: every layer is a "linear" layer, there are
no full-attention layers. We reuse sglang's Mamba/hybrid-linear state plumbing by
exposing `mamba2_cache_params`, `linear_layer_ids` (= all layers), and
`full_attention_layer_ids` (= []).
"""

import torch
from transformers.configuration_utils import PretrainedConfig

from sglang.srt.configs.mamba_utils import (
    Mamba2StateDType,
    Rwkv7CacheParams,
    Rwkv7StateShape,
)


class Rwkv7Config(PretrainedConfig):
    model_type = "rwkv7"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=65536,
        hidden_size=768,
        num_hidden_layers=12,
        head_dim=64,
        num_heads=12,
        decay_low_rank_dim=64,
        a_low_rank_dim=64,
        v_low_rank_dim=32,
        gate_low_rank_dim=128,
        intermediate_size=3072,
        hidden_ratio=4.0,
        hidden_act="sqrelu",
        norm_eps=1e-5,
        norm_bias=True,
        norm_first=True,
        max_position_embeddings=8192,
        tie_word_embeddings=False,
        attn=None,
        attn_mode="chunk",
        bos_token_id=0,
        eos_token_id=0,
        use_cache=True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.head_dim = head_dim
        # fla-hub checkpoints ship "num_heads": null (and may omit
        # intermediate_size) — derive them the way the fla reference does.
        if head_dim is None:
            raise ValueError("Rwkv7Config requires head_dim (checkpoint config)")
        if num_heads is None:
            num_heads = hidden_size // head_dim
        self.num_heads = num_heads
        self.decay_low_rank_dim = decay_low_rank_dim
        self.a_low_rank_dim = a_low_rank_dim
        self.v_low_rank_dim = v_low_rank_dim
        self.gate_low_rank_dim = gate_low_rank_dim
        if intermediate_size is None:
            intermediate_size = 32 * -(-int(hidden_size * hidden_ratio) // 32)
        self.intermediate_size = intermediate_size
        self.hidden_ratio = hidden_ratio
        self.hidden_act = hidden_act
        self.norm_eps = norm_eps
        self.norm_bias = norm_bias
        if not norm_first:
            raise NotImplementedError(
                "RWKV-7 checkpoints with norm_first=False are not supported"
            )
        self.norm_first = norm_first
        self.max_position_embeddings = max_position_embeddings
        if attn is not None:
            raise NotImplementedError(
                "fla hybrid-attention RWKV-7 variants (config 'attn' set) are "
                "not supported by this implementation"
            )
        self.attn = attn
        self.attn_mode = attn_mode
        self.use_cache = use_cache

        # ---- Standard HF/sglang ModelConfig fields (derived) ----
        # ModelConfig reads num_attention_heads / num_key_value_heads / head_dim.
        self.num_attention_heads = num_heads
        self.num_key_value_heads = num_heads

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    # ---- Hybrid-linear plumbing (RWKV-7 = every layer linear) ----
    @property
    def layers_block_type(self):
        return ["linear_attention"] * self.num_hidden_layers

    @property
    def linear_layer_ids(self):
        return list(range(self.num_hidden_layers))

    @property
    def full_attention_layer_ids(self):
        return []

    @property
    def mamba2_cache_params(self) -> Rwkv7CacheParams:
        from sglang.srt.layers.dp_attention import get_attention_tp_size

        shape = Rwkv7StateShape.create(
            tp_world_size=get_attention_tp_size(),
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )
        # Keep BOTH conv (token-shift) and temporal (recurrent S) states in fp32:
        # the default bf16 conv dtype would round the token-shift values and break
        # exact greedy reproduction against the reference implementation.
        dtype = Mamba2StateDType(conv=torch.float32, temporal=torch.float32)
        return Rwkv7CacheParams(shape=shape, layers=self.linear_layer_ids, dtype=dtype)
