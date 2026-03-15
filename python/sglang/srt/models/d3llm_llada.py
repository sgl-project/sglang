"""SGLang d3LLM-LLaDA model (LLaDA-8B with full bidirectional attention).

d3LLM-LLaDA is architecturally identical to LLaMA but uses bidirectional
(non-causal) attention and a custom weight naming scheme. This class extends
LlamaForCausalLM, patches config fields, sets encoder-only attention, and
provides weight name remapping.

Weight mapping (d3LLM-LLaDA → sglang Llama):
  model.transformer.wte.weight          → model.embed_tokens.weight
  model.transformer.ln_f.weight         → model.norm.weight
  model.transformer.ff_out.weight       → lm_head.weight
  blocks.{i}.q_proj / k_proj / v_proj   → layers.{i}.self_attn.qkv_proj (sharded)
  blocks.{i}.attn_out                   → layers.{i}.self_attn.o_proj
  blocks.{i}.attn_norm                  → layers.{i}.input_layernorm
  blocks.{i}.ff_proj                    → layers.{i}.mlp.gate_up_proj (shard 0)
  blocks.{i}.up_proj                    → layers.{i}.mlp.gate_up_proj (shard 1)
  blocks.{i}.ff_out                     → layers.{i}.mlp.down_proj
  blocks.{i}.ff_norm                    → layers.{i}.post_attention_layernorm
"""

from typing import Iterable, Optional, Tuple

import torch

from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.models.llama import LlamaForCausalLM


def _patch_llada_config(config):
    """Add Llama-compatible attributes missing from LLaDAConfig."""
    if not hasattr(config, "num_key_value_heads") or config.num_key_value_heads is None:
        config.num_key_value_heads = getattr(
            config, "n_kv_heads", config.num_attention_heads
        )
    if not hasattr(config, "intermediate_size") or config.intermediate_size is None:
        config.intermediate_size = getattr(
            config, "mlp_hidden_size", config.hidden_size * 4
        )
    if (
        not hasattr(config, "max_position_embeddings")
        or config.max_position_embeddings is None
    ):
        config.max_position_embeddings = getattr(config, "max_sequence_length", 4096)
    if not hasattr(config, "hidden_act"):
        config.hidden_act = "silu"
    if not hasattr(config, "head_dim"):
        config.head_dim = config.hidden_size // config.num_attention_heads
    if not hasattr(config, "rope_scaling"):
        config.rope_scaling = None
    if not hasattr(config, "rope_theta"):
        config.rope_theta = 500000.0
    # LLaDA has separate lm_head (ff_out) despite PretrainedConfig defaulting True
    config.tie_word_embeddings = False
    if not hasattr(config, "vocab_size") or config.vocab_size is None:
        config.vocab_size = getattr(config, "embedding_size", 126464)
    return config


# --- Weight name remapping ------------------------------------------------

# Global (non-layer) weights
_GLOBAL_MAP = {
    "model.transformer.wte.weight": "model.embed_tokens.weight",
    "model.transformer.ln_f.weight": "model.norm.weight",
    "model.transformer.ff_out.weight": "lm_head.weight",
}

# Per-layer stacked params: (llada_suffix, sglang_suffix, shard_id)
_STACKED = [
    (".q_proj.", ".self_attn.q_proj.", "q"),
    (".k_proj.", ".self_attn.k_proj.", "k"),
    (".v_proj.", ".self_attn.v_proj.", "v"),
    (".ff_proj.", ".mlp.gate_proj.", 0),
    (".up_proj.", ".mlp.up_proj.", 1),
]

# Per-layer simple renames: (llada_suffix, sglang_suffix)
_SIMPLE = [
    (".attn_out.", ".self_attn.o_proj."),
    (".attn_norm.", ".input_layernorm."),
    (".ff_norm.", ".post_attention_layernorm."),
    (".ff_out.", ".mlp.down_proj."),
]


def _remap_name(name: str):
    """Convert a LLaDA weight name to sglang Llama name + optional shard id."""
    if name in _GLOBAL_MAP:
        return _GLOBAL_MAP[name], None

    # blocks.N → layers.N
    name = name.replace("model.transformer.blocks.", "model.layers.")

    for old, new, sid in _STACKED:
        if old in name:
            return name.replace(old, new), sid

    for old, new in _SIMPLE:
        if old in name:
            return name.replace(old, new), None

    return name, None


class LLaDAModelLM(LlamaForCausalLM):
    def __init__(self, config, quant_config=None, prefix=""):
        config = _patch_llada_config(config)
        super().__init__(config, quant_config, prefix)
        # dLLM needs full logits over all tokens (not just the last one).
        self.logits_processor = LogitsProcessor(config, return_full_logits=True)
        # LLaDA uses full bidirectional (non-causal) attention.
        for layer in self.model.layers:
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "attn"):
                layer.self_attn.attn.attn_type = AttentionType.ENCODER_ONLY

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        if self.pp_group.is_last_rank and not get_embedding:
            # LLaDA predicts the current position directly (no right-shift).
            return self.logits_processor(
                input_ids,
                hidden_states,
                self.lm_head,
                forward_batch,
            )

        if get_embedding:
            return self.pooler(hidden_states, forward_batch)
        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load LLaDA checkpoint weights with name remapping to Llama layout."""
        # Llama fused-parameter mapping used by parent's weight loaders
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights:
            name, shard_id = _remap_name(name)
            if "rotary_emb.inv_freq" in name:
                continue

            if shard_id is not None:
                # Fuse into QKV or gate_up
                for fused, single, sid in stacked_params_mapping:
                    if single in name:
                        name = name.replace(single, fused)
                        break
                if name not in params_dict:
                    continue
                param = params_dict[name]
                param.weight_loader(param, loaded_weight, shard_id)
            else:
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(
                    param, "weight_loader", lambda p, w: p.data.copy_(w)
                )
                weight_loader(param, loaded_weight)


EntryClass = LLaDAModelLM
