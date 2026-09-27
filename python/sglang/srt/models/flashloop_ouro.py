# SPDX-License-Identifier: MIT
# Copyright (c) 2026 FlashLoop contributors
"""Ouro weights and residual flow executed by SGLang native layers.

Weights are shared across four traversals; paged KV layer IDs are distinct.
No Hugging Face model forward or FlashLoopEngine forward is called here.
"""

import torch
from torch import nn

from sglang.srt.configs.flashloop import component_options
from sglang.srt.layers.attention.flashloop.attention import FlashLoopAttention
from sglang.srt.layers.attention.flashloop.prefill import select_rows
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.llama import LlamaAttention, LlamaMLP


class OuroBlock(nn.Module):
    def __init__(self, config, index, quant_config, prefix):
        super().__init__()
        self.self_attn = LlamaAttention(
            config,
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            layer_id=index,
            rope_theta=config.rope_theta,
            rope_scaling=getattr(config, "rope_scaling", None),
            max_position_embeddings=config.max_position_embeddings,
            quant_config=quant_config,
            prefix=prefix + ".self_attn",
        )
        attention = self.self_attn
        attention.attn = FlashLoopAttention(
            attention.num_heads,
            attention.head_dim,
            attention.scaling,
            num_kv_heads=attention.num_kv_heads,
            layer_id=index,
            fraction=component_options(config)[1],
        )
        self.mlp = LlamaMLP(
            config.hidden_size,
            config.intermediate_size,
            config.hidden_act,
            quant_config,
            prefix + ".mlp",
        )
        for name in (
            "input_layernorm",
            "input_layernorm_2",
            "post_attention_layernorm",
            "post_attention_layernorm_2",
        ):
            setattr(self, name, RMSNorm(config.hidden_size, eps=config.rms_norm_eps))

    def forward(self, hidden, positions, batch):
        update = self.self_attn(positions, self.input_layernorm(hidden), batch)
        hidden = hidden + self.input_layernorm_2(update)
        update = self.mlp(self.post_attention_layernorm(hidden))
        return hidden + self.post_attention_layernorm_2(update)


class FlashLoopOuroForCausalLM(nn.Module):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        if quant_config is not None:
            raise ValueError("Weight quantization is not validated for this backend")
        self.config = config
        self.prefill_fractions, _, _ = component_options(config)
        count = config.flashloop_physical_layers
        self.start_layer, self.end_layer = 0, count * 4
        self.model = nn.Module()
        self.model.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.model.layers = nn.ModuleList(
            [
                OuroBlock(config, i, quant_config, f"model.layers.{i}")
                for i in range(count)
            ]
        )
        self.model.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.model.early_exit_gate = ReplicatedLinear(config.hidden_size, 1, bias=True)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.tie_weights(self.model.embed_tokens)
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(self, input_ids, positions, forward_batch, input_embeds=None, **kwargs):
        hidden = (
            self.model.embed_tokens(input_ids) if input_embeds is None else input_embeds
        )
        fractions = self.prefill_fractions
        sparse_prefill = forward_batch.forward_mode.is_extend() and min(fractions) < 1
        if sparse_prefill and any(forward_batch.extend_prefix_lens_cpu):
            raise ValueError(
                "Sparse prefill requires full prompts: disable prefix cache and chunking"
            )
        previous, eligible = None, None
        for loop in range(4):
            selection = None
            before = hidden
            if sparse_prefill and loop >= 2:
                selection = select_rows(
                    hidden,
                    previous,
                    forward_batch.extend_seq_lens_cpu,
                    fractions[loop - 2],
                    eligible,
                )
                indices, starts, counts, max_count = selection
                active_positions = positions[indices]
                active_hidden = hidden[indices]
                eligible = indices
            for index, block in enumerate(self.model.layers):
                block.self_attn.attn.layer_id = loop * len(self.model.layers) + index
                block.self_attn.attn.loop_index = loop
                if selection is None:
                    block.self_attn.attn.prefill_state = None
                    hidden = block(hidden, positions, forward_batch)
                else:
                    block.self_attn.attn.prefill_state = (
                        indices,
                        active_positions,
                        starts,
                        counts,
                        max_count,
                        (loop - 1) * len(self.model.layers) + index,
                    )
                    active_hidden = block(
                        active_hidden, active_positions, forward_batch
                    )
                    block.self_attn.attn.prefill_state = None
            if selection is None:
                hidden = self.model.norm(hidden)
            else:
                hidden = hidden.clone()
                hidden[indices] = self.model.norm(active_hidden)
            previous = before
        return self.logits_processor(input_ids, hidden, self.lm_head, forward_batch)

    def load_weights(self, weights):
        parameters = dict(self.named_parameters(remove_duplicate=False))
        loaded = set()
        rules = (
            (".q_proj.", ".qkv_proj.", "q"),
            (".k_proj.", ".qkv_proj.", "k"),
            (".v_proj.", ".qkv_proj.", "v"),
            (".gate_proj.", ".gate_up_proj.", 0),
            (".up_proj.", ".gate_up_proj.", 1),
        )
        for name, weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            target, shard = name, None
            for old, new, part in rules:
                if old in name:
                    target, shard = name.replace(old, new), part
                    break
            if target not in parameters:
                raise ValueError(f"Unexpected Ouro checkpoint tensor: {name}")
            parameter = parameters[target]
            loader = getattr(parameter, "weight_loader", default_weight_loader)
            if shard is None:
                loader(parameter, weight)
            else:
                loader(parameter, weight, shard)
            loaded.add((target, shard))
        missing = []
        for name in parameters:
            shards = (
                ("q", "k", "v")
                if ".qkv_proj." in name
                else ((0, 1) if ".gate_up_proj." in name else (None,))
            )
            for shard in shards:
                if (name, shard) not in loaded:
                    if config_tied_alias(self.config, name, loaded):
                        continue
                    missing.append((name, shard))
        if missing:
            raise ValueError(f"Missing checkpoint weights: {missing[:8]}")


def config_tied_alias(config, name, loaded):
    return (
        config.tie_word_embeddings
        and name in {"lm_head.weight", "model.embed_tokens.weight"}
        and (
            ("lm_head.weight", None) in loaded
            or ("model.embed_tokens.weight", None) in loaded
        )
    )


EntryClass = FlashLoopOuroForCausalLM
