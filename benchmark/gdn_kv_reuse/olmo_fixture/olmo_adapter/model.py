"""Experimental Olmo Hybrid implementation using native SGLang cache backends.

Projection weights and normalization/residual ordering match Transformers Olmo
Hybrid. NoPE only, TP=1; intentionally not an upstream model-support patch.
"""

import torch
from torch import nn
from transformers.models.olmo_hybrid import modeling_olmo_hybrid as hf

from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.runtime_context import get_parallel


class Attention(hf.OlmoHybridAttention):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.attn = RadixAttention(
            num_heads=config.num_attention_heads,
            head_dim=self.head_dim,
            scaling=self.scaling,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_idx,
        )

    def forward(self, x, forward_batch):
        q = self.q_norm(self.q_proj(x))
        if forward_batch.full_kv_replay:
            y = self.attn(q, None, None, forward_batch, save_kv_cache=False)
        else:
            k, v = self.k_norm(self.k_proj(x)), self.v_proj(x)
            y = self.attn(q, k, v, forward_batch)
        return self.o_proj(y)


class GatedDeltaNet(hf.OlmoHybridGatedDeltaNet):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.attn = RadixLinearAttention(
            layer_id=layer_idx,
            num_q_heads=self.num_k_heads,
            num_k_heads=self.num_k_heads,
            num_v_heads=self.num_v_heads,
            head_q_dim=self.head_k_dim,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            conv_weights=self.conv1d.weight.squeeze(1),
            activation=self.activation,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
        )

    def forward(self, x, forward_batch):
        mixed_qkv = torch.cat((self.q_proj(x), self.k_proj(x), self.v_proj(x)), dim=-1)
        y = self.attn(forward_batch, mixed_qkv, self.a_proj(x), self.b_proj(x))
        y = self.o_norm(
            y.reshape(-1, self.head_v_dim), self.g_proj(x).reshape(-1, self.head_v_dim)
        )
        return self.o_proj(y.reshape(x.shape[0], self.value_dim))


class DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.is_linear = config.layer_types[layer_idx] == "linear_attention"
        self.mlp = hf.OlmoHybridMLP(config)
        self.post_attention_layernorm = hf.OlmoHybridRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        if self.is_linear:
            self.input_layernorm = hf.OlmoHybridRMSNorm(
                config.hidden_size, config.rms_norm_eps
            )
            self.linear_attn = GatedDeltaNet(config, layer_idx)
        else:
            self.self_attn = Attention(config, layer_idx)
            self.post_feedforward_layernorm = hf.OlmoHybridRMSNorm(
                config.hidden_size, config.rms_norm_eps
            )

    def forward(self, x, forward_batch):
        if self.is_linear:
            x = x + self.linear_attn(self.input_layernorm(x), forward_batch)
            return x + self.mlp(self.post_attention_layernorm(x))
        x = x + self.post_attention_layernorm(self.self_attn(x, forward_batch))
        return x + self.post_feedforward_layernorm(self.mlp(x))


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.layers = nn.ModuleList(
            DecoderLayer(config, i) for i in range(config.num_hidden_layers)
        )
        self.norm = hf.OlmoHybridRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, ids, forward_batch, input_embeds=None):
        x = self.embed_tokens(ids) if input_embeds is None else input_embeds
        for layer in self.layers:
            x = layer(x, forward_batch)
        return self.norm(x)


class OlmoHybridForCausalLM(nn.Module):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        assert get_parallel().tp_size == get_parallel().pp_size == 1
        assert quant_config is None and config.rope_parameters["rope_theta"] is None
        assert config.linear_allow_neg_eigval and not config.attention_bias
        self.config = config
        self.model = Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self.logits_processor = LogitsProcessor(config)

    def get_num_kv_cache_layers(self):
        return len(self.config.full_attention_layer_ids)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    @torch.no_grad()
    def forward(self, input_ids, positions, forward_batch, input_embeds=None, **kwargs):
        x = self.model(input_ids, forward_batch, input_embeds)
        return self.logits_processor(input_ids, x, self.lm_head, forward_batch)

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        loaded = set()
        conv_parts = set()
        key_dim = self.config.linear_num_key_heads * self.config.linear_key_head_dim
        value_dim = (
            self.config.linear_num_value_heads * self.config.linear_value_head_dim
        )
        for name, weight in weights:
            part = next((p for p in ("q", "k", "v") if f".{p}_conv1d." in name), None)
            if part is not None:
                conv_parts.add(name)
                name = name.replace(f".{part}_conv1d.", ".conv1d.")
                start, width = {
                    "q": (0, key_dim),
                    "k": (key_dim, key_dim),
                    "v": (2 * key_dim, value_dim),
                }[part]
                default_weight_loader(params[name][start : start + width], weight)
                loaded.add(name)
                continue
            param = params[name]
            getattr(param, "weight_loader", default_weight_loader)(param, weight)
            loaded.add(name)
        assert len(conv_parts) == 3 * len(self.config.linear_layer_ids)
        assert loaded == set(params), set(params) - loaded
        return loaded


EntryClass = OlmoHybridForCausalLM
