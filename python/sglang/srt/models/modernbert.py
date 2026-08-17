# SPDX-License-Identifier: Apache-2.0
"""Inference-only ModernBERT model (encoder-only, embedding).

ModernBERT (https://arxiv.org/abs/2412.13663) is a modern encoder with rotary
position embeddings, alternating local (sliding-window) / global attention,
pre-norm blocks and a GeGLU MLP. This implementation targets the embedding use
case (e.g. ``ibm-granite/granite-embedding-english-r2``).

Attention is computed in-model with ``F.scaled_dot_product_attention`` rather
than through ``RadixAttention``: ModernBERT is bidirectional with a *symmetric*
sliding window and keeps no autoregressive KV cache, neither of which the paged
decoder attention backends express (the intel_xpu backend in particular forces
``causal`` and ignores ``AttentionType.ENCODER_ONLY``). Several other encoders in
SGLang (whisper, pixtral, ...) take the same in-model SDPA approach. RoPE reuses
the shared ``get_rope`` factory (two instances, one per attention theta).
"""

from typing import Iterable, Optional, Set, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.layers.pooler import EmbeddingPoolerOutput, Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader


def _rope_thetas(config: PretrainedConfig) -> Tuple[float, float]:
    """Return (global_theta, local_theta), robust to transformers versions.

    Newer transformers normalizes to ``rope_parameters`` keyed by
    ``full_attention`` / ``sliding_attention``; older configs expose flat
    ``global_rope_theta`` / ``local_rope_theta``.
    """
    rope_parameters = getattr(config, "rope_parameters", None)
    if rope_parameters is not None:
        return (
            float(rope_parameters["full_attention"]["rope_theta"]),
            float(rope_parameters["sliding_attention"]["rope_theta"]),
        )
    return float(config.global_rope_theta), float(config.local_rope_theta)


class ModernBertEmbeddings(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.tok_embeddings = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, enable_tp=False
        )
        self.norm = nn.LayerNorm(
            config.hidden_size, eps=config.norm_eps, bias=config.norm_bias
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.norm(self.tok_embeddings(input_ids))


class ModernBertMLP(nn.Module):
    """GeGLU MLP: Wi projects to 2*intermediate, split into (input, gate)."""

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.Wi = nn.Linear(
            config.hidden_size, int(config.intermediate_size) * 2, bias=config.mlp_bias
        )
        self.act = nn.GELU()
        self.Wo = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=config.mlp_bias
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        inp, gate = self.Wi(hidden_states).chunk(2, dim=-1)
        return self.Wo(self.act(inp) * gate)


class ModernBertAttention(nn.Module):
    def __init__(self, config: PretrainedConfig, layer_idx: int):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.Wqkv = nn.Linear(
            config.hidden_size,
            3 * self.head_dim * self.num_heads,
            bias=config.attention_bias,
        )
        self.Wo = nn.Linear(
            config.hidden_size, config.hidden_size, bias=config.attention_bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
        attn_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # hidden_states: (seq, hidden) for a single sequence; rotary_emb is the
        # shared global/local get_rope instance.
        seq_len = hidden_states.shape[0]
        qkv = self.Wqkv(hidden_states)
        qkv = qkv.view(seq_len, 3, self.num_heads * self.head_dim)
        q, k, v = qkv.unbind(dim=1)  # each (seq, heads * head_dim)

        # get_rope applies RoPE in-place on the (seq, heads*head_dim) layout.
        q, k = rotary_emb(positions, q, k)

        # SDPA expects (batch, heads, seq, head_dim); reshape from (seq, heads*hd).
        q = q.view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)

        # mask is a bool keep-mask (1,1,seq,seq) or None.
        out = F.scaled_dot_product_attention(
            q.unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            attn_mask=attn_mask,
            scale=self.head_dim**-0.5,
        )
        out = out.squeeze(0).transpose(0, 1).reshape(seq_len, -1)
        return self.Wo(out)


class ModernBertLayer(nn.Module):
    def __init__(self, config: PretrainedConfig, layer_idx: int):
        super().__init__()
        # Layer 0 has no attn_norm (embeddings.norm already pre-normalizes).
        if layer_idx == 0:
            self.attn_norm = nn.Identity()
        else:
            self.attn_norm = nn.LayerNorm(
                config.hidden_size, eps=config.norm_eps, bias=config.norm_bias
            )
        self.attn = ModernBertAttention(config, layer_idx)
        self.mlp_norm = nn.LayerNorm(
            config.hidden_size, eps=config.norm_eps, bias=config.norm_bias
        )
        self.mlp = ModernBertMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
        attn_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.attn_norm(hidden_states), positions, rotary_emb, attn_mask
        )
        hidden_states = hidden_states + self.mlp(self.mlp_norm(hidden_states))
        return hidden_states


class ModernBertModel(nn.Module):
    def __init__(
        self,
        *,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        # Alternating attention: every ``global_attn_every_n_layers``-th layer is
        # global (full bidirectional), the rest are local (symmetric window).
        # ``layer_types`` (newer transformers) is authoritative when present.
        layer_types = getattr(config, "layer_types", None)
        if layer_types is not None:
            self.is_global = [t == "full_attention" for t in layer_types]
        else:
            n = config.global_attn_every_n_layers
            self.is_global = [i % n == 0 for i in range(config.num_hidden_layers)]

        # Half-window each side. Newer configs expose ``sliding_window`` (already
        # halved); older ones expose ``local_attention`` (the full window).
        sliding_window = getattr(config, "sliding_window", None)
        if sliding_window is not None:
            self.local_window = sliding_window
        else:
            self.local_window = config.local_attention // 2

        self.embeddings = ModernBertEmbeddings(config)
        self.layers = nn.ModuleList(
            [ModernBertLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.final_norm = nn.LayerNorm(
            config.hidden_size, eps=config.norm_eps, bias=config.norm_bias
        )

        # Two shared rotary embeddings (NeoX-style), one per attention theta.
        global_theta, local_theta = _rope_thetas(config)
        self.global_rotary = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=global_theta,
            is_neox_style=True,
        )
        self.local_rotary = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=local_theta,
            is_neox_style=True,
        )

        self.pooler = Pooler(pooling_type=PoolingType.MEAN, normalize=True)

    def _local_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        # Symmetric sliding window: |i - j| <= local_window is allowed.
        # Use a BOOLEAN mask (True = keep). The Intel XPU SDPA kernel silently
        # ignores an additive float(-inf) attn_mask, but honors a bool mask.
        idx = torch.arange(seq_len, device=device)
        dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
        allowed = dist <= self.local_window
        return allowed.view(1, 1, seq_len, seq_len)

    def _encode_one(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        seq_len = hidden_states.shape[0]
        device = hidden_states.device

        local_mask = self._local_mask(seq_len, device)

        for i, layer in enumerate(self.layers):
            if self.is_global[i]:
                # Global layers are full bidirectional attention (no mask).
                hidden_states = layer(
                    hidden_states, positions, self.global_rotary, None
                )
            else:
                hidden_states = layer(
                    hidden_states, positions, self.local_rotary, local_mask
                )
        return hidden_states

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
    ) -> EmbeddingPoolerOutput:
        assert get_embedding, "ModernBertModel only supports embedding"

        hidden_states = self.embeddings(input_ids)

        # Tokens for all requests are flattened into one stream; slice per
        # request (each is an independent encoder pass with its own mask).
        seq_lens = forward_batch.extend_seq_lens_cpu
        outputs = []
        start = 0
        for seq_len in seq_lens:
            end = start + seq_len
            outputs.append(
                self._encode_one(hidden_states[start:end], positions[start:end])
            )
            start = end
        hidden_states = torch.cat(outputs, dim=0)

        hidden_states = self.final_norm(hidden_states)
        return self.pooler(hidden_states, forward_batch)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        params_dict = dict(self.named_parameters())
        loaded: Set[str] = set()
        for name, loaded_weight in weights:
            if name not in params_dict:
                # Encoder-only embedding model: skip any MLM head / unused tensors.
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded.add(name)
        return loaded


EntryClass = [ModernBertModel]
