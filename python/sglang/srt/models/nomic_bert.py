# SPDX-License-Identifier: Apache-2.0
"""Inference-only NomicBERT model (encoder-only, embedding).

`nomic-ai/nomic-embed-text-v1.5` (arch ``NomicBertModel``, ``model_type:
nomic_bert``) is a BERT-style bidirectional encoder with rotary position
embeddings, a SwiGLU gated MLP and post-norm blocks, mean-pooled for embeddings.

Attention is computed in-model with ``F.scaled_dot_product_attention`` (no KV
cache, bidirectional), matching the approach used by ``modernbert.py`` and other
SGLang encoders; RoPE reuses the shared ``get_rope`` factory. The stock server
otherwise falls back to the generic Transformers wrapper, which raises
``NotImplementedError: get_input_embeddings not auto-handled for NomicBertModel``.
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


class NomicBertEmbeddings(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.word_embeddings = VocabParallelEmbedding(
            config.vocab_size, config.n_embd, enable_tp=False
        )
        self.token_type_embeddings = VocabParallelEmbedding(
            config.type_vocab_size, config.n_embd, enable_tp=False
        )
        self.emb_ln = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # token_type_ids are all 0 for single-sequence embedding inputs.
        embeds = self.word_embeddings(input_ids)
        token_type = self.token_type_embeddings(
            torch.zeros_like(input_ids, dtype=torch.long)
        )
        return self.emb_ln(embeds + token_type)


class NomicBertGatedMLP(nn.Module):
    """SwiGLU: fc2(fc11(x) * silu(fc12(x)))."""

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.fc11 = nn.Linear(config.n_embd, config.n_inner, bias=config.mlp_fc1_bias)
        self.fc12 = nn.Linear(config.n_embd, config.n_inner, bias=config.mlp_fc1_bias)
        self.fc2 = nn.Linear(config.n_inner, config.n_embd, bias=config.mlp_fc2_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.fc11(x) * F.silu(self.fc12(x)))


class NomicBertAttention(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.num_heads = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.Wqkv = nn.Linear(
            config.n_embd, 3 * config.n_embd, bias=config.qkv_proj_bias
        )
        self.out_proj = nn.Linear(
            config.n_embd, config.n_embd, bias=config.qkv_proj_bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
    ) -> torch.Tensor:
        # hidden_states: (seq, hidden) for a single sequence (bidirectional, no mask).
        seq_len = hidden_states.shape[0]
        qkv = self.Wqkv(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)  # each (seq, heads * head_dim)

        # get_rope applies RoPE in-place on the (seq, heads*head_dim) layout.
        q, k = rotary_emb(positions, q, k)

        q = q.view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)

        out = F.scaled_dot_product_attention(
            q.unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            scale=self.head_dim**-0.5,
        )
        out = out.squeeze(0).transpose(0, 1).reshape(seq_len, -1)
        return self.out_proj(out)


class NomicBertLayer(nn.Module):
    """Post-norm block: h = norm1(attn(h) + h); h = norm2(mlp(h) + h)."""

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.attn = NomicBertAttention(config)
        self.mlp = NomicBertGatedMLP(config)
        self.norm1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.norm2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
    ) -> torch.Tensor:
        hidden_states = self.norm1(
            self.attn(hidden_states, positions, rotary_emb) + hidden_states
        )
        hidden_states = self.norm2(self.mlp(hidden_states) + hidden_states)
        return hidden_states


class NomicBertModel(nn.Module):
    def __init__(
        self,
        *,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.head_dim = config.n_embd // config.n_head

        self.embeddings = NomicBertEmbeddings(config)
        self.layers = nn.ModuleList(
            [NomicBertLayer(config) for _ in range(config.n_layer)]
        )
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=int(config.rotary_emb_base),
            is_neox_style=True,
        )
        self.pooler = Pooler(pooling_type=PoolingType.MEAN, normalize=True)

    def _encode_one(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states, positions, self.rotary_emb)
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
        assert get_embedding, "NomicBertModel only supports embedding"

        hidden_states = self.embeddings(input_ids)

        # Tokens for all requests are flattened into one stream; encode each
        # request independently (bidirectional, per-request positions).
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

        return self.pooler(hidden_states, forward_batch)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        # Checkpoint names -> module tree:
        #   embeddings.word_embeddings / embeddings.token_type_embeddings -> as-is
        #   emb_ln.* -> embeddings.emb_ln.*
        #   encoder.layers.N.{attn.Wqkv, attn.out_proj, mlp.fc11/fc12/fc2,
        #                     norm1, norm2} -> layers.N.*
        params_dict = dict(self.named_parameters())
        loaded: Set[str] = set()
        for name, loaded_weight in weights:
            if name.startswith("emb_ln."):
                name = "embeddings." + name
            elif name.startswith("encoder.layers."):
                name = name[len("encoder.") :]
            if name not in params_dict:
                # Skip unused tensors (e.g. any MLM head).
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded.add(name)
        return loaded


EntryClass = [NomicBertModel]
