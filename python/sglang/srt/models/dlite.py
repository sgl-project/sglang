"""Inference-only DLite draft model for Qwen3 targets."""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import QKVParallelLinear, RowParallelLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.rotary_embedding.utils import apply_rotary_emb
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen2 import Qwen2MLP
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.dlite_utils import parse_dlite_draft_config
from sglang.srt.utils import add_prefix
from sglang.srt.utils.hf_transformers_utils import get_rope_config
from torch import nn


class DLiteAttention(nn.Module):
    """Non-causal attention over current target features and one draft block."""

    def __init__(
        self,
        config,
        *,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        head_dim = int(
            getattr(
                config,
                "head_dim",
                hidden_size // int(config.num_attention_heads),
            )
        )
        total_num_heads = int(config.num_attention_heads)
        total_num_kv_heads = int(config.num_key_value_heads)

        self.head_dim = head_dim
        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=head_dim,
            total_num_heads=total_num_heads,
            total_num_kv_heads=total_num_kv_heads,
            bias=bool(getattr(config, "attention_bias", False)),
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.num_heads = int(self.qkv_proj.num_heads)
        self.num_kv_heads = int(self.qkv_proj.num_kv_heads)
        self.q_size = int(self.qkv_proj.q_proj_shard_size)
        self.kv_size = int(self.qkv_proj.kv_proj_shard_size)
        self.o_proj = RowParallelLinear(
            total_num_heads * head_dim,
            hidden_size,
            bias=bool(getattr(config, "attention_bias", False)),
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )
        eps = float(getattr(config, "rms_norm_eps", 1e-6))
        self.q_norm = RMSNorm(head_dim, eps=eps)
        self.k_norm = RMSNorm(head_dim, eps=eps)

        rope_theta, rope_scaling = get_rope_config(config)
        self.rotary_emb = get_rope(
            head_dim,
            rotary_dim=head_dim,
            max_position=int(getattr(config, "max_position_embeddings", 32768)),
            base=rope_theta,
            rope_scaling=rope_scaling,
        )

    def _apply_rotary(
        self, states: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        batch_size, sequence_length, num_heads, head_dim = states.shape
        flat_positions = positions.reshape(-1)
        cache = self.rotary_emb.cos_sin_cache.index_select(0, flat_positions)
        cache = cache.to(device=states.device, dtype=states.dtype)
        cos, sin = cache.chunk(2, dim=-1)
        flat_states = states.reshape(-1, num_heads, head_dim)
        rotated = apply_rotary_emb(
            flat_states,
            cos,
            sin,
            self.rotary_emb.is_neox_style,
        )
        return rotated.view(batch_size, sequence_length, num_heads, head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        query_positions: torch.Tensor,
        context_positions: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, query_length = hidden_states.shape[:2]
        context_and_query = torch.cat([target_hidden, hidden_states], dim=1)

        context_qkv, _ = self.qkv_proj(context_and_query)
        all_query, key, value = context_qkv.split(
            [self.q_size, self.kv_size, self.kv_size], dim=-1
        )
        query = all_query[:, -query_length:, :]

        query = query.view(batch_size, query_length, self.num_heads, self.head_dim)
        key = key.view(
            batch_size, context_and_query.shape[1], self.num_kv_heads, self.head_dim
        )
        value = value.view_as(key)
        query = self.q_norm(query.reshape(-1, self.head_dim)).view_as(query)
        key = self.k_norm(key.reshape(-1, self.head_dim)).view_as(key)

        key_positions = torch.cat([context_positions, query_positions], dim=1)
        query = self._apply_rotary(query, query_positions)
        key = self._apply_rotary(key, key_positions)

        kv_groups = self.num_heads // self.num_kv_heads
        if kv_groups > 1:
            key = key.repeat_interleave(kv_groups, dim=2)
            value = value.repeat_interleave(kv_groups, dim=2)

        output = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            dropout_p=0.0,
            is_causal=False,
        )
        output = output.transpose(1, 2).reshape(batch_size, query_length, -1)
        output, _ = self.o_proj(output)
        return output


class DLiteDecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        *,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        eps = float(getattr(config, "rms_norm_eps", 1e-6))
        self.self_attn = DLiteAttention(
            config,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.mlp = Qwen2MLP(
            hidden_size=hidden_size,
            intermediate_size=int(config.intermediate_size),
            hidden_act=str(config.hidden_act),
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        self.input_layernorm = RMSNorm(hidden_size, eps=eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        query_positions: torch.Tensor,
        context_positions: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            target_hidden,
            query_positions,
            context_positions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        return residual + self.mlp(hidden_states)


class DLiteSequentialHead(nn.Module):
    """Low-rank recurrent head used by released DLite checkpoints."""

    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int,
        sequential_rank: int,
        max_prediction_length: int,
        prefix: str,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.sequential_rank = int(sequential_rank)
        self.max_prediction_length = int(max_prediction_length)
        self.prev_token_embedding = VocabParallelEmbedding(
            self.vocab_size,
            self.sequential_rank,
            prefix=add_prefix("prev_token_embedding", prefix),
        )
        self.output_proj = ParallelLMHead(
            self.vocab_size,
            self.sequential_rank,
            prefix=add_prefix("output_proj", prefix),
        )
        self.state_proj = nn.Linear(
            2 * self.sequential_rank,
            2 * self.sequential_rank,
        )
        self.state_hidden_mlp = nn.Linear(
            self.sequential_rank + self.hidden_size,
            self.sequential_rank,
        )

    def _step(
        self,
        *,
        previous_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        previous = self.prev_token_embedding(previous_ids.long())
        gate_raw, candidate_raw = self.state_proj(
            torch.cat([state, previous], dim=-1)
        ).chunk(2, dim=-1)
        gate = torch.sigmoid(gate_raw)
        state = gate * state + (1.0 - gate) * torch.tanh(candidate_raw)
        latent = self.state_hidden_mlp(torch.cat([state, hidden_states], dim=-1))
        return latent, state

    def _greedy_token(self, latent: torch.Tensor) -> torch.Tensor:
        head = self.output_proj
        shard = head.shard_indices
        num_org = int(shard.num_org_elements)
        local_logits = F.linear(latent, head.weight[:num_org])
        local_max, local_arg = torch.max(local_logits, dim=-1)
        local_ids = local_arg.to(torch.int64) + int(shard.org_vocab_start_index)

        tp_group = get_parallel().tp_group
        tp_size = int(tp_group.world_size)
        if tp_size == 1:
            return local_ids

        batch_size = int(latent.shape[0])
        gathered_max = torch.empty(
            (tp_size * batch_size,),
            dtype=local_max.dtype,
            device=local_max.device,
        )
        gathered_ids = torch.empty(
            (tp_size * batch_size,),
            dtype=torch.int64,
            device=latent.device,
        )
        tp_group.all_gather_into_tensor(gathered_max, local_max.contiguous())
        tp_group.all_gather_into_tensor(gathered_ids, local_ids.contiguous())
        gathered_max = gathered_max.view(tp_size, batch_size)
        gathered_ids = gathered_ids.view(tp_size, batch_size)
        best_rank = torch.argmax(gathered_max, dim=0, keepdim=True)
        return torch.gather(gathered_ids, 0, best_rank).squeeze(0)

    def sample_greedy(
        self,
        *,
        hidden_states: torch.Tensor,
        first_previous_ids: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(
                "DLite sequential hidden_states must have shape [batch, length, "
                f"hidden], got {tuple(hidden_states.shape)}."
            )
        prediction_length = int(hidden_states.shape[1])
        if prediction_length > self.max_prediction_length:
            raise ValueError(
                f"DLite prediction length {prediction_length} exceeds checkpoint "
                f"maximum {self.max_prediction_length}."
            )

        state = hidden_states.new_zeros(hidden_states.shape[0], self.sequential_rank)
        previous_ids = first_previous_ids.long()
        outputs = []
        for position in range(prediction_length):
            latent, state = self._step(
                previous_ids=previous_ids,
                hidden_states=hidden_states[:, position, :],
                state=state,
            )
            previous_ids = self._greedy_token(latent)
            outputs.append(previous_ids.unsqueeze(1))
        return torch.cat(outputs, dim=1)


class DLiteDraftModel(nn.Module):
    """History-free DLite draft network loaded from released HF checkpoints."""

    supports_quantization = False

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.dlite_config = parse_dlite_draft_config(config)
        hidden_size = int(config.hidden_size)
        eps = float(getattr(config, "rms_norm_eps", 1e-6))

        self.layers = nn.ModuleList(
            [
                DLiteDecoderLayer(
                    config,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{layer_id}", prefix),
                )
                for layer_id in range(int(config.num_hidden_layers))
            ]
        )
        self.norm = RMSNorm(hidden_size, eps=eps)
        self.layer_depth_embedding = nn.Embedding(
            int(config.num_target_layers), hidden_size
        )
        self.context_norm = RMSNorm(hidden_size, eps=eps)
        self.register_buffer(
            "_target_layer_ids",
            torch.tensor(self.dlite_config.target_layer_ids, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_query_position_offsets",
            torch.arange(self.dlite_config.block_size, dtype=torch.long),
            persistent=False,
        )
        self.sequential_head = DLiteSequentialHead(
            vocab_size=int(config.vocab_size),
            hidden_size=hidden_size,
            sequential_rank=self.dlite_config.sequential_rank,
            max_prediction_length=self.dlite_config.proposal_length,
            prefix=add_prefix("sequential_head", prefix),
        )

    @property
    def block_size(self) -> int:
        return self.dlite_config.block_size

    @property
    def mask_token_id(self) -> int:
        return self.dlite_config.mask_token_id

    @property
    def target_layer_ids(self) -> Tuple[int, ...]:
        return self.dlite_config.target_layer_ids

    @torch.inference_mode()
    def forward_draft(
        self,
        *,
        query_embeddings: torch.Tensor,
        target_hidden: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = int(query_embeddings.shape[0])
        hidden_size = int(self.config.hidden_size)
        expected_target_width = self.dlite_config.chs_num_layers * hidden_size
        if target_hidden.ndim == 2:
            if int(target_hidden.shape[1]) != expected_target_width:
                raise ValueError(
                    "DLite target hidden width mismatch: expected "
                    f"{expected_target_width}, got {int(target_hidden.shape[1])}."
                )
            target_hidden = target_hidden.view(
                batch_size, self.dlite_config.chs_num_layers, hidden_size
            )
        if target_hidden.shape != (
            batch_size,
            self.dlite_config.chs_num_layers,
            hidden_size,
        ):
            raise ValueError(
                "DLite target_hidden must have shape "
                f"[{batch_size}, {self.dlite_config.chs_num_layers}, {hidden_size}], "
                f"got {tuple(target_hidden.shape)}."
            )
        if query_embeddings.shape != (
            batch_size,
            self.block_size,
            hidden_size,
        ):
            raise ValueError(
                "DLite query_embeddings must have shape "
                f"[{batch_size}, {self.block_size}, {hidden_size}], got "
                f"{tuple(query_embeddings.shape)}."
            )

        context = self.context_norm(
            target_hidden
            + self.layer_depth_embedding(self._target_layer_ids).unsqueeze(0)
        )
        query_positions = self._query_position_offsets.unsqueeze(0).expand(
            batch_size, -1
        )
        context_positions = torch.zeros(
            (batch_size, self.dlite_config.chs_num_layers),
            dtype=torch.long,
            device=query_embeddings.device,
        )

        hidden_states = query_embeddings
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                context,
                query_positions,
                context_positions,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states[:, -self.dlite_config.proposal_length :, :]

    @torch.inference_mode()
    def sample_greedy(
        self,
        *,
        draft_hidden: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.sequential_head.sample_greedy(
            hidden_states=draft_hidden,
            first_previous_ids=anchor_token_ids,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            name = name.removeprefix("model.")
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if f".{weight_name}." not in name:
                    continue
                mapped_name = name.replace(weight_name, param_name)
                if mapped_name not in params_dict:
                    break
                param = params_dict[mapped_name]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(mapped_name)
                break
            else:
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        return loaded_params


EntryClass = DLiteDraftModel

__all__ = [
    "DLiteAttention",
    "DLiteDecoderLayer",
    "DLiteDraftModel",
    "DLiteSequentialHead",
]
