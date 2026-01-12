# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Qwen3 DFlash draft model with non-causal attention for parallel block drafting."""

from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn

from sglang.srt.layers.dp_attention import get_attention_tp_rank, get_attention_tp_size
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import QKVParallelLinear, RowParallelLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.dflash import RMSNorm3D, build_target_layer_ids
from sglang.srt.models.utils import apply_qk_norm
from sglang.srt.utils import add_prefix


class Qwen3DFlashMLP(nn.Module):
    """Standard MLP for Qwen3 DFlash decoder layer."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.gate_up_proj = torch.nn.Linear(
            hidden_size, intermediate_size * 2, bias=False
        )
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down_proj(self.act_fn(gate) * up)


class Qwen3DFlashAttention(nn.Module):
    """DFlash attention: Q from noise only, K/V from full sequence, non-causal."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        layer_id: int = 0,
        rope_theta: float = 1000000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 32768,
        rms_norm_eps: float = 1e-6,
        attention_bias: bool = True,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_id = layer_id

        # Handle tensor parallelism
        attn_tp_size = get_attention_tp_size()
        attn_tp_rank = get_attention_tp_rank()

        self.total_num_heads = num_heads
        self.total_num_kv_heads = num_kv_heads
        self.num_heads = num_heads // attn_tp_size
        self.num_kv_heads = max(1, num_kv_heads // attn_tp_size)
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        # Q/K normalization (Qwen3 style)
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

        # Fused QKV projection for tensor parallelism
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("qkv_proj", prefix),
        )

        # Output projection
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            reduce_results=False,
            prefix=add_prefix("o_proj", prefix),
        )

        # Rotary embeddings
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: Optional[ForwardBatch],
        ctx_len: int,
    ) -> torch.Tensor:
        """Forward: returns attention output for noise positions only."""
        total_tokens = hidden_states.shape[0]
        noise_len = total_tokens - ctx_len

        # QKV projection on full input
        qkv, _ = self.qkv_proj(hidden_states)
        q_full, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Apply Q/K normalization
        q_full, k = apply_qk_norm(
            q=q_full,
            k=k,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
            head_dim=self.head_dim,
        )

        # Apply rotary embeddings to full Q and K
        q_full, k = self.rotary_emb(positions, q_full, k)

        # DFlash attention: Q from noise only, K/V from full sequence
        # Extract Q for noise positions only (original DFlash pattern)
        q = q_full[ctx_len:]  # [noise_len, q_size]

        # Reshape to [1, num_heads, seq_len, head_dim]
        q = (
            q.view(noise_len, self.num_heads, self.head_dim)
            .transpose(0, 1)
            .unsqueeze(0)
        )
        k = (
            k.view(total_tokens, self.num_kv_heads, self.head_dim)
            .transpose(0, 1)
            .unsqueeze(0)
        )
        v = (
            v.view(total_tokens, self.num_kv_heads, self.head_dim)
            .transpose(0, 1)
            .unsqueeze(0)
        )

        # Expand KV for GQA
        num_kv_groups = self.num_heads // self.num_kv_heads
        if num_kv_groups > 1:
            k = k.repeat_interleave(num_kv_groups, dim=1)
            v = v.repeat_interleave(num_kv_groups, dim=1)

        # Compute attention: Q [1, heads, noise_len, head_dim] @ K^T [1, heads, head_dim, total_len]
        # Result: [1, heads, noise_len, total_len]
        # Non-causal (bidirectional) attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            v.dtype
        )

        # Apply attention: [1, heads, noise_len, total_len] @ V [1, heads, total_len, head_dim]
        # Result: [1, heads, noise_len, head_dim]
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back to [noise_len, num_heads * head_dim]
        attn_output = (
            attn_output.squeeze(0).transpose(0, 1).contiguous().view(noise_len, -1)
        )

        # Output projection
        output, _ = self.o_proj(attn_output)
        return output


class Qwen3DFlashDecoderLayer(nn.Module):
    """DFlash decoder layer: input_layernorm on noise only, residual for noise only."""

    def __init__(
        self,
        config,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_id = layer_id

        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )

        self.self_attn = Qwen3DFlashAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=head_dim,
            layer_id=layer_id,
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
            max_position_embeddings=getattr(config, "max_position_embeddings", 32768),
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=getattr(config, "attention_bias", True),
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )

        self.mlp = Qwen3DFlashMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=getattr(config, "hidden_act", "silu"),
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: Optional[ForwardBatch],
        ctx_len: Optional[int] = None,
    ) -> torch.Tensor:
        """Forward: layernorm on noise only, return [target_hidden, updated_noise]."""
        if ctx_len is None or ctx_len == 0:
            # Fallback to standard pre-norm if ctx_len not provided
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            attn_output = self.self_attn(
                positions, hidden_states, forward_batch, ctx_len=0
            )
            hidden_states = residual + attn_output

            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states
            return hidden_states

        # Split into target_hidden and noise
        target_hidden = hidden_states[:ctx_len]  # Already normalized by hidden_norm
        noise = hidden_states[ctx_len:]  # Needs input_layernorm

        # Residual is for noise only
        noise_residual = noise

        # Apply input_layernorm to NOISE ONLY (matching original DFlash)
        noise_normed = self.input_layernorm(noise)

        # Concatenate: target_hidden (already normed) + noise (just normed)
        combined = torch.cat([target_hidden, noise_normed], dim=0)

        # Attention - output is for NOISE positions only
        attn_output = self.self_attn(
            positions, combined, forward_batch, ctx_len=ctx_len
        )

        # Residual connection for noise only
        noise = noise_residual + attn_output

        # MLP on noise only
        noise_residual = noise
        noise = self.post_attention_layernorm(noise)
        noise = self.mlp(noise)
        noise = noise_residual + noise

        # Return concatenated (target_hidden unchanged, noise updated)
        return torch.cat([target_hidden, noise], dim=0)


class Qwen3DFlashModel(nn.Module):
    """DFlash draft model body: [target_hidden, noise] -> noise output."""

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size

        # Embedding layer (for noise tokens)
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
        )

        # Target layer IDs for multi-layer feature extraction
        self.target_layer_ids = build_target_layer_ids(
            getattr(config, "num_target_layers", 28),
            config.num_hidden_layers,
        )

        # Feature compression: multi-layer target features → hidden_size
        num_selected_layers = len(self.target_layer_ids)
        self.fc = nn.Linear(
            num_selected_layers * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.hidden_norm = RMSNorm3D(config.hidden_size, eps=config.rms_norm_eps)

        # Decoder layers
        self.layers = nn.ModuleList(
            [
                Qwen3DFlashDecoderLayer(
                    config,
                    layer_id=layer_idx,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{layer_idx}", prefix),
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        # Final norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Block size for speculation
        self.block_size = getattr(config, "block_size", 16)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward: returns hidden states for noise positions only."""
        ctx_len = 0

        if input_embeds is not None:
            # Input already prepared by worker (concatenated target_hidden + noise_embeds)
            hidden_states = input_embeds
            # Get ctx_len from spec_info
            if (
                hasattr(forward_batch, "spec_info")
                and forward_batch.spec_info is not None
            ):
                if (
                    hasattr(forward_batch.spec_info, "ctx_lens")
                    and forward_batch.spec_info.ctx_lens is not None
                ):
                    # For batched case, use the first (or max) ctx_len
                    ctx_lens = forward_batch.spec_info.ctx_lens
                    ctx_len = (
                        ctx_lens[0].item() if ctx_lens.dim() > 0 else ctx_lens.item()
                    )
        else:
            # Get target hidden from spec_info
            target_hidden = forward_batch.spec_info.hidden_states

            # Handle batched target_hidden [bs, seq_len, hidden] -> [seq_len, hidden]
            if target_hidden.dim() == 3:
                target_hidden = target_hidden.squeeze(0)

            ctx_len = target_hidden.shape[0]

            # Project and normalize target hidden (THIS is the hidden_norm application)
            if target_hidden.shape[-1] != self.hidden_size:
                target_hidden = self.fc(target_hidden)
            target_hidden = self.hidden_norm(target_hidden)

            # Get noise embeddings
            noise_embeds = self.embed_tokens(input_ids)
            if noise_embeds.dim() == 3:
                noise_embeds = noise_embeds.squeeze(0)

            # Concatenate: [target_hidden, noise]
            hidden_states = torch.cat([target_hidden, noise_embeds], dim=0)

        # Process through layers, passing ctx_len for correct normalization
        for layer in self.layers:
            hidden_states = layer(
                positions, hidden_states, forward_batch, ctx_len=ctx_len
            )

        # Extract noise portion and apply final norm
        if ctx_len > 0:
            noise_hidden = hidden_states[ctx_len:]
        else:
            noise_hidden = hidden_states

        noise_hidden = self.norm(noise_hidden)

        return noise_hidden

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value


class Qwen3ForCausalLMDFlash(nn.Module):
    """Qwen3 DFlash draft model wrapper with weight loading."""

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.quant_config = quant_config

        self.model = Qwen3DFlashModel(
            config, quant_config, prefix=add_prefix("model", prefix)
        )

        # LM head for token prediction
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

        # Track if embeddings are shared from target model
        self._embed_tokens_from_target = None
        self._lm_head_from_target = None

    def get_embed_and_head(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get embedding and lm_head for sharing with target model.

        Returns:
            Tuple of (embed_weight, lm_head_weight)
        """
        embed = self.model.embed_tokens.weight
        head = self.lm_head.weight
        return embed, head

    def set_embed_and_head(self, embed: torch.Tensor, head: torch.Tensor):
        """Set embedding and lm_head from target model.

        Args:
            embed: Embedding weight from target model
            head: LM head weight from target model
        """
        self._embed_tokens_from_target = embed
        self._lm_head_from_target = head
        # Replace the embedding weight
        self.model.embed_tokens.weight = nn.Parameter(embed, requires_grad=False)
        # Replace lm_head weight
        self.lm_head.weight = nn.Parameter(head, requires_grad=False)

    def set_embed(self, embed: torch.Tensor):
        """Set only embedding from target model."""
        self._embed_tokens_from_target = embed
        self.model.embed_tokens.weight = nn.Parameter(embed, requires_grad=False)

    @property
    def target_layer_ids(self) -> List[int]:
        """Get target layer IDs from the model body."""
        return self.model.target_layer_ids

    @property
    def block_size(self) -> int:
        """Get block size from the model body."""
        return self.model.block_size

    def get_input_embeddings(self):
        """HuggingFace-compatible interface for getting input embeddings."""
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        """HuggingFace-compatible interface for setting input embeddings."""
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward: returns logits for noise tokens."""
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)

        # Simple linear projection to logits
        return torch.matmul(hidden_states, self.lm_head.weight.t())

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights with q/k/v -> qkv_proj and gate/up -> gate_up_proj mapping."""
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        stacked_weight_buffer: Dict[str, Dict] = {}

        for name, loaded_weight in weights:
            if name.startswith("model.model."):
                name = name[6:]
            elif not name.startswith("model."):
                name = "model." + name

            matched = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name in name:
                    mapped_name = name.replace(weight_name, param_name)
                    if mapped_name not in params_dict:
                        matched = True
                        break
                    param = params_dict[mapped_name]
                    weight_loader = getattr(param, "weight_loader", None)
                    if weight_loader is not None:
                        weight_loader(param, loaded_weight, shard_id)
                    else:
                        if mapped_name not in stacked_weight_buffer:
                            stacked_weight_buffer[mapped_name] = {}
                        stacked_weight_buffer[mapped_name][shard_id] = loaded_weight
                    matched = True
                    break

            if not matched and name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

        for param_name, shard_dict in stacked_weight_buffer.items():
            if param_name in params_dict and 0 in shard_dict and 1 in shard_dict:
                params_dict[param_name].data.copy_(
                    torch.cat([shard_dict[0], shard_dict[1]], dim=0)
                )


class DFlashDraftModel(Qwen3ForCausalLMDFlash):
    """Alias for HuggingFace config compatibility."""
    pass


EntryClass = [Qwen3ForCausalLMDFlash, DFlashDraftModel]
