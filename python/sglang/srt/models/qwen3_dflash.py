# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Qwen3-specific DFlash draft model implementation.

This module implements DFlash for Qwen3 target models with:
- Tensor parallelism support via QKVParallelLinear
- Non-causal attention for parallel block drafting
- Multi-layer feature extraction from target model

DFlash attention pattern:
- Q is projected from noise positions only
- K/V are projected from concatenated [target_hidden, noise_embedding]
- Non-causal (bidirectional) attention
"""

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
    """
    Qwen3-specific DFlash attention matching the original DFlash pattern.

    DFlash attention pattern:
    - Q is projected from noise_embedding only (after input_layernorm)
    - K/V are projected from concatenated [target_hidden, noise_embedding]
    - target_hidden is NOT passed through input_layernorm (already normalized by hidden_norm)
    - Non-causal (bidirectional) attention

    This implementation uses QKVParallelLinear for tensor parallelism.
    """

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
        forward_batch: ForwardBatch,
        ctx_len: int,
    ) -> torch.Tensor:
        """
        Forward pass matching original DFlash attention pattern.

        Args:
            positions: Position IDs for rotary embeddings [total_tokens]
            hidden_states: Concatenated [target_hidden, noise] [total_tokens, hidden]
            forward_batch: Unused, kept for interface compatibility
            ctx_len: Length of target_hidden portion

        Returns:
            Attention output for NOISE positions only [noise_tokens, hidden]
        """
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
    """Qwen3-specific DFlash decoder layer.

    Matches original DFlash pattern:
    - input_layernorm is applied ONLY to noise, NOT to target_hidden
    - target_hidden is already normalized by hidden_norm at model level
    - Residual connection is only for noise portion
    """

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
        forward_batch: ForwardBatch,
        ctx_len: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Forward pass matching original DFlash pattern.

        Args:
            positions: Position IDs [total_tokens]
            hidden_states: Concatenated [target_hidden, noise] [total_tokens, hidden]
            forward_batch: ForwardBatch with attention metadata
            ctx_len: Length of target_hidden portion. Required for correct normalization.

        Original DFlash pattern:
        - input_layernorm is applied ONLY to noise
        - target_hidden is NOT normalized here (already done by hidden_norm)
        """
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
    """
    Qwen3 DFlash draft model body.

    Takes concatenated [target_hidden, noise_embedding] as input,
    returns output for noise positions only.
    """

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
        """
        Forward pass matching original DFlash pattern.

        Args:
            input_ids: Token IDs for noise (used only if input_embeds is None)
            positions: Position IDs for full sequence [ctx_len + noise_len]
            forward_batch: ForwardBatch with spec_info containing target hidden states
            input_embeds: Pre-computed embeddings (concatenated target_hidden + noise)

        Returns:
            Hidden states for noise positions only [noise_tokens, hidden]
        """
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


class Qwen3ForCausalLMDFlash(nn.Module):
    """
    Qwen3 DFlash draft model wrapper.

    This is the main entry point class that handles:
    - Model initialization
    - Weight loading (separate q/k/v → fused qkv_proj)
    - Logits processing

    Compatible with TpModelWorker loading via SGLang model registry.
    """

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

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for DFlash draft model.

        Note: This method is not used by DFlashWorker, which directly accesses
        model components for better control over the attention pattern.
        Kept for interface compatibility and potential testing use.

        Args:
            input_ids: Token IDs for noise tokens
            positions: Position IDs for full sequence
            forward_batch: ForwardBatch with attention metadata
            input_embeds: Optional pre-computed embeddings

        Returns:
            Logits for next token prediction [seq_len, vocab_size]
        """
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)

        # Simple linear projection to logits
        return torch.matmul(hidden_states, self.lm_head.weight.t())

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """
        Load weights with mapping from separate q/k/v to fused qkv_proj.

        Pretrained DFlash uses separate q_proj, k_proj, v_proj.
        This maps them to fused QKVParallelLinear.
        """
        # Stacked params: (param_name, shard_name, shard_id)
        # For QKVParallelLinear: shard_id is "q", "k", "v"
        # For gate_up_proj (nn.Linear): shard_id is 0, 1 (index in concatenation)
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())

        loaded_count = 0
        skipped_weights = []

        # Collect weights for manual stacking (for nn.Linear params without weight_loader)
        stacked_weight_buffer: Dict[str, Dict] = {}

        for name, loaded_weight in weights:
            original_name = name
            # Handle prefix variations
            if name.startswith("model.model."):
                name = name[6:]  # Remove duplicate "model."
            elif not name.startswith("model."):
                # FIX: Add "model." prefix if missing (checkpoint doesn't have it)
                name = "model." + name

            # Check for stacked parameter mapping
            matched = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name in name:
                    # Map to stacked parameter
                    mapped_name = name.replace(weight_name, param_name)

                    if mapped_name not in params_dict:
                        skipped_weights.append(
                            {
                                "original": original_name,
                                "mapped": mapped_name,
                                "reason": "not in params_dict",
                            }
                        )
                        matched = True
                        break

                    param = params_dict[mapped_name]
                    weight_loader = getattr(param, "weight_loader", None)

                    if weight_loader is not None:
                        # Use proper weight_loader (QKVParallelLinear has this)
                        weight_loader(param, loaded_weight, shard_id)
                        loaded_count += 1
                    else:
                        # Manual stacking for nn.Linear (gate_up_proj)
                        if mapped_name not in stacked_weight_buffer:
                            stacked_weight_buffer[mapped_name] = {}
                        stacked_weight_buffer[mapped_name][shard_id] = loaded_weight
                    matched = True
                    break

            if not matched:
                # Direct loading for non-stacked parameters
                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_count += 1
                else:
                    skipped_weights.append(
                        {
                            "original": original_name,
                            "mapped": name,
                            "reason": "not in params_dict (direct)",
                        }
                    )

        # Now apply manually stacked weights
        for param_name, shard_dict in stacked_weight_buffer.items():
            if param_name in params_dict:
                param = params_dict[param_name]
                # For gate_up_proj: shard_id 0 = gate, 1 = up
                # Concatenate along dim 0 (output dimension)
                if 0 in shard_dict and 1 in shard_dict:
                    stacked = torch.cat([shard_dict[0], shard_dict[1]], dim=0)
                    param.data.copy_(stacked)
                    loaded_count += 1
                else:
                    skipped_weights.append(
                        {"param": param_name, "reason": "incomplete stacked weights"}
                    )


# Create a properly named class for HuggingFace architecture matching
# The HuggingFace config has "architectures": ["DFlashDraftModel"]
# This class MUST have __name__ == "DFlashDraftModel" for registry to find it
class DFlashDraftModel(Qwen3ForCausalLMDFlash):
    """DFlash draft model - same as Qwen3ForCausalLMDFlash but with matching name."""

    pass


# Register with SGLang's model registry
# Include both names so either architecture name works
EntryClass = [Qwen3ForCausalLMDFlash, DFlashDraftModel]
