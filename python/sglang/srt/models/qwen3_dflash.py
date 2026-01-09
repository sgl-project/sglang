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
Qwen3-specific DFlash draft model implementation using RadixAttention.

This module implements DFlash for Qwen3 target models using SGLang's native
RadixAttention with ENCODER_ONLY for non-causal attention. This enables:
- Native paged KV cache management
- CUDA graph support out of the box
- Tensor parallelism support
- Prefix caching integration

DFlash uses:
- Concatenated input attention (target_hidden + noise_embedding)
- Non-causal attention for parallel block drafting
- Multi-layer feature extraction from target model
"""

from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn

from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.layers.dp_attention import get_attention_tp_rank, get_attention_tp_size
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import QKVParallelLinear, RowParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import AttentionType, RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.dflash import RMSNorm3D, build_target_layer_ids
from sglang.srt.models.qwen2 import Qwen2MLP
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
    Qwen3-specific DFlash attention using RadixAttention with ENCODER_ONLY.
    
    Uses non-causal attention where input is concatenated [target_hidden, noise_embedding].
    Output is extracted for noise positions only.
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
        self.scaling = self.head_dim ** -0.5

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

        # RadixAttention with ENCODER_ONLY for non-causal attention
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            attn_type=AttentionType.ENCODER_ONLY,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """
        Forward pass with concatenated input.
        
        Args:
            positions: Position IDs for rotary embeddings [total_tokens]
            hidden_states: Concatenated [target_hidden, noise] [total_tokens, hidden]
            forward_batch: ForwardBatch with attention metadata
            
        Returns:
            Attention output [total_tokens, hidden]
        """
        # QKV projection
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        
        # Apply Q/K normalization
        q, k = apply_qk_norm(
            q=q,
            k=k,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
            head_dim=self.head_dim,
        )
        
        # Apply rotary embeddings
        q, k = self.rotary_emb(positions, q, k)
        
        # RadixAttention (ENCODER_ONLY = non-causal)
        attn_output = self.attn(q, k, v, forward_batch)
        
        # Output projection
        output, _ = self.o_proj(attn_output)
        return output


class Qwen3DFlashDecoderLayer(nn.Module):
    """Qwen3-specific DFlash decoder layer with RadixAttention."""

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
        
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        
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
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """
        Forward pass with pre-norm attention and MLP.
        
        Args:
            positions: Position IDs [total_tokens]
            hidden_states: Input [total_tokens, hidden]
            forward_batch: ForwardBatch with attention metadata
        """
        # Pre-norm + attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states, forward_batch)
        hidden_states = residual + hidden_states

        # Pre-norm + MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Qwen3DFlashModel(nn.Module):
    """
    Qwen3 DFlash draft model body using RadixAttention.
    
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
        self.layers = nn.ModuleList([
            Qwen3DFlashDecoderLayer(
                config,
                layer_id=layer_idx,
                quant_config=quant_config,
                prefix=add_prefix(f"layers.{layer_idx}", prefix),
            )
            for layer_idx in range(config.num_hidden_layers)
        ])

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
        Forward pass with concatenated input.
        
        The input is prepared by the worker:
        - hidden_states = cat([projected_target_hidden, noise_embedding], dim=0)
        - positions = [0, 1, ..., ctx_len + q_len - 1]
        
        Args:
            input_ids: Token IDs (used only if input_embeds is None)
            positions: Position IDs for full sequence
            forward_batch: ForwardBatch with spec_info containing target hidden states
            input_embeds: Pre-computed embeddings (concatenated target_hidden + noise)
            
        Returns:
            Hidden states for noise positions only [noise_tokens, hidden]
        """
        if input_embeds is not None:
            # Input already prepared by worker (concatenated target_hidden + noise_embeds)
            hidden_states = input_embeds
        else:
            # Get target hidden from spec_info
            target_hidden = forward_batch.spec_info.hidden_states
            
            # Project and normalize target hidden
            if target_hidden.shape[-1] != self.hidden_size:
                target_hidden = self.fc(target_hidden)
            target_hidden = self.hidden_norm(target_hidden)
            
            # Get noise embeddings
            noise_embeds = self.embed_tokens(input_ids)
            
            # Concatenate: [target_hidden, noise]
            hidden_states = torch.cat([target_hidden, noise_embeds], dim=0)
        
        # Track context length for output extraction
        # This comes from spec_info or is computed from shapes
        if hasattr(forward_batch, 'spec_info') and hasattr(forward_batch.spec_info, 'ctx_lens'):
            ctx_lens = forward_batch.spec_info.ctx_lens
        else:
            # Fallback: assume uniform distribution
            ctx_lens = None

        # Process through layers
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states, forward_batch)

        # Final norm
        hidden_states = self.norm(hidden_states)

        return hidden_states


class Qwen3ForCausalLMDFlash(nn.Module):
    """
    Qwen3 DFlash draft model wrapper using RadixAttention.
    
    This is the main entry point class that handles:
    - Model initialization
    - Weight loading (separate q/k/v → fused qkv_proj)
    - Logits processing
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
        
        self.model = Qwen3DFlashModel(config, quant_config, prefix=add_prefix("model", prefix))
        
        # LM head for token prediction
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )
        
        self.logits_processor = LogitsProcessor(config)

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
        
        Args:
            input_ids: Token IDs for noise tokens
            positions: Position IDs for full sequence
            forward_batch: ForwardBatch with attention metadata
            input_embeds: Optional pre-computed embeddings
            
        Returns:
            Logits for next token prediction
        """
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        
        # Get logits
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

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
                        skipped_weights.append({"original": original_name, "mapped": mapped_name, "reason": "not in params_dict"})
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
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
                    loaded_count += 1
                else:
                    skipped_weights.append({"original": original_name, "mapped": name, "reason": "not in params_dict (direct)"})
        
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
                    skipped_weights.append({"param": param_name, "reason": "incomplete stacked weights"})


# Backward compatibility alias
DFlashDraftModel = Qwen3ForCausalLMDFlash

# Register with SGLang's model registry
EntryClass = [Qwen3ForCausalLMDFlash]
