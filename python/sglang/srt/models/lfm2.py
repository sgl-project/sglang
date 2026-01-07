# sglang/srt/models/lfm2.py
# LFM2 implementation for SGLang
# Based on HuggingFace's implementation
#
# This version uses SGLang's hybrid caching infrastructure (HybridReqToTokenPool + MambaPool)
# while keeping the original working model structure and weight names.
#
# IMPORTANT: This file patches Lfm2Config at MODULE IMPORT TIME to add the properties
# required by model_runner.py for hybrid cache detection. You must also add Lfm2Config
# to the isinstance check in model_runner.py's mamba2_config property.

import logging
from typing import Iterable, List, Optional, Set, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from transformers import Lfm2Config

from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.linear import ColumnParallelLinear, QKVParallelLinear, RowParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix, make_layers

# Import for Mamba state management - use SGLang's actual classes
from sglang.srt.configs.mamba_utils import (
    Mamba2StateShape,
    Mamba2CacheParams,
)

logger = logging.getLogger(__name__)

# Debug flag - set to True to enable debug logging
DEBUG_LFM2 = False


def debug_tensor(name: str, t: torch.Tensor):
    if DEBUG_LFM2:
        logger.info(f"DEBUG {name}: shape={t.shape}, dtype={t.dtype}, "
                   f"min={t.min().item():.6f}, max={t.max().item():.6f}, "
                   f"mean={t.mean().item():.6f}, std={t.std().item():.6f}")


# ============================================================================
# Config Patching - MUST happen at module import time
# ============================================================================

def _patch_lfm2_config_class():
    """
    Patch Lfm2Config CLASS (not instance) with properties required by model_runner.py.
    
    This must happen at module import time, BEFORE model_runner.py checks the config type.
    
    model_runner.py's mamba2_config property does:
        if isinstance(config, FalconH1Config | NemotronHConfig | Lfm2Config):
            return config
    
    And then uses config.mamba2_cache_params to set up HybridReqToTokenPool.
    """
    if getattr(Lfm2Config, '_sglang_patched', False):
        return
    
    def _get_full_attention_layer_ids(self) -> List[int]:
        """Return indices of attention layers for KV cache."""
        return [i for i, lt in enumerate(self.layer_types) if lt == "full_attention"]
    
    def _get_linear_layer_ids(self) -> List[int]:
        """Return indices of conv layers for conv state cache."""
        return [i for i, lt in enumerate(self.layer_types) if lt in ("conv", "short_conv")]
    
    def _get_mamba_chunk_size(self) -> int:
        """Return chunk size for Mamba2 backend. LFM2 doesn't use chunking, return 1."""
        return 1
    
    def _get_mamba2_cache_params(self) -> Optional[Mamba2CacheParams]:
        """
        Get cache params for HybridReqToTokenPool initialization.
        
        Uses SGLang's Mamba2StateShape to describe the conv state shape.
        LFM2 only uses the conv state (no SSM temporal state), so we set
        state_size=0 which makes temporal state minimal.
        """
        from sglang.srt.layers.dp_attention import get_attention_tp_size
        
        conv_layer_ids = [i for i, lt in enumerate(self.layer_types) if lt in ("conv", "short_conv")]
        if not conv_layer_ids:
            return None
        
        hidden_size = self.hidden_size
        # conv_L_cache in config is kernel_size (e.g., 3)
        conv_kernel = int(self.conv_L_cache)
        L_cache = conv_kernel - 1  # actual cache size
        tp_size = get_attention_tp_size()
        
        # Create Mamba2StateShape compatible with SGLang's infrastructure
        # For LFM2 conv layers:
        # - Conv state shape: (hidden_size, L_cache) per layer
        # - No SSM temporal state (state_size=0)
        #
        # Mamba2StateShape.create() computes:
        # - conv_dim = intermediate_size // tp_size (for TP sharding)
        # - We want conv_dim = hidden_size, so intermediate_size = hidden_size * tp_size
        shape = Mamba2StateShape.create(
            tp_world_size=tp_size,
            intermediate_size=hidden_size * tp_size,  # Results in conv_dim = hidden_size
            n_groups=1,
            num_heads=1,
            head_dim=1,
            state_size=0,  # No SSM state, only conv state
            conv_kernel=conv_kernel,
        )
        
        return Mamba2CacheParams(shape=shape, layers=conv_layer_ids)
    
    # Patch the CLASS, not instances
    Lfm2Config.full_attention_layer_ids = property(_get_full_attention_layer_ids)
    Lfm2Config.linear_layer_ids = property(_get_linear_layer_ids)
    Lfm2Config.mamba2_cache_params = property(_get_mamba2_cache_params)
    Lfm2Config.mamba_chunk_size = property(_get_mamba_chunk_size)
    Lfm2Config._sglang_patched = True
    
    logger.info("Patched Lfm2Config class with SGLang hybrid cache properties")


# Patch at module import time - this runs when lfm2.py is imported
_patch_lfm2_config_class()


# ============================================================================
# Model Components
# ============================================================================

class Lfm2RMSNorm(nn.Module):
    """
    LFM2-specific RMSNorm that uses weight * x (NOT (1 + weight) * x like Gemma).
    This matches the HuggingFace Lfm2RMSNorm implementation exactly.
    """
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


class Lfm2MLP(nn.Module):
    """MLP with SwiGLU activation - uses w1/w2/w3 naming to match checkpoint."""
    def __init__(
        self,
        config: Lfm2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        intermediate_size = config.intermediate_size
        if config.block_auto_adjust_ff_dim:
            intermediate_size = int(2 * intermediate_size / 3)
            if config.block_ffn_dim_multiplier is not None:
                intermediate_size = int(config.block_ffn_dim_multiplier * intermediate_size)
                intermediate_size = config.block_multiple_of * (
                    (intermediate_size + config.block_multiple_of - 1) // config.block_multiple_of
                )

        self.w1 = ColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("w1", prefix),
        )
        self.w3 = ColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("w3", prefix),
        )
        self.w2 = RowParallelLinear(
            input_size=intermediate_size,
            output_size=config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("w2", prefix),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, _ = self.w1(x)
        up, _ = self.w3(x)
        h = F.silu(gate) * up
        out, _ = self.w2(h)
        return out


class Lfm2Attention(nn.Module):
    """Attention with RoPE and Q/K layernorm - matches checkpoint weight names."""
    def __init__(
        self,
        config: Lfm2Config,
        layer_id: int,
        attn_layer_id: int,  # Sequential ID for attention layers only (for KV cache)
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.attn_layer_id = attn_layer_id

        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", None) or (self.hidden_size // self.total_num_heads)
        self.scaling = self.head_dim**-0.5

        rope_parameters = getattr(config, "rope_parameters", None)
        if rope_parameters is not None and "rope_theta" in rope_parameters:
            self.rope_theta = rope_parameters["rope_theta"]
        else:
            self.rope_theta = getattr(config, "rope_theta", 10000)
        
        self.max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        self.rope_scaling = getattr(config, "rope_scaling", None)

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position_embeddings,
            rope_scaling=self.rope_scaling,
            base=self.rope_theta,
            is_neox_style=True,
            dtype=torch.get_default_dtype(),
        )

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )

        # Named out_proj to match checkpoint
        self.out_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("out_proj", prefix),
        )

        # Named q_layernorm/k_layernorm to match checkpoint
        self.q_layernorm = Lfm2RMSNorm(self.head_dim, eps=config.norm_eps)
        self.k_layernorm = Lfm2RMSNorm(self.head_dim, eps=config.norm_eps)

        self.num_local_q_heads = self.qkv_proj.num_heads
        self.num_local_kv_heads = self.qkv_proj.num_kv_heads

        self.attn = RadixAttention(
            num_heads=self.num_local_q_heads,
            head_dim=self.head_dim,
            scaling=self.scaling,
            num_kv_heads=self.num_local_kv_heads,
            layer_id=self.layer_id,  # Use global layer ID for routing in hybrid backend
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        T = hidden_states.shape[0]
        qkv, _ = self.qkv_proj(hidden_states)

        q_size = self.num_local_q_heads * self.head_dim
        kv_size = self.num_local_kv_heads * self.head_dim

        q, k, v = torch.split(qkv, [q_size, kv_size, kv_size], dim=-1)

        q = q.reshape(T, self.num_local_q_heads, self.head_dim)
        k = k.reshape(T, self.num_local_kv_heads, self.head_dim)

        q = self.q_layernorm(q.reshape(-1, self.head_dim)).reshape(T, self.num_local_q_heads, self.head_dim)
        k = self.k_layernorm(k.reshape(-1, self.head_dim)).reshape(T, self.num_local_kv_heads, self.head_dim)

        q, k = self.rotary_emb(positions, q, k)

        q = q.reshape(T, -1)
        k = k.reshape(T, -1)

        attn_out = self.attn(q, k, v, forward_batch)

        out, _ = self.out_proj(attn_out)
        return out


class Lfm2ShortConv(nn.Module):
    """
    Short conv implementation using SGLang's MambaPool for state management.
    
    This implementation:
    1. Uses nn.Linear for in_proj/out_proj (matching HF checkpoint)
    2. Accesses conv state through HybridReqToTokenPool.mamba2_layer_cache()
    3. Handles prefill and decode modes properly
    4. Is CUDA graph compatible (uses index_copy_ instead of .item())
    """

    def __init__(
        self,
        config: Lfm2Config,
        layer_idx: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        # conv_L_cache in config is the kernel size (e.g., 3), NOT kernel-1
        # The "cache" stores kernel_size - 1 values for causal conv
        self.conv_kernel = int(config.conv_L_cache)  # kernel_size from config
        self.L_cache = self.conv_kernel - 1  # actual cache size = kernel - 1
        self.bias = bool(config.conv_bias)
        self.hidden_size = config.hidden_size

        # Match HF exactly - use nn.Linear (not parallel versions)
        self.in_proj = nn.Linear(
            config.hidden_size,
            3 * config.hidden_size,
            bias=self.bias,
        )
        self.out_proj = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=self.bias,
        )

        # Depthwise conv1d with causal padding
        self.conv = nn.Conv1d(
            in_channels=config.hidden_size,
            out_channels=config.hidden_size,
            kernel_size=self.conv_kernel,
            groups=config.hidden_size,
            bias=self.bias,
            padding=self.L_cache,  # Causal padding = kernel_size - 1
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """
        Forward pass using SGLang's hybrid caching infrastructure.
        
        Conv state is accessed through:
        forward_batch.req_to_token_pool.mamba2_layer_cache(self.layer_idx)
        """
        forward_mode = forward_batch.forward_mode
        
        if forward_mode.is_idle():
            return hidden_states
        
        # Get conv cache through HybridReqToTokenPool
        # mamba2_layer_cache returns a cache object with .conv and .temporal attributes
        layer_cache = forward_batch.req_to_token_pool.mamba2_layer_cache(self.layer_idx)
        # conv is a list of tensors, one per conv state component
        # For Mamba2, conv[0] has shape [pool_size+1, conv_dim, conv_kernel-1]
        # For LFM2, this is [pool_size+1, hidden_size, L_cache]
        conv_state = layer_cache.conv[0]
        
        # Get request pool indices for current batch
        req_pool_indices = forward_batch.req_pool_indices
        
        if forward_mode.is_decode():
            return self._forward_decode(
                hidden_states, 
                conv_state,
                req_pool_indices,
            )
        else:
            # Prefill/extend mode
            seq_lens = getattr(forward_batch, 'extend_seq_lens', None)
            
            if seq_lens is not None and len(seq_lens) > 1:
                return self._forward_prefill_multi(
                    hidden_states,
                    conv_state,
                    req_pool_indices,
                    seq_lens,
                )
            else:
                return self._forward_prefill_single(
                    hidden_states,
                    conv_state,
                    req_pool_indices,
                )

    def _forward_prefill_single(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
        req_pool_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Prefill for a single sequence - matches HF slow_forward exactly."""
        T = hidden_states.shape[0]
        
        # Step 1: in_proj
        proj = self.in_proj(hidden_states)  # [T, 3H]
        
        # Step 2: transpose to [1, 3H, T] for conv
        proj_t = proj.transpose(0, 1).unsqueeze(0)  # [1, 3H, T]
        
        # Step 3: chunk into B, C, x
        B_gate, C_gate, x = proj_t.chunk(3, dim=1)  # each [1, H, T]
        
        # Step 4: Bx = B * x
        Bx = B_gate * x  # [1, H, T]
        
        # Step 5: conv with causal padding (output is truncated to T)
        conv_out = self.conv(Bx)[..., :T]  # [1, H, T]
        
        # Step 6: y = C * conv_out
        y = C_gate * conv_out  # [1, H, T]
        
        # Step 7: transpose back
        y = y.squeeze(0).transpose(0, 1)  # [T, H]
        
        # Step 8: out_proj
        y = self.out_proj(y)  # [T, H]
        
        # Store the final conv state (last L_cache values of Bx)
        if T >= self.L_cache:
            final_state = Bx[0, :, -self.L_cache:]  # [H, L_cache]
        else:
            final_state = F.pad(Bx[0], (self.L_cache - T, 0), value=0.0)  # [H, L_cache]
        
        # Store for the request using index_copy_ (CUDA graph compatible)
        # Ensure dtype matches conv_state (may be bfloat16 while final_state is float32)
        if req_pool_indices.numel() > 0:
            conv_state.index_copy_(0, req_pool_indices[:1].long(), final_state.unsqueeze(0).to(conv_state.dtype))
        
        return y

    def _forward_prefill_multi(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Process multiple sequences separately to avoid cross-contamination."""
        outputs = []
        start_idx = 0
        
        seq_lens_list = seq_lens.tolist() if isinstance(seq_lens, torch.Tensor) else list(seq_lens)
        req_pool_indices_long = req_pool_indices.long()
        
        for i, seq_len in enumerate(seq_lens_list):
            seq_len = int(seq_len)
            end_idx = start_idx + seq_len
            
            seq_hidden = hidden_states[start_idx:end_idx]
            T = seq_hidden.shape[0]
            
            # Process this sequence
            proj = self.in_proj(seq_hidden)
            proj_t = proj.transpose(0, 1).unsqueeze(0)
            B_gate, C_gate, x = proj_t.chunk(3, dim=1)
            Bx = B_gate * x
            conv_out = self.conv(Bx)[..., :T]
            y = C_gate * conv_out
            y = y.squeeze(0).transpose(0, 1)
            y = self.out_proj(y)
            
            outputs.append(y)
            
            # Store conv state for this sequence
            if T >= self.L_cache:
                final_state = Bx[0, :, -self.L_cache:]
            else:
                final_state = F.pad(Bx[0], (self.L_cache - T, 0), value=0.0)
            
            # Use index_copy_ for CUDA graph compatibility
            # Ensure dtype matches conv_state (may be bfloat16 while final_state is float32)
            conv_state.index_copy_(0, req_pool_indices_long[i:i+1], final_state.unsqueeze(0).to(conv_state.dtype))
            
            start_idx = end_idx
        
        return torch.cat(outputs, dim=0)

    def _forward_decode(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
        req_pool_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode mode: single token per sequence using cached state.
        CUDA graph compatible - uses only tensor operations, no .item() calls.
        """
        batch_size = hidden_states.shape[0]
        
        req_pool_indices_long = req_pool_indices.long()
        
        # in_proj for all tokens
        proj = self.in_proj(hidden_states)  # [B, 3H]
        
        # Split into gates
        B_gate, C_gate, x = proj.chunk(3, dim=-1)  # each [B, H]
        
        # Compute Bx
        Bx = B_gate * x  # [B, H]
        
        # Get conv weights - shape is [H, 1, kernel_size], we need [H, kernel_size]
        conv_weights = self.conv.weight[:, 0, :]  # [H, kernel_size]
        
        # Gather current states: [B, H, L_cache]
        current_states = conv_state[req_pool_indices_long]
        
        # Roll states left by 1 and insert new Bx at the end
        new_states = torch.cat([
            current_states[:, :, 1:],  # [B, H, L_cache-1]
            Bx.unsqueeze(-1)           # [B, H, 1]
        ], dim=-1)  # [B, H, L_cache]

        # Scatter updated states back using index_copy_ (CUDA graph compatible)
        # Ensure dtype matches conv_state (may be bfloat16 while new_states is float32)
        conv_state.index_copy_(0, req_pool_indices_long, new_states.to(conv_state.dtype))
        
        # Compute conv output: need full kernel_size inputs
        # Prepend zeros to get [B, H, kernel_size] for the conv operation
        # Actually, we need to apply the conv kernel to the last kernel_size values
        # The state has L_cache = kernel_size - 1 values, plus the new Bx makes kernel_size
        conv_input = torch.cat([
            current_states[:, :, -(self.conv_kernel - 1):],  # [B, H, kernel_size-1]
            Bx.unsqueeze(-1)                                  # [B, H, 1]
        ], dim=-1)  # [B, H, kernel_size]
        
        # Apply conv weights: element-wise multiply and sum
        conv_out = (conv_input * conv_weights.unsqueeze(0)).sum(dim=-1)  # [B, H]
        
        # Add bias if present
        if self.bias and self.conv.bias is not None:
            conv_out = conv_out + self.conv.bias
        
        # Apply output gate
        y = C_gate * conv_out  # [B, H]

        # Apply out_proj (ensure dtype matches model weights)
        y = self.out_proj(y.to(hidden_states.dtype))  # [B, H]

        return y


class Lfm2DecoderLayer(nn.Module):
    """Decoder layer - can be attention or conv based on config."""
    def __init__(
        self,
        config: Lfm2Config,
        layer_id: int,
        attn_layer_id: int,  # Sequential ID for attention layers (for KV cache)
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.layer_type = config.layer_types[layer_id]
        self.is_attention_layer = self.layer_type == "full_attention"

        self.operator_norm = Lfm2RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm = Lfm2RMSNorm(config.hidden_size, eps=config.norm_eps)

        if self.is_attention_layer:
            self.self_attn = Lfm2Attention(
                config=config,
                layer_id=layer_id,
                attn_layer_id=attn_layer_id,
                quant_config=quant_config,
                prefix=add_prefix("self_attn", prefix),
            )
        elif self.layer_type in ("conv", "short_conv"):
            # Named 'conv' to match checkpoint (model.layers.X.conv.*)
            self.conv = Lfm2ShortConv(
                config=config,
                layer_idx=layer_id,
                quant_config=quant_config,
                prefix=add_prefix("conv", prefix),
            )
        else:
            raise ValueError(f"Unknown layer type: {self.layer_type}")

        self.feed_forward = Lfm2MLP(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("feed_forward", prefix),
        )

    def forward(
        self,
        layer_id: int,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward matching HF exactly."""
        if not forward_batch.forward_mode.is_idle():
            residual = hidden_states
            normed = self.operator_norm(hidden_states)
            
            if self.is_attention_layer:
                hidden_states = self.self_attn(
                    positions=positions,
                    hidden_states=normed,
                    forward_batch=forward_batch
                )
            else:
                hidden_states = self.conv(
                    hidden_states=normed,
                    forward_batch=forward_batch,
                )
            
            hidden_states = hidden_states + residual
            hidden_states = hidden_states + self.feed_forward(self.ffn_norm(hidden_states))
        
        return hidden_states, residual


class Lfm2Model(nn.Module):
    def __init__(
        self,
        config: Lfm2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            prefix=add_prefix("embed_tokens", prefix),
        )

        # Compute attention layer IDs (sequential numbering for KV cache)
        attn_layer_ids = []
        attn_count = 0
        for layer_type in config.layer_types:
            if layer_type == "full_attention":
                attn_layer_ids.append(attn_count)
                attn_count += 1
            else:
                attn_layer_ids.append(-1)
        
        self.num_attention_layers = attn_count
        
        logger.info(f"LFM2 model has {attn_count} attention layers and "
                   f"{len(config.layer_types) - attn_count} conv layers "
                   f"out of {config.num_hidden_layers} total")

        def get_layer(idx: int, prefix: str, **kwargs):
            return Lfm2DecoderLayer(
                config=config,
                layer_id=idx,
                attn_layer_id=attn_layer_ids[idx],
                quant_config=quant_config,
                prefix=prefix,
            )

        self.layers = make_layers(config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers")
        
        # Named embedding_norm to match checkpoint
        self.embedding_norm = Lfm2RMSNorm(config.hidden_size, eps=config.norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)

        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(
                layer_id=i,
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                forward_batch=forward_batch,
            )

        hidden_states = self.embedding_norm(hidden_states)
        return hidden_states


class Lfm2ForCausalLM(nn.Module):
    """
    LFM2 for Causal Language Modeling.
    
    This model has a hybrid architecture with both attention and conv layers.
    - Attention layers use standard KV cache (managed by SGLang)
    - Conv layers use MambaPool for state caching (via HybridReqToTokenPool)
    
    IMPORTANT: For this to work, you must also modify model_runner.py to add
    Lfm2Config to the mamba2_config property's isinstance check:
    
        from transformers import Lfm2Config
        ...
        if isinstance(config, FalconH1Config | NemotronHConfig | Lfm2Config):
            return config
    """
    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: Lfm2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.pp_group = get_pp_group()
        assert self.pp_group.is_first_rank and self.pp_group.is_last_rank

        self.quant_config = quant_config
        self.model = Lfm2Model(config, quant_config, prefix=add_prefix("model", prefix))
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            org_num_embeddings=config.vocab_size,
            prefix=add_prefix("lm_head", prefix),
        )
        self.logits_processor = LogitsProcessor(config)
        
        # Store number of attention layers for KV cache sizing
        self.num_attention_layers = self.model.num_attention_layers

    def get_num_kv_cache_layers(self) -> int:
        """Return the number of layers that need KV cache (attention layers only)."""
        return self.num_attention_layers

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        hidden_states = self.model(
            input_ids, 
            positions, 
            forward_batch, 
            inputs_embeds,
        )
        return self.logits_processor(input_ids, hidden_states, self.lm_head, forward_batch)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_mtp: bool = False) -> Set[str]:
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

        params_dict = dict(self.named_parameters())

        logger.info("=== Model parameter names (first 30) ===")
        for i, (name, param) in enumerate(params_dict.items()):
            if i >= 30:
                break
            logger.info(f"  {name}: {param.shape}")

        loaded_params: Set[str] = set()
        conv_weights_loaded = 0
        missing_params = []
        
        embed_tokens_weight = None

        for name, loaded_weight in weights:
            original_name = name
            
            if "rotary_emb.inv_freq" in name:
                continue
            
            if "embed_tokens.weight" in name:
                embed_tokens_weight = loaded_weight

            if conv_weights_loaded < 5 and ".conv." in name:
                logger.info(f"Loading: {name}, shape: {loaded_weight.shape}")

            # Handle QKV stacking
            did_stack = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    did_stack = True
                    break
                if name not in params_dict:
                    did_stack = True
                    break
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader")
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                did_stack = True
                break
            if did_stack:
                continue

            if name.endswith(".bias") and name not in params_dict:
                continue
                
            if name not in params_dict:
                if len(missing_params) < 20:
                    missing_params.append(f"{original_name} -> {name}")
                continue

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
            
            if ".conv." in name:
                conv_weights_loaded += 1

        # Handle tied lm_head weight
        if "lm_head.weight" not in loaded_params and "lm_head.weight" in params_dict:
            if embed_tokens_weight is not None:
                logger.info("Tying lm_head.weight to embed_tokens.weight")
                param = params_dict["lm_head.weight"]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, embed_tokens_weight)
                loaded_params.add("lm_head.weight")
            else:
                logger.warning("lm_head.weight not found and no embed_tokens.weight to tie")

        if missing_params:
            logger.warning(f"Missing params (first 20): {missing_params}")
            
        logger.info(f"Loaded {conv_weights_loaded} conv weight tensors")
        logger.info(f"Total loaded params: {len(loaded_params)}")
        
        unloaded = set(params_dict.keys()) - loaded_params
        if unloaded:
            logger.warning(f"Unloaded params ({len(unloaded)}): {list(unloaded)[:10]}...")
            
        return loaded_params


EntryClass = [Lfm2ForCausalLM]
