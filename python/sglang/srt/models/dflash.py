# Adapted from the DFlash reference implementation (HF) but implemented with
# SGLang primitives (RadixAttention + SGLang KV cache). This model intentionally
# does not include token embeddings or an LM head; DFlash uses the target model's
# embedding/lm_head.

from __future__ import annotations

import logging
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.radix_attention import AttentionType, RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.utils import apply_qk_norm

logger = logging.getLogger(__name__)


class DFlashAttention(nn.Module):
    def __init__(self, config, layer_id: int) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        num_heads = int(config.num_attention_heads)
        num_kv_heads = int(getattr(config, "num_key_value_heads", num_heads))
        head_dim = int(getattr(config, "head_dim", hidden_size // num_heads))

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim

        attention_bias = bool(getattr(config, "attention_bias", False))
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))

        self.q_proj = nn.Linear(hidden_size, self.q_size, bias=attention_bias)
        self.k_proj = nn.Linear(hidden_size, self.kv_size, bias=attention_bias)
        self.v_proj = nn.Linear(hidden_size, self.kv_size, bias=attention_bias)
        self.o_proj = nn.Linear(self.q_size, hidden_size, bias=attention_bias)

        # Per-head Q/K RMSNorm, matching HF Qwen3.
        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)

        rope_theta = float(getattr(config, "rope_theta", 1000000))
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = int(getattr(config, "max_position_embeddings", 32768))
        self.rotary_emb = get_rope(
            head_dim,
            rotary_dim=head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )

        self.scaling = head_dim**-0.5
        # DFlash uses non-causal attention over the draft block.
        self.attn = RadixAttention(
            num_heads=num_heads,
            head_dim=head_dim,
            scaling=self.scaling,
            num_kv_heads=num_kv_heads,
            layer_id=layer_id,
            attn_type=AttentionType.ENCODER_ONLY,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q, k = apply_qk_norm(
            q=q,
            k=k,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
            head_dim=self.head_dim,
        )
        q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v, forward_batch)
        return self.o_proj(attn_output)


class DFlashMLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        intermediate_size = int(getattr(config, "intermediate_size", 0))
        if intermediate_size <= 0:
            raise ValueError(f"Invalid intermediate_size={intermediate_size} for DFlash MLP.")

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DFlashDecoderLayer(nn.Module):
    def __init__(self, config, layer_id: int) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))

        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = DFlashAttention(config=config, layer_id=layer_id)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = DFlashMLP(config=config)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class DFlashDraftModel(nn.Module):
    """SGLang-native DFlash draft model (no embedding / lm_head weights).

    The checkpoint provides:
      - transformer weights for `layers.*`
      - `fc.weight`, `hidden_norm.weight` for projecting target context features
      - `norm.weight` for final normalization
    """

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__()
        self.config = config

        hidden_size = int(config.hidden_size)
        num_layers = int(config.num_hidden_layers)
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))

        self.layers = nn.ModuleList(
            [DFlashDecoderLayer(config=config, layer_id=i) for i in range(num_layers)]
        )
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)

        # Project per-token target context features:
        # concat(num_layers * hidden_size) -> hidden_size
        self.fc = nn.Linear(num_layers * hidden_size, hidden_size, bias=False)
        self.hidden_norm = RMSNorm(hidden_size, eps=rms_norm_eps)

        self.block_size = int(getattr(config, "block_size", 16))

    def project_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:
        """Project concatenated target-layer hidden states into draft hidden_size."""
        return self.hidden_norm(self.fc(target_hidden))

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
        pp_proxy_tensors=None,
    ) -> torch.Tensor:
        if input_embeds is None:
            raise ValueError(
                "DFlashDraftModel requires `input_embeds` (use the target embedding)."
            )
        hidden_states = input_embeds

        for layer in self.layers:
            hidden_states = layer(positions, hidden_states, forward_batch)

        if hidden_states.numel() == 0:
            return hidden_states
        return self.norm(hidden_states)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if name.endswith(".bias") and name not in params_dict:
                # Some quantized checkpoints may have extra biases.
                continue
            param = params_dict.get(name)
            if param is None:
                # Ignore unexpected weights (e.g., HF rotary caches).
                continue
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)


EntryClass = DFlashDraftModel

