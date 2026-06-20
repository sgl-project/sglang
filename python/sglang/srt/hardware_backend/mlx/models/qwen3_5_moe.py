"""Native MLX implementation of Qwen3.5 MoE (a.k.a. Qwen3.6-35B-A3B).

This is the text-only model extracted from
`Qwen3_5MoeForConditionalGeneration`.  Vision tower and MTP head live in
their own modules (TODO: Phase 4 / Phase 3 of the plan).

Architecture summary (Qwen3.6-35B-A3B config):
  * 40 hidden layers, hidden_size 2048, head_dim 256
  * 16 attention heads, 2 KV heads (heavy GQA)
  * Hybrid attention 3:1 — `GatedDeltaNet` (linear) for 3 of every 4
    layers, `Attention` (full) every 4th
  * MoE: 256 experts, 8 active per token; SwitchGLU per expert
  * `attn_output_gate: true` on full attention
  * `partial_rotary_factor=0.25` RoPE with `mrope_section=[11,11,10]`
  * Shared expert path (single MLP, not routed)
  * `tie_word_embeddings=False`

This stub is the Phase 1 MVP.  It defines `ModelArgs` + a no-op `Model`
class so the sglang MLX backend can route Qwen3.5 MoE paths to a
sglang-owned module instead of `mlx_lm.models.qwen3_5_moe`.  Subsequent
phases will fill in the actual layers:

  Phase 2 — RoPE + Attention (full) + GatedDeltaNet (linear) + MoE +
            DecoderLayer + Model + ForCausalLM + weight load
  Phase 3 — MTP head (multi-token prediction)
  Phase 4 — Vision tower + multimodal projector
  Phase 5 — Fused SwiGLU + KV cache tuning for hybrid attention

Reference: mlx_lm 0.31.3 `qwen3_5.py` (architecture), `qwen3_5_moe.py`
(weight name remapping), `gated_delta.py` (linear attention primitive).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import BaseModelArgs


@dataclass
class TextModelArgs(BaseModelArgs):
    """Subset of fields needed to construct a Qwen3.5 MoE text model.

    The defaults match Qwen3.6-35B-A3B.  Phase 2 will add the rest of
    the fields (intermediate_size, num_experts_per_tok details, etc.).
    """

    model_type: str = "qwen3_5_moe_text"
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_hidden_layers: int = 40
    num_attention_heads: int = 16
    rms_norm_eps: float = 1e-6
    vocab_size: int = 248320
    num_key_value_heads: int = 2
    max_position_embeddings: int = 262144
    head_dim: int = 256
    full_attention_interval: int = 4

    linear_num_value_heads: int = 32
    linear_num_key_heads: int = 16
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4

    tie_word_embeddings: bool = False
    attention_bias: bool = False
    attn_output_gate: bool = True

    num_experts: int = 256
    num_experts_per_tok: int = 8
    decoder_sparse_step: int = 1
    shared_expert_intermediate_size: int = 512
    moe_intermediate_size: int = 512
    norm_topk_prob: bool = True
    router_aux_loss_coef: float = 0.001

    rope_parameters: Optional[Dict[str, Union[float, str, bool, List[int]]]] = field(
        default_factory=lambda: {
            "type": "default",
            "mrope_section": [11, 11, 10],
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
        }
    )
    partial_rotary_factor: float = 0.25
    rope_theta: float = 10000000.0
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None


class Model(nn.Module):
    """Phase 1 MVP stub.

    Holds the config and a minimal structure so sglang MLX backend can
    route to it.  Forward is intentionally a no-op — Phase 2 wires up
    the real layers.
    """

    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type

    def __call__(self, *args, **kwargs):  # pragma: no cover - stub
        raise NotImplementedError(
            "Qwen3.5 MoE native forward is implemented in Phase 2."
        )

    def sanitize(self, weights: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 1 passthrough.

        Phase 2 will remap HF weight names (e.g. `experts.gate_up_proj` →
        `switch_mlp.{gate,up}_proj.weight`, `experts.down_proj` →
        `switch_mlp.down_proj.weight`, drop `model.visual.*`, etc.) —
        see `mlx_lm.models.qwen3_5_moe.Model.sanitize` for the upstream
        version we will mirror.
        """
        return dict(weights)

    @property
    def quant_predicate(self):
        """Phase 1 predicate that excludes nothing.

        Phase 2 will return a predicate that excludes `mlp.gate` and
        `shared_expert.*` from quantization (matching the upstream
        Qwen3.5 MoE behaviour).
        """
        return lambda path, _module: True

    @property
    def layers(self):
        """Phase 1 returns an empty list so attention patching is a no-op.

        Phase 2 will return the real `DecoderLayer` list and the existing
        sglang `model_patching.py` will pick it up.
        """
        return []
