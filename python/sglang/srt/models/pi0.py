# Copyright 2025 SGLang Team
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
Inference-only π0 (Pi-Zero) VLA model for SGLang.

π0 is a Vision-Language-Action model from Physical Intelligence that uses
flow matching to generate continuous robot actions. The architecture is:

1. **PaliGemma backbone** (SigLIP vision + Gemma 2B LM):
   - SigLIP encodes images into visual tokens.
   - Gemma 2B embeds tokenized language instructions.
   - Together these form the "prefix" tokens.

2. **Gemma 300M action expert**:
   - Processes robot state, noisy actions, and timestep into "suffix" tokens.
   - Shares attention layers with the PaliGemma backbone via cross-attention.

3. **Flow matching head**:
   - Iterative denoising: starts from Gaussian noise at t=1, denoises to
     actions at t=0.
   - Each denoising step runs the action expert once with a prefix KV cache.
   - Output: continuous action chunk ``(batch_size, action_horizon, action_dim)``.

Attention pattern:
   - Prefix tokens (images + language): bidirectional attention among themselves.
   - Suffix tokens (state + actions): attend to prefix and to each other causally.
   - Prefix tokens do NOT attend to suffix tokens.

This file targets **modern transformers** (the refactored
``PaliGemmaForConditionalGeneration`` with an inner ``PaliGemmaModel`` and
the ``DynamicCache``-based KV cache). We do NOT rely on OpenPI's
``transformers_replace`` monkey-patch — instead we walk Gemma decoder
layers ourselves in ``_compute_layer_*`` so we never go through
``GemmaModel.forward`` (which has version-dependent mask/KV-cache behaviour).

Reference implementations:
   - OpenPI: openpi/src/openpi/models_pytorch/pi0_pytorch.py
   - OpenPI: openpi/src/openpi/models_pytorch/gemma_pytorch.py
   - LeRobot: lerobot/src/lerobot/policies/pi0/modeling_pi0.py

Weight source: https://huggingface.co/lerobot/pi0_base
"""

import logging
import math
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma.modeling_gemma import (
    GemmaForCausalLM,
    apply_rotary_pos_emb,
)
from transformers.models.paligemma.modeling_paligemma import (
    PaliGemmaForConditionalGeneration,
)

from sglang.srt.configs.pi0 import Pi0Config  # noqa: F401 – re-exported for processor
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
)
from sglang.srt.managers.schedule_batch import MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils.common import print_warning_once

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────
DEFAULT_ACTION_DIM = 32
DEFAULT_ACTION_HORIZON = 50
DEFAULT_MAX_TOKEN_LEN = 48
DEFAULT_NUM_INFERENCE_STEPS = 10
DEFAULT_IMAGE_RESOLUTION = (224, 224)

# Large negative value to fill masked-out positions in a float attention mask.
# Matches OpenPI's constant exactly so that numerics line up during parity.
# Ref: openpi/src/openpi/models/gemma.py
OPENPI_ATTENTION_MASK_VALUE = -2.3819763e38


# ──────────────────────────────────────────────────────────────────────
# Gemma variant configs (matches openpi/models/gemma.py get_config)
# ──────────────────────────────────────────────────────────────────────


class GemmaVariantConfig:
    def __init__(self, width, depth, mlp_dim, num_heads, num_kv_heads, head_dim):
        self.width = width
        self.depth = depth
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim


def get_gemma_config(variant: str) -> GemmaVariantConfig:
    if variant == "gemma_2b":
        return GemmaVariantConfig(2048, 18, 16384, 8, 1, 256)
    elif variant == "gemma_300m":
        return GemmaVariantConfig(1024, 18, 4096, 8, 1, 256)
    else:
        raise ValueError(f"Unknown variant: {variant}")


# ──────────────────────────────────────────────────────────────────────
# Utility functions (match openpi/models_pytorch/pi0_pytorch.py)
# ──────────────────────────────────────────────────────────────────────


def create_sinusoidal_pos_embedding(
    time: torch.Tensor,
    dimension: int,
    min_period: float = 4e-3,
    max_period: float = 4.0,
    device: torch.device = None,
) -> torch.Tensor:
    """Compute a sine/cosine positional embedding for scalar timesteps.

    Ref: openpi/models_pytorch/pi0_pytorch.py create_sinusoidal_pos_embedding
    """
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError("time tensor must be 1-D (batch_size,)")
    if device is None:
        device = time.device

    # Use float64 for the log-linear sweep and for the inner products, to
    # match the reference implementation's numerical behaviour exactly.
    dtype = torch.float64
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None].to(dtype)
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def make_att_2d_masks(
    pad_masks: torch.Tensor, att_masks: torch.Tensor
) -> torch.Tensor:
    """Build a 2D attention mask from a padding mask and an autoregressive mask.

    Ref: openpi/models_pytorch/pi0_pytorch.py make_att_2d_masks
    """
    if att_masks.ndim != 2:
        raise ValueError(f"att_masks must be 2-D, got {att_masks.ndim}-D")
    if pad_masks.ndim != 2:
        raise ValueError(f"pad_masks must be 2-D, got {pad_masks.ndim}-D")

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def _build_norm_buffers(
    norm_stats: Optional[Dict], key: str
) -> Optional[Dict[str, torch.Tensor]]:
    """Parse a ``norm_stats[key]`` entry into CPU tensors, or ``None``.

    Matches LeRobot's ``NormalizationMode``:
      - ``mean_std`` : ``forward = (x - mean) / std``, ``inverse = x * std + mean``
      - ``min_max``  : forward maps into ``[-1, 1]``, inverse maps back.
    """
    if not norm_stats or not isinstance(norm_stats, dict):
        return None
    entry = norm_stats.get(key)
    if not entry:
        return None
    mode = str(entry.get("mode", "mean_std")).lower()
    if mode == "mean_std":
        mean = entry.get("mean")
        std = entry.get("std")
        if mean is None or std is None:
            return None
        return {
            "mode": mode,
            "mean": torch.as_tensor(mean, dtype=torch.float32),
            "std": torch.as_tensor(std, dtype=torch.float32),
        }
    if mode == "min_max":
        lo = entry.get("min")
        hi = entry.get("max")
        if lo is None or hi is None:
            return None
        return {
            "mode": mode,
            "min": torch.as_tensor(lo, dtype=torch.float32),
            "max": torch.as_tensor(hi, dtype=torch.float32),
        }
    return None


def _apply_norm(
    x: torch.Tensor,
    stats: Optional[Dict[str, torch.Tensor]],
    inverse: bool,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Apply (un)normalization using the given stats. No-op if ``stats`` is None.

    Broadcasts over leading dims; stats vectors align with the last dim. When
    ``x`` has more last-dim entries than the stats (π0 pads state/action to
    ``max_dim``), only the first ``len(stats)`` entries are transformed and
    the padded tail is left untouched.
    """
    if stats is None:
        return x
    mode = stats["mode"]
    if mode == "mean_std":
        mean = stats["mean"].to(device=x.device, dtype=x.dtype)
        std = stats["std"].to(device=x.device, dtype=x.dtype)
        valid = mean.shape[0]
        head = x[..., :valid]
        if inverse:
            head = head * std + mean
        else:
            head = (head - mean) / (std + eps)
        if valid == x.shape[-1]:
            return head
        return torch.cat([head, x[..., valid:]], dim=-1)
    if mode == "min_max":
        lo = stats["min"].to(device=x.device, dtype=x.dtype)
        hi = stats["max"].to(device=x.device, dtype=x.dtype)
        valid = lo.shape[0]
        head = x[..., :valid]
        denom = (hi - lo).clamp_min(eps)
        if inverse:
            head = (head + 1.0) * 0.5 * denom + lo
        else:
            head = 2.0 * (head - lo) / denom - 1.0
        if valid == x.shape[-1]:
            return head
        return torch.cat([head, x[..., valid:]], dim=-1)
    return x


def prepare_attention_masks_4d(att_2d_masks: torch.Tensor) -> torch.Tensor:
    """Convert ``(B, S, S)`` bool masks to ``(B, 1, S, S)`` float masks.

    ``True`` → 0.0 (attend), ``False`` → ``OPENPI_ATTENTION_MASK_VALUE``.
    Ref: openpi PI0Pytorch._prepare_attention_masks_4d
    """
    att_2d_masks_4d = att_2d_masks[:, None, :, :]
    return torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)


# ──────────────────────────────────────────────────────────────────────
# Dual-backbone: PaliGemma + Action Expert
# Ref: openpi/models_pytorch/gemma_pytorch.py PaliGemmaWithExpertModel
# Ref: lerobot/policies/pi0/modeling_pi0.py PaliGemmaWithExpertModel
# ──────────────────────────────────────────────────────────────────────


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads for grouped-query attention.

    ``(B, num_kv_heads, S, D) → (B, num_kv_heads * n_rep, S, D)``.
    Matches ``transformers.models.gemma.modeling_gemma.repeat_kv``.
    """
    if n_rep == 1:
        return hidden_states
    b, nh, s, d = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(b, nh, n_rep, s, d)
    return hidden_states.reshape(b, nh * n_rep, s, d)


def _attend(query_states, key_states, value_states, attention_mask, num_kv_groups, scaling):
    """Manual eager attention: ``softmax(Q Kᵀ · scale + mask) · V``.

    Important: we slice ``attention_mask[..., :key_states.shape[-2]]`` so that
    the same ``(B, 1, Q, prefix+suffix)`` mask produced by ``denoise_step``
    can be re-used both in the prefix pass (K length = prefix) and in the
    suffix pass (K length = prefix + suffix after we concat the cached prefix).
    Stock transformers' ``eager_attention_forward`` adds the mask without
    slicing and therefore breaks when the two dims disagree.
    """
    k = _repeat_kv(key_states, num_kv_groups)
    v = _repeat_kv(value_states, num_kv_groups)
    attn_weights = torch.matmul(query_states, k.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, : k.shape[-2]]
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    return torch.matmul(attn_weights, v)


def _compute_layer_prefix_only(layer_idx, hidden_states, attention_mask, position_ids, paligemma):
    """Run one PaliGemma LM layer on the prefix only, returning the layer
    output **and** the post-RoPE ``(k, v)`` so the caller can cache them for
    the suffix pass."""
    model = paligemma.model.language_model
    layer = model.layers[layer_idx]
    residual = hidden_states
    x = layer.input_layernorm(hidden_states)

    hidden_shape = (*x.shape[:-1], -1, layer.self_attn.head_dim)
    q = layer.self_attn.q_proj(x).view(hidden_shape).transpose(1, 2)
    k = layer.self_attn.k_proj(x).view(hidden_shape).transpose(1, 2)
    v = layer.self_attn.v_proj(x).view(hidden_shape).transpose(1, 2)

    cos, sin = model.rotary_emb(v, position_ids)
    q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

    att = _attend(
        q, k, v, attention_mask,
        num_kv_groups=layer.self_attn.num_key_value_groups,
        scaling=1.0 / math.sqrt(layer.self_attn.head_dim),
    )
    att = att.transpose(1, 2).reshape(q.shape[0], -1, q.shape[1] * layer.self_attn.head_dim)

    out = layer.self_attn.o_proj(att) + residual
    after_resid = out
    out = layer.mlp(layer.post_attention_layernorm(out)) + after_resid
    return out, (k, v)


def _compute_layer_suffix_only(
    layer_idx, hidden_states, prefix_kv, attention_mask, position_ids,
    paligemma, gemma_expert,
):
    """Run one action-expert layer on the suffix, attending to the cached
    prefix K/V concatenated with freshly computed suffix K/V.

    ``prefix_kv`` is a ``(k_prefix, v_prefix)`` tuple, shape
    ``(B, num_kv_heads, prefix_len, head_dim)``, post-RoPE, produced by the
    corresponding ``_compute_layer_prefix_only`` call. Bypasses
    ``GemmaModel.forward`` entirely — no HF mask rebuild, no cache format
    gymnastics.
    """
    layer = gemma_expert.model.layers[layer_idx]
    residual = hidden_states
    x = layer.input_layernorm(hidden_states)

    hidden_shape = (*x.shape[:-1], -1, layer.self_attn.head_dim)
    q = layer.self_attn.q_proj(x).view(hidden_shape).transpose(1, 2)
    k_suf = layer.self_attn.k_proj(x).view(hidden_shape).transpose(1, 2)
    v_suf = layer.self_attn.v_proj(x).view(hidden_shape).transpose(1, 2)

    # RoPE frequencies are shared between PaliGemma and the expert.
    cos, sin = gemma_expert.model.rotary_emb(v_suf, position_ids)
    q, k_suf = apply_rotary_pos_emb(q, k_suf, cos, sin, unsqueeze_dim=1)

    # Concatenate cached prefix K/V (possibly different dtype) with suffix K/V.
    k_prefix, v_prefix = prefix_kv
    k = torch.cat([k_prefix.to(k_suf.dtype), k_suf], dim=2)
    v = torch.cat([v_prefix.to(v_suf.dtype), v_suf], dim=2)

    att = _attend(
        q, k, v, attention_mask,
        num_kv_groups=layer.self_attn.num_key_value_groups,
        scaling=1.0 / math.sqrt(layer.self_attn.head_dim),
    )
    att = att.transpose(1, 2).reshape(q.shape[0], -1, q.shape[1] * layer.self_attn.head_dim)

    out = layer.self_attn.o_proj(att) + residual
    after_resid = out
    out = layer.mlp(layer.post_attention_layernorm(out)) + after_resid
    return out


class PaliGemmaWithActionExpert(nn.Module):
    """Dual-backbone transformer: PaliGemma (Gemma 2B) + Action Expert (Gemma 300M).

    ``forward`` has two inference modes:
      - **prefix_only**: ``inputs_embeds=[prefix, None]`` + ``use_cache=True``
        → compute prefix hidden states + a layer-wise K/V cache.
      - **suffix_only**: ``inputs_embeds=[None, suffix]`` + the cache from the
        previous prefix pass → compute expert hidden states with cross-attention
        over the concatenated (prefix, suffix) K/V.

    Both modes walk Gemma decoder layers manually through ``_compute_layer_*``
    instead of ``GemmaModel.forward`` — that way we own the attention mask
    handling and the KV cache format (a plain ``list[(k, v)]`` one entry
    per layer).

    Ref: lerobot/policies/pi0/modeling_pi0.py PaliGemmaWithExpertModel
    """

    def __init__(self, vlm_config, action_expert_config):
        super().__init__()

        # Build HF PaliGemma config from the variant dims we expose.
        # NOTE: we do NOT set use_adarms / adarms_cond_dim — those are
        # OpenPI-only attributes used by π0.5, not π0.
        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        # transformers ≥ 5 uses ``dtype``; older versions still accept
        # ``torch_dtype``. We set the canonical name; HF's PretrainedConfig
        # normalizes one to the other internally.
        vlm_config_hf.text_config.dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.dtype = "float32"

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            dtype="float32",
        )

        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
        # The action expert doesn't embed tokens — it only consumes the
        # suffix state/action embeddings we feed in.
        self.gemma_expert.model.embed_tokens = None

    def embed_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode images with SigLIP vision tower + PaliGemma projector.

        We run the two steps explicitly instead of calling
        ``PaliGemmaModel.get_image_features`` because that convenience
        method historically has returned either the projected tensor (old)
        or the raw vision-tower output (new). Being explicit makes us
        independent of which transformers release we're on.
        """
        # Shapes: pixel_values (B, 3, 224, 224) → SigLIP (B, 256, 1152)
        #                                      → projector (B, 256, 2048)
        vision_outputs = self.paligemma.model.vision_tower(pixel_values)
        image_features = vision_outputs.last_hidden_state
        return self.paligemma.model.multi_modal_projector(image_features)

    def embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embed language tokens with PaliGemma's embedding table."""
        return self.paligemma.model.language_model.embed_tokens(tokens)

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[List[Optional[torch.Tensor]]] = None,
        use_cache: bool = False,
    ):
        """Dispatch to prefix_only / suffix_only and return
        ``([prefix_out, suffix_out], past_key_values_or_None)``.
        """
        num_layers = self.paligemma.config.text_config.num_hidden_layers
        pali_lm = self.paligemma.model.language_model
        expert_lm = self.gemma_expert.model

        if inputs_embeds[1] is None:
            # Prefix-only: PaliGemma LM on (images + language) tokens; the
            # per-layer post-RoPE K/V is collected into a list that the
            # suffix pass will consume directly.
            hidden_states = inputs_embeds[0]
            kv_list: List[Tuple[torch.Tensor, torch.Tensor]] = []
            for layer_idx in range(num_layers):
                hidden_states, kv = _compute_layer_prefix_only(
                    layer_idx, hidden_states, attention_mask, position_ids,
                    paligemma=self.paligemma,
                )
                kv_list.append(kv)
            hidden_states = pali_lm.norm(hidden_states)
            return [hidden_states, None], (kv_list if use_cache else None)

        if inputs_embeds[0] is not None:
            raise ValueError(
                "PaliGemmaWithActionExpert.forward only supports prefix-only "
                "or suffix-only dispatch; got both inputs_embeds populated."
            )
        # Suffix-only: action expert on (state + noisy_actions + time),
        # attending to the cached prefix K/V.
        if not isinstance(past_key_values, list):
            raise TypeError(
                "suffix_only forward expects past_key_values to be the "
                "list[(k, v)] produced by a previous prefix_only forward; "
                f"got {type(past_key_values)}"
            )
        hidden_states = inputs_embeds[1]
        for layer_idx in range(num_layers):
            hidden_states = _compute_layer_suffix_only(
                layer_idx, hidden_states, past_key_values[layer_idx],
                attention_mask, position_ids,
                paligemma=self.paligemma, gemma_expert=self.gemma_expert,
            )
        hidden_states = expert_lm.norm(hidden_states)
        return [None, hidden_states], None


# ──────────────────────────────────────────────────────────────────────
# Main π0 Model
# ──────────────────────────────────────────────────────────────────────


class Pi0ForActionPrediction(nn.Module):
    """π0 VLA model for robot action prediction via flow matching.

    Inference flow:
      1. Embed prefix (images + language) → prefix tokens.
      2. Forward prefix through PaliGemma → layer-wise KV cache.
      3. For each denoising step ``t = 1.0, 1-dt, ..., 0``:
         a. Embed suffix (state + x_t + timestep) → suffix tokens.
         b. Forward suffix through the action expert with the prefix cache.
         c. ``x_t = x_t + dt * v_t`` (Euler integration).
      4. Return ``x_0`` as the predicted action chunk.
    """

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config

        self.action_dim = getattr(config, "max_action_dim", DEFAULT_ACTION_DIM)
        # ``max_state_dim`` is independent from ``max_action_dim`` — e.g. Aloha
        # has state_dim=14, action_dim=32, both padded to their own max.  Fall
        # back to action_dim for older configs that only expose a single size.
        self.max_state_dim = getattr(config, "max_state_dim", self.action_dim)
        self.action_horizon = getattr(config, "chunk_size", DEFAULT_ACTION_HORIZON)
        self.num_inference_steps = getattr(
            config, "num_inference_steps", DEFAULT_NUM_INFERENCE_STEPS
        )

        paligemma_variant = getattr(config, "paligemma_variant", "gemma_2b")
        action_expert_variant = getattr(config, "action_expert_variant", "gemma_300m")
        vlm_config = get_gemma_config(paligemma_variant)
        expert_config = get_gemma_config(action_expert_variant)
        self.vlm_width = vlm_config.width
        self.expert_width = expert_config.width

        # Dual backbone
        self.paligemma_with_expert = PaliGemmaWithActionExpert(vlm_config, expert_config)

        # Action chunk projections (openpi pi0_pytorch.py).
        self.action_in_proj = nn.Linear(self.action_dim, self.expert_width)
        self.action_out_proj = nn.Linear(self.expert_width, self.action_dim)

        # State is projected to the expert dim once (π0 only, not π0.5).
        # Ref: lerobot modeling_pi0.py ``nn.Linear(config.max_state_dim, ...)``
        self.state_proj = nn.Linear(self.max_state_dim, self.expert_width)

        # Timestep + action fusion MLP (2W → W → W).
        self.action_time_mlp_in = nn.Linear(2 * self.expert_width, self.expert_width)
        self.action_time_mlp_out = nn.Linear(self.expert_width, self.expert_width)

        # Optional mean/std normalization stats (LeRobot NormalizerProcessorStep
        # semantics). Format on the HF config (all optional):
        #     norm_stats = {
        #         "state":  {"mode": "mean_std", "mean": [...], "std": [...]},
        #         "action": {"mode": "mean_std", "mean": [...], "std": [...]},
        #     }
        # ``mode`` may also be ``"min_max"``.
        #
        # Missing norm_stats is logged at INFO (not WARNING) because the most
        # common setup is "client pre-normalizes state and post-normalizes
        # actions using dataset stats" — that's a supported mode, not an error.
        # A malformed entry is a user bug and would surface through
        # ``_build_norm_buffers`` returning ``None`` too, but the logged
        # message below still captures it.
        self._state_norm = _build_norm_buffers(getattr(config, "norm_stats", None), "state")
        self._action_norm = _build_norm_buffers(getattr(config, "norm_stats", None), "action")
        if self._state_norm is None:
            logger.info(
                "π0: no state normalization stats on config.norm_stats — "
                "state values will pass through unchanged."
            )
        if self._action_norm is None:
            logger.info(
                "π0: no action normalization stats on config.norm_stats — "
                "returned actions are in the model's normalized space."
            )

    # ── State / action normalization ─────────────────────────────────

    def _normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        return _apply_norm(state, self._state_norm, inverse=False)

    def _unnormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return _apply_norm(actions, self._action_norm, inverse=True)

    # ── SGLang multimodal input padding ──────────────────────────────

    def pad_input_ids(
        self, input_ids: List[int], mm_inputs: MultimodalInputs
    ) -> List[int]:
        """Replace the image placeholder token runs in ``input_ids`` with the
        per-item hash values from ``mm_inputs`` so RadixAttention can use
        them as cache keys.

        Processor layout: ``[image_token × 256] × num_cameras + [lang_tokens]``.
        """
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    # ── Prefix embedding ─────────────────────────────────────────────

    def embed_prefix(
        self,
        images: List[torch.Tensor],
        image_masks: List[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the prefix embeddings, per-token padding mask, and AR mask.

        Prefix tokens form a contiguous sequence
        ``[img_cam_0..., img_cam_1..., ..., lang_tokens...]`` with
        bidirectional attention; the returned ``att_masks`` are all zeros
        because each token is free to attend to every other prefix token (the
        suffix pass will put a causal boundary right before the state token).

        Ref: openpi PI0Pytorch.embed_prefix
        """
        embs: List[torch.Tensor] = []
        pad_masks: List[torch.Tensor] = []
        att_masks: List[int] = []

        for img, img_mask in zip(images, image_masks):
            img_emb = self.paligemma_with_expert.embed_image(img)
            bsize, num_img_embs = img_emb.shape[:2]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
        # Scale by sqrt(hidden_dim) — PaliGemma convention applied only to the
        # language portion (the image slice already went through the projector).
        lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])

        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        att_masks += [0] * lang_emb.shape[1]

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=embs.device)
        att_masks = att_masks[None, :].expand(pad_masks.shape[0], -1)

        return embs, pad_masks, att_masks

    # ── Suffix embedding ─────────────────────────────────────────────

    def embed_suffix(
        self,
        state: torch.Tensor,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the suffix embeddings + masks: ``[state_token, action_tokens×H]``.

        AR mask layout: ``[1, 1, 0, 0, ..., 0]`` — the state token is a causal
        boundary (no later token attends backwards through it onto prefix
        *by mistake*), the first action token starts a new causal block, and
        the rest of the action tokens attend to each other bidirectionally.

        Ref: openpi PI0Pytorch.embed_suffix
        """
        model_dtype = self.state_proj.weight.dtype
        state = state.to(dtype=model_dtype)
        noisy_actions = noisy_actions.to(dtype=model_dtype)
        device = state.device

        # State → (B, 1, W)
        state_emb = self.state_proj(state)[:, None, :]
        bsize = state_emb.shape[0]

        # Sinusoidal timestep → (B, W) → expand across the horizon.
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features,
            min_period=4e-3, max_period=4.0, device=device,
        ).to(dtype=model_dtype)

        # Fuse action and time via (2W → W → W) MLP with SiLU.
        action_emb = self.action_in_proj(noisy_actions)  # (B, H, W)
        action_time_emb = torch.cat(
            [action_emb, time_emb[:, None, :].expand_as(action_emb)], dim=2
        )
        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        embs = torch.cat([state_emb, action_time_emb], dim=1)
        pad_masks = torch.ones(
            bsize, embs.shape[1], dtype=torch.bool, device=device
        )
        att_masks = torch.tensor(
            [1] + [1] + [0] * (self.action_horizon - 1),
            dtype=embs.dtype, device=device,
        )[None, :].expand(bsize, -1)
        return embs, pad_masks, att_masks

    # ── Denoising step ───────────────────────────────────────────────

    def denoise_step(
        self,
        state: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        past_key_values,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Apply one flow-matching denoising step: predict ``v_t`` from ``x_t``.

        Uses the prefix KV cache from ``sample_actions`` and only runs the
        action expert. Ref: openpi PI0Pytorch.denoise_step.
        """
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
            state, x_t, timestep
        )

        # Build the full (B, suffix_len, prefix_len + suffix_len) boolean mask:
        #   * suffix queries can see every *valid* prefix key (padded camera
        #     slots and padded language tokens are blocked).
        #   * within the suffix, the state token is a causal boundary; action
        #     tokens attend to each other bidirectionally.
        batch_size = prefix_pad_masks.shape[0]
        suffix_len = suffix_pad_masks.shape[1]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
            batch_size, suffix_len, prefix_len
        )
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        # Position IDs continue from where the prefix's last valid token left off.
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        full_att_2d_masks_4d = prepare_attention_masks_4d(full_att_2d_masks)

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
        )

        # Drop the state token, keep only the action tokens, project down.
        suffix_out = outputs_embeds[1][:, -self.action_horizon :]
        suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype)
        return self.action_out_proj(suffix_out)

    # ── Full action generation ───────────────────────────────────────

    @torch.no_grad()
    def sample_actions(
        self,
        images: List[torch.Tensor],
        image_masks: List[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate an action chunk via iterative flow-matching denoising.

        Convention: ``t=1`` is noise, ``t=0`` is the target — opposite of the
        published π0 paper but matches both OpenPI and LeRobot.
        Ref: openpi PI0Pytorch.sample_actions
        """
        if num_steps is None:
            num_steps = self.num_inference_steps

        bsize = state.shape[0]
        device = state.device
        if noise is None:
            noise = torch.randn(
                bsize, self.action_horizon, self.action_dim,
                dtype=torch.float32, device=device,
            )

        # 1. Prefix embeddings + mask building.
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, image_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = prepare_attention_masks_4d(prefix_att_2d_masks)

        # 2. Forward prefix through PaliGemma LM, producing a list[(k, v)] cache.
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        # 3. Euler-integrated denoising from t=1 down to t=0.
        dt = -1.0 / num_steps
        x_t = noise
        for step in range(num_steps):
            t = 1.0 + step * dt
            time_tensor = torch.full(
                (bsize,), t, dtype=torch.float32, device=device
            )
            v_t = self.denoise_step(
                state=state,
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=x_t,
                timestep=time_tensor,
            )
            x_t = x_t + dt * v_t
        return x_t

    # ── Input unpacking from SGLang's mm_inputs ──────────────────────

    def _extract_request_inputs(
        self,
        mm_input,
        device: torch.device,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[int],
    ]:
        """Pull pixel_values / image_masks / lang_tokens / lang_masks / state
        out of a single-request ``MultimodalInputs``.

        All returned tensors have a leading batch dim of 1.
        """
        if mm_input is None or not mm_input.mm_items:
            raise ValueError("π0 requires multimodal input (images + state).")

        mm_item = mm_input.mm_items[0]
        model_data = mm_item.model_specific_data or {}

        pixel_values = mm_item.feature  # (num_cameras, C, H, W)
        if not isinstance(pixel_values, torch.Tensor):
            pixel_values = torch.as_tensor(pixel_values)
        pixel_values = pixel_values.to(device=device)

        image_masks = model_data.get("image_masks")
        if image_masks is None:
            image_masks = torch.ones(pixel_values.shape[0], dtype=torch.bool)
        image_masks = torch.as_tensor(image_masks, dtype=torch.bool, device=device)

        lang_tokens = model_data.get("lang_tokens")
        if lang_tokens is None:
            raise ValueError("Processor must provide lang_tokens in model_specific_data.")
        lang_tokens = torch.as_tensor(lang_tokens, dtype=torch.long, device=device)
        if lang_tokens.dim() == 1:
            lang_tokens = lang_tokens[None, :]

        lang_masks = model_data.get("lang_attention_mask")
        if lang_masks is None:
            lang_masks = torch.ones(lang_tokens.shape, dtype=torch.bool)
        lang_masks = torch.as_tensor(lang_masks, dtype=torch.bool, device=device)
        if lang_masks.dim() == 1:
            lang_masks = lang_masks[None, :]

        # Pad / truncate state to ``max_state_dim`` — matches LeRobot.
        state_data = model_data.get("state")
        if state_data is not None:
            state = torch.as_tensor(state_data, dtype=torch.float32, device=device)
            if state.dim() == 1:
                state = state[None, :]
        else:
            state = torch.zeros(
                1, self.max_state_dim, dtype=torch.float32, device=device
            )
        if state.shape[-1] < self.max_state_dim:
            state = F.pad(state, (0, self.max_state_dim - state.shape[-1]))
        elif state.shape[-1] > self.max_state_dim:
            state = state[..., : self.max_state_dim]

        num_steps = model_data.get("num_inference_steps")
        return pixel_values, image_masks, lang_tokens, lang_masks, state, num_steps

    # ── SGLang forward interface ─────────────────────────────────────

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """SGLang-compatible forward.

        π0 is a flow-matching model, not autoregressive: on ``extend`` (prefill)
        each request runs a full denoising loop and returns its action chunk
        in place of logits; on ``decode`` we return zeros because π0 never
        produces new tokens.
        """
        if forward_batch.forward_mode.is_extend():
            device = input_ids.device
            all_actions = []

            for i in range(forward_batch.batch_size):
                mm_input = forward_batch.mm_inputs[i] if forward_batch.mm_inputs else None

                if mm_input is None or not getattr(mm_input, "mm_items", None):
                    # Warmup / CUDA-graph capture can legitimately reach here
                    # with no mm_input; emit zeros so capture doesn't crash and
                    # warn once so real text-only traffic to a VLA endpoint is
                    # still visible.
                    print_warning_once(
                        "π0: extend forward got no mm_input. Expected during "
                        "warmup/CUDA-graph capture; if seen repeatedly on real "
                        "traffic, the client is sending a non-multimodal "
                        "request to a VLA model."
                    )
                    all_actions.append(torch.zeros(
                        1, self.action_horizon, self.action_dim,
                        dtype=torch.float32, device=device,
                    ))
                    continue

                (
                    pixel_values, image_masks, lang_tokens, lang_masks,
                    state, num_steps,
                ) = self._extract_request_inputs(mm_input, device)

                # ``embed_prefix`` expects a per-camera list of (1, C, H, W)
                # tensors and matching scalar masks.
                images_list = [
                    pixel_values[c : c + 1] for c in range(pixel_values.shape[0])
                ]
                image_masks_list = [
                    image_masks[c : c + 1] for c in range(image_masks.shape[0])
                ]

                state = self._normalize_state(state)
                x_t = self.sample_actions(
                    images=images_list,
                    image_masks=image_masks_list,
                    lang_tokens=lang_tokens,
                    lang_masks=lang_masks,
                    state=state,
                    num_steps=num_steps,
                )
                x_t = self._unnormalize_actions(x_t)
                all_actions.append(x_t)

            if all_actions:
                actions = torch.cat(all_actions, dim=0)
                actions_flat = actions.reshape(actions.shape[0], -1)
            else:
                actions_flat = torch.zeros(
                    1, self.action_horizon * self.action_dim, device=device
                )
            return LogitsProcessorOutput(next_token_logits=actions_flat)

        if forward_batch.forward_mode.is_decode():
            dummy = torch.zeros(
                forward_batch.batch_size,
                self.action_horizon * self.action_dim,
                device=input_ids.device,
            )
            return LogitsProcessorOutput(next_token_logits=dummy)

        raise ValueError(f"Unsupported forward mode: {forward_batch.forward_mode}")

    # ── Weight loading ───────────────────────────────────────────────

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights from an ``lerobot/pi0_base`` safetensors checkpoint.

        Checkpoint keys (after stripping the leading ``model.`` prefix) look
        like ``paligemma_with_expert.paligemma.<sub>.*``. Modern transformers
        nests those sub-modules one level deeper (``paligemma.model.<sub>``),
        and PaliGemma ties ``lm_head.weight`` with ``embed_tokens.weight`` at
        ``post_init`` time. Two remap rules make the load lossless:

          1. ``paligemma.{vision_tower,multi_modal_projector,language_model}.*``
             → ``paligemma.model.{...}.*``
          2. ``paligemma.lm_head.weight``
             → ``paligemma.model.language_model.embed_tokens.weight``
             (the checkpoint does not store ``embed_tokens.weight`` on its
             own — only the tied ``lm_head.weight`` copy — and
             ``PaliGemmaForConditionalGeneration`` does not register
             ``lm_head.weight`` as a Parameter when tied, so without this
             rule the language embedding would silently remain at its
             random init.)

        On any remaining mismatch the loader logs a warning listing a sample
        of the unmatched checkpoint keys and model params; downstream parity
        tests surface these as a hard failure.
        """
        params_dict = dict(self.named_parameters())
        buffers_dict = dict(self.named_buffers())

        _PALIGEMMA_SUBMODULES = (
            "vision_tower",
            "multi_modal_projector",
            "language_model",
        )

        def _remap(name: str) -> str:
            # Strip the leading "model." that LeRobot's PI0Policy wrapper adds.
            if name.startswith("model."):
                name = name[len("model."):]

            # Historical MLP alias used by some early π0 checkpoints.
            if name.startswith("time_mlp_in."):
                name = "action_time_mlp_in." + name[len("time_mlp_in."):]
            elif name.startswith("time_mlp_out."):
                name = "action_time_mlp_out." + name[len("time_mlp_out."):]

            # Nested PaliGemma layout.
            for sub in _PALIGEMMA_SUBMODULES:
                flat = f"paligemma_with_expert.paligemma.{sub}."
                nested = f"paligemma_with_expert.paligemma.model.{sub}."
                if name.startswith(flat) and not name.startswith(nested):
                    return nested + name[len(flat):]

            # Tied lm_head → embed_tokens redirect (see docstring).
            if name == "paligemma_with_expert.paligemma.lm_head.weight":
                return "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"

            return name

        loaded = 0
        skipped: List[str] = []
        filled_params: set = set()
        for name, loaded_weight in weights:
            mapped = _remap(name)
            if mapped in params_dict:
                param = params_dict[mapped]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded += 1
                filled_params.add(mapped)
            elif mapped in buffers_dict:
                buffers_dict[mapped].copy_(loaded_weight)
                loaded += 1
                filled_params.add(mapped)
            else:
                skipped.append(mapped)

        # Reverse audit: any model param that got no checkpoint tensor at all
        # would be running with random init. ``rotary_emb.inv_freq`` and
        # friends are config-derived buffers, not stored in the checkpoint.
        missing_params: List[str] = []
        for pname in params_dict:
            if pname in filled_params:
                continue
            if "rotary_emb" in pname or pname.endswith(".inv_freq"):
                continue
            missing_params.append(pname)

        if missing_params or skipped:
            parts: List[str] = []
            if skipped:
                parts.append(
                    f"{len(skipped)} checkpoint key(s) had no home in the "
                    f"model (first 5: {skipped[:5]})"
                )
            if missing_params:
                parts.append(
                    f"{len(missing_params)} model param(s) received NO weight "
                    f"(first 5: {missing_params[:5]})"
                )
            logger.warning(
                "π0 load_weights: %d tensors loaded — %s.",
                loaded, "; ".join(parts),
            )
        else:
            logger.info(
                "π0 load_weights: %d tensors loaded, 0 skipped, 0 missing.",
                loaded,
            )


# ──────────────────────────────────────────────────────────────────────
# SGLang model registry entry
# ──────────────────────────────────────────────────────────────────────
EntryClass = Pi0ForActionPrediction
