# Copyright 2026 SGLang Team
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
"""HuggingFace-compatible config for NVIDIA GR00T-N1.7.

Mirrors the GR00T-N1.7-3B config.json.  Every field is declared
explicitly so checkpoint-loading paths (model + processor) can rely on them.

sglang's `ModelConfig._derive_model_shapes` reads
`hf_text_config.num_attention_heads` / `.hidden_size`; its
`get_hf_text_config` helper looks for `config.text_config` on multimodal
configs.  The GR00T robot config is flat (no text_config / vision_config)
so we overlay the Qwen3-VL backbone (Cosmos-Reason2-2B) sub-configs onto
this config during `__init__`
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from transformers import CONFIG_MAPPING, AutoConfig, PretrainedConfig

logger = logging.getLogger(__name__)


_BACKBONE_ATTRS_TO_OVERLAY: Tuple[str, ...] = (
    # Sub-configs that `get_hf_text_config` / multimodal processors read.
    "text_config",
    "vision_config",
    # VLM top-level fields sglang and HF access directly.
    "vocab_size",
    "max_position_embeddings",
    "rope_scaling",
    "rope_theta",
    "tie_word_embeddings",
    "image_token_id",
    "video_token_id",
    "vision_start_token_id",
    "vision_end_token_id",
    "bos_token_id",
    "eos_token_id",
    "pad_token_id",
    "attention_dropout",
)


def _load_backbone_config(model_name: str) -> Optional[PretrainedConfig]:
    if not model_name:
        return None
    try:
        return AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    except Exception as exc:  # pragma: no cover — logged, not fatal
        logger.warning(
            "Gr00tN1d7Config: failed to load backbone config from %s: %s",
            model_name,
            exc,
        )
        return None


class Gr00tN1d7Config(PretrainedConfig):
    model_type = "Gr00tN1d7"

    def __init__(
        self,
        # Backbone (Qwen3-VL / Cosmos-Reason2-2B)
        model_name: str = "nvidia/Cosmos-Reason2-2B",
        backbone_model_type: str = "qwen",
        backbone_embedding_dim: int = 2048,
        select_layer: int = 16,
        reproject_vision: bool = False,
        use_flash_attention: bool = True,
        load_bf16: bool = True,
        tune_llm: bool = True,
        tune_visual: bool = True,
        tune_top_llm_layers: int = 0,
        backbone_trainable_params_fp32: bool = False,
        # Action head dims
        hidden_size: int = 1024,
        input_embedding_dim: int = 1536,
        max_action_dim: int = 132,
        max_state_dim: int = 132,
        action_horizon: int = 40,
        max_num_embodiments: int = 32,
        state_history_length: int = 1,
        max_seq_len: int = 1024,
        add_pos_embed: bool = True,
        use_vlln: bool = True,
        # DiT sub-config
        use_alternate_vl_dit: bool = True,
        attend_text_every_n_blocks: int = 2,
        diffusion_model_cfg: Optional[Dict[str, Any]] = None,
        vl_self_attention_cfg: Optional[Dict[str, Any]] = None,
        use_vl_self_attention: bool = True,
        # Flow matching
        num_inference_timesteps: int = 4,
        num_timestep_buckets: int = 1000,
        noise_beta_alpha: float = 1.5,
        noise_beta_beta: float = 1.0,
        noise_s: float = 0.999,
        # Training-only (stored but unused at inference)
        attn_dropout: float = 0.2,
        state_dropout_prob: float = 0.2,
        state_gaussian_noise_std: float = 0.0,
        tune_diffusion_model: bool = True,
        tune_projector: bool = True,
        tune_vlln: bool = True,
        tune_linear: bool = True,
        # Misc (image/processor — kept so full config roundtrips cleanly)
        image_target_size: Tuple[int, int] = (256, 256),
        image_crop_size: Tuple[int, int] = (230, 230),
        shortest_image_edge: int = 256,
        crop_fraction: float = 0.95,
        color_jitter_params: Optional[Dict[str, float]] = None,
        use_albumentations: bool = True,
        formalize_language: bool = True,
        # Extras observed in the GR00T-N1.7-3B config.json (training
        # knobs — stored so to_dict roundtrips without warnings).
        apply_sincos_state_encoding: bool = False,
        exclude_state: bool = False,
        letter_box_transform: bool = False,
        random_history_crop: bool = True,
        random_rotation_angle: int = 0,
        rtc_ramp_rate: float = 6.0,
        use_future_tokens: bool = False,
        use_mean_std: bool = False,
        use_percentiles: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.backbone_model_type = backbone_model_type
        self.backbone_embedding_dim = backbone_embedding_dim
        self.select_layer = select_layer
        self.reproject_vision = reproject_vision
        self.use_flash_attention = use_flash_attention
        self.load_bf16 = load_bf16
        self.tune_llm = tune_llm
        self.tune_visual = tune_visual
        self.tune_top_llm_layers = tune_top_llm_layers
        self.backbone_trainable_params_fp32 = backbone_trainable_params_fp32

        self.hidden_size = hidden_size
        self.input_embedding_dim = input_embedding_dim
        self.max_action_dim = max_action_dim
        self.max_state_dim = max_state_dim
        self.action_horizon = action_horizon
        self.max_num_embodiments = max_num_embodiments
        self.state_history_length = state_history_length
        self.max_seq_len = max_seq_len
        self.add_pos_embed = add_pos_embed
        self.use_vlln = use_vlln

        self.use_alternate_vl_dit = use_alternate_vl_dit
        self.attend_text_every_n_blocks = attend_text_every_n_blocks
        self.diffusion_model_cfg = diffusion_model_cfg or {
            "num_layers": 32,
            "num_attention_heads": 32,
            "attention_head_dim": 48,
            "output_dim": 1024,
            "norm_type": "ada_norm",
            "interleave_self_attention": True,
            "final_dropout": True,
            "dropout": 0.2,
            "positional_embeddings": None,
        }
        self.vl_self_attention_cfg = vl_self_attention_cfg or {
            "num_layers": 4,
            "num_attention_heads": 32,
            "attention_head_dim": 64,
            "dropout": 0.2,
            "final_dropout": True,
            "positional_embeddings": None,
        }
        self.use_vl_self_attention = use_vl_self_attention

        self.num_inference_timesteps = num_inference_timesteps
        self.num_timestep_buckets = num_timestep_buckets
        self.noise_beta_alpha = noise_beta_alpha
        self.noise_beta_beta = noise_beta_beta
        self.noise_s = noise_s

        self.attn_dropout = attn_dropout
        self.state_dropout_prob = state_dropout_prob
        self.state_gaussian_noise_std = state_gaussian_noise_std
        self.tune_diffusion_model = tune_diffusion_model
        self.tune_projector = tune_projector
        self.tune_vlln = tune_vlln
        self.tune_linear = tune_linear

        self.image_target_size = list(image_target_size)
        self.image_crop_size = list(image_crop_size)
        self.shortest_image_edge = shortest_image_edge
        self.crop_fraction = crop_fraction
        self.color_jitter_params = color_jitter_params or {
            "brightness": 0.3,
            "contrast": 0.4,
            "saturation": 0.5,
            "hue": 0.08,
        }
        self.use_albumentations = use_albumentations
        self.formalize_language = formalize_language

        self.apply_sincos_state_encoding = apply_sincos_state_encoding
        self.exclude_state = exclude_state
        self.letter_box_transform = letter_box_transform
        self.random_history_crop = random_history_crop
        self.random_rotation_angle = random_rotation_angle
        self.rtc_ramp_rate = rtc_ramp_rate
        self.use_future_tokens = use_future_tokens
        self.use_mean_std = use_mean_std
        self.use_percentiles = use_percentiles

        # Overlay Qwen3-VL backbone sub-configs so sglang's
        # `get_hf_text_config` / `_derive_model_shapes` path resolves
        # text_config.num_attention_heads / .hidden_size etc. from
        # Cosmos-Reason2-2B rather than hitting this flat robot config.
        self._overlay_backbone(model_name, kwargs)

    def _overlay_backbone(self, model_name: str, init_kwargs: Dict[str, Any]) -> None:
        # If the backbone sub-configs were already materialised on the
        # incoming kwargs (e.g. from a prior serialization roundtrip),
        # leave them untouched.
        if (
            getattr(self, "text_config", None) is not None
            and getattr(self, "vision_config", None) is not None
        ):
            return
        backbone = _load_backbone_config(model_name)
        if backbone is None:
            return
        for attr in _BACKBONE_ATTRS_TO_OVERLAY:
            value = getattr(backbone, attr, None)
            if value is None:
                continue
            # Never clobber a GR00T field that a prior __init__ step or
            # incoming kwargs explicitly set.
            if attr in init_kwargs:
                continue
            if getattr(self, attr, None) not in (None, [], {}):
                continue
            setattr(self, attr, value)


# Register with transformers CONFIG_MAPPING so AutoConfig.from_pretrained()
# resolves `model_type = "Gr00tN1d7"` to `Gr00tN1d7Config` automatically.
try:
    CONFIG_MAPPING.register("Gr00tN1d7", Gr00tN1d7Config)
except Exception:
    # Already registered (e.g. module imported twice) — fall back to direct
    # assignment as lfm2_moe does.
    CONFIG_MAPPING._extra_content["Gr00tN1d7"] = Gr00tN1d7Config
