# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Multimodal processor for NVIDIA GR00T-N1.7.

Extends the Qwen3-VL image processor with GR00T-specific extras:
- The backbone HF config (Cosmos-Reason2-2B) is substituted for the GR00T
  robot config when calling the super init, because Qwen3-VL requires
  vision_start_token_id / image_token_id / video_token_id which the GR00T
  config doesn't carry.
- Image target size is pinned to 256x256 per processor_config.json.
- `proprio_state` (dict of joint->list[float]) is flattened per the
  embodiment's modality order and right-padded to `max_state_dim=132`,
  then broadcast to `(state_history_length, max_state_dim)` and stashed on
  `result.proprio_states`.
- `embodiment` (string tag) is mapped to an integer `embodiment_id` via
  EMBODIMENT_TAG_TO_PROJECTOR_INDEX (ported from Isaac-GR00T's
  processing_gr00t_n1d7.py lines 56-72).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Union

import torch

from sglang.srt.models.groot_n1d7 import Gr00tN1d7
from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor

logger = logging.getLogger(__name__)


# Ported from Isaac-GR00T/gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py
# lines 56-72.  Maps embodiment tag -> projector (embodiment MLP) index.
EMBODIMENT_TAG_TO_PROJECTOR_INDEX: Dict[str, int] = {
    "oxe_droid_relative_eef_relative_joint": 24,
    "xdof_relative_eef_relative_joint": 27,
    "xdof_relative_eef_relative_joint_subtask": 27,
    "real_g1_relative_eef_relative_joints": 25,
    "real_r1_pro_sharpa_relative_eef": 26,
    "real_r1_pro_sharpa_relative_eef_human": 26,
    "real_r1_pro_sharpa_relative_eef_maxinsights": 26,
    "real_r1_pro_sharpa_relative_eef_mecka": 26,
    "unitree_g1_full_body_with_waist_height_nav_cmd": 25,
    "simpler_env_google": 0,
    "simpler_env_widowx": 1,
    "libero_sim": 2,
    "new_embodiment": 10,
}


# Fallback modality order when `hf_config.modality_configs` isn't available.
# Matches the g1 family — the test embodiment in the plan.
_FALLBACK_G1_STATE_KEYS: List[str] = [
    "left_wrist_eef_9d",
    "right_wrist_eef_9d",
    "left_hand",
    "right_hand",
    "left_arm",
    "right_arm",
    "waist",
]

# Per-embodiment state-key fallbacks used when `hf_config.modality_configs` is
# not available.  Sourced from Isaac-GR00T's experiment_cfg modality.json for
# each embodiment.
_FALLBACK_STATE_KEYS: Dict[str, List[str]] = {
    "real_g1_relative_eef_relative_joints": list(_FALLBACK_G1_STATE_KEYS),
    "unitree_g1_full_body_with_waist_height_nav_cmd": list(_FALLBACK_G1_STATE_KEYS),
    # DROID: eef_9d (pose as xyz+rot6d) + gripper_position + joint_position.
    # Order matches Isaac's ModalityConfig(modality_keys=["eef_9d",
    # "gripper_position", "joint_position"]) for OXE_DROID_RELATIVE_EEF_*.
    "oxe_droid_relative_eef_relative_joint": [
        "eef_9d",
        "gripper_position",
        "joint_position",
    ],
}


def state_keys_for(
    embodiment: str,
    modality_configs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[str]:
    """Return the ordered list of proprio-state modality keys for an
    embodiment, preferring `modality_configs[embodiment]["state"]["modality_keys"]`
    when the HF config surfaces one, else the per-embodiment fallback
    table, else the g1 ordering (original behaviour).
    """
    if modality_configs and embodiment in modality_configs:
        cfg = modality_configs[embodiment]
        state_cfg = cfg.get("state") if isinstance(cfg, dict) else None
        if isinstance(state_cfg, dict) and "modality_keys" in state_cfg:
            return list(state_cfg["modality_keys"])
    return list(_FALLBACK_STATE_KEYS.get(embodiment, _FALLBACK_G1_STATE_KEYS))


def build_proprio_state(
    proprio: Dict[str, Sequence[float]],
    embodiment: str,
    max_state_dim: int,
    state_history_length: int,
    modality_configs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> torch.Tensor:
    """Flatten a proprio-state dict into a `(state_history_length, max_state_dim)`
    float32 tensor per the embodiment's modality ordering, right-padding with
    zeros.

    Raises ValueError if any expected key is absent or if the concatenated
    length overruns `max_state_dim`.
    """
    keys = state_keys_for(embodiment, modality_configs)
    flat: List[float] = []
    for key in keys:
        vals = proprio.get(key)
        if vals is None:
            raise ValueError(
                f"proprio_state missing key {key!r} for embodiment {embodiment!r}"
            )
        flat.extend(float(v) for v in vals)
    if len(flat) > max_state_dim:
        raise ValueError(
            f"proprio state has {len(flat)} dims, exceeds max_state_dim={max_state_dim}"
        )
    flat.extend([0.0] * (max_state_dim - len(flat)))
    state = torch.tensor(flat, dtype=torch.float32).reshape(1, max_state_dim)
    return state.expand(state_history_length, max_state_dim).clone()


def _resolve_backbone_hf_config(groot_cfg):
    """Return the Qwen3-VL (Cosmos-Reason2-2B) HF config so the base Qwen-VL
    processor finds vision_start_token_id / image_token_id / ... — the GR00T
    robot config doesn't carry those.
    """
    from transformers import AutoConfig

    model_name = getattr(groot_cfg, "model_name", None) or "nvidia/Cosmos-Reason2-2B"
    return AutoConfig.from_pretrained(model_name, trust_remote_code=True)


class Gr00tN1d7Processor(QwenVLImageProcessor):
    models = [Gr00tN1d7]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        backbone_cfg = _resolve_backbone_hf_config(hf_config)
        super().__init__(backbone_cfg, server_args, _processor, *args, **kwargs)
        # Pin the Qwen2VLImageProcessor to emit a 256x256 (= 65536 total
        # pixels) image, matching GR00T's processor_config.json.  Qwen2VL
        # parametrises size in total-pixel terms, not height/width.
        if hasattr(_processor, "image_processor"):
            target_pixels = 256 * 256
            try:
                _processor.image_processor.size = {
                    "shortest_edge": target_pixels,
                    "longest_edge": target_pixels,
                }
                _processor.image_processor.min_pixels = target_pixels
                _processor.image_processor.max_pixels = target_pixels
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "Gr00tN1d7Processor: failed to pin image size to 256x256: %s",
                    exc,
                )
        self.groot_config = hf_config
        self.modality_configs = getattr(hf_config, "modality_configs", None)
        self.max_state_dim = hf_config.max_state_dim
        self.state_history_length = hf_config.state_history_length

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        result = await super().process_mm_data_async(
            image_data, input_text, request_obj, *args, **kwargs
        )

        # GR00T reuses the shared VLA `history_traj` channel (same field
        # alpamayo uses) — robot payload lives under two dict keys:
        #   history_traj["proprio_state"]: dict[joint_name -> list[float]]
        #   history_traj["embodiment"]:    str tag, e.g. "real_g1_..."
        history_traj = getattr(request_obj, "history_traj", None) or {}
        proprio = history_traj.get("proprio_state")
        embodiment = history_traj.get("embodiment")
        if proprio is None or embodiment is None:
            return result

        state = build_proprio_state(
            proprio,
            embodiment,
            max_state_dim=self.max_state_dim,
            state_history_length=self.state_history_length,
            modality_configs=self.modality_configs,
        )
        embodiment_id = EMBODIMENT_TAG_TO_PROJECTOR_INDEX.get(embodiment)
        if embodiment_id is None:
            raise ValueError(
                f"Unknown embodiment tag {embodiment!r}; expected one of "
                f"{sorted(EMBODIMENT_TAG_TO_PROJECTOR_INDEX)}"
            )

        # Stash the processed tensor + id back onto request_obj.history_traj
        # so downstream (ForwardBatch.history_trajs[i]) gets the model-ready
        # form without re-parsing.  Also mirror onto result for multimodal
        # processors that forward the per-request mm_inputs separately.
        history_traj["proprio_state_tensor"] = state
        history_traj["embodiment_id"] = int(embodiment_id)
        try:
            request_obj.history_traj = history_traj
        except (AttributeError, TypeError):
            # Read-only request_obj (e.g. pydantic frozen model) — fall back
            # to mm_inputs-side attributes.
            pass

        result.proprio_states = [state]
        result.embodiment_ids = [int(embodiment_id)]
        return result
