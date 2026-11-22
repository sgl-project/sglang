# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Inference configuration for LeRobot-format LingBot-VLA 2.0 checkpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass
class LingBotVLAV2Config:
    """Checkpoint fields used for inference; unsupported training or runtime features are rejected."""

    chunk_size: int = 50
    n_action_steps: int = 50
    max_state_dim: int = 55
    max_action_dim: int = 55
    num_steps: int = 10
    tokenizer_path: str = "Qwen/Qwen3-VL-4B-Instruct"
    tokenizer_revision: str = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
    tokenizer_max_length: int = 72
    resize_imgs_with_padding: tuple[int, int] = (224, 224)
    image_max_pixels: int = 262144
    image_min_pixels: int = 131072
    input_features: dict[str, Any] = field(default_factory=dict)
    output_features: dict[str, Any] = field(default_factory=dict)
    state_slots: dict[str, Any] | None = None
    action_slots: dict[str, Any] | None = None
    canonical_joints: dict[str, int] = field(
        default_factory=lambda: {
            "arm.position": 14,
            "end.position": 14,
            "effector.position": 2,
            "hand.position": 12,
            "waist.position": 4,
            "head.position": 2,
            "base.velocity": 3,
            "reserved.slots": 4,
        }
    )
    normalization_mapping: dict[str, str] = field(
        default_factory=lambda: {
            "VISUAL": "IDENTITY",
            "STATE": "MEAN_STD",
            "ACTION": "MEAN_STD",
        }
    )
    inputs_normalized: bool = False
    image_key_map: dict[str, str] = field(default_factory=dict)
    policy_server_config: dict[str, Any] = field(default_factory=dict)
    expert_hidden_size: int = 768
    expert_intermediate_size: int = 2752
    action_num_attention_heads: int = 32
    action_num_key_value_heads: int = 8
    action_head_dim: int = 128
    action_fp32: bool = False
    use_moe: bool = True
    token_moe_layers: list[int] = field(default_factory=lambda: list(range(36)))
    token_num_experts: int = 32
    token_top_k: int = 4
    token_moe_intermediate_size: int = 512
    token_shared_intermediate_size: int = 704
    router_activation: str = "sigmoid"
    routed_scaling_factor: float = 4.0
    use_shared_expert_gate: bool = False
    attention_implementation: str = "sdpa"
    vit_attn_implementation: str = "sdpa"
    adanorm_time: bool = True
    final_norm_adanorm: bool = False
    vlm_causal: bool = True
    vocab_size: int = 0
    use_lm_head: bool = False
    use_cache: bool = True
    qwen3vl_use_vision_boundaries: bool = True
    align_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.normalization_mapping.get("VISUAL") != "IDENTITY":
            raise ValueError(
                "LingBot requires its checkpoint Qwen image preprocessing without dataset image normalization"
            )
        if (
            self.max_state_dim != sum(self.canonical_joints.values())
            or self.max_action_dim != self.max_state_dim
        ):
            raise ValueError(
                "Canonical joint widths must match state and action widths"
            )
        for name in (
            "chunk_size",
            "n_action_steps",
            "num_steps",
            "tokenizer_max_length",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.final_norm_adanorm and not self.adanorm_time:
            raise ValueError("final_norm_adanorm requires adanorm_time")
        if self.n_action_steps != self.chunk_size:
            raise ValueError(
                "The LingBot endpoint returns a full chunk; n_action_steps must equal chunk_size"
            )
        if not self.use_cache or self.use_lm_head:
            raise ValueError(
                "LingBot inference requires prefix KV caching, no LM head, and dynamic grid processing"
            )
        if (
            self.attention_implementation not in ("sdpa", "eager")
            or self.vit_attn_implementation != "sdpa"
        ):
            raise ValueError(
                "Supported attention: sdpa/eager for the policy, sdpa for vision"
            )
        if self.router_activation not in ("sigmoid", "softmax"):
            raise ValueError("router_activation must be sigmoid or softmax")
        if not 1 <= self.token_top_k <= self.token_num_experts:
            raise ValueError("Invalid MoE top-k")
        if len(self.camera_keys) not in (1, 2, 3):
            raise ValueError("LingBot requires one to three configured cameras")
        # note (0xtoward): boundary tokens keep adjacent cameras in separate MRoPE grids.
        if not self.qwen3vl_use_vision_boundaries:
            raise ValueError("LingBot inference requires qwen3vl_use_vision_boundaries")
        if self.image_min_pixels < 1 or self.image_max_pixels < self.image_min_pixels:
            raise ValueError("Invalid image pixel budget")
        self.slot_indices("observation.state")
        self.slot_indices("action")

    @property
    def camera_keys(self) -> list[str]:
        return [k for k, v in self.input_features.items() if v.get("type") == "VISUAL"]

    @property
    def state_dim(self) -> int:
        return self.input_features["observation.state"]["shape"][-1]

    @property
    def action_dim(self) -> int:
        return self.output_features["action"]["shape"][-1]

    def slot_indices(self, key: str) -> tuple[list[int], list[int]]:
        slots = self.state_slots if key == "observation.state" else self.action_slots
        width = self.state_dim if key == "observation.state" else self.action_dim
        if slots is None:
            if width != self.max_action_dim:
                raise ValueError(
                    f"{key}: a noncanonical robot requires an explicit slot mapping"
                )
            return list(range(width)), list(range(width))
        source, target = [], []
        offsets, offset = {}, 0
        for name, size in self.canonical_joints.items():
            offsets[name] = offset
            offset += size
        for name, mapping in slots.items():
            name = name.removeprefix(f"{key}.")
            if name not in offsets:
                raise ValueError(f"Unknown canonical slot: {name}")
            cursor = offsets[name]
            for origin in mapping["origin_keys"]:
                if set(origin) != {key}:
                    raise ValueError(f"Slot origin must be {key}")
                start, end = origin[key]["start"], origin[key]["end"]
                if not 0 <= start < end <= width:
                    raise ValueError(f"Invalid {key} span {start}:{end}")
                source.extend(range(start, end))
                target.extend(range(cursor, cursor + end - start))
                cursor += end - start
            if cursor > offsets[name] + self.canonical_joints[name]:
                raise ValueError(f"Canonical slot overflow: {name}")
        if sorted(source) != list(range(width)) or len(set(target)) != len(target):
            raise ValueError(f"{key}: each raw dimension must map exactly once")
        return source, target

    @classmethod
    def from_dict(cls, raw: dict, overrides: dict | None = None):
        raw = dict(raw)
        if raw.get("type", "lingbot_vla_v2") != "lingbot_vla_v2":
            raise ValueError("Expected a LeRobot-format lingbot_vla_v2 checkpoint")
        for key in (
            "use_relative_actions",
            "use_peft",
            "use_visual_memory",
            "use_proprioceptive_memory",
            "compile_predict_velocity",
            "use_cudagraph_denoise",
            "use_cudagraph_prefix",
            "use_cudagraph_prefix_full",
            "precompute_grid_thw",
        ):
            if raw.get(key):
                raise ValueError(f"Unsupported LingBot inference capability: {key}")
        if raw.get("rtc_config") is not None or raw.get("n_obs_steps", 1) != 1:
            raise ValueError("RTC and observation history are not supported")
        if raw.get("dtype", "float32") != "float32":
            raise ValueError("The initial LingBot inference lane requires float32")
        if raw.get("moe_implementation", "fused") != "fused":
            raise ValueError("Only the fused checkpoint expert layout is supported")
        known = {f.name for f in fields(cls)}
        values = {k: v for k, v in raw.items() if k in known}
        overrides = dict(overrides or {})
        if "num_inference_steps" in overrides:
            overrides["num_steps"] = overrides.pop("num_inference_steps")
        unknown = set(overrides) - known
        if unknown:
            raise ValueError(f"Unknown LingBot runtime overrides: {sorted(unknown)}")
        values.update(overrides)
        return cls(**values)

    @classmethod
    def from_pretrained(cls, path: str | Path, overrides: dict | None = None):
        return cls.from_dict(
            json.loads((Path(path) / "config.json").read_text()), overrides
        )
