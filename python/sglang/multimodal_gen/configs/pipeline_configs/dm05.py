# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
)

DM05_MODEL_IDS = (
    "Dexmal/DM05",
    "Dexmal/DM05-libero",
    "Dexmal/DM05-robotwin2",
    "Dexmal/DM05-MEM-Robodojo-Sim",
)


def is_dm05_model(model_path: str) -> bool:
    """Use local checkpoint metadata before interpreting a model's name."""
    path = Path(model_path).expanduser()
    if path.is_dir():
        try:
            config = json.loads((path / "config.json").read_text())
        except (OSError, ValueError):
            return False
        return isinstance(config, dict) and config.get("model_type") == "dm05"
    normalized = str(model_path).lower().replace("\\", "/")
    return any(normalized == name.lower() for name in DM05_MODEL_IDS)


def dm05_model_id(model_path: str) -> str:
    """Select deployment recipe defaults; arbitrary local exports use base defaults."""
    normalized = str(model_path).lower().replace("\\", "/")
    for name in reversed(DM05_MODEL_IDS):
        short_name = name.rsplit("/", 1)[-1].lower()
        if (
            short_name in Path(normalized).name
            or ("models--" + name.lower().replace("/", "--")) in normalized
        ):
            return name
    return DM05_MODEL_IDS[0]


@dataclass
class DM05PipelineConfig(PipelineConfig):
    """Native OpenDM action inference with checkpoint-owned Gemma3 components."""

    task_type: ModelTaskType = ModelTaskType.VLA_ACTION
    should_use_guidance: bool = False
    enable_autocast: bool = False
    generator_device: str | None = None
    policy_family: str = "dm05"

    action_horizon: int = 50
    action_dim: int = 32
    output_action_dim: int = 14
    state_dim: int = 14
    default_num_inference_steps: int = 10
    materialize_dtype: str = "bf16"
    precision_policy: str = "bf16_mixed"
    llm_attn_implementation: str = "sdpa"
    vision_attn_implementation: str = "sdpa"
    action_attn_implementation: str = "sdpa"

    processor_name: str | None = None
    norm_stats_path: str | None = None
    max_token_len: int = 1024
    image_keys: tuple[str, ...] = ("head", "left_wrist", "right_wrist")
    image_prompts: tuple[str, ...] = ("Head", "Left wrist", "Right wrist")
    image_size: tuple[int, int] = (448, 448)
    add_state: bool = True
    n_state_bins: int = 256
    clip_to_bounds: bool = True
    reject_prompt_truncation: bool = False
    normalize_task_prompt: bool = False
    prompt_is_formatted: bool = False
    default_robot_type: str | None = None
    default_control_mode: str | None = None
    default_speed: str = "0.5"
    state_desc: tuple[str, ...] | None = None
    use_absolute_action: bool = True
    compose_eef_rot: bool = False
    use_transformed_state: bool = False
    state_layout: str = "identity"
    denorm_state: bool = False
    full_action_mask: bool = False
    is_history: bool = False
    max_history_images: int = 5

    enable_global_prefix_cache: bool = False
    prefix_cache_max_entries: int = 1
    enable_prefix_cuda_graph: bool = False
    enable_action_cuda_graph: bool = True
    action_cuda_graph_max_entries: int = 4
    empty_cache_after_prefix: bool = False
    prefix_cache_layout_version: str = "dm05-prefix-v1"

    def __post_init__(self) -> None:
        self._validate_dm05_config()

    def _validate_dm05_config(self) -> None:
        for name in (
            "action_horizon",
            "action_dim",
            "output_action_dim",
            "state_dim",
            "default_num_inference_steps",
            "max_token_len",
            "n_state_bins",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.output_action_dim > self.action_dim:
            raise ValueError("output_action_dim cannot exceed action_dim")
        if self.n_state_bins < 2:
            raise ValueError("n_state_bins must be at least 2")
        for name in (
            "max_history_images",
            "prefix_cache_max_entries",
            "action_cuda_graph_max_entries",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not self.image_keys or len(set(self.image_keys)) != len(self.image_keys):
            raise ValueError("image_keys must contain unique camera names")
        if len(self.image_keys) != len(self.image_prompts):
            raise ValueError("image_keys and image_prompts must have equal lengths")
        if self.materialize_dtype not in ("bf16", "bfloat16", "fp32", "float32"):
            raise ValueError("materialize_dtype must be bf16 or fp32")
        if self.precision_policy not in ("bf16_mixed", "fp32_mixed"):
            raise ValueError("precision_policy must be bf16_mixed or fp32_mixed")
        if self.precision_policy == "fp32_mixed" and self.materialize_dtype not in (
            "fp32",
            "float32",
        ):
            raise ValueError("fp32_mixed requires FP32 model parameters")
        for name in (
            "llm_attn_implementation",
            "vision_attn_implementation",
            "action_attn_implementation",
        ):
            if getattr(self, name) not in ("eager", "sdpa"):
                raise ValueError(f"{name} must be eager or sdpa")
        if self.enable_prefix_cuda_graph:
            raise ValueError(
                "DM05 currently captures the action loop, not prefix prefill"
            )

    def check_pipeline_config(self) -> None:
        super().check_pipeline_config()
        self._validate_dm05_config()

    def apply_checkpoint_config(self, local_path: str, payload: dict) -> None:
        if payload.get("model_type") != "dm05":
            raise ValueError(f"{local_path} is not a DM05 checkpoint")
        if payload.get("action_dim", self.action_dim) != self.action_dim:
            raise ValueError("Configured action_dim does not match the DM05 checkpoint")
        self._validate_dm05_config()

    def validate_server_args(self, server_args: Any) -> None:
        self._validate_dm05_config()
        for name in (
            "num_gpus",
            "tp_size",
            "sp_degree",
            "ulysses_degree",
            "ring_degree",
        ):
            if getattr(server_args, name, 1) != 1:
                raise ValueError(
                    "DM05 uses one GPU per worker; launch independent replicas for multiple GPUs"
                )

    def prefix_cuda_graph_available(self) -> bool:
        return False

    def action_cuda_graph_available(self) -> bool:
        return self.enable_action_cuda_graph and self.action_cuda_graph_max_entries > 0

    def supports_dynamic_batching(self) -> bool:
        return True

    def supports_native_grouped_requests(self) -> bool:
        return True

    def supports_disaggregation(self) -> bool:
        return False

    def supports_openpi_endpoint(self) -> bool:
        return True

    def action_metadata(self, server_args: Any) -> dict[str, Any]:
        returned_action_dim = self.output_action_dim
        if self.state_layout == "robocoliseum_g2":
            returned_action_dim = 16
        elif self.state_layout == "robocoliseum_manipulation":
            returned_action_dim = 21
        return {
            "object": "action.metadata",
            "model": server_args.served_model_name,
            "model_path": server_args.model_path,
            "policy_family": self.policy_family,
            "input": {
                "image_keys": list(self.image_keys),
                "image_prompts": list(self.image_prompts),
                "image_size": list(self.image_size),
                "state_dim": self.state_dim,
                "history": self.is_history,
                "max_history_images": self.max_history_images if self.is_history else 0,
            },
            "output": {
                "action_type": "continuous",
                "action_horizon": self.action_horizon,
                "action_dim": returned_action_dim,
                "padded_action_dim": self.action_dim,
                "dtype": "float32",
            },
            "runtime": {
                "materialize_dtype": self.materialize_dtype,
                "precision_policy": self.precision_policy,
                "cuda_graph": {
                    "prefix_enabled": False,
                    "action_enabled": self.action_cuda_graph_available(),
                    "action_max_entries": self.action_cuda_graph_max_entries,
                },
            },
            "defaults": {
                "num_inference_steps": self.default_num_inference_steps,
                "prefix_cache": self.enable_global_prefix_cache,
                "cuda_graph": self.action_cuda_graph_available(),
            },
            "capabilities": {
                "exact_prefix_cache": True,
                "cuda_graph": self.action_cuda_graph_available(),
                "realtime_websocket": True,
                "openpi_websocket": True,
                "batch_inputs": False,
                "multiple_candidates": False,
                "history_images": self.is_history,
            },
        }

    def estimate_request_cost(self, batch: Any) -> float:
        return float(
            self.action_horizon
            * self.action_dim
            * (batch.num_inference_steps or self.default_num_inference_steps)
        )


@dataclass
class DM05LiberoPipelineConfig(DM05PipelineConfig):
    action_horizon: int = 10
    output_action_dim: int = 7
    state_dim: int = 8
    image_keys: tuple[str, ...] = ("head", "left_wrist")
    image_prompts: tuple[str, ...] = ("Head", "Left wrist")
    add_state: bool = False
    default_robot_type: str | None = "Franka"
    use_absolute_action: bool = False
    state_desc: tuple[str, ...] | None = ("joint",) * 6 + ("gripper",) * 2


@dataclass
class DM05Robotwin2PipelineConfig(DM05PipelineConfig):
    max_token_len: int = 1152
    default_robot_type: str | None = "Aloha RoboTwin2"
    use_absolute_action: bool = False
    clip_to_bounds: bool = False


@dataclass
class DM05MemoryPipelineConfig(DM05PipelineConfig):
    max_token_len: int = 1536
    default_robot_type: str | None = "Dual ARX5"
    is_history: bool = True
    max_history_images: int = 20
    state_desc: tuple[str, ...] | None = (("joint",) * 6 + ("gripper",)) * 2
    use_absolute_action: bool = False
    clip_to_bounds: bool = False
    materialize_dtype: str = "fp32"
    precision_policy: str = "fp32_mixed"
    normalize_task_prompt: bool = True
    prompt_is_formatted: bool = True


def register():
    from sglang.multimodal_gen.configs.sample.dm05 import DM05SamplingParams
    from sglang.multimodal_gen.registry import register_configs

    for model_id, cls in zip(
        DM05_MODEL_IDS,
        (
            DM05PipelineConfig,
            DM05LiberoPipelineConfig,
            DM05Robotwin2PipelineConfig,
            DM05MemoryPipelineConfig,
        ),
        strict=True,
    ):
        register_configs(
            sampling_param_cls=DM05SamplingParams,
            pipeline_config_cls=cls,
            hf_model_paths=[model_id],
        )
