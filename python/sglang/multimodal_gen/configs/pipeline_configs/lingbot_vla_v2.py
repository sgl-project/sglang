# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sglang.multimodal_gen.configs.models.lingbot_vla_v2 import LingBotVLAV2Config
from sglang.multimodal_gen.configs.pipeline_configs.base import PipelineConfig
from sglang.multimodal_gen.configs.task_type import ModelTaskType
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
    hf_hub_download,
    load_dict,
)

CHECKPOINT_REVISIONS = {
    "lerobot/lingbot_vla_v2_robotwin": "54f79a4b6e9166bda91b5010088097ff3191bca2",
    "lerobot/lingbot_vla_v2_base": "5dd4b5299ed85c782a987d58ff3f24f95a95474d",
}


def is_lingbot_vla_v2_checkpoint(model_path: str) -> bool:
    if model_path.lower() in CHECKPOINT_REVISIONS:
        return True
    try:
        config = json.loads((Path(model_path) / "config.json").read_text())
    except (OSError, ValueError):
        return False
    return isinstance(config, dict) and config.get("type") == "lingbot_vla_v2"


@dataclass
class LingBotVLAV2PipelineConfig(PipelineConfig):
    task_type: ModelTaskType = ModelTaskType.VLA_ACTION
    tokenizer_path: str | None = None
    inputs_normalized: bool | None = None
    enable_action_cuda_graph: bool = False
    policy: LingBotVLAV2Config | None = field(init=False, default=None, repr=False)

    def validate_server_args(self, server_args: Any) -> None:
        super().validate_server_args(server_args)
        if server_args.num_gpus != 1 or server_args.enable_cfg_parallel:
            raise ValueError("LingBot-VLA 2.0 requires one CUDA GPU")
        if server_args.quantization:
            raise ValueError("LingBot-VLA 2.0 requires FP32 weights")
        if any(
            (
                server_args.dit_cpu_offload,
                server_args.text_encoder_cpu_offload,
                server_args.image_encoder_cpu_offload,
                server_args.dit_layerwise_offload,
            )
        ):
            raise ValueError("LingBot-VLA 2.0 requires resident model weights")

        model_path = server_args.model_path
        config_path = Path(model_path) / "config.json"
        if not Path(model_path).is_dir():
            if server_args.revision is None:
                server_args.revision = CHECKPOINT_REVISIONS.get(model_path.lower())
            config_path = Path(
                hf_hub_download(
                    model_path, "config.json", revision=server_args.revision
                )
            )
        overrides = {}
        if self.tokenizer_path is not None:
            overrides["tokenizer_path"] = self.tokenizer_path
        if self.inputs_normalized is not None:
            overrides["inputs_normalized"] = self.inputs_normalized
        self.policy = LingBotVLAV2Config.from_dict(
            load_dict(str(config_path)), overrides
        )

    @property
    def image_keys(self) -> tuple[str, ...]:
        return tuple(
            key.removeprefix("observation.images.") for key in self.policy.camera_keys
        )

    @property
    def action_horizon(self) -> int:
        return self.policy.chunk_size

    @property
    def action_dim(self) -> int:
        return self.policy.max_action_dim

    @property
    def output_action_dim(self) -> int:
        return self.policy.action_dim

    @property
    def default_num_inference_steps(self) -> int:
        return self.policy.num_steps

    def supports_openpi_endpoint(self) -> bool:
        return True

    def action_metadata(self, server_args: Any) -> dict[str, Any]:
        return {
            "object": "action.metadata",
            "model": server_args.served_model_name,
            "model_path": server_args.model_path,
            "policy_family": "lingbot_vla_v2",
            "input": {
                "image_keys": list(self.image_keys),
                "image_size": list(self.policy.resize_imgs_with_padding),
                "state_dim": self.policy.state_dim,
                "state_normalized": self.policy.inputs_normalized,
            },
            "output": {
                "action_type": "continuous",
                "action_horizon": self.action_horizon,
                "action_dim": self.output_action_dim,
                "padded_action_dim": self.action_dim,
                "dtype": "float32",
                "normalized": self.policy.inputs_normalized,
            },
            "defaults": {"num_inference_steps": self.default_num_inference_steps},
            "capabilities": {
                "openpi_websocket": True,
                "cuda_graph": self.enable_action_cuda_graph,
                "realtime_websocket": True,
                "batch_inputs": False,
                "multiple_candidates": False,
            },
        }


def register():
    from sglang.multimodal_gen.configs.sample.lingbot_vla_v2 import (
        LingBotVLAV2SamplingParams,
    )
    from sglang.multimodal_gen.registry import register_configs

    register_configs(
        sampling_param_cls=LingBotVLAV2SamplingParams,
        pipeline_config_cls=LingBotVLAV2PipelineConfig,
        hf_model_paths=list(CHECKPOINT_REVISIONS),
        model_detectors=[is_lingbot_vla_v2_checkpoint],
    )
