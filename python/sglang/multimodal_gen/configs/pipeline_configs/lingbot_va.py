# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.vaes.wanvae import WanVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import PipelineConfig
from sglang.multimodal_gen.configs.task_type import ModelTaskType


@dataclass
class LingBotVALiberoConfig(PipelineConfig):
    task_type: ModelTaskType = ModelTaskType.VLA_ACTION
    image_keys: tuple[str, ...] = ("agentview_rgb", "eye_in_hand_rgb")
    image_size: tuple[int, int] = (128, 128)
    wrist_half_resolution: bool = False
    latent_shape: tuple[int, int] = (8, 16)
    video_steps: int = 20
    action_shift: float = 0.05
    frame_chunk_size: int = 4
    attention_window: int = 30
    action_per_frame: int = 4
    channels: tuple[int, ...] = tuple(range(7))
    low: tuple[float, ...] = (
        -0.6589285731315613,
        -0.84375,
        -0.9375,
        -0.12107142806053162,
        -0.15964286029338837,
        -0.26571428775787354,
        -1.0,
    )
    high: tuple[float, ...] = (
        0.8999999761581421,
        0.8544642925262451,
        0.9375,
        0.17142857611179352,
        0.1842857152223587,
        0.34392857551574707,
        1.0,
    )
    output_action_dim: int = 7
    action_dim: int = 7
    action_horizon: int = 16
    default_num_inference_steps: int = 50
    vae_config: WanVAEConfig = field(default_factory=WanVAEConfig)
    vae_precision: str = "bf16"
    supports_action_prefix: bool = True

    supports_independent_action_sessions: bool = True

    def supports_dynamic_batching(self):
        return True

    def supports_native_grouped_requests(self):
        return True

    def estimate_request_cost(self, batch):
        height, width = self.latent_shape
        return 2 * (
            self.frame_chunk_size * height * width // 4 * (self.video_steps + 1)
            + self.action_horizon * batch.num_inference_steps
        )

    def validate_server_args(self, server_args):
        super().validate_server_args(server_args)
        if server_args.num_gpus != 1:
            raise ValueError("LingBot-VA currently requires one GPU per replica")

    def action_metadata(self, server_args):
        return {
            "object": "action.metadata",
            "model": server_args.served_model_name,
            "policy_family": "lingbot-va",
            "input": {
                "image_keys": list(self.image_keys),
                "image_size": list(self.image_size),
            },
            "output": {
                "action_dim": self.action_dim,
                "action_horizon": self.action_horizon,
                "first_action_horizon": self.action_horizon - self.action_per_frame,
                "dtype": "float32",
            },
            "defaults": {"num_inference_steps": 50, "video_steps": self.video_steps},
            "capabilities": {
                "action_prefix": True,
                "cross_request_batching": True,
                "realtime_websocket": True,
            },
        }


@dataclass
class LingBotVARoboTwinConfig(LingBotVALiberoConfig):
    image_keys: tuple[str, ...] = ("cam_high", "cam_left_wrist", "cam_right_wrist")
    image_size: tuple[int, int] = (256, 320)
    wrist_half_resolution: bool = True
    latent_shape: tuple[int, int] = (24, 20)
    video_steps: int = 25
    action_shift: float = 1.0
    action_per_frame: int = 16
    frame_chunk_size: int = 2
    attention_window: int = 72
    action_horizon: int = 32
    action_dim: int = 16
    output_action_dim: int = 16
    channels: tuple[int, ...] = (*range(7), 28, *range(7, 14), 29)
    # In wire order: left xyz, quaternion, grip, then the right arm.
    low: tuple[float, ...] = (
        -0.06172713458538055,
        -3.6716461181640625e-05,
        -0.08783501386642456,
        -1,
        -1,
        -1,
        -1,
        0,
        -0.3547105032205582,
        -1.3113021850585938e-06,
        -0.11975435614585876,
        -1,
        -1,
        -1,
        -1,
        0,
    )
    high: tuple[float, ...] = (
        0.3462600058317184,
        0.39966784834861746,
        0.14745532035827624,
        1,
        1,
        1,
        1,
        1,
        0.034201726913452024,
        0.39142737388610793,
        0.1792279863357542,
        1,
        1,
        1,
        1,
        1,
    )


def register():
    from sglang.multimodal_gen.configs.sample.lingbot_va import LingBotVASamplingParams
    from sglang.multimodal_gen.registry import register_configs

    register_configs(
        sampling_param_cls=LingBotVASamplingParams,
        pipeline_config_cls=LingBotVALiberoConfig,
        hf_model_paths=["robbyant/lingbot-va-posttrain-libero-long"],
        model_detectors=[lambda name: name.lower() == "lingbotvaliberopipeline"],
    )

    register_configs(
        sampling_param_cls=LingBotVASamplingParams,
        pipeline_config_cls=LingBotVARoboTwinConfig,
        hf_model_paths=["robbyant/lingbot-va-posttrain-robotwin"],
        model_detectors=[lambda name: name.lower() == "lingbotvarobotwinpipeline"],
    )
