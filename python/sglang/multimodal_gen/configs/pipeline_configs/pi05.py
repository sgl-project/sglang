# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.model_deployment_config import (
    ModelDeploymentConfig,
)


@dataclass
class Pi05PipelineConfig(PipelineConfig):
    """Configuration for OpenPI / LeRobot Pi0.5 action policies."""

    task_type: ModelTaskType = ModelTaskType.VLA_ACTION
    should_use_guidance: bool = False
    enable_autocast: bool = True
    generator_device: str | None = None

    # OpenPI pi0.5 public checkpoint layout.
    pi05: bool = True
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    max_token_len: int = 200
    action_horizon: int = 50
    action_dim: int = 32
    state_dim: int = 32
    output_action_dim: int = 32
    n_action_steps: int = 50
    default_num_inference_steps: int = 10
    time_embedding_min_period: float = 4e-3
    time_embedding_max_period: float = 4.0
    tokenizer_name: str = "google/paligemma-3b-pt-224"

    image_keys: tuple[str, ...] = (
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
    )
    empty_cameras: int = 0
    image_size: tuple[int, int] = (224, 224)
    image_normalization_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    image_normalization_std: tuple[float, float, float] = (0.5, 0.5, 0.5)

    enable_global_prefix_cache: bool = True
    enable_action_cuda_graph: bool = True
    prefix_cache_max_entries: int = 1
    prefix_cache_layout_version: str = "pi05-prefix-v1"
    offload_prefix_image_encoder: bool = False
    offload_prefix_image_encoder_after_embed: bool = False
    offload_prefix_token_embedding: bool = False
    offload_prefix_language_layers: bool = False
    offload_prefix_language_layers_after_prefix: bool = False
    offload_prefix_language_layer_count_after_prefix: int = 0
    offload_prefix_language_layers_empty_cache: bool = True
    offload_action_expert_after_denoise: bool = False
    empty_cache_after_prefix: bool = False

    # Prefix VLM and action expert are separate logical groups. The concrete
    # process-group construction lands with the native model parallel kernels.
    prefix_parallel_strategy: str = "tp"
    action_parallel_strategy: str = "sp"
    parallel_layout_version: str = "pi05-split-prefix-action-v1"

    skip_unused_lm_head: bool = True
    materialize_dtype: str = "bf16"
    loader_component_map: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "vision_tower": ("paligemma_with_expert.paligemma.model.vision_tower.",),
            "paligemma": ("paligemma_with_expert.paligemma.model.language_model.",),
            "multi_modal_projector": (
                "paligemma_with_expert.paligemma.model.multi_modal_projector.",
            ),
            "action_expert": ("paligemma_with_expert.gemma_expert.",),
            "action_heads": (
                "action_in_proj.",
                "action_out_proj.",
                "time_mlp_in.",
                "time_mlp_out.",
            ),
        }
    )

    def supports_dynamic_batching(self):
        # Request observations have large in-memory image/state payloads and
        # exact-prefix cache semantics. Keep batching explicit until grouped
        # robot streams are validated.
        return False

    def estimate_request_cost(self, batch) -> float:
        return float(
            self.action_horizon * self.action_dim * self.default_num_inference_steps
        )

    def get_model_deployment_config(self) -> ModelDeploymentConfig:
        return ModelDeploymentConfig()
