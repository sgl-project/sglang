# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Any

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

    enable_global_prefix_cache: bool = False
    enable_prefix_cuda_graph: bool = True
    # Bucket prompt tokens so common serving lengths reuse the same prefix and
    # action CUDA graphs. Opt in explicitly because padding can change floating
    # point reduction order even though padded tokens remain attention-masked.
    prompt_token_buckets: tuple[int, ...] = ()
    prefix_cuda_graph_max_entries: int = 1
    enable_action_cuda_graph: bool = True
    action_cuda_graph_max_entries: int = 16
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

    def __post_init__(self) -> None:
        self._normalize_and_validate_cuda_graph_config()

    def _normalize_and_validate_cuda_graph_config(self) -> None:
        try:
            buckets = tuple(int(bucket) for bucket in self.prompt_token_buckets)
        except (TypeError, ValueError) as exc:
            raise ValueError("prompt_token_buckets must contain integers") from exc
        if any(bucket <= 0 for bucket in buckets):
            raise ValueError("prompt_token_buckets must contain positive lengths")
        if tuple(sorted(set(buckets))) != buckets:
            raise ValueError(
                "prompt_token_buckets must be strictly increasing and unique"
            )
        if buckets and buckets[-1] > self.max_token_len:
            raise ValueError(
                "prompt_token_buckets cannot exceed max_token_len "
                f"({self.max_token_len}), got {buckets[-1]}"
            )
        if self.prefix_cuda_graph_max_entries < 0:
            raise ValueError("prefix_cuda_graph_max_entries must be non-negative")
        if self.action_cuda_graph_max_entries < 0:
            raise ValueError("action_cuda_graph_max_entries must be non-negative")
        self.prompt_token_buckets = buckets

    def check_pipeline_config(self) -> None:
        super().check_pipeline_config()
        # JSON/CLI overrides are applied after dataclass construction.
        self._normalize_and_validate_cuda_graph_config()

    def update_pipeline_config(self, source_pipeline_dict: dict[str, Any]) -> None:
        # PipelineConfig treats tuples of ModelConfig specially. ``all`` is
        # vacuously true for our default empty bucket tuple, so handle this
        # variable-length scalar tuple outside the generic updater.
        source_pipeline_dict = dict(source_pipeline_dict)
        missing = object()
        buckets = source_pipeline_dict.pop("prompt_token_buckets", missing)
        super().update_pipeline_config(source_pipeline_dict)
        if buckets is not missing:
            self.prompt_token_buckets = buckets
            self._normalize_and_validate_cuda_graph_config()

    def supports_dynamic_batching(self):
        return True

    def supports_native_grouped_requests(self):
        return True

    def supports_openpi_endpoint(self) -> bool:
        return True

    def estimate_request_cost(self, batch) -> float:
        return float(
            self.action_horizon * self.action_dim * self.default_num_inference_steps
        )

    def get_model_deployment_config(self) -> ModelDeploymentConfig:
        return ModelDeploymentConfig()
