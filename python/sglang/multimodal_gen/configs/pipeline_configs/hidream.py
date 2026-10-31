from dataclasses import dataclass
from typing import ClassVar, Optional

from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
)


@dataclass
class HiDreamPipelineConfig(PipelineConfig):
    """HiDream-O1-Image unified VLM diffusion pipeline (Qwen3VL backbone;
    no separate text encoder, VAE, or scheduler modules)."""

    # requests with a reference image auto-resolve to TI2I
    task_type: ModelTaskType = ModelTaskType.T2I
    supported_task_types: ClassVar[tuple[ModelTaskType, ...]] = (
        ModelTaskType.T2I,
        ModelTaskType.TI2I,
    )
    model_precision: str = "bf16"
    should_use_guidance: bool = True
    supports_cfg_parallel: bool = False

    # sentinel resolved in __post_init__ (28 DEV / 50 FULL); serve
    # --num-inference-steps default, explicit request wins
    num_inference_steps: Optional[int] = None
    guidance_scale: float = 5.0
    shift: float = 3.0

    def __post_init__(self):
        from sglang.multimodal_gen.configs.sample.hidream import _is_full_model

        if self.num_inference_steps is None:
            self.num_inference_steps = 50 if _is_full_model(self.model_path) else 28

    def supports_dynamic_batching(self):
        return True

    # no VAE: resolution handling lives in the generation stage
    def calculate_condition_image_size(self, image, width, height):
        return None

    def prepare_calculated_size(self, image):
        return None

    def estimate_request_cost(self, batch) -> float:
        patch_size = 32
        image_tokens = (int(batch.height) // patch_size) * (
            int(batch.width) // patch_size
        )
        cfg_branches = 2 if float(batch.guidance_scale) > 1 else 1
        return float(
            image_tokens
            * int(batch.num_inference_steps)
            * cfg_branches
            * int(batch.num_outputs_per_prompt)
        )

    def supports_disaggregation(self) -> bool:
        return False


def register():
    from sglang.multimodal_gen.configs.sample.hidream import (
        HiDreamSamplingParams,
    )
    from sglang.multimodal_gen.registry import register_configs

    register_configs(
        sampling_param_cls=HiDreamSamplingParams,
        pipeline_config_cls=HiDreamPipelineConfig,
        hf_model_paths=[
            "HiDream-ai/HiDream-O1-Image-Dev-2604",
            "HiDream-ai/HiDream-O1-Image-Dev",
            "HiDream-ai/HiDream-O1-Image",
        ],
        model_detectors=[
            lambda hf_id: "hidream" in hf_id.lower(),
        ],
    )
