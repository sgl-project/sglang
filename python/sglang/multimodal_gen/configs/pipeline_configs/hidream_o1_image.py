# SPDX-License-Identifier: Apache-2.0
import math
from dataclasses import dataclass, field
from typing import Any, ClassVar

import torch

from sglang.multimodal_gen.configs.models.dits.base import DiTConfig
from sglang.multimodal_gen.configs.models.dits.hidream_o1_image import (
    HiDreamO1ImageDitConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.base import ImagePipelineConfig
from sglang.multimodal_gen.configs.sample.hidream_o1_image import _resolve_variant
from sglang.multimodal_gen.configs.task_type import ModelTaskType

# Key under which the before-denoising stage stashes the per-branch backbone
# inputs, since HiDream-O1 conditions on token ids rather than text embeddings.
HIDREAM_O1_COND_KEY = "hidream_o1_pos_cond"
HIDREAM_O1_NEG_COND_KEY = "hidream_o1_neg_cond"
# ServerArgs fields the pipeline's own loader does not read.
_UNSUPPORTED_WEIGHT_OVERRIDES = (
    "transformer_weights_path",
    "quantization",
    "quantization_ignored_layers",
    "component_paths",
    "component_weights_paths",
    "component_quantizations",
)


@dataclass
class HiDreamO1ImagePipelineConfig(ImagePipelineConfig):
    task_type: ModelTaskType = ModelTaskType.T2I
    supported_task_types: ClassVar[tuple[ModelTaskType, ...]] = (
        ModelTaskType.T2I,
        ModelTaskType.TI2I,
    )

    dit_config: DiTConfig = field(default_factory=HiDreamO1ImageDitConfig)
    dit_precision: str = "bf16"

    # No VAE and no separate text encoder: the backbone embeds the prompt and
    # the "latents" are raw 32x32 pixel patches.
    vae_tiling: bool = False
    vae_sp: bool = False

    # The DiT takes no distilled-guidance embedding; classifier-free guidance
    # is the only guidance mechanism.
    should_use_guidance: bool = False

    # Noise is drawn on the host so a seed reproduces the same image on any
    # accelerator, matching the reference implementation.
    generator_device: str = "cpu"

    def update_config_from_dict(self, args, prefix: str = "") -> None:
        super().update_config_from_dict(args, prefix)
        # model_path is only populated here in the serving path; Dev checkpoints
        # are guidance-distilled and must run the DiT in fp32.
        if _resolve_variant(self.model_path) != "full":
            self.dit_precision = "fp32"

    def validate_server_args(self, server_args: Any) -> None:
        # The pipeline builds its scheduler with the reference flow shift, so
        # these would otherwise be silently ignored.
        scheduler_overrides = [
            f"--{option.replace('_', '-')}"
            for option in ("flow_shift", "scheduler_class_override")
            if getattr(self, option) is not None
        ]
        if scheduler_overrides:
            raise ValueError(
                "HiDream-O1-Image samples with the reference scheduler and flow "
                f"shift. Drop {', '.join(scheduler_overrides)}."
            )
        if server_args.sp_degree > 1:
            raise ValueError(
                "HiDream-O1-Image mixes causal text spans with a bidirectional "
                "image span, which requires a dense [B, 1, S, S] attention mask "
                "that cannot be sharded along the sequence. Run with "
                "--ulysses-degree 1 --ring-degree 1 and use --tp-size or "
                "--cfg-parallel-size for multi-GPU instead."
            )
        if server_args.enable_breakable_cuda_graph:
            # The denoising stage's prompt_embeds are token ids, which the BCG
            # text-bucket padder would treat as length-1 embeddings and pad the
            # [B, 1, S, S] mask into a shape the transformer rejects.
            raise ValueError(
                "HiDream-O1-Image conditions on token ids rather than text "
                "embeddings, which the breakable CUDA graph prompt padding does "
                "not support yet. Drop --enable-breakable-cuda-graph."
            )
        # The pipeline builds the transformer straight from the base checkpoint
        # with no quant config, so these would otherwise be silently ignored.
        # --enable-svdquant is covered: it requires --transformer-weights-path.
        unsupported_overrides = [
            f"--{option.replace('_', '-')}"
            for option in _UNSUPPORTED_WEIGHT_OVERRIDES
            if getattr(server_args, option)
        ]
        if unsupported_overrides:
            raise ValueError(
                "HiDream-O1-Image loads only its released bfloat16 checkpoint and "
                "does not support weight or quantization overrides yet. Drop "
                f"{', '.join(unsupported_overrides)}."
            )
        super().validate_server_args(server_args)

    def supports_dynamic_batching(self):
        return True

    def estimate_request_cost(self, batch) -> float:
        # HiDream tokenizes raw pixels into 32x32 patches and has no VAE.
        patch_size = self.dit_config.arch_config.patch_size
        image_tokens = math.ceil(int(batch.width) / patch_size) * math.ceil(
            int(batch.height) / patch_size
        )
        num_frames = max(1, int(batch.num_frames or 1))
        num_outputs = max(1, int(batch.num_outputs_per_prompt or 1))
        return float(image_tokens * num_frames * num_outputs)

    def supports_batching_image_conditioning(self):
        return True

    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        del device, rotary_emb, dtype
        return batch.extra[HIDREAM_O1_COND_KEY]

    def prepare_neg_cond_kwargs(self, batch, device, rotary_emb, dtype):
        del device, rotary_emb, dtype
        return batch.extra[HIDREAM_O1_NEG_COND_KEY]

    def post_denoising_loop(self, latents: torch.Tensor, batch) -> torch.Tensor:
        # Patch tokens are never padded, so the base unpad would be a no-op that
        # only risks truncating against a stale raw_latent_shape.
        del batch
        return latents

    def calculate_condition_image_size(self, image, width, height):
        # HiDream has no VAE, so there is no latent-aligned multiple to resize
        # the reference to. The before-denoising stage sizes the reference
        # itself (aspect-preserving), so returning None skips the VAE-based
        # size adjustment in InputValidationStage, which would otherwise hit a
        # VAE-less scale factor.
        del image, width, height
        return None


def register():
    from sglang.multimodal_gen.configs.sample.hidream_o1_image import (
        HiDreamO1ImageSamplingParams,
    )
    from sglang.multimodal_gen.registry import register_configs

    register_configs(
        sampling_param_cls=HiDreamO1ImageSamplingParams,
        pipeline_config_cls=HiDreamO1ImagePipelineConfig,
        hf_model_paths=[
            "HiDream-ai/HiDream-O1-Image",
            "HiDream-ai/HiDream-O1-Image-Dev",
            "HiDream-ai/HiDream-O1-Image-Dev-2604",
        ],
        model_detectors=[lambda hf_id: "hidream-o1-image" in hf_id.lower()],
    )
