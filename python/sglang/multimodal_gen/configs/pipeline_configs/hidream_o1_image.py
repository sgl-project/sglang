# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from sglang.multimodal_gen.configs.models.dits.base import DiTConfig
from sglang.multimodal_gen.configs.models.dits.hidream_o1_image import (
    HiDreamO1ImageDitConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ImagePipelineConfig,
    ModelTaskType,
)
from sglang.multimodal_gen.configs.sample.hidream_o1_image import _resolve_variant

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
    """HiDream-O1-Image pipeline config for all three checkpoints.

    One pipeline, three variants; the config carries the recipe differences.
    All variants run the same generation stage, so every variant supports T2I,
    TI2I editing, prompt enhancement, in-stage batching, and cache-dit. The
    full checkpoint runs the reference 50-step FlowUniPC recipe with CFG 5.0
    in bf16; the Dev checkpoints are guidance-distilled and run the fixed
    28-step flash schedule in fp32 weights.
    """

    task_type: ModelTaskType = ModelTaskType.T2I

    dit_config: DiTConfig = field(default_factory=HiDreamO1ImageDitConfig)
    dit_precision: str = "bf16"

    # No VAE and no separate text encoder: the backbone embeds the prompt and
    # the "latents" are raw 32x32 pixel patches.
    vae_tiling: bool = False
    vae_sp: bool = False

    # The DiT takes no distilled-guidance embedding; classifier-free guidance
    # is the only guidance mechanism. The Dev checkpoints are distilled to
    # guidance 0.0, so no CFG branch runs there at all.
    should_use_guidance: bool = False

    # Noise is drawn on the host so a seed reproduces the same image on any
    # accelerator, matching the reference implementation.
    generator_device: str = "cpu"

    # Sentinel resolved in __post_init__ (50 FULL / fixed 28 DEV); serve
    # --num-inference-steps default, explicit request wins.
    num_inference_steps: Optional[int] = None
    guidance_scale: float = 5.0
    shift: float = 3.0

    def __post_init__(self):
        variant = _resolve_variant(self.model_path)
        if self.num_inference_steps is None:
            self.num_inference_steps = 50 if variant == "full" else 28
        # TI2I editing is a capability of every variant (the vision tower is
        # always built); this is a model capability, not a user knob.
        self.supported_task_types = (ModelTaskType.T2I, ModelTaskType.TI2I)
        if variant != "full":
            # Dev weights are held in fp32 (variant dtype policy, mirrored in
            # the sampling defaults) and forwarded under bf16 autocast.
            self.dit_precision = "fp32"
            self.should_use_guidance = True

    def validate_server_args(self, server_args: Any) -> None:
        # One generation stage serves every variant: token-id conditioning and
        # the single-checkpoint loader constraints apply to all of them.
        if server_args.sp_degree > 1:
            raise ValueError(
                "HiDream-O1-Image does not support sequence parallelism; run "
                "with --ulysses-degree 1 --ring-degree 1 and use --tp-size for "
                "multi-GPU instead."
            )
        if server_args.enable_breakable_cuda_graph:
            # The stage conditions on token ids rather than text embeddings,
            # which the breakable CUDA graph prompt padding does not support.
            raise ValueError(
                "HiDream-O1-Image conditions on token ids rather than text "
                "embeddings, which the breakable CUDA graph prompt padding does "
                "not support yet. Drop --enable-breakable-cuda-graph."
            )
        # The pipeline builds the model straight from the base checkpoint with
        # no quant config, so these would otherwise be silently ignored.
        # --enable-svdquant is covered: it requires --transformer-weights-path.
        unsupported_overrides = [
            f"--{option.replace('_', '-')}"
            for option in _UNSUPPORTED_WEIGHT_OVERRIDES
            if getattr(server_args, option)
        ]
        if unsupported_overrides:
            raise ValueError(
                "HiDream-O1-Image loads only its released checkpoint and does "
                "not support weight or quantization overrides yet. Drop "
                f"{', '.join(unsupported_overrides)}."
            )
        super().validate_server_args(server_args)

    def supports_dynamic_batching(self):
        # The generation stage batches merged requests itself.
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
