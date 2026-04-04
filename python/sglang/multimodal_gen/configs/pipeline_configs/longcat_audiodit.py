# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.longcat_audiodit import (
    LongCatAudioDiTDitConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
)


def _optional_to(value, device, *args, **kwargs):
    return None if value is None else value.to(device, *args, **kwargs)


@dataclass
class LongCatAudioDiTPipelineConfig(PipelineConfig):
    """LongCat-AudioDiT text-to-speech / voice cloning.

    Denoising lives in ``LongCatAudioDiTDenoisingStage`` (prompt-region
    rewrite, uncond zeroing, per-request CFG/APG on ``batch.extra["cfg_policy"]``).
    Cond kwargs / prompt embeds / ``post_denoising_loop`` stay here as the
    PipelineConfig hooks that stage already calls.
    """

    task_type: ModelTaskType = ModelTaskType.T2A
    dit_config: LongCatAudioDiTDitConfig = field(
        default_factory=LongCatAudioDiTDitConfig
    )

    # Checkpoint defaults; also read by DenoisingStage autocast.
    dit_precision: str = "bf16"
    vae_precision: str = "fp16"

    scheduler_class_override: str = "AudioDiTFlowMatchScheduler"

    # Embedded guidance tensor only (Flux-style). CFG/APG still runs via
    # ``batch.do_classifier_free_guidance``.
    should_use_guidance: bool = False
    enable_autocast: bool = False

    def supports_dynamic_batching(self):
        # Prompt-region lengths and APG state are per-request scalars; a merged
        # prompt list cannot be consumed. Base already excludes T2A; keep this
        # explicit so a later base-class change cannot opt LongCat in.
        return False

    def supports_disaggregation(self) -> bool:
        # Single-GPU 1D pipeline with no PD contract.
        return False

    def validate_server_args(self, server_args) -> None:
        ulysses = server_args.ulysses_degree or 1
        ring = server_args.ring_degree or 1
        if ulysses > 1 or ring > 1:
            raise ValueError(
                "LongCat-AudioDiT is single-GPU only: 1D sequence parallelism "
                "is not implemented. Leave --ulysses-degree and --ring-degree unset."
            )

    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return {
            "text_len": batch._audio_text_condition_len.to(device),
            "mask": _optional_to(batch._audio_mask, device),
            "cond_mask": _optional_to(batch._audio_cond_mask, device),
            "latent_cond": batch._audio_latent_cond.to(device, dtype),
            "return_ith_layer": int(batch._audio_repa_dit_layer),
        }

    def prepare_neg_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return {
            "text_len": batch._audio_text_condition_len.to(device),
            "mask": _optional_to(batch._audio_mask, device),
            "cond_mask": _optional_to(batch._audio_cond_mask, device),
            "latent_cond": batch._audio_empty_latent_cond.to(device, dtype),
            "return_ith_layer": int(batch._audio_repa_dit_layer),
        }

    def get_pos_prompt_embeds(self, batch):
        return batch.prompt_embeds[0]

    def get_neg_prompt_embeds(self, batch):
        return batch.negative_prompt_embeds[0]

    def post_denoising_loop(self, latents, batch):
        latent_len = batch._audio_prompt_latent_len
        if latent_len > 0:
            latents = latents[:, latent_len:]
        return latents
