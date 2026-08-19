# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class MiniMaxH3LatentPreparationStage(PipelineStage):
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
            minimax_h3_plan_from_batch,
        )

        plan = minimax_h3_plan_from_batch(batch)
        if plan is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} is a MiniMax H3 contract stage "
                "and has no implementation yet."
            )
        self._prepare_denoise_state_from_plan(batch, plan)
        self._publish_native_latent_state(batch)
        return batch

    def run_grouped_requests(
        self,
        batches: list[Req],
        server_args: ServerArgs,
    ) -> list[Req]:
        """Preserve H3's independent per-modality RNG streams per request."""
        return [self(batch, server_args) for batch in batches]

    @staticmethod
    def _publish_native_latent_state(batch: Req) -> None:
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
            MINIMAX_H3_DENOISE_STATE_EXTRA_KEY,
        )

        state = batch.extra.get(MINIMAX_H3_DENOISE_STATE_EXTRA_KEY)
        if not isinstance(state, dict):
            raise ValueError("MiniMax H3 denoise state must be a mapping")
        video_rows = state.get("initial_video_rows")
        audio_rows = state.get("initial_audio_rows")
        if not isinstance(video_rows, torch.Tensor) or video_rows.ndim != 2:
            raise ValueError("MiniMax H3 initial_video_rows must be a rank-2 tensor")
        if not isinstance(audio_rows, torch.Tensor) or audio_rows.ndim != 2:
            raise ValueError("MiniMax H3 initial_audio_rows must be a rank-2 tensor")

        latent_t = int(state["latent_t"])
        latent_h = int(state["latent_h"])
        latent_w = int(state["latent_w"])
        audio_t = int(state["audio_t"])
        batch.latents = video_rows
        batch.audio_latents = audio_rows
        batch.raw_latent_shape = (1, 24, latent_t, latent_h, latent_w)
        batch.raw_audio_latent_shape = (2, 32, audio_t)

    def _prepare_denoise_state_from_plan(self, batch: Req, plan) -> None:
        """Direct initial-noise materialization (t2va recipe):
        torch.Generator().manual_seed(seed); video rows drawn first,
        then audio rows, CPU fp32. Every task consumes the final latent grid
        frozen by the pre-queue shape resolver."""
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
            MINIMAX_H3_DENOISE_STATE_EXTRA_KEY,
        )

        if MINIMAX_H3_DENOISE_STATE_EXTRA_KEY in batch.extra:
            return
        shape = plan.shape
        geometry = str(shape["geometry"])
        if geometry != "resolved_v2":
            raise ValueError(
                "MiniMax H3 latent preparation requires pre-queue resolved_v2 "
                f"geometry, got {geometry!r}"
            )
        latent_h = int(shape["height"]) // 16
        latent_w = int(shape["width"]) // 16
        if shape.get("video_latent_t") is None or shape.get("audio_latent_t") is None:
            raise ValueError(
                "MiniMax H3 latent preparation requires pre-queue resolved "
                "temporal dimensions"
            )
        latent_t = int(shape["video_latent_t"])
        audio_t = int(shape["audio_latent_t"])

        seed = plan.seed
        if seed is None:
            seed = 42  # pinned default seed
        video_rows_n = latent_t * (latent_h // 2) * (latent_w // 2)
        audio_rows_n = audio_t * 2
        # Noise semantics:
        # - video noise is drawn on the RAW latent tensor
        #   [1, 24, T, H_lat, W_lat] in tensor layout, then patchified
        #   into packed row order;
        # - audio uses an INDEPENDENT generator re-seeded with the same
        #   seed (each modality re-seeds its own generator);
        # - no extra cond-frame noise is drawn for image-conditioned
        #   requests.
        # The same seed always reproduces the same noise.
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_tokens import (
            minimax_h3_patchify_video_latent,
        )

        gen_v = torch.Generator().manual_seed(int(seed))
        video_tensor = torch.randn(
            1,
            24,
            latent_t,
            latent_h,
            latent_w,
            generator=gen_v,
            dtype=torch.float32,
        )
        video_noise = minimax_h3_patchify_video_latent(
            video_tensor, patch_size=[1, 2, 2]
        ).to(torch.float32)
        gen_a = torch.Generator().manual_seed(int(seed))
        audio_noise = torch.randn(
            audio_rows_n, 32, generator=gen_a, dtype=torch.float32
        )
        if list(video_noise.shape) != [video_rows_n, 96]:
            raise ValueError(
                f"aligned video noise shape {list(video_noise.shape)} != "
                f"[{video_rows_n}, 96]"
            )
        batch.extra[MINIMAX_H3_DENOISE_STATE_EXTRA_KEY] = {
            "initial_video_rows": video_noise,
            "initial_audio_rows": audio_noise,
            "latent_t": latent_t,
            "latent_h": latent_h,
            "latent_w": latent_w,
            "audio_t": audio_t,
        }

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check(
            "prompt_or_embeds",
            None,
            lambda _: V.string_or_list_strings(batch.prompt)
            or V.list_not_empty(batch.prompt_embeds),
        )
        result.add_check("prompt_embeds", batch.prompt_embeds, V.list_of_tensors)
        result.add_check(
            "num_videos_per_prompt", batch.num_outputs_per_prompt, V.positive_int
        )
        result.add_check("generator", batch.generator, V.generator_or_list_generators)
        result.add_check("num_frames", batch.num_frames, V.positive_int)
        result.add_check("height", batch.height, V.positive_int)
        result.add_check("width", batch.width, V.positive_int)
        result.add_check("latents", batch.latents, V.none_or_tensor)
        return result

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("latents", batch.latents, [V.is_tensor, V.with_dims(2)])
        result.add_check(
            "audio_latents", batch.audio_latents, [V.is_tensor, V.with_dims(2)]
        )
        result.add_check("raw_latent_shape", batch.raw_latent_shape, V.is_tuple)
        result.add_check(
            "raw_audio_latent_shape", batch.raw_audio_latent_shape, V.is_tuple
        )
        return result


__all__ = [
    "MiniMaxH3LatentPreparationStage",
]
