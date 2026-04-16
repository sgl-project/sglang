# SPDX-License-Identifier: Apache-2.0
"""
KV cache denoising stage for FLUX.2-klein-9b-kv.

On step 0, reference image tokens are included in the forward pass and their
post-RoPE K/V projections are cached per attention layer.  On steps 1+, the
cached K/V is reused so the reference tokens need not be recomputed.

Adapts the standard ``DenoisingStage`` by overriding per-step methods rather
than duplicating the denoising loop.

KV cache modes (``kv_cache_mode`` parameter on the transformer forward):

    ``"extract"``
        Step 0 only.  Reference tokens are present in the input sequence.
        Each attention layer stores the post-RoPE K/V for the last
        ``num_ref_tokens`` positions.  Causal attention is used so that
        reference tokens only self-attend (preventing noise leakage into
        the cached representations).  Modulation parameters for reference
        positions use a fixed timestep (t=0) via blending.

    ``"cached"``
        Steps 1+.  Reference tokens are *not* in the input sequence.
        Each attention layer appends the previously cached K/V to the
        current K/V so that noise/text tokens can still attend to the
        reference information.

    ``None``
        Standard forward without KV caching (no reference images).
"""

from typing import Any

import torch
import torch.nn as nn

from sglang.multimodal_gen.runtime.models.dits.flux_2 import Flux2KVCache
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class Flux2KleinKVCacheDenoisingStage(DenoisingStage):
    """Denoising stage that caches reference image K/V on step 0.

    Reuses the parent ``DenoisingStage`` loop by overriding:

    - ``_prepare_denoising_loop``: creates KV cache, computes freqs_cis
      without ref positions for steps 1+, stores both on ``batch``.
    - ``_predict_noise_with_cfg``: injects per-step KV cache params into
      ``pos_cond_kwargs`` (and ``neg_cond_kwargs`` when CFG is enabled),
      clears ``batch.image_latent`` after step 0 to prevent re-concat.
    - ``_post_denoising_loop``: restores ``batch.image_latent``, clears cache.

    All intermediate state is stored on ``batch`` (not ``self``) so that
    aborted requests do not leak stale state.
    """

    def _prepare_denoising_loop(self, batch: Req, server_args: ServerArgs):
        prepared = super()._prepare_denoising_loop(batch, server_args)

        if batch.image_latent is not None:
            num_double = len(self.transformer.transformer_blocks)
            num_single = len(self.transformer.single_transformer_blocks)
            batch._kv_cache = Flux2KVCache(num_double, num_single)
            batch._kv_num_ref_tokens = batch.image_latent.shape[1]

            # freqs_cis layout: [txt, img, ref] — slice off ref for steps 1+
            cos, sin = prepared["pos_cond_kwargs"]["freqs_cis"]
            batch._kv_freqs_cis_without_ref = (
                cos[: -batch._kv_num_ref_tokens],
                sin[: -batch._kv_num_ref_tokens],
            )
        else:
            batch._kv_cache = None

        return prepared

    def _predict_noise_with_cfg(
        self,
        current_model: nn.Module,
        latent_model_input: torch.Tensor,
        timestep,
        batch: Req,
        timestep_index: int,
        attn_metadata,
        target_dtype,
        current_guidance_scale,
        image_kwargs: dict[str, Any],
        pos_cond_kwargs: dict[str, Any],
        neg_cond_kwargs: dict[str, Any],
        server_args,
        guidance,
        latents,
    ):
        kv_cache = getattr(batch, "_kv_cache", None)

        if kv_cache is not None:
            if timestep_index == 0:
                kv_params = {
                    "kv_cache": kv_cache,
                    "kv_cache_mode": "extract",
                    "num_ref_tokens": batch._kv_num_ref_tokens,
                    "ref_fixed_timestep": 0.0,
                }
            else:
                kv_params = {
                    "kv_cache": kv_cache,
                    "kv_cache_mode": "cached",
                    "num_ref_tokens": 0,
                    "freqs_cis": batch._kv_freqs_cis_without_ref,
                }

            pos_cond_kwargs = {**pos_cond_kwargs, **kv_params}
            if neg_cond_kwargs:
                neg_cond_kwargs = {**neg_cond_kwargs, **kv_params}

        result = super()._predict_noise_with_cfg(
            current_model=current_model,
            latent_model_input=latent_model_input,
            timestep=timestep,
            batch=batch,
            timestep_index=timestep_index,
            attn_metadata=attn_metadata,
            target_dtype=target_dtype,
            current_guidance_scale=current_guidance_scale,
            image_kwargs=image_kwargs,
            pos_cond_kwargs=pos_cond_kwargs,
            neg_cond_kwargs=neg_cond_kwargs,
            server_args=server_args,
            guidance=guidance,
            latents=latents,
        )

        # After step 0: clear image_latent so the parent loop won't concat
        # ref tokens on subsequent steps
        if kv_cache is not None and timestep_index == 0:
            batch._kv_saved_image_latent = batch.image_latent
            batch.image_latent = None

        return result

    def _post_denoising_loop(
        self,
        batch,
        latents,
        trajectory_latents,
        trajectory_timesteps,
        server_args,
        is_warmup=False,
    ):
        # Restore image_latent and clean up KV cache
        if hasattr(batch, "_kv_saved_image_latent"):
            batch.image_latent = batch._kv_saved_image_latent
            del batch._kv_saved_image_latent
        if hasattr(batch, "_kv_cache") and batch._kv_cache is not None:
            batch._kv_cache.clear()
            del batch._kv_cache
        if hasattr(batch, "_kv_num_ref_tokens"):
            del batch._kv_num_ref_tokens
        if hasattr(batch, "_kv_freqs_cis_without_ref"):
            del batch._kv_freqs_cis_without_ref

        return super()._post_denoising_loop(
            batch,
            latents,
            trajectory_latents,
            trajectory_timesteps,
            server_args,
            is_warmup,
        )
