# SPDX-License-Identifier: Apache-2.0

from typing import Any

import torch

from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import DenoisingStage


class ZImageDenoisingStage(DenoisingStage):
    def __init__(
        self, transformer, scheduler, pipeline=None, transformer_2=None, vae=None
    ) -> None:
        super().__init__(
            transformer,
            scheduler,
            pipeline=pipeline,
            transformer_2=transformer_2,
            vae=vae,
        )
        self._batched_cfg_conditioning: tuple[tuple, dict[str, Any]] | None = None

    def _predict_noise_with_cfg(self, **kwargs):
        batch = kwargs["batch"]
        server_args = kwargs["server_args"]
        policy = kwargs["cfg_policy"]
        latent_input = kwargs["latent_model_input"]
        if (
            len(policy.branches) != 2
            or not policy.branches[0].is_conditional
            or policy.branches[1].is_conditional
            or server_args.enable_cfg_parallel
            or server_args.enable_torch_compile
            or server_args.enable_breakable_cuda_graph
            or self._cache_dit_enabled
            or batch.enable_teacache
            or batch.enable_spectrum
            or (
                kwargs["cfg_gate_state"] is not None
                and kwargs["cfg_gate_state"]["active"]
            )
            or latent_input.shape[0] != 1
        ):
            return super()._predict_noise_with_cfg(**kwargs)

        positive, negative = policy.branches
        pos_freqs = positive.kwargs["freqs_cis"]
        neg_freqs = negative.kwargs["freqs_cis"]
        if any(
            p.shape != n.shape
            for pos_pair, neg_pair in zip(pos_freqs, neg_freqs, strict=True)
            for p, n in zip(pos_pair, neg_pair, strict=True)
        ):
            return super()._predict_noise_with_cfg(**kwargs)

        cache_key = (positive, negative, *pos_freqs, *neg_freqs)
        cached = self._batched_cfg_conditioning
        if cached is not None and all(
            a is b for a, b in zip(cached[0], cache_key, strict=True)
        ):
            conditioning = cached[1]
        else:
            conditioning = self._build_batched_cfg_conditioning(positive, negative)
            self._batched_cfg_conditioning = (cache_key, conditioning)
        positive.configure_batch(batch)
        with set_forward_context(
            current_timestep=kwargs["timestep_index"],
            attn_metadata=kwargs["attn_metadata"],
            forward_batch=batch,
            quality=batch.sampling_params.quality,
        ):
            prediction = self._predict_noise(
                current_model=kwargs["current_model"],
                latent_model_input=torch.cat([latent_input, latent_input]),
                timestep=kwargs["timestep"].reshape(-1).repeat(2),
                target_dtype=kwargs["target_dtype"],
                guidance=kwargs["guidance"],
                **conditioning,
            )
        predictions = [
            server_args.pipeline_config.slice_noise_pred(p, kwargs["latents"])
            for p in prediction.chunk(2)
        ]
        cfg_scale = server_args.pipeline_config.get_classifier_free_guidance_scale(
            batch, kwargs["current_guidance_scale"]
        )
        return policy.combine(
            predictions, batch, cfg_scale, server_args.pipeline_config
        )

    @staticmethod
    def _build_batched_cfg_conditioning(positive, negative):
        pos_freqs = positive.kwargs["freqs_cis"]
        neg_freqs = negative.kwargs["freqs_cis"]
        return {
            **positive.kwargs,
            "encoder_hidden_states": (
                positive.kwargs["encoder_hidden_states"]
                + negative.kwargs["encoder_hidden_states"]
            ),
            "caption_valid_lens": torch.cat(
                [
                    positive.kwargs["caption_valid_lens"],
                    negative.kwargs["caption_valid_lens"],
                ]
            ),
            "freqs_cis": tuple(
                tuple(
                    torch.stack([p, n]) for p, n in zip(pos_pair, neg_pair, strict=True)
                )
                for pos_pair, neg_pair in zip(pos_freqs, neg_freqs, strict=True)
            ),
        }
