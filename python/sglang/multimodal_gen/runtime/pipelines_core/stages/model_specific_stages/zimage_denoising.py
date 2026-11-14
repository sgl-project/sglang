# SPDX-License-Identifier: Apache-2.0

from typing import Any

import torch
from torch import nn

from sglang.multimodal_gen.configs.sample.sampling_params import quality_allows
from sglang.multimodal_gen.runtime.distributed.cfg_policy import CFGPolicy
from sglang.multimodal_gen.runtime.distributed.parallel_state import get_sp_world_size
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import DenoisingStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class ZImageDenoisingStage(DenoisingStage):
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        try:
            return super().forward(batch, server_args)
        finally:
            batch.extra.pop("zimage_batched_cfg_conditioning", None)

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
        cfg_policy: CFGPolicy,
        cfg_gate_state: dict[str, Any] | None,
        server_args: ServerArgs,
        guidance: torch.Tensor,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        if (
            not quality_allows(batch.sampling_params.quality, "lossless")
            or len(cfg_policy.branches) != 2
            or not cfg_policy.branches[0].is_conditional
            or cfg_policy.branches[1].is_conditional
            or server_args.enable_cfg_parallel
            or server_args.enable_torch_compile
            or server_args.enable_breakable_cuda_graph
            or self._cache_dit_enabled
            or batch.enable_teacache
            or batch.enable_spectrum
            or (cfg_gate_state is not None and cfg_gate_state["active"])
            or latent_model_input.shape[0] != 1
            or not self._matching_rope_shapes(cfg_policy)
        ):
            return super()._predict_noise_with_cfg(
                current_model=current_model,
                latent_model_input=latent_model_input,
                timestep=timestep,
                batch=batch,
                timestep_index=timestep_index,
                attn_metadata=attn_metadata,
                target_dtype=target_dtype,
                current_guidance_scale=current_guidance_scale,
                cfg_policy=cfg_policy,
                cfg_gate_state=cfg_gate_state,
                server_args=server_args,
                guidance=guidance,
                latents=latents,
            )

        positive, negative = cfg_policy.branches
        sp_size = get_sp_world_size()
        sources = (positive, negative, sp_size)
        cached = batch.extra.get("zimage_batched_cfg_conditioning")
        if (
            cached is None
            or cached[0][0] is not positive
            or cached[0][1] is not negative
            or cached[0][2] != sp_size
        ):
            conditioning = self._build_batched_cfg_conditioning(
                positive, negative, stack_rope=sp_size > 1
            )
            batch.extra["zimage_batched_cfg_conditioning"] = (sources, conditioning)
        else:
            conditioning = cached[1]
        positive.configure_batch(batch)
        with set_forward_context(
            current_timestep=timestep_index,
            attn_metadata=attn_metadata,
            forward_batch=batch,
            quality=batch.sampling_params.quality,
        ):
            prediction = self._predict_noise(
                current_model=current_model,
                latent_model_input=torch.cat([latent_model_input, latent_model_input]),
                timestep=timestep.reshape(-1).repeat(2),
                target_dtype=target_dtype,
                guidance=guidance,
                **conditioning,
            )
        predictions = [
            server_args.pipeline_config.slice_noise_pred(p, latents)
            for p in prediction.chunk(2)
        ]
        cfg_scale = server_args.pipeline_config.get_classifier_free_guidance_scale(
            batch, current_guidance_scale
        )
        return cfg_policy.combine(
            predictions, batch, cfg_scale, server_args.pipeline_config
        )

    @staticmethod
    def _matching_rope_shapes(policy: CFGPolicy) -> bool:
        positive, negative = policy.branches
        return all(
            p.shape == n.shape
            for pos_pair, neg_pair in zip(
                positive.kwargs["freqs_cis"], negative.kwargs["freqs_cis"], strict=True
            )
            for p, n in zip(pos_pair, neg_pair, strict=True)
        )

    @staticmethod
    def _build_batched_cfg_conditioning(positive, negative, *, stack_rope: bool):
        conditioning = {
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
        }
        if stack_rope:
            conditioning["freqs_cis"] = tuple(
                tuple(
                    torch.stack([p, n]) for p, n in zip(pos_pair, neg_pair, strict=True)
                )
                for pos_pair, neg_pair in zip(
                    positive.kwargs["freqs_cis"],
                    negative.kwargs["freqs_cis"],
                    strict=True,
                )
            )
        return conditioning
