# SPDX-License-Identifier: Apache-2.0
"""Denoising stage of Kandinsky 6 video SR.

Subclasses the shared :class:`~...pipelines_core.stages.denoising.DenoisingStage` instead of
bypassing it with a free-standing loop (the gap the SGLang PR review flagged, the same one
called out for the K6 TI2VA stage): this gets the constructor's attention-backend inference,
torch.compile-during-offload plumbing, and ``_maybe_enable_cache_dit_and_torch_compile`` for
free, the same hooks every other denoising stage in this repo reuses. ``forward`` is still
overridden wholesale, because the shared per-step loop (``_prepare_denoising_loop`` /
``_run_denoising_step``) assumes one global ``batch.latents`` tensor stepped once per timestep
under CFG; SR instead denoises a list of independent tile chunks, each run through the bundle's
own scheduler (``run_spec.effective_scheduler`` -- ``PiflowScheduler`` or
``FlowMatchEulerDiscreteScheduler``) with a fresh ``set_timesteps`` per chunk, text-free and
without CFG. The breakable-CUDA-graph hooks (``_maybe_get_bcg_runner`` / ``_bcg_run``) are *not*
reused here: their padding logic (``_bcg_pad_prompt_kwargs``) exists to make a captured graph's
shape independent of prompt length, which has no meaning for this text-free DiT, so wiring them
up would add a no-op indirection rather than genuine behaviour.
"""

from __future__ import annotations

import torch

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import DenoisingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.run_spec import (
    SR_CHUNKS_KEY,
    SR_DENOISED_KEY,
    SR_DIT_SPEC_KEY,
    SR_SAMPLING_SPEC_KEY,
    effective_scheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.tiled import (
    denoise_chunks,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.precision import precision_to_dtype


class Kandinsky6SRDenoisingStage(DenoisingStage):
    """Runs the bundle's scheduler over every tile chunk prepared by the latent-prep stage."""

    def __init__(self, transformer, scheduler, pipeline=None) -> None:
        super().__init__(
            transformer=transformer,
            scheduler=scheduler,
            pipeline=pipeline,
            transformer_2=None,
            vae=None,
        )

    def _owns_compile_warmup_lifecycle(self) -> bool:
        # ``forward`` below does not go through the shared ``_denoise`` loop that this guard
        # normally protects, so claim ownership explicitly (same as MiniMaxH3DenoisingStage):
        # the offload/restore wrapper is still entered through ``_offload_for_torch_compile_warmup``.
        return True

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """SR's ``forward`` reads ``batch.extra`` tile-chunk state, not the base class's
        global ``timesteps``/``prompt_embeds``/``generator`` fields (this DiT is text-free and
        steps each tile chunk through its own fresh scheduler instead of one CFG loop over
        ``batch.latents``), so the inherited validator is replaced rather than satisfied."""
        result = VerificationResult()
        result.add_check(
            "extra[SR_SAMPLING_SPEC_KEY]",
            batch.extra.get(SR_SAMPLING_SPEC_KEY),
            V.not_none,
        )
        result.add_check(
            "extra[SR_DIT_SPEC_KEY]", batch.extra.get(SR_DIT_SPEC_KEY), V.not_none
        )
        result.add_check(
            "extra[SR_CHUNKS_KEY]", batch.extra.get(SR_CHUNKS_KEY), V.list_not_empty
        )
        return result

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check(
            "extra[SR_DENOISED_KEY]", batch.extra.get(SR_DENOISED_KEY), V.list_not_empty
        )
        return result

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        dit_dtype = precision_to_dtype(
            server_args.pipeline_config.dit_precision, "dit_precision"
        )
        return [
            ComponentUse(
                self._component_stage_name(stage_name),
                "transformer",
                phase="denoise_tiles",
                target_dtype=dit_dtype,
                preferred_ready_after_request=True,
                memory_intensive=True,
            )
        ]

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        with self._offload_for_torch_compile_warmup(batch):
            return self._denoise_tiles(batch, server_args)

    def _denoise_tiles(self, batch: Req, server_args: ServerArgs) -> Req:
        spec = batch.extra.pop(SR_SAMPLING_SPEC_KEY)
        dit_spec = batch.extra.pop(SR_DIT_SPEC_KEY)
        chunks = batch.extra.pop(SR_CHUNKS_KEY)
        device = get_local_torch_device()

        # Reuse the shared cache-dit / torch.compile wiring (both are no-ops unless the
        # corresponding server arg actually requests them): ``spec.steps_per_chunk`` is this
        # request's DiT-call count, the closest analogue of the global ``num_inference_steps``
        # those hooks expect.
        self._maybe_enable_cache_dit_and_torch_compile(spec.steps_per_chunk, batch)

        scheduler = effective_scheduler(spec, self.scheduler)
        with self.use_declared_component(
            component_name="transformer", module=self.transformer, phase="denoise_tiles"
        ) as transformer:
            assert transformer is not None
            self.transformer = transformer
            total = spec.steps_per_chunk * len(chunks)
            with self.progress_bar(
                total=total, batch=batch, desc="Kandinsky6 SR denoising"
            ) as progress:
                denoised = denoise_chunks(
                    chunks,
                    transformer,
                    scheduler,
                    dit_spec=dit_spec,
                    spec=spec,
                    device=device,
                    step_context=lambda step: set_forward_context(
                        current_timestep=step, attn_metadata=None, forward_batch=batch
                    ),
                    on_step=progress.update,
                )
        self._finish_active_component_use()
        batch.extra[SR_DENOISED_KEY] = denoised
        return batch
