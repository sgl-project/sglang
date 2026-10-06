# SPDX-License-Identifier: Apache-2.0
"""Text-free SR tile denoising with shared residency, compile and cache-DiT hooks.

Each chunk resets its scheduler; the generic single-latent CFG loop and
prompt-padding BCG path do not apply."""

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
        """Validate tile-chunk state instead of the shared text/CFG loop inputs."""
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

        self._maybe_enable_cache_dit_and_torch_compile(spec.num_steps, batch)

        with self.use_declared_component(
            component_name="transformer", module=self.transformer, phase="denoise_tiles"
        ) as transformer:
            assert transformer is not None
            self.transformer = transformer
            total = spec.num_steps * len(chunks)
            with self.progress_bar(
                total=total, batch=batch, desc="Kandinsky6 SR denoising"
            ) as progress:
                denoised = denoise_chunks(
                    chunks,
                    transformer,
                    self.scheduler,
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
