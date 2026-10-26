# SPDX-License-Identifier: Apache-2.0

# Reference implementation license:
# BSD 3-Clause License
#
# (c) Meta Platforms, Inc. and affiliates.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,this list
# of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this
# list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may
# be used to endorse or promote products derived from this software without specific
# prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
# SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
# DAMAGE.

import math
from dataclasses import replace

import torch
from diffusers.utils.torch_utils import randn_tensor

from sglang.multimodal_gen.configs.sample.moss_image import (
    MOSSImageSamplingParams,
    resolve_uncond_skip_layers,
)
from sglang.multimodal_gen.configs.sample.sampling_params import quality_allows
from sglang.multimodal_gen.runtime.cache.conditioning import (
    cached_encoder_call,
    prefer_conditioning_cache,
)
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.distributed.cfg_parallel_utils import (
    run_cfg_parallel,
)
from sglang.multimodal_gen.runtime.distributed.cfg_policy import CFGPolicy
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.models.dits.moss_image import validate_skip_layers
from sglang.multimodal_gen.runtime.pipelines_core.diffusion_scheduler_utils import (
    get_or_create_request_scheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import DenoisingStage


class MOSSImageDenoisingStage(DenoisingStage):
    @staticmethod
    def _has_extra_slg(params: MOSSImageSamplingParams) -> bool:
        return bool(params.slg_layers) and params.slg_scale > 0.0

    @classmethod
    def _has_dynamic_guidance(cls, params: MOSSImageSamplingParams) -> bool:
        return cls._has_extra_slg(params) or (
            bool(resolve_uncond_skip_layers(params))
            and (params.slg_start != 0.0 or params.slg_stop != 1.0)
        )

    @staticmethod
    def _with_uncond_skip_layers(
        cfg_policy: CFGPolicy, skip_layers: tuple[int, ...]
    ) -> CFGPolicy:
        if all(
            branch.is_conditional
            or tuple(branch.kwargs.get("skip_layers", ())) == skip_layers
            for branch in cfg_policy.branches
        ):
            return cfg_policy
        # Preserve the request-wide branches when changing one step's topology.
        return replace(
            cfg_policy,
            branches=[
                branch
                if branch.is_conditional
                else replace(
                    branch, kwargs={**branch.kwargs, "skip_layers": skip_layers}
                )
                for branch in cfg_policy.branches
            ],
        )

    def _predict_noise_with_cfg(
        self,
        current_model,
        latent_model_input,
        timestep,
        batch,
        timestep_index,
        attn_metadata,
        target_dtype,
        current_guidance_scale,
        cfg_policy,
        cfg_gate_state,
        server_args,
        guidance,
        latents,
    ):
        params = batch.sampling_params
        active = (
            params.slg_start <= timestep_index / len(batch.timesteps) < params.slg_stop
        )
        uncond_layers = resolve_uncond_skip_layers(params) if active else ()
        cfg_policy = self._with_uncond_skip_layers(cfg_policy, uncond_layers)
        # A cached CFG delta represents the old branch topology. Keep the
        # reference's actual forwards when guidance changes within a request.
        if self._has_dynamic_guidance(params):
            cfg_gate_state = None
        if not (active and self._has_extra_slg(params)):
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

        config = server_args.pipeline_config

        def predict(branch):
            branch.configure_batch(batch)
            with set_forward_context(
                current_timestep=timestep_index,
                attn_metadata=attn_metadata,
                forward_batch=batch,
            ):
                prediction = self._predict_noise(
                    current_model=current_model,
                    latent_model_input=latent_model_input,
                    timestep=timestep,
                    target_dtype=target_dtype,
                    guidance=guidance,
                    **branch.kwargs,
                )
            return config.slice_noise_pred(prediction, latents).float()

        slg_branch = replace(
            cfg_policy.branches[0],
            kwargs={
                **cfg_policy.branches[0].kwargs,
                "skip_layers": tuple(params.slg_layers),
            },
        )
        expanded_policy = replace(
            cfg_policy, branches=[*cfg_policy.branches, slg_branch]
        )
        predictions = (
            run_cfg_parallel(expanded_policy, predict)
            if server_args.enable_cfg_parallel
            else [predict(branch) for branch in expanded_policy.branches]
        )
        noise_pred = cfg_policy.combine(
            predictions[:-1],
            batch,
            config.get_classifier_free_guidance_scale(batch, current_guidance_scale),
            config,
            cfg_parallel=server_args.enable_cfg_parallel,
        )
        skipped = predictions[-1]
        if not batch.do_classifier_free_guidance:
            # Reference CFG=1 SLG uses the skipped prediction as its base.
            noise_pred = skipped
        return noise_pred + params.slg_scale * (predictions[0] - skipped)

    def _before_denoising_loop(self, ctx, batch, server_args):
        super()._before_denoising_loop(ctx, batch, server_args)
        self._manage_dit_use_site(self.transformer, "transformer", batch)
        height, width = ctx.latents.shape[-2:]
        with set_forward_context(0, None):
            for branch in ctx.cfg_policy.branches:
                kwargs = branch.kwargs
                kwargs["prepared_conditioning"] = self.transformer.prepare_conditioning(
                    kwargs["encoder_hidden_states"],
                    kwargs.get("encoder_attention_mask"),
                    height,
                    width,
                    ctx.target_dtype,
                    batch_samples=quality_allows(
                        batch.sampling_params.quality, "lossless"
                    ),
                )


class MOSSImageBeforeDenoisingStage(PipelineStage):
    def __init__(self, text_encoder, tokenizer, scheduler):
        super().__init__()
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.scheduler = scheduler

    def component_uses(self, server_args, stage_name=None):
        return [
            ComponentUse(
                self._component_stage_name(stage_name),
                "text_encoder",
                memory_intensive=True,
                start_at_stage_entry=False,
            )
        ]

    @torch.no_grad()
    def _encode_prompt(self, prompts, device, max_sequence_length=4096):
        rows = []
        for prompt in prompts:
            formatted = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            inputs = self.tokenizer(
                formatted,
                return_tensors="pt",
                truncation=True,
                max_length=max_sequence_length,
            )

            def encode_row():
                self.begin_declared_component_use(
                    component_name="text_encoder", module=self.text_encoder
                )
                with set_forward_context(0, None):
                    output = self.text_encoder(
                        **inputs.to(device), use_cache=False, return_dict=True
                    )
                return output.last_hidden_state[0]

            rows.append(
                cached_encoder_call(
                    self.text_encoder,
                    (dict(inputs),),
                    {"device": str(device)},
                    encode_row,
                    namespace=self,
                    nested=False,
                    share_in_group=True,
                )
            )
        self.finish_unused_declared_component(
            component_name="text_encoder", module=self.text_encoder
        )
        length = max(row.shape[0] for row in rows)
        embeds = rows[0].new_zeros(len(rows), length, rows[0].shape[-1])
        mask = torch.zeros(len(rows), length, dtype=torch.bool, device=device)
        for index, row in enumerate(rows):
            embeds[index, : row.shape[0]] = row
            mask[index, : row.shape[0]] = True
        return embeds, mask

    @staticmethod
    def _validate_request(batch, server_args):
        params = batch.sampling_params
        try:
            valid_scale = math.isfinite(params.slg_scale) and params.slg_scale >= 0
        except (TypeError, ValueError):
            valid_scale = False
        if not valid_scale:
            raise ValueError("MOSSImage slg_scale must be finite and nonnegative.")
        validate_skip_layers(
            resolve_uncond_skip_layers(params),
            server_args.pipeline_config.dit_config.arch_config.num_layers,
        )
        validate_skip_layers(
            params.slg_layers if params.slg_layers is not None else (),
            server_args.pipeline_config.dit_config.arch_config.num_layers,
        )
        try:
            valid_window = 0 <= params.slg_start <= params.slg_stop <= 1
        except (TypeError, ValueError):
            valid_window = False
        if not valid_window:
            raise ValueError("MOSSImage requires 0 <= slg_start <= slg_stop <= 1.")
        if (
            not isinstance(batch.height, int)
            or not isinstance(batch.width, int)
            or batch.height <= 0
            or batch.width <= 0
            or batch.height % 16
            or batch.width % 16
        ):
            raise ValueError(
                "MOSSImage height and width must be positive multiples of 16."
            )
        if not math.isfinite(batch.guidance_scale) or batch.guidance_scale < 0:
            raise ValueError("MOSSImage guidance_scale must be finite and nonnegative.")
        prompts = [batch.prompt] if isinstance(batch.prompt, str) else batch.prompt
        if not prompts or not all(isinstance(p, str) for p in prompts):
            raise ValueError(
                "MOSSImage requires a prompt string or nonempty list of strings."
            )
        count = batch.num_outputs_per_prompt
        if count < 1:
            raise ValueError("num_outputs_per_prompt must be positive.")
        max_length = batch.max_sequence_length
        if max_length is None:
            max_length = batch.sampling_params.max_sequence_length
        max_length = 4096 if max_length is None else max_length
        if not 1 <= max_length <= 4096:
            raise ValueError(
                "MOSSImage max_sequence_length must be between 1 and 4096."
            )
        if batch.flow_shift is not None and (
            not math.isfinite(batch.flow_shift) or batch.flow_shift <= 0
        ):
            raise ValueError("MOSSImage flow_shift must be finite and positive.")
        return prompts, max_length

    def _prepare_prompt_embeddings(self, batch, prompts, device, max_length):
        count = batch.num_outputs_per_prompt
        embeds, mask = self._encode_prompt(prompts, device, max_length)
        batch.prompt_embeds = [embeds.repeat_interleave(count, 0)]
        batch.prompt_attention_mask = mask.repeat_interleave(count, 0)
        if batch.do_classifier_free_guidance:
            negative = (
                batch.negative_prompt if batch.negative_prompt is not None else ""
            )
            negative = (
                [negative] * len(prompts) if isinstance(negative, str) else negative
            )
            if len(negative) != len(prompts):
                raise ValueError("negative_prompt must match the prompt batch size.")
            with prefer_conditioning_cache():
                embeds, mask = self._encode_prompt(negative, device, max_length)
            batch.negative_prompt_embeds = [embeds.repeat_interleave(count, 0)]
            batch.negative_attention_mask = mask.repeat_interleave(count, 0)

    @staticmethod
    def _prepare_latents(batch, *, batch_size, latent_channels, device):
        shape = (
            batch_size,
            latent_channels,
            batch.height // 16,
            batch.width // 16,
        )
        if batch.latents is None:
            batch.latents = randn_tensor(
                shape, generator=batch.generator, device=device, dtype=torch.float32
            )
        elif tuple(batch.latents.shape) != shape:
            raise ValueError(f"MOSSImage latents must have shape {shape}.")
        else:
            batch.latents = batch.latents.to(device=device, dtype=torch.float32)
        batch.raw_latent_shape = shape

    def _prepare_timesteps(self, batch, device):
        scheduler = get_or_create_request_scheduler(
            batch, self.scheduler, isolate=batch.flow_shift is not None
        )
        if batch.flow_shift is not None:
            scheduler.set_shift(batch.flow_shift)
        scheduler.set_timesteps(batch.num_inference_steps, device=device)
        batch.timesteps = scheduler.timesteps
        batch.num_inference_steps = len(batch.timesteps)
        batch.sigmas = scheduler.sigmas.tolist()

    @torch.no_grad()
    def forward(self, batch, server_args):
        prompts, max_length = self._validate_request(batch, server_args)
        device = get_local_torch_device()
        self._prepare_prompt_embeddings(batch, prompts, device, max_length)
        self._prepare_latents(
            batch,
            batch_size=len(prompts) * batch.num_outputs_per_prompt,
            latent_channels=server_args.pipeline_config.dit_config.arch_config.latent_channels,
            device=device,
        )
        self._prepare_timesteps(batch, device)
        return batch
