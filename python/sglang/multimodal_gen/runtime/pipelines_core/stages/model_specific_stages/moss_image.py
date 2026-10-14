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

import torch
from diffusers.utils.torch_utils import randn_tensor

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.models.dits.moss_image import validate_skip_layers
from sglang.multimodal_gen.runtime.pipelines_core.diffusion_scheduler_utils import (
    get_or_create_request_scheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage


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
            )
        ]

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
            ).to(device)
            with set_forward_context(0, None):
                output = self.text_encoder(**inputs, use_cache=False, return_dict=True)
            rows.append(output.last_hidden_state[0])
        length = max(row.shape[0] for row in rows)
        embeds = rows[0].new_zeros(len(rows), length, rows[0].shape[-1])
        mask = torch.zeros(len(rows), length, dtype=torch.bool, device=device)
        for index, row in enumerate(rows):
            embeds[index, : row.shape[0]] = row
            mask[index, : row.shape[0]] = True
        return embeds, mask

    @staticmethod
    def _validate_request(batch, server_args):
        validate_skip_layers(
            batch.sampling_params.uncond_skip_layers,
            server_args.pipeline_config.dit_config.arch_config.num_layers,
        )
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
        with self.use_declared_component(
            component_name="text_encoder", module=self.text_encoder
        ) as encoder:
            self.text_encoder = encoder
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
                    raise ValueError(
                        "negative_prompt must match the prompt batch size."
                    )
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
