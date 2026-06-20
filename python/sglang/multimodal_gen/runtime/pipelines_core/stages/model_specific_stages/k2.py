"""Krea-2 (K2) pre-processing stage: text encode + latent/timestep preparation.

Consolidates everything before the denoising loop: Qwen3-VL text encoding with
the K2 system-prompt template and 12-layer hidden-state stacking, initial noise
latent packing, and the rectified-flow timestep schedule. Produces a batch the
standard DenoisingStage consumes unchanged.
"""

import numpy as np
import torch
from einops import rearrange

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# K2 text-encode template (Qwen3-VL, text only). The system prefix is stripped
# from the hidden states after encoding (drop_idx tokens); a fixed assistant
# suffix is appended so the final token positions match training.
_PREFIX = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and "
    "background:<|im_end|>\n<|im_start|>user\n"
)
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
_DROP_IDX = 34
_SUFFIX_START_IDX = 5
_MAX_LENGTH = 512
_SELECT_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)


class Krea2BeforeDenoisingStage(PipelineStage):
    def __init__(self, vae, text_encoder, tokenizer, processor, transformer, scheduler):
        super().__init__()
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.processor = processor
        self.transformer = transformer
        self.scheduler = scheduler

    def component_uses(self, server_args: ServerArgs, stage_name: str | None = None):
        # Declare the text encoder so the residency manager can CPU-offload it
        # (with --text-encoder-cpu-offload) for the denoise loop, where it is idle.
        return [
            ComponentUse(
                self._component_stage_name(stage_name),
                "text_encoder",
                target_dtype=torch.bfloat16,
            )
        ]

    @torch.no_grad()
    def _encode(self, prompts, device, dtype):
        """Reproduce the K2 conditioner: template + 12-layer hidden-state stack."""
        text = [_PREFIX + p for p in prompts]
        suffix_inputs = self.processor(
            text=[_SUFFIX] * len(text), return_tensors="pt"
        ).to(device)
        suffix_ids = suffix_inputs["input_ids"]
        suffix_mask = suffix_inputs["attention_mask"].bool()

        # Pad to the batch's longest sequence (no padding for a single prompt) so
        # the joint stream carries only valid tokens and attention needs no mask.
        # The Qwen3-VL encoder is causal with right padding, so valid-token hidden
        # states are identical to fixed max-length padding.
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="longest",
            max_length=_MAX_LENGTH + _DROP_IDX - _SUFFIX_START_IDX,
            return_tensors="pt",
        ).to(device)
        input_ids = torch.cat([inputs["input_ids"], suffix_ids], dim=1)
        mask = torch.cat([inputs["attention_mask"].bool(), suffix_mask], dim=1)

        states = self.text_encoder(
            input_ids=input_ids, attention_mask=mask, output_hidden_states=True
        )
        hiddens = torch.stack([states.hidden_states[i] for i in _SELECT_LAYERS], dim=2)
        hiddens = hiddens[:, _DROP_IDX:]
        mask = mask[:, _DROP_IDX:]
        return hiddens.to(dtype), mask

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        device = get_local_torch_device()
        pipeline_config = server_args.pipeline_config
        arch = pipeline_config.dit_config.arch_config
        dtype = torch.bfloat16

        prompts = batch.prompt if isinstance(batch.prompt, list) else [batch.prompt]
        n = len(prompts)
        height, width = int(batch.height), int(batch.width)
        patch = arch.patch
        vsf = pipeline_config.get_vae_scale_factor()

        # Text conditioning (positive + negative for CFG). The residency manager
        # pages the encoder to GPU for this block only, then offloads it for the
        # denoise loop (frees ~8GB) when --text-encoder-cpu-offload is set.
        neg_prompts = (
            batch.negative_prompt
            if isinstance(batch.negative_prompt, list)
            else [batch.negative_prompt or ""] * n
        )
        with self.use_declared_component(
            component_name="text_encoder", module=self.text_encoder
        ) as text_encoder:
            self.text_encoder = text_encoder
            prompt_embeds, prompt_mask = self._encode(prompts, device, dtype)
            neg_embeds, neg_mask = self._encode(neg_prompts, device, dtype)

        # Initial noise latents, packed to [B, S_img, channels*patch**2].
        seed = batch.seed if batch.seed is not None else 0
        lat_h, lat_w = height // vsf, width // vsf
        noise = torch.cat(
            [
                torch.randn(
                    1,
                    arch.channels,
                    lat_h,
                    lat_w,
                    device=device,
                    dtype=dtype,
                    generator=torch.Generator(device=device).manual_seed(seed + i),
                )
                for i in range(n)
            ],
            dim=0,
        )
        latents = rearrange(
            noise, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch
        )

        # Rectified-flow timestep schedule with resolution-dependent time-shift mu.
        # The repo ships a FlowMatchEulerDiscreteScheduler; drive it with the Flux
        # sigma grid linspace(1, 1/n, n) so its shifted sigmas match the K2 sampler.
        num_inference_steps = batch.num_inference_steps
        image_seq_len = (lat_h // patch) * (lat_w // patch)
        mu = pipeline_config.compute_mu(image_seq_len)
        scheduler = self.scheduler
        sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
        scheduler.set_timesteps(
            num_inference_steps, device=device, mu=mu, sigmas=sigmas
        )

        batch.prompt_embeds = [prompt_embeds]
        batch.prompt_embeds_mask = [prompt_mask]
        batch.negative_prompt_embeds = [neg_embeds]
        batch.negative_prompt_embeds_mask = [neg_mask]
        batch.latents = latents
        batch.raw_latent_shape = latents.shape
        batch.scheduler = scheduler
        # The DiT's TimeEmbed applies its own 1000x, so feed it the [0,1] sigmas
        # (the diffusers scheduler reports timesteps on the 0..1000 scale instead).
        batch.timesteps = scheduler.sigmas[:num_inference_steps]
        batch.num_inference_steps = num_inference_steps
        batch.sigmas = None
        batch.generator = torch.Generator(device=device).manual_seed(seed)
        return batch
