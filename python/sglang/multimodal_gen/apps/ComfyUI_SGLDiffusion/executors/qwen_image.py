"""
QwenImage executor for SGLang Diffusion ComfyUI integration.
"""

import torch

try:
    from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
    from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
except ImportError:
    print(
        "Error: sglang.multimodal_gen is not installed. Please install it using 'pip install sglang[diffusion]'"
    )

import comfy.ldm.common_dit

from .base import SGLDiffusionExecutor


class QwenImageExecutor(SGLDiffusionExecutor):
    """Executor for QwenImage models in ComfyUI."""

    def __init__(self, generator, model_path, model, config):
        super().__init__(generator, model_path, model, config)
        self.patch_size = 2

    def _pack_latents(self, x):
        """Process hidden states for QwenImage model."""
        bs, c, t, h, w = x.shape
        patch_size = self.patch_size
        latents = comfy.ldm.common_dit.pad_to_patch_size(
            x, (1, self.patch_size, self.patch_size)
        )
        orig_shape = latents.shape
        latents = latents.view(
            orig_shape[0],
            orig_shape[1],
            orig_shape[-3],
            orig_shape[-2] // 2,
            2,
            orig_shape[-1] // 2,
            2,
        )
        latents = latents.permute(0, 2, 3, 5, 1, 4, 6)
        latents = latents.reshape(
            orig_shape[0],
            orig_shape[-3] * (orig_shape[-2] // 2) * (orig_shape[-1] // 2),
            orig_shape[1] * 4,
        )
        return latents, orig_shape

    def _unpack_latents(self, latents, num_embeds, orig_shape, x):
        """Unpack hidden states from packed format to standard format."""
        latents = latents[:, :num_embeds].view(
            orig_shape[0],
            orig_shape[-3],
            orig_shape[-2] // 2,
            orig_shape[-1] // 2,
            orig_shape[1],
            2,
            2,
        )
        latents = latents.permute(0, 4, 1, 2, 5, 3, 6)
        latents = latents.reshape(orig_shape)[:, :, :, : x.shape[-2], : x.shape[-1]]
        return latents

    def forward(self, x, timestep, context, **kwargs):
        """Forward pass for QwenImage model."""
        latents, orig_shape = self._pack_latents(x)
        num_embeds = latents.shape[1]
        height = orig_shape[-2] * 8
        width = orig_shape[-1] * 8

        sampling_params = SamplingParams.from_user_sampling_params_args(
            self.model_path,
            server_args=self.generator.server_args,
            prompt=" ",
            guidance_scale=1.0,
            height=height,
            width=width,
            num_frames=1,
            num_inference_steps=1,
            save_output=False,
            suppress_logs=self.should_suppress_logs(timestep),
        )

        # Prepare request (converts SamplingParams to Req)
        req = prepare_request(
            server_args=self.generator.server_args,
            sampling_params=sampling_params,
        )
        # Set ComfyUI-specific inputs directly on the Req object
        req.latents = latents
        req.timesteps = timestep * 1000.0
        req.prompt_embeds = [context]
        req.raw_latent_shape = torch.tensor(latents.shape, dtype=torch.long)
        req.do_classifier_free_guidance = False
        req.generator = [
            torch.Generator("cuda") for _ in range(req.num_outputs_per_prompt)
        ]

        output_batch = self.generator._send_to_scheduler_and_wait_for_response([req])
        noise_pred = output_batch.noise_pred

        return self._unpack_latents(noise_pred, num_embeds, orig_shape, x)


class QwenImageEditExecutor(QwenImageExecutor):
    """Executor for QwenImageEdit models in ComfyUI."""

    def __init__(self, generator, model_path, model, config):
        super().__init__(generator, model_path, model, config)

    def forward(
        self,
        x,
        timestep,
        context,
        attention_mask=None,
        ref_latents=None,
        additional_t_cond=None,
        transformer_options={},
        **kwargs
    ):
        """Forward pass for QwenImageEdit model."""
        latents, orig_shape = self._pack_latents(x)
        num_embeds = latents.shape[1]
        height = orig_shape[-2] * 8
        width = orig_shape[-1] * 8

        # Prepare vae_image_sizes for the condition image (ref_latents)
        vae_image_sizes = []
        pack_ref_latents = None

        # TODO: sgld now don't support multiple condition images, so we only support one condition image for now.
        if ref_latents is not None and len(ref_latents) > 0:
            pack_ref_latents, orig_ref_shape = self._pack_latents(ref_latents[0])
            vae_image_sizes = [(orig_ref_shape[-1], orig_ref_shape[-2])]

        sampling_params = SamplingParams.from_user_sampling_params_args(
            self.model_path,
            server_args=self.generator.server_args,
            prompt=" ",
            guidance_scale=1.0,
            image_path="",
            height=height,
            width=width,
            num_frames=1,
            num_inference_steps=1,
            save_output=False,
            suppress_logs=self.should_suppress_logs(timestep),
        )

        # Prepare request (converts SamplingParams to Req)
        req = prepare_request(
            server_args=self.generator.server_args,
            sampling_params=sampling_params,
        )
        # Set ComfyUI-specific inputs directly on the Req object
        req.latents = latents
        req.image_latent = pack_ref_latents
        req.timesteps = timestep * 1000.0
        req.vae_image_sizes = vae_image_sizes
        req.prompt_embeds = [context]
        req.raw_latent_shape = torch.tensor(latents.shape, dtype=torch.long)
        req.do_classifier_free_guidance = False
        req.generator = [
            torch.Generator("cuda") for _ in range(req.num_outputs_per_prompt)
        ]

        output_batch = self.generator._send_to_scheduler_and_wait_for_response([req])
        noise_pred = output_batch.noise_pred

        return self._unpack_latents(noise_pred, num_embeds, orig_shape, x)
