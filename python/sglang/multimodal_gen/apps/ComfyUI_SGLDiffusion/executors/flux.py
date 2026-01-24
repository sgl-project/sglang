"""
Flux executor for SGLang Diffusion ComfyUI integration.
"""

import torch

try:
    from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
    from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
except ImportError:
    print(
        "Error: sglang.multimodal_gen is not installed. Please install it using 'pip install sglang[diffusion]'"
    )

from .base import SGLDiffusionExecutor


class FluxExecutor(SGLDiffusionExecutor):
    """Executor for Flux models in ComfyUI."""

    def __init__(self, generator, model_path, model, config):
        super().__init__(generator, model_path, model, config)

    def forward(self, x, timestep, context, y=None, guidance=None, **kwargs):
        """Forward pass for Flux model."""
        hidden_states = self._pack_latents(x)
        timesteps = timestep * 1000.0
        encoder_hidden_states = context
        pooled_projections = y
        guidance = guidance * 1000.0

        B, C, H, W = x.shape
        height = H * 8
        width = W * 8
        # Create SamplingParams
        sampling_params = SamplingParams.from_user_sampling_params_args(
            self.model_path,
            server_args=self.generator.server_args,
            prompt=" ",
            guidance_scale=3.5,  # Flux typically uses embedded_cfg_scale=3.5
            height=height,
            width=width,
            num_frames=1,
            num_inference_steps=1,
            save_output=False,
        )

        # Prepare request (converts SamplingParams to Req)
        req = prepare_request(
            server_args=self.generator.server_args,
            sampling_params=sampling_params,
        )
        req.latents = hidden_states  # Set as [B, S, D] format directly
        req.timesteps = timesteps  # ComfyUI's timesteps parameter
        req.prompt_embeds = [pooled_projections, encoder_hidden_states]  # [CLIP, T5]
        req.raw_latent_shape = torch.tensor(hidden_states.shape, dtype=torch.long)

        # Set pooled_projections (required by Flux)
        req.pooled_embeds = [pooled_projections]  # List format as per Req definition
        req.do_classifier_free_guidance = False
        req.generator = [
            torch.Generator("cuda") for _ in range(req.num_outputs_per_prompt)
        ]

        # Send request to scheduler
        output_batch = self.generator._send_to_scheduler_and_wait_for_response([req])
        noise_pred = output_batch.noise_pred
        return self._unpack_latents(noise_pred, H, W, C).to(x.device)
