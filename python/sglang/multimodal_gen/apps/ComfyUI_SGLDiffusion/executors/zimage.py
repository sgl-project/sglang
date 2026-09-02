"""
ZImage executor for SGLang Diffusion ComfyUI integration.
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


class ZImageExecutor(SGLDiffusionExecutor):
    """Executor for ZImage models in ComfyUI."""

    def __init__(self, generator, model_path, model, config):
        super().__init__(generator, model_path, model, config)

    def forward(self, x, timesteps, context, **kwargs):
        """Forward pass for ZImage model."""
        B, C, H, W = x.shape
        height = H * 8
        width = W * 8
        sampling_params = SamplingParams.from_user_sampling_params_args(
            self.model_path,
            server_args=self.generator.server_args,
            prompt=" ",
            guidance_scale=1.0,
            height=height,
            width=width,
            num_frames=1,  # For images
            num_inference_steps=1,  # Single step for ComfyUI
            save_output=False,
            suppress_logs=self.should_suppress_logs(timesteps),
        )

        # Prepare request (converts SamplingParams to Req)
        req = prepare_request(
            server_args=self.generator.server_args,
            sampling_params=sampling_params,
        )
        latents = x.unsqueeze(2)
        context = context.squeeze(0)
        # Set ComfyUI-specific inputs directly on the Req object
        req.latents = latents  # ComfyUI's x parameter
        req.timesteps = timesteps * 1000.0  # ComfyUI's timesteps parameter
        req.prompt_embeds = [
            context
        ]  # ComfyUI's context parameter (must be List[Tensor])
        req.raw_latent_shape = torch.tensor(latents.shape, dtype=torch.long)
        req.do_classifier_free_guidance = False
        req.generator = [
            torch.Generator("cuda") for _ in range(req.num_outputs_per_prompt)
        ]

        output_batch = self.generator._send_to_scheduler_and_wait_for_response([req])
        noise_pred = output_batch.noise_pred

        return noise_pred.permute(1, 0, 2, 3).to(x.device)
