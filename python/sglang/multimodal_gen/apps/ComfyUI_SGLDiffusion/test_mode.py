import torch

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request


class SGLDiffusionZImageNode:
    """Node to generate images using SGLang Diffusion."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "generate_image"
    CATEGORY = "SGLDiffusion"
    OUTPUT_NODE = False

    def __init__(self):
        self.ori_forward = None
        self.generator = DiffGenerator.from_pretrained(
            model_path="/root/workspace/sglang-model/Tongyi-MAI/Z-Image-Turbo",  # Replace with actual model path
            pipeline_class_name="ComfyUIZImagePipeline",
            num_gpus=1,
        )

    def forward(self, x, timesteps, context, **kwargs):
        B, C, H, W = x.shape
        height = H * 8
        width = W * 8
        sampling_params = SamplingParams.from_user_sampling_params_args(
            self.generator.server_args.model_path,
            server_args=self.generator.server_args,
            prompt=" ",
            guidance_scale=1.0,
            height=height,
            width=width,
            num_frames=1,  # For images
            num_inference_steps=1,  # Single step for ComfyUI
            comfyui_mode=True,
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

    def generate_image(self, model):
        """Generate image using SGLang Diffusion."""
        ori_forward = model.diffusion_model.forward

        model.diffusion_model.forward = self.forward

        return (model,)


class SGLDiffusionFluxNode:
    """Node to generate images using SGLang Flux."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "generate_image"
    CATEGORY = "SGLDiffusion"
    OUTPUT_NODE = False

    def __init__(self):
        self.ori_forward = None
        self.generator = DiffGenerator.from_pretrained(
            model_path="/root/workspace/sglang-model/FLUX.1-dev",  # Replace with actual model path
            pipeline_class_name="ComfyUIFluxPipeline",
            num_gpus=2,
        )

    def generate_image(self, model):
        """Generate image using SGLang Diffusion."""
        ori_forward = model.model.diffusion_model.forward

        model.model.diffusion_model.forward = self.forward

        return (model,)

    def _unpack_latents(self, latents, height, width, channels):
        batch_size = latents.shape[0]
        # channels=16, height=160, width=90
        latents = latents.view(batch_size, height // 2, width // 2, channels, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, channels, height, width)

        return latents

    def _pack_latents(self, latents):
        # torch.Size([1, 16, 160, 90]) -> torch.Size([1, 3600, 64])
        batch_size, num_channels_latents, height, width = latents.shape
        latents = latents.view(
            batch_size, num_channels_latents, height // 2, 2, width // 2, 2
        )
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(
            batch_size, (height // 2) * (width // 2), num_channels_latents * 4
        )
        return latents

    def forward(self, x, timestep, context, y=None, guidance=None, **kwargs):
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
            self.generator.server_args.model_path,
            server_args=self.generator.server_args,
            prompt=" ",
            guidance_scale=3.5,  # Flux typically uses embedded_cfg_scale=3.5
            height=height,
            width=width,
            num_frames=1,  # For images
            num_inference_steps=1,  # Single step for ComfyUI
            seed=42,
            save_output=False,
            return_frames=False,
            comfyui_mode=True,
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
        return self._unpack_latents(noise_pred, H, W, C)


NODE_CLASS_MAPPINGS = {
    "SGLDiffusionZImageNode": SGLDiffusionZImageNode,
    "SGLDiffusionFluxNode": SGLDiffusionFluxNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SGLDiffusionZImageNode": "SGLDiffusionZImageNode",
    "SGLDiffusionFluxNode": "SGLDiffusionFluxNode",
}
