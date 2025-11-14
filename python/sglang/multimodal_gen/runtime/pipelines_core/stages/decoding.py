# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
Decoding stage for diffusion pipelines.
"""

import weakref

import torch

from sglang.multimodal_gen.configs.models.vaes.base import VAEArchConfig
from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import (
    QwenImageEditPipelineConfig,
    QwenImagePipelineConfig,
)
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.loader.component_loader import VAELoader
from sglang.multimodal_gen.runtime.models.vaes.common import ParallelTiledVAE
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    PipelineStage,
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs, get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


class DecodingStage(PipelineStage):
    """
    Stage for decoding latent representations into pixel space.

    This stage handles the decoding of latent representations into the final
    output format (e.g., pixel values).
    """

    def __init__(self, vae, pipeline=None) -> None:
        self.vae: ParallelTiledVAE = vae
        self.pipeline = weakref.ref(pipeline) if pipeline else None

    @property
    def parallelism_type(self) -> StageParallelismType:
        if get_global_server_args().enable_cfg_parallel:
            return StageParallelismType.MAIN_RANK_ONLY
        return StageParallelismType.REPLICATED

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify decoding stage inputs."""
        result = VerificationResult()
        # Denoised latents for VAE decoding: [batch_size, channels, frames, height_latents, width_latents]
        # result.add_check("latents", batch.latents, [V.is_tensor, V.with_dims(5)])
        return result

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify decoding stage outputs."""
        result = VerificationResult()
        # Decoded video/images: [batch_size, channels, frames, height, width]
        # result.add_check("output", batch.output, [V.is_tensor, V.with_dims(5)])
        return result

    def scale_and_shift(
        self, vae_arch_config: VAEArchConfig, latents: torch.Tensor, server_args
    ):
        # 1. scale
        is_qwen_image = isinstance(
            server_args.pipeline_config, QwenImagePipelineConfig
        ) or isinstance(server_args.pipeline_config, QwenImageEditPipelineConfig)
        if is_qwen_image:
            scaling_factor = 1.0 / torch.tensor(
                vae_arch_config.latents_std, device=latents.device
            ).view(1, vae_arch_config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
        else:
            scaling_factor = vae_arch_config.scaling_factor

        if isinstance(scaling_factor, torch.Tensor):
            latents = latents / scaling_factor.to(latents.device, latents.dtype)
        else:
            latents = latents / scaling_factor

        # 2. shift
        if is_qwen_image:
            shift_factor = (
                torch.tensor(vae_arch_config.latents_mean)
                .view(1, vae_arch_config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
        else:
            shift_factor = getattr(vae_arch_config, "shift_factor", None)

        # Apply shifting if needed
        if shift_factor is not None:
            if isinstance(shift_factor, torch.Tensor):
                latents += shift_factor.to(latents.device, latents.dtype)
            else:
                latents += shift_factor
        return latents

    @torch.no_grad()
    def decode(self, latents: torch.Tensor, server_args: ServerArgs) -> torch.Tensor:
        """
        Decode latent representations into pixel space using VAE.

        Args:
            latents: Input latent tensor with shape (batch, channels, frames, height_latents, width_latents)
            server_args: Configuration containing:
                - disable_autocast: Whether to disable automatic mixed precision (default: False)
                - pipeline_config.vae_precision: VAE computation precision ("fp32", "fp16", "bf16")
                - pipeline_config.vae_tiling: Whether to enable VAE tiling for memory efficiency

        Returns:
            Decoded video tensor with shape (batch, channels, frames, height, width),
            normalized to [0, 1] range and moved to CPU as float32
        """
        self.vae = self.vae.to(get_local_torch_device())
        latents = latents.to(get_local_torch_device())
        # Setup VAE precision
        vae_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.vae_precision]
        vae_autocast_enabled = (
            vae_dtype != torch.float32
        ) and not server_args.disable_autocast
        vae_arch_config = server_args.pipeline_config.vae_config.arch_config

        # scale and shift
        latents = self.scale_and_shift(vae_arch_config, latents, server_args)

        # Decode latents
        with torch.autocast(
            device_type="cuda", dtype=vae_dtype, enabled=vae_autocast_enabled
        ):
            try:
                # TODO: make it more specific
                if server_args.pipeline_config.vae_tiling:
                    self.vae.enable_tiling()
            except Exception:
                pass
            if not vae_autocast_enabled:
                latents = latents.to(vae_dtype)
            image = self.vae.decode(latents)

        # De-normalize image to [0, 1] range
        image = (image / 2 + 0.5).clamp(0, 1)
        return image

    @torch.no_grad()
    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> OutputBatch:
        """
        Decode latent representations into pixel space.

        This method processes the batch through the VAE decoder, converting latent
        representations to pixel-space video/images. It also optionally decodes
        trajectory latents for visualization purposes.

        Args:
            batch: The current batch containing:
                - latents: Tensor to decode (batch, channels, frames, height_latents, width_latents)
                - return_trajectory_decoded (optional): Flag to decode trajectory latents
                - trajectory_latents (optional): Latents at different timesteps
                - trajectory_timesteps (optional): Corresponding timesteps
            server_args: Configuration containing:
                - output_type: "latent" to skip decoding, otherwise decode to pixels
                - vae_cpu_offload: Whether to offload VAE to CPU after decoding
                - model_loaded: Track VAE loading state
                - model_paths: Path to VAE model if loading needed

        Returns:
            Modified batch with:
                - output: Decoded frames (batch, channels, frames, height, width) as CPU float32
                - trajectory_decoded (if requested): List of decoded frames per timestep
        """
        # load vae if not already loaded (used for memory constrained devices)
        pipeline = self.pipeline() if self.pipeline else None
        if not server_args.model_loaded["vae"]:
            loader = VAELoader()
            self.vae = loader.load(server_args.model_paths["vae"], server_args)
            if pipeline:
                pipeline.add_module("vae", self.vae)
            server_args.model_loaded["vae"] = True

        if server_args.output_type == "latent":
            frames = batch.latents
        else:
            frames = self.decode(batch.latents, server_args)

        # decode trajectory latents if needed
        if batch.return_trajectory_decoded:
            trajectory_decoded = []
            assert (
                batch.trajectory_latents is not None
            ), "batch should have trajectory latents"
            for idx in range(batch.trajectory_latents.shape[1]):
                # batch.trajectory_latents is [batch_size, timesteps, channels, frames, height, width]
                cur_latent = batch.trajectory_latents[:, idx, :, :, :, :]
                cur_timestep = batch.trajectory_timesteps[idx]
                logger.info("decoding trajectory latent for timestep: %s", cur_timestep)
                decoded_frames = self.decode(cur_latent, server_args)
                trajectory_decoded.append(decoded_frames.cpu().float())
        else:
            trajectory_decoded = None

        # Convert to CPU float32 for compatibility
        frames = frames.cpu().float()

        # Update batch with decoded image
        output_batch = OutputBatch(
            output=frames,
            trajectory_timesteps=batch.trajectory_timesteps,
            trajectory_latents=batch.trajectory_latents,
            trajectory_decoded=trajectory_decoded,
        )

        # Offload models if needed
        if hasattr(self, "maybe_free_model_hooks"):
            self.maybe_free_model_hooks()

        if server_args.vae_cpu_offload:
            self.vae.to("cpu")

        if torch.backends.mps.is_available():
            del self.vae
            if pipeline is not None and "vae" in pipeline.modules:
                del pipeline.modules["vae"]
            server_args.model_loaded["vae"] = False

        return output_batch
