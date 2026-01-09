# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
Decoding stage for diffusion pipelines.
"""

import weakref
from collections.abc import Iterable

import torch
from tqdm.auto import tqdm

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.distributed import (
    get_local_torch_device,
    get_world_group,
)
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
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs, get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


def _ensure_tensor_decode_output(decode_output):
    """
    Ensure VAE decode output is a tensor.

    Some VAE implementations return DecoderOutput objects with a .sample attribute,
    tuples, or tensors directly. This function normalizes the output to always be a tensor.

    Args:
        decode_output: Output from VAE.decode(), can be DecoderOutput, tuple, or torch.Tensor

    Returns:
        torch.Tensor: The decoded image tensor
    """
    if isinstance(decode_output, tuple):
        return decode_output[0]
    if hasattr(decode_output, "sample"):
        return decode_output.sample
    return decode_output


def _post_process_decoded_output(decode_output: torch.Tensor) -> torch.Tensor:
    """
    Post-process decoded output from VAE.

    This includes scaling the output to [0, 1] range and moving it to CPU.

    Args:
        decode_output: Decoded output tensor from VAE

    Returns:
        torch.Tensor: Post-processed tensor in [0, 1] range on CPU
    """

    frame = _ensure_tensor_decode_output(decode_output)
    frame = (frame / 2 + 0.5).clamp(0, 1)
    return frame.cpu()


class DecodingStage(PipelineStage):
    """
    Stage for decoding latent representations into pixel space.

    This stage handles the decoding of latent representations into the final
    output format (e.g., pixel values).
    """

    def __init__(self, vae, pipeline=None) -> None:
        super().__init__()
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

    def scale_and_shift(self, latents: torch.Tensor, server_args):
        scaling_factor, shift_factor = (
            server_args.pipeline_config.get_decode_scale_and_shift(
                latents.device, latents.dtype, self.vae
            )
        )

        # 1. scale
        if isinstance(scaling_factor, torch.Tensor):
            latents = latents / scaling_factor.to(latents.device, latents.dtype)
        else:
            latents = latents / scaling_factor

        # 2. apply shifting if needed
        if shift_factor is not None:
            if isinstance(shift_factor, torch.Tensor):
                latents += shift_factor.to(latents.device, latents.dtype)
            else:
                latents += shift_factor
        return latents

    @torch.no_grad()
    def decode(
        self,
        latents: torch.Tensor,
        sampling_params: SamplingParams,
        server_args: ServerArgs,
    ) -> torch.Tensor:
        """
        Decode latent representations into pixel space using VAE.

        Args:
            latents: Input latent tensor with shape (batch, channels, frames, height_latents, width_latents)
            sampling_params: Parameters for sampling, including height and width of the output
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

        # scale and shift
        latents = self.scale_and_shift(latents, server_args)
        # Preprocess latents before decoding (e.g., unpatchify for standard Flux2 VAE)
        latents = server_args.pipeline_config.preprocess_decoding(
            latents, server_args, vae=self.vae
        )

        # Decode latents
        with torch.autocast(
            device_type=current_platform.device_type,
            dtype=vae_dtype,
            enabled=vae_autocast_enabled,
        ):
            try:
                # TODO: make it more specific
                if server_args.pipeline_config.vae_tiling:
                    self.vae.enable_tiling()
            except Exception:
                pass
            if not vae_autocast_enabled:
                latents = latents.to(vae_dtype)

            if latents.dim() == 5:
                # Decode video frames in chunks to avoid out of memory errors.
                batch_size, num_channels, num_frames, _, _ = latents.shape
                num_sample_frames = num_frames * self.vae.temporal_compression_ratio
                gpu_mem_before_decoding = current_platform.get_available_gpu_memory()

                # Estimated memory needed per frame
                # TODO: Need to consider parallel / tiling
                output_frame_size = (
                    sampling_params.height
                    * sampling_params.width
                    * num_channels
                    * latents.element_size()
                )

                num_sample_frames_per_chunk = int(
                    gpu_mem_before_decoding
                    // (output_frame_size / (1 << 30))
                    // batch_size
                )
                tile_latent_stride_num_frames = (
                    num_sample_frames_per_chunk // self.vae.temporal_compression_ratio
                )

                # Decode all frames at once
                if num_frames <= tile_latent_stride_num_frames:
                    decode_output = self.vae.decode(latents)
                    frames = _post_process_decoded_output(decode_output)
                else:
                    # Decode in chunks
                    overlap_sample_frames = (
                        self.vae.config.temporal_tiling_num_overlap_latent_frames
                        * self.vae.temporal_compression_ratio
                    )
                    frame_slices = []
                    with self.progress_bar(total=num_sample_frames) as progress_bar:
                        for i in range(0, num_frames, tile_latent_stride_num_frames):
                            latent_step = min(
                                tile_latent_stride_num_frames,
                                num_frames - i,
                            )
                            sample_frame_step = (
                                latent_step * self.vae.temporal_compression_ratio
                            )
                            latent_step_with_overlap = (
                                latent_step
                                + self.vae.config.temporal_tiling_num_overlap_latent_frames
                            )

                            decode_output = self.vae.decode(
                                latents[
                                    :, :, i : i + latent_step_with_overlap + 1, :, :
                                ]
                            )
                            if i > 0:
                                decode_output = decode_output[
                                    :, :, overlap_sample_frames + 1 :, :, :
                                ]

                            frame = _post_process_decoded_output(decode_output)
                            frame_slices.append(frame)

                            if progress_bar is not None:
                                progress_bar.update(sample_frame_step)
                        frames = torch.cat(frame_slices, dim=2)[
                            :, :, :num_sample_frames
                        ]
            else:
                # Decode image
                decode_output = self.vae.decode(latents)
                frames = _post_process_decoded_output(decode_output)
            return frames

    def load_model(self):
        # load vae if not already loaded (used for memory constrained devices)
        pipeline = self.pipeline() if self.pipeline else None
        if not self.server_args.model_loaded["vae"]:
            loader = VAELoader()
            self.vae = loader.load(
                self.server_args.model_paths["vae"], self.server_args
            )
            if pipeline:
                pipeline.add_module("vae", self.vae)
            self.server_args.model_loaded["vae"] = True

    def offload_model(self):
        # Offload models if needed
        self.maybe_free_model_hooks()

        if self.server_args.vae_cpu_offload:
            self.vae.to("cpu", non_blocking=True)

        if torch.backends.mps.is_available():
            del self.vae
            pipeline = self.pipeline() if self.pipeline else None
            if pipeline is not None and "vae" in pipeline.modules:
                del pipeline.modules["vae"]
            self.server_args.model_loaded["vae"] = False

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
                - vae_cpu_offload: Whether to offload VAE to CPU after decoding
                - model_loaded: Track VAE loading state
                - model_paths: Path to VAE model if loading needed

        Returns:
            Modified batch with:
                - output: Decoded frames (batch, channels, frames, height, width) as CPU float32
                - trajectory_decoded (if requested): List of decoded frames per timestep
        """
        # load vae if not already loaded (used for memory constrained devices)
        self.load_model()

        frames = self.decode(batch.latents, batch.sampling_params, server_args)

        # decode trajectory latents if needed
        if batch.return_trajectory_decoded:
            assert (
                batch.trajectory_latents is not None
            ), "batch should have trajectory latents"

            # 1. Batch trajectory decoding to improve GPU utilization
            # batch.trajectory_latents is [batch_size, timesteps, channels, frames, height, width]
            B, T, C, F, H, W = batch.trajectory_latents.shape
            flat_latents = batch.trajectory_latents.view(B * T, C, F, H, W)

            logger.info("decoding %s trajectory latents in batch", B * T)
            # Use the optimized batch decode
            all_decoded = self.decode(flat_latents, server_args)

            # 2. Reshape back
            # Keep on GPU to allow faster vectorized post-processing
            decoded_tensor = all_decoded.view(B, T, *all_decoded.shape[1:])

            # Convert to list of tensors (per timestep) as expected by OutputBatch
            # Each element in list is [B, channels, frames, H_out, W_out]
            trajectory_decoded = [decoded_tensor[:, i] for i in range(T)]
        else:
            trajectory_decoded = None

        # Update batch with decoded image
        output_batch = OutputBatch(
            output=frames,
            trajectory_timesteps=batch.trajectory_timesteps,
            trajectory_latents=batch.trajectory_latents,
            trajectory_decoded=trajectory_decoded,
            timings=batch.timings,
        )

        self.offload_model()

        return output_batch

    def progress_bar(
        self, iterable: Iterable | None = None, total: int | None = None
    ) -> tqdm:
        """
        Create a progress bar for the denoising process.

        Args:
            iterable: The iterable to iterate over.
            total: The total number of items.

        Returns:
            A tqdm progress bar.
        """
        local_rank = get_world_group().local_rank
        disable = local_rank != 0
        return tqdm(iterable=iterable, total=total, disable=disable)
