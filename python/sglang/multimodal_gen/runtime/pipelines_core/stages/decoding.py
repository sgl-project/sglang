# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
Decoding stage for diffusion pipelines.
"""

import weakref

import torch
import torch.nn as nn

from sglang.multimodal_gen.runtime.distributed import (
    get_decode_parallel_world_size,
    get_local_torch_device,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.vae_loader import VAELoader
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
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
from sglang.multimodal_gen.runtime.utils.precision import (
    autocast_context,
    autocast_enabled,
    resolve_precision,
    temporary_module_dtype,
)
from sglang.multimodal_gen.runtime.utils.torch_compile import (
    ActiveTargetCompiledCallable,
    build_torch_compile_kwargs,
    resolve_torch_compile_mode,
)

logger = init_logger(__name__)


def scale_and_shift_latents(latents: torch.Tensor, server_args, vae) -> torch.Tensor:
    """De-normalize latents before VAE decode (single shared implementation).

    Used by DecodingStage.scale_and_shift and by realtime stages that decode
    outside a DecodingStage instance.
    """
    scaling_factor, shift_factor = (
        server_args.pipeline_config.get_decode_scale_and_shift(
            latents.device, latents.dtype, vae
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
            latents = latents + shift_factor.to(latents.device, latents.dtype)
        else:
            latents = latents + shift_factor
    return latents


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


class DecodingStage(PipelineStage):
    """
    Stage for decoding latent representations into pixel space.

    This stage handles the decoding of latent representations into the final
    output format (e.g., pixel values).
    """

    @property
    def role_affinity(self):
        from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType

        return RoleType.DECODER

    def __init__(self, vae, pipeline=None, component_name: str = "vae") -> None:
        super().__init__()
        self.vae: ParallelTiledVAE = vae
        self.pipeline = weakref.ref(pipeline) if pipeline else None
        self.component_name = component_name
        self._compiled_vae_decode = ActiveTargetCompiledCallable()

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        vae_dtype = resolve_precision(
            server_args, self.component_name, precision_attr="vae_precision"
        )
        stage_name = self._component_stage_name(stage_name)
        return [
            ComponentUse(
                stage_name,
                self.component_name,
                target_dtype=vae_dtype,
                keep_ready_after_warmup=True,
            )
        ]

    @property
    def parallelism_type(self) -> StageParallelismType:
        server_args = get_global_server_args()
        if server_args.enable_cfg_parallel:
            if self._can_use_parallel_decode():
                return StageParallelismType.REPLICATED
            return StageParallelismType.MAIN_RANK_ONLY
        return StageParallelismType.REPLICATED

    def _can_use_parallel_decode(self) -> bool:
        return (
            model_parallel_is_initialized()
            and get_decode_parallel_world_size() > 1
            and self.vae.use_parallel_decode
        )

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
        return scale_and_shift_latents(latents, server_args, self.vae)

    def _get_vae_decode_fn(self, vae, server_args: ServerArgs):
        if not server_args.enable_torch_compile or not isinstance(vae, nn.Module):
            return vae.decode

        will_compile = (
            self._compiled_vae_decode.target_id != id(vae)
            or self._compiled_vae_decode.compiled_module is None
        )
        if current_platform.is_npu():
            compile_kwargs = build_torch_compile_kwargs(mode=None)
            if will_compile:
                logger.info("Compiling VAE decode with torchair backend on NPU")
        else:
            mode = resolve_torch_compile_mode(
                "SGLANG_VAE_TORCH_COMPILE_MODE",
                "SGLANG_TORCH_COMPILE_MODE",
                default="default",
            )
            compile_kwargs = build_torch_compile_kwargs(mode=mode)
            if will_compile:
                logger.info("Compiling VAE decode with mode: %s", mode)

        return self._compiled_vae_decode.get_or_compile(
            vae, vae.decode, compile_kwargs=compile_kwargs
        )

    @torch.no_grad()
    def decode(
        self,
        latents: torch.Tensor,
        server_args: ServerArgs,
        *,
        vae_dtype: torch.dtype,
    ) -> torch.Tensor:
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
        latents = latents.to(get_local_torch_device())
        # Setup VAE precision from user policy.
        vae_dtype = resolve_precision(
            server_args, self.component_name, precision_attr="vae_precision"
        )
        vae_autocast_enabled = autocast_enabled(vae_dtype, server_args.disable_autocast)

        # scale and shift
        latents = self.scale_and_shift(latents, server_args)
        # Preprocess latents before decoding (e.g., unpatchify for standard Flux2 VAE)
        latents = server_args.pipeline_config.preprocess_decoding(
            latents, server_args, vae=self.vae
        )
        if latents.device.type == "mps":
            torch.mps.synchronize()
            torch.mps.empty_cache()

        # Decode latents
        with autocast_context(vae_dtype, server_args.disable_autocast):
            try:
                # TODO: make it more specific
                if server_args.pipeline_config.vae_tiling:
                    self.vae.enable_tiling()
            except Exception:
                pass
            should_cast_vae = not vae_autocast_enabled
            if not vae_autocast_enabled:
                latents = latents.to(vae_dtype)
            with temporary_module_dtype(
                self.vae, vae_dtype, enabled=should_cast_vae
            ) as vae:
                decode_output = self._get_vae_decode_fn(vae, server_args)(latents)
                image = _ensure_tensor_decode_output(decode_output)

        # De-normalize image to [0, 1] range
        image = (image / 2 + 0.5).clamp(0, 1)
        return image

    def load_model(self):
        # load vae if not already loaded (used for memory constrained devices)
        pipeline = self.pipeline() if self.pipeline else None
        if not self.server_args.model_loaded[self.component_name]:
            loader = VAELoader()
            self.vae, _ = loader.load(
                self.server_args.model_paths[self.component_name],
                self.server_args,
                component_name=self.component_name,
                transformers_or_diffusers=loader.expected_library,
            )
            if pipeline:
                pipeline.add_module(self.component_name, self.vae)
            self.server_args.model_loaded[self.component_name] = True

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

        """
        # load vae if not already loaded (used for memory constrained devices)
        self.load_model()

        vae_dtype = resolve_precision(
            server_args, self.component_name, precision_attr="vae_precision"
        )
        with self.use_declared_component(
            component_name=self.component_name,
            module=self.vae,
        ) as vae:
            assert vae is not None
            self.vae = vae

            frames = self.decode(batch.latents, server_args, vae_dtype=vae_dtype)

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
                all_decoded = self.decode(
                    flat_latents, server_args, vae_dtype=vae_dtype
                )

                # 2. Reshape back
                # Keep on GPU to allow faster vectorized post-processing
                decoded_tensor = all_decoded.view(B, T, *all_decoded.shape[1:])

                # Convert to list of tensors (per timestep) as expected by OutputBatch
                # Each element in list is [B, channels, frames, H_out, W_out]
                trajectory_decoded = [decoded_tensor[:, i] for i in range(T)]
            else:
                trajectory_decoded = None

        frames = server_args.pipeline_config.post_decoding(frames, server_args)

        # Update batch with decoded image
        output_batch = OutputBatch(
            output=frames,
            trajectory_timesteps=batch.trajectory_timesteps,
            trajectory_latents=batch.trajectory_latents,
            rollout_trajectory_data=batch.rollout_trajectory_data,
            trajectory_decoded=trajectory_decoded,
            metrics=batch.metrics,
            noise_pred=None,
        )

        return output_batch
