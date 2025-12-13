# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
Denoising stage for diffusion pipelines.
"""

import inspect
import math
import os
import time
import weakref
from collections.abc import Iterable
from functools import lru_cache
from typing import Any

import torch
from einops import rearrange
from tqdm.auto import tqdm

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType, STA_Mode
from sglang.multimodal_gen.configs.pipeline_configs.wan import Wan2_2_TI2V_5B_Config
from sglang.multimodal_gen.runtime.distributed import (
    cfg_model_parallel_all_reduce,
    get_local_torch_device,
    get_sp_parallel_rank,
    get_sp_world_size,
    get_world_group,
)
from sglang.multimodal_gen.runtime.distributed.communication_op import (
    sequence_model_parallel_all_gather,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_cfg_group,
    get_classifier_free_guidance_rank,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn import (
    FlashAttentionBackend,
)
from sglang.multimodal_gen.runtime.layers.attention.selector import get_attn_backend
from sglang.multimodal_gen.runtime.layers.attention.STA_configuration import (
    configure_sta,
    save_mask_search_results,
)
from sglang.multimodal_gen.runtime.loader.component_loader import TransformerLoader
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    PipelineStage,
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.platforms.interface import AttentionBackendEnum
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.perf_logger import StageProfiler
from sglang.multimodal_gen.runtime.utils.profiler import SGLDiffusionProfiler
from sglang.multimodal_gen.utils import dict_to_3d_list, masks_like

try:
    from sglang.multimodal_gen.runtime.layers.attention.backends.sliding_tile_attn import (
        SlidingTileAttentionBackend,
    )

    st_attn_available = True
except ImportError:
    st_attn_available = False

try:
    from sglang.multimodal_gen.runtime.layers.attention.backends.vmoba import (
        VMOBAAttentionBackend,
    )
    from sglang.multimodal_gen.utils import is_vmoba_available

    vmoba_attn_available = is_vmoba_available()
except ImportError:
    vmoba_attn_available = False

try:
    from sglang.multimodal_gen.runtime.layers.attention.backends.video_sparse_attn import (
        VideoSparseAttentionBackend,
    )

    vsa_available = True
except ImportError:
    vsa_available = False

logger = init_logger(__name__)


class DenoisingStage(PipelineStage):
    """
    Stage for running the denoising loop in diffusion pipelines.

    This stage handles the iterative denoising process that transforms
    the initial noise into the final output.
    """

    def __init__(
        self, transformer, scheduler, pipeline=None, transformer_2=None, vae=None
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.transformer_2 = transformer_2

        hidden_size = self.server_args.pipeline_config.dit_config.hidden_size
        num_attention_heads = (
            self.server_args.pipeline_config.dit_config.num_attention_heads
        )
        attn_head_size = hidden_size // num_attention_heads

        # torch compile
        if self.server_args.enable_torch_compile:
            self.torch_compile_module(self.transformer)
            if transformer_2 is not None:
                self.torch_compile_module(self.transformer_2)

        self.scheduler = scheduler
        self.vae = vae
        self.pipeline = weakref.ref(pipeline) if pipeline else None

        self.attn_backend = get_attn_backend(
            head_size=attn_head_size,
            dtype=torch.float16,  # TODO(will): hack
            supported_attention_backends={
                AttentionBackendEnum.SLIDING_TILE_ATTN,
                AttentionBackendEnum.VIDEO_SPARSE_ATTN,
                AttentionBackendEnum.VMOBA_ATTN,
                AttentionBackendEnum.FA,
                AttentionBackendEnum.TORCH_SDPA,
                AttentionBackendEnum.SAGE_ATTN_3,
            },  # hack
        )

        # cfg
        self.guidance = None

        # misc
        self.profiler = None
        # cache-dit state (for delayed mounting and idempotent control)
        self._cache_dit_enabled = False
        self._cached_num_steps = None

    def torch_compile_module(self, module):
        """
        Compile a module's forward with torch.compile, and enable inductor overlap tweak if available.
        No-op if torch compile is disabled or the object has no forward.
        """
        if not self.server_args.enable_torch_compile or module is None:
            return module
        if not hasattr(module, "forward"):
            return module
        try:
            import torch._inductor.config as _inductor_cfg

            _inductor_cfg.reorder_for_compute_comm_overlap = True
        except ImportError:
            pass
        mode = os.environ.get("SGLANG_TORCH_COMPILE_MODE", "max-autotune-no-cudagraphs")
        logger.info(f"Compiling transformer with mode: {mode}")
        compiled_forward = torch.compile(getattr(module, "forward"), mode=mode)
        setattr(module, "forward", compiled_forward)
        return module

    def _maybe_enable_cache_dit(self, num_inference_steps: int) -> None:
        """Enable cache-dit on the transformers if configured (idempotent).

        This method should be called after the transformer is fully loaded
        and before torch.compile is applied.

        For dual-transformer models (e.g., Wan2.2), this enables cache-dit on both
        transformers with (potentially) different configurations.

        """
        if self._cache_dit_enabled:
            if self._cached_num_steps != num_inference_steps:
                logger.warning(
                    "num_inference_steps changed from %d to %d after cache-dit was enabled. "
                    "Continuing with initial configuration (steps=%d).",
                    self._cached_num_steps,
                    num_inference_steps,
                    self._cached_num_steps,
                )
            return
        # check if cache-dit is enabled in config
        if not envs.SGLANG_CACHE_DIT_ENABLED:
            return

        from sglang.multimodal_gen.runtime.distributed import get_world_size
        from sglang.multimodal_gen.runtime.utils.cache_dit_integration import (
            CacheDitConfig,
            enable_cache_on_dual_transformer,
            enable_cache_on_transformer,
            get_scm_mask,
        )

        if get_world_size() > 1:
            logger.warning(
                "cache-dit is disabled in distributed environment (world_size=%d). "
                "Distributed support will be added in a future version.",
                get_world_size(),
            )
            return
        # === Parse SCM configuration from envs ===
        # SCM is shared between primary and secondary transformers
        scm_preset = envs.SGLANG_CACHE_DIT_SCM_PRESET
        scm_compute_bins_str = envs.SGLANG_CACHE_DIT_SCM_COMPUTE_BINS
        scm_cache_bins_str = envs.SGLANG_CACHE_DIT_SCM_CACHE_BINS
        scm_policy = envs.SGLANG_CACHE_DIT_SCM_POLICY

        # parse custom bins if provided (both must be set together)
        scm_compute_bins = None
        scm_cache_bins = None
        if scm_compute_bins_str and scm_cache_bins_str:
            try:
                scm_compute_bins = [
                    int(x.strip()) for x in scm_compute_bins_str.split(",")
                ]
                scm_cache_bins = [int(x.strip()) for x in scm_cache_bins_str.split(",")]
            except ValueError as e:
                logger.warning("Failed to parse SCM bins: %s. SCM disabled.", e)
                scm_preset = "none"
        elif scm_compute_bins_str or scm_cache_bins_str:
            # Only one of the bins was provided - warn user
            logger.warning(
                "SCM custom bins require both compute_bins and cache_bins. "
                "Only one was provided (compute=%s, cache=%s). Falling back to preset '%s'.",
                scm_compute_bins_str,
                scm_cache_bins_str,
                scm_preset,
            )

        # generate SCM mask using cache-dit's steps_mask()
        # cache-dit handles step count validation and scaling internally
        steps_computation_mask = get_scm_mask(
            preset=scm_preset,
            num_inference_steps=num_inference_steps,
            compute_bins=scm_compute_bins,
            cache_bins=scm_cache_bins,
        )

        # build config for primary transformer (high-noise expert)
        primary_config = CacheDitConfig(
            enabled=True,
            Fn_compute_blocks=envs.SGLANG_CACHE_DIT_FN,
            Bn_compute_blocks=envs.SGLANG_CACHE_DIT_BN,
            max_warmup_steps=envs.SGLANG_CACHE_DIT_WARMUP,
            residual_diff_threshold=envs.SGLANG_CACHE_DIT_RDT,
            max_continuous_cached_steps=envs.SGLANG_CACHE_DIT_MC,
            enable_taylorseer=envs.SGLANG_CACHE_DIT_TAYLORSEER,
            taylorseer_order=envs.SGLANG_CACHE_DIT_TS_ORDER,
            num_inference_steps=num_inference_steps,
            # SCM fields
            steps_computation_mask=steps_computation_mask,
            steps_computation_policy=scm_policy,
        )

        if self.transformer_2 is not None:
            # dual transformer
            # build config for secondary transformer (low-noise expert)
            # uses secondary parameters which inherit from primary if not explicitly set
            secondary_config = CacheDitConfig(
                enabled=True,
                Fn_compute_blocks=envs.SGLANG_CACHE_DIT_SECONDARY_FN,
                Bn_compute_blocks=envs.SGLANG_CACHE_DIT_SECONDARY_BN,
                max_warmup_steps=envs.SGLANG_CACHE_DIT_SECONDARY_WARMUP,
                residual_diff_threshold=envs.SGLANG_CACHE_DIT_SECONDARY_RDT,
                max_continuous_cached_steps=envs.SGLANG_CACHE_DIT_SECONDARY_MC,
                enable_taylorseer=envs.SGLANG_CACHE_DIT_SECONDARY_TAYLORSEER,
                taylorseer_order=envs.SGLANG_CACHE_DIT_SECONDARY_TS_ORDER,
                num_inference_steps=num_inference_steps,
                # SCM fields - shared with primary
                steps_computation_mask=steps_computation_mask,
                steps_computation_policy=scm_policy,
            )

            # for dual transformers, must use BlockAdapter to enable cache on both simultaneously.
            # Don't call enable_cache separately on each transformer.
            self.transformer, self.transformer_2 = enable_cache_on_dual_transformer(
                self.transformer,
                self.transformer_2,
                primary_config,
                secondary_config,
                model_name="wan2.2",
            )
            logger.info(
                "cache-dit enabled on dual transformers (steps=%d)",
                num_inference_steps,
            )
        else:
            # single transformer
            self.transformer = enable_cache_on_transformer(
                self.transformer,
                primary_config,
                model_name="transformer",
            )
            logger.info(
                "cache-dit enabled on transformer (steps=%d, Fn=%d, Bn=%d, rdt=%.3f)",
                num_inference_steps,
                envs.SGLANG_CACHE_DIT_FN,
                envs.SGLANG_CACHE_DIT_BN,
                envs.SGLANG_CACHE_DIT_RDT,
            )

        self._cache_dit_enabled = True
        self._cached_num_steps = num_inference_steps

    @lru_cache(maxsize=8)
    def _build_guidance(self, batch_size, target_dtype, device, guidance_val):
        """Builds a guidance tensor. This method is cached."""
        return (
            torch.full(
                (batch_size,),
                guidance_val,
                dtype=torch.float32,
                device=device,
            ).to(target_dtype)
            * 1000.0
        )

    def get_or_build_guidance(self, bsz: int, dtype, device):
        """
        Get the guidance tensor, using a cached version if available.

        This method retrieves a cached guidance tensor using `_build_guidance`.
        The caching is based on batch size, dtype, device, and the guidance value,
        preventing repeated tensor creation within the denoising loop.
        """
        if self.server_args.pipeline_config.should_use_guidance:
            # TODO: should the guidance_scale be picked-up from sampling_params?
            guidance_val = self.server_args.pipeline_config.embedded_cfg_scale
            return self._build_guidance(bsz, dtype, device, guidance_val)
        else:
            return None

    @property
    def parallelism_type(self) -> StageParallelismType:
        # return StageParallelismType.CFG_PARALLEL if get_global_server_args().enable_cfg_parallel else StageParallelismType.REPLICATED
        return StageParallelismType.REPLICATED

    def _preprocess_latents_for_ti2v(
        self, latents, target_dtype, batch, server_args: ServerArgs
    ):
        # FIXME: should probably move to latent preparation stage, to handle with offload
        # Wan2.2 TI2V directly replaces the first frame of the latent with
        # the image latent instead of appending along the channel dim
        assert batch.image_latent is None, "TI2V task should not have image latents"
        assert self.vae is not None, "VAE is not provided for TI2V task"
        self.vae = self.vae.to(batch.condition_image.device)
        z = self.vae.encode(batch.condition_image).mean.float()
        if self.vae.device != "cpu" and server_args.vae_cpu_offload:
            self.vae = self.vae.to("cpu")
        if hasattr(self.vae, "shift_factor") and self.vae.shift_factor is not None:
            if isinstance(self.vae.shift_factor, torch.Tensor):
                z -= self.vae.shift_factor.to(z.device, z.dtype)
            else:
                z -= self.vae.shift_factor

        if isinstance(self.vae.scaling_factor, torch.Tensor):
            z = z * self.vae.scaling_factor.to(z.device, z.dtype)
        else:
            z = z * self.vae.scaling_factor
        # z: [B, C, 1, H, W]
        latent_model_input = latents.to(target_dtype)
        # Keep as [B, C, T, H, W] for proper broadcasting
        assert latent_model_input.ndim == 5

        # Create mask with proper shape [B, C, T, H, W]
        latent_for_mask = latent_model_input.squeeze(0)  # [C, T, H, W]
        _, reserved_frames_masks = masks_like([latent_for_mask], zero=True)
        reserved_frames_mask = reserved_frames_masks[0].unsqueeze(0)  # [1, C, T, H, W]

        # replace GLOBAL first frame with image - proper broadcasting
        # z: [B, C, 1, H, W], reserved_frames_mask: [1, C, T, H, W]
        # Both will broadcast correctly
        latents = (
            1.0 - reserved_frames_mask
        ) * z + reserved_frames_mask * latent_model_input
        assert latents.ndim == 5
        latents = latents.to(get_local_torch_device())
        batch.latents = latents

        F = batch.num_frames
        temporal_scale = (
            server_args.pipeline_config.vae_config.arch_config.scale_factor_temporal
        )
        spatial_scale = (
            server_args.pipeline_config.vae_config.arch_config.scale_factor_spatial
        )
        patch_size = server_args.pipeline_config.dit_config.arch_config.patch_size
        seq_len = (
            ((F - 1) // temporal_scale + 1)
            * (batch.height // spatial_scale)
            * (batch.width // spatial_scale)
            // (patch_size[1] * patch_size[2])
        )
        seq_len = int(math.ceil(seq_len / get_sp_world_size())) * get_sp_world_size()
        return seq_len, z, reserved_frames_masks

    def _postprocess_latents_for_ti2v(self, z, reserved_frames_masks, batch):
        rank_in_sp_group = get_sp_parallel_rank()
        sp_world_size = get_sp_world_size()

        if getattr(batch, "did_sp_shard_latents", False):
            # Shard z (image latent) along time dimension
            # z shape: [1, C, 1, H, W] - only first frame
            # Only rank 0 has the first frame after sharding
            if z.shape[2] == 1:
                # z is single frame, only rank 0 needs it
                if rank_in_sp_group == 0:
                    z_sp = z
                else:
                    # Other ranks don't have the first frame
                    z_sp = None
            else:
                # Should not happen for TI2V
                z_sp = z

            # Shard reserved_frames_mask along time dimension to match sharded latents
            # reserved_frames_mask is a list from masks_like, extract reserved_frames_mask[0] first
            # reserved_frames_mask[0] shape: [C, T, H, W]
            # All ranks need their portion of reserved_frames_mask for timestep calculation
            if reserved_frames_masks is not None:
                reserved_frames_mask = reserved_frames_masks[
                    0
                ]  # Extract tensor from list
                time_dim = reserved_frames_mask.shape[1]  # [C, T, H, W]
                if time_dim > 0 and time_dim % sp_world_size == 0:
                    reserved_frames_mask_sp_tensor = rearrange(
                        reserved_frames_mask,
                        "c (n t) h w -> c n t h w",
                        n=sp_world_size,
                    ).contiguous()
                    reserved_frames_mask_sp_tensor = reserved_frames_mask_sp_tensor[
                        :, rank_in_sp_group, :, :, :
                    ]
                    reserved_frames_mask_sp = (
                        reserved_frames_mask_sp_tensor  # Store as tensor, not list
                    )
                else:
                    reserved_frames_mask_sp = reserved_frames_mask
            else:
                reserved_frames_mask_sp = None
        else:
            # SP not enabled or latents not sharded
            z_sp = z
            reserved_frames_mask_sp = (
                reserved_frames_masks[0] if reserved_frames_masks is not None else None
            )  # Extract tensor

        return reserved_frames_mask_sp, z_sp

    def _handle_boundary_ratio(
        self,
        server_args,
        batch,
    ):
        """
        (Wan2.2) Calculate timestep to switch from high noise expert to low noise expert
        """
        boundary_ratio = server_args.pipeline_config.dit_config.boundary_ratio
        if batch.boundary_ratio is not None:
            logger.info(
                "Overriding boundary ratio from %s to %s",
                boundary_ratio,
                batch.boundary_ratio,
            )
            boundary_ratio = batch.boundary_ratio

        if boundary_ratio is not None:
            boundary_timestep = boundary_ratio * self.scheduler.num_train_timesteps
        else:
            boundary_timestep = None

        return boundary_timestep

    def _prepare_denoising_loop(self, batch: Req, server_args: ServerArgs):
        """
        Prepare all necessary invariant variables for the denoising loop.

        Args:
            batch: The current batch information.
            server_args: The inference arguments.

        Returns:
            A dictionary containing all the prepared variables for the denoising loop.
        """
        pipeline = self.pipeline() if self.pipeline else None
        if not server_args.model_loaded["transformer"]:
            loader = TransformerLoader()
            self.transformer = loader.load(
                server_args.model_paths["transformer"], server_args
            )

            # enable cache-dit before torch.compile (delayed mounting)
            self._maybe_enable_cache_dit(batch.num_inference_steps)

            if self.server_args.enable_torch_compile:
                self.transformer = torch.compile(
                    self.transformer, mode="max-autotune", fullgraph=True
                )
            if pipeline:
                pipeline.add_module("transformer", self.transformer)
            server_args.model_loaded["transformer"] = True
        else:
            self._maybe_enable_cache_dit(batch.num_inference_steps)

        # Prepare extra step kwargs for scheduler
        extra_step_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler.step,
            {"generator": batch.generator, "eta": batch.eta},
        )

        # Setup precision and autocast settings
        target_dtype = torch.bfloat16
        autocast_enabled = (
            target_dtype != torch.float32
        ) and not server_args.disable_autocast

        # Get timesteps and calculate warmup steps
        timesteps = batch.timesteps
        if timesteps is None:
            raise ValueError("Timesteps must be provided")
        num_inference_steps = batch.num_inference_steps
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        # Prepare image latents and embeddings for I2V generation
        image_embeds = batch.image_embeds
        if len(image_embeds) > 0:
            image_embeds = [
                image_embed.to(target_dtype) for image_embed in image_embeds
            ]

        # Prepare STA parameters
        if st_attn_available and self.attn_backend == SlidingTileAttentionBackend:
            self.prepare_sta_param(batch, server_args)

        # Get latents and embeddings
        latents = batch.latents
        prompt_embeds = batch.prompt_embeds
        # Removed Tensor truthiness assert to avoid GPU sync
        neg_prompt_embeds = None
        if batch.do_classifier_free_guidance:
            neg_prompt_embeds = batch.negative_prompt_embeds
            assert neg_prompt_embeds is not None
            # Removed Tensor truthiness assert to avoid GPU sync

        boundary_timestep = self._handle_boundary_ratio(server_args, batch)

        # specifically for Wan2_2_TI2V_5B_Config, not applicable for FastWan2_2_TI2V_5B_Config
        should_preprocess_for_wan_ti2v = (
            server_args.pipeline_config.task_type == ModelTaskType.TI2V
            and batch.condition_image is not None
            and type(server_args.pipeline_config) is Wan2_2_TI2V_5B_Config
        )

        # TI2V specific preparations - before SP sharding
        if should_preprocess_for_wan_ti2v:
            seq_len, z, reserved_frames_masks = self._preprocess_latents_for_ti2v(
                latents, target_dtype, batch, server_args
            )
        else:
            seq_len, z, reserved_frames_masks = (
                None,
                None,
                None,
            )

        # Handle sequence parallelism after TI2V processing
        self._preprocess_sp_latents(batch, server_args)
        latents = batch.latents

        # Shard z and reserved_frames_mask for TI2V if SP is enabled
        if should_preprocess_for_wan_ti2v:
            reserved_frames_mask_sp, z_sp = self._postprocess_latents_for_ti2v(
                z, reserved_frames_masks, batch
            )
        else:
            reserved_frames_mask_sp, z_sp = (
                reserved_frames_masks[0] if reserved_frames_masks is not None else None
            ), z

        guidance = self.get_or_build_guidance(
            # TODO: replace with raw_latent_shape?
            latents.shape[0],
            latents.dtype,
            latents.device,
        )

        image_kwargs = self.prepare_extra_func_kwargs(
            getattr(self.transformer, "forward", self.transformer),
            {
                # TODO: make sure on-device
                "encoder_hidden_states_image": image_embeds,
                "mask_strategy": dict_to_3d_list(None, t_max=50, l_max=60, h_max=24),
            },
        )

        pos_cond_kwargs = self.prepare_extra_func_kwargs(
            getattr(self.transformer, "forward", self.transformer),
            {
                "encoder_hidden_states_2": batch.clip_embedding_pos,
                "encoder_attention_mask": batch.prompt_attention_mask,
            }
            | server_args.pipeline_config.prepare_pos_cond_kwargs(
                batch,
                self.device,
                getattr(self.transformer, "rotary_emb", None),
                dtype=target_dtype,
            ),
        )

        if batch.do_classifier_free_guidance:
            neg_cond_kwargs = self.prepare_extra_func_kwargs(
                getattr(self.transformer, "forward", self.transformer),
                {
                    "encoder_hidden_states_2": batch.clip_embedding_neg,
                    "encoder_attention_mask": batch.negative_attention_mask,
                }
                | server_args.pipeline_config.prepare_neg_cond_kwargs(
                    batch,
                    self.device,
                    getattr(self.transformer, "rotary_emb", None),
                    dtype=target_dtype,
                ),
            )
        else:
            neg_cond_kwargs = {}

        return {
            "extra_step_kwargs": extra_step_kwargs,
            "target_dtype": target_dtype,
            "autocast_enabled": autocast_enabled,
            "timesteps": timesteps,
            "num_inference_steps": num_inference_steps,
            "num_warmup_steps": num_warmup_steps,
            "image_kwargs": image_kwargs,
            "pos_cond_kwargs": pos_cond_kwargs,
            "neg_cond_kwargs": neg_cond_kwargs,
            "latents": latents,
            "prompt_embeds": prompt_embeds,
            "neg_prompt_embeds": neg_prompt_embeds,
            "boundary_timestep": boundary_timestep,
            "z": z_sp,  # Use SP-sharded version
            # ndim == 5
            "reserved_frames_mask": reserved_frames_mask_sp,  # Use SP-sharded version
            "seq_len": seq_len,
            "guidance": guidance,
        }

    def _post_denoising_loop(
        self,
        batch: Req,
        latents: torch.Tensor,
        trajectory_latents: list,
        trajectory_timesteps: list,
        server_args: ServerArgs,
    ):
        # Gather results if using sequence parallelism
        if trajectory_latents:
            trajectory_tensor = torch.stack(trajectory_latents, dim=1)
            trajectory_timesteps_tensor = torch.stack(trajectory_timesteps, dim=0)
        else:
            trajectory_tensor = None
            trajectory_timesteps_tensor = None

        # Gather results if using sequence parallelism
        latents, trajectory_tensor = self._postprocess_sp_latents(
            batch, latents, trajectory_tensor
        )

        if trajectory_tensor is not None and trajectory_timesteps_tensor is not None:
            batch.trajectory_timesteps = trajectory_timesteps_tensor.cpu()
            batch.trajectory_latents = trajectory_tensor.cpu()

        # Update batch with final latents
        batch.latents = self.server_args.pipeline_config.post_denoising_loop(
            latents, batch
        )

        # Save STA mask search results if needed
        if (
            st_attn_available
            and self.attn_backend == SlidingTileAttentionBackend
            and server_args.STA_mode == STA_Mode.STA_SEARCHING
        ):
            self.save_sta_search_results(batch)

        # deallocate transformer if on mps
        pipeline = self.pipeline() if self.pipeline else None
        if torch.backends.mps.is_available():
            logger.info(
                "Memory before deallocating transformer: %s",
                torch.mps.current_allocated_memory(),
            )
            del self.transformer
            if pipeline is not None and "transformer" in pipeline.modules:
                del pipeline.modules["transformer"]
            server_args.model_loaded["transformer"] = False
            logger.info(
                "Memory after deallocating transformer: %s",
                torch.mps.current_allocated_memory(),
            )

    def _preprocess_sp_latents(self, batch: Req, server_args: ServerArgs):
        """Shard latents for Sequence Parallelism if applicable."""
        if get_sp_world_size() <= 1:
            return

        if batch.latents is not None:
            (
                batch.latents,
                did_shard,
            ) = server_args.pipeline_config.shard_latents_for_sp(batch, batch.latents)
            batch.did_sp_shard_latents = did_shard
        else:
            batch.did_sp_shard_latents = False

        # For I2I tasks like QwenImageEdit, the image_latent (input image) should be
        # replicated on all SP ranks, not sharded, as it provides global context.
        if (
            server_args.pipeline_config.task_type != ModelTaskType.I2I
            and batch.image_latent is not None
        ):
            batch.image_latent, _ = server_args.pipeline_config.shard_latents_for_sp(
                batch, batch.image_latent
            )

    def _postprocess_sp_latents(
        self,
        batch: Req,
        latents: torch.Tensor,
        trajectory_tensor: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Gather latents after Sequence Parallelism if they were sharded."""
        if get_sp_world_size() > 1 and getattr(batch, "did_sp_shard_latents", False):
            latents = self.server_args.pipeline_config.gather_latents_for_sp(latents)
            if trajectory_tensor is not None:
                # trajectory_tensor shapes:
                # - video: [b, num_steps, c, t_local, h, w] -> gather on dim=3
                # - image: [b, num_steps, s_local, d] -> gather on dim=2
                trajectory_tensor = trajectory_tensor.to(get_local_torch_device())
                gather_dim = 3 if trajectory_tensor.dim() >= 5 else 2
                trajectory_tensor = sequence_model_parallel_all_gather(
                    trajectory_tensor, dim=gather_dim
                )
                if gather_dim == 2 and hasattr(batch, "raw_latent_shape"):
                    orig_s = batch.raw_latent_shape[1]
                    if trajectory_tensor.shape[2] > orig_s:
                        trajectory_tensor = trajectory_tensor[:, :, :orig_s, :]
        return latents, trajectory_tensor

    def step_profile(self):
        profiler = SGLDiffusionProfiler.get_instance()
        if profiler:
            profiler.step_denoising_step()

    def _manage_device_placement(
        self,
        model_to_use: torch.nn.Module,
        model_to_offload: torch.nn.Module | None,
        server_args: ServerArgs,
    ):
        """
        Manages the offload / load behavior of dit
        """
        if not server_args.dit_cpu_offload:
            return

        # Offload the unused model if it's on CUDA
        if (
            model_to_offload is not None
            and next(model_to_offload.parameters()).device.type == "cuda"
        ):
            model_to_offload.to("cpu")

        # Load the model to use if it's on CPU
        if (
            model_to_use is not None
            and next(model_to_use.parameters()).device.type == "cpu"
        ):
            model_to_use.to(get_local_torch_device())

    def _select_and_manage_model(
        self,
        t_int: int,
        boundary_timestep: float | None,
        server_args: ServerArgs,
        batch: Req,
    ):
        if boundary_timestep is None or t_int >= boundary_timestep:
            # High-noise stage
            current_model = self.transformer
            model_to_offload = self.transformer_2
            current_guidance_scale = batch.guidance_scale
        else:
            # Low-noise stage
            current_model = self.transformer_2
            model_to_offload = self.transformer
            current_guidance_scale = batch.guidance_scale_2

        self._manage_device_placement(current_model, model_to_offload, server_args)

        assert current_model is not None, "The model for the current step is not set."
        return current_model, current_guidance_scale

    def expand_timestep_before_forward(
        self,
        batch: Req,
        server_args: ServerArgs,
        t_device,
        target_dtype,
        seq_len: int | None,
        reserved_frames_mask,
    ):
        bsz = batch.raw_latent_shape[0]
        should_preprocess_for_wan_ti2v = (
            server_args.pipeline_config.task_type == ModelTaskType.TI2V
            and batch.condition_image is not None
            and type(server_args.pipeline_config) is Wan2_2_TI2V_5B_Config
        )

        # expand timestep
        if should_preprocess_for_wan_ti2v:
            # Explicitly cast t_device to the target float type at the beginning.
            # This ensures any precision-based rounding (e.g., float32(999.0) -> bfloat16(1000.0))
            # is applied consistently *before* it's used by any rank.
            t_device_rounded = t_device.to(target_dtype)

            local_seq_len = seq_len
            if get_sp_world_size() > 1 and getattr(
                batch, "did_sp_shard_latents", False
            ):
                local_seq_len = seq_len // get_sp_world_size()

            if get_sp_parallel_rank() == 0 and reserved_frames_mask is not None:
                # Rank 0 has the first frame, create a special timestep tensor
                # NOTE: The spatial downsampling in the next line is suspicious but kept
                # to match original model's potential training configuration.
                temp_ts = (
                    reserved_frames_mask[0][:, ::2, ::2] * t_device_rounded
                ).flatten()

                # Pad to full local sequence length
                temp_ts = torch.cat(
                    [
                        temp_ts,
                        temp_ts.new_ones(local_seq_len - temp_ts.size(0))
                        * t_device_rounded,
                    ]
                )
                timestep = temp_ts.unsqueeze(0).repeat(bsz, 1)
            else:
                # Other ranks get a uniform timestep tensor of the correct shape [B, local_seq_len]
                timestep = t_device.repeat(bsz, local_seq_len)
        else:
            timestep = t_device.repeat(bsz)
        return timestep

    def post_forward_for_ti2v_task(
        self, batch: Req, server_args: ServerArgs, reserved_frames_mask, latents, z
    ):
        """
        For Wan2.2 ti2v task, global first frame should be replaced with encoded image after each timestep
        """
        should_preprocess_for_wan_ti2v = (
            server_args.pipeline_config.task_type == ModelTaskType.TI2V
            and batch.condition_image is not None
            and type(server_args.pipeline_config) is Wan2_2_TI2V_5B_Config
        )
        if should_preprocess_for_wan_ti2v:
            # Apply TI2V mask blending with SP-aware z and reserved_frames_mask.
            # This ensures the first frame is always the condition image after each step.
            # This is only applied on rank 0, where z is not None.
            if z is not None and reserved_frames_mask is not None:
                # z: [1, C, 1, H, W]
                # latents: [1, C, T_local, H, W]
                # reserved_frames_mask: [C, T_local, H, W]
                # Unsqueeze mask to [1, C, T_local, H, W] for broadcasting.
                # z will broadcast along the time dimension.
                latents = (
                    1.0 - reserved_frames_mask.unsqueeze(0)
                ) * z + reserved_frames_mask.unsqueeze(0) * latents

        return latents

    @torch.no_grad()
    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        """
        Run the denoising loop.

        Args:
            batch: The current batch information.
            server_args: The inference arguments.

        Returns:
            The batch with denoised latents.
        """
        # Prepare variables for the denoising loop

        prepared_vars = self._prepare_denoising_loop(batch, server_args)
        extra_step_kwargs = prepared_vars["extra_step_kwargs"]
        target_dtype = prepared_vars["target_dtype"]
        autocast_enabled = prepared_vars["autocast_enabled"]
        timesteps = prepared_vars["timesteps"]
        num_inference_steps = prepared_vars["num_inference_steps"]
        num_warmup_steps = prepared_vars["num_warmup_steps"]
        image_kwargs = prepared_vars["image_kwargs"]
        pos_cond_kwargs = prepared_vars["pos_cond_kwargs"]
        neg_cond_kwargs = prepared_vars["neg_cond_kwargs"]
        latents = prepared_vars["latents"]
        boundary_timestep = prepared_vars["boundary_timestep"]
        z = prepared_vars["z"]
        reserved_frames_mask = prepared_vars["reserved_frames_mask"]
        seq_len = prepared_vars["seq_len"]
        guidance = prepared_vars["guidance"]

        # Initialize lists for ODE trajectory
        trajectory_timesteps: list[torch.Tensor] = []
        trajectory_latents: list[torch.Tensor] = []

        # Run denoising loop
        denoising_start_time = time.time()

        # to avoid device-sync caused by timestep comparison
        timesteps_cpu = timesteps.cpu()
        num_timesteps = timesteps_cpu.shape[0]
        with torch.autocast(
            device_type=("cuda" if torch.cuda.is_available() else "cpu"),
            dtype=target_dtype,
            enabled=autocast_enabled,
        ):
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t_host in enumerate(timesteps_cpu):
                    # Skip if interrupted
                    if hasattr(self, "interrupt") and self.interrupt:
                        continue

                    with StageProfiler(
                        f"denoising_step_{i}", logger=logger, timings=batch.timings
                    ):
                        t_int = int(t_host.item())
                        t_device = timesteps[i]
                        current_model, current_guidance_scale = (
                            self._select_and_manage_model(
                                t_int=t_int,
                                boundary_timestep=boundary_timestep,
                                server_args=server_args,
                                batch=batch,
                            )
                        )

                        # Expand latents for I2V
                        latent_model_input = latents.to(target_dtype)
                        if batch.image_latent is not None:
                            assert (
                                not server_args.pipeline_config.task_type
                                == ModelTaskType.TI2V
                            ), "image latents should not be provided for TI2V task"
                            latent_model_input = torch.cat(
                                [latent_model_input, batch.image_latent], dim=1
                            ).to(target_dtype)

                        timestep = self.expand_timestep_before_forward(
                            batch,
                            server_args,
                            t_device,
                            target_dtype,
                            seq_len,
                            reserved_frames_mask,
                        )

                        latent_model_input = self.scheduler.scale_model_input(
                            latent_model_input, t_device
                        )

                        # Predict noise residual
                        attn_metadata = self._build_attn_metadata(i, batch, server_args)
                        noise_pred = self._predict_noise_with_cfg(
                            current_model=current_model,
                            latent_model_input=latent_model_input,
                            timestep=timestep,
                            batch=batch,
                            timestep_index=i,
                            attn_metadata=attn_metadata,
                            target_dtype=target_dtype,
                            current_guidance_scale=current_guidance_scale,
                            image_kwargs=image_kwargs,
                            pos_cond_kwargs=pos_cond_kwargs,
                            neg_cond_kwargs=neg_cond_kwargs,
                            server_args=server_args,
                            guidance=guidance,
                            latents=latents,
                        )

                        # Compute the previous noisy sample
                        latents = self.scheduler.step(
                            model_output=noise_pred,
                            timestep=t_device,
                            sample=latents,
                            **extra_step_kwargs,
                            return_dict=False,
                        )[0]

                        latents = self.post_forward_for_ti2v_task(
                            batch, server_args, reserved_frames_mask, latents, z
                        )

                        # save trajectory latents if needed
                        if batch.return_trajectory_latents:
                            trajectory_timesteps.append(t_host)
                            trajectory_latents.append(latents)

                        # Update progress bar
                        if i == num_timesteps - 1 or (
                            (i + 1) > num_warmup_steps
                            and (i + 1) % self.scheduler.order == 0
                            and progress_bar is not None
                        ):
                            progress_bar.update()

                        self.step_profile()

        denoising_end_time = time.time()

        if num_timesteps > 0:
            self.log_info(
                "average time per step: %.4f seconds",
                (denoising_end_time - denoising_start_time) / len(timesteps),
            )

        self._post_denoising_loop(
            batch=batch,
            latents=latents,
            trajectory_latents=trajectory_latents,
            trajectory_timesteps=trajectory_timesteps,
            server_args=server_args,
        )
        return batch

    # TODO: this will extends the preparation stage, should let subclass/passed-in variables decide which to prepare
    def prepare_extra_func_kwargs(self, func, kwargs) -> dict[str, Any]:
        """
        Prepare extra kwargs for the scheduler step / denoise step.

        Args:
            func: The function to prepare kwargs for.
            kwargs: The kwargs to prepare.
        """
        import functools

        # Handle cache-dit's partial wrapping logic.
        # Cache-dit wraps the forward method with functools.partial where args[0] is the instance.
        # We access `_original_forward` if available to inspect the underlying signature.
        # See: https://github.com/vipshop/cache-dit
        if isinstance(func, functools.partial) and func.args:
            func = getattr(func.args[0], "_original_forward", func)

        # Unwrap any decorators (e.g. functools.wraps)
        target_func = inspect.unwrap(func)

        # Filter kwargs based on the signature
        params = inspect.signature(target_func).parameters
        return {k: v for k, v in kwargs.items() if k in params}

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
        if local_rank == 0:
            return tqdm(iterable=iterable, total=total)
        else:
            return tqdm(iterable=iterable, total=total, disable=True)

    def rescale_noise_cfg(
        self, noise_cfg, noise_pred_text, guidance_rescale=0.0
    ) -> torch.Tensor:
        """
        Rescale noise prediction according to guidance_rescale.

        Based on findings of "Common Diffusion Noise Schedules and Sample Steps are Flawed"
        (https://arxiv.org/pdf/2305.08891.pdf), Section 3.4.

        Args:
            noise_cfg: The noise prediction with guidance.
            noise_pred_text: The text-conditioned noise prediction.
            guidance_rescale: The guidance rescale factor.

        Returns:
            The rescaled noise prediction.
        """
        std_text = noise_pred_text.std(
            dim=list(range(1, noise_pred_text.ndim)), keepdim=True
        )
        std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
        # Rescale the results from guidance (fixes overexposure)
        noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
        # Mix with the original results from guidance by factor guidance_rescale
        noise_cfg = (
            guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
        )
        return noise_cfg

    def _build_attn_metadata(
        self, i: int, batch: Req, server_args: ServerArgs
    ) -> Any | None:
        """
        Build attention metadata for custom attention backends.

        Args:
            i: The current timestep index.
            batch: The current batch information.
            server_args: The inference arguments.

        Returns:
            The attention metadata, or None if not applicable.
        """
        attn_metadata = None
        self.attn_metadata_builder_cls = self.attn_backend.get_builder_cls()
        if self.attn_metadata_builder_cls:
            self.attn_metadata_builder = self.attn_metadata_builder_cls()
        if (st_attn_available and self.attn_backend == SlidingTileAttentionBackend) or (
            vsa_available and self.attn_backend == VideoSparseAttentionBackend
        ):
            attn_metadata = self.attn_metadata_builder.build(
                current_timestep=i,
                raw_latent_shape=batch.raw_latent_shape[2:5],
                patch_size=server_args.pipeline_config.dit_config.patch_size,
                STA_param=batch.STA_param,
                VSA_sparsity=server_args.VSA_sparsity,
                device=get_local_torch_device(),
            )
        elif vmoba_attn_available and self.attn_backend == VMOBAAttentionBackend:
            moba_params = server_args.moba_config.copy()
            moba_params.update(
                {
                    "current_timestep": i,
                    "raw_latent_shape": batch.raw_latent_shape[2:5],
                    "patch_size": server_args.pipeline_config.dit_config.patch_size,
                    "device": get_local_torch_device(),
                }
            )
        elif self.attn_backend == FlashAttentionBackend:
            attn_metadata = self.attn_metadata_builder.build(
                raw_latent_shape=batch.raw_latent_shape
            )
        else:
            return None

        assert attn_metadata is not None, "attn_metadata cannot be None"

        return attn_metadata

    def _predict_noise(
        self,
        current_model,
        latent_model_input,
        timestep,
        prompt_embeds,
        target_dtype,
        guidance: torch.Tensor,
        **kwargs,
    ):
        return current_model(
            hidden_states=latent_model_input,
            encoder_hidden_states=prompt_embeds,
            timestep=timestep,
            guidance=guidance,
            **kwargs,
        )

    def _predict_noise_with_cfg(
        self,
        current_model: torch.nn.Module,
        latent_model_input: torch.Tensor,
        timestep,
        batch: Req,
        timestep_index: int,
        attn_metadata,
        target_dtype,
        current_guidance_scale,
        image_kwargs: dict[str, Any],
        pos_cond_kwargs: dict[str, Any],
        neg_cond_kwargs: dict[str, Any],
        server_args,
        guidance,
        latents,
    ):
        """
        Predict the noise residual with classifier-free guidance.

        Args:
            current_model: The transformer model to use for the current step.
            latent_model_input: The input latents for the model.
            timestep: The expanded timestep tensor.
            batch: The current batch information.
            timestep_index: The current timestep index.
            attn_metadata: Attention metadata for custom backends.
            target_dtype: The target data type for autocasting.
            current_guidance_scale: The guidance scale for the current step.
            image_kwargs: Keyword arguments for image conditioning.
            pos_cond_kwargs: Keyword arguments for positive prompt conditioning.
            neg_cond_kwargs: Keyword arguments for negative prompt conditioning.

        Returns:
            The predicted noise.
        """
        noise_pred_cond: torch.Tensor | None = None
        noise_pred_uncond: torch.Tensor | None = None
        cfg_rank = get_classifier_free_guidance_rank()
        # positive pass
        if not (server_args.enable_cfg_parallel and cfg_rank != 0):
            batch.is_cfg_negative = False
            with set_forward_context(
                current_timestep=timestep_index,
                attn_metadata=attn_metadata,
                forward_batch=batch,
            ):
                noise_pred_cond = self._predict_noise(
                    current_model=current_model,
                    latent_model_input=latent_model_input,
                    timestep=timestep,
                    prompt_embeds=server_args.pipeline_config.get_pos_prompt_embeds(
                        batch
                    ),
                    target_dtype=target_dtype,
                    guidance=guidance,
                    **image_kwargs,
                    **pos_cond_kwargs,
                )
                # TODO: can it be moved to after _predict_noise_with_cfg?
                noise_pred_cond = server_args.pipeline_config.slice_noise_pred(
                    noise_pred_cond, latents
                )
        if not batch.do_classifier_free_guidance:
            # If CFG is disabled, we are done. Return the conditional prediction.
            return noise_pred_cond

        # negative pass
        if not server_args.enable_cfg_parallel or cfg_rank != 0:
            batch.is_cfg_negative = True
            with set_forward_context(
                current_timestep=timestep_index,
                attn_metadata=attn_metadata,
                forward_batch=batch,
            ):
                noise_pred_uncond = self._predict_noise(
                    current_model=current_model,
                    latent_model_input=latent_model_input,
                    timestep=timestep,
                    prompt_embeds=server_args.pipeline_config.get_neg_prompt_embeds(
                        batch
                    ),
                    target_dtype=target_dtype,
                    guidance=guidance,
                    **image_kwargs,
                    **neg_cond_kwargs,
                )
                noise_pred_uncond = server_args.pipeline_config.slice_noise_pred(
                    noise_pred_uncond, latents
                )

        # Combine predictions
        if server_args.enable_cfg_parallel:
            # Each rank computes its partial contribution and we sum via all-reduce:
            #   final = s*cond + (1-s)*uncond
            if cfg_rank == 0:
                assert noise_pred_cond is not None
                partial = current_guidance_scale * noise_pred_cond
            else:
                assert noise_pred_uncond is not None
                partial = (1 - current_guidance_scale) * noise_pred_uncond

            noise_pred = cfg_model_parallel_all_reduce(partial)

            # Guidance rescale: broadcast std(cond) from rank 0, compute std(cfg) locally
            if batch.guidance_rescale > 0.0:
                std_cfg = noise_pred.std(
                    dim=list(range(1, noise_pred.ndim)), keepdim=True
                )
                if cfg_rank == 0:
                    assert noise_pred_cond is not None
                    std_text = noise_pred_cond.std(
                        dim=list(range(1, noise_pred_cond.ndim)), keepdim=True
                    )
                else:
                    std_text = torch.empty_like(std_cfg)
                # Broadcast std_text from local src=0 to all ranks in CFG group
                std_text = get_cfg_group().broadcast(std_text, src=0)
                noise_pred_rescaled = noise_pred * (std_text / std_cfg)
                noise_pred = (
                    batch.guidance_rescale * noise_pred_rescaled
                    + (1 - batch.guidance_rescale) * noise_pred
                )
            return noise_pred
        else:
            # Serial CFG: both cond and uncond are available locally
            assert noise_pred_cond is not None and noise_pred_uncond is not None
            noise_pred = noise_pred_uncond + current_guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )

            if batch.guidance_rescale > 0.0:
                noise_pred = self.rescale_noise_cfg(
                    noise_pred,
                    noise_pred_cond,
                    guidance_rescale=batch.guidance_rescale,
                )
            return noise_pred

    def prepare_sta_param(self, batch: Req, server_args: ServerArgs):
        """
        Prepare Sliding Tile Attention (STA) parameters and settings.

        Args:
            batch: The current batch information.
            server_args: The inference arguments.
        """
        # TODO(kevin): STA mask search, currently only support Wan2.1 with 69x768x1280
        STA_mode = server_args.STA_mode
        skip_time_steps = server_args.skip_time_steps
        if batch.timesteps is None:
            raise ValueError("Timesteps must be provided")
        timesteps_num = batch.timesteps.shape[0]

        logger.info("STA_mode: %s", STA_mode)
        if (batch.num_frames, batch.height, batch.width) != (
            69,
            768,
            1280,
        ) and STA_mode != "STA_inference":
            raise NotImplementedError(
                "STA mask search/tuning is not supported for this resolution"
            )

        if (
            STA_mode == STA_Mode.STA_SEARCHING
            or STA_mode == STA_Mode.STA_TUNING
            or STA_mode == STA_Mode.STA_TUNING_CFG
        ):
            size = (batch.width, batch.height)
            if size == (1280, 768):
                # TODO: make it configurable
                sparse_mask_candidates_searching = [
                    "3, 1, 10",
                    "1, 5, 7",
                    "3, 3, 3",
                    "1, 6, 5",
                    "1, 3, 10",
                    "3, 6, 1",
                ]
                sparse_mask_candidates_tuning = [
                    "3, 1, 10",
                    "1, 5, 7",
                    "3, 3, 3",
                    "1, 6, 5",
                    "1, 3, 10",
                    "3, 6, 1",
                ]
                full_mask = ["3,6,10"]
            else:
                raise NotImplementedError(
                    "STA mask search is not supported for this resolution"
                )
        layer_num = self.transformer.config.num_layers
        # specific for HunyuanVideo
        if hasattr(self.transformer.config, "num_single_layers"):
            layer_num += self.transformer.config.num_single_layers
        head_num = self.transformer.config.num_attention_heads

        if STA_mode == STA_Mode.STA_SEARCHING:
            STA_param = configure_sta(
                mode=STA_Mode.STA_SEARCHING,
                layer_num=layer_num,
                head_num=head_num,
                time_step_num=timesteps_num,
                mask_candidates=sparse_mask_candidates_searching + full_mask,
                # last is full mask; Can add more sparse masks while keep last one as full mask
            )
        elif STA_mode == STA_Mode.STA_TUNING:
            STA_param = configure_sta(
                mode=STA_Mode.STA_TUNING,
                layer_num=layer_num,
                head_num=head_num,
                time_step_num=timesteps_num,
                mask_search_files_path=f"output/mask_search_result_pos_{size[0]}x{size[1]}/",
                mask_candidates=sparse_mask_candidates_tuning,
                full_attention_mask=[int(x) for x in full_mask[0].split(",")],
                skip_time_steps=skip_time_steps,  # Use full attention for first 12 steps
                save_dir=f"output/mask_search_strategy_{size[0]}x{size[1]}/",  # Custom save directory
                timesteps=timesteps_num,
            )
        elif STA_mode == STA_Mode.STA_TUNING_CFG:
            STA_param = configure_sta(
                mode=STA_Mode.STA_TUNING_CFG,
                layer_num=layer_num,
                head_num=head_num,
                time_step_num=timesteps_num,
                mask_search_files_path_pos=f"output/mask_search_result_pos_{size[0]}x{size[1]}/",
                mask_search_files_path_neg=f"output/mask_search_result_neg_{size[0]}x{size[1]}/",
                mask_candidates=sparse_mask_candidates_tuning,
                full_attention_mask=[int(x) for x in full_mask[0].split(",")],
                skip_time_steps=skip_time_steps,
                save_dir=f"output/mask_search_strategy_{size[0]}x{size[1]}/",
                timesteps=timesteps_num,
            )
        elif STA_mode == STA_Mode.STA_INFERENCE:
            import sglang.multimodal_gen.envs as envs

            config_file = envs.SGLANG_DIFFUSION_ATTENTION_CONFIG
            if config_file is None:
                raise ValueError("SGLANG_DIFFUSION_ATTENTION_CONFIG is not set")
            STA_param = configure_sta(
                mode=STA_Mode.STA_INFERENCE,
                layer_num=layer_num,
                head_num=head_num,
                time_step_num=timesteps_num,
                load_path=config_file,
            )

        batch.STA_param = STA_param
        batch.mask_search_final_result_pos = [[] for _ in range(timesteps_num)]
        batch.mask_search_final_result_neg = [[] for _ in range(timesteps_num)]

    def save_sta_search_results(self, batch: Req):
        """
        Save the STA mask search results.

        Args:
            batch: The current batch information.
        """
        size = (batch.width, batch.height)
        if size == (1280, 768):
            # TODO: make it configurable
            sparse_mask_candidates_searching = [
                "3, 1, 10",
                "1, 5, 7",
                "3, 3, 3",
                "1, 6, 5",
                "1, 3, 10",
                "3, 6, 1",
            ]
        else:
            raise NotImplementedError(
                "STA mask search is not supported for this resolution"
            )

        if batch.mask_search_final_result_pos is not None and batch.prompt is not None:
            save_mask_search_results(
                [dict(layer_data) for layer_data in batch.mask_search_final_result_pos],
                prompt=str(batch.prompt),
                mask_strategies=sparse_mask_candidates_searching,
                output_dir=f"output/mask_search_result_pos_{size[0]}x{size[1]}/",
            )
        if batch.mask_search_final_result_neg is not None and batch.prompt is not None:
            save_mask_search_results(
                [dict(layer_data) for layer_data in batch.mask_search_final_result_neg],
                prompt=str(batch.prompt),
                mask_strategies=sparse_mask_candidates_searching,
                output_dir=f"output/mask_search_result_neg_{size[0]}x{size[1]}/",
            )

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify denoising stage inputs."""
        result = VerificationResult()
        result.add_check("timesteps", batch.timesteps, [V.is_tensor, V.min_dims(1)])
        # disable temporarily for image-generation models
        # result.add_check("latents", batch.latents, [V.is_tensor, V.with_dims(5)])
        result.add_check("prompt_embeds", batch.prompt_embeds, V.list_not_empty)
        result.add_check("image_embeds", batch.image_embeds, V.is_list)
        # result.add_check(
        #     "image_latent", batch.image_latent, V.none_or_tensor_with_dims(5)
        # )
        result.add_check(
            "num_inference_steps", batch.num_inference_steps, V.positive_int
        )
        result.add_check("guidance_scale", batch.guidance_scale, V.non_negative_float)
        result.add_check("eta", batch.eta, V.non_negative_float)
        result.add_check("generator", batch.generator, V.generator_or_list_generators)
        result.add_check(
            "do_classifier_free_guidance",
            batch.do_classifier_free_guidance,
            V.bool_value,
        )
        result.add_check(
            "negative_prompt_embeds",
            batch.negative_prompt_embeds,
            lambda x: not batch.do_classifier_free_guidance or V.list_not_empty(x),
        )
        return result

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify denoising stage outputs."""
        result = VerificationResult()
        # result.add_check("latents", batch.latents, [V.is_tensor, V.with_dims(5)])
        return result
