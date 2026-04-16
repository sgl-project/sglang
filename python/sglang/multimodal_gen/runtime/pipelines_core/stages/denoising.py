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
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, fields
from functools import lru_cache
from typing import Any

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from sglang.jit_kernel.nvfp4 import prewarm_nvfp4_jit_modules
from sglang.multimodal_gen import envs
from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType, STA_Mode
from sglang.multimodal_gen.configs.pipeline_configs.flux import (
    Flux2PipelineConfig,
    FluxPipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.zimage import ZImagePipelineConfig
from sglang.multimodal_gen.runtime.cache.cache_dit_integration import (
    CacheDitConfig,
    enable_cache_on_dual_transformer,
    enable_cache_on_transformer,
    get_scm_mask,
    refresh_context_on_dual_transformer,
    refresh_context_on_transformer,
)
from sglang.multimodal_gen.runtime.distributed import (
    cfg_model_parallel_all_reduce,
    get_local_torch_device,
    get_sp_group,
    get_sp_world_size,
    get_tp_group,
    get_world_group,
    get_world_size,
)
from sglang.multimodal_gen.runtime.distributed.communication_op import (
    sequence_model_parallel_all_gather,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_cfg_group,
    get_classifier_free_guidance_rank,
)
from sglang.multimodal_gen.runtime.layers.attention.selector import get_attn_backend
from sglang.multimodal_gen.runtime.layers.attention.STA_configuration import (
    configure_sta,
    save_mask_search_results,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.transformer_loader import (
    TransformerLoader,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    PipelineStage,
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.wan_ti2v import (
    blend_wan_ti2v_latents,
    expand_wan_ti2v_timestep,
    prepare_wan_ti2v_latents,
    prepare_wan_ti2v_sp_inputs,
    should_apply_wan_ti2v,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.platforms import (
    AttentionBackendEnum,
    current_platform,
)
from sglang.multimodal_gen.runtime.post_training.rollout_denoising_mixin import (
    RolloutDenoisingMixin,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.layerwise_offload import OffloadableDiTMixin
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.perf_logger import StageProfiler
from sglang.multimodal_gen.runtime.utils.profiler import SGLDiffusionProfiler
from sglang.multimodal_gen.utils import dict_to_3d_list
from sglang.srt.utils.common import get_compiler_backend

logger = init_logger(__name__)


@dataclass(slots=True)
class DenoisingContext:
    """Loop-scoped state shared across the denoising skeleton and its hooks."""

    extra_step_kwargs: dict[str, Any]
    target_dtype: torch.dtype
    autocast_enabled: bool
    timesteps: torch.Tensor
    num_inference_steps: int
    num_warmup_steps: int
    image_kwargs: dict[str, Any]
    pos_cond_kwargs: dict[str, Any]
    neg_cond_kwargs: dict[str, Any]
    latents: torch.Tensor
    boundary_timestep: float | None
    z: torch.Tensor | None
    reserved_frames_mask: torch.Tensor | None
    seq_len: int | None
    guidance: torch.Tensor
    is_warmup: bool
    trajectory_timesteps: list[torch.Tensor] = field(default_factory=list)
    trajectory_latents: list[torch.Tensor] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_kwargs(self) -> dict[str, Any]:
        """Return a shallow field mapping for derived context construction."""
        return {item.name: getattr(self, item.name) for item in fields(self)}


@dataclass(slots=True)
class DenoisingStepState:
    """Per-step hot-path state computed once and reused within a denoising step."""

    step_index: int
    t_host: torch.Tensor
    t_device: torch.Tensor
    t_int: int
    current_model: Any
    current_guidance_scale: Any
    attn_metadata: Any | None


class DenoisingStage(PipelineStage, RolloutDenoisingMixin):
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
        for transformer in filter(None, [self.transformer, self.transformer_2]):
            self._maybe_enable_torch_compile(transformer)

        self.scheduler = scheduler
        self.vae = vae
        self.pipeline = weakref.ref(pipeline) if pipeline else None

        # TODO(will): hack, should use the actual one in dit
        self.attn_backend = get_attn_backend(
            head_size=attn_head_size,
            dtype=torch.float16,
        )

        # cfg
        self.guidance = None

        # misc
        self.profiler = None
        # cache-dit state (for delayed mounting and idempotent control)
        self._cache_dit_enabled = False
        self._cached_num_steps = None
        self._is_warmed_up = False

    def _maybe_enable_torch_compile(self, module: object) -> None:
        """
        Compile a module with torch.compile, and enable inductor overlap tweak if available.
        No-op if torch compile is disabled or the object is not a nn.Module.
        """
        if not self.server_args.enable_torch_compile or not isinstance(
            module, nn.Module
        ):
            return
        compile_kwargs: dict[str, Any] = {"fullgraph": False, "dynamic": None}

        if current_platform.is_npu():
            backend = get_compiler_backend()
            compile_kwargs["backend"] = backend
            compile_kwargs["dynamic"] = False
            logger.info("Compiling transformer with torchair backend on NPU")
        else:
            try:
                import torch._inductor.config as _inductor_cfg

                _inductor_cfg.reorder_for_compute_comm_overlap = True
            except ImportError:
                pass
            mode = os.environ.get(
                "SGLANG_TORCH_COMPILE_MODE", "max-autotune-no-cudagraphs"
            )
            compile_kwargs["mode"] = mode
            logger.info(f"Compiling transformer with mode: {mode}")

        if self._needs_nvfp4_jit_prewarm(module):
            logger.info(
                "Prewarming NVFP4 JIT modules before torch.compile to avoid "
                "Dynamo tracing JIT initialization."
            )
            prewarm_nvfp4_jit_modules()

        # TODO(triple-mu): support customized fullgraph and dynamic in the future
        module.compile(**compile_kwargs)

    @staticmethod
    def _needs_nvfp4_jit_prewarm(module: nn.Module) -> bool:
        for submodule in module.modules():
            quant_method = getattr(submodule, "quant_method", None)
            if quant_method is None:
                continue
            if type(quant_method).__name__ == "ModelOptFp4LinearMethod":
                return True
        return False

    def _maybe_enable_cache_dit(
        self, num_inference_steps: int | tuple[int, int], batch: Req
    ) -> None:
        """Enable cache-dit on the transformers if configured (idempotent).

        This method should be called after the transformer is fully loaded
        and before torch.compile is applied.

        For dual-transformer models (e.g., Wan2.2), this enables cache-dit on both
        transformers with (potentially) different configurations.

        """
        if isinstance(num_inference_steps, tuple):
            num_high_noise_steps, num_low_noise_steps = num_inference_steps

        # NOTE: When a new request arrives, we need to refresh the cache-dit context.
        if self._cache_dit_enabled:
            scm_preset = envs.SGLANG_CACHE_DIT_SCM_PRESET
            scm_preset = None if scm_preset == "none" else scm_preset
            if isinstance(num_inference_steps, tuple):
                refresh_context_on_dual_transformer(
                    self.transformer,
                    self.transformer_2,
                    num_high_noise_steps,
                    num_low_noise_steps,
                    scm_preset=scm_preset,
                )
            else:
                refresh_context_on_transformer(
                    self.transformer,
                    num_inference_steps,
                    scm_preset=scm_preset,
                )
            return

        # check if cache-dit is enabled in config
        if not envs.SGLANG_CACHE_DIT_ENABLED or batch.is_warmup:
            return

        world_size = get_world_size()
        parallelized = world_size > 1

        sp_group = None
        tp_group = None
        if parallelized:
            sp_group_candidate = get_sp_group()
            tp_group_candidate = get_tp_group()

            sp_world_size = sp_group_candidate.world_size if sp_group_candidate else 1
            tp_world_size = tp_group_candidate.world_size if tp_group_candidate else 1

            has_sp = sp_world_size > 1
            has_tp = tp_world_size > 1

            sp_group = sp_group_candidate.device_group if has_sp else None
            tp_group = tp_group_candidate.device_group if has_tp else None

            logger.info(
                "cache-dit enabled in distributed environment (world_size=%d, has_sp=%s, has_tp=%s)",
                world_size,
                has_sp,
                has_tp,
            )
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
            num_inference_steps=(
                num_inference_steps
                if isinstance(num_inference_steps, int)
                else num_high_noise_steps
            ),
            compute_bins=scm_compute_bins,
            cache_bins=scm_cache_bins,
        )

        if isinstance(num_inference_steps, tuple):
            steps_computation_mask_2 = get_scm_mask(
                preset=scm_preset,
                num_inference_steps=num_low_noise_steps,
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
            num_inference_steps=(
                num_inference_steps
                if isinstance(num_inference_steps, int)
                else num_high_noise_steps
            ),
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
                num_inference_steps=num_low_noise_steps,
                # SCM fields - shared with primary
                steps_computation_mask=steps_computation_mask_2,
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
                sp_group=sp_group,
                tp_group=tp_group,
            )
            logger.info(
                "cache-dit enabled on dual transformers (steps=%d, %d)",
                num_high_noise_steps,
                num_low_noise_steps,
            )
        else:
            # single transformer
            self.transformer = enable_cache_on_transformer(
                self.transformer,
                primary_config,
                model_name="transformer",
                sp_group=sp_group,
                tp_group=tp_group,
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
        if isinstance(
            self.server_args.pipeline_config, FluxPipelineConfig
        ) and not isinstance(self.server_args.pipeline_config, Flux2PipelineConfig):
            guidance_val = guidance_val * 1000.0
        return torch.full(
            (batch_size,),
            guidance_val,
            dtype=target_dtype,
            device=device,
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

        Returns:
            A context object containing the invariant state for the denoising loop.
        """
        assert self.transformer is not None
        pipeline = self.pipeline() if self.pipeline else None

        boundary_timestep = self._handle_boundary_ratio(server_args, batch)
        # Get timesteps and calculate warmup steps
        timesteps = batch.timesteps
        num_inference_steps = batch.num_inference_steps
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        if self.transformer_2 is not None:
            assert boundary_timestep is not None, "boundary_timestep must be provided"
            num_high_noise_steps = (timesteps >= boundary_timestep).sum().item()
            num_low_noise_steps = num_inference_steps - num_high_noise_steps
            cache_dit_num_inference_steps = (num_high_noise_steps, num_low_noise_steps)
        else:
            cache_dit_num_inference_steps = num_inference_steps

        if not server_args.model_loaded["transformer"]:
            # FIXME: reuse more code
            loader = TransformerLoader()
            self.transformer = loader.load(
                server_args.model_paths["transformer"], server_args, "transformer"
            )
            # enable cache-dit before torch.compile (delayed mounting)
            self._maybe_enable_cache_dit(cache_dit_num_inference_steps, batch)
            self._maybe_enable_torch_compile(self.transformer)
            if pipeline:
                pipeline.add_module("transformer", self.transformer)
            server_args.model_loaded["transformer"] = True
        else:
            self._maybe_enable_cache_dit(cache_dit_num_inference_steps, batch)

        if batch.rollout:
            self._maybe_prepare_rollout(batch)

        # Prepare extra step kwargs for scheduler
        extra_step_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler.step,
            {"generator": batch.generator, "eta": batch.eta, "batch": batch},
        )

        # Setup precision and autocast settings
        target_dtype = torch.bfloat16
        autocast_enabled = (
            target_dtype != torch.float32
        ) and not server_args.disable_autocast

        # Prepare image latents and embeddings for I2V generation
        image_embeds = batch.image_embeds
        if len(image_embeds) > 0:
            image_embeds = [
                image_embed.to(target_dtype) for image_embed in image_embeds
            ]

        # Prepare STA parameters
        if self.attn_backend.get_enum() == AttentionBackendEnum.SLIDING_TILE_ATTN:
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

        should_preprocess_for_wan_ti2v = should_apply_wan_ti2v(batch, server_args)

        # TI2V specific preparations - before SP sharding
        if should_preprocess_for_wan_ti2v:
            seq_len, z, reserved_frames_masks = prepare_wan_ti2v_latents(
                self.vae,
                latents,
                target_dtype,
                batch,
                server_args,
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
            reserved_frames_mask_sp, z_sp = prepare_wan_ti2v_sp_inputs(
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
            )
            | dict(
                encoder_hidden_states=server_args.pipeline_config.get_pos_prompt_embeds(
                    batch
                )
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
                )
                | dict(
                    encoder_hidden_states=server_args.pipeline_config.get_neg_prompt_embeds(
                        batch
                    )
                ),
            )
        else:
            neg_cond_kwargs = {}

        return DenoisingContext(
            extra_step_kwargs=extra_step_kwargs,
            target_dtype=target_dtype,
            autocast_enabled=autocast_enabled,
            timesteps=timesteps,
            num_inference_steps=num_inference_steps,
            num_warmup_steps=num_warmup_steps,
            image_kwargs=image_kwargs,
            pos_cond_kwargs=pos_cond_kwargs,
            neg_cond_kwargs=neg_cond_kwargs,
            latents=latents,
            boundary_timestep=boundary_timestep,
            z=z_sp,
            reserved_frames_mask=reserved_frames_mask_sp,
            seq_len=seq_len,
            guidance=guidance,
            is_warmup=batch.is_warmup,
        )

    def _before_denoising_loop(
        self, ctx: DenoisingContext, batch: Req, server_args: ServerArgs
    ) -> None:
        """Prepare scheduler state before entering the shared denoising loop."""
        self.scheduler.set_begin_index(0)

    def _prepare_step_state(
        self,
        ctx: DenoisingContext,
        batch: Req,
        server_args: ServerArgs,
        step_index: int,
        t_host: torch.Tensor,
        timesteps_cpu: torch.Tensor,
    ) -> DenoisingStepState:
        """Build the per-step state shared by the loop and model-specific hooks."""
        t_int = int(t_host.item())
        t_device = ctx.timesteps[step_index]
        current_model, current_guidance_scale = self._select_and_manage_model(
            t_int=t_int,
            boundary_timestep=ctx.boundary_timestep,
            server_args=server_args,
            batch=batch,
        )
        attn_metadata = self._prepare_step_attn_metadata(
            ctx=ctx,
            batch=batch,
            server_args=server_args,
            step_index=step_index,
            t_int=t_int,
            timesteps_cpu=timesteps_cpu,
        )
        return DenoisingStepState(
            step_index=step_index,
            t_host=t_host,
            t_device=t_device,
            t_int=t_int,
            current_model=current_model,
            current_guidance_scale=current_guidance_scale,
            attn_metadata=attn_metadata,
        )

    def _prepare_step_attn_metadata(
        self,
        ctx: DenoisingContext,
        batch: Req,
        server_args: ServerArgs,
        step_index: int,
        t_int: int,
        timesteps_cpu: torch.Tensor,
    ) -> Any | None:
        """Build attention metadata for the current denoising step."""
        # Keep attention metadata preparation overridable so model-specific stages
        # can preserve their original semantics without duplicating step state setup.
        return self._build_attn_metadata(
            step_index,
            batch,
            server_args,
            timestep_value=t_int,
            timesteps=timesteps_cpu,
        )

    def _get_prompt_embeds_validator(
        self, batch: Req
    ) -> Callable[[Any], bool] | list[Callable[[Any], bool]]:
        """Return the prompt-embedding validator used by verify_input."""
        del batch
        return V.list_not_empty

    def _get_negative_prompt_embeds_validator(
        self, batch: Req
    ) -> Callable[[Any], bool] | list[Callable[[Any], bool]]:
        """Return the negative-prompt validator used by verify_input."""
        return lambda x: not batch.do_classifier_free_guidance or V.list_not_empty(x)

    def _run_denoising_step(
        self,
        ctx: DenoisingContext,
        step: DenoisingStepState,
        batch: Req,
        server_args: ServerArgs,
    ) -> None:
        """Run one scheduler-backed denoising step in the shared base path.

        Model-specific stages should override this instead of the whole loop whenever possible to achieve better performance
        """
        # 1. Prepare latent inputs in the model's compute dtype.
        latent_model_input = ctx.latents.to(ctx.target_dtype)
        if batch.image_latent is not None:
            assert (
                not server_args.pipeline_config.task_type == ModelTaskType.TI2V
            ), "image latents should not be provided for TI2V task"
            latent_model_input = torch.cat(
                [latent_model_input, batch.image_latent], dim=1
            ).to(ctx.target_dtype)

        # 2. Expand the timestep to the shape expected by the current model.
        timestep = self.expand_timestep_before_forward(
            batch,
            server_args,
            step.t_device,
            ctx.target_dtype,
            ctx.seq_len,
            ctx.reserved_frames_mask,
        )

        # 3. Apply scheduler-side input scaling before the model forward.
        latent_model_input = self.scheduler.scale_model_input(
            latent_model_input, step.t_device
        )

        # 4. Run the model prediction path, including CFG when enabled.
        noise_pred = self._predict_noise_with_cfg(
            current_model=step.current_model,
            latent_model_input=latent_model_input,
            timestep=timestep,
            batch=batch,
            timestep_index=step.step_index,
            attn_metadata=step.attn_metadata,
            target_dtype=ctx.target_dtype,
            current_guidance_scale=step.current_guidance_scale,
            image_kwargs=ctx.image_kwargs,
            pos_cond_kwargs=ctx.pos_cond_kwargs,
            neg_cond_kwargs=ctx.neg_cond_kwargs,
            server_args=server_args,
            guidance=ctx.guidance,
            latents=ctx.latents,
        )
        if server_args.comfyui_mode:
            batch.noise_pred = noise_pred

        # 5. Advance the scheduler state with the predicted noise.
        ctx.latents = self.scheduler.step(
            model_output=noise_pred,
            timestep=step.t_device,
            sample=ctx.latents,
            **ctx.extra_step_kwargs,
            return_dict=False,
        )[0]

        # 6. Re-apply any model-specific latent constraints after the update.
        ctx.latents = self.post_forward_for_ti2v_task(
            batch,
            server_args,
            ctx.reserved_frames_mask,
            ctx.latents,
            ctx.z,
        )

    def _record_trajectory(
        self,
        ctx: DenoisingContext,
        step: DenoisingStepState,
        batch: Req,
        server_args: ServerArgs,
    ) -> None:
        """Append the current step to the returned latent trajectory, if requested."""
        if not batch.return_trajectory_latents:
            return
        ctx.trajectory_timesteps.append(step.t_host)
        ctx.trajectory_latents.append(ctx.latents)

    def _finalize_denoising_loop(
        self, ctx: DenoisingContext, batch: Req, server_args: ServerArgs
    ) -> None:
        """Finalize the shared loop by handing state to post-denoising processing."""
        self._post_denoising_loop(
            batch=batch,
            latents=ctx.latents,
            trajectory_latents=ctx.trajectory_latents,
            trajectory_timesteps=ctx.trajectory_timesteps,
            server_args=server_args,
            is_warmup=ctx.is_warmup,
        )

    def _post_denoising_loop(
        self,
        batch: Req,
        latents: torch.Tensor,
        trajectory_latents: list,
        trajectory_timesteps: list,
        server_args: ServerArgs,
        is_warmup: bool = False,
        *args,
        **kwargs,
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

        # Gather noise_pred if using sequence parallelism
        # noise_pred has the same shape as latents (sharded along sequence dimension)
        if (
            get_sp_world_size() > 1
            and getattr(batch, "did_sp_shard_latents", False)
            and server_args.comfyui_mode
            and hasattr(batch, "noise_pred")
            and batch.noise_pred is not None
        ):
            batch.noise_pred = server_args.pipeline_config.gather_noise_pred_for_sp(
                batch, batch.noise_pred
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
            not is_warmup
            and self.attn_backend.get_enum() == AttentionBackendEnum.SLIDING_TILE_ATTN
            and server_args.attention_backend_config.STA_mode == "STA_SEARCHING"
        ):
            self.save_sta_search_results(batch)

        # Capture references before potential deletion on MPS
        dits = list(filter(None, [self.transformer, self.transformer_2]))

        # deallocate transformer if on mps
        pipeline = self.pipeline() if self.pipeline else None
        if torch.backends.mps.is_available() and not is_warmup:
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

        # reset offload managers with prefetching first layer for next forward
        for dit in dits:
            if isinstance(dit, OffloadableDiTMixin):
                # release all DiT weights to avoid peak VRAM usage, which may increasing the latency for next req
                # TODO: should be make this an option?
                for manager in dit.layerwise_offload_managers:
                    manager.release_all()

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

        # image_latent must be sharded consistently with latents when it is
        # concatenated along the sequence dimension in the denoising loop.
        if batch.image_latent is not None:
            sp_video_metadata = {
                name: getattr(batch, name)
                for name in (
                    "sp_video_latent_num_frames",
                    "sp_video_start_frame",
                    "sp_video_tokens_per_frame",
                    "sp_video_valid_token_count",
                )
                if hasattr(batch, name)
            }
            batch.image_latent, _ = server_args.pipeline_config.shard_latents_for_sp(
                batch, batch.image_latent
            )
            for name, value in sp_video_metadata.items():
                setattr(batch, name, value)

    def _postprocess_sp_latents(
        self,
        batch: Req,
        latents: torch.Tensor,
        trajectory_tensor: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Gather latents after Sequence Parallelism if they were sharded."""
        if get_sp_world_size() > 1 and getattr(batch, "did_sp_shard_latents", False):
            latents = self.server_args.pipeline_config.gather_latents_for_sp(
                latents, batch=batch
            )
            if trajectory_tensor is not None:
                # trajectory_tensor shapes:
                # - video: [b, num_steps, c, t_local, h, w] -> gather on dim=3
                # - image: [b, num_steps, s_local, d] -> gather on dim=2
                trajectory_tensor = trajectory_tensor.to(get_local_torch_device())
                if isinstance(self.server_args.pipeline_config, ZImagePipelineConfig):
                    trajectory_tensor = (
                        self.server_args.pipeline_config.gather_latents_for_sp(
                            trajectory_tensor, batch=batch
                        )
                    )
                else:
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
        model_to_use: nn.Module,
        model_to_offload: nn.Module | None,
        server_args: ServerArgs,
    ):
        """
        Manages the offload / load behavior of dit
        """
        if not server_args.dit_cpu_offload:
            return

        # FSDP manages offloading internally
        if server_args.use_fsdp_inference:
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
        should_preprocess_for_wan_ti2v = should_apply_wan_ti2v(batch, server_args)

        # expand timestep
        if should_preprocess_for_wan_ti2v:
            assert seq_len is not None, "Wan TI2V requires a token sequence length."
            timestep = expand_wan_ti2v_timestep(
                batch,
                t_device,
                target_dtype,
                seq_len,
                reserved_frames_mask,
            )
        else:
            timestep = t_device.repeat(bsz)
        return timestep

    def post_forward_for_ti2v_task(
        self, batch: Req, server_args: ServerArgs, reserved_frames_mask, latents, z
    ):
        """Re-apply Wan TI2V first-frame conditioning after each denoising step."""
        should_preprocess_for_wan_ti2v = should_apply_wan_ti2v(batch, server_args)
        if should_preprocess_for_wan_ti2v:
            latents = blend_wan_ti2v_latents(latents, reserved_frames_mask, z)

        return latents

    @torch.no_grad()
    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        """
        Run the denoising loop.
        """
        ctx = self._prepare_denoising_loop(batch, server_args)
        if batch.rollout:
            self._maybe_init_denoising_env_collection(
                batch=batch,
                pipeline_config=server_args.pipeline_config,
                image_kwargs=ctx.image_kwargs,
                pos_cond_kwargs=ctx.pos_cond_kwargs,
                neg_cond_kwargs=ctx.neg_cond_kwargs,
                guidance=ctx.guidance,
            )
        denoising_start_time = time.time()
        self._before_denoising_loop(ctx, batch, server_args)
        # to avoid device-sync caused by timestep comparison
        timesteps_cpu = ctx.timesteps.cpu()
        num_timesteps = timesteps_cpu.shape[0]
        with torch.autocast(
            device_type=current_platform.device_type,
            dtype=ctx.target_dtype,
            enabled=ctx.autocast_enabled,
        ):
            with self.progress_bar(total=ctx.num_inference_steps) as progress_bar:
                for step_index, t_host in enumerate(timesteps_cpu):
                    with StageProfiler(
                        f"denoising_step_{step_index}",
                        logger=logger,
                        metrics=batch.metrics,
                        perf_dump_path_provided=batch.perf_dump_path is not None,
                        record_as_step=True,
                    ):
                        step = self._prepare_step_state(
                            ctx,
                            batch,
                            server_args,
                            step_index,
                            t_host,
                            timesteps_cpu,
                        )
                        # Capture the raw (pre-scale, pre-I2V-concat) noisy latent
                        # x_{t_i} for rollout trajectory collection. Must run
                        # BEFORE _run_denoising_step so ctx.latents is still the
                        # pre-step value. Gated on batch.rollout to keep the
                        # non-rollout path strictly untouched.
                        if batch.rollout:
                            self._maybe_append_dit_trajectory_step(
                                batch=batch,
                                latents=ctx.latents,
                                timestep_value=step.t_host,
                            )
                        self._run_denoising_step(ctx, step, batch, server_args)
                        self._record_trajectory(ctx, step, batch, server_args)

                        if step_index == num_timesteps - 1 or (
                            (step_index + 1) > ctx.num_warmup_steps
                            and (step_index + 1) % self.scheduler.order == 0
                            and progress_bar is not None
                        ):
                            progress_bar.update()

                        if not ctx.is_warmup:
                            self.step_profile()

        denoising_end_time = time.time()

        if num_timesteps > 0 and not ctx.is_warmup:
            self.log_info(
                "average time per step: %.4f seconds",
                (denoising_end_time - denoising_start_time) / len(ctx.timesteps),
            )

        # Rollout postprocessing must run BEFORE _finalize_denoising_loop so
        # the final scheduler.step output (ctx.latents) is still SP-sharded and
        # can be gathered uniformly alongside the per-step dit_trajectory via
        # gather_stacked_latents_for_sp.
        if batch.rollout:
            self._postprocess_rollout_outputs(
                batch=batch,
                latents=ctx.latents,
                server_args=server_args,
            )
        self._finalize_denoising_loop(ctx, batch, server_args)
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
        """
        local_rank = get_world_group().local_rank
        disable = local_rank != 0
        return tqdm(iterable=iterable, total=total, disable=disable)

    def _rescale_noise_cfg(
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

    def _apply_cfg_normalization(
        self,
        noise_pred: torch.Tensor,
        noise_pred_cond: torch.Tensor,
        cfg_normalization: float,
    ) -> torch.Tensor:
        factor = float(cfg_normalization)
        cond_f = noise_pred_cond.float()
        pred_f = noise_pred.float()
        ori_norm = torch.linalg.vector_norm(cond_f)
        new_norm = torch.linalg.vector_norm(pred_f)
        max_norm = ori_norm * factor

        if new_norm > max_norm:
            noise_pred = noise_pred * (max_norm / new_norm)
        return noise_pred

    def _apply_cfg_normalization_parallel(
        self,
        noise_pred: torch.Tensor,
        noise_pred_cond: torch.Tensor | None,
        cfg_normalization: float,
        cfg_rank: int,
    ) -> torch.Tensor:
        # In cfg-parallel mode, only rank 0 has the conditional branch locally,
        # so the reference norm has to be broadcast to the other ranks
        factor = float(cfg_normalization)
        pred_f = noise_pred.float()
        new_norm = torch.linalg.vector_norm(pred_f)
        if cfg_rank == 0:
            assert noise_pred_cond is not None
            ori_norm = torch.linalg.vector_norm(noise_pred_cond.float())
        else:
            ori_norm = torch.empty_like(new_norm)
        ori_norm = get_cfg_group().broadcast(ori_norm, src=0)
        max_norm = ori_norm * factor

        if new_norm > max_norm:
            noise_pred = noise_pred * (max_norm / new_norm)
        return noise_pred

    def _apply_guidance_rescale_parallel(
        self,
        noise_pred: torch.Tensor,
        noise_pred_cond: torch.Tensor | None,
        guidance_rescale: float,
        cfg_rank: int,
    ) -> torch.Tensor:
        # Guidance rescale is still defined against the conditional branch, so
        # cfg-parallel needs to broadcast that statistic to every rank
        std_cfg = noise_pred.std(dim=list(range(1, noise_pred.ndim)), keepdim=True)
        if cfg_rank == 0:
            assert noise_pred_cond is not None
            std_text = noise_pred_cond.std(
                dim=list(range(1, noise_pred_cond.ndim)), keepdim=True
            )
        else:
            std_text = torch.empty_like(std_cfg)
        std_text = get_cfg_group().broadcast(std_text, src=0)
        noise_pred_rescaled = noise_pred * (std_text / std_cfg)
        return (
            guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_pred
        )

    def _apply_model_specific_cfg_postprocess(
        self,
        batch: Req,
        noise_pred: torch.Tensor,
        noise_pred_cond: torch.Tensor | None,
        cfg_rank: int,
    ) -> torch.Tensor:
        # keep model-specific CFG behavior out of the main denoising loop
        # for cfg-parallel, broadcast cond noise first so the hook sees the same
        # inputs as the serial path.
        if cfg_rank == 0:
            assert noise_pred_cond is not None
            cond_noise = noise_pred_cond
        else:
            # TODO: cache this?
            cond_noise = torch.empty_like(noise_pred)
        cond_noise = get_cfg_group().broadcast(cond_noise, src=0)

        # qwen-image uses true_cfg_scale, match the per-token norm back to the conditional branch
        return self.server_args.pipeline_config.postprocess_cfg_noise(
            batch, noise_pred, cond_noise
        )

    def _combine_cfg_parallel(
        self,
        batch: Req,
        noise_pred_cond: torch.Tensor | None,
        noise_pred_uncond: torch.Tensor | None,
        cfg_scale: float,
        cfg_rank: int,
    ) -> torch.Tensor:
        # cfg-parallel splits cond / uncond across ranks and reconstructs the
        # final CFG result with an all-reduce.
        if cfg_rank == 0:
            assert noise_pred_cond is not None
            partial = cfg_scale * noise_pred_cond
        else:
            assert noise_pred_uncond is not None
            partial = (1 - cfg_scale) * noise_pred_uncond

        noise_pred = cfg_model_parallel_all_reduce(partial)

        if batch.cfg_normalization and float(batch.cfg_normalization) > 0:
            noise_pred = self._apply_cfg_normalization_parallel(
                noise_pred,
                noise_pred_cond,
                batch.cfg_normalization,
                cfg_rank,
            )

        if batch.guidance_rescale > 0.0:
            noise_pred = self._apply_guidance_rescale_parallel(
                noise_pred,
                noise_pred_cond,
                batch.guidance_rescale,
                cfg_rank,
            )

        return self._apply_model_specific_cfg_postprocess(
            batch, noise_pred, noise_pred_cond, cfg_rank
        )

    def _combine_cfg_serial(
        self,
        batch: Req,
        noise_pred_cond: torch.Tensor,
        noise_pred_uncond: torch.Tensor,
        cfg_scale: float,
    ) -> torch.Tensor:
        # Serial CFG keeps both branches local and is the reference path that
        # model-specific postprocessing hooks should match.
        noise_pred = noise_pred_uncond + cfg_scale * (
            noise_pred_cond - noise_pred_uncond
        )

        if batch.cfg_normalization and float(batch.cfg_normalization) > 0:
            noise_pred = self._apply_cfg_normalization(
                noise_pred,
                noise_pred_cond,
                batch.cfg_normalization,
            )

        if batch.guidance_rescale > 0.0:
            noise_pred = self._rescale_noise_cfg(
                noise_pred,
                noise_pred_cond,
                guidance_rescale=batch.guidance_rescale,
            )

        return self.server_args.pipeline_config.postprocess_cfg_noise(
            batch, noise_pred, noise_pred_cond
        )

    def _build_attn_metadata(
        self,
        i: int,
        batch: Req,
        server_args: ServerArgs,
        *,
        timestep_value: int | None = None,
        timesteps: torch.Tensor | None = None,
    ) -> Any | None:
        """
        Build attention metadata for custom attention backends.

        Args:
            i: The current timestep index.
        """
        attn_metadata = None
        self.attn_metadata_builder = None
        try:
            self.attn_metadata_builder_cls = self.attn_backend.get_builder_cls()
        except NotImplementedError:
            self.attn_metadata_builder_cls = None
        if self.attn_metadata_builder_cls:
            self.attn_metadata_builder = self.attn_metadata_builder_cls()
        if (
            self.attn_backend.get_enum() == AttentionBackendEnum.SLIDING_TILE_ATTN
            or self.attn_backend.get_enum() == AttentionBackendEnum.VIDEO_SPARSE_ATTN
        ):
            attn_metadata = self.attn_metadata_builder.build(
                current_timestep=i,
                raw_latent_shape=batch.raw_latent_shape[2:5],
                patch_size=server_args.pipeline_config.dit_config.patch_size,
                STA_param=batch.STA_param,
                VSA_sparsity=server_args.attention_backend_config.VSA_sparsity,
                device=get_local_torch_device(),
            )
        elif (
            self.attn_backend.get_enum() == AttentionBackendEnum.SPARSE_VIDEO_GEN_2_ATTN
        ):
            if timestep_value is None or timesteps is None:
                raise ValueError(
                    "timestep_value and timesteps must be provided for SVG2 attention metadata"
                )

            svg2_cfg = server_args.attention_backend_config or {}
            num_layers = server_args.pipeline_config.dit_config.num_layers
            if (
                server_args.pipeline_config.dit_config.prefix.lower() == "hunyuan"
                and hasattr(server_args.pipeline_config.dit_config, "num_single_layers")
            ):
                num_layers += server_args.pipeline_config.dit_config.num_single_layers
            first_layers_fp = svg2_cfg.get("svg2_first_layers_fp", 0.03)
            if first_layers_fp <= 1.0:
                first_layers_fp = math.floor(first_layers_fp * num_layers)
            first_layers_fp = max(0, min(int(first_layers_fp), num_layers))

            first_times_fp = svg2_cfg.get("svg2_first_times_fp", 0.2)
            if first_times_fp <= 1.0:
                num_fp_steps = math.floor(first_times_fp * len(timesteps))
                if num_fp_steps > 0:
                    first_times_fp = float(timesteps[num_fp_steps - 1].item() - 1)
                else:
                    first_times_fp = float(timesteps.max().item() + 1)

            current_timestep = int(timestep_value)

            cache = batch.extra.get("svg2_cache")
            if cache is None:
                from sglang.multimodal_gen.runtime.layers.attention.backends.sparse_video_gen_2_attn import (
                    Svg2Cache,
                )

                cache = Svg2Cache()
                batch.extra["svg2_cache"] = cache

            patch_size = server_args.pipeline_config.dit_config.patch_size
            if isinstance(patch_size, list):
                patch_size = tuple(patch_size)
            if isinstance(patch_size, int):
                patch_size_t = getattr(
                    server_args.pipeline_config.dit_config, "patch_size_t", None
                )
                if patch_size_t is not None:
                    patch_size = (patch_size_t, patch_size, patch_size)

            context_length = 0
            prompt_length = None
            if server_args.pipeline_config.dit_config.prefix.lower() == "hunyuan":
                prompt_embeds = server_args.pipeline_config.get_pos_prompt_embeds(batch)
                if isinstance(prompt_embeds, list):
                    text_embeds = prompt_embeds[0] if prompt_embeds else None
                else:
                    text_embeds = prompt_embeds
                if isinstance(text_embeds, torch.Tensor) and text_embeds.ndim >= 2:
                    context_length = int(text_embeds.shape[1])
                if context_length > 0 and batch.prompt_attention_mask:
                    mask = batch.prompt_attention_mask[0]
                    if isinstance(mask, torch.Tensor):
                        if mask.shape[-1] > context_length:
                            mask = mask[:, -context_length:]
                        prompt_length = int(mask[0].sum().item())
                if prompt_length is None:
                    prompt_length = context_length

            attn_metadata = self.attn_metadata_builder.build(
                current_timestep=current_timestep,
                raw_latent_shape=batch.raw_latent_shape,
                patch_size=patch_size,
                num_q_centroids=svg2_cfg.get("svg2_num_q_centroids", 300),
                num_k_centroids=svg2_cfg.get("svg2_num_k_centroids", 1000),
                top_p_kmeans=svg2_cfg.get("svg2_top_p_kmeans", 0.9),
                min_kc_ratio=svg2_cfg.get("svg2_min_kc_ratio", 0.1),
                kmeans_iter_init=svg2_cfg.get("svg2_kmeans_iter_init", 50),
                kmeans_iter_step=svg2_cfg.get("svg2_kmeans_iter_step", 2),
                zero_step_kmeans_init=svg2_cfg.get("svg2_zero_step_kmeans_init", False),
                first_layers_fp=first_layers_fp,
                first_times_fp=first_times_fp,
                context_length=context_length,
                prompt_length=prompt_length,
                cache=cache,
                calculate_density=False,  # only need density when doing head load balancing
            )
        elif self.attn_backend.get_enum() == AttentionBackendEnum.VMOBA_ATTN:
            moba_params = server_args.attention_backend_config.moba_config.copy()
            moba_params.update(
                {
                    "current_timestep": i,
                    "raw_latent_shape": batch.raw_latent_shape[2:5],
                    "patch_size": server_args.pipeline_config.dit_config.patch_size,
                    "device": get_local_torch_device(),
                }
            )
        elif self.attn_backend.get_enum() == AttentionBackendEnum.FA:
            attn_metadata = self.attn_metadata_builder.build(
                raw_latent_shape=batch.raw_latent_shape
            )
        else:
            # attn_metadata can be None for SDPA attention backend
            return None

        return attn_metadata

    def _predict_noise(
        self,
        current_model,
        latent_model_input,
        timestep,
        target_dtype,
        guidance: torch.Tensor,
        **kwargs,
    ):
        return current_model(
            hidden_states=latent_model_input,
            timestep=timestep,
            guidance=guidance,
            **kwargs,
        )

    def _predict_noise_with_cfg(
        self,
        current_model: nn.Module,
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
                    target_dtype=target_dtype,
                    guidance=guidance,
                    **image_kwargs,
                    **neg_cond_kwargs,
                )
                noise_pred_uncond = server_args.pipeline_config.slice_noise_pred(
                    noise_pred_uncond, latents
                )
        cfg_scale = server_args.pipeline_config.get_classifier_free_guidance_scale(
            batch, current_guidance_scale
        )

        if server_args.enable_cfg_parallel:
            return self._combine_cfg_parallel(
                batch,
                noise_pred_cond,
                noise_pred_uncond,
                cfg_scale,
                cfg_rank,
            )

        assert noise_pred_cond is not None and noise_pred_uncond is not None
        return self._combine_cfg_serial(
            batch,
            noise_pred_cond,
            noise_pred_uncond,
            cfg_scale,
        )

    def prepare_sta_param(self, batch: Req, server_args: ServerArgs):
        """
        Prepare Sliding Tile Attention (STA) parameters and settings.
        """
        # TODO(kevin): STA mask search, currently only support Wan2.1 with 69x768x1280
        try:
            STA_mode = STA_Mode[server_args.attention_backend_config.STA_mode]
        except Exception as e:
            logger.error(f"Passed STA_mode: {STA_mode} doesn't exist")
            raise e
        skip_time_steps = server_args.attention_backend_config.skip_time_steps
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
        result.add_check(
            "prompt_embeds",
            batch.prompt_embeds,
            self._get_prompt_embeds_validator(batch),
        )
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
            self._get_negative_prompt_embeds_validator(batch),
        )
        return result

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify denoising stage outputs."""
        result = VerificationResult()
        # result.add_check("latents", batch.latents, [V.is_tensor, V.with_dims(5)])
        return result
