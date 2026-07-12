# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch

from sglang.multimodal_gen.runtime.distributed.communication_op import (
    cfg_model_parallel_all_gather,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_cfg_group,
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
)
from sglang.multimodal_gen.runtime.managers.dreamzero_session_cache import (
    BRANCH_COND,
    BRANCH_UNCOND,
    DreamZeroCachePoolManager,
    enter_request_cache,
    record_session_timing,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    PipelineStage,
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE


@dataclass
class DreamZeroInputContext:
    request_inputs: dict[str, Any]
    dtype: torch.dtype
    device: torch.device
    clip_feature: torch.Tensor
    y_full: torch.Tensor
    latent_video: torch.Tensor
    batch_size: int


@dataclass
class DreamZeroNoiseContext:
    action_dim: int
    max_state_dim: int
    action_horizon: int
    num_frame_per_block: int
    noise_obs: torch.Tensor
    noise_action: torch.Tensor
    frame_seqlen: int
    seq_len: int


@dataclass
class DreamZeroBranchContext:
    local_branch_indices: list[int]
    local_prompt_embs: list[torch.Tensor]
    cfg_rank: int | None
    cfg_world_size: int | None


@dataclass
class DreamZeroCacheContext:
    request_cache: Any
    session_state: Any
    slots: list[int]
    current_start_frame: int
    kv_caches: list[list[torch.Tensor]]
    crossattn_caches: list[list[dict[str, Any]]]


@dataclass
class DreamZeroDenoisingContext:
    inputs: DreamZeroInputContext
    noise: DreamZeroNoiseContext
    branches: DreamZeroBranchContext
    cache: DreamZeroCacheContext


@dataclass
class SchedulerStepState:
    step_index: int
    video_timestep: torch.Tensor
    action_timestep: torch.Tensor
    prev_predictions: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    skip_state: dict[str, int]


class DreamZeroCausalDenoisingStage(PipelineStage):
    """One-shot DreamZero causal video/action denoising stage."""

    def __init__(
        self,
        transformer: torch.nn.Module,
        scheduler: Any | None = None,
        cache_manager: DreamZeroCachePoolManager | None = None,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.scheduler = scheduler
        self.cache_manager = cache_manager

    @property
    def parallelism_type(self) -> StageParallelismType:
        if getattr(self.server_args, "enable_cfg_parallel", False):
            return StageParallelismType.CFG_PARALLEL
        return StageParallelismType.REPLICATED

    @property
    def role_affinity(self):
        from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType

        return RoleType.DENOISER

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        stage_name = self._component_stage_name(stage_name)
        return [
            ComponentUse(
                stage_name,
                "transformer",
                target_dtype=PRECISION_TO_TYPE[
                    server_args.pipeline_config.dit_precision
                ],
            )
        ]

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check(
            "dreamzero_action_pred",
            getattr(batch, "dreamzero_action_pred", None),
            torch.is_tensor,
        )
        return result

    @staticmethod
    def _module_dtype(module: torch.nn.Module) -> torch.dtype:
        try:
            return next(module.parameters()).dtype
        except StopIteration:
            return torch.bfloat16

    @staticmethod
    def _module_device(module: torch.nn.Module) -> torch.device:
        try:
            return next(module.parameters()).device
        except StopIteration:
            return torch.device(current_platform.device_type)

    @staticmethod
    def _make_noise(shape, *, seed: int, device: torch.device, dtype: torch.dtype):
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        return torch.randn(shape, generator=generator, device=device, dtype=dtype)

    def _create_kv_caches(
        self,
        *,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        local_heads: int | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        model = self.transformer
        num_heads = int(getattr(model, "local_num_heads", getattr(model, "num_heads")))
        if local_heads is not None:
            num_heads = local_heads
        head_dim = int(getattr(model, "dim")) // int(getattr(model, "num_heads"))

        def new_cache() -> list[torch.Tensor]:
            return [
                torch.zeros(
                    [2, batch_size, 0, num_heads, head_dim],
                    dtype=dtype,
                    device=device,
                )
                for _ in range(int(getattr(model, "num_layers")))
            ]

        return new_cache(), new_cache()

    def _create_crossattn_caches(
        self, *, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        model = self.transformer
        return (
            [{"is_init": False} for _ in range(model.num_layers)],
            [{"is_init": False} for _ in range(model.num_layers)],
        )

    @staticmethod
    def _cfg_parallel_active(server_args: ServerArgs, num_branches: int) -> bool:
        if not getattr(server_args, "enable_cfg_parallel", False):
            return False
        cfg_world_size = get_classifier_free_guidance_world_size()
        if cfg_world_size != 2 or num_branches != 2:
            raise RuntimeError(
                "DreamZero CFG parallel currently supports exactly two CFG ranks "
                f"and two branches, got cfg_world_size={cfg_world_size}, "
                f"branches={num_branches}"
            )
        return True

    @staticmethod
    def _cfg_local_branch_indices(
        server_args: ServerArgs, branch_indices: list[int]
    ) -> tuple[list[int], int | None, int]:
        if not DreamZeroCausalDenoisingStage._cfg_parallel_active(
            server_args,
            len(branch_indices),
        ):
            return branch_indices, None, 1
        cfg_rank = get_classifier_free_guidance_rank()
        if cfg_rank >= len(branch_indices):
            raise RuntimeError(
                "DreamZero CFG rank has no branch assignment: "
                f"cfg_rank={cfg_rank}, branches={branch_indices}"
            )
        return (
            [branch_indices[cfg_rank]],
            cfg_rank,
            get_classifier_free_guidance_world_size(),
        )

    @staticmethod
    def _combine_cfg_parallel_predictions(
        *,
        local_prediction: tuple[torch.Tensor, torch.Tensor],
        cfg_rank: int,
        cfg_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video, action = local_prediction
        if cfg_rank not in (0, 1):
            raise ValueError(
                "DreamZero two-branch CFG only supports cfg ranks 0 and 1, "
                f"got {cfg_rank}"
            )
        branches = cfg_model_parallel_all_gather(
            video,
            dim=0,
            separate_tensors=True,
        )
        if not isinstance(branches, list) or len(branches) != 2:
            raise RuntimeError(
                "DreamZero CFG all-gather must return cond/uncond tensors"
            )
        flow_pred_cond, flow_pred_uncond = branches
        flow_pred = flow_pred_uncond + cfg_scale * (flow_pred_cond - flow_pred_uncond)

        flow_pred_action_cond = action if cfg_rank == 0 else torch.empty_like(action)
        flow_pred_action_cond = get_cfg_group().broadcast(
            flow_pred_action_cond,
            src=0,
        )
        return flow_pred, flow_pred_action_cond

    def _run_diffusion_steps(
        self,
        *,
        noisy_input: torch.Tensor,
        timestep: torch.Tensor,
        action: torch.Tensor | None,
        timestep_action: torch.Tensor | None,
        state: torch.Tensor | None,
        context: list[torch.Tensor],
        seq_len: int,
        y: torch.Tensor,
        clip_feature: torch.Tensor,
        kv_caches: list[list[torch.Tensor]],
        crossattn_caches: list[list[dict[str, Any]]],
        current_start_frame: int,
        update_kv_cache: bool,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        predictions = []
        for local_index, prompt_emb in enumerate(context):
            kv_cache = kv_caches[local_index]
            obs_noise_pred, action_noise_pred, updated_kv_caches = self.transformer(
                x=noisy_input,
                timestep=timestep,
                action=action,
                timestep_action=timestep_action,
                state=state,
                embodiment_id=None,
                context=prompt_emb,
                seq_len=seq_len,
                y=y,
                clip_feature=clip_feature,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_caches[local_index],
                current_start_frame=current_start_frame,
                enable_sequence_parallel=(
                    self.server_args.pipeline_config.dreamzero_sequence_parallel_size
                    > 1
                ),
            )
            if update_kv_cache:
                for block_index, updated in enumerate(updated_kv_caches):
                    kv_cache[block_index] = updated.detach()
            if action_noise_pred is None:
                action_noise_pred = obs_noise_pred.new_zeros(())
            predictions.append((obs_noise_pred.detach(), action_noise_pred.detach()))
        return predictions

    @staticmethod
    def _should_run_model(
        *,
        step_index: int,
        current_timestep: torch.Tensor,
        prev_predictions: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        dit_step_mask: tuple[bool, ...] | None,
        dynamic_cache_schedule: bool,
        skip_state: dict[str, int],
    ) -> bool:
        if not dynamic_cache_schedule:
            if dit_step_mask is None:
                return True
            return bool(dit_step_mask[step_index])

        if len(prev_predictions) < 2:
            return True

        if skip_state["countdown"] > 1:
            skip_state["countdown"] -= 1
            return False
        if skip_state["countdown"] == 1:
            skip_state["countdown"] = 0
            return True

        del current_timestep
        v_last = prev_predictions[-1][1].flatten(1).float()
        v_prev = prev_predictions[-2][1].flatten(1).float()
        sim = torch.nn.functional.cosine_similarity(v_last, v_prev, dim=1).mean()

        for threshold, countdown in ((0.95, 4), (0.93, 2)):
            if sim > threshold:
                skip_state["countdown"] = countdown
                return False

        return True

    @staticmethod
    def _scheduler_train_timesteps(scheduler: Any | None) -> int:
        if scheduler is None:
            return 1000
        value = getattr(scheduler, "num_train_timesteps", None)
        if value is not None:
            return int(value)
        config = getattr(scheduler, "config", None)
        value = getattr(config, "num_train_timesteps", None)
        if value is None and isinstance(config, dict):
            value = config.get("num_train_timesteps")
        return int(value) if value is not None else 1000

    @staticmethod
    def _require_tensor(batch: Req, field_name: str) -> torch.Tensor:
        value = getattr(batch, field_name, None)
        if not torch.is_tensor(value):
            raise ValueError(
                f"DreamZero denoising requires batch.{field_name}; "
                "run image/VAE encoding stages before denoising"
            )
        return value

    def _new_unipc_scheduler(self) -> Any:
        scheduler_cls = (
            self.scheduler.__class__
            if self.scheduler is not None
            else FlowUniPCMultistepScheduler
        )
        return scheduler_cls(
            num_train_timesteps=self._scheduler_train_timesteps(self.scheduler),
            shift=1,
            use_dynamic_shifting=False,
        )

    @staticmethod
    def _scheduler_step(
        scheduler: Any,
        *,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        step_index: int,
    ) -> torch.Tensor:
        try:
            return scheduler.step(
                model_output=model_output,
                timestep=timestep,
                sample=sample,
                step_index=step_index,
                return_dict=False,
            )[0]
        except TypeError as exc:
            if "step_index" not in str(exc):
                raise
            return scheduler.step(
                model_output=model_output,
                timestep=timestep,
                sample=sample,
                return_dict=False,
            )[0]

    def _validate_dreamzero_batch(self, batch: Req) -> None:
        if not hasattr(batch, "dreamzero_inputs"):
            raise ValueError("DreamZero denoising requires batch.dreamzero_inputs")
        if not getattr(batch, "dreamzero_prompt_embs", None):
            raise ValueError("DreamZero denoising requires batch.dreamzero_prompt_embs")

    def _materialize_input_tensors(
        self,
        batch: Req,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> DreamZeroInputContext:
        inputs: dict[str, Any] = batch.dreamzero_inputs
        clip_feature = self._require_tensor(batch, "dreamzero_clip_feature").to(
            device=device, dtype=dtype
        )
        y_full = self._require_tensor(batch, "dreamzero_y").to(
            device=device, dtype=dtype
        )
        latent_video = self._require_tensor(batch, "dreamzero_latent_video").to(
            device=device, dtype=dtype
        )
        batch.dreamzero_clip_feature = clip_feature
        batch.dreamzero_y = y_full
        batch.dreamzero_latent_video = latent_video
        return DreamZeroInputContext(
            request_inputs=inputs,
            dtype=dtype,
            device=device,
            clip_feature=clip_feature,
            y_full=y_full,
            latent_video=latent_video,
            batch_size=latent_video.shape[0],
        )

    def _prepare_noise_tensors(
        self,
        batch: Req,
        server_args: ServerArgs,
        *,
        latent_video: torch.Tensor,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> DreamZeroNoiseContext:
        arch_config = server_args.pipeline_config.dit_config.arch_config
        action_dim = arch_config.action_dim
        max_state_dim = arch_config.max_state_dim
        action_horizon = server_args.pipeline_config.action_horizon
        num_frame_per_block = arch_config.num_frame_per_block
        seed = batch.seed[0] if isinstance(batch.seed, list) else batch.seed
        noise_obs = self._make_noise(
            (
                batch_size,
                latent_video.shape[1],
                num_frame_per_block,
                latent_video.shape[3],
                latent_video.shape[4],
            ),
            seed=seed,
            device=device,
            dtype=dtype,
        )
        noise_action = self._make_noise(
            (batch_size, action_horizon, action_dim),
            seed=seed,
            device=device,
            dtype=dtype,
        )
        patch_size = arch_config.patch_size
        frame_seqlen = (noise_obs.shape[3] // patch_size[1]) * (
            noise_obs.shape[4] // patch_size[2]
        )
        return DreamZeroNoiseContext(
            action_dim=action_dim,
            max_state_dim=max_state_dim,
            action_horizon=action_horizon,
            num_frame_per_block=num_frame_per_block,
            noise_obs=noise_obs,
            noise_action=noise_action,
            frame_seqlen=frame_seqlen,
            seq_len=num_frame_per_block * frame_seqlen,
        )

    def _prepare_context(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> DreamZeroDenoisingContext:
        dtype = self._module_dtype(self.transformer)
        device = self._module_device(self.transformer)
        input_ctx = self._materialize_input_tensors(batch, dtype=dtype, device=device)
        local_attn_size = int(getattr(self.transformer, "local_attn_size", -1))
        request_cache, session_state = enter_request_cache(
            batch,
            self.cache_manager,
            local_attn_size=local_attn_size,
            batch_size=input_ctx.batch_size,
        )
        slots = request_cache.slot_indices
        current_start_frame = request_cache.uniform_current_start_frame(
            self.cache_manager
        )
        record_session_timing(batch, "kv_layout_materialize_ms", 0.0)

        noise_ctx = self._prepare_noise_tensors(
            batch,
            server_args,
            latent_video=input_ctx.latent_video,
            batch_size=input_ctx.batch_size,
            dtype=dtype,
            device=device,
        )
        branch_ctx = self._prepare_cfg_branches(
            batch,
            server_args,
            device=device,
            dtype=dtype,
        )
        kv_caches, crossattn_caches = self._prepare_branch_caches(
            request_cache=request_cache,
            session_state=session_state,
            slots=slots,
            current_start_frame=current_start_frame,
            local_branch_indices=branch_ctx.local_branch_indices,
            cfg_rank=branch_ctx.cfg_rank,
            batch_size=input_ctx.batch_size,
            dtype=dtype,
            device=device,
        )
        cache_ctx = DreamZeroCacheContext(
            request_cache=request_cache,
            session_state=session_state,
            slots=slots,
            current_start_frame=current_start_frame,
            kv_caches=kv_caches,
            crossattn_caches=crossattn_caches,
        )
        return DreamZeroDenoisingContext(
            inputs=input_ctx,
            noise=noise_ctx,
            branches=branch_ctx,
            cache=cache_ctx,
        )

    def _prepare_cfg_branches(
        self,
        batch: Req,
        server_args: ServerArgs,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> DreamZeroBranchContext:
        cfg_parallel = bool(getattr(server_args, "enable_cfg_parallel", False))
        source_prompt_embs = batch.dreamzero_prompt_embs
        prompt_embs = [emb.to(device=device, dtype=dtype) for emb in source_prompt_embs]
        if not cfg_parallel and len(prompt_embs) == 1:
            prompt_embs = [prompt_embs[0], prompt_embs[0]]
        if cfg_parallel:
            if len(prompt_embs) not in (1, 2):
                raise RuntimeError(
                    "DreamZero CFG parallel expects either one rank-local "
                    "prompt embedding or both cond/uncond prompt embeddings, "
                    f"got {len(prompt_embs)}"
                )
            branch_indices = [BRANCH_COND, BRANCH_UNCOND]
        else:
            branch_indices = list(range(len(prompt_embs)))
        local_branch_indices, cfg_rank, cfg_world_size = self._cfg_local_branch_indices(
            server_args,
            branch_indices,
        )
        if cfg_rank is not None and len(prompt_embs) == 1:
            local_prompt_embs = prompt_embs
        else:
            local_prompt_embs = [
                prompt_embs[branch_indices.index(branch_index)]
                for branch_index in local_branch_indices
            ]
        return DreamZeroBranchContext(
            local_branch_indices=local_branch_indices,
            local_prompt_embs=local_prompt_embs,
            cfg_rank=cfg_rank,
            cfg_world_size=cfg_world_size,
        )

    def _prepare_branch_caches(
        self,
        *,
        request_cache: Any,
        session_state: Any,
        slots: list[int],
        current_start_frame: int,
        local_branch_indices: list[int],
        cfg_rank: int | None,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[list[list[torch.Tensor]], list[list[dict[str, Any]]]]:
        if current_start_frame == 0:
            kv_cache_pair = list(
                self._create_kv_caches(
                    batch_size=batch_size,
                    dtype=dtype,
                    device=device,
                )
            )
            crossattn_cache_pair = list(
                self._create_crossattn_caches(
                    batch_size=batch_size,
                    dtype=dtype,
                    device=device,
                )
            )
            if cfg_rank is None:
                return kv_cache_pair, crossattn_cache_pair
            return [kv_cache_pair[cfg_rank]], [crossattn_cache_pair[cfg_rank]]
        return (
            [
                session_state.gather_kv(branch_index, slots)
                for branch_index in local_branch_indices
            ],
            [
                session_state.gather_crossattn(branch_index, slots)
                for branch_index in local_branch_indices
            ],
        )

    def _scatter_active_branch_caches(self, ctx: DreamZeroDenoisingContext) -> None:
        for local_index, branch_index in enumerate(ctx.branches.local_branch_indices):
            ctx.cache.session_state.scatter_kv(
                branch_index,
                ctx.cache.slots,
                ctx.cache.kv_caches[local_index],
            )
            ctx.cache.session_state.scatter_crossattn(
                branch_index,
                ctx.cache.slots,
                ctx.cache.crossattn_caches[local_index],
            )

    def _run_prefill(self, ctx: DreamZeroDenoisingContext) -> None:
        if ctx.cache.current_start_frame == 0:
            zero_timestep = torch.zeros(
                [ctx.inputs.batch_size, 1],
                device=ctx.inputs.device,
                dtype=torch.int64,
            )
            self._run_diffusion_steps(
                noisy_input=ctx.inputs.latent_video,
                timestep=zero_timestep,
                action=None,
                timestep_action=None,
                state=None,
                context=ctx.branches.local_prompt_embs,
                seq_len=ctx.noise.frame_seqlen,
                y=ctx.inputs.y_full[:, :, 0:1],
                clip_feature=ctx.inputs.clip_feature,
                kv_caches=ctx.cache.kv_caches,
                crossattn_caches=ctx.cache.crossattn_caches,
                current_start_frame=0,
                update_kv_cache=True,
            )
            self._scatter_active_branch_caches(ctx)
            ctx.cache.request_cache.mark_current_start_frame(self.cache_manager, 1)
            ctx.cache.current_start_frame = 1
        if ctx.cache.current_start_frame != 1:
            current_ref_latents = ctx.inputs.latent_video[
                :, :, -ctx.noise.num_frame_per_block :
            ]
            y_start = ctx.cache.current_start_frame - ctx.noise.num_frame_per_block
            if ctx.cache.current_start_frame <= ctx.inputs.y_full.shape[2]:
                y_prefill = ctx.inputs.y_full[
                    :, :, y_start : ctx.cache.current_start_frame
                ]
            else:
                y_prefill = ctx.inputs.y_full[:, :, -ctx.noise.num_frame_per_block :]
            zero_timestep = torch.zeros(
                [ctx.inputs.batch_size, ctx.noise.num_frame_per_block],
                device=ctx.inputs.device,
                dtype=torch.int64,
            )
            self._run_diffusion_steps(
                noisy_input=current_ref_latents,
                timestep=zero_timestep,
                action=None,
                timestep_action=None,
                state=None,
                context=ctx.branches.local_prompt_embs,
                seq_len=ctx.noise.seq_len,
                y=y_prefill,
                clip_feature=ctx.inputs.clip_feature,
                kv_caches=ctx.cache.kv_caches,
                crossattn_caches=ctx.cache.crossattn_caches,
                current_start_frame=y_start,
                update_kv_cache=True,
            )
            self._scatter_active_branch_caches(ctx)

    def _maybe_return_prefill_only(
        self,
        batch: Req,
        ctx: DreamZeroDenoisingContext,
    ) -> Req | None:
        if os.environ.get("DREAMZERO_SELF_CONTAINED_PREFILL_ONLY", "0") != "1":
            return None
        batch.dreamzero_prefill_only = True
        batch.dreamzero_action_pred = torch.zeros(
            ctx.inputs.batch_size,
            ctx.noise.action_horizon,
            ctx.noise.action_dim,
            device=ctx.inputs.device,
            dtype=torch.float32,
        )
        record_session_timing(batch, "kv_split_append_ms", 0.0)
        record_session_timing(batch, "session_scatter_ms", 0.0)
        self._set_request_metadata(batch, ctx)
        batch.output = batch.dreamzero_action_pred
        return batch

    def _prepare_rollout_state(self, ctx: DreamZeroDenoisingContext) -> torch.Tensor:
        state = ctx.inputs.request_inputs.get("state")
        if state is None:
            return torch.zeros(
                ctx.inputs.batch_size,
                1,
                ctx.noise.max_state_dim,
                device=ctx.inputs.device,
                dtype=ctx.inputs.dtype,
            )
        return state.to(device=ctx.inputs.device, dtype=ctx.inputs.dtype)

    def _rollout_step_prediction(
        self,
        *,
        server_args: ServerArgs,
        ctx: DreamZeroDenoisingContext,
        state: torch.Tensor,
        noisy_input: torch.Tensor,
        noisy_action: torch.Tensor,
        step_state: SchedulerStepState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        timestep = (
            torch.ones(
                [ctx.inputs.batch_size, ctx.noise.num_frame_per_block],
                device=ctx.inputs.device,
                dtype=torch.int64,
            )
            * step_state.video_timestep
        )
        timestep_action = (
            torch.ones(
                [ctx.inputs.batch_size, ctx.noise.action_horizon],
                device=ctx.inputs.device,
                dtype=torch.int64,
            )
            * step_state.action_timestep
        )
        y_start = ctx.cache.current_start_frame
        y_end = y_start + ctx.noise.num_frame_per_block
        y = ctx.inputs.y_full[:, :, y_start:y_end]
        if y.shape[2] < ctx.noise.num_frame_per_block:
            y = ctx.inputs.y_full[:, :, -ctx.noise.num_frame_per_block :]

        should_run_model = self._should_run_model(
            step_index=step_state.step_index,
            current_timestep=step_state.video_timestep,
            prev_predictions=step_state.prev_predictions,
            dit_step_mask=server_args.pipeline_config.dit_step_mask,
            dynamic_cache_schedule=getattr(
                server_args.pipeline_config,
                "dynamic_cache_schedule",
                False,
            ),
            skip_state=step_state.skip_state,
        )
        if not should_run_model:
            assert (
                step_state.prev_predictions
            ), "prev_predictions must be set when DreamZero skips a DiT step"
            _, flow_pred, flow_pred_action_cond = step_state.prev_predictions[-1]
            return flow_pred, flow_pred_action_cond

        predictions = self._run_diffusion_steps(
            noisy_input=noisy_input.transpose(1, 2).contiguous(),
            timestep=timestep,
            action=noisy_action,
            timestep_action=timestep_action,
            state=state,
            context=ctx.branches.local_prompt_embs,
            seq_len=ctx.noise.seq_len,
            y=y,
            clip_feature=ctx.inputs.clip_feature,
            kv_caches=ctx.cache.kv_caches,
            crossattn_caches=ctx.cache.crossattn_caches,
            current_start_frame=ctx.cache.current_start_frame,
            update_kv_cache=False,
        )
        cfg_scale = server_args.pipeline_config.cfg_scale
        if ctx.branches.cfg_rank is None:
            flow_pred_cond, flow_pred_action_cond = predictions[0]
            flow_pred_uncond, _ = predictions[1]
            flow_pred = flow_pred_uncond + cfg_scale * (
                flow_pred_cond - flow_pred_uncond
            )
        else:
            flow_pred, flow_pred_action_cond = self._combine_cfg_parallel_predictions(
                local_prediction=predictions[0],
                cfg_rank=ctx.branches.cfg_rank,
                cfg_scale=cfg_scale,
            )
        step_state.prev_predictions.append(
            (step_state.video_timestep, flow_pred, flow_pred_action_cond)
        )
        if len(step_state.prev_predictions) > 2:
            step_state.prev_predictions.pop(0)
        return flow_pred, flow_pred_action_cond

    def _run_action_rollout(
        self,
        batch: Req,
        server_args: ServerArgs,
        ctx: DreamZeroDenoisingContext,
    ) -> None:
        state = self._prepare_rollout_state(ctx)
        scheduler = self._new_unipc_scheduler()
        action_scheduler = self._new_unipc_scheduler()
        for rollout_scheduler in (scheduler, action_scheduler):
            rollout_scheduler.set_timesteps(
                server_args.pipeline_config.default_num_inference_steps,
                device=ctx.inputs.device,
                shift=server_args.pipeline_config.flow_shift,
            )

        noisy_input = ctx.noise.noise_obs.transpose(1, 2).contiguous()
        noisy_action = ctx.noise.noise_action
        prev_predictions: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        skip_state = {"countdown": 0}
        for step_index, video_timestep in enumerate(scheduler.timesteps):
            step_state = SchedulerStepState(
                step_index=step_index,
                video_timestep=video_timestep,
                action_timestep=action_scheduler.timesteps[step_index],
                prev_predictions=prev_predictions,
                skip_state=skip_state,
            )
            flow_pred, flow_pred_action_cond = self._rollout_step_prediction(
                server_args=server_args,
                ctx=ctx,
                state=state,
                noisy_input=noisy_input,
                noisy_action=noisy_action,
                step_state=step_state,
            )
            noisy_input = self._scheduler_step(
                scheduler,
                model_output=flow_pred.transpose(1, 2),
                timestep=step_state.video_timestep,
                sample=noisy_input,
                step_index=step_index,
            )
            noisy_action = self._scheduler_step(
                action_scheduler,
                model_output=flow_pred_action_cond,
                timestep=step_state.action_timestep,
                sample=noisy_action,
                step_index=step_index,
            )
        batch.dreamzero_action_pred = noisy_action.float()

    def _set_request_metadata(
        self,
        batch: Req,
        ctx: DreamZeroDenoisingContext,
    ) -> None:
        batch.dreamzero_current_start_frame = (
            ctx.cache.request_cache.current_start_frames(self.cache_manager)
        )
        batch.dreamzero_cfg_rank = ctx.branches.cfg_rank
        batch.dreamzero_cfg_world_size = ctx.branches.cfg_world_size
        batch.dreamzero_cfg_branches_per_step = len(ctx.branches.local_branch_indices)
        self._record_session_cache_overhead(batch)

    def _finalize_request(
        self,
        batch: Req,
        ctx: DreamZeroDenoisingContext,
    ) -> Req:
        ctx.cache.current_start_frame += ctx.noise.num_frame_per_block
        ctx.cache.request_cache.mark_current_start_frame(
            self.cache_manager,
            ctx.cache.current_start_frame,
        )
        ctx.cache.session_state.scatter_visual(
            ctx.cache.slots,
            latent_video=ctx.inputs.latent_video,
        )
        record_session_timing(batch, "kv_split_append_ms", 0.0)
        record_session_timing(batch, "session_scatter_ms", 0.0)
        if hasattr(batch, "dreamzero_kv_caches"):
            delattr(batch, "dreamzero_kv_caches")
        if hasattr(batch, "dreamzero_crossattn_caches"):
            delattr(batch, "dreamzero_crossattn_caches")
        ctx.cache.kv_caches = []
        ctx.cache.crossattn_caches = []
        self._set_request_metadata(batch, ctx)
        batch.output = batch.dreamzero_action_pred
        return batch

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        self._validate_dreamzero_batch(batch)
        with self.use_declared_component(
            component_name="transformer", module=self.transformer
        ) as transformer:
            assert transformer is not None
            self.transformer = transformer
            ctx = self._prepare_context(batch, server_args)
            self._run_prefill(ctx)
            prefill_result = self._maybe_return_prefill_only(batch, ctx)
            if prefill_result is not None:
                return prefill_result
            self._run_action_rollout(batch, server_args, ctx)
            return self._finalize_request(batch, ctx)

    @staticmethod
    def _record_session_cache_overhead(batch: Req) -> None:
        timing = getattr(batch, "dreamzero_session_timing", {})
        overhead_ms = sum(
            float(timing.get(key, 0.0))
            for key in (
                "session_gather_ms",
                "kv_layout_materialize_ms",
                "kv_split_append_ms",
                "session_scatter_ms",
            )
        )
        timing["session_cache_overhead_ms"] = overhead_ms
        timing["session_store_overhead_ms"] = overhead_ms
        batch.dreamzero_session_timing = timing
