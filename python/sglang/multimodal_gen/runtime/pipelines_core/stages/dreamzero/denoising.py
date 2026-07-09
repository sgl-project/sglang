# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
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
from sglang.multimodal_gen.runtime.managers.dreamzero_session_store import (
    SessionStore,
    get_request_session_state,
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


class DreamZeroCausalDenoisingStage(PipelineStage):
    """One-shot DreamZero causal video/action denoising stage."""

    def __init__(
        self,
        transformer: torch.nn.Module,
        scheduler: Any | None = None,
        session_store: SessionStore | None = None,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.scheduler = scheduler
        self.session_store = session_store

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
                target_dtype=PRECISION_TO_TYPE[server_args.pipeline_config.dit_precision],
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
        return (
            [
                torch.zeros(
                    [2, batch_size, 0, num_heads, head_dim],
                    dtype=dtype,
                    device=device,
                )
                for _ in range(int(getattr(model, "num_layers")))
            ],
            [
                torch.zeros(
                    [2, batch_size, 0, num_heads, head_dim],
                    dtype=dtype,
                    device=device,
                )
                for _ in range(int(getattr(model, "num_layers")))
            ],
        )

    def _create_single_kv_cache(
        self,
        *,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        local_heads: int | None = None,
    ) -> list[torch.Tensor]:
        return self._create_kv_caches(
            batch_size=batch_size,
            dtype=dtype,
            device=device,
            local_heads=local_heads,
        )[0]

    def _create_crossattn_caches(
        self, *, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        model = self.transformer
        return (
            [{"is_init": False} for _ in range(model.num_layers)],
            [{"is_init": False} for _ in range(model.num_layers)],
        )

    def _create_single_crossattn_cache(self) -> list[dict[str, Any]]:
        return [{"is_init": False} for _ in range(self.transformer.num_layers)]

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
        crossattn_caches: list[list[torch.Tensor]],
        current_start_frame: int,
        update_kv_cache: bool,
        branch_indices: list[int] | None = None,
        debug_calls: list[dict[str, Any]] | None = None,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        predictions = []
        if branch_indices is None:
            branch_indices = list(range(len(context)))
        if len(branch_indices) != len(context):
            raise ValueError("branch_indices must match the number of CFG contexts")
        for local_index, (branch_index, prompt_emb) in enumerate(
            zip(branch_indices, context, strict=True)
        ):
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
                    kv_cache[block_index] = updated.clone()
            if action_noise_pred is None:
                action_noise_pred = obs_noise_pred.new_zeros(())
            debug_call_capture_limit = int(
                os.environ.get(
                    "DREAMZERO_DEBUG_DIT_CALL_CAPTURE_LIMIT",
                    "0",
                )
            )
            debug_action_only = (
                os.environ.get("DREAMZERO_DEBUG_DIT_ACTION_ONLY", "0") == "1"
            )
            capture_debug_call = (
                not debug_action_only or action is not None
            )
            if debug_calls is not None and capture_debug_call and (
                debug_call_capture_limit <= 0
                or len(debug_calls) < debug_call_capture_limit
            ):
                debug_layers = getattr(
                    self.transformer, "_dreamzero_last_layer_debug", None
                )
                debug_calls.append(
                    {
                        "branch_index": branch_index,
                        "current_start_frame": current_start_frame,
                        "update_kv_cache": update_kv_cache,
                        "has_action": action is not None,
                        "action_input_length": (
                            action.shape[1] if action is not None else 0
                        ),
                        "state_input_length": (
                            state.shape[1] if state is not None else 0
                        ),
                        "input_video": noisy_input.detach().cpu().clone(),
                        "context": prompt_emb.detach().cpu().clone(),
                        "input_action": (
                            action.detach().cpu().clone()
                            if action is not None
                            else None
                        ),
                        "timestep": timestep.detach().cpu().clone(),
                        "timestep_action": (
                            timestep_action.detach().cpu().clone()
                            if timestep_action is not None
                            else None
                        ),
                        "video": obs_noise_pred.detach().cpu().clone(),
                        "action": action_noise_pred.detach().cpu().clone(),
                        "layers": debug_layers if debug_layers is not None else [],
                    }
                )
            predictions.append((obs_noise_pred.clone(), action_noise_pred.clone()))
        return predictions

    @staticmethod
    def _combine_cfg_parallel_predictions(
        *,
        video: torch.Tensor,
        action: torch.Tensor,
        cfg_scale: float,
        cfg_rank: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        flow_pred = flow_pred_uncond + cfg_scale * (
            flow_pred_cond - flow_pred_uncond
        )

        flow_pred_action_cond = (
            action if cfg_rank == 0 else torch.empty_like(action)
        )
        flow_pred_action_cond = get_cfg_group().broadcast(
            flow_pred_action_cond,
            src=0,
        )
        return flow_pred, flow_pred_action_cond

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
        if os.environ.get("DREAMZERO_USE_GROOT_UNIPC_SCHEDULER", "0") == "1":
            from groot.vla.model.dreamzero.modules.flow_unipc_multistep_scheduler import (
                FlowUniPCMultistepScheduler as GrootFlowUniPCMultistepScheduler,
            )

            scheduler_cls = GrootFlowUniPCMultistepScheduler
        else:
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

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if not hasattr(batch, "dreamzero_inputs"):
            raise ValueError("DreamZero denoising requires batch.dreamzero_inputs")
        if not getattr(batch, "dreamzero_prompt_embs", None):
            raise ValueError("DreamZero denoising requires batch.dreamzero_prompt_embs")

        with self.use_declared_component(
            component_name="transformer", module=self.transformer
        ) as transformer:
            assert transformer is not None
            self.transformer = transformer

            dtype = self._module_dtype(self.transformer)
            device = self._module_device(self.transformer)
            inputs: dict[str, Any] = batch.dreamzero_inputs
            clip_feature = self._require_tensor(
                batch, "dreamzero_clip_feature"
            ).to(device=device, dtype=dtype)
            y_full = self._require_tensor(batch, "dreamzero_y").to(
                device=device, dtype=dtype
            )
            latent_video = self._require_tensor(
                batch, "dreamzero_latent_video"
            ).to(device=device, dtype=dtype)
            batch.dreamzero_clip_feature = clip_feature
            batch.dreamzero_y = y_full
            batch.dreamzero_latent_video = latent_video
            batch_size = latent_video.shape[0]
            cfg_parallel = bool(
                getattr(server_args, "enable_cfg_parallel", False)
            )
            cfg_rank = 0
            cfg_world_size = 1
            if cfg_parallel:
                cfg_world_size = get_classifier_free_guidance_world_size()
                if cfg_world_size != 2:
                    raise ValueError(
                        "DreamZero CFG parallel requires cfg_parallel_degree=2, "
                        f"got {cfg_world_size}"
                    )
                cfg_rank = get_classifier_free_guidance_rank()
                # This is the second CFG_PARALLEL stage. ParallelExecutor has
                # broadcast rank 0's batch again, so switch back to this
                # worker's SessionStore entry. The text stage marked any
                # explicit reset as consumed, preventing a second reset here.
                if hasattr(batch, "dreamzero_session_state"):
                    delattr(batch, "dreamzero_session_state")
            session_state = get_request_session_state(
                batch,
                self.session_store,
                local_attn_size=int(
                    getattr(self.transformer, "local_attn_size", -1)
                ),
            )

            action_dim = server_args.pipeline_config.dit_config.arch_config.action_dim
            max_state_dim = (
                server_args.pipeline_config.dit_config.arch_config.max_state_dim
            )
            action_horizon = server_args.pipeline_config.action_horizon
            num_frame_per_block = (
                server_args.pipeline_config.dit_config.arch_config.num_frame_per_block
            )
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

            patch_size = server_args.pipeline_config.dit_config.arch_config.patch_size
            frame_seqlen = (
                noise_obs.shape[3] // patch_size[1]
            ) * (noise_obs.shape[4] // patch_size[2])
            seq_len = num_frame_per_block * frame_seqlen
            source_prompt_embs = batch.dreamzero_prompt_embs
            if cfg_parallel and session_state.cached_prompt_embs is not None:
                source_prompt_embs = session_state.cached_prompt_embs
            prompt_embs = [
                emb.to(device=device, dtype=dtype) for emb in source_prompt_embs
            ]
            if cfg_parallel:
                if len(prompt_embs) != 1:
                    if len(prompt_embs) < 2:
                        raise ValueError(
                            "DreamZero CFG parallel requires one rank-local prompt "
                            "embedding or both CFG prompt embeddings"
                        )
                    prompt_embs = [prompt_embs[cfg_rank]]
                branch_indices = [cfg_rank]
            else:
                if len(prompt_embs) == 1:
                    prompt_embs = [prompt_embs[0], prompt_embs[0]]
                branch_indices = list(range(len(prompt_embs)))

            if cfg_parallel and session_state.current_start_frame == 0:
                local_kv_cache = self._create_single_kv_cache(
                    batch_size=batch_size,
                    dtype=dtype,
                    device=device,
                )
                local_crossattn_cache = self._create_single_crossattn_cache()
                if cfg_rank == 0:
                    session_state.kv_cache1 = local_kv_cache
                    session_state.kv_cache_neg.clear()
                    session_state.crossattn_cache = local_crossattn_cache
                    session_state.crossattn_cache_neg.clear()
                else:
                    session_state.kv_cache1.clear()
                    session_state.kv_cache_neg = local_kv_cache
                    session_state.crossattn_cache.clear()
                    session_state.crossattn_cache_neg = local_crossattn_cache
            elif session_state.current_start_frame == 0:
                (
                    session_state.kv_cache1,
                    session_state.kv_cache_neg,
                ) = self._create_kv_caches(
                    batch_size=batch_size,
                    dtype=dtype,
                    device=device,
                )
                if (
                    not session_state.crossattn_cache
                    or not session_state.crossattn_cache_neg
                ):
                    (
                        session_state.crossattn_cache,
                        session_state.crossattn_cache_neg,
                    ) = self._create_crossattn_caches(
                        batch_size=batch_size,
                        dtype=dtype,
                        device=device,
                    )
            elif cfg_parallel:
                local_kv_cache = (
                    session_state.kv_cache1
                    if cfg_rank == 0
                    else session_state.kv_cache_neg
                )
                local_crossattn_cache = (
                    session_state.crossattn_cache
                    if cfg_rank == 0
                    else session_state.crossattn_cache_neg
                )
                if not local_kv_cache or not local_crossattn_cache:
                    raise RuntimeError(
                        "DreamZero streaming session is missing rank-local CFG "
                        f"cache state for cfg_rank={cfg_rank}"
                    )
            elif (
                not session_state.kv_cache1
                or not session_state.kv_cache_neg
                or not session_state.crossattn_cache
                or not session_state.crossattn_cache_neg
            ):
                raise RuntimeError(
                    "DreamZero streaming session is missing DiT cache state"
                )

            if cfg_parallel:
                kv_caches = [
                    session_state.kv_cache1
                    if cfg_rank == 0
                    else session_state.kv_cache_neg
                ]
                crossattn_caches = [
                    session_state.crossattn_cache
                    if cfg_rank == 0
                    else session_state.crossattn_cache_neg
                ]
            else:
                kv_caches = [
                    session_state.kv_cache1,
                    session_state.kv_cache_neg,
                ]
                crossattn_caches = [
                    session_state.crossattn_cache,
                    session_state.crossattn_cache_neg,
                ]
            debug_dit_calls = (
                []
                if os.environ.get("DREAMZERO_DEBUG_DIT_CALLS", "0") == "1"
                else None
            )

            if session_state.current_start_frame == 0:
                zero_timestep = torch.zeros(
                    [batch_size, 1], device=device, dtype=torch.int64
                )
                self._run_diffusion_steps(
                    noisy_input=latent_video,
                    timestep=zero_timestep,
                    action=None,
                    timestep_action=None,
                    state=None,
                    context=prompt_embs,
                    seq_len=frame_seqlen,
                    y=y_full[:, :, 0:1],
                    clip_feature=clip_feature,
                    kv_caches=kv_caches,
                    crossattn_caches=crossattn_caches,
                    current_start_frame=0,
                    update_kv_cache=True,
                    branch_indices=branch_indices,
                    debug_calls=debug_dit_calls,
                )
                session_state.current_start_frame = 1
            if session_state.current_start_frame != 1:
                current_start_frame = session_state.current_start_frame
                current_ref_latents = latent_video[:, :, -num_frame_per_block:]
                y_start = current_start_frame - num_frame_per_block
                if current_start_frame <= y_full.shape[2]:
                    y_prefill = y_full[:, :, y_start:current_start_frame]
                else:
                    y_prefill = y_full[:, :, -num_frame_per_block:]
                zero_timestep = torch.zeros(
                    [batch_size, num_frame_per_block],
                    device=device,
                    dtype=torch.int64,
                )
                self._run_diffusion_steps(
                    noisy_input=current_ref_latents,
                    timestep=zero_timestep,
                    action=None,
                    timestep_action=None,
                    state=None,
                    context=prompt_embs,
                    seq_len=seq_len,
                    y=y_prefill,
                    clip_feature=clip_feature,
                    kv_caches=kv_caches,
                    crossattn_caches=crossattn_caches,
                    current_start_frame=y_start,
                    update_kv_cache=True,
                    branch_indices=branch_indices,
                    debug_calls=debug_dit_calls,
                )
            if os.environ.get("DREAMZERO_SELF_CONTAINED_PREFILL_ONLY", "0") == "1":
                batch.dreamzero_prefill_only = True
                batch.dreamzero_action_pred = torch.zeros(
                    batch_size,
                    action_horizon,
                    action_dim,
                    device=device,
                    dtype=torch.float32,
                )
                if debug_dit_calls is not None:
                    batch.dreamzero_dit_debug_calls = debug_dit_calls
                batch.dreamzero_current_start_frame = (
                    session_state.current_start_frame
                )
                batch.dreamzero_cfg_rank = cfg_rank
                batch.dreamzero_cfg_world_size = cfg_world_size
                batch.dreamzero_cfg_branches_per_step = (
                    1 if cfg_parallel else len(prompt_embs)
                )
                batch.output = batch.dreamzero_action_pred
                return batch

            state = inputs.get("state")
            if state is None:
                state = torch.zeros(
                    batch_size, 1, max_state_dim, device=device, dtype=dtype
                )
            else:
                state = state.to(device=device, dtype=dtype)

            scheduler = self._new_unipc_scheduler()
            action_scheduler = self._new_unipc_scheduler()
            scheduler.set_timesteps(
                server_args.pipeline_config.num_inference_steps,
                device=device,
                shift=server_args.pipeline_config.flow_shift,
            )
            action_scheduler.set_timesteps(
                server_args.pipeline_config.num_inference_steps,
                device=device,
                shift=server_args.pipeline_config.flow_shift,
            )

            noisy_input = noise_obs.transpose(1, 2).contiguous()
            noisy_action = noise_action
            prev_predictions: list[
                tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            ] = []
            skip_state = {"countdown": 0}
            for step_index, video_timestep in enumerate(scheduler.timesteps):
                action_timestep = action_scheduler.timesteps[step_index]
                timestep = torch.ones(
                    [batch_size, num_frame_per_block],
                    device=device,
                    dtype=torch.int64,
                ) * video_timestep
                timestep_action = torch.ones(
                    [batch_size, action_horizon],
                    device=device,
                    dtype=torch.int64,
                ) * action_timestep
                y_start = session_state.current_start_frame
                y_end = y_start + num_frame_per_block
                y = y_full[:, :, y_start:y_end]
                if y.shape[2] < num_frame_per_block:
                    y = y_full[:, :, -num_frame_per_block:]

                should_run_model = self._should_run_model(
                    step_index=step_index,
                    current_timestep=video_timestep,
                    prev_predictions=prev_predictions,
                    dit_step_mask=server_args.pipeline_config.dit_step_mask,
                    dynamic_cache_schedule=getattr(
                        server_args.pipeline_config,
                        "dynamic_cache_schedule",
                        False,
                    ),
                    skip_state=skip_state,
                )
                if should_run_model:
                    predictions = self._run_diffusion_steps(
                        noisy_input=noisy_input.transpose(1, 2).contiguous(),
                        timestep=timestep,
                        action=noisy_action,
                        timestep_action=timestep_action,
                        state=state,
                        context=prompt_embs,
                        seq_len=seq_len,
                        y=y,
                        clip_feature=clip_feature,
                        kv_caches=kv_caches,
                        crossattn_caches=crossattn_caches,
                        current_start_frame=session_state.current_start_frame,
                        update_kv_cache=False,
                        branch_indices=branch_indices,
                        debug_calls=debug_dit_calls,
                    )
                    cfg_scale = server_args.pipeline_config.cfg_scale
                    if cfg_parallel:
                        local_video, local_action = predictions[0]
                        flow_pred, flow_pred_action_cond = (
                            self._combine_cfg_parallel_predictions(
                                video=local_video,
                                action=local_action,
                                cfg_scale=cfg_scale,
                                cfg_rank=cfg_rank,
                            )
                        )
                    else:
                        flow_pred_cond, flow_pred_action_cond = predictions[0]
                        flow_pred_uncond, _ = predictions[1]
                        flow_pred = flow_pred_uncond + cfg_scale * (
                            flow_pred_cond - flow_pred_uncond
                        )
                    prev_predictions.append(
                        (video_timestep, flow_pred, flow_pred_action_cond)
                    )
                    if len(prev_predictions) > 2:
                        prev_predictions.pop(0)
                else:
                    assert prev_predictions, (
                        "prev_predictions must be set when DreamZero skips a DiT step"
                    )
                    _, flow_pred, flow_pred_action_cond = prev_predictions[-1]

                noisy_input = self._scheduler_step(
                    scheduler,
                    model_output=flow_pred.transpose(1, 2),
                    timestep=video_timestep,
                    sample=noisy_input,
                    step_index=step_index,
                )
                noisy_action = self._scheduler_step(
                    action_scheduler,
                    model_output=flow_pred_action_cond,
                    timestep=action_timestep,
                    sample=noisy_action,
                    step_index=step_index,
                )

            batch.dreamzero_action_pred = noisy_action.float()
            if session_state.current_start_frame == 1:
                batch.dreamzero_video_pred = torch.cat(
                    [latent_video.transpose(1, 2), noisy_input], dim=1
                ).transpose(1, 2)
            else:
                batch.dreamzero_video_pred = noisy_input.transpose(1, 2)
            session_state.current_start_frame += num_frame_per_block
            session_state.latent_video = latent_video
            batch.dreamzero_kv_caches = kv_caches
            batch.dreamzero_crossattn_caches = crossattn_caches
            batch.dreamzero_current_start_frame = (
                session_state.current_start_frame
            )
            batch.dreamzero_cfg_rank = cfg_rank
            batch.dreamzero_cfg_world_size = cfg_world_size
            batch.dreamzero_cfg_branches_per_step = (
                1 if cfg_parallel else len(prompt_embs)
            )
            if debug_dit_calls is not None:
                batch.dreamzero_dit_debug_calls = debug_dit_calls
            batch.output = batch.dreamzero_action_pred
            return batch
