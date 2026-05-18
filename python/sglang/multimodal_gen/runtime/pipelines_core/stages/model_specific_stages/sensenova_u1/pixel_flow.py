# SPDX-License-Identifier: Apache-2.0
"""Run SenseNova U1 pixel-flow against live SRT context state."""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image

from sglang.multimodal_gen.configs.sample.sensenova_u1 import (
    SenseNovaU1PixelFlowCFG,
    resolve_sensenova_u1_pixel_flow_cfg,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.omni.core.protocol import (
    ColocatedContextOps,
    ContextOps,
    GeneratedSegment,
    TemporaryForwardPrepared,
)

_U1_T2I_CFG_UNCONDITION_ROLE = "u1_t2i_cfg_uncondition"
_U1_INTERLEAVE_TEXT_UNCONDITION_ROLE = "u1_interleave_text_uncondition"
_U1_EDIT_IMG_CONDITION_ROLE = "u1_edit_img_condition"
_U1_EDIT_UNCONDITION_ROLE = "u1_edit_uncondition"
_U1_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_U1_IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True, slots=True)
class SenseNovaU1PixelFlowForwardContext:
    prepared: Any
    indexes_image: Any
    position_count: int


@dataclass(slots=True)
class SenseNovaU1PixelFlowPrepared:
    width: int
    height: int
    patch_size: int
    merge_size: int
    token_h: int
    token_w: int
    grid_h: int
    grid_w: int
    steps: int
    seed: int
    noise_scale: float
    t_eps: float
    image_prediction: Any
    gen_grid_hw: Any
    timesteps: Any
    cfg_step_mask: tuple[bool, ...]
    cfg: SenseNovaU1PixelFlowCFG
    commit_generated_image: bool
    condition: SenseNovaU1PixelFlowForwardContext
    img_condition: SenseNovaU1PixelFlowForwardContext | None
    uncondition: SenseNovaU1PixelFlowForwardContext | None


@dataclass(frozen=True, slots=True)
class SenseNovaU1PixelFlowDenoiseOutput:
    prepared: SenseNovaU1PixelFlowPrepared
    image_prediction: Any


@dataclass(frozen=True, slots=True)
class _SenseNovaU1GenerationContext:
    session_id: str
    position_count: int
    condition_path_role: str | None = None


class SenseNovaU1PixelFlowStage(DenoisingStage):
    """Run U1 pixel-flow denoising with SRT-owned transformer state."""

    def __init__(self) -> None:
        # u1 borrows transformer/session state from the colocated srt runtime
        PipelineStage.__init__(self)

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        return []

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        return PipelineStage.verify_input(self, batch, server_args)

    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        """
        different from standard multimodal_gen pipeline, omni pipeline may return a Req, carrying some intermediate states
        """
        context_ops = _require_context_ops(batch)
        model = _require_model(context_ops)
        temporary_forward_runner = _require_temporary_forward_runner(context_ops)
        (
            u1_context,
            cfg_img_condition_u1_context,
            cfg_uncondition_u1_context,
        ) = _resolve_u1_contexts(context_ops=context_ops, batch=batch)
        prepared = self._prepare(
            model=model,
            context_metadata=dict(context_ops.metadata),
            batch=batch,
            u1_context=u1_context,
            cfg_img_condition_u1_context=cfg_img_condition_u1_context,
            cfg_uncondition_u1_context=cfg_uncondition_u1_context,
        )
        image_prediction = self._denoise(
            model=model,
            temporary_forward_runner=temporary_forward_runner,
            prepared=prepared,
        )
        batch.sensenova_u1_pixel_flow = SenseNovaU1PixelFlowDenoiseOutput(
            prepared=prepared,
            image_prediction=image_prediction,
        )
        return batch

    def _prepare(
        self,
        *,
        model: Any,
        context_metadata: dict[str, Any],
        batch: Any,
        u1_context: Any,
        cfg_img_condition_u1_context: Any | None = None,
        cfg_uncondition_u1_context: Any | None = None,
    ) -> SenseNovaU1PixelFlowPrepared:
        import torch

        sampling_params = batch.sampling_params
        cfg = resolve_sensenova_u1_pixel_flow_cfg(sampling_params)
        if (
            cfg.needs_img_condition
            and cfg_img_condition_u1_context is None
            and not cfg.needs_uncondition
            and cfg_uncondition_u1_context is not None
        ):
            cfg_img_condition_u1_context = cfg_uncondition_u1_context
            cfg_uncondition_u1_context = None
        if cfg.needs_img_condition and cfg_img_condition_u1_context is None:
            raise RuntimeError(
                "SenseNova U1 pixel-flow CFG requires an image-condition context"
            )
        if cfg.needs_uncondition and cfg_uncondition_u1_context is None:
            raise RuntimeError(
                "SenseNova U1 pixel-flow CFG requires an uncondition context"
            )

        width, height = _batch_image_size(batch)
        patch_size = int(model.config.vision_config.patch_size)
        merge_size = int(1 / float(model.config.downsample_ratio))
        divisor = patch_size * merge_size
        if width % divisor or height % divisor:
            raise ValueError(
                "SenseNova U1 pixel-flow image size must be divisible by "
                f"{divisor}, got {width}x{height}"
            )

        token_h = height // divisor
        token_w = width // divisor
        grid_h = height // patch_size
        grid_w = width // patch_size
        steps = int(sampling_params.num_inference_steps or 0)
        if steps <= 0:
            raise ValueError(f"num_inference_steps must be positive, got {steps}")
        t_eps = float(sampling_params.t_eps)
        commit_generated_image = sampling_params.omni_generation_mode == "interleave"

        device = _model_device(model)
        seed = _batch_seed(batch)
        generator = _new_torch_generator(seed=seed, device=device)
        noise_scale = float(_noise_scale_for_image(model, grid_h=grid_h, grid_w=grid_w))
        dtype = _model_dtype(model)
        image_prediction = noise_scale * torch.randn(
            (1, 3, height, width),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        # keep shape metadata on cpu so vision embedding loop bounds do not sync per step
        gen_grid_hw = torch.tensor([[grid_h, grid_w]], dtype=torch.long)
        timestep_shift = float(sampling_params.timestep_shift)
        timesteps = torch.linspace(0.0, 1.0, steps + 1, device=device)
        timesteps = _apply_time_schedule(
            model,
            timesteps,
            image_seq_len=token_h * token_w,
            timestep_shift=timestep_shift,
        )
        cfg_step_mask = _build_cfg_step_mask(
            model=model,
            cfg=cfg,
            steps=steps,
            image_seq_len=token_h * token_w,
            timestep_shift=timestep_shift,
        )
        packed_seqlens = torch.tensor(
            [token_h * token_w], dtype=torch.int32, device=device
        )
        condition = _build_forward_context(
            u1_context,
            token_h=token_h,
            token_w=token_w,
            packed_seqlens=packed_seqlens,
            attention_math_mode=context_metadata.get("attention_math_mode"),
            device=device,
        )
        img_condition = None
        if cfg_img_condition_u1_context is not None:
            img_condition = _build_forward_context(
                cfg_img_condition_u1_context,
                token_h=token_h,
                token_w=token_w,
                packed_seqlens=packed_seqlens,
                attention_math_mode=context_metadata.get("attention_math_mode"),
                device=device,
            )
        uncondition = None
        if cfg_uncondition_u1_context is not None:
            uncondition = _build_forward_context(
                cfg_uncondition_u1_context,
                token_h=token_h,
                token_w=token_w,
                packed_seqlens=packed_seqlens,
                attention_math_mode=context_metadata.get("attention_math_mode"),
                device=device,
            )

        return SenseNovaU1PixelFlowPrepared(
            width=width,
            height=height,
            patch_size=patch_size,
            merge_size=merge_size,
            token_h=token_h,
            token_w=token_w,
            grid_h=grid_h,
            grid_w=grid_w,
            steps=steps,
            seed=seed,
            noise_scale=noise_scale,
            t_eps=t_eps,
            image_prediction=image_prediction,
            gen_grid_hw=gen_grid_hw,
            timesteps=timesteps,
            cfg_step_mask=cfg_step_mask,
            cfg=cfg,
            commit_generated_image=commit_generated_image,
            condition=condition,
            img_condition=img_condition,
            uncondition=uncondition,
        )

    def _denoise(
        self,
        *,
        model: Any,
        temporary_forward_runner: Any,
        prepared: SenseNovaU1PixelFlowPrepared,
    ) -> Any:

        image_prediction = prepared.image_prediction

        for step_i in range(prepared.steps):
            timestep = prepared.timesteps[step_i]
            next_timestep = prepared.timesteps[step_i + 1]
            z = _patchify(
                image_prediction,
                prepared.patch_size * prepared.merge_size,
            )
            image_input = _patchify(
                image_prediction,
                prepared.patch_size,
                channel_first=True,
            )
            image_embeds = model.extract_feature(
                image_input.view(prepared.grid_h * prepared.grid_w, -1),
                gen_model=True,
                grid_hw=prepared.gen_grid_hw,
            ).view(1, prepared.token_h * prepared.token_w, -1)
            timestep_values = timestep.expand(prepared.token_h * prepared.token_w)
            timestep_embeddings = model.fm_modules["timestep_embedder"](
                timestep_values
            ).view(1, prepared.token_h * prepared.token_w, -1)
            if getattr(model.config, "add_noise_scale_embedding", False):
                noise_values = torch.full_like(
                    timestep_values,
                    prepared.noise_scale / float(model.config.noise_scale_max_value),
                )
                timestep_embeddings = timestep_embeddings + model.fm_modules[
                    "noise_scale_embedder"
                ](noise_values).view(1, prepared.token_h * prepared.token_w, -1)
            image_embeds = image_embeds + timestep_embeddings

            v_condition = self._predict_v(
                model=model,
                temporary_forward_runner=temporary_forward_runner,
                forward_context=prepared.condition,
                image_embeds=image_embeds,
                timestep=timestep,
                z=z,
                t_eps=prepared.t_eps,
            )
            use_cfg = prepared.cfg_step_mask[step_i]
            v_pred = self._combine_cfg_velocity(
                model=model,
                temporary_forward_runner=temporary_forward_runner,
                prepared=prepared,
                image_embeds=image_embeds,
                timestep=timestep,
                z=z,
                v_condition=v_condition,
                use_cfg=use_cfg,
            )
            if prepared.cfg.needs_cfg and use_cfg:
                v_pred = self._apply_cfg_renorm(
                    v_condition=v_condition,
                    v_pred=v_pred,
                    cfg=prepared.cfg,
                )

            z = z + (next_timestep - timestep) * v_pred
            image_prediction = _unpatchify(
                z,
                prepared.patch_size * prepared.merge_size,
                prepared.height,
                prepared.width,
            )
        return image_prediction

    def _combine_cfg_velocity(
        self,
        *,
        model: Any,
        temporary_forward_runner: Any,
        prepared: SenseNovaU1PixelFlowPrepared,
        image_embeds: Any,
        timestep: Any,
        z: Any,
        v_condition: Any,
        use_cfg: bool,
    ) -> Any:
        cfg = prepared.cfg
        if not use_cfg or not cfg.needs_cfg:
            return v_condition
        if cfg.img_scale == 1.0:
            v_img_condition = self._predict_v(
                model=model,
                temporary_forward_runner=temporary_forward_runner,
                forward_context=_require_forward_context(prepared.img_condition),
                image_embeds=image_embeds,
                timestep=timestep,
                z=z,
                t_eps=prepared.t_eps,
            )
            return v_img_condition + cfg.text_scale * (v_condition - v_img_condition)
        if cfg.text_scale == cfg.img_scale:
            v_uncondition = self._predict_v(
                model=model,
                temporary_forward_runner=temporary_forward_runner,
                forward_context=_require_forward_context(prepared.uncondition),
                image_embeds=image_embeds,
                timestep=timestep,
                z=z,
                t_eps=prepared.t_eps,
            )
            return v_uncondition + cfg.text_scale * (v_condition - v_uncondition)

        v_img_condition = self._predict_v(
            model=model,
            temporary_forward_runner=temporary_forward_runner,
            forward_context=_require_forward_context(prepared.img_condition),
            image_embeds=image_embeds,
            timestep=timestep,
            z=z,
            t_eps=prepared.t_eps,
        )
        v_uncondition = self._predict_v(
            model=model,
            temporary_forward_runner=temporary_forward_runner,
            forward_context=_require_forward_context(prepared.uncondition),
            image_embeds=image_embeds,
            timestep=timestep,
            z=z,
            t_eps=prepared.t_eps,
        )
        return (
            v_uncondition
            + cfg.text_scale * (v_condition - v_img_condition)
            + cfg.img_scale * (v_img_condition - v_uncondition)
        )

    def _predict_v(
        self,
        *,
        model: Any,
        temporary_forward_runner: Any,
        forward_context: SenseNovaU1PixelFlowForwardContext,
        image_embeds: Any,
        timestep: Any,
        z: Any,
        t_eps: float,
    ) -> Any:
        def run_forward(forward_batch: Any) -> Any:
            # 1. each noise step calls the AR model with temporary SRT KV
            return _predict_pixel_flow_from_srt(
                model,
                image_embeds=image_embeds,
                indexes_image=forward_context.indexes_image,
                temp_forward_batch=forward_batch,
                timestep=timestep,
                z=z,
                t_eps=t_eps,
            )

        # 2. srt owns build/forward/release so forward batch does not cross threads
        return temporary_forward_runner(
            prepared=forward_context.prepared,
            forward=run_forward,
        )

    @staticmethod
    def _apply_cfg_renorm(
        *,
        v_condition: Any,
        v_pred: Any,
        cfg: SenseNovaU1PixelFlowCFG,
    ) -> Any:
        cfg_renorm_type = cfg.renorm_type
        if cfg_renorm_type == "none":
            return v_pred
        if cfg_renorm_type == "global":
            norm_v_condition = v_condition.norm(dim=(1, 2), keepdim=True)
            norm_v_cfg = v_pred.norm(dim=(1, 2), keepdim=True)
        elif cfg_renorm_type == "channel":
            norm_v_condition = v_condition.norm(dim=-1, keepdim=True)
            norm_v_cfg = v_pred.norm(dim=-1, keepdim=True)
        else:
            raise ValueError(
                "Unsupported SenseNova U1 pixel-flow CFG renorm type: "
                f"{cfg_renorm_type}"
            )
        scale = (norm_v_condition / (norm_v_cfg + 1e-8)).clamp(
            min=cfg.renorm_min, max=1.0
        )
        return v_pred * scale


class SenseNovaU1PixelFlowDecodeStage(PipelineStage):
    """Finalize U1 pixel-flow tensors into an omni image segment."""

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        return []

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        return PipelineStage.verify_input(self, batch, server_args)

    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        context_ops = _require_context_ops(batch)
        model = _require_model(context_ops)
        denoise_output = _require_denoise_output(batch)
        segment = self._finalize_image_segment(
            model=model,
            prepared=denoise_output.prepared,
            image_prediction=denoise_output.image_prediction,
        )
        batch.generated_segment = segment
        batch.output = _image_to_numpy_batch(segment.image)
        return batch

    def _finalize_image_segment(
        self,
        *,
        model: Any,
        prepared: SenseNovaU1PixelFlowPrepared,
        image_prediction: Any,
    ) -> GeneratedSegment:
        array = (
            (image_prediction[0].float() * 0.5 + 0.5)
            .clamp(0, 1)
            .permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy()
        )
        image = Image.fromarray((array * 255.0).round().astype(np.uint8), "RGB")
        commit_image = None
        if prepared.commit_generated_image:
            # commit reuses final vision embeddings instead of re-encoding CPU pixels
            commit_image = {
                "precomputed_embeddings": _extract_final_image_embeddings(
                    model=model,
                    prepared=prepared,
                    image_prediction=image_prediction,
                ),
                "grid_hw": prepared.gen_grid_hw[:1],
                "pad_hash": id(image_prediction),
            }
        cfg = prepared.cfg
        return GeneratedSegment(
            type="image",
            image=image,
            metadata={
                "generation_kind": "pixel_flow",
                "native_context_pixel_flow": True,
                "temporary_context_kv": True,
                "timesteps": prepared.steps,
                "seed": prepared.seed,
                "width": prepared.width,
                "height": prepared.height,
                "grid": (prepared.token_h, prepared.token_w),
                "generation_position_start": prepared.condition.position_count,
                "condition_position_count": prepared.condition.position_count,
                "cfg_img_condition_position_count": _forward_context_position(
                    prepared.img_condition
                ),
                "cfg_uncondition_position_count": _forward_context_position(
                    prepared.uncondition
                ),
                "noise_scale": prepared.noise_scale,
                "t_eps": prepared.t_eps,
                "cfg_text_scale": cfg.text_scale,
                "cfg_img_scale": cfg.img_scale,
                "cfg_renorm_type": cfg.renorm_type if cfg.needs_cfg else "none",
            },
            commit_payload=commit_image,
        )


def _require_denoise_output(batch: Req) -> SenseNovaU1PixelFlowDenoiseOutput:
    return batch.sensenova_u1_pixel_flow


def _resolve_u1_contexts(
    *,
    context_ops: ContextOps,
    batch: Req,
) -> tuple[
    _SenseNovaU1GenerationContext,
    _SenseNovaU1GenerationContext | None,
    _SenseNovaU1GenerationContext | None,
]:
    u1_context = _require_context(
        context_ops,
        "SenseNova U1 pixel-flow has no context position count",
    )
    cfg_img_condition_context = None
    cfg_uncondition_context = None
    sampling_params = batch.sampling_params
    mode = sampling_params.omni_generation_mode
    cfg = resolve_sensenova_u1_pixel_flow_cfg(sampling_params)

    t2i_uncondition_role = context_ops.get_role(
        "t2i_cfg_uncondition_role",
        _U1_T2I_CFG_UNCONDITION_ROLE,
    )
    interleave_text_uncondition_role = context_ops.get_role(
        "interleave_text_uncondition_role",
        _U1_INTERLEAVE_TEXT_UNCONDITION_ROLE,
    )
    edit_img_condition_role = context_ops.get_role(
        "edit_img_condition_role",
        _U1_EDIT_IMG_CONDITION_ROLE,
    )
    edit_uncondition_role = context_ops.get_role(
        "edit_uncondition_role",
        _U1_EDIT_UNCONDITION_ROLE,
    )

    # condition path roles select the CFG branch whose SRT KV is read during denoise
    if mode == "edit":
        if cfg.needs_img_condition:
            cfg_img_condition_context = _require_context(
                context_ops,
                "SenseNova U1 edit image CFG requires condition path position count",
                edit_img_condition_role,
            )
        if cfg.needs_uncondition:
            cfg_uncondition_context = _require_context(
                context_ops,
                "SenseNova U1 edit uncondition CFG requires condition path position count",
                edit_uncondition_role,
            )
    elif mode == "interleave":
        if cfg.needs_img_condition:
            cfg_img_condition_context = _require_context(
                context_ops,
                "SenseNova U1 interleave text CFG requires condition path position count",
                interleave_text_uncondition_role,
            )
        if cfg.needs_uncondition:
            cfg_uncondition_context = _require_context(
                context_ops,
                "SenseNova U1 interleave image CFG requires condition path position count",
                t2i_uncondition_role,
            )
    elif cfg.text_scale > 1.0:
        cfg_img_condition_context = _require_context(
            context_ops,
            "SenseNova U1 pixel-flow CFG requires condition path position count",
            t2i_uncondition_role,
        )
    return u1_context, cfg_img_condition_context, cfg_uncondition_context


def _require_context(
    context_ops: ContextOps,
    message: str,
    condition_path_role: str | None = None,
) -> _SenseNovaU1GenerationContext:
    position_count = context_ops.get_position_count(
        condition_path_role=condition_path_role
    )
    if position_count is None:
        suffix = (
            f" condition path {condition_path_role}"
            if condition_path_role is not None
            else ""
        )
        raise RuntimeError(f"{message} for context {context_ops.session_id}{suffix}")
    return _SenseNovaU1GenerationContext(
        session_id=context_ops.session_id,
        condition_path_role=condition_path_role,
        position_count=int(position_count),
    )


def _require_context_ops(batch: Req) -> ColocatedContextOps:
    context_ops = batch.omni_context_ops
    if context_ops is None:
        raise RuntimeError("SenseNova U1 pixel-flow requires batch.omni_context_ops")
    if not isinstance(context_ops, ColocatedContextOps):
        raise TypeError(
            "SenseNova U1 pixel-flow requires colocated context ops because it "
            "borrows the live SRT model and temporary KV slots during denoise"
        )
    if context_ops.generation_kind != "pixel_flow":
        raise ValueError(
            "SenseNova U1 pixel-flow requires generation_kind='pixel_flow', got "
            f"{context_ops.generation_kind!r}"
        )
    return context_ops


def _build_forward_context(
    context: Any,
    *,
    token_h: int,
    token_w: int,
    packed_seqlens: Any,
    attention_math_mode: str | None,
    device: Any,
) -> SenseNovaU1PixelFlowForwardContext:
    position_count = int(context.position_count)
    indexes_image = _build_t2i_image_indexes(
        token_h=token_h,
        token_w=token_w,
        text_len=position_count,
        device=device,
    )
    # srt owns the true kv token count; u1 logical positions collapse image tokens
    prepared = TemporaryForwardPrepared(
        generation_input={
            "packed_seqlens": packed_seqlens,
            "packed_position_ids": indexes_image,
            "extend_num_tokens": token_h * token_w,
            "attention_math_mode": attention_math_mode,
            "synchronize_before_release": True,
        },
        srt_session_id=context.session_id,
        condition_path_role=context.condition_path_role,
    )
    return SenseNovaU1PixelFlowForwardContext(
        prepared=prepared,
        indexes_image=indexes_image,
        position_count=position_count,
    )


def _batch_image_size(batch: Any) -> tuple[int, int]:
    sampling_params = batch.sampling_params
    height = _first_int(
        getattr(batch, "height", None),
        getattr(sampling_params, "height", None),
        default=1024,
    )
    width = _first_int(
        getattr(batch, "width", None),
        getattr(sampling_params, "width", None),
        default=1024,
    )
    return width, height


def _build_t2i_image_indexes(
    *,
    token_h: int,
    token_w: int,
    text_len: int,
    device: Any,
) -> Any:
    import torch

    t_image = torch.full(
        (token_h * token_w,),
        int(text_len),
        dtype=torch.long,
        device=device,
    )
    idx = torch.arange(token_h * token_w, device=device, dtype=torch.long)
    h_image = idx // token_w
    w_image = idx % token_w
    return torch.stack([t_image, h_image, w_image], dim=0)


def _apply_time_schedule(
    model: Any,
    timesteps: Any,
    *,
    image_seq_len: int,
    timestep_shift: float,
) -> Any:
    import torch

    sigma = 1 - timesteps
    cfg = model.config
    schedule = str(cfg.time_schedule)
    if timestep_shift != 1:
        schedule = "standard"
    if schedule == "standard":
        shift = float(timestep_shift)
        sigma = shift * sigma / (1 + (shift - 1) * sigma)
    elif schedule == "dynamic":
        mu = _calculate_dynamic_mu(model, image_seq_len)
        mu_t = timesteps.new_tensor(mu)
        time_shift_type = str(cfg.time_shift_type)
        if time_shift_type == "exponential":
            shift = torch.exp(mu_t)
            sigma = shift * sigma / (1 + (shift - 1) * sigma)
        elif time_shift_type == "linear":
            sigma = mu_t / (mu_t + (1 / sigma - 1))
        else:
            raise ValueError(
                f"Unsupported SenseNova U1 time_shift_type: {time_shift_type}"
            )
    else:
        raise ValueError(f"Unsupported SenseNova U1 time_schedule: {schedule}")
    return 1 - sigma


def _build_cfg_step_mask(
    *,
    model: Any,
    cfg: SenseNovaU1PixelFlowCFG,
    steps: int,
    image_seq_len: int,
    timestep_shift: float,
) -> tuple[bool, ...]:
    if not cfg.needs_cfg:
        return (False,) * steps
    # cfg boundaries are control flow; compute them on cpu to avoid per-step device sync
    timesteps = torch.linspace(0.0, 1.0, steps + 1)
    timesteps = _apply_time_schedule(
        model,
        timesteps,
        image_seq_len=image_seq_len,
        timestep_shift=timestep_shift,
    )
    return tuple(
        _should_apply_cfg_value(cfg, float(timesteps[i])) for i in range(steps)
    )


def _noise_scale_for_image(model: Any, *, grid_h: int, grid_w: int) -> float:
    cfg = model.config
    merge_size = int(1 / float(cfg.downsample_ratio))
    noise_scale = float(cfg.noise_scale)
    noise_scale_mode = str(cfg.noise_scale_mode)
    if noise_scale_mode in {"resolution", "dynamic", "dynamic_sqrt"}:
        base = float(cfg.noise_scale_base_image_seq_len)
        scale = math.sqrt((grid_h * grid_w) / (merge_size**2) / base)
        noise_scale = scale * float(cfg.noise_scale)
        if noise_scale_mode == "dynamic_sqrt":
            noise_scale = math.sqrt(noise_scale)
    return min(noise_scale, float(cfg.noise_scale_max_value))


def _calculate_dynamic_mu(model: Any, image_seq_len: int) -> float:
    cfg = model.config
    denom = int(cfg.max_image_seq_len) - int(cfg.base_image_seq_len)
    if denom == 0:
        return float(cfg.base_shift)
    slope = (float(cfg.max_shift) - float(cfg.base_shift)) / denom
    bias = float(cfg.base_shift) - slope * int(cfg.base_image_seq_len)
    return float(image_seq_len) * slope + bias


def _new_torch_generator(*, seed: int, device: Any) -> Any:
    import torch

    return torch.Generator(device=device).manual_seed(int(seed))


def _batch_seed(batch: Any) -> int:
    seeds = getattr(batch, "seeds", None)
    if seeds:
        return int(seeds[0])
    seed = getattr(batch, "seed", None)
    if isinstance(seed, list):
        if not seed:
            raise ValueError("SenseNova U1 seed list must not be empty")
        return int(seed[0])
    if seed is not None:
        return int(seed)
    return 0


def _first_int(*values: Any, default: int) -> int:
    for value in values:
        if value is not None:
            return int(value)
    return int(default)


def _model_device(model: Any) -> Any:
    vision_model = getattr(model, "vision_model", None)
    device = getattr(vision_model, "device", None)
    if device is not None:
        return device
    return next(model.parameters()).device


def _model_dtype(model: Any) -> Any:
    vision_model = getattr(model, "vision_model", None)
    dtype = getattr(vision_model, "dtype", None)
    if dtype is not None:
        return dtype
    return next(model.parameters()).dtype


def _require_model(context_ops: ColocatedContextOps) -> Any:
    return context_ops.get_model()


def _require_temporary_forward_runner(context_ops: ColocatedContextOps) -> Any:
    return context_ops.run_temporary_forward


def _should_apply_cfg_value(
    cfg: SenseNovaU1PixelFlowCFG, timestep_value: float
) -> bool:
    if cfg.start == 0.0:
        return 0.0 <= timestep_value < cfg.end
    return cfg.start < timestep_value < cfg.end


def _require_forward_context(
    context: SenseNovaU1PixelFlowForwardContext | None,
) -> SenseNovaU1PixelFlowForwardContext:
    if context is None:
        raise RuntimeError("SenseNova U1 pixel-flow CFG forward context is missing")
    return context


def _patchify(images: Any, patch_size: int, *, channel_first: bool = False) -> Any:
    import torch

    h, w = images.shape[2] // patch_size, images.shape[3] // patch_size
    x = images.reshape(images.shape[0], 3, h, patch_size, w, patch_size)
    if channel_first:
        x = torch.einsum("nchpwq->nhwcpq", x)
    else:
        x = torch.einsum("nchpwq->nhwpqc", x)
    return x.reshape(images.shape[0], h * w, patch_size**2 * 3)


def _unpatchify(
    x: Any,
    patch_size: int,
    h: int | None = None,
    w: int | None = None,
) -> Any:
    import torch

    if h is None or w is None:
        h = w = int(x.shape[1] ** 0.5)
    else:
        h = h // patch_size
        w = w // patch_size
    x = x.reshape(x.shape[0], h, w, patch_size, patch_size, 3)
    x = torch.einsum("nhwpqc->nchpwq", x)
    return x.reshape(x.shape[0], 3, h * patch_size, w * patch_size)


def _predict_pixel_flow_from_srt(
    model: Any,
    *,
    image_embeds: Any,
    indexes_image: Any,
    temp_forward_batch: Any,
    timestep: Any,
    z: Any,
    t_eps: float,
) -> Any:
    batch_size, image_token_num = image_embeds.shape[:2]
    # forward with temporary forward batch
    hidden_states = model.language_model.forward_u1_gen_embeds(
        input_embeds=image_embeds.reshape(-1, image_embeds.shape[-1]),
        positions=indexes_image,
        forward_batch=temp_forward_batch,
    ).view(batch_size, image_token_num, -1)
    x_pred = model.fm_modules["fm_head"](hidden_states).view(
        batch_size,
        image_token_num,
        -1,
    )

    t = timestep.to(device=z.device)
    return (x_pred - z) / (1 - t).clamp_min(float(t_eps))


def _extract_final_image_embeddings(
    *,
    model: Any,
    prepared: SenseNovaU1PixelFlowPrepared,
    image_prediction: Any,
) -> Any:
    pixel_values = (image_prediction[0].float() * 0.5 + 0.5).clamp(0, 1)
    mean = pixel_values.new_tensor(_U1_IMAGENET_MEAN).view(3, 1, 1)
    std = pixel_values.new_tensor(_U1_IMAGENET_STD).view(3, 1, 1)
    image_input = _patchify(
        ((pixel_values - mean) / std).unsqueeze(0),
        prepared.patch_size,
        channel_first=True,
    )
    embeddings = model.extract_feature(
        image_input.view(prepared.grid_h * prepared.grid_w, -1),
        gen_model=False,
        grid_hw=prepared.gen_grid_hw,
    )
    return embeddings.view(prepared.token_h * prepared.token_w, -1).detach()


def _forward_context_position(context: Any | None) -> int | None:
    if context is None:
        return None
    return context.position_count


def _image_to_numpy_batch(image: Any) -> Any:
    import numpy as np
    from PIL import Image

    if isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"))
    else:
        array = np.asarray(image)
    if array.ndim == 3:
        array = array[None, ...]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array
