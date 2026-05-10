# SPDX-License-Identifier: Apache-2.0
"""Run SenseNova U1 pixel-flow against live SRT context state."""

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from PIL import Image

from sglang.multimodal_gen.configs.sample.sensenova_u1 import (
    SenseNovaU1PixelFlowCFG,
    resolve_sensenova_u1_pixel_flow_cfg,
)
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.omni.protocol import ContextOps, TemporaryForwardPrepared

_U1_T2I_CFG_UNCONDITION_ROLE = "u1_t2i_cfg_uncondition"
_U1_INTERLEAVE_TEXT_UNCONDITION_ROLE = "u1_interleave_text_uncondition"
_U1_EDIT_IMG_CONDITION_ROLE = "u1_edit_img_condition"
_U1_EDIT_UNCONDITION_ROLE = "u1_edit_uncondition"


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
    image_prediction: Any
    gen_grid_hw: Any
    timesteps: Any
    cfg: SenseNovaU1PixelFlowCFG
    condition: SenseNovaU1PixelFlowForwardContext
    img_condition: SenseNovaU1PixelFlowForwardContext | None
    uncondition: SenseNovaU1PixelFlowForwardContext | None


@dataclass(frozen=True, slots=True)
class SenseNovaU1GeneratedSegment:
    type: str
    image: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    commit_image: Any | None = None


@dataclass(frozen=True, slots=True)
class _SenseNovaU1GenerationContext:
    session_id: str
    position_count: int
    condition_path_role: str | None = None


class SenseNovaU1PixelFlowStage(PipelineStage):
    """Run U1 pixel-flow generation inside one stage-local execution boundary."""

    def __init__(self) -> None:
        super().__init__()

    @property
    def role_affinity(self):
        return RoleType.DENOISER

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
        forward_batch_provider = _require_forward_batch_provider(context_ops)
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
            forward_batch_provider=forward_batch_provider,
            prepared=prepared,
        )
        segment = self._decode(prepared, image_prediction)
        batch.generated_segment = segment
        batch.output = _image_to_numpy_batch(segment.image)
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
        steps = int(getattr(sampling_params, "num_inference_steps", None) or 0)
        if steps <= 0:
            raise ValueError(f"num_inference_steps must be positive, got {steps}")

        device = _model_device(model)
        dtype = _model_dtype(model)
        seed = _batch_seed(batch)
        generator = _session_torch_generator(
            context_metadata,
            seed=seed,
            device=device,
        )
        noise_scale = float(_noise_scale_for_image(model, grid_h=grid_h, grid_w=grid_w))
        image_prediction = noise_scale * torch.randn(
            (1, 3, height, width),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        gen_grid_hw = torch.tensor([[grid_h, grid_w]], device=device, dtype=torch.long)
        timesteps = torch.linspace(0.0, 1.0, steps + 1, device=device)
        timesteps = _apply_time_schedule(
            model,
            timesteps,
            image_seq_len=token_h * token_w,
            timestep_shift=float(getattr(sampling_params, "timestep_shift", 1.0)),
        )
        packed_seqlens = torch.tensor(
            [token_h * token_w], dtype=torch.int32, device=device
        )
        condition = _build_forward_context(
            u1_context,
            token_h=token_h,
            token_w=token_w,
            packed_seqlens=packed_seqlens,
            device=device,
        )
        img_condition = None
        if cfg_img_condition_u1_context is not None:
            img_condition = _build_forward_context(
                cfg_img_condition_u1_context,
                token_h=token_h,
                token_w=token_w,
                packed_seqlens=packed_seqlens,
                device=device,
            )
        uncondition = None
        if cfg_uncondition_u1_context is not None:
            uncondition = _build_forward_context(
                cfg_uncondition_u1_context,
                token_h=token_h,
                token_w=token_w,
                packed_seqlens=packed_seqlens,
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
            image_prediction=image_prediction,
            gen_grid_hw=gen_grid_hw,
            timesteps=timesteps,
            cfg=cfg,
            condition=condition,
            img_condition=img_condition,
            uncondition=uncondition,
        )

    def _denoise(
        self,
        *,
        model: Any,
        forward_batch_provider: Any,
        prepared: SenseNovaU1PixelFlowPrepared,
    ) -> Any:
        import torch

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
                forward_batch_provider=forward_batch_provider,
                forward_context=prepared.condition,
                image_embeds=image_embeds,
                timestep=timestep,
                z=z,
            )
            use_cfg = _should_apply_cfg(prepared.cfg, timestep)
            v_pred = self._combine_cfg_velocity(
                model=model,
                forward_batch_provider=forward_batch_provider,
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
        forward_batch_provider: Any,
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
                forward_batch_provider=forward_batch_provider,
                forward_context=_require_forward_context(prepared.img_condition),
                image_embeds=image_embeds,
                timestep=timestep,
                z=z,
            )
            return v_img_condition + cfg.text_scale * (v_condition - v_img_condition)
        if cfg.text_scale == cfg.img_scale:
            v_uncondition = self._predict_v(
                model=model,
                forward_batch_provider=forward_batch_provider,
                forward_context=_require_forward_context(prepared.uncondition),
                image_embeds=image_embeds,
                timestep=timestep,
                z=z,
            )
            return v_uncondition + cfg.text_scale * (v_condition - v_uncondition)

        v_img_condition = self._predict_v(
            model=model,
            forward_batch_provider=forward_batch_provider,
            forward_context=_require_forward_context(prepared.img_condition),
            image_embeds=image_embeds,
            timestep=timestep,
            z=z,
        )
        v_uncondition = self._predict_v(
            model=model,
            forward_batch_provider=forward_batch_provider,
            forward_context=_require_forward_context(prepared.uncondition),
            image_embeds=image_embeds,
            timestep=timestep,
            z=z,
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
        forward_batch_provider: Any,
        forward_context: SenseNovaU1PixelFlowForwardContext,
        image_embeds: Any,
        timestep: Any,
        z: Any,
    ) -> Any:
        forward_batch_context = forward_batch_provider(
            prepared=forward_context.prepared,
            generation_query_embeds=image_embeds,
            timestep=timestep,
        )
        forward_batch = getattr(
            forward_batch_context,
            "forward_batch",
            forward_batch_context,
        )
        try:
            return _predict_pixel_flow_from_srt(
                model,
                image_embeds=image_embeds,
                indexes_image=forward_context.indexes_image,
                forward_batch=forward_batch,
                timestep=timestep,
                z=z,
            )
        finally:
            release = getattr(forward_batch_context, "release", None)
            if callable(release):
                release()

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

    def _decode(
        self,
        prepared: SenseNovaU1PixelFlowPrepared,
        image_prediction: Any,
    ) -> SenseNovaU1GeneratedSegment:

        array = (
            (image_prediction[0].float() * 0.5 + 0.5)
            .clamp(0, 1)
            .permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy()
        )
        image = Image.fromarray((array * 255.0).round().astype(np.uint8), "RGB")
        commit_image = {
            "pixel_values": image_prediction.detach().to(torch.bfloat16).cpu(),
            "value_range": "minus_one_to_one",
            "grid_hw": prepared.gen_grid_hw[:1].detach().cpu(),
        }
        cfg = prepared.cfg
        return SenseNovaU1GeneratedSegment(
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
                "cfg_text_scale": cfg.text_scale,
                "cfg_img_scale": cfg.img_scale,
                "cfg_renorm_type": cfg.renorm_type if cfg.needs_cfg else "none",
            },
            commit_image=commit_image,
        )


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
    mode = getattr(sampling_params, "omni_generation_mode", None)
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


def _require_context_ops(batch: Req) -> ContextOps:
    context_ops = batch.omni_context_ops
    if context_ops is None:
        raise RuntimeError("SenseNova U1 pixel-flow requires batch.omni_context_ops")
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
    device: Any,
) -> SenseNovaU1PixelFlowForwardContext:
    position_count = int(context.position_count)
    indexes_image = _build_t2i_image_indexes(
        token_h=token_h,
        token_w=token_w,
        text_len=position_count,
        device=device,
    )
    prepared = TemporaryForwardPrepared(
        generation_input={
            "packed_seqlens": packed_seqlens,
            "packed_position_ids": indexes_image,
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


def _session_torch_generator(
    metadata: dict[str, Any],
    *,
    seed: int,
    device: Any,
) -> Any:
    import torch

    key = "_u1_pixel_flow_generator"
    device_str = str(device)
    state = metadata.get(key)
    if (
        isinstance(state, dict)
        and state.get("seed") == int(seed)
        and state.get("device") == device_str
        and isinstance(state.get("generator"), torch.Generator)
    ):
        return state["generator"]
    generator = torch.Generator(device=device).manual_seed(int(seed))
    metadata[key] = {
        "seed": int(seed),
        "device": device_str,
        "generator": generator,
    }
    return generator


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


def _require_model(context_ops: ContextOps) -> Any:
    return context_ops.get_model()


def _require_forward_batch_provider(context_ops: ContextOps) -> Any:
    return context_ops.build_temporary_forward_batch


def _should_apply_cfg(cfg: SenseNovaU1PixelFlowCFG, timestep: Any) -> bool:
    timestep_value = float(timestep)
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
    forward_batch: Any,
    timestep: Any,
    z: Any,
) -> Any:
    batch_size, image_token_num = image_embeds.shape[:2]
    hidden_states = model.language_model.forward_u1_gen_embeds(
        input_embeds=image_embeds.reshape(-1, image_embeds.shape[-1]),
        positions=indexes_image,
        forward_batch=forward_batch,
    ).view(batch_size, image_token_num, -1)
    x_pred = model.fm_modules["fm_head"](hidden_states).view(
        batch_size,
        image_token_num,
        -1,
    )

    t = timestep.to(device=z.device, dtype=z.dtype)
    return (x_pred - z) / (1 - t).clamp_min(float(getattr(model.config, "t_eps", 0.02)))


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
