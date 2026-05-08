# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from sglang.multimodal_gen.configs.sample.sensenova_u1 import (
    SenseNovaU1PixelFlowCFG,
    resolve_sensenova_u1_pixel_flow_cfg,
)


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


class SenseNovaU1PixelFlowPreparer:
    def __init__(self, model: Any) -> None:
        self.model = model

    def forward(
        self,
        *,
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
        patch_size = int(self.model.config.vision_config.patch_size)
        merge_size = int(1 / float(self.model.config.downsample_ratio))
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

        device = _model_device(self.model)
        dtype = _model_dtype(self.model)
        seed = int(getattr(batch, "seed", None) or 0)
        generator = _session_torch_generator(
            context_metadata,
            seed=seed,
            device=device,
        )
        noise_scale = float(
            _noise_scale_for_image(self.model, grid_h=grid_h, grid_w=grid_w)
        )
        image_prediction = noise_scale * torch.randn(
            (1, 3, height, width),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        gen_grid_hw = torch.tensor([[grid_h, grid_w]], device=device, dtype=torch.long)
        timesteps = torch.linspace(0.0, 1.0, steps + 1, device=device)
        timesteps = _apply_time_schedule(
            self.model,
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
    prepared = SimpleNamespace(
        generation_input={
            "packed_seqlens": packed_seqlens,
            "packed_position_ids": indexes_image,
        },
        session_id=context.session_id,
        sidecar_role=context.sidecar_role,
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
