# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image

from sglang.multimodal_gen.configs.sensenova_u1 import (
    DEFAULT_CFG_INTERVAL,
    DEFAULT_CFG_NORM,
    DEFAULT_ENABLE_TIMESTEP_SHIFT,
    DEFAULT_T_EPS,
    DEFAULT_THINK_MODE,
    DEFAULT_TIMESTEP_SHIFT,
    SENSENOVA_U1_REQUEST_EXTRA_KEY,
    SENSENOVA_U1_RESOLUTION_ALIGNMENT,
)
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify.utils import (
    smart_resize,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
    OutputBatch,
    Req,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.vision import load_image

DEFAULT_INPUT_MAX_PIXELS = 2048 * 2048
MIN_INPUT_MAX_PIXELS = 512 * 512


def _denorm_sensenova_output(x: torch.Tensor) -> torch.Tensor:
    """Convert SenseNova's normalized image tensor from [-1, 1] to [0, 1]."""
    return ((x.float() + 1.0) * 0.5).clamp(0, 1)


def _auto_input_max_pixels(num_images: int) -> int:
    if num_images <= 0:
        raise ValueError(
            "SenseNova-U1 image editing requires at least one input image."
        )
    full_resolution_image_budget = 2
    if num_images <= full_resolution_image_budget:
        return DEFAULT_INPUT_MAX_PIXELS
    total_budget = full_resolution_image_budget * DEFAULT_INPUT_MAX_PIXELS
    return max(MIN_INPUT_MAX_PIXELS, total_budget // num_images)


def _flatten_rgba_to_rgb(image: Image.Image) -> Image.Image:
    if image.mode != "RGBA":
        return image.convert("RGB")
    background = Image.new("RGB", image.size, (255, 255, 255))
    background.paste(image, mask=image.split()[3])
    return background


def _resize_input_to_budget(
    image: Image.Image,
    *,
    do_resize: bool,
    input_max_pixels: int | None,
) -> Image.Image:
    image = _flatten_rgba_to_rgb(image)
    if not do_resize or input_max_pixels is None:
        return image
    resized_height, resized_width = smart_resize(
        height=image.height,
        width=image.width,
        factor=SENSENOVA_U1_RESOLUTION_ALIGNMENT,
        min_pixels=input_max_pixels,
        max_pixels=input_max_pixels,
    )
    if (resized_width, resized_height) == image.size:
        return image
    return image.resize((resized_width, resized_height), Image.LANCZOS)


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def _image_input_to_list(image_input: Any) -> list[Image.Image]:
    if image_input is None:
        return []
    if isinstance(image_input, list):
        items = image_input
    else:
        items = [image_input]

    images = []
    for item in items:
        if isinstance(item, Image.Image):
            images.append(item)
        else:
            images.append(load_image(str(item)))
    return images


def _prepare_edit_images(
    batch: Req, options: SenseNovaU1GenerationOptions
) -> list[Image.Image]:
    images = _image_input_to_list(getattr(batch, "condition_image", None))
    if not images:
        images = _image_input_to_list(getattr(batch, "image_path", None))
    if not images:
        return []

    input_max_pixels = options.input_max_pixels
    if input_max_pixels is None:
        input_max_pixels = _auto_input_max_pixels(len(images))
    return [
        _resize_input_to_budget(
            image,
            do_resize=options.do_resize,
            input_max_pixels=input_max_pixels,
        )
        for image in images
    ]


def _resolve_edit_output_size(
    batch: Req, edit_images: list[Image.Image]
) -> tuple[int, int]:
    """Preserve the first input image's aspect ratio for SenseNova image edits."""
    if not edit_images:
        return int(batch.width), int(batch.height)

    target_pixels = int(batch.width) * int(batch.height)
    resized_height, resized_width = smart_resize(
        height=edit_images[0].height,
        width=edit_images[0].width,
        factor=SENSENOVA_U1_RESOLUTION_ALIGNMENT,
        min_pixels=target_pixels,
        max_pixels=target_pixels,
    )
    return resized_width, resized_height


@dataclass(frozen=True)
class SenseNovaU1GenerationOptions:
    cfg_norm: str = DEFAULT_CFG_NORM
    timestep_shift: float = DEFAULT_TIMESTEP_SHIFT
    enable_timestep_shift: bool = DEFAULT_ENABLE_TIMESTEP_SHIFT
    cfg_interval: tuple[float, float] = DEFAULT_CFG_INTERVAL
    t_eps: float = DEFAULT_T_EPS
    think_mode: bool = DEFAULT_THINK_MODE
    img_cfg_scale: float = 1.0
    input_max_pixels: int | None = None
    do_resize: bool = True

    @classmethod
    def from_batch(cls, batch: Req) -> SenseNovaU1GenerationOptions:
        extra = batch.extra.get(SENSENOVA_U1_REQUEST_EXTRA_KEY, {})
        return cls(
            cfg_norm=extra.get("cfg_norm", DEFAULT_CFG_NORM),
            timestep_shift=float(extra.get("timestep_shift", DEFAULT_TIMESTEP_SHIFT)),
            enable_timestep_shift=bool(
                extra.get("enable_timestep_shift", DEFAULT_ENABLE_TIMESTEP_SHIFT)
            ),
            cfg_interval=tuple(
                float(value)
                for value in extra.get("cfg_interval", DEFAULT_CFG_INTERVAL)
            ),
            t_eps=float(extra.get("t_eps", DEFAULT_T_EPS)),
            think_mode=bool(extra.get("think_mode", DEFAULT_THINK_MODE)),
            img_cfg_scale=float(extra.get("img_cfg_scale", 1.0)),
            input_max_pixels=(
                None
                if extra.get("input_max_pixels") is None
                else int(extra.get("input_max_pixels"))
            ),
            do_resize=_coerce_bool(extra.get("do_resize", True)),
        )


class SenseNovaU1GenerationStage(PipelineStage):
    def __init__(self, model: torch.nn.Module, tokenizer: Any):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    @property
    def role_affinity(self) -> RoleType:
        return RoleType.DENOISER

    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        del server_args
        options = SenseNovaU1GenerationOptions.from_batch(batch)
        if int(batch.num_outputs_per_prompt) != 1:
            raise ValueError(
                "SenseNova-U1 expects output expansion before generation; "
                f"got num_outputs_per_prompt={batch.num_outputs_per_prompt}."
            )
        seed = batch.seed[0] if isinstance(batch.seed, list) else int(batch.seed)

        edit_images = _prepare_edit_images(batch, options)
        image_size = (
            _resolve_edit_output_size(batch, edit_images)
            if edit_images
            else (int(batch.width), int(batch.height))
        )

        common_kwargs = dict(
            image_size=image_size,
            cfg_scale=float(batch.guidance_scale),
            cfg_norm=options.cfg_norm,
            timestep_shift=options.timestep_shift,
            enable_timestep_shift=options.enable_timestep_shift,
            cfg_interval=options.cfg_interval,
            num_steps=int(batch.num_inference_steps),
            batch_size=1,
            t_eps=options.t_eps,
            think_mode=options.think_mode,
            seed=seed,
        )
        if edit_images:
            if options.cfg_norm == "cfg_zero_star":
                raise ValueError(
                    "cfg_zero_star is only supported for SenseNova-U1 text-to-image, "
                    "not image editing."
                )
            out = self.model.it2i_generate(
                self.tokenizer,
                batch.prompt,
                edit_images,
                img_cfg_scale=options.img_cfg_scale,
                **common_kwargs,
            )
        else:
            out = self.model.t2i_generate(
                self.tokenizer,
                batch.prompt,
                **common_kwargs,
            )
        think_text = None
        if options.think_mode:
            images, think_text = out
        else:
            images = out

        images = _denorm_sensenova_output(images)
        samples = [sample.contiguous() for sample in images]
        usage = {"think_text": think_text} if think_text is not None else None
        return OutputBatch(
            output=samples,
            metrics=batch.metrics,
            usage=usage,
        )
