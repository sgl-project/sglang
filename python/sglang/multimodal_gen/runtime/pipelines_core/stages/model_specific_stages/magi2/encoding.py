# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import numpy as np
import torch
from PIL import Image

from sglang.multimodal_gen.runtime.distributed import (
    get_local_torch_device,
    get_sp_group,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.magi2.preparation import (
    prompt_is_non_empty,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs

_CANVAS_COLOR = (255, 255, 255)

FIGURE_ONE = "<Figure 1>"
_REFERENCE_HINT = f"The first frame refers to {FIGURE_ONE}"


def resize_pad(image: Image.Image, *, height: int, width: int) -> Image.Image:
    """Letterbox, never crop: the image conditions the whole clip rather than becoming a frame of it."""
    source_w, source_h = image.size
    if source_w <= 0 or source_h <= 0:
        raise ValueError(f"invalid image size {image.size}")

    scale = min(width / source_w, height / source_h)
    target_w = max(1, round(source_w * scale))
    target_h = max(1, round(source_h * scale))
    resized = image.convert("RGB").resize(
        (target_w, target_h), resample=Image.Resampling.LANCZOS
    )
    canvas = Image.new("RGB", (width, height), _CANVAS_COLOR)
    canvas.paste(resized, ((width - target_w) // 2, (height - target_h) // 2))
    return canvas


def target_size(
    image: Image.Image, *, generation_height: int, generation_width: int
) -> tuple[int, int]:
    long_edge = max(generation_height, generation_width)
    if image.width > image.height:
        return round(image.height * long_edge / image.width), long_edge
    return long_edge, round(image.width * long_edge / image.height)


def resample_to(image: Image.Image, height: int, width: int) -> Image.Image:
    if image.size == (width, height):
        return image
    return image.resize((width, height), resample=Image.Resampling.LANCZOS)


def aligned_size(height: int, width: int, *, align: int) -> tuple[int, int]:
    """``align`` is TWICE the VAE spatial compression ratio, not the ratio itself."""
    return max(align, height - height % align), max(align, width - width % align)


def ensure_figure_phrase(prompt: str) -> str:
    """Without the phrase the pooled embedding is zero and the image conditions nothing, with no error."""
    if FIGURE_ONE in prompt:
        return prompt
    try:
        parsed = json.loads(prompt)
        if not isinstance(parsed, dict):
            raise ValueError("prompt JSON is not an object")
    except (json.JSONDecodeError, TypeError, ValueError):
        return prompt + f"reference_layer:{_REFERENCE_HINT}"
    parsed["reference_layer"] = [_REFERENCE_HINT]
    return json.dumps(parsed, ensure_ascii=False)


def pool_figure_embedding(
    *,
    prompt: str,
    tokenizer,
    prompt_embeds: torch.Tensor,
    phrase: str = FIGURE_ONE,
) -> torch.Tensor:
    """Uses character offsets, not a token-id search, because the phrase spans several subwords."""
    width = prompt_embeds.shape[-1]
    zero = prompt_embeds.new_zeros(width)

    start_char = prompt.find(phrase)
    if start_char < 0:
        return zero
    end_char = start_char + len(phrase)

    encoded = tokenizer(
        [prompt], return_tensors="pt", return_offsets_mapping=True, truncation=False
    )
    offsets = encoded["offset_mapping"][0]

    rows: list[int] = []
    for index, (begin, end) in enumerate(offsets.tolist()):
        # Special tokens report a (0, 0) span; keeping them would pool padding.
        if begin == 0 and end == 0 and index != 0:
            continue
        if max(begin, start_char) < min(end, end_char):
            rows.append(index)
    if not rows:
        return zero

    selection = torch.tensor(rows, device=prompt_embeds.device)
    return prompt_embeds.index_select(0, selection).mean(dim=0)


class Magi2TextEncodingStage(PipelineStage):
    def __init__(
        self,
        *,
        text_encoder,
        tokenizer,
        tokenizer_kwargs: dict,
        skip_layer: int,
    ) -> None:
        super().__init__()
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.tokenizer_kwargs = tokenizer_kwargs
        self.skip_layer = skip_layer
        # Cached on the prompt text, which every rank has: if one rank hit the cache
        # and another missed, _encode_and_share's broadcast would hang.
        self._negative_prompt: str | None = None
        self._negative_embeds: list[torch.Tensor] | None = None

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("prompt", batch.prompt, prompt_is_non_empty)
        return result

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        # Non-source SP ranks never run the encoder, so the use is not opened at entry.
        return [
            ComponentUse(
                stage_name=self._component_stage_name(stage_name),
                component_name="text_encoder",
                start_at_stage_entry=False,
            )
        ]

    def _encode(
        self, prompts: list[str], *, device: torch.device
    ) -> list[torch.Tensor]:
        """Batching pads to the longest prompt; only the real tokens may be packed."""
        tokens = self.tokenizer(prompts, **self.tokenizer_kwargs).to(device)
        with (
            self.use_declared_component(
                component_name="text_encoder", module=self.text_encoder
            ),
            torch.no_grad(),
            set_forward_context(current_timestep=0, attn_metadata=None),
        ):
            output = self.text_encoder(**tokens, output_hidden_states=True)

        # Not the final state: the DiT was trained on features from that depth.
        hidden = output.hidden_states[-(self.skip_layer + 1)]
        lengths = tokens["attention_mask"].sum(dim=1).tolist()
        return [hidden[row, : int(length)] for row, length in enumerate(lengths)]

    def _encode_and_share(self, prompts: list[str]) -> list[torch.Tensor]:
        """Only the source rank runs the encoder; residency streams its layers onto that rank's GPU."""
        sp_group = get_sp_group()
        is_src = sp_group.rank_in_group == 0
        device = get_local_torch_device()

        embeds: list[torch.Tensor] | None = None
        if is_src:
            embeds = self._encode(prompts, device=device)

        if sp_group.world_size == 1:
            return embeds

        # Shapes travel first: branches are trimmed, so ranks cannot infer them.
        meta = sp_group.broadcast_object(
            [(tuple(e.shape), e.dtype) for e in embeds] if is_src else None
        )
        if not is_src:
            embeds = [
                torch.empty(shape, dtype=dtype, device=device) for shape, dtype in meta
            ]
        for embed in embeds:
            sp_group.broadcast(embed, src=0)
        return embeds

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        use_guidance = server_args.pipeline_config.should_use_guidance

        # One call per prompt: batching pads the caption to the negative prompt's length.
        embeds = self._encode_and_share([batch.prompt])

        batch.prompt_embeds = [embeds[0]]
        if use_guidance:
            negative = batch.negative_prompt or ""
            if negative != self._negative_prompt:
                self._negative_embeds = self._encode_and_share([negative])
                self._negative_prompt = negative
            batch.negative_prompt_embeds = list(self._negative_embeds)

        counts = batch.extra["magi2_ref_patch_counts"]
        if counts:
            batch.extra["magi2_ref_special"] = torch.stack(
                [
                    pool_figure_embedding(
                        prompt=batch.prompt,
                        tokenizer=self.tokenizer,
                        prompt_embeds=embeds[0],
                        phrase=f"<Figure {index + 1}>",
                    )
                    for index in range(len(counts))
                ],
                dim=0,
            )
        else:
            batch.extra["magi2_ref_special"] = None
        return batch


class Magi2ImageEncodingStage(PipelineStage):
    """Runs before text encoding: the text stage pools each image's ``<Figure N>`` phrase to prefix its patches."""

    def __init__(
        self,
        *,
        vae,
        spatial_compression_ratio: int,
        latents_mean: tuple[float, ...],
        latents_std: tuple[float, ...],
    ) -> None:
        super().__init__()
        self.vae = vae
        # 2x the compression ratio: the ratio alone gives a taller latent grid than
        # was trained on.
        self.ref_align = 2 * spatial_compression_ratio
        self.latents_mean = torch.tensor(latents_mean).view(1, -1, 1, 1, 1)
        self.latents_std = torch.tensor(latents_std).view(1, -1, 1, 1, 1)

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        # Opened only when the request carries images.
        return [
            ComponentUse(
                stage_name=self._component_stage_name(stage_name),
                component_name="vae",
                start_at_stage_entry=False,
            )
        ]

    def _encode_images(
        self, images: list[Image.Image], *, params
    ) -> tuple[list[torch.Tensor], list[tuple[int, int]]]:
        device = get_local_torch_device()
        weight = next(self.vae.parameters())

        patches: list[torch.Tensor] = []
        latent_hw: list[tuple[int, int]] = []
        for image in images:
            height, width = target_size(
                image,
                generation_height=params.preview_height,
                generation_width=params.preview_width,
            )
            # Letterbox onto the unrounded canvas, then resample to the aligned one;
            # the direct route pads where the reference squashes.
            fitted = resize_pad(image, height=height, width=width)
            fitted = resample_to(
                fitted,
                *aligned_size(height, width, align=self.ref_align),
            )
            pixels = _to_pixel_tensor(fitted, device=device, dtype=weight.dtype)
            with torch.no_grad():
                encoded = self.vae.encode(pixels)
            latent = encoded.mean if hasattr(encoded, "mean") else encoded
            # sglang's Wan encode returns the raw distribution; without this the
            # patches sit at ~std scale.
            latent = (latent - self.latents_mean.to(latent)) / self.latents_std.to(
                latent
            )
            latent = latent[:, :, 0]
            channels, latent_h, latent_w = latent.shape[1:]
            patches.append(latent.reshape(channels, latent_h * latent_w).t().float())
            latent_hw.append((latent_h, latent_w))
        return patches, latent_hw

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        images = _condition_images(batch)
        batch.extra["magi2_ref_patches"] = None
        batch.extra["magi2_ref_latent_hw"] = []
        batch.extra["magi2_ref_patch_counts"] = []
        if not images:
            return batch

        with self.use_declared_component(component_name="vae", module=self.vae):
            patches, latent_hw = self._encode_images(
                images, params=batch.sampling_params
            )

        batch.extra["magi2_ref_patches"] = torch.cat(patches, dim=0)
        batch.extra["magi2_ref_latent_hw"] = latent_hw
        batch.extra["magi2_ref_patch_counts"] = [h * w for h, w in latent_hw]

        # Must happen before text encoding: a prompt that never names the image
        # pools a zero vector and stops conditioning on it, with no error.
        batch.prompt = ensure_figure_phrase(batch.prompt)
        self.log_info(
            "[magi2] conditioning on %d image(s), latent grids %s",
            len(images),
            latent_hw,
        )
        return batch


def _condition_images(batch: Req) -> list[Image.Image]:
    candidates = batch.condition_image or batch.image_path
    if candidates is None:
        return []
    if not isinstance(candidates, (list, tuple)):
        candidates = [candidates]

    images: list[Image.Image] = []
    for candidate in candidates:
        if candidate is None:
            continue
        if isinstance(candidate, Image.Image):
            images.append(candidate)
        elif isinstance(candidate, str):
            images.append(Image.open(candidate))
        else:
            raise TypeError(
                "MAGI-2 needs conditioning images as paths or PIL images, not "
                f"{type(candidate)}: it VAE-encodes the original rather than a "
                "preprocessed tensor"
            )
    return images


def _to_pixel_tensor(
    image: Image.Image, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    pixels = torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.float32))
    pixels = pixels.permute(2, 0, 1) / 127.5 - 1.0
    return pixels.to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(2)
