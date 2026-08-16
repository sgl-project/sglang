# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from sglang.multimodal_gen.runtime.distributed import (
    get_local_torch_device,
    get_sp_group,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.magi2 import (
    ref_images as magi2_ref_images,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


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

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("prompt", batch.prompt, V.string_not_empty)
        return result

    def _encode(
        self, prompts: list[str], *, device: torch.device
    ) -> list[torch.Tensor]:
        """Batching pads to the longest prompt; only the real tokens may be packed."""
        tokens = self.tokenizer(prompts, **self.tokenizer_kwargs).to(device)
        with torch.no_grad():
            output = self.text_encoder(**tokens, output_hidden_states=True)

        # Not the final state: the DiT was trained on features from that depth.
        hidden = output.hidden_states[-(self.skip_layer + 1)]
        lengths = tokens["attention_mask"].sum(dim=1).tolist()
        return [hidden[row, : int(length)] for row, length in enumerate(lengths)]

    def _encode_and_share(self, prompts: list[str]) -> list[torch.Tensor]:
        """The forward stays on the CPU: staging the encoder to a card costs more than it saves, and spikes on the resident DiTs."""
        sp_group = get_sp_group()
        is_src = sp_group.rank_in_group == 0
        device = get_local_torch_device()

        embeds: list[torch.Tensor] | None = None
        if is_src:
            embeds = [
                embed.to(device)
                for embed in self._encode(prompts, device=torch.device("cpu"))
            ]

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
        prompts = [batch.prompt]
        if use_guidance:
            prompts.append(batch.negative_prompt or "")

        embeds = self._encode_and_share(prompts)

        batch.prompt_embeds = [embeds[0]]
        if use_guidance:
            batch.negative_prompt_embeds = [embeds[1]]

        counts = batch.extra["magi2_ref_patch_counts"]
        if counts:
            batch.extra["magi2_ref_special"] = torch.stack(
                [
                    magi2_ref_images.pool_figure_embedding(
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

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        images = _condition_images(batch)
        batch.extra["magi2_ref_patches"] = None
        batch.extra["magi2_ref_latent_hw"] = []
        batch.extra["magi2_ref_patch_counts"] = []
        if not images:
            return batch

        params = batch.sampling_params
        device = get_local_torch_device()
        weight = next(self.vae.parameters())

        patches: list[torch.Tensor] = []
        latent_hw: list[tuple[int, int]] = []
        for image in images:
            height, width = magi2_ref_images.target_size(
                image,
                generation_height=params.preview_height,
                generation_width=params.preview_width,
            )
            # Letterbox onto the unrounded canvas, then resample to the aligned one;
            # the direct route pads where the reference squashes.
            fitted = magi2_ref_images.resize_pad(image, height=height, width=width)
            fitted = magi2_ref_images.resample_to(
                fitted,
                *magi2_ref_images.aligned_size(height, width, align=self.ref_align),
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

        batch.extra["magi2_ref_patches"] = torch.cat(patches, dim=0)
        batch.extra["magi2_ref_latent_hw"] = latent_hw
        batch.extra["magi2_ref_patch_counts"] = [h * w for h, w in latent_hw]

        # Must happen before text encoding: a prompt that never names the image
        # pools a zero vector and stops conditioning on it, with no error.
        batch.prompt = magi2_ref_images.ensure_figure_phrase(batch.prompt)
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
