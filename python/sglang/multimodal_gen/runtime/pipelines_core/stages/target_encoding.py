# SPDX-License-Identifier: Apache-2.0
"""Encode-only training targets: what the denoiser would generate, as its latents."""

from io import BytesIO

import torch
from PIL import Image, ImageOps

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs


def open_target_image(path: str) -> Image.Image:
    from sglang.srt.utils.common import get_image_bytes

    image = Image.open(BytesIO(get_image_bytes(path)))
    return ImageOps.exif_transpose(image).convert("RGB")


class TargetEncodingStage(PipelineStage):
    """Encode ``encode_target`` into ``extra["encode_target"][stream]``.

    A pipeline that supports encode-only training adds a subclass to its image
    encoder role; the stage is a no-op for requests without a target, so the
    same pipeline still serves generation. Each stream maps to a dict of
    tensors in the denoiser's layout, at least ``latents``.
    """

    streams: tuple[str, ...] = ("visual",)

    @property
    def role_affinity(self) -> RoleType:
        return RoleType.IMAGE_ENCODER

    def prepare_visual(self, path: str, batch: Req) -> Image.Image:
        """The target as the denoiser would generate it at the request size:
        resized to cover ``width`` x ``height`` and center cropped."""
        return ImageOps.fit(
            open_target_image(path),
            (batch.width, batch.height),
            Image.Resampling.LANCZOS,
        )

    def encode_visual(
        self, image: Image.Image, batch: Req, server_args: ServerArgs
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        targets = batch.encode_target
        if not targets:
            return batch
        unsupported = set(targets) - set(self.streams)
        if unsupported:
            raise ValueError(
                f"{type(self).__name__} encodes targets {list(self.streams)}, "
                f"got {sorted(unsupported)}"
            )
        batch.extra["encode_target"] = {
            "visual": self.encode_visual(
                self.prepare_visual(targets["visual"], batch), batch, server_args
            )
        }
        return batch
