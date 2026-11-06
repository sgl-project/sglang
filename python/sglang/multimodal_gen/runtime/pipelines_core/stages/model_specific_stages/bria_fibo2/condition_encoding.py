# SPDX-License-Identifier: Apache-2.0
"""Text conditioning for Fibo 2.0, whose one checkpoint serves both paths.

`ImageEncodingStage` encodes a prompt the way an editing model needs: Qwen3-VL
sees the images to edit next to the prompt and returns the bundles the
transformer consumes. It returns early when a request carries no image, which is
right for checkpoints that ship separate generation and editing pipelines, but
sglang resolves the pipeline class from `model_index.json` and Fibo 2.0 has a
single checkpoint, so one pipeline class has to drive both paths.

The two stages below are the two halves of that: a standard text-encoding stage
that steps aside when the request carries an image, and the image-aware stage
that steps aside when it does not. Both are registered as ordinary stages rather
than one wrapping the other, so each keeps its own stage name and therefore its
own component-residency bookkeeping; a wrapping stage would leave the inner one
unregistered and its encoder would never be moved onto the device.

Both halves hand the same six bundles to the transformer, because the assembly
lives in `BriaFibo2PipelineConfig.postprocess_text_funcs`.
"""

from __future__ import annotations

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.image_encoding import (
    ImageEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding import (
    TextEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class BriaFibo2TextEncodingStage(TextEncodingStage):
    """Text-only encoding, for the requests that carry no image to edit."""

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if batch.condition_image is not None:
            return batch
        return super().forward(batch, server_args)

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        if batch.condition_image is not None:
            return VerificationResult()
        return super().verify_input(batch, server_args)

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        if batch.condition_image is not None:
            return VerificationResult()
        return super().verify_output(batch, server_args)


class BriaFibo2ImageEncodingStage(ImageEncodingStage):
    """Image-aware encoding, for edit requests; the generation path skips it."""

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if batch.condition_image is None:
            return batch
        return super().forward(batch, server_args)

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        if batch.condition_image is None:
            return VerificationResult()
        return super().verify_input(batch, server_args)

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        if batch.condition_image is None:
            return VerificationResult()
        return super().verify_output(batch, server_args)


__all__ = ["BriaFibo2ImageEncodingStage", "BriaFibo2TextEncodingStage"]
