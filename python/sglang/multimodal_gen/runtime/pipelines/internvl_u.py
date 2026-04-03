# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
InternVL-U unified pipeline for SGLang multimodal_gen.

Loads the full InternVL-U pipeline (VLM + DiT + VAE) as a single unit,
handling text -> VLM hidden states -> DiT denoising -> VAE decode -> image.

Requires:
  - InternVL-U source: set INTERNVLU_SRC_PATH or pip install internvlu
  - Model weights: OpenGVLab/InternVL-U (HuggingFace, gated)

Optional:
  - INTERNVLU_TORCH_COMPILE=1: Enable torch.compile for DiT acceleration
  - FLASH_ATTN_PATCH_PATH: Path to flash_attn SDPA patch (if flash_attn unavailable)
"""

import os
import sys
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def _setup_internvlu_imports():
    """Ensure InternVL-U source code is importable."""
    src_path = envs.INTERNVLU_SRC_PATH
    if src_path and src_path not in sys.path:
        sys.path.insert(0, src_path)

    patch_path = envs.FLASH_ATTN_PATCH_PATH
    if patch_path and patch_path not in sys.path:
        sys.path.insert(0, patch_path)

    try:
        import flash_attn_patch  # noqa: F401
    except ImportError:
        logger.warning(
            "flash_attn_patch is not available; set FLASH_ATTN_PATCH_PATH if InternVL-U requires the SDPA patch"
        )

    try:
        from internvlu import InternVLUPipeline
        from internvlu.processing_internvlu import InternVLUProcessor
    except ImportError as exc:
        raise ImportError(
            "InternVL-U source not found. Either:\n"
            "  1. pip install internvlu, or\n"
            "  2. Set INTERNVLU_SRC_PATH to the InternVL-U repo directory\n"
            "  See: https://github.com/OpenGVLab/InternVL-U"
        ) from exc

    return InternVLUPipeline, InternVLUProcessor


class InternVLUExecutionStage(PipelineStage):
    """Execute full InternVL-U pipeline: prompt -> VLM -> DiT -> image."""

    def __init__(self, pipe: Any, processor: Any):
        super().__init__()
        self.pipe = pipe
        self.processor = processor

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        prompt = batch.prompt or ""

        height = batch.height if batch.height is not None else 512
        width = batch.width if batch.width is not None else 512
        num_steps = (
            batch.num_inference_steps if batch.num_inference_steps is not None else 20
        )
        size = batch.size
        if isinstance(size, str) and "x" in size:
            try:
                width_str, height_str = size.split("x")
                width, height = int(width_str), int(height_str)
            except ValueError:
                pass

        logger.info(
            "Generating %dx%d (%d steps): %.50s", width, height, num_steps, prompt
        )

        with torch.no_grad():
            result = self.pipe(
                prompt=prompt,
                generation_mode="image",
                height=height,
                width=width,
                num_inference_steps=num_steps,
                processor=self.processor,
            )

        if result.images:
            img = result.images[0]
            if isinstance(img, Image.Image):
                batch.output = (
                    torch.from_numpy(np.array(img).astype(np.float32) / 255.0)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                )
            elif isinstance(img, torch.Tensor):
                batch.output = img.unsqueeze(0) if img.dim() == 3 else img
            else:
                batch.output = img
        else:
            batch.output = None

        return batch


class InternVLUDiffusionPipeline(ComposedPipelineBase):
    """SGLang pipeline for InternVL-U: unified VLM + DiT image generation."""

    pipeline_name = "InternVLUDiffusionPipeline"
    _required_config_modules = []

    def load_modules(
        self, server_args: ServerArgs, loaded_modules: Optional[Dict] = None
    ) -> Dict[str, Any]:
        InternVLUPipeline, InternVLUProcessor = _setup_internvlu_imports()

        model_path = self.model_path
        dtype = torch.float16

        logger.info("Loading InternVL-U pipeline from %s", model_path)

        self._pipe = InternVLUPipeline.from_pretrained(
            model_path, torch_dtype=dtype
        ).to("cuda")

        processor_path = os.path.join(model_path, "processor")
        self._processor = InternVLUProcessor.from_pretrained(processor_path)

        # Optional torch.compile acceleration
        if envs.INTERNVLU_TORCH_COMPILE:
            logger.info("Applying torch.compile to generation decoder")
            self._pipe.generation_decoder.decoder = torch.compile(
                self._pipe.generation_decoder.decoder, mode="reduce-overhead"
            )

        mem_gb = torch.cuda.max_memory_allocated() / 1024**3
        logger.info("InternVL-U loaded, GPU: %.1f GB", mem_gb)

        return {"internvlu_pipeline": self._pipe, "processor": self._processor}

    def create_pipeline_stages(self, server_args: ServerArgs):
        self.add_stage(
            InternVLUExecutionStage(self._pipe, self._processor),
            "internvlu_execution",
        )

    def initialize_pipeline(self, server_args: ServerArgs):
        """Warmup with a small generation to trigger JIT compilation."""
        logger.info("Warming up InternVL-U pipeline...")
        try:
            self._pipe(
                prompt="warmup",
                generation_mode="image",
                height=256,
                width=256,
                num_inference_steps=2,
                processor=self._processor,
            )
            logger.info("Warmup complete")
        except Exception as e:
            logger.warning("Warmup failed (non-fatal): %s", e)


EntryClass = [InternVLUDiffusionPipeline]
