# SPDX-License-Identifier: Apache-2.0

import json
import re

import PIL.Image

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    PipelineStage,
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_video_moe.rewriter_prompts import (
    IMAGE_STEP1_EXPAND,
    IMAGE_STEP2_MAP,
    VIDEO_DURATION_EN,
    VIDEO_DURATION_ZH,
    VIDEO_STEP1_EXPAND,
    VIDEO_STEP2_MAP,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.vision import load_image

_CJK = re.compile(r"[　-〿㐀-䶿一-鿿＀-￯]")
_FENCED_JSON = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)


def build_expand_prompt(mode: str, prompt: str, duration: int) -> str:
    if mode == "t2i":
        return f"{IMAGE_STEP1_EXPAND}\n\nUser image prompt:\n{prompt}"
    template = VIDEO_DURATION_ZH if _CJK.search(prompt) else VIDEO_DURATION_EN
    duration_line = template.format(duration=duration)
    return f"{VIDEO_STEP1_EXPAND}\n\n{prompt}\n\n{duration_line}"


def build_map_prompt(mode: str, detailed: str, duration: int) -> str:
    if mode == "t2i":
        return f"{IMAGE_STEP2_MAP}\n\nDETAILED CAPTION:\n{detailed}"
    return (
        f"{VIDEO_STEP2_MAP}\n\nVideo Duration: {duration} seconds\n\n"
        f"DETAILED CAPTION:\n{detailed}\n\nOutput the JSON now."
    )


def parse_caption(raw: str) -> dict | None:
    """Read the mapping stage output, tolerating code fences and stray text."""

    text = (raw or "").strip()
    fenced = _FENCED_JSON.search(text)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        caption = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return caption if isinstance(caption, dict) else None


def resolve_mode(batch: Req) -> str:
    if int(batch.num_frames) == 1:
        return "t2i"
    return "ti2v" if batch.image_path else "t2v"


def needs_rewrite(prompt: str) -> bool:
    """A structured caption is already in distribution, so leave it alone."""

    return not prompt.lstrip().startswith("{")


def condition_image(batch: Req) -> PIL.Image.Image:
    """The condition frame as a PIL image, loading it if the stage runs first."""

    image = batch.condition_image
    if isinstance(image, list):
        image = image[0]
    if not isinstance(image, PIL.Image.Image):
        image = load_image(batch.image_path)
    return image.convert("RGB")


class LingBotVideoPromptRewriteStage(PipelineStage):
    """Expand a plain prompt into the structured caption the DiT was trained on.

    Two turns against the rewriter backend: expand without the rewriter adapter,
    then map that expansion to JSON with the adapter enabled.
    """

    def __init__(self, backend):
        super().__init__()
        self.backend = backend

    @property
    def parallelism_type(self) -> StageParallelismType:
        # Rank 0 rewrites and broadcasts, so every rank denoises the same caption.
        return StageParallelismType.MAIN_RANK_ONLY_AND_SEND_TO_OTHERS

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        # Dynamic batching merges compatible text requests into a prompt list.
        batched = isinstance(batch.prompt, list)
        prompts = batch.prompt if batched else [batch.prompt]

        mode = resolve_mode(batch)
        duration = max(1, round(int(batch.num_frames) / max(int(batch.fps), 1)))
        image = condition_image(batch) if mode == "ti2v" else None
        rewritten = [self._rewrite(p, mode, duration, image) for p in prompts]

        batch.prompt = rewritten if batched else rewritten[0]
        return batch

    def _rewrite(
        self, prompt: str, mode: str, duration: int, image: PIL.Image.Image | None
    ) -> str:
        if not needs_rewrite(prompt):
            return prompt
        detailed = self.backend.generate(
            build_expand_prompt(mode, prompt, duration), image, use_lora=False
        )
        raw = self.backend.generate(
            build_map_prompt(mode, detailed, duration), image, use_lora=True
        )
        caption = parse_caption(raw)
        if caption is None:
            # Falling back to the plain prompt would denoise out of distribution
            # and emit a broken video rather than fail.
            raise ValueError(
                "Prompt rewriting produced no parseable structured caption. "
                "Pass a structured JSON caption directly, or check the rewriter backend."
            )
        return json.dumps(caption, ensure_ascii=False, separators=(",", ":"))
