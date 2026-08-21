# Adapted from LingBot-Video (https://github.com/Robbyant/lingbot-video).
#
# SPDX-License-Identifier: Apache-2.0

import json
import re

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    PipelineStage,
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_video_moe.rewriter import (
    condition_image,
    parse_caption,
    resolve_mode,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_video_moe.rewriter_prompts import (
    NEGATIVE_PRUNE,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs

_ROOT = "universal_negative"


def _hint_pattern(*hints: str) -> re.Pattern:
    """Match any hint as a whole word, so "knight" does not read as "night"."""

    alternatives = "|".join(re.escape(hint) for hint in hints)
    return re.compile(rf"(?<![a-z0-9])(?:{alternatives})(?![a-z0-9])")


# Whole-category removal needs the caption to ask for it outright.
_BLOCK_HINTS = {
    "physical_plausibility": _hint_pattern(
        "fantasy",
        "surreal",
        "dreamlike",
        "dream-like",
        "magic",
        "magical",
        "supernatural",
        "physics-bending",
        "physics bending",
        "impossible physics",
        "anti-gravity",
        "antigravity",
        "zero gravity",
        "zero-gravity",
        "weightless",
        "weightlessness",
        "floating in space",
        "outer space",
        "astronaut",
    ),
    "artistic_style": _hint_pattern(
        "painting",
        "illustration",
        "cartoon",
        "drawing",
        "sketch",
        "cgi",
        "3d render",
        "3d-render",
        "digital art",
        "anime",
        "stylized animation",
        "claymation",
        "stop motion",
        "stop-motion",
    ),
}

# Terms the model tends to leave in even when the caption clearly asks for them.
_FORCED_DELETIONS = (
    (
        _hint_pattern(
            "dark",
            "dim",
            "dimly",
            "low light",
            "low-light",
            "night",
            "nighttime",
            "moody",
            "gloomy",
            "ominous",
            "deep shadow",
            "deep shadows",
            "dark shadows",
        ),
        ("underexposed", "subject hidden in darkness", "crushed blacks"),
    ),
    (
        _hint_pattern(
            "motion blur",
            "motion-blur",
            "blurred background",
            "blurred landscape",
            "blurred scenery",
            "blurred surroundings",
            "speed blur",
            "long exposure",
        ),
        ("motion blur",),
    ),
)


def categorized_negative(negative_prompt: str) -> dict | None:
    """The negative prompt as categories of terms, or None if it is free text.

    Only the shipped shape can be pruned: a term kept by category, in order. A
    request may send anything, so the schema is checked before it is indexed.
    """

    parsed = parse_caption(negative_prompt)
    if not isinstance(parsed, dict):
        return None
    categories = parsed.get(_ROOT)
    if not isinstance(categories, dict):
        return None
    for terms in categories.values():
        if not isinstance(terms, list):
            return None
        if not all(isinstance(term, str) for term in terms):
            return None
    return {_ROOT: categories}


def build_prune_prompt(caption: str, mode: str, default: dict) -> str:
    return (
        f"{NEGATIVE_PRUNE}\n\n## MODE: {mode}\n\n"
        f"## INTENDED CONTENT (structured caption):\n```json\n{caption}\n```\n\n"
        "## DEFAULT NEGATIVE (delete the contradicting terms, keep the rest):\n"
        f"```json\n{json.dumps(default, ensure_ascii=False)}\n```\n\n"
        "Output ONLY the edited negative JSON now."
    )


def prune_negative(default: dict, pruned: dict | None, caption: str) -> dict:
    """Keep the default's terms and order, dropping only what the model deleted."""

    kept = pruned.get(_ROOT) if isinstance(pruned, dict) else None
    lowered = caption.lower()
    out = {}
    for category, terms in default[_ROOT].items():
        survivors = kept.get(category) if isinstance(kept, dict) else None
        if not isinstance(survivors, list):
            out[category] = list(terms)
            continue
        survivor_set = set(survivors)
        out[category] = [term for term in terms if term in survivor_set]
        hints = _BLOCK_HINTS.get(category)
        if not out[category] and hints is not None and not hints.search(lowered):
            out[category] = list(terms)
    for hints, deletions in _FORCED_DELETIONS:
        if not hints.search(lowered):
            continue
        for category, terms in out.items():
            out[category] = [term for term in terms if term not in deletions]
    return {_ROOT: out}


class LingBotVideoAutoNegativeStage(PipelineStage):
    """Drop the negative-prompt terms that fight this request's own caption.

    The shipped negative is deliberately over-complete and self-contradictory, so
    terms the caption legitimately wants push the sample away from the request.
    """

    def __init__(self, backend):
        super().__init__()
        self.backend = backend

    @property
    def parallelism_type(self) -> StageParallelismType:
        return StageParallelismType.MAIN_RANK_ONLY_AND_SEND_TO_OTHERS

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        # supports_dynamic_batching() turns merging off for this config, so the
        # prompt is one caption and the negative it prunes belongs to it.
        default = categorized_negative(batch.negative_prompt)
        if default is None:
            # A free-text negative has no categories to prune.
            return batch
        caption = batch.prompt
        mode = resolve_mode(batch)
        image = condition_image(batch) if mode == "ti2v" else None
        raw = self.backend.generate(
            build_prune_prompt(caption, mode, default), image, use_lora=False
        )
        negative = prune_negative(default, parse_caption(raw), caption)
        batch.negative_prompt = json.dumps(
            negative, ensure_ascii=False, separators=(",", ":")
        )
        return batch
