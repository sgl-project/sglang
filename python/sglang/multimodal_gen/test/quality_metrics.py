"""CLIP / CLIP-IQA / ImageReward scoring for generated images.

The three metrics see different inputs on purpose: CLIP scores an image against
its own prompt (adherence), CLIP-IQA scores the image alone (no-reference
aesthetics), ImageReward scores the pair with a human-preference model. None of
them measures realism -- all three rate generated images above real photographs
-- so a value is low only relative to a control, never absolutely.

Distinct from the CLIP *similarity* in ``test_utils.py``, which compares a
generated image against a stored reference and answers "did the pipeline
change?". This module answers "is the output any good?" and needs no reference.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import msgspec
import numpy as np
from PIL import Image

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.srt.environ import envs

logger = init_logger(__name__)

# torchmetrics' CLIPScore default, and the backbone published FLUX/SD numbers use.
# Comparable only within one backbone: ViT-L/14 reads several points away.
CLIP_PROMPT_MODEL_NAME = "openai/clip-vit-base-patch32"

IMAGE_REWARD_MODEL_NAME = "ImageReward-v1.0"

QUALITY_THRESHOLD_PATH = (
    Path(__file__).resolve().parent / "server" / "quality_thresholds.json"
)

_EVAL_EXTRA_HINT = (
    'CLIP-IQA needs the eval extra: pip install "sglang[diffusion-eval]" '
    "(torchmetrics + piq). CLIP prompt adherence needs no extra -- it runs on the "
    "transformers already pinned by sglang."
)

_IMAGE_REWARD_HINT = (
    "ImageReward cannot share this interpreter: it imports ImageReward/ReFL.py, "
    "which needs transformers<5 while sglang pins transformers==5.12.1. Install it "
    "into a separate venv and point SGLANG_TEST_IMAGE_REWARD_PYTHON at that "
    "python: pip install image-reward 'transformers==4.46.3' 'diffusers==0.31.0'."
)

# Reads one {"model_name", "pairs": [[prompt, image_path], ...]} object on stdin,
# prints {"scores": [...]} in the same order. Runs outside sglang's interpreter.
_IMAGE_REWARD_SCRIPT = """
import json, sys
import ImageReward
payload = json.load(sys.stdin)
# CPU on purpose, as for CLIP above: ImageReward.load() takes a visible GPU by
# default, and the generation server still holds it when a test scores output.
model = ImageReward.load(payload["model_name"], device="cpu")
scores = [model.score(prompt, path) for prompt, path in payload["pairs"]]
print(json.dumps({"scores": scores}))
"""

_clip_prompt_cache: dict[str, Any] = {}
_clip_iqa_cache: dict[str, Any] = {}


class QualityScores(msgspec.Struct, frozen=True, omit_defaults=True):
    # None means "not computed", never "computed as zero".
    clip_score: float | None = None
    clip_iqa: float | None = None
    image_reward: float | None = None


class QualityThresholds(msgspec.Struct, frozen=True, omit_defaults=True):
    # None means the metric is neither enforced nor computed for this case.
    min_clip_score: float | None = None
    min_clip_iqa: float | None = None
    min_image_reward: float | None = None

    def enforces_anything(self) -> bool:
        return any(
            value is not None
            for value in (self.min_clip_score, self.min_clip_iqa, self.min_image_reward)
        )


def load_quality_thresholds(
    case_id: str,
    metadata: dict[str, Any] | None = None,
) -> QualityThresholds:
    if metadata is None:
        metadata = _load_quality_threshold_json()
    resolved = {
        **metadata.get("defaults", {}),
        **metadata.get("cases", {}).get(case_id, {}),
    }
    return QualityThresholds(
        min_clip_score=_optional_float(resolved.get("min_clip_score")),
        min_clip_iqa=_optional_float(resolved.get("min_clip_iqa")),
        min_image_reward=_optional_float(resolved.get("min_image_reward")),
    )


def _load_quality_threshold_json() -> dict[str, Any]:
    # One table for every backend: the floors are loose relative to what a GPU
    # family changes, unlike latency, which is why perf_baselines/ is per-device.
    with QUALITY_THRESHOLD_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def compute_quality_scores(
    *,
    image: np.ndarray,
    prompt: str,
    thresholds: QualityThresholds,
) -> QualityScores:
    """Score only the metrics the case enforces; each raises on failure."""
    return QualityScores(
        clip_score=(
            None
            if thresholds.min_clip_score is None
            else compute_clip_score(image=image, prompt=prompt)
        ),
        clip_iqa=(
            None if thresholds.min_clip_iqa is None else compute_clip_iqa(image=image)
        ),
        image_reward=(
            None
            if thresholds.min_image_reward is None
            else compute_image_reward(image=image, prompt=prompt)
        ),
    )


def check_quality(
    *,
    scores: QualityScores,
    thresholds: QualityThresholds,
) -> list[str]:
    """Return one message per violated threshold; empty means passed."""
    failures = []
    for name, score, minimum in (
        ("clip_score", scores.clip_score, thresholds.min_clip_score),
        ("clip_iqa", scores.clip_iqa, thresholds.min_clip_iqa),
        ("image_reward", scores.image_reward, thresholds.min_image_reward),
    ):
        if minimum is None:
            continue
        # An enforced metric that produced no value is a failure, not a pass:
        # a missing dependency or a broken scorer must not read as green.
        if score is None:
            failures.append(
                f"{name}: not computed, but threshold {minimum} is enforced"
            )
        elif score < minimum:
            failures.append(f"{name}: {score:.4f} < {minimum}")
    return failures


def format_quality_scores(scores: QualityScores) -> str:
    parts = [
        f"{name}={value:.4f}"
        for name, value in (
            ("clip", scores.clip_score),
            ("clip_iqa", scores.clip_iqa),
            ("image_reward", scores.image_reward),
        )
        if value is not None
    ]
    return ", ".join(parts) if parts else "no metrics computed"


def compute_clip_score(*, image: np.ndarray, prompt: str) -> float:
    """CLIP prompt adherence, 100 * cosine, matching torchmetrics' CLIPScore.

    Hand-rolled: torchmetrics 1.9's CLIPScore calls ``.norm()`` on what the pinned
    transformers 5.12.1 returns as an object (Lightning-AI/torchmetrics#3244).
    """
    import torch

    model, processor = _get_clip_prompt_model()
    inputs = processor(
        text=[prompt],
        images=[_as_rgb_uint8(image)],
        return_tensors="pt",
        padding=True,
        # Truncates to CLIP's 77-token context; torchmetrics instead slices
        # input_ids, dropping the EOS it pools at, so the two disagree past 77.
        truncation=True,
    )

    with torch.no_grad():
        image_features = _projected_features(
            model.get_image_features(inputs["pixel_values"])
        )
        text_features = _projected_features(
            model.get_text_features(inputs["input_ids"], inputs["attention_mask"])
        )
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        cosine = (image_features * text_features).sum(dim=-1)

    # No clamp at zero: torchmetrics 1.9 does not clamp either, and a negative
    # cosine is a signal worth seeing rather than flooring to 0.
    return float(100.0 * cosine.item())


def _projected_features(output: Any) -> Any:
    # transformers 5.12.1 returns BaseModelOutputWithPooling with the projected
    # embedding on pooler_output; 4.x returned that tensor directly.
    return output.pooler_output


def _get_clip_prompt_model() -> tuple[Any, Any]:
    if "model" not in _clip_prompt_cache:
        from transformers import CLIPModel, CLIPProcessor

        logger.info(
            "Loading CLIP model for prompt adherence: %s", CLIP_PROMPT_MODEL_NAME
        )
        # CPU on purpose: the generation server still holds the GPU when a test
        # scores its output, and one image per case is not worth a transfer.
        _clip_prompt_cache["model"] = (
            CLIPModel.from_pretrained(CLIP_PROMPT_MODEL_NAME).eval().to("cpu")
        )
        _clip_prompt_cache["processor"] = CLIPProcessor.from_pretrained(
            CLIP_PROMPT_MODEL_NAME
        )
    return _clip_prompt_cache["model"], _clip_prompt_cache["processor"]


def compute_clip_iqa(*, image: np.ndarray) -> float:
    """No-reference CLIP-IQA on the canonical `clip_iqa` weights, in [0, 1]."""
    import torch

    metric = _get_clip_iqa_metric()
    tensor = (
        torch.from_numpy(_as_rgb_uint8(image)).permute(2, 0, 1).unsqueeze(0).float()
    )
    score = float(metric(tensor).item())
    # The metric is cached across images and __call__ accumulates state nothing
    # here consumes; drop it per image rather than growing it over a sweep.
    metric.reset()
    return score


def _get_clip_iqa_metric() -> Any:
    if "metric" not in _clip_iqa_cache:
        try:
            from torchmetrics.multimodal import CLIPImageQualityAssessment
        except ImportError as exc:
            raise ImportError(_EVAL_EXTRA_HINT) from exc

        # data_range must match the scale of the tensors handed in below;
        # torchmetrics defaults to 1.0 unvalidated, so [0, 255] returns noise.
        _clip_iqa_cache["metric"] = CLIPImageQualityAssessment(
            model_name_or_path="clip_iqa",
            data_range=255.0,
        )
    return _clip_iqa_cache["metric"]


def image_reward_python() -> str | None:
    return envs.SGLANG_TEST_IMAGE_REWARD_PYTHON.get()


def compute_image_reward(*, image: np.ndarray, prompt: str) -> float:
    """ImageReward-v1.0 preference score, run in a separate interpreter."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        image_path = Path(tmp_dir) / "image.png"
        Image.fromarray(_as_rgb_uint8(image)).save(image_path)
        return compute_image_rewards(pairs=[(prompt, image_path)])[0]


def compute_image_rewards(*, pairs: list[tuple[str, Path]]) -> list[float]:
    """Score already-saved images, one interpreter launch for the whole list.

    The launch loads a 1.7 GB checkpoint, so a sweep must not call
    compute_image_reward in a loop.
    """
    python = image_reward_python()
    if python is None:
        raise RuntimeError(_IMAGE_REWARD_HINT)

    payload = json.dumps(
        {
            "model_name": IMAGE_REWARD_MODEL_NAME,
            "pairs": [[prompt, str(path)] for prompt, path in pairs],
        }
    )
    completed = subprocess.run(
        # -I: an inherited PYTHONPATH/PYTHONHOME would put sglang's transformers 5
        # ahead of that interpreter's own transformers<5 -- the conflict avoided.
        [python, "-I", "-c", _IMAGE_REWARD_SCRIPT],
        input=payload,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"ImageReward scorer failed (exit {completed.returncode}) using {python}:\n"
            f"{completed.stderr.strip()}\n{_IMAGE_REWARD_HINT}"
        )

    scores = _parse_image_reward_stdout(completed.stdout)
    if len(scores) != len(pairs):
        raise RuntimeError(
            f"ImageReward scorer returned {len(scores)} scores for {len(pairs)} images"
        )
    return scores


def _parse_image_reward_stdout(stdout: str) -> list[float]:
    # ImageReward's loader prints progress bars and a checkpoint banner; the
    # scores are the last line, so parse from the end rather than the start.
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return [float(score) for score in json.loads(line)["scores"]]
    raise RuntimeError(f"ImageReward scorer printed no score:\n{stdout}")


def _as_rgb_uint8(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected an RGB HWC image, got shape={image.shape}")
    # Refuse anything but uint8 rather than rescale: a [0, 1] float array cast to
    # uint8 is a near-black image that still scores, i.e. a wrong number.
    if image.dtype != np.uint8:
        raise ValueError(
            f"Expected a uint8 [0, 255] image, got dtype={image.dtype}; "
            "convert with PIL before scoring"
        )
    return image
