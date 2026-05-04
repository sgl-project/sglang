# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

BAGEL_KEY_CATEGORY_QWEN2_SHARED = "qwen2_shared"
BAGEL_KEY_CATEGORY_MOT_GEN_BRANCH = "mot_gen_branch"
BAGEL_KEY_CATEGORY_BAGEL_OUTER = "bagel_outer"
BAGEL_KEY_CATEGORY_VIT_VAE = "vit_vae"
BAGEL_KEY_CATEGORY_UNKNOWN = "unknown"

BAGEL_KEY_CATEGORIES = (
    BAGEL_KEY_CATEGORY_QWEN2_SHARED,
    BAGEL_KEY_CATEGORY_MOT_GEN_BRANCH,
    BAGEL_KEY_CATEGORY_BAGEL_OUTER,
    BAGEL_KEY_CATEGORY_VIT_VAE,
    BAGEL_KEY_CATEGORY_UNKNOWN,
)

_BAGEL_OUTER_PREFIXES = (
    "time_embedder.",
    "vae2llm.",
    "llm2vae.",
    "latent_pos_embed.",
    "connector.",
    "vit_pos_embed.",
)
_VIT_VAE_PREFIXES = (
    "vit_model.",
    "vae_model.",
    "vae.",
    "autoencoder.",
    "encoder.",
    "decoder.",
    "quant_conv.",
    "post_quant_conv.",
)
_VIT_VAE_SUBSTRINGS = (
    ".vit_model.",
    ".vae_model.",
    ".autoencoder.",
)


class BAGELCheckpointKeyError(RuntimeError):
    """Raised when BAGEL checkpoint keys cannot be loaded or classified."""


@dataclass(frozen=True)
class BAGELCheckpointKeyRecord:
    key: str
    category: str


@dataclass(frozen=True)
class BAGELCheckpointKeySummary:
    total: int
    counts: dict[str, int]
    examples: dict[str, tuple[str, ...]]

    def ratio(self, category: str) -> float:
        if self.total == 0:
            return 0.0
        return self.counts.get(category, 0) / self.total


def classify_bagel_checkpoint_key(key: str) -> str:
    """Classify one BAGEL checkpoint key by the SRT-native loading surface."""

    if _is_mot_gen_branch_key(key):
        return BAGEL_KEY_CATEGORY_MOT_GEN_BRANCH
    if _is_qwen2_shared_key(key):
        return BAGEL_KEY_CATEGORY_QWEN2_SHARED
    if key.startswith(_BAGEL_OUTER_PREFIXES):
        return BAGEL_KEY_CATEGORY_BAGEL_OUTER
    if key.startswith(_VIT_VAE_PREFIXES) or any(
        marker in key for marker in _VIT_VAE_SUBSTRINGS
    ):
        return BAGEL_KEY_CATEGORY_VIT_VAE
    return BAGEL_KEY_CATEGORY_UNKNOWN


def classify_bagel_checkpoint_keys(
    keys: Iterable[str],
) -> list[BAGELCheckpointKeyRecord]:
    return [
        BAGELCheckpointKeyRecord(key=key, category=classify_bagel_checkpoint_key(key))
        for key in keys
    ]


def summarize_bagel_checkpoint_keys(
    keys: Iterable[str],
    *,
    max_examples_per_category: int = 5,
) -> BAGELCheckpointKeySummary:
    counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = defaultdict(list)

    for key in keys:
        category = classify_bagel_checkpoint_key(key)
        counts[category] += 1
        if len(examples[category]) < max_examples_per_category:
            examples[category].append(key)

    normalized_counts = {
        category: counts.get(category, 0) for category in BAGEL_KEY_CATEGORIES
    }
    return BAGELCheckpointKeySummary(
        total=sum(normalized_counts.values()),
        counts=normalized_counts,
        examples={
            category: tuple(examples.get(category, ()))
            for category in BAGEL_KEY_CATEGORIES
        },
    )


def load_bagel_checkpoint_keys(path: str | Path) -> tuple[str, ...]:
    """Load tensor keys from a BAGEL checkpoint file, directory, or HF index."""

    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise BAGELCheckpointKeyError(
            f"BAGEL checkpoint path does not exist: {checkpoint_path}"
        )

    if checkpoint_path.is_dir():
        return _load_bagel_checkpoint_keys_from_dir(checkpoint_path)
    if checkpoint_path.suffix == ".json":
        return _load_safetensors_index_keys(checkpoint_path)
    if checkpoint_path.suffix == ".safetensors":
        return _load_safetensors_file_keys(checkpoint_path)

    raise BAGELCheckpointKeyError(
        "BAGEL checkpoint key classifier expects a directory, "
        f".safetensors file, or safetensors index json, got {checkpoint_path}"
    )


def format_bagel_checkpoint_key_summary(summary: BAGELCheckpointKeySummary) -> str:
    lines = [f"total: {summary.total}"]
    for category in BAGEL_KEY_CATEGORIES:
        count = summary.counts.get(category, 0)
        lines.append(f"{category}: {count} ({summary.ratio(category):.2%})")
        for key in summary.examples.get(category, ()):
            lines.append(f"  - {key}")
    return "\n".join(lines)


def _is_mot_gen_branch_key(key: str) -> bool:
    return "_moe_gen" in key or key.endswith(".norm_moe_gen.weight")


def _is_qwen2_shared_key(key: str) -> bool:
    if key.startswith("language_model."):
        return True
    return key.startswith(
        (
            "model.embed_tokens.",
            "model.layers.",
            "model.norm.",
            "lm_head.",
        )
    )


def _load_bagel_checkpoint_keys_from_dir(checkpoint_dir: Path) -> tuple[str, ...]:
    index_files = sorted(checkpoint_dir.glob("*.safetensors.index.json"))
    if index_files:
        return _load_safetensors_index_keys(index_files[0])

    ema_file = checkpoint_dir / "ema.safetensors"
    if ema_file.exists():
        return _load_safetensors_file_keys(ema_file)

    safetensor_files = sorted(checkpoint_dir.glob("*.safetensors"))
    if not safetensor_files:
        raise BAGELCheckpointKeyError(
            f"No safetensors checkpoint found in BAGEL directory: {checkpoint_dir}"
        )

    keys: list[str] = []
    for safetensor_file in safetensor_files:
        keys.extend(_load_safetensors_file_keys(safetensor_file))
    return tuple(sorted(set(keys)))


def _load_safetensors_index_keys(index_path: Path) -> tuple[str, ...]:
    with index_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict):
        raise BAGELCheckpointKeyError(
            f"Safetensors index missing weight_map: {index_path}"
        )
    return tuple(sorted(str(key) for key in weight_map))


def _load_safetensors_file_keys(safetensor_path: Path) -> tuple[str, ...]:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise BAGELCheckpointKeyError(
            "Reading .safetensors keys requires the safetensors package"
        ) from exc

    with safe_open(safetensor_path, framework="pt", device="cpu") as f:
        return tuple(sorted(f.keys()))


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Classify BAGEL checkpoint keys by SRT-native loading surface."
    )
    parser.add_argument("checkpoint", help="BAGEL checkpoint directory/file/index")
    parser.add_argument(
        "--max-examples-per-category",
        type=int,
        default=5,
        help="Number of example keys to print for each category.",
    )
    args = parser.parse_args(argv)

    keys = load_bagel_checkpoint_keys(args.checkpoint)
    summary = summarize_bagel_checkpoint_keys(
        keys,
        max_examples_per_category=args.max_examples_per_category,
    )
    print(format_bagel_checkpoint_key_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
