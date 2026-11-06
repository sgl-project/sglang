from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_EAGLE,
    DEFAULT_TARGET_MODEL_EAGLE,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class _ModeConfig:
    model_path: str
    json_model_override_args: Optional[str] = None
    server_args: tuple[str, ...] = ()


_MODE_CONFIGS: dict[str, _ModeConfig] = {
    "mha": _ModeConfig(
        model_path="Qwen/Qwen3-0.6B",
    ),
    "swa": _ModeConfig(
        model_path="google/gemma-4-E2B-it",
    ),
    "dsv4": _ModeConfig(
        model_path="deepseek-ai/DeepSeek-V4-Flash",
    ),
    "eagle": _ModeConfig(
        model_path=DEFAULT_TARGET_MODEL_EAGLE,
        server_args=(
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-draft-model-path",
            DEFAULT_DRAFT_MODEL_EAGLE,
            "--speculative-num-steps",
            "3",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "4",
            "--context-length",
            "4096",
        ),
    ),
}
