from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True, slots=True, kw_only=True)
class _ModeConfig:
    model_path: str
    json_model_override_args: Optional[str] = None


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
}
