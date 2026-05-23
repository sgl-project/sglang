from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True, slots=True, kw_only=True)
class _ModeConfig:
    """Mode-specific server launch config so per-mode test classes only set
    `model_mode = "mha"` / `"swa"`, not individual flags. All flags collected here.

    Fields:
        model_path: HF model id used by popen_launch_server.
        json_model_override_args: JSON string passed to --json-model-override-args, or
            None to omit the flag entirely.
    """

    model_path: str
    json_model_override_args: Optional[str] = None


_MODE_CONFIGS: dict[str, _ModeConfig] = {
    "mha": _ModeConfig(
        model_path="Qwen/Qwen3-0.6B",
    ),
    "swa": _ModeConfig(
        model_path="google/gemma-4-E2B-it",
    ),
}
