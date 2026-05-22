from __future__ import annotations

import json
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
        model_path="google/gemma-3-1b-it",
        # Gemma 3 1B-it's HF config carries layer-typed rope params; SGLang's
        # parser also needs an explicit rope_type / factor on full_attention,
        # otherwise the swa-mode server fails to launch. Passing these via
        # --json-model-override-args avoids touching the model source.
        json_model_override_args=json.dumps(
            {
                "rope_parameters": {
                    "sliding_attention": {
                        "rope_type": "default",
                        "rope_theta": 10000,
                    },
                    "full_attention": {
                        "rope_type": "default",
                        "rope_theta": 1000000,
                        "factor": 8.0,
                    },
                },
            }
        ),
    ),
}
