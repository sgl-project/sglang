"""Default kwargs for spinning up an Engine in mock-model + canary mode.

Mock-model mode is testing-only; the default-filling logic lives here so the
main code (server_args) does not have to know about it.
"""

from __future__ import annotations

import json
import os
from typing import Any


def mock_model_engine_kwargs(**overrides: Any) -> dict[str, Any]:
    """Return Engine() kwargs that wire up mock-model + canary together.

    Defaults:
        load_format = "dummy"            (no real weights loaded)
        json_model_override_args = '{"num_hidden_layers": 1}'
        sampling_backend = "oracle"      (gate for install_token_oracle_from_env)
        kv_canary = "raise"              (mock-model without canary is mostly pointless)

    Also sets ``SGLANG_KV_CANARY_INPUT_CHECK=1`` in the current process env so
    the canary's input-id verification path turns on when the engine starts.
    This is a side effect because input-check is mock-model-only and is no
    longer a server arg; the env var is the only injection path.

    Caller-supplied overrides win; for json_model_override_args, the override
    dict is merged on top of the default so callers can add extra keys without
    losing num_hidden_layers=1.
    """
    os.environ["SGLANG_KV_CANARY_INPUT_CHECK"] = "1"

    defaults: dict[str, Any] = {
        "load_format": "dummy",
        "json_model_override_args": json.dumps({"num_hidden_layers": 1}),
        "sampling_backend": "oracle",
        "kv_canary": "raise",
    }
    if "json_model_override_args" in overrides:
        user_dict = json.loads(overrides.pop("json_model_override_args"))
        merged = {"num_hidden_layers": 1, **user_dict}
        defaults["json_model_override_args"] = json.dumps(merged)
    defaults.update(overrides)
    return defaults
