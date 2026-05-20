from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs


_MOCK_MODEL_DEFAULTS: Dict[str, Any] = {
    "load_format": "dummy",
    "sampling_backend": "oracle",
    "kv_canary": "raise",
    "kv_canary_input_check_mode": "ON",
}

_DEFAULT_SENTINELS: Dict[str, Any] = {
    "load_format": "auto",
    "sampling_backend": None,
    "kv_canary": "off",
    "kv_canary_input_check_mode": None,
}


def apply_mock_model_defaults(server_args: "ServerArgs") -> "ServerArgs":
    """If server_args opts into mock_model, fill in sensible defaults for the dependent flags.

    Read trigger: server_args.mock_model_enabled (a new sglang flag added in this PR).

    When enabled, the following flags get default values only if the user has NOT already set them:
        - load_format = "dummy"            (no real weights loaded)
        - sampling_backend = "oracle"      (so install_oracle_sampler is the live backend)
        - kv_canary = "raise"        (mock_model without canary is mostly pointless)
        - kv_canary_input_check_mode = "ON"  (turn on the input-id verification path)

    Additionally injects ``{"num_hidden_layers": 1}`` into ``json_model_override_args`` so the
    mock model loads with a single layer. A user-supplied ``num_hidden_layers`` is preserved.

    User-specified values are preserved - apply_mock_model_defaults only fills holes. Returns the
    (possibly mutated) ServerArgs. Idempotent: applying twice is a no-op since holes are already filled.

    Called once during ServerArgs post-parse, BEFORE ModelRunner instantiation.
    """
    if not server_args.mock_model_enabled:
        return server_args

    updates: Dict[str, Any] = {}
    for field_name, default_value in _MOCK_MODEL_DEFAULTS.items():
        current = getattr(server_args, field_name)
        sentinel = _DEFAULT_SENTINELS[field_name]
        if current == sentinel:
            updates[field_name] = default_value

    merged_json = _merge_num_hidden_layers(server_args.json_model_override_args)
    if merged_json != server_args.json_model_override_args:
        updates["json_model_override_args"] = merged_json

    if not updates:
        return server_args

    return dataclasses.replace(server_args, **updates)


def _merge_num_hidden_layers(raw: str) -> str:
    parsed: Dict[str, Any] = json.loads(raw) if raw else {}
    if "num_hidden_layers" in parsed:
        return raw
    merged: Dict[str, Any] = {"num_hidden_layers": 1, **parsed}
    return json.dumps(merged)
