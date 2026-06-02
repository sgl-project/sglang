# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0

import json
from typing import Any, Optional

_CONSTRAINT_KEYS = ("json_schema", "regex", "ebnf", "structural_tag")


def response_format_to_json_schema(
    response_format: Optional[dict[str, Any]],
) -> Optional[str]:
    """Convert an OpenAI response_format dict to SGLang's json_schema string."""
    if not isinstance(response_format, dict):
        return None

    response_format_type = response_format.get("type")
    if response_format_type == "json_object":
        return '{"type": "object"}'
    if response_format_type != "json_schema":
        return None

    json_schema = response_format.get("json_schema")
    if isinstance(json_schema, dict) and "schema" in json_schema:
        schema = json_schema["schema"]
    else:
        schema = response_format.get("schema")

    if schema is None:
        return None
    if isinstance(schema, str):
        return schema
    return json.dumps(schema)


def apply_response_format_to_sampling_params(
    sampling_params: Optional[Any],
    response_format: Optional[dict[str, Any]],
) -> Optional[Any]:
    """Apply response_format to sampling params unless a constraint already exists."""
    if response_format is None:
        return sampling_params

    if isinstance(sampling_params, list):
        return [
            apply_response_format_to_sampling_params(params, response_format)
            for params in sampling_params
        ]

    params = dict(sampling_params or {})
    if any(params.get(key) for key in _CONSTRAINT_KEYS):
        return params

    json_schema = response_format_to_json_schema(response_format)
    if json_schema is not None:
        params["json_schema"] = json_schema
    return params
