# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Checkpoint loading for response-template parsers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sglang.srt.parser.chat_parsing.response_templates import ResponseTemplate

RESPONSE_TEMPLATE_CONFIG_KEY = "response_template"
SUPPORTED_RESPONSE_TEMPLATE_FIELDS = frozenset(
    {
        "thinking",
        "content",
        "tool_calls",
    }
)


def validate_response_template_for_serving(template: dict) -> ResponseTemplate:
    """Validate the template and reject semantic fields serving cannot route."""
    from sglang.srt.parser.chat_parsing.response_templates import (
        load_response_template,
    )

    loaded = load_response_template(template)
    fields = set(loaded.fields)
    unsupported = (fields - SUPPORTED_RESPONSE_TEMPLATE_FIELDS) | (
        set(loaded.defaults) - {"role"}
    )
    if unsupported:
        raise ValueError(
            "response_template contains unsupported semantic fields: "
            f"{sorted(unsupported)}. Supported fields are: "
            f"{sorted(SUPPORTED_RESPONSE_TEMPLATE_FIELDS)}"
        )
    for name in ("thinking", "content"):
        field = template.get("fields", {}).get(name)
        if not isinstance(field, dict):
            continue
        if (
            field.get("content", "text") != "text"
            or field.get("transform") is not None
            or field.get("join") is not None
            or field.get("content_args", {}).get("strip") is True
        ):
            raise ValueError(
                f"response_template field {name!r} uses semantics that cannot "
                "be streamed by the OpenAI serving adapter"
            )
    return loaded


def resolve_detector_response_template(
    tokenizer: Any | None,
    fallback: dict | None,
) -> dict | None:
    """Load `response_template` from the tokenizer, then use the fallback."""
    if tokenizer is None:
        return fallback

    template = getattr(tokenizer, RESPONSE_TEMPLATE_CONFIG_KEY, None)
    if isinstance(template, dict):
        return template

    init_kwargs = getattr(tokenizer, "init_kwargs", None) or {}
    template = init_kwargs.get(RESPONSE_TEMPLATE_CONFIG_KEY)
    if isinstance(template, dict):
        return template

    return fallback
