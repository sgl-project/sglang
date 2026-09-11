# Copyright 2023-2024 SGLang Team
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
"""Normalization shared by chat-shaped API inputs."""

from typing import Any, Dict


def _has_message_level_tools(messages: Any) -> bool:
    if not isinstance(messages, list):
        return False
    return any(
        isinstance(message, dict)
        and isinstance(message.get("role"), str)
        and message["role"].lower() in ("system", "developer")
        and bool(message.get("tools"))
        for message in messages
    )


def set_tool_choice_default(values):
    if values.get("tool_choice") is None:
        if values.get("tools") is None and not _has_message_level_tools(
            values.get("messages")
        ):
            values["tool_choice"] = "none"
        else:
            values["tool_choice"] = "auto"
    return values


def validate_reasoning_effort_type(value):
    if isinstance(value, bool):
        raise ValueError("reasoning_effort must not be a boolean")
    return value


def normalize_reasoning_inputs(values: Dict):
    r = values.get("reasoning")
    thinking = None

    if r is not None and isinstance(r, dict):
        effort = r.get("effort")
        if effort is None:
            effort = r.get("reasoning_effort")
        if isinstance(effort, str) and effort in {
            "none",
            "minimal",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        }:
            values["reasoning_effort"] = effort
        elif isinstance(effort, (int, float)) and not isinstance(effort, bool):
            values["reasoning_effort"] = float(effort)
        elif isinstance(effort, str):
            # Keep parity with the top-level reasoning_effort field, whose
            # lax union coerces numeric strings; range checks then apply.
            try:
                values["reasoning_effort"] = float(effort)
            except ValueError as exc:
                raise ValueError(f"invalid reasoning effort: {effort!r}") from exc
        elif effort is not None:
            raise ValueError(f"invalid reasoning effort: {effort!r}")

        enabled = (
            r.get("enabled") if r.get("enabled") is not None else r.get("enable", False)
        )
        if isinstance(enabled, str):
            enabled = enabled.strip().lower() in {"1", "true", "yes", "y", "on"}
        if enabled:
            thinking = True

    effort = values.get("reasoning_effort")
    if effort is not None:
        thinking = effort != "none"

    if thinking is not None:
        ctk = values.get("chat_template_kwargs")
        if not isinstance(ctk, dict):
            ctk = {}
        # different models check different keys:
        # - "thinking" for deepseek-v3, kimi_k2
        # - "enable_thinking" for qwen3, glm45, nemotron_3, interns1
        ctk.setdefault("thinking", thinking)
        ctk.setdefault("enable_thinking", thinking)
        values["chat_template_kwargs"] = ctk

    return values


def set_json_schema(values):
    response_format = values.get("response_format")
    if not response_format:
        return values

    if response_format.get("type") != "json_schema":
        return values

    schema = response_format.pop("schema", None)
    json_schema = response_format.get("json_schema")

    if json_schema:
        return values

    if schema:
        name_ = schema.get("title", "Schema")
        strict_ = None
        if "properties" in schema and "strict" in schema["properties"]:
            item = schema["properties"].pop("strict", None)
            strict_ = bool(item and item.get("default", False))

        response_format["json_schema"] = {
            "name": name_,
            "schema": schema,
            "strict": strict_,
        }

    return values
