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

from typing import Any

RESPONSE_TEMPLATE_CONFIG_KEY = "response_template"


def resolve_detector_response_template(
    tokenizer: Any | None,
    fallback: dict | None,
) -> dict | None:
    """Load `response_template` from the tokenizer, then use the fallback."""
    if tokenizer is None:
        return fallback

    cached = getattr(tokenizer, RESPONSE_TEMPLATE_CONFIG_KEY, None)
    if isinstance(cached, dict):
        return cached

    init_kwargs = getattr(tokenizer, "init_kwargs", None) or {}
    template = init_kwargs.get(RESPONSE_TEMPLATE_CONFIG_KEY)
    if isinstance(template, dict):
        setattr(tokenizer, RESPONSE_TEMPLATE_CONFIG_KEY, template)
        return template

    return fallback
