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
"""Completion templates."""


import dataclasses
import logging
from enum import auto

from sglang.srt.entrypoints.openai.protocol import CompletionRequest

logger = logging.getLogger(__name__)
completion_template_name = None


class FimPosition:
    """Position of fim middle token."""

    MIDDLE = auto()
    END = auto()


@dataclasses.dataclass
class CompletionTemplate:
    """A class that manages completion prompt templates. only for code completion currently."""

    # The name of this template
    name: str

    # the fim begin token
    fim_begin_token: str

    # The fim middle token
    fim_middle_token: str

    # The fim end token
    fim_end_token: str

    # The position of the fim middle token
    fim_position: FimPosition


# A global registry for all completion templates
completion_templates: dict[str, CompletionTemplate] = {}


def register_completion_template(template: CompletionTemplate, override: bool = False):
    """Register a new completion template."""
    if not override:
        assert (
            template.name not in completion_templates
        ), f"{template.name} has been registered."

    completion_templates[template.name] = template


def completion_template_exists(template_name: str) -> bool:
    return template_name in completion_templates


def is_completion_template_defined() -> bool:
    global completion_template_name
    return completion_template_name is not None


def generate_completion_prompt_from_request(request: CompletionRequest) -> str:
    global completion_template_name
    if request.suffix == "":
        return request.prompt

    return generate_completion_prompt(
        request.prompt, request.suffix, completion_template_name
    )


def generate_completion_prompt(prompt: str, suffix: str, template_name: str) -> str:

    completion_template = completion_templates[template_name]
    fim_begin_token = completion_template.fim_begin_token
    fim_middle_token = completion_template.fim_middle_token
    fim_end_token = completion_template.fim_end_token
    fim_position = completion_template.fim_position

    if fim_position == FimPosition.MIDDLE:
        prompt = f"{fim_begin_token}{prompt}{fim_middle_token}{suffix}{fim_end_token}"
    elif fim_position == FimPosition.END:
        prompt = f"{fim_begin_token}{prompt}{fim_end_token}{suffix}{fim_middle_token}"

    return prompt


register_completion_template(
    CompletionTemplate(
        name="deepseek_coder",
        fim_begin_token="<｜fim▁begin｜>",
        fim_middle_token="<｜fim▁hole｜>",
        fim_end_token="<｜fim▁end｜>",
        fim_position=FimPosition.MIDDLE,
    )
)


register_completion_template(
    CompletionTemplate(
        name="star_coder",
        fim_begin_token="<fim_prefix>",
        fim_middle_token="<fim_middle>",
        fim_end_token="<fim_suffix>",
        fim_position=FimPosition.END,
    )
)

register_completion_template(
    CompletionTemplate(
        name="qwen_coder",
        fim_begin_token="<|fim_prefix|>",
        fim_middle_token="<|fim_middle|>",
        fim_end_token="<|fim_suffix|>",
        fim_position=FimPosition.END,
    )
)
