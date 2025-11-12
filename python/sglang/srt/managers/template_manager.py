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
"""
Centralized template management for chat templates and completion templates.

This module provides a unified interface for managing both chat conversation templates
and code completion templates, eliminating global state and improving modularity.
"""

import json
import logging
import os
import re
from typing import Optional

from sglang.srt.parser.code_completion_parser import (
    CompletionTemplate,
    FimPosition,
    completion_template_exists,
    register_completion_template,
)
from sglang.srt.parser.conversation import (
    Conversation,
    SeparatorStyle,
    chat_template_exists,
    get_conv_template_by_model_path,
    register_conv_template,
)
from sglang.srt.parser.jinja_template_utils import detect_jinja_template_content_format

logger = logging.getLogger(__name__)


class TemplateManager:
    """
    Centralized manager for chat and completion templates.

    This class encapsulates all template-related state and operations,
    eliminating the need for global variables and providing a clean
    interface for template management.
    """

    def __init__(self):
        self._chat_template_name: Optional[str] = None
        self._completion_template_name: Optional[str] = None
        self._jinja_template_content_format: Optional[str] = "openai"
        self._force_reasoning: bool = False

    @property
    def chat_template_name(self) -> Optional[str]:
        """Get the current chat template name."""
        return self._chat_template_name

    @property
    def completion_template_name(self) -> Optional[str]:
        """Get the current completion template name."""
        return self._completion_template_name

    @property
    def jinja_template_content_format(self) -> Optional[str]:
        """Get the detected template content format ('string' or 'openai' or None)."""
        return self._jinja_template_content_format

    @property
    def force_reasoning(self) -> bool:
        """
        Check if the current chat template enforces reasoning/thinking.

        Returns:
            True if the template contains reasoning patterns like <think> tags
        """
        return self._force_reasoning

    def _detect_reasoning_pattern(self, template: str) -> bool:
        """
        Detect if the chat template contains reasoning/thinking patterns.
        """
        if template is None:
            return False

        # TODO: remove this hard code the reasoning pattern
        force_reasoning_pattern = r"<\|im_start\|>assistant\\n<think>\\n"
        has_reasoning = re.search(force_reasoning_pattern, template) is not None

        if has_reasoning:
            logger.info("Detected the force reasoning pattern in chat template.")

        return has_reasoning

    def load_chat_template(
        self, tokenizer_manager, chat_template_arg: Optional[str], model_path: str
    ) -> None:
        """
        Load a chat template from various sources.

        Args:
            tokenizer_manager: The tokenizer manager instance
            chat_template_arg: Template name, file path, or None to auto-detect
            model_path: Path to the model
        """
        if chat_template_arg:
            self._load_explicit_chat_template(tokenizer_manager, chat_template_arg)
        else:
            # Guess chat template from model path
            self.guess_chat_template_from_model_path(model_path)

            # If no pre-defined template was found, fallback to HuggingFace template
            if self._chat_template_name is None:
                # Try HuggingFace template first
                hf_template = self._resolve_hf_chat_template(tokenizer_manager)
                if hf_template:
                    # override the chat template
                    if tokenizer_manager.tokenizer:
                        tokenizer_manager.tokenizer.chat_template = hf_template
                    self._jinja_template_content_format = (
                        detect_jinja_template_content_format(hf_template)
                    )
                    logger.info(
                        f"Using default HuggingFace chat template with detected content format: {self._jinja_template_content_format}"
                    )
                else:
                    # Default to string content format if no template was found
                    self._jinja_template_content_format = "string"
                    logger.info(
                        "No chat template found, defaulting to 'string' content format"
                    )

        # Detect reasoning pattern from chat template
        if tokenizer_manager.tokenizer:
            self._force_reasoning = self._detect_reasoning_pattern(
                tokenizer_manager.tokenizer.chat_template
            )

    def _load_explicit_chat_template(
        self, tokenizer_manager, chat_template_arg: str
    ) -> None:
        """Load explicitly specified chat template."""
        logger.info(f"Loading chat template from argument: {chat_template_arg}")

        if chat_template_exists(chat_template_arg):
            self._chat_template_name = chat_template_arg
            return

        if not os.path.exists(chat_template_arg):
            raise RuntimeError(
                f"Chat template {chat_template_arg} is not a built-in template name "
                "or a valid chat template file path."
            )

        if chat_template_arg.endswith(".jinja"):
            self._load_jinja_template(tokenizer_manager, chat_template_arg)
        else:
            self._load_json_chat_template(chat_template_arg)

    def guess_chat_template_from_model_path(self, model_path: str) -> None:
        """
        Infer chat template name from model path.

        Args:
            model_path: Path to the model
        """
        template_name = get_conv_template_by_model_path(model_path)
        if template_name is not None:
            logger.info(f"Inferred chat template from model path: {template_name}")
            self._chat_template_name = template_name

    def load_completion_template(self, completion_template_arg: str) -> None:
        """
        Load completion template for code completion.

        Args:
            completion_template_arg: Template name or file path
        """
        logger.info(f"Loading completion template: {completion_template_arg}")

        if not completion_template_exists(completion_template_arg):
            if not os.path.exists(completion_template_arg):
                raise RuntimeError(
                    f"Completion template {completion_template_arg} is not a built-in template name "
                    "or a valid completion template file path."
                )

            self._load_json_completion_template(completion_template_arg)
        else:
            self._completion_template_name = completion_template_arg

    def initialize_templates(
        self,
        tokenizer_manager,
        model_path: str,
        chat_template: Optional[str] = None,
        completion_template: Optional[str] = None,
    ) -> None:
        """
        Initialize all templates based on provided configuration.

        Args:
            tokenizer_manager: The tokenizer manager instance
            model_path: Path to the model
            chat_template: Optional chat template name/path
            completion_template: Optional completion template name/path
        """
        # Load chat template
        self.load_chat_template(tokenizer_manager, chat_template, model_path)

        # Load completion template
        if completion_template:
            self.load_completion_template(completion_template)

    def _load_jinja_template(self, tokenizer_manager, template_path: str) -> None:
        """Load a Jinja template file."""
        with open(template_path, "r") as f:
            chat_template = "".join(f.readlines()).strip("\n")
        tokenizer_manager.tokenizer.chat_template = chat_template.replace("\\n", "\n")
        self._chat_template_name = None
        # Detect content format from the loaded template
        self._jinja_template_content_format = detect_jinja_template_content_format(
            chat_template
        )
        logger.info(
            f"Detected user specified Jinja chat template with content format: {self._jinja_template_content_format}"
        )

    def _load_json_chat_template(self, template_path: str) -> None:
        """Load a JSON chat template file."""
        assert template_path.endswith(
            ".json"
        ), "unrecognized format of chat template file"

        with open(template_path, "r") as filep:
            template = json.load(filep)
            try:
                sep_style = SeparatorStyle[template["sep_style"]]
            except KeyError:
                raise ValueError(
                    f"Unknown separator style: {template['sep_style']}"
                ) from None

            register_conv_template(
                Conversation(
                    name=template["name"],
                    system_template=template["system"] + "\n{system_message}",
                    system_message=template.get("system_message", ""),
                    roles=(template["user"], template["assistant"]),
                    sep_style=sep_style,
                    sep=template.get("sep", "\n"),
                    stop_str=template["stop_str"],
                ),
                override=True,
            )
        self._chat_template_name = template["name"]

    def _load_json_completion_template(self, template_path: str) -> None:
        """Load a JSON completion template file."""
        assert template_path.endswith(
            ".json"
        ), "unrecognized format of completion template file"

        with open(template_path, "r") as filep:
            template = json.load(filep)
            try:
                fim_position = FimPosition[template["fim_position"]]
            except KeyError:
                raise ValueError(
                    f"Unknown fim position: {template['fim_position']}"
                ) from None

            register_completion_template(
                CompletionTemplate(
                    name=template["name"],
                    fim_begin_token=template["fim_begin_token"],
                    fim_middle_token=template["fim_middle_token"],
                    fim_end_token=template["fim_end_token"],
                    fim_position=fim_position,
                ),
                override=True,
            )
        self._completion_template_name = template["name"]

    def _resolve_hf_chat_template(self, tokenizer_manager) -> Optional[str]:
        """
        Resolve HuggingFace chat template.

        Returns the chat template string if found, None otherwise.
        """
        try:
            if processor := tokenizer_manager.processor:
                if hasattr(processor, "chat_template") and processor.chat_template:
                    return processor.chat_template
            if tokenizer := tokenizer_manager.tokenizer:
                if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
                    return tokenizer.chat_template
        except Exception as e:
            logger.debug(f"Error getting chat template: {e}")

        logger.debug("No HuggingFace chat template found")
        return None
