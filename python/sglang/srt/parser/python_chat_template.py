"""Chat templates rendered in Python instead of the checkpoint's Jinja template.

When one applies to the served checkpoint, the serving tokenizer's
``apply_chat_template`` renders through Python, so every caller keeps the
transformers calling convention and the request pipeline in front of it is
unchanged; the features sglang would otherwise derive from the Jinja source
(content format, reasoning mode, tool-result reordering, inline system
messages, the parsers ``auto`` resolves to) are declared on the template
instead.
"""

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from transformers.utils import get_json_schema

from sglang.srt.parser.glm53_chat_template import render_glm53, render_glm53_flash
from sglang.srt.parser.template_detection import (
    ReasoningToggleConfig,
    is_glm53_template,
)


@dataclass(frozen=True)
class PythonChatTemplate:
    """A chat template implemented in Python, with the features sglang needs.

    ``render(messages, tools=None, add_generation_prompt=False, **kwargs)``
    receives the same variables a Jinja template does and returns the prompt.
    """

    name: str
    render: Callable[..., str]
    content_format: str
    reasoning_config: Optional[ReasoningToggleConfig]
    may_reorder_tool_results: bool
    supports_inline_system: bool
    reasoning_parser: Optional[str] = None
    tool_call_parser: Optional[str] = None

    @property
    def force_reasoning(self) -> bool:
        return self.reasoning_config is not None and self.reasoning_config.always_on


# Matching what template detection derives from both models' Jinja source.
_GLM53_FEATURES: Dict[str, Any] = dict(
    content_format="openai",
    reasoning_config=ReasoningToggleConfig(special_case="always"),
    may_reorder_tool_results=True,
    supports_inline_system=True,
    reasoning_parser="glm45",
    tool_call_parser="glm47",
)

GLM53 = PythonChatTemplate(name="glm-5.3", render=render_glm53, **_GLM53_FEATURES)
GLM53_FLASH = PythonChatTemplate(
    name="glm-5.3-flash", render=render_glm53_flash, **_GLM53_FEATURES
)

PYTHON_CHAT_TEMPLATES: Dict[str, PythonChatTemplate] = {
    template.name: template for template in (GLM53, GLM53_FLASH)
}

_GLM53_TEMPLATES_BY_ARCH = {
    "GlmMoeDsaForCausalLM": GLM53,
    "Glm5NextForConditionalGeneration": GLM53_FLASH,
}


def get_python_chat_template(name: str) -> Optional[PythonChatTemplate]:
    """Look up a built-in Python chat template by name (``--chat-template``)."""
    return PYTHON_CHAT_TEMPLATES.get(name)


def may_replace_jinja_template(hf_chat_template: Optional[str]) -> bool:
    """Whether a built-in Python template can replace this Jinja template.

    Only the checkpoint's template is inspected, so callers can skip loading
    the model config for every other template.
    """
    return isinstance(hf_chat_template, str) and is_glm53_template(hf_chat_template)


def resolve_python_chat_template(
    hf_config: Any, hf_chat_template: Optional[str]
) -> Optional[PythonChatTemplate]:
    """The Python template that replaces this checkpoint's Jinja template.

    GLM-5, GLM-5.1 and GLM-5.2 share ``GlmMoeDsaForCausalLM`` with GLM-5.3 but
    ship different templates, so the architecture alone does not decide: the
    checkpoint's template must be a GLM-5.3 one, and its multimodal block must
    match the architecture (GLM-5.3 is text-only, GLM-5.3-Flash multimodal).
    """
    architectures = getattr(hf_config, "architectures", None) or []
    template = _GLM53_TEMPLATES_BY_ARCH.get(architectures[0] if architectures else "")
    if template is None or not may_replace_jinja_template(hf_chat_template):
        return None
    multimodal = "<|begin_of_image|>" in hf_chat_template
    if multimodal != (template is GLM53_FLASH):
        return None
    return template


def _tool_schemas(tools):
    # Same conversion transformers applies before rendering a Jinja template.
    if tools is None:
        return None
    schemas = []
    for tool in tools:
        if isinstance(tool, dict):
            schemas.append(tool)
        elif inspect.isfunction(tool) or inspect.ismethod(tool):
            schemas.append(get_json_schema(tool))
        else:
            raise ValueError(
                "Tools should either be a JSON schema, or a callable function with "
                "type hints and a docstring suitable for auto-conversion to a schema."
            )
    return schemas


def install_python_chat_template(tokenizer: Any, template: PythonChatTemplate) -> None:
    """Make ``tokenizer.apply_chat_template`` render with ``template``.

    Mirrors transformers' ``apply_chat_template``: tool schemas, special tokens
    as template variables, batched conversations, and tokenization without
    added special tokens. A caller-supplied ``chat_template`` and the Jinja-only
    ``continue_final_message`` / ``return_assistant_tokens_mask`` modes still go
    to the original Jinja implementation.
    """
    jinja_apply_chat_template = getattr(
        tokenizer, "_sglang_jinja_apply_chat_template", tokenizer.apply_chat_template
    )

    def apply_chat_template(
        conversation,
        tools=None,
        documents=None,
        chat_template=None,
        add_generation_prompt=False,
        continue_final_message=False,
        tokenize=True,
        padding=False,
        truncation=False,
        max_length=None,
        return_tensors=None,
        return_dict=True,
        return_assistant_tokens_mask=False,
        tokenizer_kwargs=None,
        **kwargs,
    ):
        if (
            chat_template is not None
            or continue_final_message
            or return_assistant_tokens_mask
        ):
            return jinja_apply_chat_template(
                conversation,
                tools=tools,
                documents=documents,
                chat_template=chat_template,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=continue_final_message,
                tokenize=tokenize,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                return_tensors=return_tensors,
                return_dict=return_dict,
                return_assistant_tokens_mask=return_assistant_tokens_mask,
                tokenizer_kwargs=tokenizer_kwargs,
                **kwargs,
            )

        if isinstance(conversation, (list, tuple)) and len(conversation) == 0:
            raise ValueError(
                "Cannot apply chat template to an empty conversation. Provide at "
                "least one message."
            )
        is_batched = isinstance(conversation, (list, tuple)) and (
            isinstance(conversation[0], (list, tuple))
            or hasattr(conversation[0], "messages")
        )
        conversations = conversation if is_batched else [conversation]
        tool_schemas = _tool_schemas(tools)
        if documents is not None:
            for document in documents:
                if not isinstance(document, dict):
                    raise TypeError(
                        "Documents should be a list of dicts with 'title' and "
                        "'text' keys!"
                    )

        # As for Jinja, kwargs override special tokens of the same name.
        template_kwargs = {**tokenizer.special_tokens_map, **kwargs}
        rendered = [
            template.render(
                messages=getattr(chat, "messages", chat),
                tools=tool_schemas,
                documents=documents,
                add_generation_prompt=add_generation_prompt,
                **template_kwargs,
            )
            for chat in conversations
        ]
        rendered_chat = rendered if is_batched else rendered[0]
        if not tokenize:
            return rendered_chat

        out = tokenizer(
            rendered_chat,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            add_special_tokens=False,
            return_tensors=return_tensors,
            **(tokenizer_kwargs or {}),
        )
        return out if return_dict else out["input_ids"]

    tokenizer._sglang_jinja_apply_chat_template = jinja_apply_chat_template
    tokenizer.apply_chat_template = apply_chat_template
