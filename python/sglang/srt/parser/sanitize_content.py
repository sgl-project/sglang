"""Sanitize user content in chat templates for safer tokenization.

This module provides a ``sanitize_content`` Jinja filter that chat template
authors can use to explicitly mark user-controlled content boundaries.  When
the filter is used, sglang renders user content as a SHA256-based placeholder,
then tokenizes template fragments normally and user content with
``split_special_tokens=True``.

This prevents user-provided strings that look like special tokens from being
tokenized as model special tokens, while preserving existing behavior for
templates that do not opt in.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from datetime import datetime
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, Final, List, Optional, cast

import jinja2
import jinja2.ext
import jinja2.sandbox

if TYPE_CHECKING:
    pass

logger = __import__("logging").getLogger(__name__)

_SANITIZE_CONTENT_FILTER_NAME: Final[str] = "sanitize_content"
_SANITIZED_CONTENT_PLACEHOLDER_PREFIX: Final[str] = "<sglang_user_input_"
_SANITIZED_CONTENT_PLACEHOLDER_RE: Final[re.Pattern[str]] = re.compile(
    r"<sglang_user_input_[0-9a-f]{64}>"
)
_CONTINUE_FINAL_MESSAGE_TAG: Final[str] = "CONTINUE_FINAL_MESSAGE_TAG "


def _make_sanitized_content_placeholder(content: str) -> str:
    """Create a SHA256-based placeholder for user content."""
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return f"{_SANITIZED_CONTENT_PLACEHOLDER_PREFIX}{digest}>"


def _tokenize_sanitized_content(tokenizer: Any, content: str) -> List[int]:
    """Tokenize user content with split_special_tokens=True to prevent
    special token injection."""
    tokens = tokenizer.tokenize(content, split_special_tokens=True)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    return cast(List[int], token_ids)


def _split_sanitized_rendered_chat(rendered_chat: str):
    """Split rendered chat into (is_content, text) tuples."""
    last_end = 0
    for match in _SANITIZED_CONTENT_PLACEHOLDER_RE.finditer(rendered_chat):
        if match.start() > last_end:
            yield False, rendered_chat[last_end : match.start()]
        yield True, match.group(0)
        last_end = match.end()
    if last_end < len(rendered_chat):
        yield False, rendered_chat[last_end:]


def _restore_sanitized_placeholders(
    rendered_chat: str,
    content_map: Dict[str, str],
) -> str:
    """Restore sanitized placeholders to original content."""
    return _SANITIZED_CONTENT_PLACEHOLDER_RE.sub(
        lambda match: content_map.get(match.group(0), match.group(0)),
        rendered_chat,
    )


def _compile_sanitized_jinja_template(
    chat_template: str,
    sanitize_content_fn,
) -> jinja2.Template:
    """Compile a Jinja template with the sanitize_content filter available."""

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(
            x,
            ensure_ascii=ensure_ascii,
            indent=indent,
            separators=separators,
            sort_keys=sort_keys,
        )

    def strftime_now(format):
        return datetime.now().strftime(format)

    jinja_env = jinja2.sandbox.ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
        extensions=[jinja2.ext.loopcontrols],
    )
    jinja_env.filters["tojson"] = tojson
    jinja_env.filters[_SANITIZE_CONTENT_FILTER_NAME] = sanitize_content_fn
    jinja_env.globals["raise_exception"] = raise_exception
    jinja_env.globals["strftime_now"] = strftime_now
    return jinja_env.from_string(chat_template)


@lru_cache(maxsize=32)
def _should_use_sanitized_chat_template(chat_template: str) -> bool:
    """Check if a chat template uses the sanitize_content filter."""
    try:
        env = jinja2.sandbox.ImmutableSandboxedEnvironment(
            trim_blocks=True,
            lstrip_blocks=True,
            extensions=[jinja2.ext.loopcontrols],
        )
        env.filters[_SANITIZE_CONTENT_FILTER_NAME] = lambda value: value
        parsed_content = env.parse(chat_template)
        return any(
            node.name == _SANITIZE_CONTENT_FILTER_NAME
            for node in parsed_content.find_all(jinja2.nodes.Filter)
        )
    except Exception as e:
        logger.debug(f"Error when parsing Jinja template for sanitize_content: {e}")
        return False


def _render_jinja_template_with_sanitized_content(
    tokenizer: Any,
    conversation: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
    chat_template: str,
    *,
    add_generation_prompt: bool = False,
    continue_final_message: bool = False,
    documents: Optional[List[Dict[str, str]]] = None,
    **kwargs,
) -> tuple:
    """Render a chat template with sanitized user content.

    Returns (rendered_chat, content_map).
    """
    content_map: Dict[str, str] = {}
    continue_placeholder: Optional[str] = None

    def sanitize_content(value: Any) -> str:
        nonlocal continue_placeholder

        content = value if isinstance(value, str) else str(value)
        if content.endswith(_CONTINUE_FINAL_MESSAGE_TAG):
            content = content[: -len(_CONTINUE_FINAL_MESSAGE_TAG)]
            placeholder_content = content + _CONTINUE_FINAL_MESSAGE_TAG
            placeholder = _make_sanitized_content_placeholder(placeholder_content)
            continue_placeholder = placeholder
        else:
            placeholder = _make_sanitized_content_placeholder(content)

        content_map[placeholder] = content
        return placeholder

    if continue_final_message:
        if add_generation_prompt:
            raise ValueError(
                "continue_final_message and add_generation_prompt are not "
                "compatible. Use continue_final_message when you want the "
                "model to continue the final message, and add_generation_prompt "
                "when you want to add a header that will prompt it to start a "
                "new assistant message instead."
            )

        conversation = copy.deepcopy(conversation)
        final_message = conversation[-1].get("content")
        if final_message is None:
            raise ValueError(
                "continue_final_message is set but the final message has no "
                "content to continue!"
            )
        if isinstance(final_message, (list, tuple)):
            for content_block in reversed(final_message):
                if "text" in content_block:
                    content_block["text"] = (
                        content_block["text"] + _CONTINUE_FINAL_MESSAGE_TAG
                    )
                    break
            else:
                raise ValueError(
                    "continue_final_message is set but we could not find any "
                    "text to continue in the final message!"
                )
        else:
            conversation[-1]["content"] = final_message + _CONTINUE_FINAL_MESSAGE_TAG

    template = _compile_sanitized_jinja_template(chat_template, sanitize_content)
    template_kwargs = {**getattr(tokenizer, "special_tokens_map", {}), **kwargs}
    rendered_chat = template.render(
        messages=conversation,
        tools=tools,
        documents=documents,
        add_generation_prompt=add_generation_prompt,
        **template_kwargs,
    )

    if continue_final_message:
        if continue_placeholder is None or continue_placeholder not in rendered_chat:
            raise ValueError(
                "continue_final_message is set but the final message does not "
                "appear in the chat after applying the chat template! This can "
                "happen if the chat template deletes portions of the final "
                "message. Please verify the chat template and final message in "
                "your chat to ensure they are compatible."
            )
        tag_loc = rendered_chat.rindex(continue_placeholder)
        rendered_chat = rendered_chat[: tag_loc + len(continue_placeholder)]

    return rendered_chat, content_map


def safe_apply_chat_template(
    tokenizer: Any,
    conversation: List[Dict[str, Any]],
    *,
    chat_template: Optional[str] = None,
    tokenize: bool = True,
    add_generation_prompt: bool = False,
    tools: Optional[List[Dict[str, Any]]] = None,
    continue_final_message: bool = False,
    documents: Optional[List[Dict[str, str]]] = None,
    return_assistant_tokens_mask: bool = False,
    return_dict: bool = False,
    return_tensors: Optional[str] = None,
    padding: bool = False,
    truncation: bool = False,
    max_length: Optional[int] = None,
    tokenizer_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Any:
    """Apply a chat template, with sanitize_content support if the template
    opts in via the ``sanitize_content`` Jinja filter.

    When the template does NOT use ``sanitize_content``, this falls through to
    the tokenizer's native ``apply_chat_template``, preserving all existing
    behaviour.

    When the template DOES use ``sanitize_content``:

    * User content is replaced with SHA256-based placeholders during rendering.
    * Template fragments are tokenized normally.
    * User content is tokenized with ``split_special_tokens=True``.
    * For ``tokenize=False``, placeholders are restored to the original text.
    """
    # Resolve the chat template
    if chat_template is None:
        chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template is None:
        raise ValueError(
            "Cannot apply chat template: no chat_template provided and "
            "tokenizer has no chat_template."
        )

    if isinstance(chat_template, dict):
        # Handle dict of named templates – use the first one
        chat_template = next(iter(chat_template.values()))

    if not _should_use_sanitized_chat_template(chat_template):
        # Fall through to the tokenizer's native apply_chat_template
        return tokenizer.apply_chat_template(
            conversation=conversation,
            chat_template=chat_template,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            continue_final_message=continue_final_message,
            documents=documents,
            return_assistant_tokens_mask=return_assistant_tokens_mask,
            return_dict=return_dict,
            return_tensors=return_tensors,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            tokenizer_kwargs=tokenizer_kwargs,
            **kwargs,
        )

    # --- Sanitized content path ---
    if return_assistant_tokens_mask:
        raise ValueError(
            "return_assistant_tokens_mask is not supported with "
            "sanitize_content chat templates."
        )
    if return_dict:
        raise ValueError(
            "return_dict=True is not supported with sanitize_content chat templates."
        )
    if return_tensors is not None:
        raise ValueError(
            "return_tensors is not supported with sanitize_content chat templates."
        )
    if padding:
        raise ValueError(
            "padding is not supported with sanitize_content chat templates."
        )
    if tokenizer_kwargs:
        raise ValueError(
            "tokenizer_kwargs is not supported with sanitize_content chat templates."
        )

    rendered_chat, content_map = _render_jinja_template_with_sanitized_content(
        tokenizer,
        conversation,
        tools,
        chat_template,
        add_generation_prompt=add_generation_prompt,
        continue_final_message=continue_final_message,
        documents=documents,
        **kwargs,
    )

    if not tokenize:
        return _restore_sanitized_placeholders(rendered_chat, content_map)

    token_ids: List[int] = []
    for is_content, part in _split_sanitized_rendered_chat(rendered_chat):
        if is_content:
            content = content_map.get(part)
            if content is None:
                token_ids.extend(tokenizer.encode(part, add_special_tokens=False))
            else:
                token_ids.extend(_tokenize_sanitized_content(tokenizer, content))
        else:
            token_ids.extend(tokenizer.encode(part, add_special_tokens=False))

    if truncation and max_length is not None and len(token_ids) > max_length:
        if getattr(tokenizer, "truncation_side", "right") == "left":
            token_ids = token_ids[-max_length:]
        else:
            token_ids = token_ids[:max_length]

    return token_ids
