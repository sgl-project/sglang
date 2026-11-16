"""ERNIE 5.5 Flash chat encoding adapter.

The checkpoint renders its chat template in Python (``ErnieChatTokenizer``),
not Jinja, so prompts are built by the tokenizer's own
``apply_chat_template`` rather than the default Jinja path. Reasoning and
tool-call output use the standard ``ernie55`` text parsers, so this adapter
owns prompt encoding only (``output`` is ``None``).

Text, tool calling and the thinking toggle are supported; the template's
multimodal branches are left for a later upgrade (``collects_openai_media``
is ``False``).
"""

from __future__ import annotations

import inspect
import logging
import sys
from typing import TYPE_CHECKING, Any, Optional

from sglang.srt.entrypoints.openai.chat_encoding_adapters import OPENAI_CHAT_WIRE

if TYPE_CHECKING:
    from sglang.srt.entrypoints.openai.chat_encoding_adapters import ChatOutputHooks
    from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

logger = logging.getLogger(__name__)

# Keys the adapter passes to apply_chat_template itself; a client copy in
# chat_template_kwargs would collide with the explicit argument.
_RESERVED_TEMPLATE_KWARGS = frozenset(
    {
        "messages",
        "tools",
        "add_generation_prompt",
        "continue_final_message",
        "tokenize",
        "return_dict",
    }
)
_accepted_render_kwargs_cache: dict[type, Optional[frozenset[str]]] = {}
_warned_template_kwargs: set[str] = set()


def _accepted_render_kwargs(tokenizer: Any) -> Optional[frozenset[str]]:
    """Keyword names ErnieChatTokenizer's renderer accepts; None means "any".

    The tokenizer's ``render_chat`` forwards ``**kwargs`` to the module-level
    ``render_chat`` in its own module, whose signature is explicit; introspect
    that to learn which keys the strict renderer will not reject.
    """
    cls = type(tokenizer)
    if cls not in _accepted_render_kwargs_cache:
        accepted: Optional[frozenset[str]] = None
        for fn in (
            getattr(sys.modules.get(cls.__module__), "render_chat", None),
            getattr(tokenizer, "render_chat", None),
        ):
            if not callable(fn):
                continue
            try:
                params = list(inspect.signature(fn).parameters.values())
            except (TypeError, ValueError):
                continue
            if any(p.kind is p.VAR_KEYWORD for p in params):
                continue  # forwards everything; prefer the explicit signature
            accepted = frozenset(
                p.name
                for p in params
                if p.kind in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD)
            )
            break
        _accepted_render_kwargs_cache[cls] = accepted
    return _accepted_render_kwargs_cache[cls]


def _supported_template_kwargs(
    tokenizer: Any, template_kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Drop chat_template_kwargs the strict Python renderer would reject.

    SGLang forwards ``request.chat_template_kwargs`` verbatim. A Jinja template
    silently ignores names it never reads (e.g. a frontend's DeepSeek-style
    ``thinking``), but ErnieChatTokenizer's renderer raises on an unknown
    keyword. Keep Jinja's leniency at the serving layer; the tokenizer stays
    strict. The thinking intent is applied separately via ``enable_thinking``.
    """
    out = dict(template_kwargs)
    accepted = _accepted_render_kwargs(tokenizer)
    drop = [
        k
        for k in out
        if k in _RESERVED_TEMPLATE_KWARGS
        or (accepted is not None and k not in accepted)
    ]
    for key in drop:
        out.pop(key)
        if key not in _warned_template_kwargs:
            _warned_template_kwargs.add(key)
            logger.warning(
                "ignoring chat_template_kwargs[%r]: not accepted by %s render path",
                key,
                type(tokenizer).__name__,
            )
    return out


class Ernie55ChatEncodingAdapter:
    # Prompt ids come straight from the tokenizer; no media to collect, and
    # history tool-call arguments stay parsed JSON (the template re-serializes).
    collects_openai_media = False
    history_tool_call_arguments = "strict"
    renders_response_format = False
    uncounted_prompt_tokens = 0
    wire = OPENAI_CHAT_WIRE

    def __init__(
        self,
        *,
        tokenizer: Any,
        reasoning_parser: Optional[str],
        tool_call_parser: Optional[str],
        chat_hooks: ChatOutputHooks,
        default_chat_template_kwargs: Optional[dict[str, Any]] = None,
    ):
        if tokenizer is None:
            raise ValueError(
                "ERNIE 5.5 chat encoding needs the checkpoint's Python tokenizer "
                "(ErnieChatTokenizer, load it with --trust-remote-code); serving "
                "chat with --skip-tokenizer-init is unsupported"
            )
        self._tokenizer = tokenizer
        self._default_chat_template_kwargs = dict(default_chat_template_kwargs or {})
        self._hooks = chat_hooks
        # ErnieChatTokenizer builds its encode tables on first use (0.8 s measured
        # on the 1110 checkpoint); pay that at startup, not on the first request.
        tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
        )

    @property
    def output(self) -> None:
        # Reasoning/tool output is parsed by the standard ernie55 text parsers.
        return None

    def validate_request(self, request: ChatCompletionRequest) -> Optional[str]:
        return None

    def encode_messages(
        self,
        *,
        messages: list[dict[str, Any]],
        request: ChatCompletionRequest,
        tools: Optional[list[dict]],
        thinking: bool,
        image_data: list,
    ) -> list[int]:
        """Render with the checkpoint's own Python template. ``thinking`` and
        ``image_data`` are unused: thinking is left to the enable_thinking toggle
        (see below), and multimodal is out of scope for this adapter."""
        kwargs: dict[str, Any] = dict(self._default_chat_template_kwargs)
        kwargs.update(request.chat_template_kwargs or {})
        # Keep only keys the strict Python renderer accepts (the adapter sets
        # tokenize/return_dict/add_generation_prompt/tools itself); this drops a
        # frontend's DeepSeek-style ``thinking`` the renderer would reject.
        kwargs = _supported_template_kwargs(self._tokenizer, kwargs)
        # Do not inject enable_thinking from the handler's `thinking`: that is
        # resolved from the `thinking` key (default off), while the reasoning
        # parser's force_reasoning reads `enable_thinking` (default on). Injecting
        # it here renders a thinking-off prompt under a force-reasoning parser, so
        # the answer lands in reasoning_content. Let both default to the
        # enable_thinking toggle instead. A reasoning_effort only when the client set one.
        if request.reasoning_effort is not None:
            kwargs.setdefault("reasoning_effort", request.reasoning_effort)
        continue_final = bool(
            request.continue_final_message
            and messages
            and messages[-1].get("role") == "assistant"
        )
        if not continue_final:
            # A trailing assistant turn becomes a user turn, as on the Jinja path.
            messages, _ = self._hooks.split_assistant_prefix(messages, request)
        # The renderer continues the final turn itself, after that turn's think
        # block; the generic append_assistant_prefix hook would put the prefix
        # inside the opened think block and prepend <|ERNIE_MASK|> to it.
        return self._tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=not continue_final,
            continue_final_message=continue_final,
            tokenize=True,
            return_dict=False,
            **kwargs,
        )
