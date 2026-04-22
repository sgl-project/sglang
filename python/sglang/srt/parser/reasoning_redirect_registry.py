"""Startup-time registry that decides whether the reasoning-EOS redirect logit
processor (``ReasoningEosRedirectLogitProcessor``) can be used for a given
reasoning parser + tokenizer combination.

The processor relies on three things being true:

1. The reasoning parser is in our whitelist (so we know the think-start /
   think-end tokens and the relevant chat-end tokens).
2. The tokenizer encodes the think_end token as a *single* token. The
   processor edits one logit row per step, so multi-token think_end is not
   supported (the request will silently fall back to Path B if enabled).
3. We can resolve at least one chat-end token id to redirect away from.

If any of these checks fail we return ``None`` and the caller leaves the
request alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Set

from sglang.srt.parser.reasoning_parser import ReasoningParser


@dataclass
class ReasoningRedirectConfig:
    think_start_token_id: Optional[int]
    think_end_token_id: int
    redirect_eos_token_ids: List[int]
    force_reasoning: bool


# Per-reasoning-parser whitelist + the additional chat-end tokens that we
# want to treat as "EOS-class" for the redirect trigger. The model's primary
# ``tokenizer.eos_token_id`` is always added on top of this.
_REDIRECT_WHITELIST = {
    "deepseek-r1": {
        "force_reasoning": True,
        "extra_chat_end_tokens": [],
    },
    "deepseek-v3": {
        "force_reasoning": False,
        "extra_chat_end_tokens": ["<|im_end|>", "<|end_of_sentence|>"],
    },
    "qwen3": {
        "force_reasoning": False,
        "extra_chat_end_tokens": ["<|im_end|>"],
    },
    "qwen3-thinking": {
        "force_reasoning": True,
        "extra_chat_end_tokens": ["<|im_end|>"],
    },
    "kimi": {
        "force_reasoning": False,
        "extra_chat_end_tokens": ["<|im_end|>", "[EOS]"],
    },
    "kimi_k2": {
        "force_reasoning": False,
        "extra_chat_end_tokens": ["<|im_end|>", "[EOS]"],
    },
    "glm45": {
        "force_reasoning": False,
        "extra_chat_end_tokens": ["<|user|>", "<|endoftext|>"],
    },
    "minimax": {
        "force_reasoning": True,
        "extra_chat_end_tokens": ["<|im_end|>"],
    },
    "minimax-append-think": {
        "force_reasoning": False,
        "extra_chat_end_tokens": ["<|im_end|>"],
    },
    "mimo": {
        "force_reasoning": False,
        "extra_chat_end_tokens": ["<|im_end|>"],
    },
    "interns1": {
        "force_reasoning": False,
        "extra_chat_end_tokens": ["<|im_end|>"],
    },
    "step3": {
        "force_reasoning": True,
        "extra_chat_end_tokens": ["<|im_end|>"],
    },
    "step3p5": {
        "force_reasoning": True,
        "extra_chat_end_tokens": ["<|im_end|>"],
    },
    "nemotron_3": {
        "force_reasoning": False,
        "extra_chat_end_tokens": ["<|im_end|>"],
    },
}


def _encode_to_single_token(tokenizer: Any, text: Optional[str]) -> Optional[int]:
    if not text or tokenizer is None:
        return None
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
    except Exception:
        return None
    if isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], int):
        return ids[0]
    return None


def _collect_eos_ids(tokenizer: Any) -> Set[int]:
    out: Set[int] = set()
    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, int):
        out.add(eos)
    elif isinstance(eos, (list, tuple, set)):
        for v in eos:
            if isinstance(v, int):
                out.add(v)
    return out


def is_supported_reasoning_parser(name: Optional[str]) -> bool:
    return bool(name) and name.lower() in _REDIRECT_WHITELIST


def build_redirect_config(
    reasoning_parser_name: Optional[str],
    tokenizer: Any,
    extra_chat_end_token_ids: Optional[Iterable[int]] = None,
) -> Optional[ReasoningRedirectConfig]:
    """Return a config the redirect processor can consume, or None.

    Returning None means: do not enable the path-1 redirect processor for this
    reasoning parser / tokenizer (multi-token think_end, unknown parser, etc.).
    The caller can still enable Fallback B independently.
    """
    if reasoning_parser_name is None:
        return None
    name = reasoning_parser_name.lower()
    entry = _REDIRECT_WHITELIST.get(name)
    if entry is None:
        return None

    detector_cls = ReasoningParser.DetectorMap.get(name)
    if detector_cls is None:
        return None

    # Construct detector with default kwargs only — we just need the tag strings.
    try:
        detector = detector_cls(stream_reasoning=True)
    except Exception:
        return None

    think_end_id = _encode_to_single_token(tokenizer, detector.think_end_token)
    if think_end_id is None:
        return None
    think_start_id = _encode_to_single_token(tokenizer, detector.think_start_token)

    eos_ids: Set[int] = _collect_eos_ids(tokenizer)
    for tok_str in entry.get("extra_chat_end_tokens", []):
        tok_id = _encode_to_single_token(tokenizer, tok_str)
        if tok_id is not None:
            eos_ids.add(tok_id)

    if extra_chat_end_token_ids:
        for v in extra_chat_end_token_ids:
            if isinstance(v, int):
                eos_ids.add(v)

    # Never include think_end itself in the EOS-redirect set — that would
    # cause us to mask out the very token we want to keep.
    eos_ids.discard(think_end_id)

    if not eos_ids:
        return None

    return ReasoningRedirectConfig(
        think_start_token_id=think_start_id,
        think_end_token_id=think_end_id,
        redirect_eos_token_ids=sorted(eos_ids),
        force_reasoning=entry["force_reasoning"],
    )
