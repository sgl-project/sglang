"""Tokens for a FIXED system prompt: inject once + cache the system prefix.

The system prompt is rendered by the chat template once and its token ids
cached. Each chat request then becomes ``cached_system_ids + ids(rest)`` so the
shared system block is neither re-rendered nor re-tokenized per request.

Correctness is guaranteed by a one-time probe: we compare
``render(system) + render(rest)`` against a single ``render(full)`` for the same
messages. Jinja chat templates are per-message, so the split normally reproduces
the full token stream exactly; templates that emit context outside the
per-message loop (a leading BOS token) are handled by stripping the duplicated
leading BOS from the ``rest`` render. If the probe ever disagrees, prefix
caching is disabled and the caller falls back to a single full render, so output
remains byte-identical.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from sglang.srt.system_prompt.system_prompt import aligned_prefix_len

logger = logging.getLogger(__name__)


class SystemPrefix:
    """Cached token prefix for a fixed system prompt."""

    def __init__(self, tokenizer, system_text: str) -> None:
        self.tokenizer = tokenizer
        self.system_text = system_text
        self._system_msg = {"role": "system", "content": system_text}
        self._prefix_ids: Optional[List[int]] = None
        self._encode_kwargs: dict = {}
        self._enabled: bool = False
        self._padded_content: str = system_text
        self._boundary_len: int = 0
        # When a chat template auto-injects a default system message into any
        # render whose list does not start with a `system` role (e.g. Qwen2.5-VL
        # -> "You are a helpful assistant."), we cache that auto-block and strip
        # it from the per-request `render([rest])` before appending the cached
        # real system prefix. None when not needed.
        self._auto_prefix: Optional[List[int]] = None

    # -- public ---------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def prefix_ids(self) -> Optional[List[int]]:
        return self._prefix_ids

    @property
    def prefix_len(self) -> int:
        return len(self._prefix_ids) if self._prefix_ids is not None else 0

    @property
    def boundary_len(self) -> int:
        """System-prefix length in tokens, aligned to the cache chunk boundary.

        This is the position the scheduler truncates radix KV / mamba state at,
        so it is valid regardless of whether the request is multimodal (the
        system block is text-only and sits before any media). 0 when disabled.
        """
        return self._boundary_len

    @property
    def padded_content(self) -> str:
        """The system text with padding appended so its render is chunk-aligned."""
        return self._padded_content

    def inject_message(self) -> Dict[str, str]:
        """The message to inject so the real prompt contains the aligned prefix."""
        return {"role": "system", "content": self._padded_content}

    def aligned_prefix_len(self) -> int:
        """The cached prefix length (padded to a mamba chunk boundary)."""
        return self.prefix_len

    def prepare(self, chunk: int, encode_kwargs: dict) -> None:
        """Run the token-identity probe once and cache the system prefix.

        Two paths, selected by template capability:
        * Explicit: the template can render a bare system message (handles
          top-BOS templates correctly).
        * Measured: the template rejects a system-only render (e.g. Qwen 3.5/3.6)
          -> system-block length = ``len([system,user]) - len([user])``.
        Either way the injected system text is padded to a cache-chunk boundary
        and token-identity ``prefix + ids(rest) == full`` is verified. On any
        failure caching stays disabled (callers fall back to the full render).
        """
        self._encode_kwargs = encode_kwargs
        self._enabled = False
        if _prepare_explicit(self, chunk, encode_kwargs):
            return
        if _prepare_measured(self, chunk, encode_kwargs):
            return
        if _prepare_auto_strip(self, chunk, encode_kwargs):
            return
        self._boundary_len = 0
        logger.info("SystemPrompt prefix caching disabled by identity probe")

    # -- request-time ---------------------------------------------------
    def split_request(
        self, messages_after_system: List[Dict[str, str]], encode_kwargs: dict
    ) -> Optional[List[int]]:
        """Return ``prefix + ids(rest)`` for a chat, or None to use the full path.

        ``messages_after_system`` is the message list with the system message as
        the first element. Returns None when prefix caching is disabled or only
        a system message is present (nothing to concatenate).
        """
        if not self._enabled or len(messages_after_system) < 2:
            return None
        self._encode_kwargs = encode_kwargs
        rest = self._render(
            messages_after_system[1:], add_generation_prompt=True, encode_kwargs=encode_kwargs
        )
        if rest is not None and self._auto_prefix:
            # Strip the template's auto-injected default system block from the
            # head of the user render so `cached_sys + rest` matches a full render.
            ap = len(self._auto_prefix)
            if len(rest) >= ap and rest[:ap] == self._auto_prefix:
                rest = rest[ap:]
        rest = _strip_leading_bos(rest, getattr(self.tokenizer, "bos_token_id", None))
        return list(self._prefix_ids) + rest

    # -- internal -------------------------------------------------------
    def _render(self, msgs, add_generation_prompt: bool, encode_kwargs: dict) -> Optional[List[int]]:
        try:
            rendered = self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=add_generation_prompt
            )
            return self.tokenizer.encode(rendered, **encode_kwargs)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("SystemPrompt render failed: %s", e)
            return None

    def _strip_leading_bos(self, token_ids: List[int]) -> List[int]:
        return _strip_leading_bos(token_ids, getattr(self.tokenizer, "bos_token_id", None))


def _strip_leading_bos(token_ids: List[int], bos_id: Optional[int]) -> List[int]:
    if bos_id is None or not token_ids or token_ids[0] != bos_id:
        return token_ids
    return token_ids[1:]


def _one_token_seed(tokenizer, encode_kwargs: dict) -> str:
    """Return a literal string that tokenizes to exactly one token (with the
    given encode kwargs). Used to pad the injected system text to a cache-chunk
    boundary without adding template-controlled characters.
    """
    candidates = [
        getattr(tokenizer, "pad_token", None),
        getattr(tokenizer, "eos_token", None),
        getattr(tokenizer, "sep_token", None),
        "!",
    ]
    for cand in candidates:
        if not cand:
            continue
        try:
            if len(tokenizer.encode(cand, **encode_kwargs)) == 1:
                return cand
        except Exception:
            continue
    return " "


def _prepare_explicit(seam, chunk: int, encode_kwargs: dict) -> bool:
    """Probe for templates that can render a bare system message."""
    sample_user = {"role": "user", "content": "hi"}
    try:
        sys_ids = seam._render(
            [seam._system_msg], add_generation_prompt=False, encode_kwargs=encode_kwargs
        )
        if sys_ids is None:
            return False
        user_ids = seam._render(
            [sample_user], add_generation_prompt=True, encode_kwargs=encode_kwargs
        )
        full_ids = seam._render(
            [seam._system_msg, sample_user],
            add_generation_prompt=True,
            encode_kwargs=encode_kwargs,
        )
        if full_ids is None or user_ids is None:
            return False
        rest = _strip_leading_bos(user_ids, getattr(seam.tokenizer, "bos_token_id", None))
        if sys_ids + rest != full_ids:
            return False
        pad = aligned_prefix_len(len(sys_ids), chunk) - len(sys_ids)
        if pad > 0:
            ph = _one_token_seed(seam.tokenizer, encode_kwargs)
            seam._padded_content = seam.system_text + ph * pad
            seam._system_msg = {"role": "system", "content": seam._padded_content}
            sys_ids = seam._render(
                [seam._system_msg],
                add_generation_prompt=False,
                encode_kwargs=encode_kwargs,
            )
            if sys_ids is None:
                return False
        seam._prefix_ids = list(sys_ids)
        seam._boundary_len = len(sys_ids)
        seam._enabled = True
    except Exception:
        return False
    logger.info(
        "SystemPrompt prefix caching enabled (explicit): prefix=%d tokens (chunk=%d)",
        seam._boundary_len,
        chunk,
    )
    return True


def _prepare_measured(seam, chunk: int, encode_kwargs: dict) -> bool:
    """Fallback for templates that require a user turn to render (e.g. Qwen3.x)."""
    sample_user = {"role": "user", "content": "hi"}
    try:
        user_ids = seam._render(
            [sample_user], add_generation_prompt=True, encode_kwargs=encode_kwargs
        )
        full_ids = seam._render(
            [seam._system_msg, sample_user],
            add_generation_prompt=True,
            encode_kwargs=encode_kwargs,
        )
        if full_ids is None or user_ids is None:
            return False
        sys_len = len(full_ids) - len(user_ids)
        pad = aligned_prefix_len(max(sys_len, 0), chunk) - max(sys_len, 0)
        if pad > 0 and sys_len > 0:
            ph = _one_token_seed(seam.tokenizer, encode_kwargs)
            seam._padded_content = seam.system_text + ph * pad
            seam._system_msg = {"role": "system", "content": seam._padded_content}
            full_ids = seam._render(
                [seam._system_msg, sample_user],
                add_generation_prompt=True,
                encode_kwargs=encode_kwargs,
            )
            if full_ids is None:
                return False
            sys_len = len(full_ids) - len(user_ids)
        if sys_len <= 0:
            return False
        boundary = sys_len
        prefix = list(full_ids[:boundary])
        rest = _strip_leading_bos(user_ids, getattr(seam.tokenizer, "bos_token_id", None))
        if prefix + rest != full_ids:
            return False
        seam._prefix_ids = prefix
        seam._boundary_len = boundary
        seam._enabled = True
    except Exception:
        return False
    logger.info(
        "SystemPrompt prefix caching enabled (measured): prefix=%d tokens (chunk=%d)",
        seam._boundary_len,
        chunk,
    )
    return True


def _prepare_auto_strip(seam, chunk: int, encode_kwargs: dict) -> bool:
    """Last resort for templates that auto-insert a default system message.

    When a template injects a default `system` block into any render whose
    message list does not start with a `system` role (e.g. Qwen2.5-VL inserts
    "You are a helpful assistant."), rendering the user part alone differs from
    the full conversation by exactly that fixed leading block. We detect it via
    the longest common SUFFIX of the user-only render and the full render (the
    shared user block), recovering both the real system prefix (the full render
    minus that suffix) and the auto-block (the user render minus that suffix).
    ``split_request`` then strips the auto-block so that
    ``cached_real_system + rest == full``.
    """
    sample_user = {"role": "user", "content": "hi"}
    try:
        user_ids = seam._render(
            [sample_user], add_generation_prompt=True, encode_kwargs=encode_kwargs
        )
        full_ids = seam._render(
            [seam._system_msg, sample_user],
            add_generation_prompt=True,
            encode_kwargs=encode_kwargs,
        )
        if user_ids is None or full_ids is None:
            return False
        n = min(len(full_ids), len(user_ids))
        i = 0
        while i < n and full_ids[-1 - i] == user_ids[-1 - i]:
            i += 1
        if i == 0:
            return False
        shared_tail = i
        sys_prefix = list(full_ids[: len(full_ids) - shared_tail])
        auto_prefix = list(user_ids[: len(user_ids) - shared_tail])
        if not sys_prefix or not auto_prefix:
            return False

        # Stability check: the recovered system prefix must not depend on the
        # specific user content, otherwise user tokens leaked into it.
        sample2 = {"role": "user", "content": "hello world!"}
        full2 = seam._render(
            [seam._system_msg, sample2],
            add_generation_prompt=True,
            encode_kwargs=encode_kwargs,
        )
        user2 = seam._render(
            [sample2], add_generation_prompt=True, encode_kwargs=encode_kwargs
        )
        if full2 is not None and user2 is not None:
            k = min(len(full2), len(user2))
            j = 0
            while j < k and full2[-1 - j] == user2[-1 - j]:
                j += 1
            sys2 = list(full2[: len(full2) - j])
            if sys2 != sys_prefix:
                return False

        seam._auto_prefix = auto_prefix
        seam._prefix_ids = sys_prefix
        seam._boundary_len = len(sys_prefix)
        seam._enabled = True
    except Exception:
        return False
    logger.info(
        "SystemPrompt prefix caching enabled (auto-strip): prefix=%d tokens (chunk=%d)",
        seam._boundary_len,
        chunk,
    )
    return True