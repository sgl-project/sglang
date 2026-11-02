import json
from abc import ABC, abstractmethod
from array import array
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

import dill
import msgspec
import orjson
import torch

from sglang.srt.models.unlimited_ocr_v2_support.window_params import KEY_LOOKAHEAD

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

UOCR_PAGE_GUARD_PROCESSOR = "sglang:uocr-page-guard"


@lru_cache(maxsize=None)
def _cache_from_str(json_str: str):
    """Deserialize a json string to a Callable object.
    This function is cached to avoid redundant deserialization.
    """
    if json_str == UOCR_PAGE_GUARD_PROCESSOR:
        return DeepseekOCRNoRepeatNGramLogitProcessor
    data = orjson.loads(json_str)
    return dill.loads(bytes.fromhex(data["callable"]))


class CustomLogitProcessor(ABC):
    """Abstract base class for callable functions."""

    @abstractmethod
    def __call__(
        self,
        logits: torch.Tensor,
        custom_param_list: Optional[List[Dict[str, Any]]] = None,
    ) -> torch.Tensor:
        """Define the callable behavior.

        The returned tensor must have the same shape as `logits`: the caller
        writes it back row for row and does not broadcast a reduced result.
        """
        raise NotImplementedError

    @classmethod
    def to_str(cls) -> str:
        """Serialize the callable function to a JSON-compatible string."""
        return json.dumps({"callable": dill.dumps(cls).hex()})

    @classmethod
    def from_str(cls, json_str: str):
        """Deserialize a callable function from a JSON string."""
        return _cache_from_str(json_str)()


class DisallowedTokensLogitsProcessor(CustomLogitProcessor):
    def __call__(
        self,
        logits: torch.Tensor,
        custom_param_list: Optional[List[Dict[str, Any]]] = None,
    ) -> torch.Tensor:
        disallowed_token_ids = custom_param_list[0]["token_ids"]
        assert all(disallowed_token_ids == c["token_ids"] for c in custom_param_list), (
            f"{custom_param_list=}"
        )
        logits[..., disallowed_token_ids] = -float("inf")
        return logits


def supports_sampling_mask(serialized_processor: str) -> bool:
    """Hard exclusion preserves the relative logits needed for mask-based replay."""
    try:
        return isinstance(
            CustomLogitProcessor.from_str(serialized_processor),
            DisallowedTokensLogitsProcessor,
        )
    except Exception:
        return False


def _open_thinking_start(ids: list[int], start_id: int, end_id: int) -> int:
    """Return the index of the start token of the currently open thinking block, or -1."""
    for idx in reversed(range(len(ids))):
        if ids[idx] == start_id:
            return idx
        if ids[idx] == end_id:
            return -1
    return -1


class ThinkingBudgetLogitProcessor(CustomLogitProcessor):
    """A logit processor that controls the length of thinking."""

    THINKING_START_TOKEN_ID: int
    THINKING_END_TOKEN_ID: int
    NEW_LINE_TOKEN_ID: int

    def __call__(self, logits, custom_param_list: list[dict[str, Any]]):
        if custom_param_list is None or not custom_param_list:
            return logits
        for i, param_dict in enumerate(custom_param_list):
            if param_dict is None:
                continue

            thinking_budget: int | None = param_dict.get("thinking_budget")

            # Skip if thinking_budget is unset, or not an integer, or negative
            if (
                thinking_budget is None
                or not isinstance(thinking_budget, int)
                or thinking_budget < 0
            ):
                continue
            req: Req = param_dict.get("__req__")
            cur_ids: list[int] = [*req.origin_input_ids, *req.output_ids]

            # Check if out of thinking stage
            start_index = _open_thinking_start(
                cur_ids, self.THINKING_START_TOKEN_ID, self.THINKING_END_TOKEN_ID
            )
            if start_index < 0:
                continue

            # Count the number of tokens after the thinking start token
            num_tokens_after_start = len(cur_ids) - start_index - 1

            if num_tokens_after_start < thinking_budget:
                continue

            # Ensure new line token before thinking end token
            if not req.output_ids or req.output_ids[-1] != self.NEW_LINE_TOKEN_ID:
                logits[i, :] = -float("inf")
                logits[i, self.NEW_LINE_TOKEN_ID] = 0.0
                continue

            # Assign highest probability to the thinking end token
            logits[i, :] = -float("inf")
            logits[i, self.THINKING_END_TOKEN_ID] = 0.0

        return logits


class Glm4MoeThinkingBudgetLogitProcessor(ThinkingBudgetLogitProcessor):
    """A logit processor that controls the length of thinking for GLM-4.5 / GLM-4.6 / GLM-4.5V / GLM-4.6V models."""

    THINKING_START_TOKEN_ID: int = 151350
    THINKING_END_TOKEN_ID: int = 151351
    NEW_LINE_TOKEN_ID: int = 198


class Qwen3ThinkingBudgetLogitProcessor(ThinkingBudgetLogitProcessor):
    """A logit processor that controls the length of thinking for Qwen3 models."""

    THINKING_START_TOKEN_ID: int = 151667
    THINKING_END_TOKEN_ID: int = 151668
    NEW_LINE_TOKEN_ID: int = 198


class DeepSeekR1ThinkingBudgetLogitProcessor(ThinkingBudgetLogitProcessor):
    """A logit processor that controls the length of thinking for DeepSeek-R1 models."""

    THINKING_START_TOKEN_ID: int = 128798
    THINKING_END_TOKEN_ID: int = 128799
    NEW_LINE_TOKEN_ID: int = 201


class InklingThinkingBudgetLogitProcessor(ThinkingBudgetLogitProcessor):
    """A logit processor that controls the length of thinking for Inkling models."""

    THINKING_START_TOKEN_ID: int = 200008
    THINKING_END_TOKEN_ID: int = 200010
    NEW_LINE_TOKEN_ID: int = 198


def _tail_has_repeat(
    token_ids, *, min_pattern_size: int, max_pattern_size: int, min_count: int
) -> bool:
    """Detect a repeated tail pattern within the configured size range."""
    n_tok = len(token_ids)
    lo = max(1, min_pattern_size)
    for pattern_len in range(lo, max_pattern_size + 1):
        # Increasing pattern_len only needs more tokens; once it doesn't fit,
        # nothing longer will either.
        if pattern_len * min_count > n_tok:
            break
        matched = True
        for n in range(1, pattern_len + 1):
            target = token_ids[-n]
            for m in range(1, min_count):
                if token_ids[-(pattern_len * m + n)] != target:
                    matched = False
                    break
            if not matched:
                break
        if matched:
            return True
    return False


class PageGuardState(msgspec.Struct):
    """Running page count of one request's page guard; mutable."""

    # ``<PAGE>`` markers the output may hold: one per page it transcribes.
    expected: int
    # Output tokens scanned so far, markers among them, index of the last one.
    seen: int = 0
    marks: int = 0
    last: int = -1


def _window_pages(req) -> tuple[int, bool]:
    """Leading prompt pages a document window does not transcribe, and whether
    it ends on the marker past its own pages (Unlimited-OCR v2 reuse)."""
    st = req.uocr_doc
    if st is None:
        return 0, False
    return st.context_pages, st.end_on_marker


def _count_markers(st: PageGuardState, output_ids, page_token_id: int) -> None:
    """Advance the running marker count to the current output."""
    if st.seen > len(output_ids):
        st.seen, st.marks, st.last = 0, 0, -1
    for i in range(st.seen, len(output_ids)):
        if output_ids[i] == page_token_id:
            st.marks += 1
            st.last = i
    st.seen = len(output_ids)


def _page_guard(
    req,
    *,
    page_token_id: int,
    max_page_tokens: int,
    eos_id: int,
    stop_only: bool = False,
    lookahead: int = 0,
) -> Optional[tuple[Optional[int], Optional[int]]]:
    """Hold the output to one page per prompt page: ``(forced, banned)``.

    ``forced`` is the only token allowed this step, ``banned`` one to mask;
    None leaves the step alone. The ``k``-th ``<PAGE>`` marker opens page ``k``:

    - fewer markers than pages: EOS is masked;
    - on the last page: ``<PAGE>`` is masked;
    - a marker past the last page: stop now;
    - a page that reached ``max_page_tokens``: turn the page, or stop on the last.

    A prompt with no marker at all is left alone: banning EOS there would never
    let the request stop.
    """
    context_pages, end_on_marker = _window_pages(req)
    # A window that reads lookahead pages, like an instruction-less reuse
    # window, ends on the marker past its own pages; a model does not stop there.
    end_on_marker = end_on_marker or lookahead > 0
    # A stop-only guard exists for those windows; a window that fell back to
    # its full prompt with no lookahead stops itself and is left alone.
    if stop_only and not end_on_marker:
        return None
    st = req.page_guard_state
    if st is None:
        st = PageGuardState(
            expected=sum(1 for t in req.origin_input_ids if t == page_token_id)
            - context_pages
            - lookahead
        )
        req.page_guard_state = st
    if st.expected <= 0:
        return None
    # Scanning the whole output every step costs O(output) per token; keep a
    # running count and only read the tail that arrived since the last step.
    _count_markers(st, req.output_ids, page_token_id)
    if st.marks > st.expected:
        return eos_id, None
    if stop_only:
        return None
    if 0 < max_page_tokens <= st.seen - (st.last + 1):
        return (page_token_id if st.marks < st.expected else eos_id), None
    if st.marks < st.expected:
        return None, eos_id
    # The marker past the window's own pages is its stop signal (forced to EOS
    # above), so it must not be masked on the last own page.
    if end_on_marker:
        return None
    return None, page_token_id


# custom_params key: search generated tokens only for repeated n-grams, as vLLM
# does; unset keeps the prompt+output domain. Unlimited-OCR v2 serving sets it.
NGRAM_OUTPUT_ONLY_KEY = "ngram_output_only"


def _optional_int(value) -> Optional[int]:
    return int(value) if value is not None else None


class _RepeatStop(msgspec.Struct, frozen=True):
    min_pattern_size: int = 0
    max_pattern_size: int = 0
    min_count: int = 0
    eos_id: Optional[int] = None


def _repeat_stop_params(params: Dict[str, Any]) -> _RepeatStop:
    # One parse: an invalid value disables the repeat stop and the page guard.
    try:
        return _RepeatStop(
            min_pattern_size=int(params.get("rep_min_pattern_size") or 0),
            max_pattern_size=int(params.get("rep_max_pattern_size") or 0),
            min_count=int(params.get("rep_min_count") or 0),
            eos_id=_optional_int(params.get("eos_token_id")),
        )
    except (TypeError, ValueError):
        return _RepeatStop()


def _repeat_stop(req, rep: _RepeatStop, *, vocab_size: int) -> Optional[int]:
    """EOS if the output ends in a periodic cycle (opt-in), else None.

    N-gram blocking can only shift such a cycle's phase, so it needs a hard stop.
    """
    if (
        rep.max_pattern_size > 0
        and rep.min_count >= 2
        and rep.eos_id is not None
        and 0 <= rep.eos_id < vocab_size
        and _tail_has_repeat(
            req.output_ids,
            min_pattern_size=rep.min_pattern_size,
            max_pattern_size=rep.max_pattern_size,
            min_count=rep.min_count,
        )
    ):
        return rep.eos_id
    return None


def _page_guard_action(
    req, params: Dict[str, Any], *, eos_id: Optional[int], vocab_size: int
) -> Optional[tuple[Optional[int], Optional[int]]]:
    """The opt-in page guard's ``(forced, banned)`` for this step, or None.

    The page count needs ``page_token_id``, the length cap also a positive
    ``max_page_tokens``.
    """
    try:
        max_page_tokens = int(params.get("max_page_tokens") or 0)
        page_token_id = _optional_int(params.get("page_token_id"))
    except (TypeError, ValueError):
        return None
    if (
        page_token_id is None
        or eos_id is None
        or not 0 <= page_token_id < vocab_size
        or not 0 <= eos_id < vocab_size
    ):
        return None
    return _page_guard(
        req,
        page_token_id=page_token_id,
        max_page_tokens=max_page_tokens,
        eos_id=eos_id,
        stop_only=bool(params.get("page_stop_only")),
        lookahead=int(params.get(KEY_LOOKAHEAD) or 0),
    )


def _ngram_banned_tokens(req, params: Dict[str, Any], *, vocab_size: int) -> List[int]:
    """Tokens that would repeat an n-gram inside the sliding window."""
    try:
        ngram_size = int(params.get("ngram_size") or 0)
        window_size = int(params.get("window_size") or 0)
    except (TypeError, ValueError):
        return []
    if ngram_size <= 0 or window_size <= 0:
        return []

    # vLLM hands the processor generated tokens only, so the window
    # slides over the output, not the prompt.
    if params.get(NGRAM_OUTPUT_ONLY_KEY):
        sequence = req.output_ids[-window_size:]
    elif len(req.output_ids) >= window_size:
        sequence = req.output_ids[-window_size:]
    else:
        # only the searched tail can contribute a repeated n-gram
        prompt_tokens = window_size - len(req.output_ids)
        sequence = req.origin_input_ids[-prompt_tokens:] + req.output_ids
    if len(sequence) < ngram_size:
        return []

    search_start = max(0, len(sequence) - window_size)
    search_end = len(sequence) - ngram_size + 1
    if search_end <= search_start:
        return []

    if ngram_size > 1:
        current_prefix = sequence[-(ngram_size - 1) :]
    else:
        current_prefix = array("q")

    banned_tokens: Set[int] = set()
    for idx in range(search_start, search_end):
        if ngram_size == 1 or sequence[idx : idx + ngram_size - 1] == current_prefix:
            banned_tokens.add(sequence[idx + ngram_size - 1])

    whitelist_ids = params.get("whitelist_token_ids") or []
    try:
        whitelist = {int(token_id) for token_id in whitelist_ids}
    except (TypeError, ValueError):
        whitelist = set()

    banned_tokens.difference_update(whitelist)
    # Ignore multimodal pad sentinels above the vocabulary range.
    return [t for t in banned_tokens if 0 <= t < vocab_size]


# Adapted from DeepSeek's implementation: https://github.com/deepseek-ai/DeepSeek-OCR/blob/main/DeepSeek-OCR-master/DeepSeek-OCR-vllm/process/ngram_norepeat.py
class DeepseekOCRNoRepeatNGramLogitProcessor(CustomLogitProcessor):
    """Block repeated n-grams, cap per-page length, optionally stop periodic cycles."""

    def __call__(
        self,
        logits: torch.Tensor,
        custom_param_list: Optional[List[Dict[str, Any]]] = None,
    ) -> torch.Tensor:
        if not custom_param_list:
            return logits

        vocab_size = logits.shape[-1]
        # Collected and applied at once: a CUDA write through Python indices
        # copies them from pageable memory and blocks the host on the forward.
        edits = _RowEdits()
        for batch_idx, params in enumerate(custom_param_list):
            if not params:
                continue

            req = params.get("__req__")
            if req is None:
                continue

            rep = _repeat_stop_params(params)
            forced = _repeat_stop(req, rep, vocab_size=vocab_size)
            if forced is None:
                action = _page_guard_action(
                    req, params, eos_id=rep.eos_id, vocab_size=vocab_size
                )
                if action is not None:
                    forced, banned = action
                    if forced is None and banned is not None:
                        edits.ban(row=batch_idx, cols=[banned])
            if forced is not None:
                edits.force(row=batch_idx, col=forced)
                continue

            indices = _ngram_banned_tokens(req, params, vocab_size=vocab_size)
            if indices:
                edits.ban(row=batch_idx, cols=indices)

        edits.apply(logits)
        return logits


class _RowEdits:
    """Per-row logit edits, written with one non-blocking index upload."""

    def __init__(self) -> None:
        self.forced_rows: List[int] = []
        self.forced_cols: List[int] = []
        self.ban_rows: List[int] = []
        self.ban_cols: List[int] = []

    def force(self, *, row: int, col: int) -> None:
        self.forced_rows.append(row)
        self.forced_cols.append(col)

    def ban(self, *, row: int, cols: List[int]) -> None:
        self.ban_rows.extend([row] * len(cols))
        self.ban_cols.extend(cols)

    def apply(self, logits: torch.Tensor) -> None:
        """Force each forced row to its token, then mask the banned cells.

        A forced row has no bans, so the order of the writes does not matter.
        """
        n_forced = len(self.forced_rows)
        if not n_forced and not self.ban_rows:
            return
        index = torch.tensor(
            [self.forced_rows + self.ban_rows, self.forced_cols + self.ban_cols],
            dtype=torch.int64,
        )
        if logits.is_cuda:
            index = index.pin_memory().to(logits.device, non_blocking=True)
        rows, cols = index[0], index[1]
        if n_forced:
            logits.index_fill_(0, rows[:n_forced], -float("inf"))
            logits.index_put_((rows[:n_forced], cols[:n_forced]), logits.new_zeros(()))
        if self.ban_rows:
            logits.index_put_(
                (rows[n_forced:], cols[n_forced:]), logits.new_full((), -float("inf"))
            )
