import json
from abc import ABC, abstractmethod
from array import array
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Set

import dill
import orjson
import torch

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


@lru_cache(maxsize=None)
def _cache_from_str(json_str: str):
    """Deserialize a json string to a Callable object.
    This function is cached to avoid redundant deserialization.
    """
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


def _open_thinking_start(ids: Sequence[int], start_id: int, end_id: int) -> int:
    """Return the index of the start token of the currently open thinking block, or -1."""
    for idx in reversed(range(len(ids))):
        if ids[idx] == start_id:
            return idx
        if ids[idx] == end_id:
            return -1
    return -1


def _open_thinking_tokens_after_start(
    origin_input_ids: Sequence[int],
    output_ids: Sequence[int],
    start_id: int,
    end_id: int,
) -> int:
    """Return the number of tokens after the currently open thinking start token, or -1."""
    for idx in reversed(range(len(output_ids))):
        token_id = output_ids[idx]
        if token_id == start_id:
            return len(output_ids) - idx - 1
        if token_id == end_id:
            return -1
    in_idx = _open_thinking_start(origin_input_ids, start_id, end_id)
    if in_idx < 0:
        return -1
    return len(origin_input_ids) + len(output_ids) - in_idx - 1


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
                or isinstance(thinking_budget, bool)
                or not isinstance(thinking_budget, int)
                or thinking_budget < 0
            ):
                continue
            req: Req = param_dict.get("__req__")
            if req is None:
                continue

            num_tokens_after_start = _open_thinking_tokens_after_start(
                req.origin_input_ids,
                req.output_ids,
                self.THINKING_START_TOKEN_ID,
                self.THINKING_END_TOKEN_ID,
            )
            if num_tokens_after_start < 0 or num_tokens_after_start < thinking_budget:
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


# Adapted from DeepSeek's implementation: https://github.com/deepseek-ai/DeepSeek-OCR/blob/main/DeepSeek-OCR-master/DeepSeek-OCR-vllm/process/ngram_norepeat.py
class DeepseekOCRNoRepeatNGramLogitProcessor(CustomLogitProcessor):
    """Block n-gram repetitions within a sliding window for DeepSeek-OCR outputs."""

    def __call__(
        self,
        logits: torch.Tensor,
        custom_param_list: Optional[List[Dict[str, Any]]] = None,
    ) -> torch.Tensor:
        if not custom_param_list:
            return logits

        for batch_idx, params in enumerate(custom_param_list):
            if not params:
                continue

            req = params.get("__req__")
            if req is None:
                continue

            try:
                ngram_size = int(params.get("ngram_size") or 0)
                window_size = int(params.get("window_size") or 0)
            except (TypeError, ValueError):
                continue

            if ngram_size <= 0 or window_size <= 0:
                continue

            in_len = len(req.origin_input_ids)
            out_len = len(req.output_ids)
            total_len = in_len + out_len
            tail_len = min(total_len, window_size)
            if tail_len < ngram_size:
                continue

            if out_len >= tail_len:
                window = req.output_ids[out_len - tail_len :]
            elif out_len == 0:
                window = req.origin_input_ids[in_len - tail_len :]
            else:
                prefix_needed = tail_len - out_len
                window = [
                    *req.origin_input_ids[in_len - prefix_needed :],
                    *req.output_ids,
                ]

            if ngram_size == 1:
                banned_tokens: Set[int] = set(window)
            else:
                prefix_len = ngram_size - 1
                current_prefix = window[-prefix_len:]
                search_end = len(window) - ngram_size + 1
                banned_tokens = {
                    window[idx + prefix_len]
                    for idx in range(search_end)
                    if window[idx : idx + prefix_len] == current_prefix
                }

            whitelist_ids = params.get("whitelist_token_ids") or []
            try:
                whitelist = {int(token_id) for token_id in whitelist_ids}
            except (TypeError, ValueError):
                whitelist = set()

            banned_tokens.difference_update(whitelist)

            if not banned_tokens:
                continue

            indices = list(banned_tokens)
            logits[batch_idx, indices] = -float("inf")

        return logits
