import json
from abc import ABC, abstractmethod
from array import array
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

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
        """Define the callable behavior."""
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
        assert all(
            disallowed_token_ids == c["token_ids"] for c in custom_param_list
        ), f"{custom_param_list=}"
        logits[..., disallowed_token_ids] = -float("inf")
        return logits


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
            if (
                self.THINKING_START_TOKEN_ID not in cur_ids
                or self.THINKING_END_TOKEN_ID in cur_ids
            ):
                continue

            # Find the index of the thinking start token
            start_index = cur_ids.index(self.THINKING_START_TOKEN_ID)

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

            sequence = req.origin_input_ids + req.output_ids
            if len(sequence) < ngram_size:
                continue

            search_start = max(0, len(sequence) - window_size)
            search_end = len(sequence) - ngram_size + 1
            if search_end <= search_start:
                continue

            if ngram_size > 1:
                current_prefix = sequence[-(ngram_size - 1) :]
            else:
                current_prefix = array("q")

            banned_tokens: Set[int] = set()
            for idx in range(search_start, search_end):
                ngram = sequence[idx : idx + ngram_size]
                if ngram_size == 1 or ngram[:-1] == current_prefix:
                    banned_tokens.add(ngram[-1])

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


# Port of mineru-vl-utils' MinerULogitsProcessor (HF-style ``no_repeat_ngram_size``):
# https://github.com/opendatalab/mineru-vl-utils/blob/main/mineru_vl_utils/logits_processor/vllm_v1_no_repeat_ngram.py
class MinerUNoRepeatNGramLogitProcessor(CustomLogitProcessor):
    """Forbid repeating any ``ngram_size``-gram over the generated output.

    This is the SGLang equivalent of ``mineru_vl_utils.MinerULogitsProcessor``,
    the ``no_repeat_ngram_size`` guard the vLLM / Transformers paths apply for
    MinerU2.5. A token is banned whenever emitting it would reproduce an
    ``ngram_size``-token sequence that already appeared earlier in the output,
    which breaks long verbatim repetition loops on degenerate documents while
    leaving short legitimate repeats (e.g. table cells) untouched.

    Custom params:
        ngram_size (int): n-gram length to forbid repeating. Defaults to 100
            (matching mineru-vl-utils) only when the key is absent; an explicit
            value of ``0`` or less disables the processor. ``ngram_size=1`` bans
            every token already seen in the scanned output.
        window_size (int, optional): only scan the last ``window_size`` output
            tokens for repeats. A positive value bounds the per-step cost; the
            default (``None``, or any non-positive value) scans the full output
            history, which is faithful to mineru-vl-utils but costs ``O(L)`` per
            decode step (``O(L^2)`` over an ``L``-token generation). Set
            ``window_size`` for long outputs.

    Note: this rescans the output each step (like the sibling
    ``DeepseekOCRNoRepeatNGramLogitProcessor``) rather than keeping a persistent
    cross-step cache, since the processor instance is shared across requests and
    the interface offers no per-request teardown hook.
    """

    DEFAULT_NGRAM_SIZE = 100

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
                ngram_size = int(params.get("ngram_size", self.DEFAULT_NGRAM_SIZE))
            except (TypeError, ValueError):
                continue
            if ngram_size <= 0:
                continue

            window = params.get("window_size")
            try:
                window_size = int(window) if window is not None else None
            except (TypeError, ValueError):
                window_size = None

            # MinerU operates on the generated tokens only (not the prompt).
            output_ids: List[int] = req.output_ids
            if window_size is not None and window_size > 0:
                output_ids = output_ids[-window_size:]
            if len(output_ids) < ngram_size:
                continue

            # The trailing (ngram_size - 1)-gram that the next token would extend.
            current_prefix = (
                tuple(output_ids[-(ngram_size - 1) :]) if ngram_size > 1 else tuple()
            )
            # Cheap filter: only positions whose prefix ends in the same token as
            # current_prefix can match, so we skip the tuple build/compare for the
            # rest (keeps the common case ~O(L) instead of O(L * ngram_size)).
            prefix_last = current_prefix[-1] if ngram_size > 1 else None

            # Ban every token that already completed this same prefix earlier.
            banned_tokens: Set[int] = set()
            for idx in range(len(output_ids) - ngram_size + 1):
                if ngram_size > 1 and output_ids[idx + ngram_size - 2] != prefix_last:
                    continue
                if tuple(output_ids[idx : idx + ngram_size - 1]) == current_prefix:
                    banned_tokens.add(output_ids[idx + ngram_size - 1])

            if banned_tokens:
                logits[batch_idx, list(banned_tokens)] = -float("inf")

        return logits
