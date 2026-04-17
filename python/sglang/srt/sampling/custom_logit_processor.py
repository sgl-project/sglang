import json
from abc import ABC, abstractmethod
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


class ReasoningEosRedirectLogitProcessor(CustomLogitProcessor):
    """Redirect EOS-class tokens to the think_end token while a request is still
    in the reasoning section.

    Triggers when ``softmax(logits / temperature)[redirect_eos_token_ids].sum()``
    exceeds ``prob_threshold``. When triggered, the processor surgically masks
    the EOS class to ``-inf`` and lifts the think_end token to a strict argmax
    so that the next step samples the think_end token. All other token logits
    are left untouched.

    Per-request params (passed via ``SamplingParams.custom_params``):

    - ``think_end_token_id`` (int, required)
    - ``think_start_token_id`` (Optional[int]): used to detect entry into the
      reasoning section. Pass ``None`` together with ``force_reasoning=True``
      for models whose chat template never prefills a think-start token (e.g.
      original DeepSeek-R1).
    - ``redirect_eos_token_ids`` (List[int], required, non-empty): the EOS /
      chat-end token id set whose probability mass triggers the redirect.
    - ``prob_threshold`` (float, default 0.5)
    - ``force_reasoning`` (bool, default False)
    """

    def __call__(
        self,
        logits: torch.Tensor,
        custom_param_list: Optional[List[Dict[str, Any]]] = None,
    ) -> torch.Tensor:
        if not custom_param_list:
            return logits

        for i, params in enumerate(custom_param_list):
            if not params:
                continue
            req = params.get("__req__")
            think_end_id = params.get("think_end_token_id")
            eos_ids = params.get("redirect_eos_token_ids")
            if req is None or think_end_id is None or not eos_ids:
                continue

            think_start_id = params.get("think_start_token_id")
            try:
                threshold = float(params.get("prob_threshold", 0.5))
            except (TypeError, ValueError):
                continue
            force_reasoning = bool(params.get("force_reasoning", False))

            cur_ids = req.output_ids
            origin_ids = req.origin_input_ids

            # Skip if the request has already left the reasoning section.
            if think_end_id in cur_ids:
                continue

            # Skip if the request has not entered the reasoning section yet.
            saw_start = force_reasoning or (
                think_start_id is not None
                and (think_start_id in cur_ids or think_start_id in origin_ids)
            )
            if not saw_start:
                continue

            row = logits[i]
            temperature = (
                getattr(req.sampling_params, "temperature", 1.0)
                if req.sampling_params is not None
                else 1.0
            )
            try:
                temperature = float(temperature)
            except (TypeError, ValueError):
                temperature = 1.0
            if temperature <= 0.0:
                temperature = 1.0

            scaled = row.float() / max(temperature, 1e-6)
            probs = torch.softmax(scaled, dim=-1)
            eos_indices = torch.as_tensor(
                eos_ids, device=probs.device, dtype=torch.long
            )
            eos_prob_sum = probs.index_select(0, eos_indices).sum().item()
            if eos_prob_sum < threshold:
                continue

            eos_max = row[eos_indices].max().item()
            row[eos_indices] = -float("inf")
            new_max = max(row.max().item(), eos_max)
            row[think_end_id] = new_max + 1.0

        return logits


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

            sequence: List[int] = req.origin_input_ids + req.output_ids
            if len(sequence) < ngram_size:
                continue

            search_start = max(0, len(sequence) - window_size)
            search_end = len(sequence) - ngram_size + 1
            if search_end <= search_start:
                continue

            if ngram_size > 1:
                current_prefix = tuple(sequence[-(ngram_size - 1) :])
            else:
                current_prefix = tuple()

            banned_tokens: Set[int] = set()
            for idx in range(search_start, search_end):
                ngram = sequence[idx : idx + ngram_size]
                if ngram_size == 1 or tuple(ngram[:-1]) == current_prefix:
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
