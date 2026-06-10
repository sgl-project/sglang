import json
import logging
from abc import ABC, abstractmethod
from array import array
from collections import deque
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

import dill
import orjson
import torch

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


@lru_cache(maxsize=None)
def _cache_from_str(json_str: str):
    """Deserialize a json string to a Callable object.
    This function is cached to avoid redundant deserialization.
    """
    data = orjson.loads(json_str)
    return dill.loads(bytes.fromhex(data["callable"]))


@lru_cache(maxsize=1)
def _get_logit_processor_collector_impl():
    from sglang.srt.observability.metrics_collector import LogitProcessorCollector

    return LogitProcessorCollector()


def _is_logit_processor_metrics_rank() -> bool:
    try:
        from sglang.srt.layers.dp_attention import get_attention_tp_rank

        return get_attention_tp_rank() == 0
    except (AssertionError, ValueError):
        try:
            from sglang.srt.distributed import get_tensor_model_parallel_rank

            return get_tensor_model_parallel_rank() == 0
        except (AssertionError, ValueError):
            return True


def _get_logit_processor_collector():
    try:
        from sglang.srt.server_args import get_global_server_args

        if not get_global_server_args().enable_metrics:
            return None
    except ValueError:
        return None
    if not _is_logit_processor_metrics_rank():
        return None
    return _get_logit_processor_collector_impl()


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


# Default n-gram repetition-truncation tuning. These are empirically tuned
# defaults; they can be overridden per request through ``custom_params``.
DEFAULT_REPETITION_TRUNCATION_NGRAM_SIZE = 300
DEFAULT_REPETITION_TRUNCATION_WINDOW_SIZE = 8300
DEFAULT_REPETITION_TRUNCATION_MIN_REPEAT = 15
DEFAULT_REPETITION_TRUNCATION_MIN_CONTENT_LENGTH = 16384


class RepetitionTruncationLogitProcessor(CustomLogitProcessor):
    """Detect n-gram repetition in a request's output and truncate via forced EOS.

    Model-agnostic: works for any autoregressive LM. When an n-gram repeats at
    least ``min_repeat`` times within a sliding window over the most recent
    ``window_size`` output tokens, the request is finished with a forced EOS and
    a ``repetition_truncation`` finish reason.

    Tuning knobs are read from each request's ``custom_params`` and fall back to
    the module-level ``DEFAULT_REPETITION_TRUNCATION_*`` constants:
      - ``ngram_size``: length of the repeated n-gram to look for.
      - ``window_size``: size of the sliding window (in output tokens).
      - ``min_repeat``: how many times an n-gram must occur to count as a repeat
        (clamped to a minimum of 2 so a single occurrence never triggers).
      - ``min_content_length``: minimum output length before detection kicks in.
      - ``eos_token_id``: the token to force; defaults to the request's first
        ``eos_token_ids`` entry, so no model-specific id is hard-coded.

    Set ``SGLANG_DEBUG_REPETITION_TRUNCATION_DETECT_ONLY=1`` to mark the finish
    reason without actually truncating (for observing detection in production).
    """

    REQ_PARAM_KEY: str = "__req__"
    CACHE_ATTR: str = "_repetition_truncation_cache"
    DETECT_FLAG_ATTR: str = "_repetition_truncation_detected"
    ACTION_SKIP: str = "skip"
    ACTION_FORCE_STOP: str = "force_stop"

    @staticmethod
    def _read_int(params: Dict[str, Any], key: str, default: int) -> int:
        value = params.get(key)
        if value is None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _resolve_eos_token_id(params: Dict[str, Any], req: "Req") -> Optional[int]:
        eos = params.get("eos_token_id")
        if eos is not None:
            try:
                return int(eos)
            except (TypeError, ValueError):
                return None
        eos_token_ids = getattr(req, "eos_token_ids", None)
        if eos_token_ids:
            return next(iter(eos_token_ids))
        return None

    def __call__(
        self,
        logits: torch.Tensor,
        custom_param_list: Optional[List[Dict[str, Any]]] = None,
    ) -> torch.Tensor:
        if not custom_param_list:
            return logits

        from sglang.srt.environ import envs

        debug_detect_only = envs.SGLANG_DEBUG_REPETITION_TRUNCATION_DETECT_ONLY.get()
        collector = _get_logit_processor_collector()
        # Per-request decision cache for this batch (spec decoding can repeat a
        # request across several rows; we want one decision applied to all rows).
        per_req_action = {}

        for batch_idx, params in enumerate(custom_param_list):
            if not params:
                continue

            req = params.get(self.REQ_PARAM_KEY)
            if req is None:
                continue

            eos_token_id = self._resolve_eos_token_id(params, req)

            # A request already flagged for truncation keeps forcing EOS on every
            # subsequent row until it is filtered out of the batch.
            if req.repetition_detected and not debug_detect_only:
                if eos_token_id is not None:
                    logits[batch_idx, :] = -float("inf")
                    logits[batch_idx, eos_token_id] = 0.0
                continue

            action = per_req_action.get(req)
            if action is not None:
                # Same request seen again in this batch: reuse the decision.
                if action == self.ACTION_SKIP:
                    continue
                if eos_token_id is not None:
                    logits[batch_idx, :] = -float("inf")
                    logits[batch_idx, eos_token_id] = 0.0
                continue

            ngram_size = self._read_int(
                params, "ngram_size", DEFAULT_REPETITION_TRUNCATION_NGRAM_SIZE
            )
            window_size = self._read_int(
                params, "window_size", DEFAULT_REPETITION_TRUNCATION_WINDOW_SIZE
            )
            min_repeat = max(
                2,
                self._read_int(
                    params, "min_repeat", DEFAULT_REPETITION_TRUNCATION_MIN_REPEAT
                ),
            )
            min_content_length = self._read_int(
                params,
                "min_content_length",
                DEFAULT_REPETITION_TRUNCATION_MIN_CONTENT_LENGTH,
            )

            if ngram_size <= 1 or window_size <= 0 or window_size < ngram_size:
                logger.warning(
                    "RepetitionTruncation processor skipped due to invalid config: "
                    "window_size=%s, ngram_size=%s",
                    window_size,
                    ngram_size,
                )
                per_req_action[req] = self.ACTION_SKIP
                continue

            sequence = req.output_ids
            seq_len = len(sequence)
            if seq_len < min_content_length or seq_len < ngram_size:
                per_req_action[req] = self.ACTION_SKIP
                continue

            if collector is not None and not getattr(req, self.DETECT_FLAG_ATTR, False):
                collector.repetition_truncation_detected_count.inc()
                setattr(req, self.DETECT_FLAG_ATTR, True)

            repeated = self._update_cache_and_detect(
                req, sequence, seq_len, ngram_size, window_size, min_repeat
            )
            if not repeated:
                per_req_action[req] = self.ACTION_SKIP
                continue

            if collector is not None and req.repetition_detected is False:
                collector.repetition_truncation_truncated_count.inc()

            req.repetition_detected = True
            if debug_detect_only:
                per_req_action[req] = self.ACTION_SKIP
                continue

            if req.to_finish is None:
                from sglang.srt.managers.schedule_batch import FINISH_REPEAT_TRUNCATION

                req.to_finish = FINISH_REPEAT_TRUNCATION()

            if eos_token_id is not None:
                logits[batch_idx, :] = -float("inf")
                logits[batch_idx, eos_token_id] = 0.0
            per_req_action[req] = self.ACTION_FORCE_STOP

        return logits

    def _update_cache_and_detect(
        self,
        req: "Req",
        sequence,
        seq_len: int,
        ngram_size: int,
        window_size: int,
        min_repeat: int,
    ) -> bool:
        """Incrementally maintain a sliding-window n-gram count for ``req``.

        Instead of rescanning the whole output on every decode step, the cache
        appends only the newly formed n-grams and evicts those that fall outside
        the last ``window_size`` tokens. Returns True if any n-gram in the window
        currently occurs at least ``min_repeat`` times.
        """
        cache = getattr(req, self.CACHE_ATTR, None)
        if cache is None:
            cache = {
                "last_len": 0,
                "queue_start": 0,
                "queue": deque(),
                "ngram_counts": {},
            }
            setattr(req, self.CACHE_ATTR, cache)

        last_len = cache["last_len"]
        if seq_len < last_len:
            # Sequence can shrink on speculative-decoding retraction; reset.
            cache["queue"].clear()
            cache["ngram_counts"].clear()
            cache["queue_start"] = 0
            last_len = 0

        if seq_len != last_len:
            start_i = max(0, last_len - ngram_size + 1)
            end_i = seq_len - ngram_size
            if end_i >= start_i:
                queue = cache["queue"]
                ngram_counts = cache["ngram_counts"]
                if not queue:
                    cache["queue_start"] = start_i
                for i in range(start_i, end_i + 1):
                    ngram = tuple(sequence[i : i + ngram_size])
                    queue.append(ngram)
                    ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
            cache["last_len"] = seq_len

            # Evict n-grams that start before the sliding window.
            min_start = max(0, seq_len - window_size)
            queue = cache["queue"]
            ngram_counts = cache["ngram_counts"]
            queue_start = cache["queue_start"]
            while queue and queue_start < min_start:
                ngram = queue.popleft()
                count = ngram_counts[ngram] - 1
                if count:
                    ngram_counts[ngram] = count
                else:
                    del ngram_counts[ngram]
                queue_start += 1
            cache["queue_start"] = queue_start

        ngram_counts = cache["ngram_counts"]
        if not ngram_counts:
            return False
        return any(count >= min_repeat for count in ngram_counts.values())


# Maps the --default-custom-logit-processor choice to its processor class. This
# is the single source of truth for server-side default processors.
DEFAULT_CUSTOM_LOGIT_PROCESSOR_REGISTRY = {
    "RepetitionTruncationLogitProcessor": RepetitionTruncationLogitProcessor,
}
