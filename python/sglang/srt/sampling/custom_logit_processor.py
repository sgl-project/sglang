import hashlib
import hmac
import json
import logging
import os
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

import dill
import orjson
import torch

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)

# Registry of trusted CustomLogitProcessor subclasses.
# Only classes registered here can be deserialized from client requests.
_TRUSTED_PROCESSORS: Dict[str, type] = {}

# Secret key for HMAC signature verification of serialized processors.
# When set (via SGLANG_LOGIT_PROCESSOR_SECRET env var), only payloads
# signed with this key are accepted. When unset, signature verification
# is skipped but the class allowlist is still enforced.
_PROCESSOR_SECRET = os.environ.get("SGLANG_LOGIT_PROCESSOR_SECRET", "")


def register_logit_processor(cls):
    """Decorator to register a CustomLogitProcessor subclass as trusted."""
    _TRUSTED_PROCESSORS[cls.__name__] = cls
    return cls


def _verify_signature(payload_hex: str, signature: str) -> bool:
    """Verify HMAC-SHA256 signature of the serialized payload."""
    if not _PROCESSOR_SECRET:
        return True  # Skip verification if no secret configured
    expected = hmac.new(
        _PROCESSOR_SECRET.encode(),
        bytes.fromhex(payload_hex),
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


@lru_cache(maxsize=None)
def _cache_from_str(json_str: str):
    """Deserialize a json string to a Callable object.

    Security: Only accepts payloads whose deserialized class is in the
    _TRUSTED_PROCESSORS registry. Optionally verifies HMAC signature
    when SGLANG_LOGIT_PROCESSOR_SECRET is configured.
    """
    data = orjson.loads(json_str)
    payload_hex = data["callable"]

    # Verify HMAC signature if secret is configured
    signature = data.get("signature", "")
    if _PROCESSOR_SECRET and not _verify_signature(payload_hex, signature):
        raise ValueError(
            "Invalid signature for custom logit processor payload. "
            "Set SGLANG_LOGIT_PROCESSOR_SECRET and sign payloads with "
            "CustomLogitProcessor.to_str() on a trusted machine."
        )

    # Deserialize and validate against allowlist
    obj = dill.loads(bytes.fromhex(payload_hex))

    # obj should be a class (subclass of CustomLogitProcessor)
    if isinstance(obj, type):
        if obj.__name__ not in _TRUSTED_PROCESSORS:
            raise ValueError(
                f"Untrusted logit processor class: {obj.__name__}. "
                f"Only registered processors are allowed: "
                f"{list(_TRUSTED_PROCESSORS.keys())}. "
                f"Use @register_logit_processor to register custom classes."
            )
    else:
        raise ValueError(
            f"Expected a CustomLogitProcessor class, got {type(obj).__name__}. "
            "Custom logit processors must be classes, not arbitrary objects."
        )

    return obj


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
        """Serialize the callable function to a JSON-compatible string.

        If SGLANG_LOGIT_PROCESSOR_SECRET is set, the payload is signed
        with HMAC-SHA256 for tamper protection.
        """
        payload_hex = dill.dumps(cls).hex()
        result = {"callable": payload_hex}
        if _PROCESSOR_SECRET:
            sig = hmac.new(
                _PROCESSOR_SECRET.encode(),
                bytes.fromhex(payload_hex),
                hashlib.sha256,
            ).hexdigest()
            result["signature"] = sig
        return json.dumps(result)

    @classmethod
    def from_str(cls, json_str: str):
        """Deserialize a callable function from a JSON string.

        Only accepts classes registered via @register_logit_processor.
        """
        return _cache_from_str(json_str)()


@register_logit_processor
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


@register_logit_processor
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


@register_logit_processor
class Glm4MoeThinkingBudgetLogitProcessor(ThinkingBudgetLogitProcessor):
    """A logit processor that controls the length of thinking for GLM-4.5 / GLM-4.6 / GLM-4.5V / GLM-4.6V models."""

    THINKING_START_TOKEN_ID: int = 151350
    THINKING_END_TOKEN_ID: int = 151351
    NEW_LINE_TOKEN_ID: int = 198


@register_logit_processor
class Qwen3ThinkingBudgetLogitProcessor(ThinkingBudgetLogitProcessor):
    """A logit processor that controls the length of thinking for Qwen3 models."""

    THINKING_START_TOKEN_ID: int = 151667
    THINKING_END_TOKEN_ID: int = 151668
    NEW_LINE_TOKEN_ID: int = 198


@register_logit_processor
class DeepSeekR1ThinkingBudgetLogitProcessor(ThinkingBudgetLogitProcessor):
    """A logit processor that controls the length of thinking for DeepSeek-R1 models."""

    THINKING_START_TOKEN_ID: int = 128798
    THINKING_END_TOKEN_ID: int = 128799
    NEW_LINE_TOKEN_ID: int = 201


# Adapted from DeepSeek's implementation: https://github.com/deepseek-ai/DeepSeek-OCR/blob/main/DeepSeek-OCR-master/DeepSeek-OCR-vllm/process/ngram_norepeat.py
@register_logit_processor
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
