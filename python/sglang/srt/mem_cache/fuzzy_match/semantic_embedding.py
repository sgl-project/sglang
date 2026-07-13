# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Embedding-based fuzzy-match provider backed by SemBlend."""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import torch

from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.mem_cache.fuzzy_match.config import FuzzyMatchConfig
from sglang.srt.mem_cache.fuzzy_match.fuzzy_match_provider import (
    FuzzyMatchProvider,
    FuzzyMatchResult,
    FuzzyMatchSegment,
    QualitySignals,
)

logger = logging.getLogger(__name__)


class SemanticEmbeddingProvider(FuzzyMatchProvider):
    """Lazy SemBlend wrapper for SGLang's fuzzy-match interface."""

    _MIN_SEMBLEND_VERSION = "0.3.12"

    def __init__(self, config: FuzzyMatchConfig):
        super().__init__(config)
        try:
            import semblend
        except ImportError as e:
            raise ImportError(
                "fuzzy_match_provider='SemanticEmbedding' requires the "
                "`semblend` package. Install with: "
                "pip install 'sglang[fuzzy-semantic]' (or "
                f"pip install 'semblend>={self._MIN_SEMBLEND_VERSION}')"
            ) from e

        installed = getattr(semblend, "__version__", "0.0.0")
        if _version_lt(installed, self._MIN_SEMBLEND_VERSION):
            raise ImportError(
                f"fuzzy_match_provider='SemanticEmbedding' requires "
                f"semblend>={self._MIN_SEMBLEND_VERSION} (installed: "
                f"{installed}). Upgrade with: "
                f"pip install -U 'semblend>={self._MIN_SEMBLEND_VERSION}'"
            )
        try:
            from semblend.integration.sglang.config import SemBlendProviderConfig
            from semblend.integration.sglang.provider import SemBlendProviderAdapter
        except ImportError as e:
            raise ImportError(
                f"semblend>={self._MIN_SEMBLEND_VERSION} is installed but the "
                f"sglang integration entrypoints are missing ({e}). "
                f"This usually means an editable install of a dev branch. "
                f"Reinstall with: pip install -U "
                f"'semblend>={self._MIN_SEMBLEND_VERSION}'"
            ) from e

        self._adapter_cls = SemBlendProviderAdapter
        self._adapter_config = SemBlendProviderConfig.from_dict(
            {
                "min_similarity": config.fuzzy_semantic_threshold,
                "min_reuse_ratio": config.fuzzy_min_reuse_ratio,
                "min_match_length": config.fuzzy_min_match_length,
                "max_entries": config.semantic_max_entries,
                "block_size": config.fuzzy_block_size,
                "embedding_model_name": config.embedding_model_name,
                "model_arch": config.model_arch,
            }
        )
        self._adapter = self._adapter_cls(config=self._adapter_config)
        self._fallback_tokenizer = None
        self._fallback_tokenizer_name = os.environ.get("SEMBLEND_MODEL_NAME")
        logger.info(
            "SemanticEmbeddingProvider initialized: threshold=%.2f, "
            "min_reuse=%.2f, model_arch=%s",
            config.fuzzy_semantic_threshold,
            config.fuzzy_min_reuse_ratio,
            config.model_arch,
        )

    # ------------------------------------------------------------------
    # FuzzyMatchProvider contract
    # ------------------------------------------------------------------

    def cache_on_request_finished(
        self,
        request,
        token_ids: List[int],
        kv_cache: torch.Tensor,
        cache_start_pos: int,
        cache_end_pos: int,
        radix_tree=None,
    ) -> bool:
        import time as _time

        request_id = getattr(request, "rid", None) or getattr(
            request, "request_id", None
        )
        if request_id is None:
            logger.debug(
                "[SemanticEmbedding] request has no rid; skipping registration"
            )
            return False
        request_id = str(request_id)
        if request_id.startswith(HEALTH_CHECK_RID_PREFIX):
            logger.debug(
                "[SemanticEmbedding] skipping health-check donor registration: %s",
                request_id,
            )
            return False
        if cache_end_pos <= cache_start_pos:
            logger.debug(
                "[SemanticEmbedding] skipping donor registration for empty "
                "cache range: rid=%s start=%d end=%d",
                request_id,
                cache_start_pos,
                cache_end_pos,
            )
            return False
        t_decode_start = _time.monotonic()
        prompt_text = self._decode(request, token_ids[cache_start_pos:cache_end_pos])
        t_decode_ms = (_time.monotonic() - t_decode_start) * 1000
        t_submit_start = _time.monotonic()
        ok = self._adapter.register_donor(
            request_id=request_id,
            token_ids=list(token_ids),
            kv_cache=kv_cache,
            cache_start_pos=cache_start_pos,
            cache_end_pos=cache_end_pos,
            prompt_text=prompt_text,
            extra_key=getattr(request, "extra_key", None),
            radix_tree=radix_tree,
        )
        t_submit_ms = (_time.monotonic() - t_submit_start) * 1000
        logger.info(
            "[FUZZY] cache_on_request_finished: rid=%s tokens=%d "
            "decode=%.1fms submit=%.1fms ok=%s",
            request_id,
            cache_end_pos - cache_start_pos,
            t_decode_ms,
            t_submit_ms,
            ok,
        )
        return ok

    def match_on_prefix_miss(
        self,
        prompt_token_ids: List[int],
        already_matched_len: int,
        request=None,
        extra_key=None,
    ) -> Optional[FuzzyMatchResult]:
        prompt_text = self._decode(request, prompt_token_ids[already_matched_len:])
        adapter_result = self._adapter.match(
            prompt_token_ids=list(prompt_token_ids),
            already_matched_len=already_matched_len,
            prompt_text=prompt_text,
            extra_key=extra_key,
        )
        if adapter_result is None:
            return None
        return _adapter_to_sglang_result(adapter_result)

    def on_donor_inserted(self, request, donor_last_node_id: int) -> None:
        """Forward the donor's TreeNode id from RadixCache to the adapter."""
        request_id = getattr(request, "rid", None) or getattr(
            request, "request_id", None
        )
        if request_id is None:
            return
        self._adapter.on_donor_inserted(
            request_id=str(request_id),
            donor_last_node_id=donor_last_node_id,
        )

    def on_cache_reset(self) -> None:
        """Clear provider-side donor state after SGLang flushes the cache."""
        old_adapter = getattr(self, "_adapter", None)
        clear = getattr(old_adapter, "clear", None)
        if callable(clear):
            try:
                clear()
                logger.info("[FUZZY] SemanticEmbeddingProvider cleared adapter state")
                return
            except Exception:
                logger.debug("[SemanticEmbedding] adapter clear failed", exc_info=True)

        executor = getattr(old_adapter, "_register_executor", None)
        if executor is not None:
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                executor.shutdown(wait=False)
            except Exception:
                logger.debug(
                    "[SemanticEmbedding] adapter executor shutdown failed",
                    exc_info=True,
                )

        self._adapter = self._adapter_cls(config=self._adapter_config)
        logger.info("[FUZZY] SemanticEmbeddingProvider reset adapter state")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _decode(self, request, token_ids: List[int]) -> str:
        """Decode tokens for SemBlend donor/query text.

        SGLang's OpenAI serving path does not always attach a tokenizer to the
        scheduler request object. Keep the request-local tokenizer as the first
        choice, then lazily fall back to the configured model tokenizer so donor
        registration and semantic lookup still receive prompt text.
        """
        tokenizer = None
        if request is not None:
            tokenizer = getattr(request, "tokenizer", None)
        if tokenizer is None:
            tokenizer = self._get_fallback_tokenizer()
        if tokenizer is None:
            logger.debug("[SemanticEmbedding] no tokenizer available for decode")
            return ""
        try:
            return tokenizer.decode(token_ids, skip_special_tokens=False)
        except Exception:  # pragma: no cover - tokenizer interfaces vary
            logger.debug("[SemanticEmbedding] tokenizer decode failed", exc_info=True)
            return ""

    def _get_fallback_tokenizer(self):
        if self._fallback_tokenizer is not None:
            return self._fallback_tokenizer
        if not self._fallback_tokenizer_name:
            return None
        try:
            from transformers import AutoTokenizer

            self._fallback_tokenizer = AutoTokenizer.from_pretrained(
                self._fallback_tokenizer_name,
                trust_remote_code=False,
            )
            logger.info(
                "[SemanticEmbedding] loaded fallback tokenizer: %s",
                self._fallback_tokenizer_name,
            )
        except Exception:
            logger.warning(
                "[SemanticEmbedding] failed to load fallback tokenizer: %s",
                self._fallback_tokenizer_name,
                exc_info=True,
            )
            self._fallback_tokenizer_name = None
        return self._fallback_tokenizer


def _adapter_to_sglang_result(adapter_result) -> FuzzyMatchResult:
    """Translate ``semblend.integration.sglang.types.FuzzyMatchResult`` into
    SGLang's ``FuzzyMatchResult`` shape.

    Field names are kept identical so this is a straightforward dataclass
    copy. Segments are mapped element-wise.
    """
    segments = None
    if adapter_result.segments is not None:
        segments = [
            FuzzyMatchSegment(
                target_positions=_as_tensor(s.target_positions),
                donor_positions=_as_tensor(s.donor_positions),
                donor_node_id=s.donor_node_id,
                donor_offset=s.donor_offset,
                length=s.length,
                donor_kv_indices=(
                    _as_tensor(s.donor_kv_indices)
                    if s.donor_kv_indices is not None
                    else None
                ),
                donor_req_id=s.donor_req_id,
                layer_recompute_mask=s.layer_recompute_mask,
            )
            for s in adapter_result.segments
        ]

    quality = None
    if adapter_result.quality_signals is not None:
        qs = adapter_result.quality_signals
        quality = QualitySignals(
            cosine_similarity=qs.cosine_similarity,
            reuse_ratio=qs.reuse_ratio,
            confidence_tier=qs.confidence_tier,
            passed_quality_gate=qs.passed_quality_gate,
            rejection_reason=qs.rejection_reason,
        )

    return FuzzyMatchResult(
        cached_token_count=adapter_result.cached_token_count,
        cached_token_ids=list(adapter_result.cached_token_ids),
        prompt_token_count=adapter_result.prompt_token_count,
        kv_cache_indices=_as_tensor(adapter_result.kv_cache_indices),
        position_offset=adapter_result.position_offset,
        cached_start_pos=adapter_result.cached_start_pos,
        segments=segments,
        layer_recompute_mask=adapter_result.layer_recompute_mask,
        quality_signals=quality,
        donor_last_node_id=getattr(adapter_result, "donor_last_node_id", None),
        match_entry=adapter_result._match_entry,
    )


def _as_tensor(obj) -> torch.Tensor:
    """Coerce list/numpy/torch input to a torch.Tensor of int64 indices."""
    if obj is None:
        return torch.empty(0, dtype=torch.int64)
    if isinstance(obj, torch.Tensor):
        return obj
    return torch.as_tensor(list(obj), dtype=torch.int64)


def _version_lt(installed: str, minimum: str) -> bool:
    """Compare two semver-style strings without pulling in `packaging`.

    Returns True iff ``installed`` is strictly less than ``minimum`` on a
    component-by-component numeric comparison of the leading dotted prefix.
    Non-numeric suffixes (e.g. ".dev0", "rc1") are treated as zero on each
    side, which is conservative for a "must-be-at-least" gate.
    """

    def parts(v: str) -> list[int]:
        out = []
        for chunk in v.split("."):
            digits = ""
            for ch in chunk:
                if ch.isdigit():
                    digits += ch
                else:
                    break
            out.append(int(digits) if digits else 0)
        return out

    a, b = parts(installed), parts(minimum)
    n = max(len(a), len(b))
    a += [0] * (n - len(a))
    b += [0] * (n - len(b))
    return a < b
