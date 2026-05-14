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

"""Embedding-based semantic fuzzy-match provider.

Wires SGLang's ``FuzzyMatchProvider`` interface to the in-process
embedding + alignment pipeline shipped in the ``semblend`` pip package.
The dependency is imported lazily so SGLang installs that do not select
``fuzzy_match_provider="SemanticEmbedding"`` never need it on disk.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch

from sglang.srt.mem_cache.fuzzy_match.config import FuzzyMatchConfig
from sglang.srt.mem_cache.fuzzy_match.fuzzy_match_provider import (
    FuzzyMatchProvider,
    FuzzyMatchResult,
    FuzzyMatchSegment,
    QualitySignals,
)

logger = logging.getLogger(__name__)


class SemanticEmbeddingProvider(FuzzyMatchProvider):
    """Embedding-based semantic fuzzy match.

    Construction lazily imports the ``semblend`` package; an unselected
    ``SemanticEmbedding`` does not require the package to be installed.
    The provider is process-local (in-process embedding + numpy ANN); no
    network or service dependency.
    """

    _MIN_SEMBLEND_VERSION = "0.3.7"

    def __init__(self, config: FuzzyMatchConfig):
        super().__init__(config)
        try:
            import semblend
        except ImportError as e:
            raise ImportError(
                "fuzzy_match_provider='SemanticEmbedding' requires the "
                "`semblend` package. Install with: "
                f"pip install 'semblend>={self._MIN_SEMBLEND_VERSION}'"
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

        adapter_config = SemBlendProviderConfig.from_dict(
            {
                "min_similarity": config.fuzzy_semantic_threshold,
                "min_reuse_ratio": config.fuzzy_min_reuse_ratio,
                "min_match_length": config.fuzzy_min_match_length,
                "max_entries": config.fuzzy_non_prefix_max_entries,
                "block_size": config.fuzzy_block_size,
                "embedding_use_gpu": config.embedding_use_gpu,
                "embedding_model_name": config.embedding_model_name,
                "model_arch": config.model_arch,
                "enable_bathtub": config.enable_bathtub,
                "top_k": config.fuzzy_top_k,
                "quality_gate_ppl_threshold": config.quality_gate_ppl_threshold,
                "discovery_only": config.discovery_only,
            }
        )
        self._adapter = SemBlendProviderAdapter(config=adapter_config)
        logger.info(
            "SemanticEmbeddingProvider initialized: threshold=%.2f, "
            "min_reuse=%.2f, model_arch=%s, enable_bathtub=%s",
            config.fuzzy_semantic_threshold,
            config.fuzzy_min_reuse_ratio,
            config.model_arch,
            config.enable_bathtub,
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
        prompt_text = self._decode(request, token_ids[cache_start_pos:cache_end_pos])
        request_id = getattr(request, "rid", None) or getattr(request, "request_id", None)
        if request_id is None:
            logger.debug("[SemanticEmbedding] request has no rid; skipping registration")
            return False
        return self._adapter.register_donor(
            request_id=str(request_id),
            token_ids=list(token_ids),
            kv_cache=kv_cache,
            cache_start_pos=cache_start_pos,
            cache_end_pos=cache_end_pos,
            prompt_text=prompt_text,
            radix_tree=radix_tree,
        )

    def match_on_prefix_miss(
        self,
        prompt_token_ids: List[int],
        already_matched_len: int,
    ) -> Optional[FuzzyMatchResult]:
        prompt_text = self._decode(None, prompt_token_ids[already_matched_len:])
        adapter_result = self._adapter.match(
            prompt_token_ids=list(prompt_token_ids),
            already_matched_len=already_matched_len,
            prompt_text=prompt_text,
        )
        if adapter_result is None:
            return None
        return _adapter_to_sglang_result(adapter_result)

    def on_donor_inserted(self, request, donor_last_node_id: int) -> None:
        """Forward the donor's TreeNode id from RadixCache to the adapter.

        Called by RadixCache.cache_finished_req after the donor's KV has been
        inserted. Without this, the donor's slots aren't lock_ref'd at match
        time and LRU eviction frees them while a recipient request is
        consuming them, tripping the SGLang pool-leak detector.
        """
        request_id = getattr(request, "rid", None) or getattr(request, "request_id", None)
        if request_id is None:
            return
        self._adapter.on_donor_inserted(
            request_id=str(request_id),
            donor_last_node_id=donor_last_node_id,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _decode(self, request, token_ids: List[int]) -> str:
        """Decode tokens to text via the request's tokenizer if available.

        SGLang's scheduler plumbs the tokenizer through several layers; we
        accept best-effort here. The adapter has its own tokenizer fallback,
        so this is a fast-path optimization, not a hard requirement.
        """
        tokenizer = None
        if request is not None:
            tokenizer = getattr(request, "tokenizer", None)
        if tokenizer is None:
            return ""
        try:
            return tokenizer.decode(token_ids, skip_special_tokens=False)
        except Exception:  # pragma: no cover - tokenizer interfaces vary
            return ""


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
        _match_entry=adapter_result._match_entry,
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
