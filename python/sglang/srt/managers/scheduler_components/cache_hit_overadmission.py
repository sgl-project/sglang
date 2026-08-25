"""
Copyright 2026 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


@dataclass(frozen=True, slots=True)
class CacheHitOveradmissionDecision:
    allowed: bool
    reason: str
    cached_tokens: int = 0
    input_tokens: int = 0
    uncached_tokens: int = 0
    remaining_new_tokens: int = 0

    @property
    def hit_ratio(self) -> float:
        return self.cached_tokens / self.input_tokens if self.input_tokens > 0 else 0.0


def evaluate_cache_hit_overadmission(
    req: Req,
    *,
    min_hit_ratio: float,
    max_uncached_prefill_tokens: int,
    max_new_tokens: int,
) -> CacheHitOveradmissionDecision:
    """Evaluate request-shape eligibility after device prefix matching.

    Resource availability is intentionally not handled here. The scheduler
    checks request rows and Mamba slots directly, while ``PrefillAdder`` owns
    the exact full-KV budget and prefix-node locking transaction.
    """

    shape_rejection = evaluate_cache_hit_overadmission_shape(
        req, max_new_tokens=max_new_tokens
    )
    if shape_rejection is not None:
        return shape_rejection

    def reject(reason: str, **kwargs) -> CacheHitOveradmissionDecision:
        return CacheHitOveradmissionDecision(False, reason, **kwargs)

    if any(
        getattr(req, field, 0) > 0
        for field in (
            "host_hit_length",
            "swa_host_hit_length",
            "storage_hit_length",
            "mamba_host_hit_length",
        )
    ):
        return reject("non_device_cache")

    input_tokens = len(req.full_untruncated_fill_ids)
    cached_tokens = min(len(req.prefix_indices), input_tokens)
    uncached_tokens = input_tokens - cached_tokens
    declared_new_tokens = req.sampling_params.max_new_tokens
    remaining_new_tokens = (
        max(int(declared_new_tokens) - len(req.output_ids), 0)
        if declared_new_tokens is not None
        else max_new_tokens + 1
    )
    details = dict(
        cached_tokens=cached_tokens,
        input_tokens=input_tokens,
        uncached_tokens=uncached_tokens,
        remaining_new_tokens=remaining_new_tokens,
    )

    if input_tokens <= 0:
        return reject("empty_input", **details)
    if cached_tokens <= 0:
        return reject("no_device_cache", **details)
    if cached_tokens / input_tokens < min_hit_ratio:
        return reject("hit_ratio", **details)
    if uncached_tokens > max_uncached_prefill_tokens:
        return reject("uncached_prefill", **details)
    if remaining_new_tokens > max_new_tokens:
        return reject("max_new_tokens", **details)

    return CacheHitOveradmissionDecision(True, "admitted", **details)


def evaluate_cache_hit_overadmission_shape(
    req: Req,
    *,
    max_new_tokens: int,
) -> Optional[CacheHitOveradmissionDecision]:
    """Reject request shapes that do not need prefix matching to classify."""
    if req.is_retracted:
        return CacheHitOveradmissionDecision(False, "retracted")
    if req.output_ids:
        return CacheHitOveradmissionDecision(False, "continuation")
    if req.sampling_params.ignore_eos:
        return CacheHitOveradmissionDecision(False, "ignore_eos")
    if req.session is not None:
        return CacheHitOveradmissionDecision(False, "session")
    if (
        req.multimodal_inputs is not None
        or req.input_embeds is not None
        or req.positional_embed_overrides is not None
    ):
        return CacheHitOveradmissionDecision(False, "unsupported_input")

    declared_new_tokens = req.sampling_params.max_new_tokens
    if declared_new_tokens is None or int(declared_new_tokens) > max_new_tokens:
        return CacheHitOveradmissionDecision(
            False,
            "max_new_tokens",
            remaining_new_tokens=(
                max_new_tokens + 1
                if declared_new_tokens is None
                else int(declared_new_tokens)
            ),
        )
    return None
