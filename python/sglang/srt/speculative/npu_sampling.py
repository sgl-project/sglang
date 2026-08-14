# Copyright 2023-2026 SGLang Team
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
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from sglang.srt.runtime_context import get_exec, get_spec

if TYPE_CHECKING:
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo


def validate_npu_target_only_sampling(
    *,
    tree_topk: int,
    num_draft_tokens: int,
    max_tree_depth: int,
    retrieve_index_shape: tuple[int, ...],
    logits_shape: tuple[int, ...],
    batch_size: int,
    use_rejection_sampling: bool,
    threshold_single: float,
    threshold_acc: float,
    sampling_backend: str,
) -> None:
    """Validate the lossless linear-chain shortcut used for sampled NPU targets."""
    if tree_topk != 1:
        raise NotImplementedError(
            "NPU non-greedy speculative sampling supports only topk=1; "
            f"got tree_topk={tree_topk}"
        )
    if use_rejection_sampling:
        raise NotImplementedError(
            "NPU non-greedy speculative sampling is target-only; disable "
            "speculative rejection sampling"
        )
    if threshold_single != 1.0 or threshold_acc != 1.0:
        raise ValueError(
            "NPU sampled-target verification requires both speculative "
            "acceptance thresholds to equal 1.0"
        )
    expected_tree_shape = (batch_size, num_draft_tokens)
    if logits_shape[0:1] != (batch_size * num_draft_tokens,):
        raise ValueError(
            "NPU sampled-target logits must have one row per verify token; "
            f"got {logits_shape}"
        )
    if (
        max_tree_depth != num_draft_tokens
        or retrieve_index_shape != expected_tree_shape
    ):
        raise ValueError(
            "NPU sampled-target verification requires an untruncated linear "
            f"chain of shape {expected_tree_shape}"
        )
    if sampling_backend not in ("ascend", "pytorch"):
        raise NotImplementedError(
            "NPU non-greedy speculative sampling supports only the 'ascend' "
            f"and 'pytorch' backends; got {sampling_backend!r}"
        )


def sample_npu_target_tokens(
    *,
    next_token_logits: torch.Tensor,
    sampling_info: "SamplingBatchInfo",
    positions: torch.Tensor,
    tree_topk: int,
    num_draft_tokens: int,
    max_tree_depth: int,
    retrieve_index_shape: tuple[int, ...],
    batch_size: int,
) -> torch.Tensor:
    """Sample target tokens for the lossless NPU topk=1 target-only path."""
    spec = get_spec()
    backend = get_exec().kernel.sampling_backend
    validate_npu_target_only_sampling(
        tree_topk=tree_topk,
        num_draft_tokens=num_draft_tokens,
        max_tree_depth=max_tree_depth,
        retrieve_index_shape=retrieve_index_shape,
        logits_shape=tuple(next_token_logits.shape),
        batch_size=batch_size,
        use_rejection_sampling=spec.speculative_use_rejection_sampling,
        threshold_single=spec.speculative_accept_threshold_single,
        threshold_acc=spec.speculative_accept_threshold_acc,
        sampling_backend=backend,
    )

    from sglang.srt.layers.sampler import (
        sampling_from_probs_torch,
        top_k_top_p_min_p_sampling_from_logits_ascend,
        top_k_top_p_min_p_sampling_from_probs_torch,
    )

    temperatures = torch.repeat_interleave(
        sampling_info.temperatures, num_draft_tokens, dim=0
    )
    # Preserve the original logits for downstream logprob calculation.
    sampling_logits = next_token_logits.clone().div_(temperatures)
    seeds = (
        None
        if sampling_info.sampling_seed is None
        else torch.repeat_interleave(
            sampling_info.sampling_seed, num_draft_tokens, dim=0
        )
    )
    needs_filtering = (
        sampling_info.need_top_k_sampling
        or sampling_info.need_top_p_sampling
        or sampling_info.need_min_p_sampling
    )
    if not needs_filtering:
        return sampling_from_probs_torch(
            F.softmax(sampling_logits, dim=-1),
            sampling_seed=seeds,
            positions=positions,
        )

    top_ks = torch.repeat_interleave(
        sampling_info.top_ks, num_draft_tokens, dim=0
    ).clone()
    top_ps = torch.repeat_interleave(sampling_info.top_ps, num_draft_tokens, dim=0)
    min_ps = torch.repeat_interleave(sampling_info.min_ps, num_draft_tokens, dim=0)
    if backend == "ascend":
        return top_k_top_p_min_p_sampling_from_logits_ascend(
            sampling_logits,
            top_ks,
            top_ps,
            min_ps,
            sampling_info.need_min_p_sampling,
            seeds,
            positions,
        )
    return top_k_top_p_min_p_sampling_from_probs_torch(
        F.softmax(sampling_logits, dim=-1),
        top_ks,
        top_ps,
        min_ps,
        sampling_info.need_min_p_sampling,
        seeds,
        positions,
    )
