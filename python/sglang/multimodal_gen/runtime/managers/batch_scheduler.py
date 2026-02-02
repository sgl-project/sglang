# SPDX-License-Identifier: Apache-2.0
"""
Memory-aware batch scheduler for diffusion model serving.

This module provides batching functionality that:
- Groups requests by compatible configurations (resolution, frames, steps)
- Estimates memory requirements to determine safe batch sizes
- Prevents starvation with configurable max wait time
- Adapts to actual memory usage over time
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Tuple

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req

logger = init_logger(__name__)


class RequestConfig(NamedTuple):
    """
    Hashable key for grouping compatible requests.

    Requests can only be batched together if they have identical RequestConfig.
    This ensures tensor shapes are compatible for batched execution.
    """

    height: int
    width: int
    num_frames: int
    num_inference_steps: int
    has_cfg: bool  # guidance_scale > 1.0 and negative_prompt exists

    @classmethod
    def from_req(cls, req: "Req") -> "RequestConfig":
        """Create RequestConfig from a Req object."""

        # Handle potential list types for dimensions
        height = req.height[0] if isinstance(req.height, list) else (req.height or 480)
        width = req.width[0] if isinstance(req.width, list) else (req.width or 848)
        num_frames = (
            req.num_frames[0] if isinstance(req.num_frames, list) else req.num_frames
        )

        return cls(
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=req.num_inference_steps,
            has_cfg=req.do_classifier_free_guidance,
        )

    def __str__(self) -> str:
        return f"({self.height}×{self.width}×{self.num_frames}f, {self.num_inference_steps}steps, cfg={self.has_cfg})"


@dataclass
class QueuedRequest:
    """Wrapper for queued requests with metadata."""

    identity: bytes
    req: "Req"
    config: RequestConfig
    arrival_time: float = field(default_factory=time.monotonic)

    @property
    def wait_time(self) -> float:
        """Time in seconds since request arrived."""
        return time.monotonic() - self.arrival_time


class BatchScheduler:
    """
    Memory-aware batch scheduler that groups compatible requests.

    Features:
    - Groups requests by compatible configurations (same resolution, frames, steps)
    - Estimates memory to determine safe batch sizes
    - Prevents starvation with max wait time
    - Adapts to actual memory usage over time
    """

    def __init__(
        self,
        max_batch_size: int = 8,
        max_wait_time_s: float = 30.0,
    ):
        """
        Args:
            max_batch_size: Hard cap on number of requests per batch
            max_wait_time_s: Force dispatch after this wait time (starvation prevention)
        """
        self.max_batch_size = max_batch_size
        self.max_wait_time_s = max_wait_time_s

        # Buckets: config -> deque of QueuedRequest
        self._buckets: Dict[RequestConfig, deque] = defaultdict(deque)

        # Track total queue size for monitoring
        self._total_queued = 0

        # Track current batch for result distribution
        self._current_batch_identities: List[bytes] = []
        self._current_batch_config: Optional[RequestConfig] = None
        self._current_batch_size: int = 0
        self._current_batch_per_request_counts: list[int] = []

    def add_request(self, identity: bytes, req: "Req") -> None:
        """Add a new request to the appropriate bucket."""
        if req.batch_size > self.max_batch_size:
            raise ValueError(
                f"Request batch size ({req.batch_size}) exceeds max_batch_size ({self.max_batch_size})"
            )

        config = RequestConfig.from_req(req)
        queued = QueuedRequest(identity=identity, req=req, config=config)
        self._buckets[config].append(queued)
        self._total_queued += 1

        logger.debug(f"Queued request {req.request_id} into bucket {config}")

    def add_requests(self, requests: List[Tuple[bytes, "Req"]]) -> None:
        """Add multiple requests."""
        for identity, req in requests:
            self.add_request(identity, req)

    @property
    def queue_size(self) -> int:
        """Total number of queued requests."""
        return self._total_queued

    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return self._total_queued == 0

    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics for monitoring."""
        stats = {
            "total_queued": self._total_queued,
            "num_buckets": len([b for b in self._buckets.values() if b]),
            "buckets": {},
        }
        for config, queue in self._buckets.items():
            if queue:
                stats["buckets"][str(config)] = {
                    "count": len(queue),
                    "oldest_wait_s": queue[0].wait_time if queue else 0,
                }
        return stats

    def _get_oldest_bucket(self) -> Optional[RequestConfig]:
        """Find bucket with the oldest waiting request."""
        oldest_config = None
        oldest_time = float("inf")

        for config, queue in self._buckets.items():
            if queue and queue[0].arrival_time < oldest_time:
                oldest_time = queue[0].arrival_time
                oldest_config = config

        return oldest_config

    def _select_best_bucket(self) -> Optional[RequestConfig]:
        """
        Select the best bucket to process next.

        Strategy:
        1. If any request has waited > max_wait_time_s, prioritize that bucket (starvation prevention)
        2. Otherwise, prefer buckets with larger potential batch sizes
        3. Tie-break by oldest request
        """
        best_config = None
        best_batch = -1
        best_oldest_wait = -1.0

        for config, queue in self._buckets.items():
            if not queue:
                continue

            oldest_wait = queue[0].wait_time

            # Starvation check
            if oldest_wait > self.max_wait_time_s:
                logger.info(
                    f"Starvation prevention: forcing bucket {config} after {oldest_wait:.1f}s wait"
                )
                return config

            # Biggest queue => biggest potential batch (or clamp to max batch if you have one)
            potential_batch = len(queue)
            if (potential_batch > best_batch) or (
                potential_batch == best_batch and oldest_wait > best_oldest_wait
            ):
                best_batch = potential_batch
                best_oldest_wait = oldest_wait
                best_config = config

        return best_config

    def _select_batch_from_bucket(self, config: RequestConfig) -> List[QueuedRequest]:
        """Select requests from a bucket up to max batch size."""
        queue = self._buckets[config]
        if not queue:
            return []

        # Collect batch
        batch = []

        while queue and len(batch) + queue[0].req.batch_size <= self.max_batch_size:
            batch.append(queue.popleft())
            self._total_queued -= 1

        return batch

    def _merge_requests(
        self, queued_items: List[QueuedRequest]
    ) -> Tuple["Req", List[int]]:
        """
        Merge multiple Req objects into a single batched Req.
        All requests must have compatible configurations.
        """
        if len(queued_items) == 1:
            return queued_items[0].req, [queued_items[0].req.batch_size]

        from copy import deepcopy

        base_req = queued_items[0].req

        # Collect all prompts
        all_prompts = []
        all_negative_prompts = []
        all_seeds = []
        per_prompt_num_outputs = []
        per_req_num_outputs = []

        for item in queued_items:
            req = item.req

            all_prompts.extend(req.prompts_as_list)

            # Handle negative prompt
            if req.sampling_params.negative_prompt:
                all_negative_prompts.extend(req.negative_prompts_as_list)

            if req.seeds:
                all_seeds.extend(req.seeds)
            elif req.sampling_params.seed is not None:
                for i in range(req.batch_size):
                    all_seeds.append(req.sampling_params.seed + i)

            num_prompts = len(req.prompts_as_list)
            if req.sampling_params.per_prompt_num_outputs:
                per_prompt_num_outputs.extend(
                    req.sampling_params.per_prompt_num_outputs
                )
                per_req_num_outputs.append(
                    sum(req.sampling_params.per_prompt_num_outputs)
                )
            else:
                per_prompt_num_outputs.extend(
                    [req.sampling_params.num_outputs_per_prompt] * num_prompts
                )
                per_req_num_outputs.append(
                    req.sampling_params.num_outputs_per_prompt * num_prompts
                )

        self._current_batch_per_request_counts = per_prompt_num_outputs

        # Create merged request
        merged = deepcopy(base_req)
        merged.sampling_params.prompt = all_prompts
        merged.sampling_params.negative_prompt = (
            all_negative_prompts if all_negative_prompts else None
        )
        merged.seeds = all_seeds if all_seeds else None
        # merged.seed = None  # Use seeds list instead
        merged.sampling_params.seed = queued_items[0].req.sampling_params.seed
        merged.sampling_params.request_id = (
            f"batch_{len(queued_items)}_{base_req.sampling_params.request_id}"
        )
        merged.sampling_params.per_prompt_num_outputs = per_prompt_num_outputs

        logger.info(f"Merged {len(queued_items)} requests into batch")

        return merged, per_req_num_outputs

    def get_next_batch(
        self,
    ) -> Optional[Tuple[List[bytes], List[int], "Req", RequestConfig]]:
        """
        Get the next batch to process.
        Returns:
            Tuple of (identities, per_req_num_outputs, merged_req, config) or None if queue is empty
        """
        if self.is_empty():
            return None

        # Select best bucket
        best_config = self._select_best_bucket()
        if best_config is None:
            return None

        # Form batch from selected bucket
        batch = self._select_batch_from_bucket(best_config)
        if not batch:
            return None

        # Extract identities and merge requests
        identities = [item.identity for item in batch]
        merged_req, per_req_num_outputs = self._merge_requests(batch)

        # Store for result distribution
        self._current_batch_identities = identities
        self._current_batch_config = best_config
        self._current_batch_size = len(batch)

        logger.info(
            f"Formed batch of {len(batch)} requests from bucket {best_config}, "
            f"queue remaining: {self._total_queued}"
            f"buckets: {[(b,len(v)) for b, v in self._buckets.items()]}"
        )

        return identities, per_req_num_outputs, merged_req, best_config
