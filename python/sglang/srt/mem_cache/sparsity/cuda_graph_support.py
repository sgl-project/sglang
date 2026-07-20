from __future__ import annotations

import dataclasses
from typing import Any, Optional

import torch

_DEFAULT_CONTEXT_BUCKETS = (10 * 1024, 33 * 1024)


class SparseCudaGraphRuntimeProvider:
    def __init__(self, coordinator: Any):
        self.coordinator = coordinator
        self.page_size = coordinator.page_size
        self.max_context_len = coordinator.req_to_token_pool.max_context_len
        if self.page_size <= 0 or self.max_context_len <= 0:
            raise ValueError("Sparse CUDA graph dimensions must be positive")

        self.max_num_pages = (
            self.max_context_len + self.page_size - 1
        ) // self.page_size
        context_buckets = coordinator.config.sparse_extra_config.get(
            "cuda_graph_context_buckets", _DEFAULT_CONTEXT_BUCKETS
        )
        if (
            not isinstance(context_buckets, (list, tuple))
            or not context_buckets
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
                for value in context_buckets
            )
        ):
            raise ValueError(
                "cuda_graph_context_buckets must contain positive integers"
            )

        page_buckets = {
            min(
                (context_len + self.page_size - 1) // self.page_size,
                self.max_num_pages,
            )
            for context_len in context_buckets
        }
        page_buckets.add(self.max_num_pages)
        self.page_buckets = tuple(sorted(page_buckets))

    def cuda_graph_capture_variants(self):
        return self.page_buckets

    def select_cuda_graph_variant(self, forward_batch):
        seq_lens = getattr(forward_batch, "seq_lens_cpu", None)
        if torch.is_tensor(seq_lens):
            if seq_lens.device.type != "cpu":
                return True, self.page_buckets[-1]
            max_seq_len = int(seq_lens.max().item()) if seq_lens.numel() else 0
        elif seq_lens is None:
            return True, self.page_buckets[-1]
        else:
            try:
                max_seq_len = max((int(value) for value in seq_lens), default=0)
            except TypeError:
                return True, self.page_buckets[-1]

        required_pages = (max_seq_len + self.page_size - 1) // self.page_size
        runtime_variant = next(
            (capacity for capacity in self.page_buckets if capacity >= required_pages),
            None,
        )
        return runtime_variant is not None, runtime_variant

    def _publish_variant(self, forward_batch, runtime_variant) -> None:
        if runtime_variant not in self.page_buckets:
            raise ValueError(
                f"Unknown sparse CUDA graph page capacity: {runtime_variant!r}"
            )
        forward_batch.runtime_sparse_page_capacity = runtime_variant

    def prepare_cuda_graph_capture(self, forward_batch, runtime_variant) -> None:
        self._publish_variant(forward_batch, runtime_variant)

    def prepare_cuda_graph_replay(self, forward_batch, runtime_variant) -> None:
        self._publish_variant(forward_batch, runtime_variant)

    def prepare_cuda_graph_capture_context(self, context):
        if not dataclasses.is_dataclass(context):
            return context
        field_names = {field.name for field in dataclasses.fields(context)}
        if "runtime_sparse_coordinator" not in field_names:
            return context
        return dataclasses.replace(context, runtime_sparse_coordinator=self.coordinator)


def create_cuda_graph_runtime_provider(
    coordinator: Any,
) -> Optional[SparseCudaGraphRuntimeProvider]:
    if not is_runtime_sparse_cuda_graph_available(coordinator):
        return None
    return SparseCudaGraphRuntimeProvider(coordinator)


def is_runtime_sparse_cuda_graph_available(coordinator: Any) -> bool:
    config = coordinator.config
    enabled = config.sparse_extra_config.get("enable_cuda_graph_retrieval", True)
    if not isinstance(enabled, bool):
        raise ValueError("enable_cuda_graph_retrieval must be a boolean")
    algorithm_cls = type(coordinator.algorithm)
    if not enabled or not getattr(
        algorithm_cls, "supports_fixed_cuda_graph_capacity", False
    ):
        return False
    if config.backend not in ("fa3", "flashattention"):
        return False
    if torch.device(coordinator.device).type != "cuda" or torch.version.hip is not None:
        return False
    return True
