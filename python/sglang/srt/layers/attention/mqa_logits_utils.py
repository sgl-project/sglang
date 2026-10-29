"""Memory budget for the fp32 MQA-logits scratch of the DeepSeek sparse
attention indexers (DSA, DSV4, DSV4.1).

The indexer scores every query row against every key column into one fp32
logits matrix that no pool sized by mem_fraction_static accounts for. These
helpers bound that matrix and slice it by query rows.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Callable, Iterator, Optional, Set, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_schedule
from sglang.srt.utils import get_device_module, is_hip, is_xpu
from sglang.srt.utils.common import ceil_div

logger = logging.getLogger(__name__)

MQA_LOGITS_BYTES_PER_ELEM = 4
MQA_LOGITS_STATIC_SKIP_ELEMS = 8_000_000
MQA_LOGITS_TOTAL_MEM_FRACTION = 0.3
# aiter's fp8_mqa_logits only compiles below 2 GiB of logits (buffer_store).
MQA_LOGITS_MAX_BYTES_ROCM = 2**31 - 1
# DeepGEMM pads the logits row stride to 1024 bytes, i.e. 256 fp32 columns.
MQA_LOGITS_ROW_ALIGN_ELEMS = 256


def mqa_logits_free_mem_fraction() -> float:
    return envs.SGLANG_DSA_MQA_LOGITS_FREE_MEM_FRACTION.get()


def mqa_logits_needs_budget_check(*, num_rows: int, num_cols: int) -> bool:
    return num_rows * num_cols >= MQA_LOGITS_STATIC_SKIP_ELEMS


def mqa_logits_row_bytes(num_cols: int) -> int:
    aligned_cols = (
        ceil_div(num_cols, MQA_LOGITS_ROW_ALIGN_ELEMS) * MQA_LOGITS_ROW_ALIGN_ELEMS
    )
    return aligned_cols * MQA_LOGITS_BYTES_PER_ELEM


def mqa_logits_static_budget_bytes(*, device_index: int) -> int:
    """Budget from configuration alone (no device query); safe during graph capture."""
    total_mem = get_device_module().get_device_properties(device_index).total_memory
    total_mem_budget = int(total_mem * MQA_LOGITS_TOTAL_MEM_FRACTION)
    mem_fraction_static = get_schedule().mem_fraction_static
    if mem_fraction_static is None:
        budget = total_mem_budget
    else:
        static_free_mem = int(total_mem * max(0.0, 1.0 - mem_fraction_static))
        budget = min(
            int(static_free_mem * mqa_logits_free_mem_fraction()), total_mem_budget
        )
    return max(1, budget)


def mqa_logits_budget_bytes(*, device_index: int, allow_sync: bool) -> int:
    """Static budget capped by current free memory; mem_get_info syncs, so
    callers pass allow_sync=False under CUDA graph capture."""
    budget = mqa_logits_static_budget_bytes(device_index=device_index)
    if allow_sync and not is_xpu():
        free_mem, _ = torch.cuda.mem_get_info(device_index)
        budget = min(int(free_mem * mqa_logits_free_mem_fraction()), budget)
    if is_hip():
        budget = min(budget, MQA_LOGITS_MAX_BYTES_ROCM)
    return max(1, budget)


def mqa_logits_should_chunk(
    *,
    num_rows: int,
    num_cols: int,
    get_budget_bytes: Callable[[], int],
    rocm: bool,
) -> Tuple[bool, int]:
    """Whether a [num_rows, num_cols] fp32 logits matrix must be row-chunked.

    Returns (need_chunk, effective_budget_bytes). Matrices below
    MQA_LOGITS_STATIC_SKIP_ELEMS never chunk and the budget is not queried
    (it may synchronize the host), hence the callable. On ROCm the budget is
    also capped by aiter's logits ceiling.
    """
    if not mqa_logits_needs_budget_check(num_rows=num_rows, num_cols=num_cols):
        return False, 0
    budget_bytes = get_budget_bytes()
    if rocm:
        budget_bytes = min(budget_bytes, MQA_LOGITS_MAX_BYTES_ROCM)
    logits_bytes = num_rows * num_cols * MQA_LOGITS_BYTES_PER_ELEM
    return logits_bytes > budget_bytes, budget_bytes


def mqa_logits_rows_per_chunk(
    *, num_rows: int, row_bytes: int, budget_bytes: int
) -> Optional[int]:
    """Query rows per chunk so one logits chunk fits the budget; None if all rows fit."""
    if num_rows * row_bytes <= budget_bytes:
        return None
    rows = max(budget_bytes // max(row_bytes, 1), 1)
    return int(rows) if rows < num_rows else None


class MqaLogitsPool:
    """A fixed amount of device memory held for logits tiles and their scratch.

    A budget read from free memory shrinks as the caching allocator holds more of
    the device, which a long-running server always ends up doing, and then the
    tiles collapse to a few rows. This pool instead takes ``capacity_bytes`` once,
    on the first tile of each stream, into a private allocator pool. The caching
    allocator keeps a live private pool's segments out of every other allocation
    and out of ``empty_cache``, so later tiles of at most ``capacity_bytes`` reuse
    them whatever the rest of the server holds. Size tiles by ``capacity_bytes``.
    """

    def __init__(self, *, capacity_bytes: int, device: torch.device):
        self.capacity_bytes = max(1, int(capacity_bytes))
        self.device = device
        self._pool: Optional[torch.cuda.MemPool] = None
        # Cached blocks serve only the stream they were freed on.
        self._reserved_streams: Set[int] = set()

    @contextlib.contextmanager
    def allocating(self) -> Iterator[None]:
        """Route this thread's allocations on the device into the pool. Not for
        graph capture: captured tensors belong to the graph's own pool."""
        if self._pool is None:
            self._pool = torch.cuda.MemPool()
        with torch.cuda.use_mem_pool(self._pool, self.device):
            stream = torch.cuda.current_stream(self.device).cuda_stream
            if stream not in self._reserved_streams:
                self._reserve(stream)
                self._reserved_streams.add(stream)
            yield

    def _reserve(self, stream: int) -> None:
        try:
            held = torch.empty(
                self.capacity_bytes, dtype=torch.uint8, device=self.device
            )
        except torch.OutOfMemoryError as e:
            raise RuntimeError(
                f"Could not reserve {self.capacity_bytes / (1 << 30):.2f} GiB for "
                "the indexer's logits tiles; lower --mem-fraction-static or "
                "SGLANG_DSA_MQA_LOGITS_FREE_MEM_FRACTION"
            ) from e
        del held
        logger.info(
            "Reserved %.2f GiB for indexer logits tiles on %s, stream %#x",
            self.capacity_bytes / (1 << 30),
            self.device,
            stream,
        )
