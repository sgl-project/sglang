"""Bounded CPU feature hashing for the tokenizer's event loop."""

import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import MultimodalDataItem


def _is_cpu_feature(value: object) -> bool:
    # Unknown transports stay on the caller thread, where their device context
    # and lifetime are established. In particular, do not reconstruct CUDA IPC.
    if isinstance(value, torch.Tensor):
        return value.device.type == "cpu"
    if isinstance(value, np.ndarray):
        return True
    if isinstance(value, list):
        return bool(value) and all(_is_cpu_feature(part) for part in value)
    return False


def _set_pad_values(items: list[MultimodalDataItem]) -> None:
    for item in items:
        item.set_pad_value()


class MultimodalHashExecutor:
    """Keep at most ``max_workers`` native hash jobs in flight per processor.

    Admission happens before submission, so the thread pool has no unbounded
    work queue. A cancelled caller retains its inputs and admission slot until
    the native reader stops, including when cancellation is repeated.
    """

    def __init__(self, *, max_workers: int = 2):
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="sglang-mm-hash"
        )
        self._slots = asyncio.Semaphore(max_workers)

    async def set_pad_values(self, items: list[MultimodalDataItem]) -> None:
        cpu_items = []
        for item in items:
            if not isinstance(item, MultimodalDataItem):
                continue
            feature = (
                item.feature
                if item.feature is not None
                else item.precomputed_embeddings
            )
            if (
                item.pad_value is None
                and item.hash is None
                and not envs.SGLANG_MM_SKIP_COMPUTE_HASH.get()
                and _is_cpu_feature(feature)
            ):
                cpu_items.append(item)
            else:
                item.set_pad_value()

        if not cpu_items:
            return
        async with self._slots:
            future = asyncio.wrap_future(
                self._executor.submit(_set_pad_values, cpu_items)
            )
            cancelled = False
            while not future.done():
                try:
                    await asyncio.shield(future)
                except asyncio.CancelledError:
                    cancelled = True
                except Exception:
                    break
            if cancelled:
                # Retrieve a worker exception as well; cancellation still wins.
                if not future.cancelled():
                    future.exception()
                raise asyncio.CancelledError
            future.result()

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)
