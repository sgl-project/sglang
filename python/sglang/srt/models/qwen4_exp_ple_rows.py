"""Page-cache row reads and host hashing for Qwen4 PLE."""

from typing import Protocol

import numpy as np

from sglang.srt.environ import envs
from sglang.srt.models.qwen4_exp_ple_table import PleFilePageHints


class PleRowSource(Protocol):
    row_bytes: int

    def fetch_rows(self, local_ids: np.ndarray, out: np.ndarray) -> None: ...

    def prefetch_rows(self, local_ids: np.ndarray) -> None: ...

    def close(self) -> None: ...


class PageCacheRowSource:
    """Read shard-local rows from the loader-owned mapping."""

    def __init__(self, path: str, row_bytes: int, table: np.ndarray):
        self.row_bytes = row_bytes
        self._closed = False
        self._prefetcher = (
            PleFilePageHints(path, row_bytes)
            if envs.SGLANG_QWEN4_PLE_FILE_PREFETCH.get()
            else None
        )
        self._table = table

    def prefetch_rows(self, local_ids: np.ndarray) -> None:
        if self._prefetcher is not None:
            self._prefetcher.advise_rows(local_ids)

    def fetch_rows(self, local_ids: np.ndarray, out: np.ndarray) -> None:
        ids = np.asarray(local_ids)
        if ids.size and (ids.min() < 0 or ids.max() >= len(self._table)):
            raise IndexError("row outside the local vocabulary shard")
        # IDs are bounds-checked above; "clip" avoids the buffered "raise" path.
        np.take(self._table, ids, axis=0, out=out, mode="clip")

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            if self._prefetcher is not None:
                self._prefetcher.close()


def hash_contexts_numpy(
    contexts: np.ndarray,
    multipliers: np.ndarray,
    vocab_sizes: np.ndarray,
    offsets: np.ndarray,
    eos_token_id: int,
) -> np.ndarray:
    """Match the signed int64 arithmetic and EOS segments of _hash_contexts."""
    contexts = np.asarray(contexts, dtype=np.int64)
    size = len(multipliers)
    heads = len(vocab_sizes) // (size - 1)
    blocks = []
    past_eos = np.zeros(len(contexts), dtype=bool)
    mixed = contexts[:, -1] * multipliers[0]
    for shift in range(1, size):
        token = contexts[:, -1 - shift]
        past_eos |= token == eos_token_id
        token = np.where(past_eos, eos_token_id, token)
        mixed = np.bitwise_xor(mixed, token * multipliers[shift])
        begin = (shift - 1) * heads
        end = begin + heads
        blocks.append(mixed[:, None] % vocab_sizes[begin:end] + offsets[begin:end])
    return np.concatenate(blocks, axis=1)
