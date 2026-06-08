# SPDX-License-Identifier: Apache-2.0
"""Block KV cache for OmniDreams autoregressive rollout.

Faithful port of the FlashDreams production ``BlockKVCache``
(``flashdreams/core/attention/kvcache.py``): causal attention with a fixed-size
local window plus optional sink tokens.

Layout along ``seq_dim``: ``[sink tokens | local window tokens]``. Sink tokens
are never evicted; the local window rolls left by ``chunk_size`` once full.
Cross-chunk causality in OmniDreams comes ONLY from this window (the attention
op itself is full bidirectional SDPA with no causal mask).

Differences from the FlashDreams original:
- ``torch.sym_min`` / ``torch.sym_max`` are replaced with the plain builtins.
  The bounds (``_n_cached``, ``total_size``, ``chunk_size``) are Python ints, so
  the two are eager-equivalent; the ``sym_*`` variants only matter for symbolic
  CUDA-graph tracing, and plain ``min``/``max`` keep this importable + testable
  on CPU without a CUDA build.

Per-step usage::

    cache.before_update(chunk_idx)   # roll window if steady-state
    cache.update(k, v)               # write this chunk's K/V
    k_all, v_all = cache.cached_k(), cache.cached_v()
    cache.after_update(chunk_idx)    # bookkeeping
"""

from dataclasses import dataclass, field

import torch
from torch import Tensor
from typing_extensions import Self


@dataclass
class BlockKVCache:
    """KV cache for causal attention with a fixed-size local window + sink.

    Keys/values may have arbitrary shape ``[..., total_size, ...]``; the rolling
    (sequence) dimension is ``seq_dim`` (may be negative). ``total_size`` equals
    ``sink_size + window_size`` and must be divisible by ``chunk_size``. Chunks
    are non-overlapping: each update appends one ``chunk_size``-token chunk at the
    next logical position.

    ``chunk_idx`` (0, 1, 2, ...) is the chunk's index in the full sequence, not a
    cache offset. A ``chunk_idx`` of ``prev + 1`` appends (or, in steady-state,
    writes after a left-roll); a ``chunk_idx`` equal to ``prev`` overwrites the
    same positions (used to refresh K/V after a renoise/finalize pass).
    """

    k_shape: tuple[int, ...]
    v_shape: tuple[int, ...]
    seq_dim: int
    chunk_size: int
    window_size: int
    sink_size: int = 0
    device: torch.device | str = torch.device("cpu")
    dtype: torch.dtype = torch.float32

    _prev_chunk_idx: int = -1
    _curr_chunk_idx: int | None = None
    _n_cached: int = 0

    _k: Tensor = field(init=False)
    _v: Tensor = field(init=False)

    # ----- properties ------------------------------------------------------- #
    @property
    def size(self) -> int:
        """Number of valid cached tokens visible to attention."""
        if self._curr_chunk_idx is None:
            return self._n_cached
        return self._visible_end()

    @property
    def write_end(self) -> int:
        """Right edge of the current chunk in the physical cache layout."""
        assert (
            self._curr_chunk_idx is not None
        ), "Must call before_update() before write_end"
        return self.size

    # ----- construction ----------------------------------------------------- #
    @classmethod
    def from_tensor(cls, k: Tensor, v: Tensor, seq_dim: int) -> Self:
        """Build a single-chunk cache pre-filled with the given K/V tensors."""
        cache = cls(
            k_shape=tuple(k.shape),
            v_shape=tuple(v.shape),
            seq_dim=seq_dim,
            chunk_size=k.shape[seq_dim],
            window_size=k.shape[seq_dim],
            device=k.device,
            dtype=k.dtype,
        )
        cache.before_update(0)
        cache.update(k, v)
        cache.after_update(0)
        cache._curr_chunk_idx = 0
        return cache

    def __post_init__(self) -> None:
        assert (
            self.k_shape[:-1] == self.v_shape[:-1]
        ), "k and v must have the same shape except for the last dimension"

        tensor_dim = len(self.k_shape)
        assert (
            -tensor_dim <= self.seq_dim < tensor_dim
        ), f"seq_dim must be in [-{tensor_dim}, {tensor_dim}), got {self.seq_dim}"
        # Normalize seq_dim to a non-negative index.
        self.seq_dim = self.seq_dim if self.seq_dim >= 0 else self.seq_dim + tensor_dim

        assert self.sink_size >= 0, "sink_size must be non-negative"

        expected_length = self.sink_size + self.window_size
        assert self.k_shape[self.seq_dim] == expected_length, (
            f"k_shape[seq_dim] ({self.k_shape[self.seq_dim]}) must equal "
            f"sink_size + window_size ({expected_length})"
        )
        assert (self.window_size + self.sink_size) % self.chunk_size == 0, (
            f"window_size + sink_size ({self.window_size + self.sink_size}) must be "
            f"divisible by chunk_size ({self.chunk_size})"
        )

        self._k = torch.empty(self.k_shape, device=self.device, dtype=self.dtype)
        self._v = torch.empty(self.v_shape, device=self.device, dtype=self.dtype)

    # ----- internal helpers ------------------------------------------------- #
    def _seq_slice(self, start: int | None, end: int | None) -> tuple[slice | int, ...]:
        """Index tuple selecting ``[start:end]`` on ``seq_dim``, all else full."""
        idx: list[slice | int] = [slice(None)] * len(self.k_shape)
        idx[self.seq_dim] = slice(start, end)
        return tuple(idx)

    def _roll_local_window_left(self) -> None:
        """Shift the local window left by ``chunk_size`` (steady-state only)."""
        total_size = self._k.shape[self.seq_dim]
        assert (
            total_size == self._n_cached
        ), f"Expected full cache: {total_size=} != {self._n_cached=}"
        tokens_to_keep = self.window_size - self.chunk_size
        if tokens_to_keep > 0:
            src_start = self.sink_size + self.chunk_size
            src_end = total_size
            dst_start = self.sink_size
            dst_end = self.sink_size + tokens_to_keep
            dst_slice = self._seq_slice(dst_start, dst_end)
            src_slice = self._seq_slice(src_start, src_end)
            self._k[dst_slice] = self._k[src_slice].clone()
            self._v[dst_slice] = self._v[src_slice].clone()

    def _current_chunk_overlaps_sink(self) -> bool:
        assert (
            self._curr_chunk_idx is not None
        ), "Must call before_update() before checking sink overlap"
        return (
            self.sink_size > 0
            and self._curr_chunk_idx * self.chunk_size < self.sink_size
        )

    def _current_write_bounds(self) -> tuple[int, int]:
        """Physical cache range written by the current update."""
        assert (
            self._curr_chunk_idx is not None
        ), "Must call before_update() before computing write bounds"
        total_size = self._k.shape[self.seq_dim]
        assert (
            self.chunk_size <= total_size
        ), f"chunk_size ({self.chunk_size}) must be <= cache size ({total_size})"
        if self._curr_chunk_idx == self._prev_chunk_idx + 1:
            write_start = min(self._n_cached, total_size - self.chunk_size)
            write_end = write_start + self.chunk_size
        elif self._curr_chunk_idx == self._prev_chunk_idx:
            write_end = min(self._n_cached, total_size)
            write_start = max(write_end - self.chunk_size, 0)
        else:
            raise ValueError(
                f"{self._curr_chunk_idx=} should be either "
                f"{self._prev_chunk_idx + 1} or {self._prev_chunk_idx}."
            )
        return write_start, write_end

    def _write_current_chunk(self, k: Tensor, v: Tensor) -> None:
        """Write the current chunk through a filling/steady-compatible path."""
        write_start, write_end = self._current_write_bounds()
        read_start = 0
        read_end = write_end - write_start

        if (
            self.sink_size > 0
            and not self._current_chunk_overlaps_sink()
            and write_start < self.sink_size
        ):
            write_start = self.sink_size
            keep_size = write_end - write_start
            read_end = self.chunk_size
            read_start = read_end - keep_size

        sl_read = self._seq_slice(read_start, read_end)
        sl_write = self._seq_slice(write_start, write_end)
        self._k[sl_write] = k[sl_read]
        self._v[sl_write] = v[sl_read]

    def _visible_end(self) -> int:
        """Right edge of cached tokens visible to attention this update."""
        assert (
            self._curr_chunk_idx is not None
        ), "Must call before_update() before computing visible cache size"
        total_size = self._k.shape[self.seq_dim]
        if self._curr_chunk_idx == self._prev_chunk_idx + 1:
            return min(self._n_cached + self.chunk_size, total_size)
        if self._curr_chunk_idx == self._prev_chunk_idx:
            return min(self._n_cached, total_size)
        raise ValueError(
            f"{self._curr_chunk_idx=} should be either "
            f"{self._prev_chunk_idx + 1} or {self._prev_chunk_idx}."
        )

    # ----- public lifecycle ------------------------------------------------- #
    def is_steady_state(self) -> bool:
        """True if the cache is full (steady-state phase)."""
        assert (
            self._curr_chunk_idx is not None
        ), "Must call before_update() before is_steady_state()"
        total_size = self._k.shape[self.seq_dim]
        is_full = total_size == self._n_cached
        is_overlapping_with_sink = (
            self.sink_size > 0
            and self._curr_chunk_idx * self.chunk_size < self.sink_size
        )
        return is_full and not is_overlapping_with_sink

    def before_update(self, chunk_idx: int) -> None:
        """Prepare the cache before writing (roll window if steady-state)."""
        assert (
            self._curr_chunk_idx is None
        ), "Must call after_update() before before_update()"
        self._curr_chunk_idx = chunk_idx
        if chunk_idx == self._prev_chunk_idx:
            return
        assert chunk_idx == self._prev_chunk_idx + 1, (
            "Expected the new chunk_idx to be +1 from the previous chunk_idx, "
            f"got {chunk_idx} != {self._prev_chunk_idx} + 1"
        )
        if self.is_steady_state():
            self._roll_local_window_left()

    def update(self, k: Tensor, v: Tensor) -> None:
        """Write the new chunk's K/V into the cache."""
        assert (
            self._curr_chunk_idx is not None
        ), "Must call before_update() before update()"
        chunk_size_k = k.shape[self.seq_dim]
        chunk_size_v = v.shape[self.seq_dim]
        assert chunk_size_k == self.chunk_size, (
            f"Expected input k chunk_size {self.chunk_size} at seq_dim "
            f"{self.seq_dim}, got {chunk_size_k}"
        )
        assert chunk_size_v == self.chunk_size, (
            f"Expected input v chunk_size {self.chunk_size} at seq_dim "
            f"{self.seq_dim}, got {chunk_size_v}"
        )
        self._write_current_chunk(k, v)

    def after_update(self, chunk_idx: int) -> None:
        """Finalize bookkeeping after writing the chunk."""
        assert (
            chunk_idx == self._curr_chunk_idx
        ), f"Expected chunk_idx to be {self._curr_chunk_idx}, got {chunk_idx}"
        if self._curr_chunk_idx == self._prev_chunk_idx + 1:
            if not self.is_steady_state():
                self._n_cached += self.chunk_size
            self._prev_chunk_idx += 1
        elif self._curr_chunk_idx == self._prev_chunk_idx:
            pass
        else:
            raise ValueError(
                f"{self._curr_chunk_idx=} should be either "
                f"{self._prev_chunk_idx + 1} or {self._prev_chunk_idx}."
            )
        self._curr_chunk_idx = None

    def cached_k(self) -> Tensor:
        """Cached keys (valid prefix while filling, full buffer in steady-state)."""
        return self._k[self._seq_slice(0, self.size)]

    def cached_v(self) -> Tensor:
        """Cached values (valid prefix while filling, full buffer in steady-state)."""
        return self._v[self._seq_slice(0, self.size)]

    def reset(self) -> None:
        """Reset the cache to its initial empty state."""
        self._prev_chunk_idx = -1
        self._n_cached = 0
        self._curr_chunk_idx = None
