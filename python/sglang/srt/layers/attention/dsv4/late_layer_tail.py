"""DeepSeek-V4.1 decoder SWA bounded replay: tail rows and their metadata.

Device-neutral. The CUDA and HIP attention backends import Triton at module
scope; NPU backends import this module instead to share the same row layout.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import msgspec
import torch

from sglang.srt.layers.cp.interleave import InterleaveContextParallelMetadata
from sglang.srt.model_executor.forward_mode import CaptureHiddenMode
from sglang.srt.utils.device import async_h2d

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

# DeepSeek-V4.1 late layers attend through a sliding window of this width.
SWA_WINDOW = 128


class LateLayerTail(msgspec.Struct, frozen=True):
    """Rows the layers from late_layer_start on run over under decoder SWA bounded
    replay: the last tail tokens of each request in the extend."""

    token_indices: torch.Tensor
    positions: torch.Tensor
    extend_seq_lens: torch.Tensor
    extend_seq_lens_cpu: List[int]
    swa_out_cache_loc: torch.Tensor
    # Set when the tail is the extend's last rows (one request): a view, not a gather.
    contiguous_start: Optional[int] = None
    # prefill CP: this rank's tail rows padded to the largest share; cp_metadata is that layout
    pad_rows: int = 0
    cp_metadata: Optional[InterleaveContextParallelMetadata] = None
    local_lens_cpu: Optional[List[int]] = None
    # Original extend rows in global request order; token_indices is rank-local under CP.
    global_token_indices: Optional[torch.Tensor] = None
    req_global: Optional[torch.Tensor] = None
    pos_global: Optional[torch.Tensor] = None
    # Prefill CUDA graph: token_indices/positions are persistent buffers of a fixed
    # capacity (real rows first, padding after); scatter_indices maps every row back
    # into the extend, padding rows to a dummy row past the end.
    scatter_indices: Optional[torch.Tensor] = None
    real_rows_count: Optional[int] = None

    def rows(self, t: torch.Tensor) -> torch.Tensor:
        rows = self.real_rows(t)
        if self.pad_rows:
            rows = torch.cat([rows, rows.new_zeros((self.pad_rows, *rows.shape[1:]))])
        return rows

    def real_rows(self, t: torch.Tensor) -> torch.Tensor:
        return _tail_rows(
            t, token_indices=self.token_indices, contiguous_start=self.contiguous_start
        )


def _tail_rows(
    t: torch.Tensor, *, token_indices: torch.Tensor, contiguous_start: Optional[int]
) -> torch.Tensor:
    if contiguous_start is not None:
        # Same rows as t[token_indices]; t may carry MLP-sync padding past the extend.
        return t[contiguous_start : contiguous_start + token_indices.shape[0]]
    return t[token_indices]


def _late_layer_whole_extend(
    *,
    extend_lens: List[int],
    logprob_start_lens: Optional[List[int]],
    full_hidden_states: bool,
) -> Optional[List[bool]]:
    """Requests that run their whole extend with plain windows under decoder SWA
    bounded replay: those that read prompt rows (prompt logprobs, or every hidden
    state outside DSpark, whose draft reads tail rows only). None when no request
    in the batch does."""
    if full_hidden_states:
        return [True] * len(extend_lens)
    if logprob_start_lens is None:
        return None
    whole = [start < n for start, n in zip(logprob_start_lens, extend_lens)]
    return whole if any(whole) else None


def late_layer_whole_extend(
    forward_batch: ForwardBatch, *, dspark: bool
) -> Optional[List[bool]]:
    """`_late_layer_whole_extend` of a forward batch. dspark comes from the server:
    a capture batch carries DSpark's FULL capture mode without its spec_algorithm."""
    return _late_layer_whole_extend(
        extend_lens=forward_batch.extend_seq_lens_cpu,
        logprob_start_lens=(
            forward_batch.extend_logprob_start_lens_cpu
            if forward_batch.return_logprob
            else None
        ),
        full_hidden_states=forward_batch.capture_hidden_mode == CaptureHiddenMode.FULL
        and not dspark,
    )


def late_layer_num_tokens(forward_batch: ForwardBatch, *, dspark: bool) -> int:
    """Rows a rank runs from late_layer_start on: an extend's tail rows, as
    late_layer_tail_layout picks them, and every row otherwise."""
    if not forward_batch.forward_mode.is_extend_without_speculative():
        return forward_batch.input_ids.shape[0]
    whole_extend = late_layer_whole_extend(forward_batch, dspark=dspark)
    return sum(
        late_layer_tail_lens(
            forward_batch.extend_seq_lens_cpu, SWA_WINDOW, whole_extend
        )
    )


def _pad_rows(t: torch.Tensor, *, num_rows: int) -> torch.Tensor:
    assert t.shape[0] <= num_rows, (t.shape[0], num_rows)
    if t.shape[0] == num_rows:
        return t
    return torch.nn.functional.pad(t, (0, num_rows - t.shape[0]))


def late_layer_tail_lens(
    extend_lens_cpu: list[int],
    tail_len: int,
    whole_extend: Optional[list[bool]] = None,
) -> list[int]:
    """Each request's late-layer tail: its last min(tail_len, extend_len) tokens,
    or its whole extend where whole_extend is set."""
    if whole_extend is None:
        return [min(tail_len, n) for n in extend_lens_cpu]
    return [
        n if whole else min(tail_len, n)
        for n, whole in zip(extend_lens_cpu, whole_extend)
    ]


def late_layer_tail_layout(
    *,
    extend_lens_cpu: list[int],
    seq_lens_cpu: list[int],
    tail_len: int,
    device: torch.device,
    whole_extend: Optional[list[bool]] = None,
) -> tuple[torch.Tensor, list[int], torch.Tensor]:
    """Tail rows of each prefill extend: its last min(tail_len, extend_len) tokens,
    or its whole extend where whole_extend is set.
    Returns (token indices into the extend, per-request tail lengths, per-row
    absolute window floor). The floor is seq_len - tail_len, so a tail shorter
    than tail_len reads rows the request wrote before this extend. A whole
    extend keeps every row's plain window."""
    if whole_extend is not None and any(whole_extend):
        return _whole_extend_tail_layout(
            extend_lens_cpu=extend_lens_cpu,
            seq_lens_cpu=seq_lens_cpu,
            tail_len=tail_len,
            device=device,
            whole_extend=whole_extend,
        )
    tail_lens_cpu = late_layer_tail_lens(extend_lens_cpu, tail_len)
    if len(extend_lens_cpu) == 1:
        n, t, s = extend_lens_cpu[0], tail_lens_cpu[0], seq_lens_cpu[0]
        floor = torch.full((t,), max(0, s - tail_len), dtype=torch.int32, device=device)
        return torch.arange(n - t, n, device=device), tail_lens_cpu, floor
    # One H2D copy for the three length vectors; launch count does not grow with bs.
    lens = async_h2d(
        [extend_lens_cpu, tail_lens_cpu, seq_lens_cpu], dtype=torch.int64, device=device
    )
    extend_lens, tail_lens, seq_lens = lens[0], lens[1], lens[2]
    total = sum(tail_lens_cpu)
    req = torch.repeat_interleave(
        torch.arange(len(tail_lens_cpu), device=device), tail_lens, output_size=total
    )
    offs = (
        torch.arange(total, device=device)
        - (torch.cumsum(tail_lens, 0) - tail_lens)[req]
    )
    token_indices = (torch.cumsum(extend_lens, 0) - tail_lens)[req] + offs
    floor = (seq_lens - tail_len).clamp_min(0)[req].to(torch.int32)
    return token_indices, tail_lens_cpu, floor


def _whole_extend_tail_layout(
    *,
    extend_lens_cpu: list[int],
    seq_lens_cpu: list[int],
    tail_len: int,
    device: torch.device,
    whole_extend: list[bool],
) -> tuple[torch.Tensor, list[int], torch.Tensor]:
    tail_lens_cpu = late_layer_tail_lens(extend_lens_cpu, tail_len, whole_extend)
    token_indices, floor, start = [], [], 0
    for n, t, s, whole in zip(
        extend_lens_cpu, tail_lens_cpu, seq_lens_cpu, whole_extend
    ):
        token_indices.append(torch.arange(start + n - t, start + n))
        # A whole extend's rows sit at s - t or later, so its floor never cuts
        # their plain window.
        row_floor = max(0, s - t - tail_len) if whole else max(0, s - tail_len)
        floor.append(torch.full((t,), row_floor, dtype=torch.int32))
        start += n
    return (
        torch.cat(token_indices).to(device, non_blocking=True),
        tail_lens_cpu,
        torch.cat(floor).to(device, non_blocking=True),
    )
