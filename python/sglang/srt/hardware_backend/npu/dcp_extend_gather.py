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
# ==============================================================================

"""NPU DSA extend under DCP that gathers the prefix KV, the GPU MLA scheme.

Pure index math and buffers, no torch_npu, so CPU tests can import it.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, NamedTuple, Optional, Sequence, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.utils import dsa_use_prefill_cp
from sglang.srt.layers.dcp.layout import dcp_local_row, get_dcp_lens
from sglang.srt.layers.dcp.metadata import DecodeContextParallelMetadata
from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


class DcpExtendGatherPiece(NamedTuple):
    """One collective of the extend gather. The all-gather lays the sends out
    rank-major, this chunk's own KV follows, and ``index`` maps output rows
    ``[out_start, out_end)`` onto that scratch."""

    send_start: int
    send_end: int
    extend_start: int
    extend_end: int
    out_start: int
    out_end: int
    index: torch.Tensor
    scratch_rows: int


class DcpExtendGatherPlan(NamedTuple):
    """How one rank gathers a batch's prefix KV at extend. Sends are padded
    to ``padded_lens`` so every rank's is ``send_rows`` long; the output is each
    request's prefix then its extend, in order, tiled by ``pieces``."""

    local_lens: List[int]
    padded_lens: List[int]
    send_rows: int
    scratch_rows: int
    pieces: List[DcpExtendGatherPiece]


# A dataclass, not a msgspec.Struct, because its base is one.
@dataclass
class NPUDecodeContextParallelMetadata(DecodeContextParallelMetadata):
    # Set once per forward by build_dcp_extend_gather_plan and only read by the
    # layers. None means this forward takes the query gather instead.
    extend_gather_plan: Optional[DcpExtendGatherPlan] = None
    # The current layer's gathered (latent, rope) KV. NPU keeps the two apart
    # in the pool and in npu_sparse_flash_attention, so not dcp_kv_buffer.
    extend_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None


# Above this many new tokens a DSA extend gathers the KV whatever the bytes say.
# The query gather's fp32 ``[out, lse]`` exchange grows with the new tokens and
# is held several times over: a 16k-token batch of three 1M-prefix requests ran
# out of memory there on A3, while 4,096-token chunks run on main. Memory-driven,
# so a deployment can move it.
DCP_EXTEND_QUERY_GATHER_MAX_TOKENS = (
    envs.SGLANG_NPU_DCP_EXTEND_QUERY_GATHER_MAX_TOKENS.get()
)


def dcp_extend_gathers_kv(
    prefix_lens: Sequence[int],
    extend_lens: Sequence[int],
    num_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    kv_elem_size: int,
) -> bool:
    """Should this DSA extend gather the prefix KV rather than the query?

    Gathering the query costs, per query token, every head's query plus the
    fp32 ``[out, lse]`` merge; gathering the KV costs one latent row per prefix
    token, whatever the head count. Both scale by ``(dcp - 1) / dcp``, so it
    cancels. For GLM-5.2 (64 heads, 512 + 64 bf16) the crossover is a tail of
    prefix / 178. Ties go to the KV gather, so an extend with no prefix -- which
    gathers nothing -- takes it. A batch counts each request's prefix, shared or
    not, because the KV gather sends each one.

    Bytes are not the only cost: past ``DCP_EXTEND_QUERY_GATHER_MAX_TOKENS``
    the query gather's memory decides, and the KV gather is taken.
    """
    if sum(extend_lens) > DCP_EXTEND_QUERY_GATHER_MAX_TOKENS:
        return True
    row = (kv_lora_rank + qk_rope_head_dim) * kv_elem_size
    per_query_token = num_heads * (row + (kv_lora_rank + 1) * 4)
    return sum(prefix_lens) * row <= sum(extend_lens) * per_query_token


# Reusable device buffers for the extend gather, keyed by purpose, dtype, device
# and row shape. See dcp_extend_gather_buffer.
_dcp_extend_gather_buffers: Dict[Tuple, torch.Tensor] = {}


def dcp_extend_gather_buffer(name: str, ref: torch.Tensor, rows: int) -> torch.Tensor:
    """A reusable ``rows``-row buffer shaped and typed like ``ref``. Grow-only.

    Allocated per layer, the peak moved with whatever the allocator held when a
    layer asked for another ~1.1 GiB. Two callers sharing ``name`` share the
    buffer and must not hold slices across each other's calls.
    """
    key = (name, ref.dtype, ref.device, tuple(ref.shape[1:]))
    buf = _dcp_extend_gather_buffers.get(key)
    if buf is None or buf.shape[0] < rows:
        # Drop every reference first, or the peak is briefly both buffers.
        _dcp_extend_gather_buffers.pop(key, None)
        buf = None
        buf = torch.empty((rows, *ref.shape[1:]), dtype=ref.dtype, device=ref.device)
        _dcp_extend_gather_buffers[key] = buf
        logger.info(
            "DCP extend gather buffer %r reserved: %d rows, %.3f GiB",
            name,
            rows,
            buf.numel() * buf.element_size() / (1 << 30),
        )
    return buf[:rows]


_piece_rows = envs.SGLANG_NPU_DCP_EXTEND_GATHER_PIECE_ROWS.get()
DCP_EXTEND_GATHER_PIECE_ROWS = _piece_rows if _piece_rows > 0 else 1 << 62


def build_dcp_extend_gather_plan(
    forward_batch: "ForwardBatch",
    interleave_size: int,
    device: torch.device,
) -> Optional[DcpExtendGatherPlan]:
    """This rank's plan if this forward gathers the prefix KV, else None.

    The one place that decides per forward; the layers only test the result,
    so these checks do not repeat on every layer. Indices land on ``device``.
    """
    mode = forward_batch.forward_mode
    # Target verify and DSA prefill CP keep their own attention paths.
    if (
        not mode.is_extend()
        or mode.is_target_verify()
        or dsa_use_prefill_cp(forward_batch)
    ):
        return None
    parallel = get_parallel()
    plan = plan_dcp_extend_gather(
        forward_batch.extend_prefix_lens_cpu,
        forward_batch.extend_seq_lens_cpu,
        parallel.dcp_size,
        parallel.dcp_rank,
        DCP_EXTEND_GATHER_PIECE_ROWS,
        interleave_size,
    )
    return plan._replace(
        pieces=[piece._replace(index=piece.index.to(device)) for piece in plan.pieces]
    )


def plan_dcp_extend_gather(
    prefix_lens: Sequence[int],
    extend_lens: Sequence[int],
    dcp_size: int,
    dcp_rank: int,
    max_piece_gather_rows: int,
    interleave_size: int = 1,
) -> DcpExtendGatherPlan:
    """Plan the extend-time gather of the prefix KV into position order.

    Position p lives on rank ``(p // B) % dcp_size`` at local row
    ``dcp_local_row(p)``, ``B = interleave_size``. Local shards are
    concatenated per request, as ``dcp_local_prefix_kv_indices`` lists them.

    Pieces gather at most ``max_piece_gather_rows`` rows each, and cut on whole
    ``B``-blocks: that keeps local rows ``[row, row + take)`` covering global
    positions ``[row * dcp_size, (row + take) * dcp_size)`` on every rank. The
    pieces are identical on every rank; only ``local_lens`` is not.
    """
    prefix_lens = [int(p) for p in prefix_lens]
    extend_lens = [int(e) for e in extend_lens]
    b = interleave_size
    lens = torch.tensor(prefix_lens, dtype=torch.int64)
    local_lens = get_dcp_lens(lens, dcp_size, dcp_rank, interleave_size=b).tolist()
    # Rank 0 holds the most; every rank pads to it.
    padded_lens = get_dcp_lens(lens, dcp_size, 0, interleave_size=b).tolist()
    # Rounded down to a whole block, so a cut never splits one rank's run.
    piece_rows = max(b, (max_piece_gather_rows // dcp_size) // b * b)

    pieces = []
    parts = []
    send_start = extend_start = out_start = 0
    send_end = extend_end = out_end = 0
    send_offset = 0
    for prefix_len, extend_len, padded_len in zip(
        prefix_lens, extend_lens, padded_lens
    ):
        row = 0
        while row < padded_len:
            room = piece_rows - (send_end - send_start)
            # A cut inside a request lands on a whole block of it: the previous
            # request's rows need not fill whole blocks of the piece.
            take = padded_len - row if padded_len - row <= room else room // b * b
            if take == 0:
                pieces.append(
                    _dcp_extend_gather_piece(
                        parts,
                        dcp_size,
                        send_start,
                        send_end,
                        extend_start,
                        extend_end,
                        out_start,
                        out_end,
                        b,
                    )
                )
                parts = []
                send_start, extend_start, out_start = send_end, extend_end, out_end
                continue
            positions = torch.arange(
                row * dcp_size,
                min((row + take) * dcp_size, prefix_len),
                dtype=torch.int64,
            )
            parts.append(("prefix", positions, send_offset))
            row += take
            send_end += take
            out_end += positions.numel()
        if extend_len:
            parts.append(("extend", extend_end, extend_len))
            extend_end += extend_len
            out_end += extend_len
        send_offset += padded_len
    if parts:
        pieces.append(
            _dcp_extend_gather_piece(
                parts,
                dcp_size,
                send_start,
                send_end,
                extend_start,
                extend_end,
                out_start,
                out_end,
                b,
            )
        )
    return DcpExtendGatherPlan(
        local_lens=local_lens,
        padded_lens=padded_lens,
        send_rows=sum(padded_lens),
        scratch_rows=max((p.scratch_rows for p in pieces), default=0),
        pieces=pieces,
    )


def _dcp_extend_gather_piece(
    parts,
    dcp_size: int,
    send_start: int,
    send_end: int,
    extend_start: int,
    extend_end: int,
    out_start: int,
    out_end: int,
    interleave_size: int = 1,
) -> DcpExtendGatherPiece:
    """Close one piece: index its output rows into its rank-major scratch."""
    send_len = send_end - send_start
    gathered_rows = send_len * dcp_size
    b = interleave_size
    index = []
    for kind, first, second in parts:
        if kind == "prefix":
            positions, request_send_offset = first, second
            owner = (positions // b) % dcp_size
            local_row = dcp_local_row(positions, dcp_size, b)
            index.append(
                owner * send_len + (request_send_offset - send_start) + local_row
            )
        else:
            start = gathered_rows + first - extend_start
            index.append(torch.arange(start, start + second, dtype=torch.int64))
    return DcpExtendGatherPiece(
        send_start=send_start,
        send_end=send_end,
        extend_start=extend_start,
        extend_end=extend_end,
        out_start=out_start,
        out_end=out_end,
        index=torch.cat(index),
        scratch_rows=gathered_rows + extend_end - extend_start,
    )
