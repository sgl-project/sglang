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
from sglang.srt.layers.dcp.layout import (
    DcpPackedReadPlan,
    dcp_local_row,
    get_dcp_lens,
    plan_dcp_packed_read,
)
from sglang.srt.layers.dcp.metadata import DecodeContextParallelMetadata
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import print_info_once

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
    # Whether the gathered-KV operator may drop its causal crop (sparse_mode
    # 0), computed once at prepare: only when every prefix reaches index_topk.
    # The token shard, the packed read and the sparse operator read it.
    extend_crop_free: bool = False
    # The packed-read path runs its once-per-forward write plan on the first
    # layer; this records that it has.
    packed_write_planned: bool = False
    # DcpPackedReadPlan for this forward, owned by dcp_packed_read_plan below.
    # None = packed read not applicable.
    packed_read_plan: Optional[object] = None
    # True once packed_read_plan has been computed for this forward.
    packed_read_plan_done: bool = False
    # Cached rank-major kv lens for the packed read; see dcp_packed_kv_lens.
    packed_kv_lens: Optional[torch.Tensor] = None


def dcp_extend_crop_free(prefix_lens: Sequence[int], index_topk: Optional[int]) -> bool:
    """May an extend over the gathered KV drop the operator's causal crop?

    Only if every prefix reaches ``index_topk``: below it the top-k takes every
    key, so the crop is what keeps the result causal.
    """
    ok = bool(
        index_topk is not None
        and prefix_lens
        and all(p + 1 >= index_topk for p in prefix_lens)
    )
    if not ok and index_topk is not None and prefix_lens:
        print_info_once(
            f"DCP extend keeps the operator's causal crop: a request here has a "
            f"prefix under index_topk={index_topk} (shortest {min(prefix_lens)})"
        )
    return ok


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


def _dsa_kv_store_is_fp8() -> bool:
    """Whether the active DSA KV pool stores FP8 (``dsa_kv_cache_store_fp8`` /
    MXFP8). The packed single-record read does not support it: under the FP8
    store ``get_mla_kv_buffer`` returns the packed uint8 view with ``rope=None``,
    which the packed path's ``_pad_dcp_packed_send`` cannot consume. Defensive:
    the forward context is not always up (a plan probe can run before a forward),
    so a missing context means "not FP8" and the other refusals still apply.
    """
    try:
        from sglang.srt.model_executor.forward_context import get_token_to_kv_pool

        return bool(getattr(get_token_to_kv_pool(), "dsa_kv_cache_store_fp8", False))
    except Exception:
        return False


def dcp_packed_read_plan(
    forward_batch: "ForwardBatch", index_topk: Optional[int]
) -> Optional[DcpPackedReadPlan]:
    """This forward's packed-buffer plan, or None to keep the permuting path.

    Resolved once per forward and cached on the NPU DCP metadata. It lives here,
    beside the metadata it caches on, rather than in the NPU model module,
    because BOTH the model (to fill the buffer and remap the top-k) and the
    attention backend (to set the operator's KV length and drop its causal
    crop) have to agree on it, and a backend importing a model module inverts
    the layering.

    Every refusal is logged once. A silent refusal here would look exactly like
    a feature that is on and doing nothing.
    """
    md = forward_batch.attn_dcp_metadata
    if md.packed_read_plan_done:
        return md.packed_read_plan

    plan = None
    extend_lens = forward_batch.extend_seq_lens_cpu
    prefix_lens = forward_batch.extend_prefix_lens_cpu
    if _dsa_kv_store_is_fp8():
        # FP8 DSA KV store (dsa_kv_cache_store_fp8 / MXFP8): the packed
        # single-record read is not supported for it (get_mla_kv_buffer returns
        # rope=None). The permuting path handles FP8 (it packs uint8 and the
        # backend's quant branch dequants), so decline to that fallback -- the
        # v1 packed-read refusal, re-expressed as a plan admission.
        print_info_once(
            "DCP packed read is off: the DSA KV is stored in FP8 "
            "(dsa_kv_cache_store_fp8); the packed single-record read is not "
            "supported for it, so the permuting path is used"
        )
    elif index_topk is None:
        # No top-k to measure, so the causal bound below cannot be checked.
        # Refuse, and CACHE the refusal: a forward has to take one path the
        # whole way down. Resolving later would leave earlier layers' top-k
        # unremapped while later layers read a packed buffer.
        print_info_once(
            "DCP packed read is off: this forward reached the extend gather "
            "with no top-k, so the causal bound cannot be checked"
        )
    elif not extend_lens or prefix_lens is None:
        print_info_once("DCP packed read is off: no CPU length metadata on this extend")
    elif len(extend_lens) != 1:
        # ONE REQUEST, for the layout reason in DcpPackedReadPlan: the
        # all-gather is rank-major over the WHOLE send, so with two requests a
        # request's rows land in dcp_size separate blocks and the operator's
        # cumulative actual_seq_lengths_kv cannot describe them. The permuting
        # path handles those.
        print_info_once(
            f"DCP packed read is off for multi-request extends "
            f"({len(extend_lens)} requests here); the all-gather is rank-major "
            "over the whole send, so no request is contiguous in it"
        )
    elif not md.extend_crop_free:
        # Computed once at prepare (dcp_extend_crop_free); it has already
        # logged which request fell short.
        print_info_once(
            "DCP packed read is off: this chunk needs the operator's causal "
            "crop, and the packed layout cannot use one"
        )
    else:
        plan = plan_dcp_packed_read(
            prefix_lens[0], extend_lens[0], get_parallel().dcp_size
        )
        print_info_once(
            "DCP packed read is ON: the sparse operator reads the all-gather's "
            "own rank-major output and the top-k is remapped instead"
        )
    md.packed_read_plan = plan
    md.packed_read_plan_done = True
    return plan


def dcp_packed_kv_lens(forward_batch, plan: DcpPackedReadPlan, device) -> torch.Tensor:
    """The operator's ``actual_seq_lengths_kv`` for a packed read, built once.

    One request, so one entry, and it is the whole buffer: with the causal crop
    off there is nothing to shorten, and the padding rows the ranks sent to keep
    their sends equal are never named by any remapped index.

    Cached on the DCP metadata rather than built where it is used, because that
    use is once per layer and ``torch.tensor(list, device=npu)`` is a blocking
    host-to-device copy -- 78 of them a forward, each draining the queue.
    """
    md = forward_batch.attn_dcp_metadata
    cached = md.packed_kv_lens
    if cached is None:
        cached = torch.tensor([plan.rows], dtype=torch.int32, device=device)
        md.packed_kv_lens = cached
    return cached


def _dcp_all_gather_async(group, out: torch.Tensor, send: torch.Tensor):
    """The chunked gather's async collective; returns a Work handle or None.

    torch.distributed directly rather than a GroupCoordinator method: the
    chunked DCP gather is the only async all-gather user on NPU, so the async
    path stays NPU-side and GroupCoordinator keeps its synchronous API.
    """
    return torch.distributed.all_gather_into_tensor(
        out, send, group=group.device_group, async_op=True
    )


def dcp_issue_gather_chunks(
    group, out: torch.Tensor, send: torch.Tensor, plan: DcpPackedReadPlan
):
    """Fill ``out``'s prefix region with the all-gather of ``send``.

    D2v2: ``plan.chunks > 1`` issues one sub-all-gather per chunk instead of
    one monolithic collective. Chunk k moves local rows [st, st + sz) of every
    rank -- the first chunks-1 hold q = send_rows // chunks rows, the last
    q + send_rows % chunks -- and each lands in the CONTIGUOUS slice
    ``out[C*st : C*(st + sz)]`` of the very buffer the packed read expects
    (chunk-major, see DcpPackedReadPlan), so no extra buffer and no copy. With
    ``plan.chunks == 1`` this is exactly the old single collective.

    The point is not the bytes (they are the same) but the AIV episode: the
    monolithic gather was one 9-16 ms AIV kernel (93% MTE2) that starved the
    compute stream's own AIV work for its whole flight; N sub-collectives of
    roughly 1-2 ms leave the inter-collective windows in which that work runs.
    See _DCP_GATHER_CHUNKS in layers/dcp/layout.py for the measurement.

    Order every chunk's completion before the caller consumes the prefix or
    records its ready event; communication may use a separate backend stream.
    """
    c = plan.dcp_size
    if plan.chunks <= 1:
        group.all_gather_into_tensor(out[: plan.gathered_rows], send)
        return ()
    q, rem = divmod(plan.send_rows, plan.chunks)
    n = plan.chunks
    works = []
    st = 0
    for k in range(n):
        sz = q + (rem if k == n - 1 else 0)
        works.append(
            _dcp_all_gather_async(
                group, out[c * st : c * (st + sz)], send[st : st + sz]
            )
        )
        st += sz
    for work in works:
        if work is not None:
            work.wait()
    return tuple(works)
