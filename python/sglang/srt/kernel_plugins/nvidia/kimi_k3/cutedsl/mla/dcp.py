# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""MLA decode under decode context parallelism (DCP = TP, rank d owns head block d)
without collectives, three kernels per layer. Each rank's KV cache holds 1 / DCP of
every context.

  MlaQuery     FP8 query rows [B, H, 576] (+ the KV insert where this rank owns the
               slot); each CTA multicasts its bytes into every rank's Lamport q
               mailbox [B, DCP*H, 576], then polls its share until all landed
  tokenspeed   all DCP*H heads over the local KV shard -> BF16 rows + LSE (log2)
  DcpValue     re-arms the q mailbox; rows pushed to their head owner's Lamport
               mailbox (peer stores); the owner LSE-merges its H heads over the DCP
               sources, then W_UV and the gate (padding from the global lengths)
"""

import math
from functools import cache

import cutlass.cute as cute
import torch
import torch.distributed._symmetric_memory as symm_mem
from cutlass import Float32, Int32, Int64
from cutlass.cute.runtime import make_fake_stream

from sglang.kernels.jit.kimi_k3.host import check_residency, fake
from sglang.kernels.ops.communication.kimi_k3_specialized.mla_dcp import DcpValue
from sglang.kernels.ops.communication.kimi_k3_specialized.mla_query import (
    _require_deadline_ns,
)
from sglang.srt.kernel_plugins.nvidia.kimi_k3.cutedsl.common.host import (
    validate_init_barrier_group,
)


class DcpBox:
    """Model-owned DCP state over ``group``, Lamport mailboxes (sentinel-filled,
    re-armed by their reader): q [2, CAP, DCP*H, 576] FP8 bytes (multicast stores),
    partials [2, DCP, CAP, H, 512] BF16 and LSE [2, DCP, CAP, H] FP32 (peer stores),
    and the local tokenspeed rows / LSE. Parity alternates by layer. ``interleave``:
    the KV cache's DCP interleave (global masking of spec verify rows)."""

    def __init__(
        self,
        group,
        device,
        size,
        rank,
        heads,
        cap=64,
        interleave=1,
        *,
        deadline_ns,
        init_barrier_group=None,
    ):
        self.deadline_ns = _require_deadline_ns(deadline_ns)
        if size not in (8, 16) or size * heads != 96 or not 0 <= rank < size:
            raise ValueError("Kimi DCP needs TP8/12-head or TP16/6-head rank topology")
        if cap != 64 or interleave != 1:
            raise ValueError(
                "Kimi DCP needs 64-row mailboxes and token-striped KV ownership "
                "(interleave=1)"
            )
        if not torch.distributed.is_initialized():
            raise RuntimeError("Kimi DCP needs an initialized process group")
        if (
            torch.distributed.get_world_size(group) != size
            or torch.distributed.get_rank(group) != rank
        ):
            raise ValueError("Kimi DCP rank topology differs from its process group")
        init_group = validate_init_barrier_group(group, init_barrier_group)
        self.size, self.rank, self.heads, self.cap = size, rank, heads, cap
        self.group = group
        self.interleave = interleave
        empty = symm_mem.empty
        self.q = empty((2, cap, size * heads, 576), dtype=torch.uint8, device=device)
        shape = (2, size, cap, heads)
        self.partial = empty((*shape, 512), dtype=torch.bfloat16, device=device)
        self.lse = empty(shape, dtype=torch.float32, device=device)
        tensors = (self.q, self.partial, self.lse)
        self.handles = handles = tuple(symm_mem.rendezvous(t, group) for t in tensors)
        if any(len(handle.buffer_ptrs) != size for handle in handles):
            raise RuntimeError("Kimi DCP symmetric handles have a different rank count")
        self.q_mc = int(handles[0].multicast_ptr or 0)
        if not self.q_mc:
            raise RuntimeError("Kimi K3 specialized MLA DCP needs multicast memory")
        i64 = dict(dtype=torch.int64, device=device)
        self.peers = torch.tensor(handles[1].buffer_ptrs, **i64)
        self.lse_peers = torch.tensor(handles[2].buffer_ptrs, **i64)
        self.q.fill_(0xFF)
        self.partial.view(torch.int32).fill_(-0x80000000)
        self.lse.view(torch.int32).fill_(-1)
        rows = (cap, 1, size * heads)
        self.rows = torch.empty((*rows, 512), dtype=torch.bfloat16, device=device)
        self.rows_lse = torch.empty(rows, dtype=torch.float32, device=device)
        torch.accelerator.synchronize(device)
        torch.distributed.barrier(group=init_group)

    def mc(self, parity):
        """The q mailbox multicast address of ``parity``."""
        if parity not in (0, 1):
            raise ValueError("Kimi DCP mailbox parity must be 0 or 1")
        return Int64(self.q_mc + parity * self.q[0].numel())


@cache
def compile_value(
    heads, size, batch, cap, uv, pitch, device, sms=148, q=1, *, deadline_ns
):
    """DcpValue of one batch (``q`` rows per request); ``uv``: the (shape, stride,
    dtype) of W_UV, ``pitch``: the projected rows' (the gate follows the 2112 query /
    KV columns)."""
    if cap != 64:
        raise ValueError("Kimi DCP value requires its prepared 64-row mailbox")
    kernel = DcpValue(heads, size, batch, sms, q, deadline_ns=deadline_ns)
    f32, i32 = torch.float32, torch.int32
    w_uv, gate = fake(*uv), fake((batch, heads * 128), (pitch, 1))
    rows = fake((batch, 1, size * heads, 512))
    rows_lse = fake((batch, 1, size * heads), None, f32, 4)
    lens = fake((batch // q,), None, i32, 4)
    box = fake((size, cap, heads, 512))
    lbox = fake((size, cap, heads), None, f32, 4)
    peers = fake((size,), None, torch.int64, 8)
    mailbox = fake((cap, size * heads, 576), None, torch.uint8)
    output = fake((batch, heads * 128), (heads * 128, 1))
    args: tuple = (rows, rows_lse, lens, box, lbox, peers, peers, mailbox, w_uv)
    args += (gate, lens, output, Int32(0), Int32(0), make_fake_stream())
    with torch.accelerator.device_index(device):
        compiled = cute.compile(kernel, *args)
        check_residency(compiled, device, math.prod(kernel.grid), 512, "Kimi DCP value")
        return compiled


_WORKSPACE: dict[tuple[torch.device, int], torch.Tensor] = {}


def _decode_metadata(md, device, heads, q_len, batch, max_seq, min_split, pack=False):
    """Cache-independent launch geometry shared by preparation and execution."""
    arch = torch.cuda.get_device_capability(device)
    tiler, _ = md.select_mla_decode_tilers(
        heads, q_len, is_fp8=True, compute_capability=arch
    )
    pack = pack and tiler[0] == 128
    fold = 1 if pack else md.get_mla_decode_fold_sq_factor(heads, q_len, tiler[0])
    rows = (q_len // fold, heads * fold)
    if pack:
        rows = (md.compute_q_tile_layout(heads, q_len, 128)[1], 128)
    sms = md.get_num_sm(device)
    split, size = md._get_split_kv_and_workspace_size(
        batch, *rows, 512, sms, max_seq, torch.float8_e4m3fn, tiler, min_split
    )
    return arch, pack, fold, sms, split, size


def _reserve_workspace(device, heads, size, sms):
    key = device, heads
    if key not in _WORKSPACE or _WORKSPACE[key].numel() < size:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "the MLA split-KV workspace must be sized before capture"
            )
        bound = max(size, sms * heads * 513 * 4)
        _WORKSPACE[key] = torch.empty(bound, dtype=torch.int8, device=device)
    return _WORKSPACE[key]


def prepare_workspace(
    device, heads, max_seq, batches=(1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64), q=1
):
    """Reserve ordinary-decode tokenspeed scratch before KV-pool sizing.

    Only host geometry is queried; this neither compiles nor launches a kernel,
    and does not need a cache. Every prepared width shares the returned buffer.
    """
    from tokenspeed_mla import mla_decode as md

    if q != 1 or not isinstance(max_seq, int) or max_seq <= 0:
        raise ValueError(
            "Kimi workspace preparation needs ordinary decode and a positive context"
        )
    batches = tuple(batches)
    if not batches or any(
        not isinstance(batch, int) or not 1 <= batch <= 64 for batch in batches
    ):
        raise ValueError("Kimi workspace preparation needs token-row widths in [1, 64]")
    device = torch.device(device)
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Kimi MLA workspace must be prepared before capture")
    maximum = 0
    sms = None
    for batch in batches:
        metadata = _decode_metadata(
            md, device, heads, q, batch, max_seq, fill_splits(batch, device)
        )
        sms = metadata[3]
        maximum = max(maximum, metadata[5])
    return _reserve_workspace(device, heads, maximum, sms)


def stock_decode(
    query,
    cache,
    block_table,
    lengths,
    max_seq,
    scales,
    out,
    lse=None,
    min_split=1,
    pdl=True,
    cp=(1, 0, 1),
    totals=None,
    pack=False,
):
    """tokenspeed_mla_decode's FP8 path with caller-owned ``out`` and ``lse`` (log2)
    buffers and no per-call allocation: the split count follows from (B, max_seq,
    min_split); ``lengths`` bound the work, empty splits are skipped by its reducer.
    The q rows of a request mask causally; under DCP in global positions (``cp``:
    world, rank, interleave; ``totals``: the global lengths). ``pack``: M128 tiles hold
    consecutive (query row, head) rows across the q rows."""
    import tvm_ffi
    from tokenspeed_mla import mla_decode as md

    batch, q_len, heads, _ = query.shape
    fp8, device = torch.float8_e4m3fn, query.device
    arch, pack, fold, sms, split, size = _decode_metadata(
        md, device, heads, q_len, batch, max_seq, min_split, pack
    )
    # One split-KV workspace per (device, heads), shared by every MLA layer: the calls
    # are stream ordered. prepare_workspace reserves ordinary-decode widths before
    # KV sizing; prepare_cache warms all shapes before capture. Growing during a
    # capture would leave earlier graphs referring to the freed buffer.
    key = device, heads
    _reserve_workspace(device, heads, size, sms)
    kernel = md._get_compiled_mla_kernel(
        torch_dtype=fp8,
        page_size=cache.shape[1],
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        is_persistent=False,
        is_var_seq=True,
        is_var_split_kv=False,
        compute_capability=arch,
        has_local_visible_lens=False,
        skip_correction_threshold=8.0,
        is_workspace_size_zero=size == 0,
        fold_sq_factor=fold,
        causal_mask=True,
        num_heads=heads,
        seq_len_q=q_len,
        window_left=-1,
        cp_world=cp[0],
        cp_rank=cp[1],
        cp_interleave_size=cp[2],
        use_pdl=pdl,
        return_lse=lse is not None,
        reducer_d_tiles=md._get_reducer_d_tiles(batch, q_len, heads, sms, split),
        reducer_max_splits=md._get_reducer_max_splits(split),
        pack_q=pack,
        partial_fp16=False,
    )
    cache = cache.view(fp8)
    softmax, output_scale = scales
    args = [query[..., :512], query[..., 512:], cache[..., :512], cache[..., 512:]]
    args += [block_table, out, lse, _WORKSPACE[key][:size] if size else None]
    args += [Int32(split), lengths, lengths if totals is None else totals, None]
    args += [Float32(softmax), Float32(output_scale)]
    with tvm_ffi.use_torch_stream():
        kernel(*args)


def fill_splits(batch, device):
    """Splits filling the SMs (one CTA pair each); tokenspeed caps itself at 32."""
    from tokenspeed_mla import get_num_sm

    return min(256, max(1, get_num_sm(device) // (2 * batch)))


def local_max(max_context, size):
    """Longest local shard of a ``max_context`` sequence (block-interleaved)."""
    return math.ceil(max_context / (size * 64)) * 64
