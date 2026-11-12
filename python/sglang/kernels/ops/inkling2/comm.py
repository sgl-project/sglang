from functools import cache

import torch
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl

from sglang.kernels.jit.utils import load_jit
from sglang.kernels.ops.inkling2.moe import fence_system

_RANK_ORDER_MAX_BYTES = {2: 1200 * 1024, 4: 512 * 1024, 8: 288 * 1024, 16: 64 * 1024}


@triton.jit
def _rank_sum(X, Y, N: tl.constexpr, WORLD: tl.constexpr, BLOCK: tl.constexpr):
    indices = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    total = tl.full((BLOCK,), 0, tl.float32)
    for rank in tl.static_range(WORLD):
        value = tl.load(X + rank * N + indices, indices < N, other=0).to(tl.float32)
        total = total + tl.where(value == 0, 0.0, value)
    tl.store(Y + indices, total, indices < N)


@cache
def _multicast_buffer(group_name, device, dtype, capacity):
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Warm up the collective size before CUDA graph capture")
    buffer = symm_mem.empty(capacity, device=device, dtype=dtype)
    handle = symm_mem.rendezvous(buffer, group_name)
    if not handle.multicast_ptr:
        raise RuntimeError("Inkling2 tensor parallelism requires NVLink multicast")
    return buffer


def all_reduce(x: torch.Tensor, group, *, multicast: bool = False) -> torch.Tensor:
    world = group.world_size
    if world == 1 or x.numel() == 0:
        return x
    if world not in _RANK_ORDER_MAX_BYTES:
        raise ValueError("Inkling2 tensor parallelism requires 2, 4, 8, or 16 ranks")
    if not x.is_cuda or x.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("Inkling2 collectives require BF16 or FP32 CUDA tensors")
    if x.nbytes % 4:
        raise ValueError("Inkling2 collective payloads must be 4-byte aligned")
    x = x.contiguous()
    if (
        not multicast
        and torch.cuda.get_device_capability(x.device)[0] == 10
        and x.nbytes < _RANK_ORDER_MAX_BYTES[world]
        and x.nbytes % 16 == 0
    ):
        peers = group.all_gather(x, dim=0)
        output = torch.empty_like(x)
        _rank_sum[(triton.cdiv(x.numel(), 256),)](
            peers, output, x.numel(), world, 256, enable_fp_fusion=False
        )
        return output
    comm = group.torch_symm_mem_comm
    if comm is not None and comm.should_torch_symm_mem_allreduce(x):
        buffer = comm.buffer[: x.numel()]
    else:
        capacity = 1 << (x.numel() - 1).bit_length()
        buffer = _multicast_buffer(
            group.device_group.group_name, x.device, x.dtype, capacity
        )[: x.numel()]
    buffer.copy_(x.flatten())
    torch.ops.symm_mem.multimem_all_reduce_(
        buffer, "sum", group.device_group.group_name
    )
    return buffer.view_as(x).clone()


def multicast_all_reduce_input(group, shape, dtype):
    """A view of the symmetric multicast buffer for a producer to write an all-reduce
    input into; multicast_all_reduce reduces it in place."""
    numel = shape[0] * shape[1]
    capacity = 1 << (numel - 1).bit_length()
    buffer = _multicast_buffer(
        group.device_group.group_name,
        torch.device("cuda", torch.cuda.current_device()),
        dtype,
        capacity,
    )
    return buffer[:numel].view(shape)


def multicast_all_reduce(x, group):
    torch.ops.symm_mem.multimem_all_reduce_(
        x.view(-1), "sum", group.device_group.group_name
    )
    return x


ONESHOT_MAX_TOKENS = 64
# Larger reductions run the in-place multimem kernel with entry/exit barriers.
INPLACE_MAX_TOKENS = 512
PREFILL_INPLACE_MAX_TOKENS = 32768


def _oneshot_blocks(tokens):
    # Tuned on GB300 TP4 inside CUDA graphs.
    return 1 if tokens == 1 else 4 if tokens <= 4 else 8 if tokens <= 8 else 32


class _OneShotAllReduce:
    """Multimem full one-shot all-reduce for decode-sized activations.

    Every rank issues the same multimem.ld_reduce as torch's multimem all-reduce
    over the whole range, so results are identical. Inputs alternate between
    two symmetric regions: an input region is rewritten two calls later, after
    every peer has passed the next call's entry barrier."""

    def __init__(self, group, hidden, dtype, device, inplace_tokens):
        from sglang.kernels.ops.communication import inkling_all_reduce as jit

        self.jit = jit
        self.region = ONESHOT_MAX_TOKENS * hidden
        self.inplace_tokens = inplace_tokens
        name = group.device_group.group_name
        self.buffer = symm_mem.empty(3 * self.region, device=device, dtype=dtype)
        self.handle = symm_mem.rendezvous(self.buffer, name)
        self.inplace = symm_mem.empty(
            inplace_tokens * hidden, device=device, dtype=dtype
        )
        self.inplace_handle = symm_mem.rendezvous(self.inplace, name)
        self.flags = symm_mem.empty(
            jit.flags_numel(group.world_size), device=device, dtype=torch.uint32
        )
        self.flags.zero_()
        self.flags_handle = symm_mem.rendezvous(self.flags, name)
        self.flags_handle.barrier()
        self.state = torch.zeros(jit.STATE_SIZE, device=device, dtype=torch.uint32)
        jit.compile_inkling_all_reduce(dtype, group.world_size)
        self.world = group.world_size
        self.current = 0

    def input(self, rows, cols):
        if rows > ONESHOT_MAX_TOKENS:
            return self.inplace[: rows * cols].view(rows, cols)
        start = self.current * self.region
        return self.buffer[start : start + rows * cols].view(rows, cols)

    def __call__(self, x, acc_f32=False):
        n = x.numel()
        if x.data_ptr() == self.inplace.data_ptr():
            kernel, blocks, block_size = (
                self.jit.select_ar_config(x.shape[0], self.world)
                if x.shape[0] > INPLACE_MAX_TOKENS
                else ("v3b" if x.shape[0] > 256 else "v3", 32, 1024)
            )
            if kernel not in ("v3", "v3b"):
                kernel, blocks, block_size = "v3b", 32, 1024
            self.jit.inkling_multimem_one_shot_fused(
                self.inplace[:n],
                self.inplace_handle.multicast_ptr,
                self.flags_handle.buffer_ptrs_dev,
                self.state.data_ptr(),
                self.handle.rank,
                self.world,
                n,
                blocks,
                block_size,
                per_block_barrier=kernel == "v3b",
                acc_f32=acc_f32,
            )
            return x
        start = self.current * self.region
        output = self.buffer[2 * self.region : 2 * self.region + n]
        self.jit.inkling_multimem_full_oneshot(
            self.buffer[start : start + n],
            output,
            self.handle.multicast_ptr + start * x.element_size(),
            self.flags_handle.buffer_ptrs_dev,
            self.state.data_ptr(),
            self.handle.rank,
            self.world,
            n,
            _oneshot_blocks(x.shape[0]),
            1024,
        )
        self.current = 1 - self.current
        return output.view(x.shape)


_ONESHOT: dict = {}


def oneshot_all_reduce_input(group, rows, cols, dtype, tag=""):
    """A view of the next one-shot all-reduce input region for a producer to write
    into, or None when the one-shot path does not apply (including when NVLink
    multicast is unavailable)."""
    if (
        group.world_size not in (4, 8)
        or dtype != torch.bfloat16
        or rows <= 0
        or (rows * cols) % 8
    ):
        return None
    key = (group.device_group.group_name, cols, tag)
    if key not in _ONESHOT:
        if torch.cuda.is_current_stream_capturing():
            return None
        from sglang.srt.runtime_context import cutedsl_moe_max_num_tokens

        reducer = _OneShotAllReduce(
            group,
            cols,
            dtype,
            torch.device("cuda", torch.cuda.current_device()),
            min(
                max(INPLACE_MAX_TOKENS, cutedsl_moe_max_num_tokens()),
                PREFILL_INPLACE_MAX_TOKENS,
            ),
        )
        _ONESHOT[key] = reducer if reducer.handle.multicast_ptr else None
    reducer = _ONESHOT[key]
    if reducer is None or rows > reducer.inplace_tokens:
        return None
    return reducer.input(rows, cols)


def oneshot_all_reduce(x, group, acc_f32=False, tag=""):
    """All-reduce x, which must be the view returned by oneshot_all_reduce_input
    (same tag; tags own separate buffers). The result is a view that the next
    one-shot all-reduce with that tag overwrites. acc_f32 (in-place sizes only)
    accumulates in FP32."""
    return _ONESHOT[(group.device_group.group_name, x.shape[1], tag)](x, acc_f32)


@cache
def _exchange_module(hidden, top_k, world):
    return load_jit(
        "inkling2_exchange",
        str(hidden),
        str(top_k),
        str(world),
        cuda_files=["inkling2/exchange.cuh"],
        cuda_wrappers=[
            ("combine", f"inkling2_exchange::combine<{hidden}, {top_k}, {world}>")
        ],
    )


class RoutedExchange:
    """EP combine of routed expert rows without an all-reduce. Each rank writes
    the rows it owns into its symmetric buffer; after a per-CTA flag barrier,
    every rank sums each token's rows in top-k order, reading each row from its
    owner. Equal to unweighted_post_reorder(all_reduce(routed_rows)): every row
    is nonzero on one rank only. Row buffers alternate between two slots by a
    device call count (graph replays included); a slot is rewritten two calls
    later, after every peer passed the next call's barrier."""

    def __init__(self, group, max_rows, hidden, device):
        name = group.device_group.group_name
        self.world, self.hidden, self.max_rows = group.world_size, hidden, max_rows
        # Persistent grid: every CTA must be resident for the per-CTA barrier.
        self.max_blocks = (
            2 * torch.cuda.get_device_properties(device).multi_processor_count
        )
        self.buffer = symm_mem.empty(
            2 * max_rows * hidden, device=device, dtype=torch.bfloat16
        )
        self.handle = symm_mem.rendezvous(self.buffer, name)
        self.flags = symm_mem.empty(
            self.max_blocks * self.world, device=device, dtype=torch.uint32
        )
        self.flags.zero_()
        self.flags_handle = symm_mem.rendezvous(self.flags, name)
        self.flags_handle.barrier()
        self.state = torch.zeros(self.max_blocks, device=device, dtype=torch.uint32)
        self.buffer_ptrs = torch.tensor(
            self.handle.buffer_ptrs, device=device, dtype=torch.int64
        )
        self.flag_ptrs = torch.tensor(
            self.flags_handle.buffer_ptrs, device=device, dtype=torch.int64
        )
        # [call count, CTAs finished in the current call]
        self.calls = torch.zeros(2, device=device, dtype=torch.int64)
        self.rank = self.handle.rank

    def combine(self, src2dst, sorted_ids, num_local, top_k):
        tokens = src2dst.numel() // top_k
        out = torch.empty(
            (tokens, self.hidden), device=src2dst.device, dtype=torch.bfloat16
        )
        if tokens:
            _exchange_module(self.hidden, top_k, self.world).combine(
                self.buffer_ptrs,
                self.flag_ptrs,
                self.state,
                src2dst,
                sorted_ids,
                out,
                self.calls,
                min(tokens, self.max_blocks),
                self.max_rows * self.hidden,
                self.rank,
                num_local,
            )
        return out


_ROUTED_EXCHANGE: dict = {}


def prepare_routed_exchange(group, max_rows, hidden):
    """Allocate the routed exchange (collective; call outside graph capture)."""
    key = (group.device_group.group_name, hidden)
    if key not in _ROUTED_EXCHANGE and group.world_size > 1:
        _ROUTED_EXCHANGE[key] = RoutedExchange(
            group, max_rows, hidden, torch.device("cuda", torch.cuda.current_device())
        )


@triton.jit
def _peer_barrier(FlagPtrs, State, rank, WORLD: tl.constexpr):
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()
    epoch = tl.load(State) + 1
    tl.store(State, epoch)
    for peer in tl.static_range(WORLD):
        flags = tl.load(FlagPtrs + peer).to(tl.pointer_type(tl.uint32))
        tl.atomic_xchg(flags + rank, epoch, sem="release", scope="sys")
    own = tl.load(FlagPtrs + rank).to(tl.pointer_type(tl.uint32))
    for peer in tl.static_range(WORLD):
        seen = tl.atomic_add(own + peer, 0, sem="acquire", scope="sys")
        while seen < epoch:
            seen = tl.atomic_add(own + peer, 0, sem="acquire", scope="sys")


@triton.jit
def _prefill_exchange_combine(
    Rows,
    OutPtrs,
    Src2Dst,
    tokens,
    per_rank,
    rank,
    WORLD: tl.constexpr,
    TOP_K: tl.constexpr,
    H: tl.constexpr,
    CHUNKS: tl.constexpr,
    BLOCK: tl.constexpr,
    THREADS: tl.constexpr,
):
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()
    item = tl.program_id(0)
    src = rank * per_rank + item // CHUNKS
    if src >= tokens:
        return
    src = src.to(tl.int64)
    cols = (item % CHUNKS) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < H
    total = tl.zeros([BLOCK], dtype=tl.float32)
    for k in tl.static_range(TOP_K):
        dst = tl.load(Src2Dst + src * TOP_K + k).to(tl.int64)
        total += tl.load(Rows + dst * H + cols, mask) * 1.0
    value = total.to(tl.bfloat16)
    for peer in tl.static_range(WORLD):
        out = tl.load(OutPtrs + peer).to(tl.pointer_type(tl.bfloat16))
        tl.store(out + src * H + cols, value, mask)
    fence_system(THREADS)


class PrefillExchange:
    """EP combine of routed expert rows for prefill-sized batches. Rank r
    combines the r-th 1/world of the tokens (per_rank tokens each): owners
    push each row they computed into the row buffer of the rank combining its
    token; after a barrier, every rank sums its tokens' top-k rows in top-k
    order and stores the sums into every rank's output buffer; a second
    barrier publishes them. Equal to unweighted_post_reorder(all_reduce(
    routed_rows)): every row is nonzero on one rank only. Peers write a rank's
    row buffer only after the previous call's second barrier, and its output
    buffer only after it reached the next call's first barrier, so one slot of
    each suffices."""

    calls = None

    def __init__(self, group, max_tokens, top_k, hidden, device):
        name = group.device_group.group_name
        self.world, self.hidden, self.top_k = group.world_size, hidden, top_k
        self.max_tokens, self.max_rows = max_tokens, max_tokens * top_k
        self.buffer = symm_mem.empty(
            self.max_rows * hidden, device=device, dtype=torch.bfloat16
        )
        self.handle = symm_mem.rendezvous(self.buffer, name)
        self.out = symm_mem.empty(
            max_tokens * hidden, device=device, dtype=torch.bfloat16
        )
        self.out_handle = symm_mem.rendezvous(self.out, name)
        self.flags = symm_mem.empty(self.world, device=device, dtype=torch.uint32)
        self.flags.zero_()
        self.flags_handle = symm_mem.rendezvous(self.flags, name)
        self.flags_handle.barrier()
        self.state = torch.zeros(1, device=device, dtype=torch.uint32)
        self.row_ptrs, self.out_ptrs, self.flag_ptrs = (
            torch.tensor(h.buffer_ptrs, device=device, dtype=torch.int64)
            for h in (self.handle, self.out_handle, self.flags_handle)
        )
        self.rank = self.handle.rank

    def barrier(self):
        _peer_barrier[(1,)](
            self.flag_ptrs,
            self.state,
            self.rank,
            self.world,
            num_warps=1,
            launch_pdl=True,
        )

    def per_rank(self, tokens):
        return triton.cdiv(tokens, self.world)

    def combine(self, src2dst, top_k):
        tokens = src2dst.numel() // top_k
        self.barrier()
        per_rank = self.per_rank(tokens)
        chunks = triton.cdiv(self.hidden, 512)
        _prefill_exchange_combine[(per_rank * chunks,)](
            self.buffer,
            self.out_ptrs,
            src2dst,
            tokens,
            per_rank,
            self.rank,
            self.world,
            top_k,
            self.hidden,
            chunks,
            512,
            128,
            num_warps=4,
            launch_pdl=True,
        )
        self.barrier()
        return self.out[: tokens * self.hidden].view(tokens, self.hidden)


_PREFILL_EXCHANGE: dict = {}


def prepare_prefill_exchange(group, max_tokens, top_k, hidden):
    """Allocate the prefill exchange (collective; call outside graph capture)."""
    key = (group.device_group.group_name, hidden)
    if key not in _PREFILL_EXCHANGE and group.world_size > 1 and max_tokens > 0:
        _PREFILL_EXCHANGE[key] = PrefillExchange(
            group,
            max_tokens,
            top_k,
            hidden,
            torch.device("cuda", torch.cuda.current_device()),
        )


def prefill_exchange(group, rows, hidden):
    exchange = _PREFILL_EXCHANGE.get((group.device_group.group_name, hidden))
    return exchange if exchange is not None and rows <= exchange.max_rows else None


ROUTED_EXCHANGE_MIN_ROWS = 2048


def routed_exchange(group, rows, hidden):
    if not _ROUTED_EXCHANGE or rows < ROUTED_EXCHANGE_MIN_ROWS:
        return None
    exchange = _ROUTED_EXCHANGE.get((group.device_group.group_name, hidden))
    return exchange if exchange is not None and rows <= exchange.max_rows else None
