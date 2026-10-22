# SPDX-License-Identifier: Apache-2.0
"""Copy-engine transport for N-rank Ulysses all-to-all on one host.

The two-rank ``ipc_a2a`` design widened to every peer: each rank maps the
peers' staging buffers into its own device context and writes its chunk for a
peer straight into that peer's buffer. A contiguous device-to-device copy runs
on the copy engine, so the exchange takes no SMs from the kernels around it --
NCCL's all-to-all is an SM kernel, and next to a full-occupancy attention it
slows that attention by more than the exchange costs.

Per peer, a sequence counter published after the copy orders the data before
the flag, and the receiver spins on that peer's entry. The plain exchange's
double-buffered slots alternate per call; a slot is only rewritten two calls
later, after a spin on every peer, and each peer publishes that call only after
it has consumed the slot in stream order. The pipelined attention needs one set
of slots (see ``_PipelinePool``).
"""

import dataclasses
import functools
import logging
import socket
from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.distributed as dist

from sglang.kernels.ops.communication.ipc_a2a import load_ipc_a2a_sync
from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.distributed.device_communicators.ipc_a2a import (
    _Unsupported,
    ipc_shareable_zeros,
)

logger = logging.getLogger(__name__)

# Group counts tried, in order, when the caller leaves the choice to the
# transport: MiniMax-H3's 56 heads give 28, 14 and 7 per rank at Ulysses 2, 4
# and 8, and 7 groups divide all three; Wan's 40 and 12 heads need 5 and 3.
_AUTO_PIPELINE_GROUPS = (7, 5, 4, 3, 2)
# Each group's attention call must fill this many waves of 128-row tiles: below
# it the call's tail costs more than the overlap saves (cuDNN SDPA on B300 runs
# Wan 1.3B at Ulysses 4 in three one-head groups 44% slower than sequentially).
_MIN_GROUP_WAVES = 4


@functools.lru_cache(maxsize=None)
def _sm_count(device: int) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


@dataclass(frozen=True)
class _PipelinePlan:
    """Rows and heads of one pipelined attention call.

    Each part (q, k, v) is a ``[batch, s_local, heads, head_dim]`` sequence
    shard plus ``replicated`` rows that every rank holds and that stay out of
    the exchange (text tokens), attended before the gathered sequence when
    ``replicated_first`` and after it otherwise. A receive slot holds one part
    of one head group as attention reads it: its replicated rows and every
    rank's shard in rank order. The output keeps q's rows in this rank's
    layout: its replicated rows and this rank's shard.
    """

    batch: int
    s_local: int
    heads: tuple[int, int, int]
    head_dim: int
    replicated: tuple[int, int, int]
    replicated_first: bool
    world: int
    groups: int

    def group_heads(self, part: int) -> int:
        return self.heads[part] // (self.world * self.groups)

    def rows(self, part: int) -> int:
        return self.replicated[part] + self.world * self.s_local

    def shard_row(self, part: int, rank: int) -> int:
        return (self.replicated[part] if self.replicated_first else 0) + (
            rank * self.s_local
        )

    def replicated_row(self, part: int) -> int:
        return 0 if self.replicated_first else self.world * self.s_local

    @property
    def out_rows(self) -> int:
        return self.replicated[0] + self.s_local

    @property
    def out_shard_row(self) -> int:
        return self.replicated[0] if self.replicated_first else 0

    @property
    def out_replicated_row(self) -> int:
        return 0 if self.replicated_first else self.s_local

    def slot_numel(self, part: int) -> int:
        return self.batch * self.rows(part) * self.group_heads(part) * self.head_dim

    def out_block_numel(self) -> int:
        return self.batch * self.out_rows * self.group_heads(0) * self.head_dim


def _auto_pipeline_groups(plan: _PipelinePlan) -> int:
    """The most head groups whose calls still fill the GPU, or 0 for none."""
    tiles = plan.batch * -(-plan.rows(0) // 128)
    floor = _MIN_GROUP_WAVES * _sm_count(torch.cuda.current_device())
    for n in _AUTO_PIPELINE_GROUPS:
        split = plan.world * n
        if (
            all(h % split == 0 for h in plan.heads)
            and plan.heads[0] // split * tiles >= floor
        ):
            return n
    return 0


def _member_devices(group, device: int) -> list[int]:
    """Local CUDA ordinal of every member of a same-host group."""
    world_size = dist.get_world_size(group=group)
    members: list = [None] * world_size
    dist.all_gather_object(members, (socket.gethostname(), device), group=group)
    if len({host for host, _ in members}) != 1:
        raise _Unsupported("requires every Ulysses rank on the same host")
    devices = [dev for _, dev in members]
    if len(set(devices)) != world_size:
        raise _Unsupported(f"Ulysses ranks share CUDA devices {devices}")
    return devices


_CU_MEMORYTYPE_DEVICE = 2


def _memcpy2d_struct():
    import ctypes

    class Memcpy2D(ctypes.Structure):  # CUDA_MEMCPY2D
        _fields_ = [
            ("srcXInBytes", ctypes.c_size_t),
            ("srcY", ctypes.c_size_t),
            ("srcMemoryType", ctypes.c_int),
            ("srcHost", ctypes.c_void_p),
            ("srcDevice", ctypes.c_uint64),
            ("srcArray", ctypes.c_void_p),
            ("srcPitch", ctypes.c_size_t),
            ("dstXInBytes", ctypes.c_size_t),
            ("dstY", ctypes.c_size_t),
            ("dstMemoryType", ctypes.c_int),
            ("dstHost", ctypes.c_void_p),
            ("dstDevice", ctypes.c_uint64),
            ("dstArray", ctypes.c_void_p),
            ("dstPitch", ctypes.c_size_t),
            ("WidthInBytes", ctypes.c_size_t),
            ("Height", ctypes.c_size_t),
        ]

    return Memcpy2D


_Memcpy2D = _memcpy2d_struct()


class _StreamMemOps:
    """Front-end signal and wait (cuStreamWriteValue32 / cuStreamWaitValue32).

    Neither holds an SM, so a wait queued next to a persistent attention kernel
    costs it nothing (a spin kernel would pin an SM and push an attention CTA
    into the tail), and a signal queued behind a copy lands when the copy
    retires instead of when an SM frees up.
    """

    def __init__(self):
        import ctypes

        cuda = ctypes.CDLL("libcuda.so.1")
        args = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint]
        self._write = cuda.cuStreamWriteValue32_v2
        self._wait = cuda.cuStreamWaitValue32_v2
        for fn in (self._write, self._wait):
            fn.argtypes = args
            fn.restype = ctypes.c_int
        self._copy2d = cuda.cuMemcpy2DAsync_v2
        self._copy2d.argtypes = [ctypes.POINTER(_Memcpy2D), ctypes.c_void_p]
        self._copy2d.restype = ctypes.c_int
        # drivers can disable stream memory operations; find out here, inside
        # the group's agreed initialization, rather than mid-forward
        flag = torch.zeros(1, dtype=torch.int32, device="cuda")
        stream = torch.cuda.Stream()
        try:
            self.write(stream, flag, 1)
            self.wait(stream, flag, 1)
        except RuntimeError as e:
            raise _Unsupported(f"stream memory operations unavailable ({e})") from e
        stream.synchronize()

    def write(self, stream, flag: torch.Tensor | int, value: int) -> None:
        # default flags fence the stream's earlier writes (the copies) first
        ptr = flag if isinstance(flag, int) else flag.data_ptr()
        rc = self._write(stream.cuda_stream, ptr, value & 0xFFFFFFFF, 0)
        if rc:
            raise RuntimeError(f"cuStreamWriteValue32 failed with CUresult {rc}")

    def wait(self, stream, flag: torch.Tensor | int, value: int) -> None:
        # CU_STREAM_WAIT_VALUE_GEQ compares cyclically, so counters may wrap
        ptr = flag if isinstance(flag, int) else flag.data_ptr()
        rc = self._wait(stream.cuda_stream, ptr, value & 0xFFFFFFFF, 0)
        if rc:
            raise RuntimeError(f"cuStreamWaitValue32 failed with CUresult {rc}")

    def copy2d(
        self,
        stream,
        dst: int,
        dst_pitch: int,
        src: int,
        src_pitch: int,
        width: int,
        height: int,
    ) -> None:
        """`height` rows of `width` bytes between pitched device pointers, on the
        copy engine (a peer mapping is a device pointer like any other)."""
        desc = _Memcpy2D(
            srcMemoryType=_CU_MEMORYTYPE_DEVICE,
            srcDevice=src,
            srcPitch=src_pitch,
            dstMemoryType=_CU_MEMORYTYPE_DEVICE,
            dstDevice=dst,
            dstPitch=dst_pitch,
            WidthInBytes=width,
            Height=height,
        )
        rc = self._copy2d(desc, stream.cuda_stream)
        if rc:
            raise RuntimeError(f"cuMemcpy2DAsync failed with CUresult {rc}")


class _PipelinePool:
    """Receive slots, output blocks and flags that every pipelined call of one
    dtype shares, grown to the largest call seen rather than kept per shape.

    One set serves every call, with no slot per call parity. Rank r sends a
    group's output blocks only after attending over its slot, and peer p
    returns from a call only after merging all of r's blocks; p writes r's
    slots for the next call only after that, so after r has read them. And p
    writes its next block into r's output block only after attending over r's
    rows of that call, which r sends only after its own previous call
    returned, so after r merged p's previous block.
    """

    def __init__(self, state, dtype, in_numel: int, out_numel: int, groups: int):
        world = state.world
        self.sizes = (in_numel, out_numel, groups)
        zeros = lambda n, dt=dtype: ipc_shareable_zeros(n, dtype=dt)
        self.inb = state._share(zeros(in_numel), state.group)
        self.outb = state._share(zeros(out_numel), state.group)
        # fin[p]: groups peer p has delivered; fout[g, p]: calls whose group g
        # peer p has delivered. Groups finish on separate streams, so a single
        # output counter could be lowered by a signal that lands late.
        self.fin = state._share(zeros(world, dt=torch.int32), state.group)
        self.fout = state._share(zeros(groups * world, dt=torch.int32), state.group)
        self.inb_ptr = [t.data_ptr() for t in self.inb]
        self.outb_ptr = [t.data_ptr() for t in self.outb]
        self.fin_ptr = [t.data_ptr() for t in self.fin]
        self.fout_ptr = [t.data_ptr() for t in self.fout]
        self.calls = 0
        self.groups_sent = 0
        # the caller's q/k for each peer block when a fill writes them; v moves
        # straight from its source
        self.send = None

    def nbytes(self, esz: int) -> int:
        return (self.sizes[0] + self.sizes[1]) * esz


class _PipelineSlots:
    """Where one plan's receive slots and output blocks sit in the pool.

    Receive slots, per head group and part: one contiguous ``[batch, rows,
    group_heads, head_dim]`` tensor laid out as attention reads it (see
    ``_PipelinePlan``). Output blocks are source-major: block ``(source,
    group)`` is that rank's output rows for this rank, one ``[batch, out_rows,
    group_heads, head_dim]`` block in the output's row order, copied into the
    merged output as it arrives.
    """

    def __init__(self, plan: _PipelinePlan, dtype):
        self.plan = plan
        self.dtype = dtype
        numel = [plan.slot_numel(part) for part in range(3)]
        self.part_numel = numel
        self.part_offset = (0, numel[0], numel[0] + numel[1])
        self.group_numel = sum(numel)
        self.out_numel = plan.out_block_numel()
        self.in_total = plan.groups * self.group_numel
        self.out_total = plan.world * plan.groups * self.out_numel
        # raw addresses: the copies are issued per call from plain integers,
        # since a tensor view per copy costs more host time than the copy
        self.esz = torch.empty((), dtype=dtype).element_size()
        self.rows = [plan.rows(part) for part in range(3)]
        self.row_bytes = [
            plan.group_heads(part) * plan.head_dim * self.esz for part in range(3)
        ]
        self.send_numel = 2 * plan.s_local * plan.group_heads(0) * plan.head_dim

    def slot_ptr(self, pool, member, group, part, batch_row, row) -> int:
        elems = group * self.group_numel + self.part_offset[part]
        return (
            pool.inb_ptr[member]
            + elems * self.esz
            + (batch_row * self.rows[part] + row) * self.row_bytes[part]
        )

    def out_block_ptr(self, pool, member, source, group) -> int:
        block = (source * self.plan.groups + group) * self.out_numel
        return pool.outb_ptr[member] + block * self.esz

    def slot(self, pool, member: int, group: int, part: int) -> torch.Tensor:
        """``member``'s receive slot, mapped in this rank's context."""
        plan = self.plan
        start = group * self.group_numel + self.part_offset[part]
        return (
            pool.inb[member]
            .narrow(0, start, self.part_numel[part])
            .view(plan.batch, plan.rows(part), plan.group_heads(part), plan.head_dim)
        )

    def send_block(self, pool, peer: int, group: int) -> torch.Tensor:
        plan = self.plan
        need = plan.world * plan.groups * self.send_numel
        if pool.send is None or pool.send.numel() < need:
            pool.send = torch.empty(need, dtype=self.dtype, device="cuda")
        return pool.send.narrow(
            0, (peer * plan.groups + group) * self.send_numel, self.send_numel
        ).view(2, plan.s_local, plan.group_heads(0), plan.head_dim)


class IpcA2AMultiState:
    def __init__(self):
        self.ops = None
        # insertion-ordered so eviction is identical on every rank
        self.staging = OrderedDict()
        self.flags = None
        self.peer_flags = None
        self.seq = None
        self.timed_out = None
        self.peer_timed_out = None
        self.budget_ns = 0
        self.max_buffers = 0
        self.calls = 0
        self.rank = None
        self.world = 0
        self.group = None
        self.failed = False
        self.inited = False
        self.memops = None
        self.pipe_in = None
        self.pipe_group_streams = []
        # pipeline shapes whose buffers did not fit on some rank; agreed by all
        self.declined = set()
        # (shape, caller signature) checked against the sequential exchange,
        # by outcome; agreed by all
        self.verified = set()
        self.mismatched = set()
        # call shapes -> the plan they resolve to, None when they cannot
        # pipeline; repeated calls skip straight to the answer
        self.plans = {}
        # dtype -> the _PipelinePool every pipelined call of that dtype uses
        self.pools = {}
        # ("pipeline", plan, dtype) -> its _PipelineSlots
        self.slots = {}

    def reset(self) -> None:
        """Drop mappings that belong to a model-parallel group being replaced."""
        self.__init__()

    def drop_staging(self) -> None:
        if not self.staging and not self.pools:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.staging.clear()
        self.pools.clear()

    def _share(self, t: torch.Tensor, group) -> list[torch.Tensor]:
        """Every member's `t`, peers' re-opened in the LOCAL device context
        (a mapping is only dereferenceable from the context that opened it)."""
        from torch.multiprocessing.reductions import reduce_tensor

        handles: list = [None] * self.world
        dist.all_gather_object(handles, reduce_tensor(t), group=group)
        dev = torch.cuda.current_device()
        out = []
        for member, (fn, args) in enumerate(handles):
            if member == self.rank:
                out.append(t)
                continue
            args = list(args)
            for i, v in enumerate(args):
                if isinstance(v, torch.device):
                    args[i] = torch.device(f"cuda:{dev}")
                elif isinstance(v, int) and i == 6:
                    # rebuild_cuda_tensor positional device index
                    args[i] = dev
            out.append(fn(*args))
        return out

    def init(self, group) -> None:
        import ctypes

        self.rank = dist.get_rank(group=group)
        self.world = dist.get_world_size(group=group)
        self.group = group
        dev = torch.cuda.current_device()
        devices = _member_devices(group, dev)
        peers = [d for member, d in enumerate(devices) if member != self.rank]
        missing = [d for d in peers if not torch.cuda.can_device_access_peer(dev, d)]
        memops_error = None
        try:
            memops = _StreamMemOps()
        except _Unsupported as e:
            memops_error = e
        # agree before the handle exchange below: a rank that bailed out here
        # alone would leave the others blocked inside that collective
        usable = int(not missing and memops_error is None)
        agreed = torch.tensor([usable], dtype=torch.int32, device="cuda")
        dist.all_reduce(agreed, op=dist.ReduceOp.MIN, group=group)
        if agreed.item() == 0:
            if missing:
                raise _Unsupported(
                    f"a Ulysses rank lacks peer-to-peer access (CUDA device {dev} "
                    f"cannot reach {missing})"
                )
            raise _Unsupported(
                str(memops_error) if memops_error else "a Ulysses peer is unsupported"
            )
        cudart = ctypes.CDLL("libcudart.so")
        for peer_dev in peers:
            # kernel-level dereference of peer mappings needs explicit peer access
            cudart.cudaDeviceEnablePeerAccess(peer_dev, 0)
        self.ops = load_ipc_a2a_sync()
        self.memops = memops
        # flags[p] is written by peer p; seq[p] counts the calls sent to p
        self.flags = torch.zeros(self.world, dtype=torch.int32, device="cuda")
        self.seq = torch.zeros(self.world, dtype=torch.int32, device="cuda")
        self.timed_out = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.budget_ns = int(envs.SGLANG_DIFFUSION_IPC_A2A_TIMEOUT_MS * 1e6)
        self.max_buffers = envs.SGLANG_DIFFUSION_IPC_A2A_MAX_BUFFERS
        self.peer_flags = self._share(self.flags, group)
        self.peer_timed_out = self._share(self.timed_out, group)
        self.inited = True

    def get_staging(self, chunk: int, dtype: torch.dtype) -> list[torch.Tensor] | None:
        """Every member's two-slot `[2, world, chunk]` receive buffer (mine at
        my rank). Creation is a group collective, so every rank must reach a
        new key at the same call site; a miss during capture returns None."""
        key = (chunk, dtype)
        buffers = self.staging.get(key)
        if buffers is None:
            if torch.cuda.is_current_stream_capturing():
                return None
            local = torch.zeros(2, self.world, chunk, dtype=dtype, device="cuda")
            buffers = self._share(local, self.group)
            self._insert(key, buffers)
        return buffers

    def _insert(self, key, value) -> None:
        if len(self.staging) >= self.max_buffers:
            # queued copies may still target the evicted peer mappings
            torch.cuda.synchronize()
            self.staging.popitem(last=False)
        self.staging[key] = value

    def pipelined_attention(
        self,
        q,
        k,
        v,
        attend,
        groups: int,
        fill=None,
        sequential=None,
        signature=None,
        replicated=(None, None, None),
        replicated_first: bool = True,
    ):
        """Ulysses attention with the head exchange pipelined against it.

        q/k/v are ``[batch, s_local, heads, head_dim]`` sequence shards with
        every head (``[s_local, heads, head_dim]`` reads as batch 1 and returns
        that shape); k and v may have fewer heads than q (GQA).
        ``replicated`` optionally holds, per part, ``[batch, n, heads,
        head_dim]`` rows every rank holds (text tokens); they stay out of the
        exchange, attend before the gathered sequence when
        ``replicated_first`` and after it otherwise, and their output, all
        heads, leads (or trails) the returned rows.

        ``attend(q, k, v)`` runs attention on one head group, ``[batch, rows,
        heads / (world * groups), head_dim]`` per part in the sequence order
        above, and must treat heads independently, which keeps the result
        bit-identical to the sequential exchange. Group g's copies run on the
        copy engine while group g-1 attends, and each group attends on its own
        stream and sends its output from there, so no stream waits on a kernel
        still running on another. Every copy, the merge into the output
        included, is a pitched copy-engine copy, so the exchange takes no SMs.
        Returns the output rows of this rank, or None when this call cannot
        pipeline. ``groups < 0`` picks the count and steps aside, on every
        rank, when the buffers would not fit.

        ``fill(head_start, head_count, q_dst, k_dst)``, when given (batch 1,
        no replicated rows, as many k as q heads), writes those heads' q and k
        into two [s_local, head_count, head_dim] blocks in place of a copy
        (the caller's QK-norm writes its output there); blocks for this rank
        go straight into its own receive slot, and v always moves straight
        from ``v``. It is only called once the call is known to pipeline.

        ``sequential()``, when given, returns what the sequential exchange
        gives for this call. The first call per plan and ``signature`` (the
        caller's attention setup) runs both and keeps the pipeline for them
        only if every rank sees the same bytes, so a backend that does not
        treat a head group as it treats those heads in the full call never
        changes a result. It runs the sequential exchange first, with the
        pool released, so that call never holds both at once.
        """
        world, r = self.world, self.rank
        squeeze = q.dim() == 3
        if squeeze:
            q, k, v = q[None], k[None], v[None]
        parts = (q, k, v)
        replicated = tuple(replicated)
        batch, s_local, _, head_dim = q.shape
        auto = groups < 0
        shapes = (
            q.shape,
            k.shape,
            v.shape,
            tuple(None if t is None else t.shape for t in replicated),
            replicated_first,
            groups,
            fill is not None,
        )
        if shapes not in self.plans:
            self.plans[shapes] = self._plan(
                q, k, v, replicated, replicated_first, groups, fill is not None
            )
        plan = self.plans[shapes]
        if plan is None:
            return None
        groups = plan.groups

        # each token's heads of one group must be one contiguous run to move
        # with a pitched copy
        copied = [v] if fill is not None else [q, k, v]
        copied += [t for t in replicated if t is not None]
        if not all(t.stride(-1) == 1 and t.stride(-2) == head_dim for t in copied):
            return None
        key = ("pipeline", plan, q.dtype)
        check = None if sequential is None else (key, signature)
        if check in self.mismatched:
            return None
        if check in self.verified:
            check = None
        slots = self.slots.get(key)
        if slots is None:
            slots = self.slots[key] = _PipelineSlots(plan, q.dtype)
        pool = self.pools.get(q.dtype)
        sizes = (slots.in_total, slots.out_total, groups)
        if pool is not None:
            sizes = tuple(map(max, sizes, pool.sizes))
        reference = None
        if pool is None or pool.sizes != sizes or check is not None:
            if torch.cuda.is_current_stream_capturing() or key in self.declined:
                return None
            esz = q.element_size()
            held = 0 if pool is None else pool.nbytes(esz)
            need = (sizes[0] + sizes[1]) * esz
            if fill is not None:
                need += plan.world * groups * slots.send_numel * esz
            if auto and not self._fits_on_every_rank(need, reclaimable=held):
                logger.info(
                    "Ulysses pipeline buffers (%.1f GiB) do not fit; using the "
                    "sequential exchange for this shape",
                    need / 2**30,
                )
                self.declined.add(key)
                return None
            if pool is not None:
                # queued copies may still target the old mappings
                torch.cuda.synchronize()
                del self.pools[q.dtype]
                pool = None
            if check is not None:
                reference = sequential()
            pool = self.pools[q.dtype] = _PipelinePool(self, q.dtype, *sizes)
        if self.pipe_in is None:
            self.pipe_in = torch.cuda.Stream()
        while len(self.pipe_group_streams) < groups:
            self.pipe_group_streams.append(torch.cuda.Stream())
        mem, cin = self.memops, self.pipe_in
        main = torch.cuda.current_stream()
        base = pool.groups_sent
        pool.groups_sent += groups
        pool.calls += 1
        call = pool.calls
        peers = [(r + step) % world for step in range(1, world)]
        esz = q.element_size()
        head_bytes = head_dim * esz
        row_bytes = slots.row_bytes
        # first head of block (p, g) of a part
        start = lambda p, g, part: (p * groups + g) * plan.group_heads(part)
        # (address, batch pitch, token pitch) of each part's shard and replicated rows
        src = [(t.data_ptr(), t.stride(0) * esz, t.stride(1) * esz) for t in parts]
        rep_src = [
            None if t is None else (t.data_ptr(), t.stride(0) * esz, t.stride(1) * esz)
            for t in replicated
        ]
        my_shard_row = [plan.shard_row(part, r) for part in range(3)]

        def send_rows(stream, part, p, g):
            # my shard of one part and head group into rank p's receive slot
            ptr, batch_pitch, token_pitch = src[part]
            ptr += start(p, g, part) * head_bytes
            for b in range(batch):
                mem.copy2d(
                    stream,
                    slots.slot_ptr(pool, p, g, part, b, my_shard_row[part]),
                    row_bytes[part],
                    ptr + b * batch_pitch,
                    token_pitch,
                    row_bytes[part],
                    s_local,
                )

        def place_replicated(stream, part, g):
            # rows every rank holds go straight into my own slot
            n = plan.replicated[part]
            if not n:
                return
            ptr, batch_pitch, token_pitch = rep_src[part]
            ptr += start(r, g, part) * head_bytes
            row = plan.replicated_row(part)
            for b in range(batch):
                mem.copy2d(
                    stream,
                    slots.slot_ptr(pool, r, g, part, b, row),
                    row_bytes[part],
                    ptr + b * batch_pitch,
                    token_pitch,
                    row_bytes[part],
                    n,
                )

        ready = main.record_event()
        own, sent = [], []
        if fill is not None:
            # group by group: the peers' blocks first, so their copies start
            # while this rank fills its own block straight into its slot
            own_rows = lambda part, g: slots.slot(pool, r, g, part)[
                0, my_shard_row[part] : my_shard_row[part] + s_local
            ]
            hg = plan.group_heads(0)
            for g in range(groups):
                for p in peers:
                    qk = slots.send_block(pool, p, g)
                    fill(start(p, g, 0), hg, qk[0], qk[1])
                sent.append(main.record_event())
                fill(start(r, g, 0), hg, own_rows(0, g), own_rows(1, g))
                own.append(main.record_event())
        own_cin = []
        cin.wait_event(ready)
        with torch.cuda.stream(cin):
            for g in range(groups):
                for p in (r, *peers):
                    send_rows(cin, 2, p, g)
                    if fill is None:
                        send_rows(cin, 0, p, g)
                        send_rows(cin, 1, p, g)
                    if p == r:
                        for part in range(3):
                            place_replicated(cin, part, g)
                        own_cin.append(cin.record_event())
                if fill is not None:
                    cin.wait_event(sent[g])
                    chunk = s_local * row_bytes[0]
                    for p in peers:
                        q_slot = slots.slot_ptr(pool, p, g, 0, 0, my_shard_row[0])
                        k_slot = slots.slot_ptr(pool, p, g, 1, 0, my_shard_row[1])
                        # q then k: two rows of `chunk`, one part apart in the slot
                        mem.copy2d(
                            cin,
                            q_slot,
                            k_slot - q_slot,
                            slots.send_block(pool, p, g).data_ptr(),
                            chunk,
                            chunk,
                            2,
                        )
                for p in peers:
                    mem.write(cin, pool.fin_ptr[p] + 4 * r, base + g + 1)
        # Head block b = p * groups + g of the output is group g of rank p's
        # heads. Each group stream writes its blocks as soon as they exist, so
        # the merge overlaps the groups still attending instead of trailing them.
        hq = plan.group_heads(0)
        merged = torch.empty(
            batch,
            plan.out_rows,
            plan.heads[0],
            head_dim,
            dtype=q.dtype,
            device=q.device,
        )
        merged_ptr = merged.data_ptr()
        merged_row = plan.heads[0] * head_bytes
        out_row = row_bytes[0]
        block_batch = plan.out_rows * out_row
        shard_out, rep_out = plan.out_shard_row, plan.out_replicated_row
        n_rep, rep_row = plan.replicated[0], plan.replicated_row(0)
        done = []
        for g in range(groups):
            stream = self.pipe_group_streams[g]
            if fill is not None:
                stream.wait_event(own[g])
            stream.wait_event(own_cin[g])
            with torch.cuda.stream(stream):
                for p in peers:
                    mem.wait(stream, pool.fin_ptr[r] + 4 * p, base + g + 1)
                views = [slots.slot(pool, r, g, part) for part in range(3)]
                if squeeze:
                    out = attend(*(t[0] for t in views)).contiguous()[None]
                else:
                    out = attend(*views).contiguous()
                out_ptr = out.data_ptr()
                out_batch = out.stride(0) * esz
                for p in peers:
                    # this peer's rows and the replicated rows every rank needs,
                    # one contiguous run per batch row
                    block = slots.out_block_ptr(pool, p, r, g)
                    runs = [(shard_out, plan.shard_row(0, p), s_local)]
                    if n_rep:
                        runs.append((rep_out, rep_row, n_rep))
                    for dst_row, src_row, n in runs:
                        mem.copy2d(
                            stream,
                            block + dst_row * out_row,
                            block_batch,
                            out_ptr + src_row * out_row,
                            out_batch,
                            n * out_row,
                            batch,
                        )
                for p in peers:
                    mem.write(stream, pool.fout_ptr[p] + 4 * (g * world + r), call)
                own_head = merged_ptr + start(r, g, 0) * head_bytes
                for b in range(batch):
                    runs = [(shard_out, my_shard_row[0], s_local)]
                    if n_rep:
                        runs.append((rep_out, rep_row, n_rep))
                    for dst_row, src_row, n in runs:
                        mem.copy2d(
                            stream,
                            own_head + (b * plan.out_rows + dst_row) * merged_row,
                            merged_row,
                            out_ptr + b * out_batch + src_row * out_row,
                            out_row,
                            out_row,
                            n,
                        )
                for p in peers:
                    mem.wait(stream, pool.fout_ptr[r] + 4 * (g * world + p), call)
                    # the block and the merged output both have one row pitch
                    # across the batch, so a single copy covers every batch row
                    mem.copy2d(
                        stream,
                        merged_ptr + start(p, g, 0) * head_bytes,
                        merged_row,
                        slots.out_block_ptr(pool, r, p, g),
                        out_row,
                        out_row,
                        batch * plan.out_rows,
                    )
                done.append(stream.record_event())
        for event in done:
            main.wait_event(event)
        # the next call's fill rewrites `send` and the caller may free `v`, both
        # read by the copy stream
        main.wait_stream(cin)
        if squeeze:
            merged = merged[0]
        if check is not None:
            return self._first_sight(check, merged, reference)
        return merged

    def _plan(self, q, k, v, replicated, replicated_first, groups, filled):
        """The plan these shapes pipeline with, or None; rank-independent."""
        world = self.world
        batch, s_local, _, head_dim = q.shape
        parts = (q, k, v)
        rows = tuple(0 if t is None else t.shape[1] for t in replicated)
        if (
            k.shape != v.shape
            or any(
                t.shape[:2] != (batch, s_local) or t.shape[3] != head_dim for t in parts
            )
            or rows[1] != rows[2]
            or any(
                t is not None and t.shape != (batch, n, parts[i].shape[2], head_dim)
                for i, (t, n) in enumerate(zip(replicated, rows))
            )
            or (filled and (batch != 1 or any(rows) or q.shape != k.shape))
        ):
            return None
        plan = _PipelinePlan(
            batch=batch,
            s_local=s_local,
            heads=tuple(t.shape[2] for t in parts),
            head_dim=head_dim,
            replicated=rows,
            replicated_first=replicated_first,
            world=world,
            groups=groups,
        )
        if groups < 0:
            plan = dataclasses.replace(plan, groups=_auto_pipeline_groups(plan))
        if not plan.groups or any(h % (world * plan.groups) for h in plan.heads):
            return None
        return plan

    def _first_sight(self, check, merged, reference):
        """Keep the pipeline for `check` only if it matches the sequential
        exchange's `reference` byte for byte on every rank; a group collective."""
        same = torch.tensor(
            [int(torch.equal(merged, reference))], dtype=torch.int32, device="cuda"
        )
        dist.all_reduce(same, op=dist.ReduceOp.MIN, group=self.group)
        if same.item():
            self.verified.add(check)
            return merged
        self.mismatched.add(check)
        plan = check[0][1]
        logger.info(
            "Ulysses pipeline over %d head groups differs from the sequential "
            "exchange for %s under %s; keeping the sequential exchange",
            plan.groups,
            plan,
            check[1],
        )
        return reference

    def _fits_on_every_rank(self, nbytes: int, reclaimable: int = 0) -> bool:
        """Whether every rank can spare twice `nbytes`, counting the allocator's
        cached blocks and `reclaimable` bytes it is about to free; a group
        collective, so all ranks reach the same answer."""
        free, _ = torch.cuda.mem_get_info()
        cached = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
        fits = torch.tensor(
            [int(free + cached + reclaimable >= 2 * nbytes)],
            dtype=torch.int32,
            device="cuda",
        )
        dist.all_reduce(fits, op=dist.ReduceOp.MIN, group=self.group)
        return bool(fits.item())

    def exchange(self, send: torch.Tensor) -> torch.Tensor | None:
        """``all_to_all_single`` with equal splits: row p of the contiguous
        ``[world, chunk]`` `send` goes to rank p, and row p of the returned
        ``[world, chunk]`` view is what rank p sent me."""
        if send.shape[0] != self.world or not send.is_contiguous():
            return None
        buffers = self.get_staging(send[0].numel(), send.dtype)
        if buffers is None:
            return None
        slot = self.calls % 2
        self.calls += 1
        r = self.rank
        buffers[r][slot, r].copy_(send[r].view(-1), non_blocking=True)
        # Step k sends to r + k: every step is a permutation, so each rank
        # receives from exactly one peer at a time instead of all of them
        # converging on the same destination.
        peers = [(r + k) % self.world for k in range(1, self.world)]
        for p in peers:
            buffers[p][slot, r].copy_(send[p].view(-1), non_blocking=True)
            self.ops.bump_signal(
                self.seq.narrow(0, p, 1), self.peer_flags[p].narrow(0, r, 1)
            )
        for p in peers:
            self.ops.spin_wait(
                self.flags.narrow(0, p, 1),
                self.seq.narrow(0, p, 1),
                self.timed_out,
                self.peer_timed_out[p],
                self.budget_ns,
            )
        return buffers[r][slot]

    def check_timeout(self) -> None:
        """Retire the transport on every rank if any rank's spin expired.

        A spin only flags the peer it waited on, so agree across the whole group
        at the request boundary: a rank that retired alone would post an NCCL
        all-to-all that its IPC peers never join. Expiry means a peer went
        silent for the whole budget and that exchange returned incomplete data,
        so this raises instead of serving the result.
        """
        if self.failed or not self.inited:
            return
        expired = self.timed_out.clone()
        dist.all_reduce(expired, op=dist.ReduceOp.MAX, group=self.group)
        if expired.item() == 0:
            return
        self.failed = True
        raise RuntimeError(
            "IPC all-to-all gave up waiting for a peer after "
            f"{envs.SGLANG_DIFFUSION_IPC_A2A_TIMEOUT_MS:g} ms, so that exchange "
            "returned incomplete data. The transport is now disabled on every "
            "rank; retry the request over NCCL, or set "
            "SGLANG_DIFFUSION_IPC_A2A_MULTI=0."
        )


IPC_A2A_MULTI = IpcA2AMultiState()


def ipc_a2a_multi_ready(group, enabled: bool | None = None) -> bool:
    """True when the N-rank transport is enabled and initialized on every
    member of `group` (initializes lazily on the first eager call)."""
    from sglang.multimodal_gen.runtime.platforms import current_platform

    if not (envs.SGLANG_DIFFUSION_IPC_A2A_MULTI if enabled is None else enabled):
        return False
    if IPC_A2A_MULTI.group is not None and IPC_A2A_MULTI.group is not group:
        IPC_A2A_MULTI.reset()
    if IPC_A2A_MULTI.failed:
        return False
    if IPC_A2A_MULTI.inited:
        return True
    if not current_platform.is_cuda() or torch.cuda.is_current_stream_capturing():
        return False
    ok = True
    try:
        IPC_A2A_MULTI.init(group)
    except _Unsupported as e:
        logger.info("IPC all-to-all (multi-rank) unavailable (%s); using NCCL", e)
        ok = False
    except Exception as e:
        logger.debug("IPC all-to-all (multi-rank) initialization failed", exc_info=True)
        logger.warning("IPC all-to-all (multi-rank) unavailable (%s); using NCCL", e)
        ok = False
    # every member must take the same path, or an IPC rank waits on an NCCL one
    agreed = torch.tensor([int(ok)], dtype=torch.int32, device="cuda")
    dist.all_reduce(agreed, op=dist.ReduceOp.MIN, group=group)
    if agreed.item() == 0:
        IPC_A2A_MULTI.reset()
        IPC_A2A_MULTI.failed = True
        return False
    return True


def ulysses_pipelined_attention(
    q,
    k,
    v,
    attend,
    groups: int,
    fill=None,
    sequential=None,
    signature=None,
    replicated=(None, None, None),
    replicated_first: bool = True,
):
    """Pipelined Ulysses attention over the copy-engine transport, or None
    when the Ulysses group cannot use it (the caller keeps its NCCL path).
    See ``IpcA2AMultiState.pipelined_attention`` for the arguments."""
    from sglang.multimodal_gen.runtime.distributed.parallel_state import get_sp_group

    group = get_sp_group().ulysses_group
    if group is None or not ipc_a2a_multi_ready(group, enabled=True):
        return None
    return IPC_A2A_MULTI.pipelined_attention(
        q,
        k,
        v,
        attend,
        groups,
        fill=fill,
        sequential=sequential,
        signature=signature,
        replicated=replicated,
        replicated_first=replicated_first,
    )
