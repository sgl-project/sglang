# SPDX-License-Identifier: Apache-2.0
"""Copy-engine transport for N-rank Ulysses all-to-all on one host.

The two-rank ``ipc_a2a`` design widened to every peer: each rank maps the
peers' staging buffers into its own device context and writes its chunk for a
peer straight into that peer's buffer. A contiguous device-to-device copy runs
on the copy engine, so the exchange takes no SMs from the kernels around it --
NCCL's all-to-all is an SM kernel, and next to a full-occupancy attention it
slows that attention by more than the exchange costs.

Per peer, a sequence counter published after the copy orders the data before
the flag, and the receiver spins on that peer's entry. Double-buffered slots
alternate per call; a slot is only rewritten two calls later, after a spin on
every peer, and each peer publishes that call only after it has consumed the
slot in stream order.
"""

import logging
import socket
from collections import OrderedDict

import torch
import torch.distributed as dist

from sglang.kernels.ops.communication.ipc_a2a import load_ipc_a2a_sync
from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.distributed.device_communicators.ipc_a2a import (
    _Unsupported,
)

logger = logging.getLogger(__name__)

# Group counts tried, in order, when the caller leaves the choice to the
# transport: MiniMax-H3's 56 heads give 28, 14 and 7 per rank at Ulysses 2, 4
# and 8, and 7 groups divide all three.
_AUTO_PIPELINE_GROUPS = (7, 4, 2)


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

    def write(self, stream, flag: torch.Tensor, value: int) -> None:
        # default flags fence the stream's earlier writes (the copies) first
        rc = self._write(stream.cuda_stream, flag.data_ptr(), value & 0xFFFFFFFF, 0)
        if rc:
            raise RuntimeError(f"cuStreamWriteValue32 failed with CUresult {rc}")

    def wait(self, stream, flag: torch.Tensor, value: int) -> None:
        # CU_STREAM_WAIT_VALUE_GEQ compares cyclically, so counters may wrap
        rc = self._wait(stream.cuda_stream, flag.data_ptr(), value & 0xFFFFFFFF, 0)
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


class _PipelineBuffers:
    """Receive slots and flags for one pipelined-attention shape.

    Input slots are [2, groups, 3, world, chunk]: group g's q (k, v) rows from
    every rank form one contiguous [S, group_heads, head_dim] tensor for
    attention. Output slots are source-major ([2, world, groups, chunk]), each
    block a peer's rows for one head group, copied into the merged output as it
    arrives.
    """

    def __init__(self, state, s_local, heads, head_dim, groups, dtype):
        world = state.world
        self.group_heads = heads // world // groups
        rows = s_local * self.group_heads
        zeros = lambda *shape, dt=dtype: torch.zeros(*shape, dtype=dt, device="cuda")
        self.inb = state._share(
            zeros(2, groups, 3, world, rows * head_dim), state.group
        )
        self.outb = state._share(zeros(2, world, groups, rows * head_dim), state.group)
        # fin[p]: groups peer p has delivered; fout[g, p]: calls whose group g
        # peer p has delivered. Groups finish on separate streams, so a single
        # output counter could be lowered by a signal that lands late.
        self.fin = state._share(zeros(world, dt=torch.int32), state.group)
        self.fout = state._share(zeros(groups, world, dt=torch.int32), state.group)
        # the caller's q/k for each peer block; v moves straight from its source
        self.send = torch.empty(
            world * groups,
            2,
            s_local,
            self.group_heads,
            head_dim,
            dtype=dtype,
            device="cuda",
        )
        self.calls = 0


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

    def reset(self) -> None:
        """Drop mappings that belong to a model-parallel group being replaced."""
        self.__init__()

    def drop_staging(self) -> None:
        if not self.staging:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.staging.clear()

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

    def pipelined_attention(self, q, k, v, attend, groups: int, fill=None):
        """Ulysses attention with the head exchange pipelined against it.

        q/k/v are [s_local, heads, head_dim], sequence-sharded with every head.
        ``attend(q, k, v)`` runs attention on [S, heads / (world * groups),
        head_dim] tensors of one head group and must treat heads independently,
        which keeps the result bit-identical to the sequential exchange.
        Group g's copies run on the copy engine while group g-1 attends, and
        each group attends on its own stream and sends its output from there,
        so no stream waits on a kernel still running on another. Every copy,
        the merge into the output included, is a pitched copy-engine copy, so
        the exchange takes no SMs. Returns [s_local, heads, head_dim], or None
        when this call cannot pipeline. ``groups < 0`` picks the count and
        steps aside, on every rank, when the buffers would not fit.

        ``fill(head_start, head_count, q_dst, k_dst)``, when given, writes those
        heads' q and k into two [s_local, head_count, head_dim] blocks in place
        of a copy (the caller's QK-norm writes its output there); blocks for
        this rank go straight into its own receive slot, and v always moves
        straight from ``v``. It is only called once the call is known to
        pipeline.
        """
        world, r = self.world, self.rank
        s_local, heads, head_dim = q.shape
        auto = groups < 0
        if auto:
            groups = next(
                (n for n in _AUTO_PIPELINE_GROUPS if heads % (world * n) == 0), 0
            )
        # each token's heads of one group must be one contiguous run to move
        # with a pitched copy
        copied = (v,) if fill is not None else (q, k, v)
        if (
            not groups
            or heads % (world * groups)
            or not (q.shape == k.shape == v.shape)
            or any(t.stride(2) != 1 or t.stride(1) != head_dim for t in copied)
        ):
            return None
        key = ("pipeline", s_local, heads, head_dim, groups, q.dtype)
        bufs = self.staging.get(key)
        if bufs is None:
            if torch.cuda.is_current_stream_capturing() or key in self.declined:
                return None
            # send + both receive and output slots: 10 * s_local * heads * head_dim
            need = 10 * s_local * heads * head_dim * q.element_size()
            if auto and not self._fits_on_every_rank(need):
                logger.info(
                    "Ulysses pipeline buffers (%.1f GiB) do not fit; using the "
                    "sequential exchange for this shape",
                    need / 2**30,
                )
                self.declined.add(key)
                return None
            bufs = _PipelineBuffers(self, s_local, heads, head_dim, groups, q.dtype)
            self._insert(key, bufs)
        if self.pipe_in is None:
            self.pipe_in = torch.cuda.Stream()
        while len(self.pipe_group_streams) < groups:
            self.pipe_group_streams.append(torch.cuda.Stream())
        mem, cin, hg = self.memops, self.pipe_in, bufs.group_heads
        main = torch.cuda.current_stream()
        slot = bufs.calls % 2
        base = bufs.calls * groups
        bufs.calls += 1
        call = bufs.calls
        peers = [(r + step) % world for step in range(1, world)]
        esz = q.element_size()
        width = hg * head_dim * esz  # one token's heads of one group
        chunk = s_local * width  # one rank's rows of one group, one of q/k/v
        start = lambda p, g: (p * groups + g) * hg  # first head of block (p, g)
        slot_of = lambda p, g, part: bufs.inb[p][slot, g, part, r]  # my rows there

        def send_rows(stream, t, p, g, part):
            src = t[:, start(p, g)]
            mem.copy2d(
                stream,
                slot_of(p, g, part).data_ptr(),
                width,
                src.data_ptr(),
                t.stride(0) * esz,
                width,
                s_local,
            )

        ready = main.record_event()
        own, sent = [], []
        if fill is not None:
            # group by group: the peers' blocks first, so their copies start
            # while this rank fills its own block straight into its slot
            block = lambda t: t.view(s_local, hg, head_dim)
            for g in range(groups):
                for p in peers:
                    qk = bufs.send[p * groups + g]
                    fill(start(p, g), hg, qk[0], qk[1])
                sent.append(main.record_event())
                fill(start(r, g), hg, block(slot_of(r, g, 0)), block(slot_of(r, g, 1)))
                own.append(main.record_event())
        own_cin = []
        cin.wait_event(ready)
        with torch.cuda.stream(cin):
            for g in range(groups):
                for p in (r, *peers):
                    send_rows(cin, v, p, g, 2)
                    if fill is None:
                        send_rows(cin, q, p, g, 0)
                        send_rows(cin, k, p, g, 1)
                    if p == r:
                        own_cin.append(cin.record_event())
                if fill is not None:
                    cin.wait_event(sent[g])
                    for p in peers:
                        # q then k: two rows of `chunk`, one part apart in the slot
                        mem.copy2d(
                            cin,
                            slot_of(p, g, 0).data_ptr(),
                            slot_of(p, g, 1).data_ptr() - slot_of(p, g, 0).data_ptr(),
                            bufs.send[p * groups + g].data_ptr(),
                            chunk,
                            chunk,
                            2,
                        )
                for p in peers:
                    mem.write(cin, bufs.fin[p].narrow(0, r, 1), base + g + 1)
        # Head block b = p * groups + g of the output is group g of rank p's
        # heads. Each group stream writes its blocks as soon as they exist, so
        # the merge overlaps the groups still attending instead of trailing them.
        merged = torch.empty(s_local, heads, head_dim, dtype=q.dtype, device=q.device)
        merged_blocks = merged.view(s_local, world * groups, hg, head_dim)
        merged_pitch = heads * head_dim * esz
        done = []
        for g in range(groups):
            stream = self.pipe_group_streams[g]
            if fill is not None:
                stream.wait_event(own[g])
            stream.wait_event(own_cin[g])
            with torch.cuda.stream(stream):
                for p in peers:
                    mem.wait(stream, bufs.fin[r].narrow(0, p, 1), base + g + 1)
                part = lambda i: bufs.inb[r][slot, g, i].view(
                    world * s_local, hg, head_dim
                )
                out = attend(part(0), part(1), part(2)).contiguous()
                rows = lambda p: out[p * s_local : (p + 1) * s_local]
                for p in peers:
                    bufs.outb[p][slot, r, g].copy_(
                        rows(p).reshape(-1), non_blocking=True
                    )
                for p in peers:
                    mem.write(stream, bufs.fout[p][g].narrow(0, r, 1), call)
                mem.copy2d(
                    stream,
                    merged_blocks[:, r * groups + g].data_ptr(),
                    merged_pitch,
                    rows(r).data_ptr(),
                    width,
                    width,
                    s_local,
                )
                for p in peers:
                    mem.wait(stream, bufs.fout[r][g].narrow(0, p, 1), call)
                    mem.copy2d(
                        stream,
                        merged_blocks[:, p * groups + g].data_ptr(),
                        merged_pitch,
                        bufs.outb[r][slot, p, g].data_ptr(),
                        width,
                        width,
                        s_local,
                    )
                done.append(stream.record_event())
        for event in done:
            main.wait_event(event)
        # the next call's fill rewrites `send` and the caller may free `v`, both
        # read by the copy stream
        main.wait_stream(cin)
        return merged

    def _fits_on_every_rank(self, nbytes: int) -> bool:
        """Whether every rank can spare twice `nbytes`, counting the allocator's
        cached blocks; a group collective, so all ranks reach the same answer."""
        free, _ = torch.cuda.mem_get_info()
        cached = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
        fits = torch.tensor(
            [int(free + cached >= 2 * nbytes)], dtype=torch.int32, device="cuda"
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


def ulysses_pipelined_attention(q, k, v, attend, groups: int, fill=None):
    """Pipelined Ulysses attention over the copy-engine transport, or None
    when the Ulysses group cannot use it (the caller keeps its NCCL path)."""
    from sglang.multimodal_gen.runtime.distributed.parallel_state import get_sp_group

    group = get_sp_group().ulysses_group
    if group is None or not ipc_a2a_multi_ready(group, enabled=True):
        return None
    return IPC_A2A_MULTI.pipelined_attention(q, k, v, attend, groups, fill=fill)
