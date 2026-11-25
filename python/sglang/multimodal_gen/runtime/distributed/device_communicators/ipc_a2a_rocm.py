# SPDX-License-Identifier: Apache-2.0
"""Copy-engine (SDMA) transport for the Ulysses head-group pipeline on ROCm.

The ROCm counterpart of ``ipc_a2a_multi``'s pipelined attention for the
collective pipeline's layout: q/k/v are packed destination-major by the same
SM kernel, each peer's chunk of a head group moves on the SDMA engines
(``hipMemcpyDeviceToDeviceNoCU``) straight into that peer's receive slot, and
every head group attends on its own stream. RCCL's all-to-all is a kernel of
up to NCCL_MIN_NCHANNELS CTAs, and next to the attention it slows that
attention by more than the exchange costs; the SDMA copies take no CUs.

Signals are 4-byte SDMA copies from a table of sequence numbers into the
peer's flag word, queued behind the data copy on the same stream, so a flag
lands only after its data; ``hipStreamWriteValue32`` is a blit kernel on this
runtime and would take CUs. The receiver waits with ``hipStreamWaitValue32``
(GEQ). Attention reads the receive slots as the collective path's receive
buffer is read, so the result is bit-identical to it.

Slot reuse needs no extra handshake: rank p writes rank r's receive slots for
call c + 1 only after p's call c returned, which waited for r's output blocks
of every group of call c, each sent after r attended over those slots; and p
writes r's output block of call c + 1 only after attending over r's rows of
call c + 1, sent after r's call c returned, which read p's blocks of call c.
"""

import ctypes
import logging
import os

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

_NOCU = 1024  # hipMemcpyDeviceToDeviceNoCU
_WAIT_GEQ = 0
# flag values come from this table; every rank advances its counters alike,
# so all of them reset together before the table runs out
_SEQ_TABLE = int(os.environ.get("SGLANG_DIFFUSION_ROCM_CE_SEQ_TABLE", 1 << 20))
_POOL_HEADROOM = 1.25
# "group": pack q|k|v group by group with torch.cat so the first group's
# copies start before the rest is packed; "kernel": the collective path's
# one-pass pack kernel
_GROUP_PACK = os.environ.get("SGLANG_DIFFUSION_ROCM_CE_PACK", "kernel") == "group"
# with a caller's fill: "group" launches it once per head group so the first
# group's copies start while the rest fills; "all": one launch for every group
_FILL_SPLIT = os.environ.get("SGLANG_DIFFUSION_ROCM_CE_FILL_SPLIT", "all") == "group"
# "first": group 0 in one launch, the other groups in a second
_FILL_FIRST = os.environ.get("SGLANG_DIFFUSION_ROCM_CE_FILL_SPLIT", "all") == "first"
# "merged": peers write each output block straight into this rank's merged
# [rows, heads, head_dim] buffer (pitched copies), so no merge pass follows
_OUT_MERGED = os.environ.get("SGLANG_DIFFUSION_ROCM_CE_OUT", "blocks") == "merged"
# with a fill: merge each group's output blocks on the caller's stream as soon
# as they arrive (it idles while the groups attend), not in one pass at the end
_GROUP_MERGE = os.environ.get("SGLANG_DIFFUSION_ROCM_CE_MERGE", "end") == "group"
# with a fill: at most this many group attentions in flight (0: no limit);
# group g starts after group g - N has attended
_MAX_CONC = int(os.environ.get("SGLANG_DIFFUSION_ROCM_CE_MAX_CONC", "0"))
# the last head group, which nothing else overlaps (comma list):
#   klast: its output blocks go to every peer in one copy kernel, not in one
#     SDMA copy per peer one after another on its stream (an extra stream per
#     peer stalls the other groups: its waits hold their streams back)
#   tlast: its key-split tail runs on a side stream beside its full-wave launch
_LAST = frozenset(
    x for x in os.environ.get("SGLANG_DIFFUSION_ROCM_CE_LAST", "").split(",") if x
)


class _Unsupported(RuntimeError):
    pass


class _Hip:
    def __init__(self):
        path = None
        with open("/proc/self/maps") as maps:
            for line in maps:
                if "libamdhip64" in line:
                    path = line.split()[-1]
                    break
        lib = ctypes.CDLL(path or "libamdhip64.so")
        vp, sz, u32 = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32
        lib.hipMemcpyAsync.argtypes = [vp, vp, sz, ctypes.c_int, vp]
        lib.hipMemcpy2DAsync.argtypes = [vp, sz, vp, sz, sz, sz, ctypes.c_int, vp]
        lib.hipStreamWaitValue32.argtypes = [vp, vp, u32, ctypes.c_uint, u32]
        lib.hipGetErrorString.argtypes = [ctypes.c_int]
        lib.hipGetErrorString.restype = ctypes.c_char_p
        for fn in (lib.hipMemcpyAsync, lib.hipMemcpy2DAsync, lib.hipStreamWaitValue32):
            fn.restype = ctypes.c_int
        self.lib = lib

    def _check(self, rc, what):
        if rc:
            raise RuntimeError(
                f"{what} failed: {self.lib.hipGetErrorString(rc).decode()}"
            )

    def copy(self, stream, dst: int, src: int, nbytes: int) -> None:
        self._check(
            self.lib.hipMemcpyAsync(dst, src, nbytes, _NOCU, stream.cuda_stream),
            "hipMemcpyAsync(DeviceToDeviceNoCU)",
        )

    def copy2d(self, stream, dst, dpitch, src, spitch, width, height) -> None:
        self._check(
            self.lib.hipMemcpy2DAsync(
                dst, dpitch, src, spitch, width, height, _NOCU, stream.cuda_stream
            ),
            "hipMemcpy2DAsync(DeviceToDeviceNoCU)",
        )

    def wait(self, stream, addr: int, value: int) -> None:
        self._check(
            self.lib.hipStreamWaitValue32(
                stream.cuda_stream, addr, value, _WAIT_GEQ, 0xFFFFFFFF
            ),
            "hipStreamWaitValue32",
        )


def _share(t: torch.Tensor, group, rank: int, world: int) -> list[torch.Tensor]:
    """Every member's ``t``, the peers' opened over IPC in this device."""
    from torch.multiprocessing.reductions import reduce_tensor

    mine = [None if m == rank else reduce_tensor(t) for m in range(world)]
    handles: list = [None] * world
    dist.all_gather_object(handles, mine, group=group)
    dev = torch.cuda.current_device()
    out = []
    for member, theirs in enumerate(handles):
        if member == rank:
            out.append(t)
            continue
        fn, args = theirs[rank]
        args = list(args)
        for i, v in enumerate(args):
            if isinstance(v, torch.device):
                args[i] = torch.device(f"cuda:{dev}")
            elif isinstance(v, int) and i == 6:
                args[i] = dev
        out.append(fn(*args))
    return out


class _Pool:
    """Receive slots (packed q|k|v per head group) and output blocks, shared."""

    def __init__(self, state, dtype, in_numel: int, out_numel: int, max_groups: int):
        g, r, w = state.group, state.rank, state.world
        self.dtype = dtype
        self.sizes = (in_numel, out_numel, max_groups)
        self.inb = _share(torch.zeros(in_numel, dtype=dtype, device="cuda"), g, r, w)
        self.outb = _share(torch.zeros(out_numel, dtype=dtype, device="cuda"), g, r, w)
        # fin[p]: groups peer p has delivered; fout[g * world + p]: calls whose
        # group g peer p has delivered (groups finish on separate streams)
        zeros = lambda n: torch.zeros(n, dtype=torch.int32, device="cuda")
        self.fin = _share(zeros(w), g, r, w)
        self.fout = _share(zeros(max_groups * w), g, r, w)
        self.inb_ptr = [t.data_ptr() for t in self.inb]
        self.outb_ptr = [t.data_ptr() for t in self.outb]
        # merged outputs: outb and outb2 alternate per call, so a returned
        # output is rewritten only two calls later
        self.outb2 = None
        self.out_numel = out_numel
        self.fin_ptr = [t.data_ptr() for t in self.fin]
        self.fout_ptr = [t.data_ptr() for t in self.fout]
        self.groups_sent = 0
        self.calls = 0
        # local send blocks of the group-by-group pack
        self.send = None


class IpcA2ARocmState:
    def __init__(self):
        self.group = None
        self.inited = False
        self.failed = False
        self.pools = {}
        self.copy_streams = []
        self.group_streams = []
        self.tail_stream = None

    def reset(self):
        self.__init__()

    def init(self, group) -> None:
        self.group = group
        self.rank = dist.get_rank(group=group)
        self.world = dist.get_world_size(group=group)
        self.hip = _Hip()
        dev = torch.cuda.current_device()
        devs = [None] * self.world
        dist.all_gather_object(devs, dev, group=group)
        missing = [
            d
            for d in devs
            if d != dev and not torch.cuda.can_device_access_peer(dev, d)
        ]
        if missing:
            raise _Unsupported(f"device {dev} cannot reach {missing}")
        self.seqtab = torch.arange(_SEQ_TABLE, dtype=torch.int32, device="cuda")
        self.peers = [(self.rank + s) % self.world for s in range(1, self.world)]
        self.copy_streams = [torch.cuda.Stream() for _ in self.peers]
        self.inited = True

    def drop(self) -> None:
        if self.pools:
            torch.cuda.synchronize()
            self.pools.clear()
            dist.barrier(group=self.group)
            torch.cuda.ipc_collect()

    def _pool(self, dtype, in_numel, out_numel, groups) -> _Pool:
        pool = self.pools.get(dtype)
        if pool is not None and (
            pool.sizes[0] >= in_numel
            and pool.sizes[1] >= out_numel
            and pool.sizes[2] >= groups
        ):
            return pool
        if pool is not None:
            # queued copies may still target the old mappings
            torch.cuda.synchronize()
            del self.pools[dtype]
            del pool
            dist.barrier(group=self.group)
            torch.cuda.ipc_collect()
        pool = _Pool(
            self,
            dtype,
            int(in_numel * _POOL_HEADROOM),
            int(out_numel * _POOL_HEADROOM),
            max(groups, 8),
        )
        self.pools[dtype] = pool
        return pool

    def _maybe_reset_counters(self, pool: _Pool, groups: int) -> None:
        # identical on every rank: the counters advance with the call sequence
        if (
            pool.groups_sent + groups < _SEQ_TABLE - 1
            and pool.calls + 1 < _SEQ_TABLE - 1
        ):
            return
        torch.cuda.synchronize()
        dist.barrier(group=self.group)
        pool.fin[self.rank].zero_()
        pool.fout[self.rank].zero_()
        torch.cuda.synchronize()
        dist.barrier(group=self.group)
        pool.groups_sent = 0
        pool.calls = 0

    def pipelined_attention(self, q, k, v, attend, groups: int, pack=None, fill=None):
        """``ulysses_head_group_pipelined_attention`` on the copy engines.

        q/k/v: ``[rows, heads, head_size]`` row shards; ``pack(q, k, v, n)``
        returns the destination-major ``[n, rows, heads / n, 3 * head_size]``
        pack the collective path exchanges. Same result bytes as that path.
        ``fill(send, own, first_group, n_groups)``, when given, writes those
        groups' blocks itself (from q/k/v it was built over): the peers' into
        ``send`` ``[world, groups, rows, gh, 3 * head_size]`` and this rank's
        into its receive slots ``own`` ``[groups, world, rows, gh, 3 * head_size]``.
        """
        world, r = self.world, self.rank
        rows, heads, hd = q.shape
        gh = heads // (world * groups)
        esz = q.element_size()
        g_in = world * rows * gh * 3 * hd  # elements per group's receive slot
        g_out = world * rows * gh * hd
        pool = self._pool(q.dtype, groups * g_in, groups * g_out, groups)
        self._maybe_reset_counters(pool, groups)
        while len(self.group_streams) < groups:
            self.group_streams.append(torch.cuda.Stream())
        hip, seq = self.hip, self.seqtab
        main = torch.cuda.current_stream()

        recv = pool.inb[r][: groups * g_in].view(groups, world, rows, gh, 3 * hd)
        outb = pool.outb[r][: groups * g_out].view(groups, world, rows, gh, hd)
        base = pool.groups_sent
        pool.groups_sent += groups
        pool.calls += 1
        call = pool.calls
        chunk_in = rows * gh * 3 * hd * esz
        chunk_out = rows * gh * hd * esz
        if fill is not None:
            if pool.send is None or pool.send.numel() < groups * g_in:
                pool.send = torch.empty(groups * g_in, dtype=q.dtype, device=q.device)
            packed = pool.send[: groups * g_in].view(world, groups, rows, gh, 3 * hd)
            if _FILL_SPLIT:
                ready = []
                for g in range(groups):
                    fill(packed, recv, g, 1)
                    ready.append(main.record_event())
            elif _FILL_FIRST and groups > 1:
                # group 0 alone, so its copies start while the rest fills
                fill(packed, recv, 0, 1)
                first = main.record_event()
                fill(packed, recv, 1, groups - 1)
                ready = [first] + [main.record_event()] * (groups - 1)
            else:
                fill(packed, recv, 0, groups)
                ready = [main.record_event()] * groups
            own = ready
        elif _GROUP_PACK:
            # group by group, so group 0 is on the copy engines while the
            # later groups pack; this rank's blocks go straight into its slots.
            # Each block is its heads' q|k|v per head, as the pack kernel lays it
            if pool.send is None or pool.send.numel() < groups * g_in:
                pool.send = torch.empty(groups * g_in, dtype=q.dtype, device=q.device)
            packed = pool.send[: groups * g_in].view(world, groups, rows, gh, 3 * hd)
            ready, own = [], []
            for g in range(groups):
                for w in [*self.peers, r]:
                    block = slice((w * groups + g) * gh, (w * groups + g + 1) * gh)
                    dst = recv[g, r] if w == r else packed[w, g]
                    torch.cat((q[:, block], k[:, block], v[:, block]), dim=-1, out=dst)
                    if w == self.peers[-1]:
                        ready.append(main.record_event())
                own.append(main.record_event())
        else:
            packed = pack(q, k, v, world * groups).view(world, groups, rows, gh, 3 * hd)
            ready = [main.record_event()] * groups
            # this rank's own chunks: a same-device copy on the SMs, one pass
            # for every group before the first attends
            recv[:, r].copy_(packed[r])
            own = [main.record_event()] * groups
        pk_ptr = packed.data_ptr()
        for i, p in enumerate(self.peers):
            st = self.copy_streams[i]
            for g in range(groups):
                if g == 0 or ready[g] is not ready[g - 1]:
                    st.wait_event(ready[g])
                hip.copy(
                    st,
                    pool.inb_ptr[p] + ((g * world + r) * rows * gh * 3 * hd) * esz,
                    pk_ptr + ((p * groups + g) * rows * gh * 3 * hd) * esz,
                    chunk_in,
                )
                hip.copy(st, pool.fin_ptr[p] + 4 * r, seq[base + g + 1].data_ptr(), 4)
        # the merged output rides with the fill only, so the plain path keeps
        # the collective layout
        out_merged = _OUT_MERGED and fill is not None
        if out_merged:
            if pool.outb2 is None:
                # a group collective; every rank's call sequence is the same
                pool.outb2 = _share(
                    torch.zeros(pool.out_numel, dtype=q.dtype, device=q.device),
                    self.group,
                    r,
                    world,
                )
                pool.outb2_ptr = [t.data_ptr() for t in pool.outb2]
            mbuf, mptr = (
                (pool.outb, pool.outb_ptr) if call % 2 else (pool.outb2, pool.outb2_ptr)
            )
            merged = mbuf[r][: rows * heads * hd].view(rows, heads, hd)
            row_pitch = heads * hd * esz
            width = gh * hd * esz
        max_conc = _MAX_CONC if fill is not None else 0
        done, attended = [], []
        for g in range(groups):
            st = self.group_streams[g]
            st.wait_event(own[g])
            if max_conc and g >= max_conc:
                st.wait_event(attended[g - max_conc])
            last = g == groups - 1
            with torch.cuda.stream(st):
                for p in self.peers:
                    hip.wait(st, pool.fin_ptr[r] + 4 * p, base + g + 1)
                qkv = recv[g].view(world * rows, gh, 3 * hd)
                if last and "tlast" in _LAST:
                    from sglang.multimodal_gen.runtime.layers.attention.backends import (
                        aiter as _aiter,
                    )

                    if self.tail_stream is None:
                        self.tail_stream = torch.cuda.Stream()
                    _aiter._TAIL_STREAM = self.tail_stream
                    try:
                        out = attend(*qkv.split(hd, dim=-1))
                    finally:
                        _aiter._TAIL_STREAM = None
                else:
                    out = attend(*qkv.split(hd, dim=-1))
                attended.append(st.record_event())
                if not out.is_contiguous():
                    out = out.contiguous()
                optr = out.data_ptr()
                if out_merged:
                    # my heads of group g: block (r * groups + g) of every row
                    col = (r * groups + g) * gh
                    for p in self.peers:
                        hip.copy2d(
                            st,
                            mptr[p] + col * hd * esz,
                            row_pitch,
                            optr + p * chunk_out,
                            width,
                            width,
                            rows,
                        )
                        hip.copy(
                            st,
                            pool.fout_ptr[p] + 4 * (g * world + r),
                            seq[call].data_ptr(),
                            4,
                        )
                    merged[:, col : col + gh].copy_(out[r * rows : (r + 1) * rows])
                    done.append(st.record_event())
                    continue
                if last and "klast" in _LAST:
                    n = rows * gh * hd
                    at = (g * world + r) * n
                    torch._foreach_copy_(
                        [pool.outb[p][at : at + n] for p in self.peers],
                        [out[p * rows : (p + 1) * rows].view(-1) for p in self.peers],
                    )
                for p in self.peers:
                    if not (last and "klast" in _LAST):
                        hip.copy(
                            st,
                            pool.outb_ptr[p] + ((g * world + r) * rows * gh * hd) * esz,
                            optr + p * chunk_out,
                            chunk_out,
                        )
                    hip.copy(
                        st,
                        pool.fout_ptr[p] + 4 * (g * world + r),
                        seq[call].data_ptr(),
                        4,
                    )
                outb[g, r].copy_(out[r * rows : (r + 1) * rows])
                done.append(st.record_event())
        # the output waits go on the caller's stream, after every group was
        # queued: HIP multiplexes streams onto GPU_MAX_HW_QUEUES hardware
        # queues, and a wait queued on a group stream could hold back the next
        # group's attention sharing its queue
        group_merge = _GROUP_MERGE and fill is not None and not out_merged
        if group_merge:
            merged = torch.empty(rows, heads, hd, dtype=q.dtype, device=q.device)
            merged_groups = merged.view(rows, world, groups, gh, hd)
        for g in range(groups):
            main.wait_event(done[g])
            for p in self.peers:
                hip.wait(main, pool.fout_ptr[r] + 4 * (g * world + p), call)
            if group_merge:
                # [rank, row, head, dim] -> [row, rank, head, dim] of group g
                merged_groups[:, :, g].copy_(outb[g].permute(1, 0, 2, 3))
        # the next call's pack rewrites `packed`, read by the copy streams
        for st in self.copy_streams:
            main.wait_stream(st)
        if out_merged or group_merge:
            return merged
        # [group, rank, row, head, dim] -> [row, rank, group, head, dim]
        return outb.permute(2, 1, 0, 3, 4).reshape(rows, heads, hd)


IPC_A2A_ROCM = IpcA2ARocmState()


def rocm_ce_ready(group) -> bool:
    """True when the copy-engine pipeline is enabled and initialized on every
    member of ``group`` (initializes lazily on the first eager call)."""
    if os.environ.get("SGLANG_DIFFUSION_ROCM_CE_A2A", "0") not in ("1", "true", "True"):
        return False
    if IPC_A2A_ROCM.group is not None and IPC_A2A_ROCM.group is not group:
        IPC_A2A_ROCM.reset()
    if IPC_A2A_ROCM.failed:
        return False
    if IPC_A2A_ROCM.inited:
        return True
    if torch.cuda.is_current_stream_capturing():
        return False
    ok = True
    try:
        IPC_A2A_ROCM.init(group)
    except Exception as e:
        logger.warning("ROCm copy-engine all-to-all unavailable (%s); using RCCL", e)
        ok = False
    agreed = torch.tensor([int(ok)], dtype=torch.int32, device="cuda")
    dist.all_reduce(agreed, op=dist.ReduceOp.MIN, group=group)
    if agreed.item() == 0:
        IPC_A2A_ROCM.reset()
        IPC_A2A_ROCM.failed = True
        return False
    return True
