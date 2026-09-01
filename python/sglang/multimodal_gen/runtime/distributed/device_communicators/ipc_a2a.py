# SPDX-License-Identifier: Apache-2.0
"""CUDA-IPC transport for 2-rank Ulysses all-to-all.

Each rank maps the peer's staging buffers into its own device context
(handles are re-opened locally, so every access is same-device semantics over
NVLink) and writes its half of the exchange directly into them. Rank
synchronization is a GPU-side sequence counter: bump_signal publishes my new
sequence number into the peer's flag after my writes, spin_wait blocks my
stream until the peer has published the same number. Both primitives are
plain kernels on local memory, so the whole exchange is CUDA-graph
capturable. Double-buffered slots alternate per call; a slot is only rewritten
after an intervening spin_wait, which orders the rewrite after the peer's
read of that slot.
"""

import logging
import os
import socket
from collections import OrderedDict

import torch
import torch.distributed as dist

from sglang.multimodal_gen import envs

logger = logging.getLogger(__name__)

_SYNC_DECL = (
    "void spin_wait(torch::Tensor flag, torch::Tensor target, torch::Tensor timed_out,"
    " torch::Tensor peer_timed_out, int64_t budget_ns);\n"
    "void bump_signal(torch::Tensor seq, torch::Tensor peer_flag);"
)
_SYNC_SRC = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
__device__ __forceinline__ unsigned long long now_ns() {
    // %globaltimer is a nanosecond wall clock, so the budget needs no SM-clock
    // conversion -- cudaDevAttrClockRate is not dependable across architectures
    // (B200 reports 120 MHz, which would shrink the timeout ~16x).
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
__global__ void spin_wait_kernel(volatile int* flag, const int* target,
                                 int* timed_out, int* peer_timed_out,
                                 unsigned long long budget_ns) {
    int t = *target;
    unsigned long long start = now_ns();
    while (*flag < t) {
        if (now_ns() - start > budget_ns) {
            // Give up rather than hang the stream forever. The peer never
            // published, so this exchange's data is incomplete. Flag it on the
            // peer as well as here: both ranks must retire the transport at the
            // same request boundary, or the one that switched to NCCL would post
            // a collective the other never posts.
            *timed_out = 1;
            *peer_timed_out = 1;
            __threadfence_system();
            return;
        }
    }
    __threadfence_system();
}
__global__ void bump_signal_kernel(int* seq, volatile int* peer_flag) {
    int v = *seq + 1;
    *seq = v;
    __threadfence_system();
    *peer_flag = v;
}
void spin_wait(torch::Tensor flag, torch::Tensor target, torch::Tensor timed_out,
               torch::Tensor peer_timed_out, int64_t budget_ns) {
    spin_wait_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
        (volatile int*)flag.data_ptr<int>(), target.data_ptr<int>(),
        timed_out.data_ptr<int>(), peer_timed_out.data_ptr<int>(),
        (unsigned long long)budget_ns);
}
void bump_signal(torch::Tensor seq, torch::Tensor peer_flag) {
    bump_signal_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
        seq.data_ptr<int>(), (volatile int*)peer_flag.data_ptr<int>());
}
"""


class _Unsupported(RuntimeError):
    """This topology cannot run the transport -- an expected outcome, not a bug."""


def _peer_cuda_device(group, rank: int, device: int) -> int:
    """Return the peer's local CUDA ordinal for a two-rank same-host group."""
    world_size = dist.get_world_size(group=group)
    if world_size != 2:
        raise _Unsupported(
            f"requires a two-rank Ulysses group, got world size {world_size}"
        )

    members: list[tuple[str, int] | None] = [None] * world_size
    dist.all_gather_object(
        members,
        (socket.gethostname(), device),
        group=group,
    )
    local = members[rank]
    peer = members[1 - rank]
    if local is None or peer is None:
        raise _Unsupported("could not discover both Ulysses group members")
    if local[0] != peer[0]:
        raise _Unsupported(
            "requires both Ulysses ranks on the same host "
            f"(got {local[0]!r} and {peer[0]!r})"
        )
    if local[1] != device:
        raise _Unsupported(
            f"group rank {rank} reported CUDA device {local[1]}, expected {device}"
        )
    if peer[1] == device:
        raise _Unsupported(f"both Ulysses ranks are assigned CUDA device {device}")
    return peer[1]


class IpcA2AState:
    def __init__(self):
        self.ops = None
        # insertion-ordered so eviction is identical on every rank (all ranks
        # create and touch the same keys in the same order)
        self.staging = OrderedDict()
        self.flag = None
        self.peer_flag = None
        self.my_seq = None
        self.timed_out = None
        self.peer_timed_out = None
        self.budget_ns = 0
        self.max_buffers = 0
        self.calls = 0
        self.rank = None
        self.group = None
        self.failed = False
        self.inited = False

    def reset(self) -> None:
        """Drop mappings that belong to a model-parallel group being replaced."""
        self.__init__()

    def _share(self, t, group):
        """Exchange `t` with the peer via torch IPC, re-opening the handle in
        the LOCAL device context (the mapping is only dereferenceable from the
        context that opened it)."""
        from torch.multiprocessing.reductions import reduce_tensor

        fn, args = reduce_tensor(t)
        mine = [(fn, args)]
        theirs = [None]
        r0 = dist.get_global_rank(group, 0)
        r1 = dist.get_global_rank(group, 1)
        if self.rank == 0:
            dist.broadcast_object_list(mine, src=r0, group=group)
            dist.broadcast_object_list(theirs, src=r1, group=group)
        else:
            dist.broadcast_object_list(theirs, src=r0, group=group)
            dist.broadcast_object_list(mine, src=r1, group=group)
        pf, pa = theirs[0]
        pa = list(pa)
        dev = torch.cuda.current_device()
        for i, v in enumerate(pa):
            if isinstance(v, torch.device):
                pa[i] = torch.device(f"cuda:{dev}")
            elif isinstance(v, int) and i == 6:
                # rebuild_cuda_tensor positional device index
                pa[i] = dev
        return pf(*pa)

    def init(self, group):
        import ctypes

        from torch.utils.cpp_extension import load_inline

        self.rank = dist.get_rank(group=group)
        self.group = group
        dev = torch.cuda.current_device()
        peer_dev = _peer_cuda_device(group, self.rank, dev)
        try:
            has_peer_access = torch.cuda.can_device_access_peer(dev, peer_dev)
        except RuntimeError as e:
            raise _Unsupported(
                f"could not query peer access between CUDA devices {dev} and {peer_dev}: {e}"
            ) from e
        if not has_peer_access:
            raise _Unsupported(
                f"no peer-to-peer access between CUDA devices {dev} and {peer_dev}"
            )
        # kernel-level dereference of peer mappings needs explicit peer access
        ctypes.CDLL("libcudart.so").cudaDeviceEnablePeerAccess(peer_dev, 0)
        build_dir = os.path.join(
            envs.SGLANG_DIFFUSION_CACHE_ROOT, f"ipc_a2a_sync_r{dev}"
        )
        os.makedirs(build_dir, exist_ok=True)
        self.ops = load_inline(
            name="ipc_a2a_sync",
            cpp_sources=_SYNC_DECL,
            cuda_sources=_SYNC_SRC,
            functions=["spin_wait", "bump_signal"],
            extra_cuda_cflags=["-O3"],
            build_directory=build_dir,
            verbose=False,
        )
        self.flag = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.my_seq = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.timed_out = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.budget_ns = int(envs.SGLANG_DIFFUSION_IPC_A2A_TIMEOUT_MS * 1e6)
        self.max_buffers = envs.SGLANG_DIFFUSION_IPC_A2A_MAX_BUFFERS
        self.peer_flag = self._share(self.flag, group)
        self.peer_timed_out = self._share(self.timed_out, group)
        self.inited = True

    def get_staging(self, n_local, n_peer, dtype, group):
        """Local buffer of `n_local` elements (the peer writes into it) paired
        with the peer's mapped buffer of `n_peer` elements (we write into it).
        Creation is a paired collective, so both ranks must reach a new key at
        the same call site. Returns None on a miss during CUDA graph capture:
        the IPC handle exchange allocates and broadcasts, which is illegal
        inside capture, so callers fall back to NCCL and the graph bakes that
        path; pre-warmed keys keep the IPC fast path (its copies and
        spin/bump kernels are capture-safe)."""
        key = (n_local, n_peer, dtype)
        pair = self.staging.get(key)
        if pair is None:
            if torch.cuda.is_current_stream_capturing():
                return None
            local = torch.zeros(2, n_local, dtype=dtype, device="cuda")
            peer = self._share(local, group)
            pair = (local, peer)
            if len(self.staging) >= self.max_buffers:
                # Evicting drops my buffer while the peer still maps it, so both
                # ranks must drop the same key: they share the insertion order.
                self.staging.popitem(last=False)
            self.staging[key] = pair
        return pair

    def exchange(self, group, send, recv_shape):
        """Symmetric exchange: write my contiguous `send` into the peer's
        staging slot, return my staging slot viewed as `recv_shape`."""
        n_send = send.numel()
        n_recv = 1
        for v in recv_shape:
            n_recv *= v
        pair = self.get_staging(n_recv, n_send, send.dtype, group)
        if pair is None:
            return None
        local, peer = pair
        slot = self.next_slot()
        peer[slot].narrow(0, 0, n_send).copy_(send.view(-1), non_blocking=True)
        self.signal_and_wait()
        return local[slot].narrow(0, 0, n_recv).view(recv_shape)

    def next_slot(self):
        slot = self.calls % 2
        self.calls += 1
        return slot

    def signal_and_wait(self):
        self.signal()
        self.wait()

    def signal(self):
        self.ops.bump_signal(self.my_seq, self.peer_flag)

    def wait(self):
        self.ops.spin_wait(
            self.flag,
            self.my_seq,
            self.timed_out,
            self.peer_timed_out,
            self.budget_ns,
        )

    def check_timeout(self) -> None:
        """Retire the transport on every rank if any rank's spin expired.

        The flag is a device read, so this belongs at a request boundary, never
        inside a capture. It reads as local, but the spin kernel sets the flag on
        both sides, so a timeout retires the transport on both: a rank that
        retired alone would post an all_to_all its peer -- still on IPC -- never
        posts, and the NCCL watchdog would take the process down.

        Expiry means the peer went silent for the whole budget, which is a hang,
        not a slow step, and the exchange that gave up returned incomplete data.
        So this raises rather than quietly serving a corrupted result.
        """
        if self.failed or not self.inited or self.timed_out.item() == 0:
            return
        self.failed = True
        raise RuntimeError(
            "IPC all-to-all gave up waiting for its peer after "
            f"{envs.SGLANG_DIFFUSION_IPC_A2A_TIMEOUT_MS:g} ms, so that exchange "
            "returned incomplete data. The transport is now disabled on every "
            "rank; retry the request over NCCL. Raise "
            "SGLANG_DIFFUSION_IPC_A2A_TIMEOUT_MS if a rank can legitimately "
            "stall this long (layerwise offload, expert-tower swaps), or set "
            "SGLANG_DIFFUSION_IPC_A2A=0."
        )


IPC_A2A = IpcA2AState()


def ipc_a2a_ready(group) -> bool:
    """True when the IPC transport is enabled and initialized (initializes
    lazily on the first eager call; never inside a graph capture)."""
    from sglang.multimodal_gen.runtime.distributed import get_tp_world_size
    from sglang.multimodal_gen.runtime.platforms import current_platform

    if not envs.SGLANG_DIFFUSION_IPC_A2A:
        return False
    if IPC_A2A.group is not None and IPC_A2A.group is not group:
        IPC_A2A.reset()
    if IPC_A2A.failed:
        return False
    # CUDA-IPC is only implemented for TP1 two-rank Ulysses groups. The
    # initializer resolves the actual local device ordinals of both members,
    # so CFG replicas can use independent GPU pairs on the same host.
    if not current_platform.is_cuda() or get_tp_world_size() > 1:
        return False
    if IPC_A2A.inited:
        return True
    if torch.cuda.is_current_stream_capturing():
        return False
    try:
        IPC_A2A.init(group)
        return True
    except _Unsupported as e:
        logger.info("IPC all-to-all unavailable (%s); using NCCL", e)
        IPC_A2A.failed = True
        return False
    except Exception as e:
        logger.debug("IPC all-to-all initialization failed", exc_info=True)
        logger.warning("IPC all-to-all unavailable (%s); using NCCL", e)
        IPC_A2A.failed = True
        return False
