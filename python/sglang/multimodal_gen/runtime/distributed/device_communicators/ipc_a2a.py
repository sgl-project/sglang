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

import torch
import torch.distributed as dist

from sglang.multimodal_gen import envs

logger = logging.getLogger(__name__)

_SYNC_DECL = (
    "void spin_wait(torch::Tensor flag, torch::Tensor target);\n"
    "void bump_signal(torch::Tensor seq, torch::Tensor peer_flag);"
)
_SYNC_SRC = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
__global__ void spin_wait_kernel(volatile int* flag, const int* target) {
    int t = *target;
    while (*flag < t) {}
    __threadfence_system();
}
__global__ void bump_signal_kernel(int* seq, volatile int* peer_flag) {
    int v = *seq + 1;
    *seq = v;
    __threadfence_system();
    *peer_flag = v;
}
void spin_wait(torch::Tensor flag, torch::Tensor target) {
    spin_wait_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
        (volatile int*)flag.data_ptr<int>(), target.data_ptr<int>());
}
void bump_signal(torch::Tensor seq, torch::Tensor peer_flag) {
    bump_signal_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
        seq.data_ptr<int>(), (volatile int*)peer_flag.data_ptr<int>());
}
"""


class IpcA2AState:
    def __init__(self):
        self.ops = None
        self.staging = {}
        self.flag = None
        self.peer_flag = None
        self.my_seq = None
        self.calls = 0
        self.rank = None
        self.failed = False
        self.inited = False

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
        dev = torch.cuda.current_device()
        if not torch.cuda.can_device_access_peer(dev, 1 - dev):
            raise RuntimeError("no P2P access between the two devices")
        # kernel-level dereference of peer mappings needs explicit peer access
        ctypes.CDLL("libcudart.so").cudaDeviceEnablePeerAccess(1 - dev, 0)
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
        self.peer_flag = self._share(self.flag, group)
        self.inited = True

    def get_staging(self, n_local, n_peer, dtype, group):
        """Local buffer of `n_local` elements (the peer writes into it) paired
        with the peer's mapped buffer of `n_peer` elements (we write into it).
        Creation is a paired collective, so both ranks must reach a new key at
        the same call site."""
        key = (n_local, n_peer, dtype)
        pair = self.staging.get(key)
        if pair is None:
            local = torch.zeros(2, n_local, dtype=dtype, device="cuda")
            peer = self._share(local, group)
            pair = (local, peer)
            self.staging[key] = pair
        return pair

    def exchange(self, group, send, recv_shape):
        """Symmetric exchange: write my contiguous `send` into the peer's
        staging slot, return my staging slot viewed as `recv_shape`."""
        n_send = send.numel()
        n_recv = 1
        for v in recv_shape:
            n_recv *= v
        local, peer = self.get_staging(n_recv, n_send, send.dtype, group)
        slot = self.next_slot()
        peer[slot].narrow(0, 0, n_send).copy_(send.view(-1), non_blocking=True)
        self.signal_and_wait()
        return local[slot].narrow(0, 0, n_recv).view(recv_shape)

    def next_slot(self):
        slot = self.calls % 2
        self.calls += 1
        return slot

    def signal_and_wait(self):
        self.ops.bump_signal(self.my_seq, self.peer_flag)
        self.ops.spin_wait(self.flag, self.my_seq)

    def signal(self):
        self.ops.bump_signal(self.my_seq, self.peer_flag)

    def wait(self):
        self.ops.spin_wait(self.flag, self.my_seq)


IPC_A2A = IpcA2AState()


def ipc_a2a_ready(group) -> bool:
    """True when the IPC transport is enabled and initialized (initializes
    lazily on the first eager call; never inside a graph capture)."""
    if not envs.SGLANG_DIFFUSION_IPC_A2A or IPC_A2A.failed:
        return False
    if IPC_A2A.inited:
        return True
    if torch.cuda.is_current_stream_capturing():
        return False
    try:
        IPC_A2A.init(group)
        return True
    except Exception:
        logger.exception("IPC all-to-all init failed; falling back to NCCL")
        IPC_A2A.failed = True
        return False
