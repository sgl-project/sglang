# SPDX-License-Identifier: Apache-2.0
"""Peer-readable staging for the partial sums of a two-rank TP group.

A row-parallel layer's all-reduce moves each rank's partial sum to the other
rank and adds the two. Here each rank's GEMM writes its partial into a buffer
the peer has mapped and the rank publishes a sequence number; the epilogue
kernel that would have read the reduced tensor reads both partials instead, so
the transfer is that kernel's loads and the separate reduce pass is gone.

Two slots alternate per call. A rank rewrites a slot two calls later, queued
behind its wait for the peer's next publish, and the peer issues that publish
in stream order after the kernel that read the slot.
"""

import logging

import torch
import torch.distributed as dist

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.distributed.device_communicators.ipc_a2a import (
    _Unsupported,
    ipc_shareable_zeros,
)
from sglang.multimodal_gen.runtime.distributed.device_communicators.ipc_a2a_multi import (
    _member_devices,
    _StreamMemOps,
)
from sglang.multimodal_gen.runtime.platforms import current_platform

logger = logging.getLogger(__name__)


def _open_peer(t: torch.Tensor, group, rank: int) -> torch.Tensor:
    """The peer's `t`, opened in this rank's device context."""
    from torch.multiprocessing.reductions import reduce_tensor

    handles: list = [None, None]
    dist.all_gather_object(handles, reduce_tensor(t), group=group)
    fn, args = handles[1 - rank]
    dev = torch.cuda.current_device()
    args = list(args)
    for i, v in enumerate(args):
        if isinstance(v, torch.device):
            args[i] = torch.device(f"cuda:{dev}")
        elif isinstance(v, int) and i == 6:
            # rebuild_cuda_tensor positional device index
            args[i] = dev
    return fn(*args)


class TpPeerPartials:
    def __init__(self):
        self.group = None
        self.rank = None
        self.memops = None
        # written by the peer: calls whose partial it has finished
        self.flag = None
        self.peer_flag = None
        # dtype -> (mine, theirs), each [2, numel]
        self.buffers = {}
        self.calls = 0
        self.failed = False
        self.inited = False

    def reset(self) -> None:
        self.__init__()

    def init(self, tp_group) -> None:
        import ctypes

        group = tp_group.device_group
        if dist.get_world_size(group=group) != 2:
            raise _Unsupported("requires a two-rank TP group")
        self.rank = dist.get_rank(group=group)
        dev = torch.cuda.current_device()
        peer_dev = _member_devices(group, dev)[1 - self.rank]
        reachable = torch.cuda.can_device_access_peer(dev, peer_dev)
        memops_error = None
        try:
            memops = _StreamMemOps()
        except _Unsupported as e:
            memops_error = e
        # agree before the handle exchange: a rank that bailed out here alone
        # would leave the other blocked inside that collective
        agreed = torch.tensor(
            [int(reachable and memops_error is None)], dtype=torch.int32, device="cuda"
        )
        dist.all_reduce(agreed, op=dist.ReduceOp.MIN, group=group)
        if agreed.item() == 0:
            raise _Unsupported(
                str(memops_error)
                if memops_error
                else "the TP ranks lack peer-to-peer access"
            )
        # a kernel dereferencing the peer mapping needs explicit peer access
        ctypes.CDLL("libcudart.so").cudaDeviceEnablePeerAccess(peer_dev, 0)
        self.memops = memops
        self.flag = ipc_shareable_zeros(1, dtype=torch.int32)
        self.peer_flag = _open_peer(self.flag, group, self.rank)
        self.group = group
        self.inited = True

    def ready(self, tp_group) -> bool:
        if not envs.SGLANG_DIFFUSION_TP_PEER_PARTIALS or self.failed:
            return False
        if torch.compiler.is_compiling() or not current_platform.is_cuda():
            return False
        if torch.cuda.is_current_stream_capturing():
            return False
        if self.group is not None and tp_group.device_group is not self.group:
            self.reset()
        if self.inited:
            return True
        ok = True
        try:
            self.init(tp_group)
        except _Unsupported as e:
            logger.info("TP peer partials unavailable (%s); using all-reduce", e)
            ok = False
        except Exception as e:
            logger.warning("TP peer partials unavailable (%s); using all-reduce", e)
            ok = False
        # both ranks must take the same path, or one waits on a publish the
        # other never issues
        agreed = torch.tensor([int(ok)], dtype=torch.int32, device="cuda")
        dist.all_reduce(agreed, op=dist.ReduceOp.MIN, group=tp_group.device_group)
        if agreed.item() == 0:
            self.reset()
            self.failed = True
            return False
        return True

    def next_slot(
        self, numel: int, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """This call's flat (mine, theirs) staging views of `numel` elements.

        Growing the buffers is a group collective, so both ranks must reach a
        new size at the same call.
        """
        pair = self.buffers.get(dtype)
        if pair is None or pair[0].shape[1] < numel:
            # queued kernels may still read the old mappings
            torch.cuda.synchronize()
            self.buffers.pop(dtype, None)
            del pair
            mine = ipc_shareable_zeros(2, numel, dtype=dtype)
            pair = self.buffers[dtype] = (
                mine,
                _open_peer(mine, self.group, self.rank),
            )
        self.calls += 1
        slot = self.calls & 1
        return pair[0][slot, :numel], pair[1][slot, :numel]

    def publish_and_wait(self) -> None:
        """Tell the peer this call's partial is written; hold the stream until
        the peer's is. Neither takes an SM."""
        stream = torch.cuda.current_stream()
        self.memops.write(stream, self.peer_flag, self.calls)
        self.memops.wait(stream, self.flag, self.calls)


TP_PEER_PARTIALS = TpPeerPartials()


def staged_partials(
    layer, hidden: torch.Tensor, like: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Run the RowParallelLinear ``layer`` on ``hidden`` without its all-reduce:
    this rank's and the peer's partial sums of the ``like``-shaped output, as
    ``[rows, dim]``, for a kernel that adds them as it reads.

    None, before anything is exchanged, when the layer or the group cannot.
    Every rank decides alike: the checks read only what the ranks share.
    """
    if (
        layer.tp_size != 2
        or not layer.reduce_results
        or not layer.supports_partial_into(hidden)
        or not TP_PEER_PARTIALS.ready(layer.tp_group)
    ):
        return None
    dim = like.shape[-1]
    mine, theirs = TP_PEER_PARTIALS.next_slot(like.numel(), like.dtype)
    mine, theirs = mine.view(-1, dim), theirs.view(-1, dim)
    layer.partial_into(hidden, mine)
    TP_PEER_PARTIALS.publish_and_wait()
    return mine, theirs
