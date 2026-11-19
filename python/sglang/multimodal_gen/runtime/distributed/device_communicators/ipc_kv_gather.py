# SPDX-License-Identifier: Apache-2.0
"""Device-IPC K/V gather for a two-rank K/V-gather SP group (opt-in).

The K/V-gather SP path all-gathers K and V once per attention layer. Those are
the only two collectives in the step, and on a gfx1250 socket they run at about
60 GB/s: a 25 MB all-gather takes ~450 us where a device copy of the same bytes
takes ~10 us. Each rank already has peer access to the other's memory, so the
bytes do not need a collective at all -- only the ordering does.

So each rank copies K and V into a staging buffer the peer has mapped over
device IPC, a one-element collective orders the two streams, and each rank then
copies the peer's half straight out of that mapping.

Ordering: the token collective's kernel starts after everything already queued
on this rank's stream (so after my staging write) and completes only once the
peer's kernel has run (so after the peer's staging write); my read of the peer
mapping is queued behind it. Staging is double-buffered, so a slot is only
rewritten two calls later, after an intervening collective that orders the
rewrite behind the peer's read of that slot.

Enable with SGLANG_DIFFUSION_IPC_KV_GATHER=1 on ROCm. Anything unsupported --
a non-ROCm platform, an SP group that is not two ranks, a pair without
peer-to-peer access or not on one host, batch > 1, unequal shards, a failed IPC
handle exchange, or a staging miss during CUDA graph capture -- falls back to
the NCCL gather and returns the same result.

Scope: init() admits a group only if (1) it has exactly two ranks, (2) both ranks
are on one host, and (3) torch reports peer access between their devices. Nothing
here distinguishes "same SoC/socket" from "same host with peer access", and there
is no hierarchical path (IPC inside a SoC, a collective across SoCs). A group of
more than two ranks, or one spanning hosts, keeps the NCCL gather. Failure is
sticky: once init() fails (including an IPC handle exchange failure), this
process stays on NCCL; a staging miss only falls back for that call.
"""

import logging
from collections import OrderedDict

import torch
import torch.distributed as dist

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.layers.utils import register_custom_op

logger = logging.getLogger(__name__)

# Read once at import: the wrapper below is traced by torch.compile, and a
# module-level constant folds away instead of becoming a graph break. The ROCm
# and peer-access gates live in init() instead, so importing this module from
# an attention layer does not force platform resolution at import time.
ENABLED = envs.SGLANG_DIFFUSION_IPC_KV_GATHER
_CHECK = envs.SGLANG_DIFFUSION_IPC_KV_GATHER_CHECK


class _State:
    def __init__(self):
        self.failed = False
        self.inited = False
        self.rank = None
        self.group = None
        self.token = None
        self.token_out = None
        # insertion-ordered so eviction picks the same key on every rank (all
        # ranks create and touch the same keys in the same order)
        self.staging = OrderedDict()
        self.calls = 0
        self.checked = 0
        self.bad = None

    def _share(self, t):
        """Exchange ``t`` with the peer over torch IPC, re-opening the handle in
        the LOCAL device context (a mapping is only dereferenceable from the
        context that opened it)."""
        from torch.multiprocessing.reductions import reduce_tensor

        fn, args = reduce_tensor(t)
        mine, theirs = [(fn, args, tuple(t.shape))], [None]
        r0 = dist.get_global_rank(self.group, 0)
        r1 = dist.get_global_rank(self.group, 1)
        if self.rank == 0:
            dist.broadcast_object_list(mine, src=r0, group=self.group)
            dist.broadcast_object_list(theirs, src=r1, group=self.group)
        else:
            dist.broadcast_object_list(theirs, src=r0, group=self.group)
            dist.broadcast_object_list(mine, src=r1, group=self.group)
        pf, pa, peer_shape = theirs[0]
        if tuple(peer_shape) != tuple(t.shape):
            raise RuntimeError(
                f"unequal SP shards: local {tuple(t.shape)} vs peer {peer_shape}"
            )
        pa = list(pa)
        dev = t.device
        for i, v in enumerate(pa):
            if isinstance(v, torch.device):
                pa[i] = dev
            elif isinstance(v, int) and i == 6:
                # rebuild_cuda_tensor positional device index
                pa[i] = dev.index
        return pf(*pa)

    def init(self, device):
        from sglang.multimodal_gen.runtime.distributed.parallel_state import (
            get_sp_group,
        )
        from sglang.multimodal_gen.runtime.platforms import current_platform

        from .ipc_a2a import _peer_cuda_device

        # Nothing in the transport is vendor-specific -- the handles and the
        # barrier are the same on CUDA -- but gfx1250 is the only place it has
        # been measured, and the collective it replaces is already correct
        # everywhere, so another platform keeps that rather than running an
        # unvalidated transport.
        if not current_platform.is_rocm():
            raise RuntimeError("validated on ROCm only")
        sp_group = get_sp_group()
        self.group = sp_group.device_group
        self.rank = sp_group.rank_in_group
        if dist.get_world_size(group=self.group) != 2:
            raise RuntimeError("requires a two-rank SP group")
        # A peer mapping is only dereferenceable with peer access enabled, so a
        # pair that does not have it has to keep the collective. Unlike
        # ipc_a2a this does not also require NVLink: the ordering here is a
        # collective rather than a GPU-side flag spin, so a slower link makes
        # the exchange slower instead of stalling it. _peer_cuda_device also
        # rejects a pair that is not on one host; it compares hostnames, not
        # sockets, and can_device_access_peer can be true across sockets (e.g.
        # over xGMI), so a cross-socket pair on one host is admitted.
        peer_dev = _peer_cuda_device(self.group, self.rank, device.index)
        if not torch.cuda.can_device_access_peer(device.index, peer_dev):
            raise RuntimeError(
                "no peer-to-peer access between CUDA devices "
                f"{device.index} and {peer_dev}"
            )
        self.token = torch.zeros(1, dtype=torch.int32, device=device)
        self.token_out = torch.zeros(2, dtype=torch.int32, device=device)
        self.inited = True

    def _nccl(self, k, v):
        """The transport this replaces; also the fallback, so both produce the
        same [2, 1, 2*S, ...] stack of gathered K and V."""
        s = k.shape[1]
        out = torch.empty((2, 1, 2 * s, *k.shape[2:]), dtype=k.dtype, device=k.device)
        for i, t in enumerate((k, v)):
            tmp = torch.empty((2, *t.shape[1:]), dtype=t.dtype, device=t.device)
            dist.all_gather_into_tensor(tmp, t.contiguous()[0], group=self.group)
            out[i, 0].copy_(tmp.reshape(2 * s, *t.shape[2:]))
        return out

    def _get_staging(self, k):
        """Local buffer the peer writes into, paired with the peer's mapping.

        Creation is a paired collective, so both ranks must reach a new key at
        the same call site. Returns None on a miss during CUDA graph capture:
        the handle exchange allocates and broadcasts, which is illegal inside
        capture, so the caller falls back to NCCL and the graph bakes that in.
        A pre-warmed key keeps the IPC path, whose copies and token collective
        are capture-safe.
        """
        key = (tuple(k.shape), k.dtype)
        pair = self.staging.get(key)
        if pair is None:
            if k.is_cuda and torch.cuda.is_current_stream_capturing():
                return None
            local = torch.zeros((2, 2, *k.shape), dtype=k.dtype, device=k.device)
            pair = (local, self._share(local))
            max_buffers = envs.SGLANG_DIFFUSION_IPC_KV_GATHER_MAX_BUFFERS
            if len(self.staging) >= max_buffers:
                # Evicting drops my buffer while the peer still maps it, so both
                # ranks must drop the same key: they share the insertion order.
                self.staging.popitem(last=False)
            self.staging[key] = pair
            logger.info(
                "IPC K/V gather: rank %d staging %s %s", self.rank, key[0], key[1]
            )
        return pair

    def gather(self, k, v):
        """Runs eagerly inside the custom op, on both ranks at the same call."""
        if not self.inited and not self.failed:
            try:
                self.init(k.device)
            except Exception as e:  # noqa: BLE001
                logger.warning("IPC K/V gather unavailable (%s); using NCCL", e)
                self.failed = True
        if self.failed:
            return self._nccl(k, v)
        pair = self._get_staging(k)
        if pair is None:
            return self._nccl(k, v)
        local, peer = pair
        slot = self.calls % 2
        self.calls += 1
        mine = local[slot]
        mine[0].copy_(k)
        mine[1].copy_(v)
        # Orders my read of the peer mapping behind the peer's staging write.
        dist.all_gather_into_tensor(self.token_out, self.token, group=self.group)
        s = k.shape[1]
        out = torch.empty(
            (2, k.shape[0], 2 * s, *k.shape[2:]), dtype=k.dtype, device=k.device
        )
        # Per-tensor slabs are contiguous at batch 1, so these are plain device
        # memcpys rather than strided elementwise copies.
        lo, po = self.rank * s, (1 - self.rank) * s
        theirs = peer[slot]
        for i in range(2):
            out[i, 0, lo : lo + s].copy_(mine[i, 0])
            out[i, 0, po : po + s].copy_(theirs[i, 0])
        if _CHECK:
            # Accumulate the verdict on device and read it back rarely, so the
            # check does not serialize the ranks and hide an ordering bug.
            ref = self._nccl(k, v)
            if self.bad is None:
                self.bad = torch.zeros((), dtype=torch.int64, device=k.device)
            self.bad += (out != ref).any().to(torch.int64)
            self.checked += 1
            if self.checked % 500 == 0:
                logger.info(
                    "IPC K/V gather check: rank %d %d calls, %d mismatches",
                    self.rank,
                    self.checked,
                    int(self.bad.item()),
                )
        return out

    def reset(self):
        """Drop mappings that belong to an SP group being replaced."""
        self.__init__()


IPC_KV_GATHER = _State()


@register_custom_op(
    op_name="ipc_kv_gather",
    fake_impl=lambda k, v: k.new_empty((2, k.shape[0], 2 * k.shape[1], *k.shape[2:])),
)
def _ipc_kv_gather(k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return IPC_KV_GATHER.gather(k, v)


def ipc_kv_gather(k: torch.Tensor, v: torch.Tensor, world_size: int):
    """Return (K, V) gathered along dim 1 in rank order, or None to fall back.

    Only static checks live here: this wrapper is traced by torch.compile, so
    all stateful work stays inside the opaque custom op.
    """
    if (
        not ENABLED
        or world_size != 2
        or k.dim() != 4
        or k.shape[0] != 1
        or k.shape != v.shape
        or k.dtype != v.dtype
    ):
        return None
    # Flux2 K and V are strided views of one fused qkv projection. Materialize
    # them in the compiled graph (one fused clone each) so the op's staging
    # writes are contiguous memcpys.
    out = _ipc_kv_gather(k.contiguous(), v.contiguous())
    return out[0], out[1]
