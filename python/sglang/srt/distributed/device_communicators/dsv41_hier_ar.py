"""Two-step collectives for the 8×V100 hybrid NVLink mesh.

TP8 custom-AR is disabled: the eight cards are not a 1-hop clique. They *are*
two full NVLink quads {0–3} and {4–7} with four NVLink bridges 0–4, 1–5, 2–6,
3–7. Same 2×4 split that Ulysses SP8 used for AR and A2A: step inside the
quad, then across the pair. Decode-sized tensors only; large prefill stays
on the 8-rank NCCL communicator.

Default backend is PyNCCL (CUDA-graph capturable). Custom-AR uses the
pre-registered IPC staging buffer (never graph-pool pointer IPC —
``custom_all_reduce.cuh:614`` on V100). Pair CA is on whenever hier AR is
on. Quad CA is behind ``SGLANG_DSV41_HIER_AR_CA`` (in-graph 1-stage
on the NVLink clique; relaunch57 lost that kernel only in eager).
"""

from __future__ import annotations

import ctypes
import logging
from typing import List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# Decode ARs on this model are 10–200 KiB. 20 MiB prefill chunks stay 8-rank.
_MAX_HIER_BYTES = 2 * 1024 * 1024
# Below the quad custom-AR's 512 KiB 1stage limit, so the chain the push kernel
# replaces is 1stage in both steps; larger tensors keep the chain.
_PUSH_SLOT_BYTES = 256 * 1024
_WORLD = 8
_QUAD = 4
_PAIR = 2


def partition_quads_and_pairs(
    ranks: Sequence[int],
) -> Tuple[List[List[int]], List[List[int]]]:
    """PCI-order 8 ranks → two NVLink quads and the four cross-quad pairs."""
    r = list(ranks)
    if len(r) != 8:
        raise ValueError(f"hierarchical AR needs 8 ranks, got {len(r)}")
    quads = [r[:4], r[4:]]
    pairs = [[r[i], r[i + 4]] for i in range(4)]
    return quads, pairs


def simulate_two_step_a2a(send_by_rank: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    """Reference 2-step all-to-all (quad then pair) for the PCI 2×4 mesh.

    ``send_by_rank[src]`` is dest-major ``[8, n]``: row ``d`` is what ``src``
    sends to dest ``d``. Returns dest-local ``[8, n]`` at each dest: row
    ``s`` is what arrived from source ``s``.
    """
    if len(send_by_rank) != _WORLD:
        raise ValueError(f"need {_WORLD} rank buffers, got {len(send_by_rank)}")
    n = send_by_rank[0].shape[-1]
    quad_recv: List[Optional[torch.Tensor]] = [None] * _WORLD
    for q in (0, 1):
        members = [q * _QUAD + i for i in range(_QUAD)]
        sends = []
        for r in members:
            # [dest_quad, dest_local, n] -> [dest_local, dest_quad, n]
            sends.append(
                send_by_rank[r].view(_PAIR, _QUAD, n).permute(1, 0, 2).contiguous()
            )
        for l_dst, rdst in enumerate(members):
            parts = [sends[l_src][l_dst] for l_src in range(_QUAD)]
            quad_recv[rdst] = torch.stack(parts, 0)  # [src_local, dest_quad, n]
    out: List[torch.Tensor] = [send_by_rank[0].new_empty(_WORLD, n) for _ in range(_WORLD)]
    for local in range(_QUAD):
        a, b = local, local + _QUAD
        qa = quad_recv[a]
        qb = quad_recv[b]
        assert qa is not None and qb is not None
        sa = qa.permute(1, 0, 2).contiguous()  # [dest_quad, src_local, n]
        sb = qb.permute(1, 0, 2).contiguous()
        # pair rank 0 is the low-quad rank; all_to_all send[i] -> pair rank i
        out[a] = torch.stack([sa[0], sb[0]], 0).reshape(_WORLD, n)
        out[b] = torch.stack([sa[1], sb[1]], 0).reshape(_WORLD, n)
    return out


class _DeviceBuffer:
    """Exposes a raw cudaMalloc range to torch.as_tensor without copying."""

    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 2,
        }


def _staging_view(ca, device: torch.device) -> torch.Tensor:
    """uint8 view of this rank's registered custom-AR staging buffer."""
    return torch.as_tensor(
        _DeviceBuffer(ca.buffer_ptrs[ca.rank], ca.max_size), device=device
    )


class Dsv41HierAllReduce:
    def __init__(self, ranks: Sequence[int], rank: int, device: torch.device):
        quads, pairs = partition_quads_and_pairs(ranks)
        self.rank = rank
        self.device = device
        self.quad_cpu: Optional[ProcessGroup] = None
        self.quad_dev: Optional[ProcessGroup] = None
        self.pair_cpu: Optional[ProcessGroup] = None
        self.pair_dev: Optional[ProcessGroup] = None
        for q in quads:
            cpu = dist.new_group(q, backend="gloo")
            dev = dist.new_group(q, backend="nccl")
            if rank in q:
                self.quad_cpu, self.quad_dev = cpu, dev
        for p in pairs:
            cpu = dist.new_group(p, backend="gloo")
            dev = dist.new_group(p, backend="nccl")
            if rank in p:
                self.pair_cpu, self.pair_dev = cpu, dev
        if self.quad_cpu is None or self.pair_cpu is None:
            raise RuntimeError("rank not in a V100 quad/pair for hierarchical AR")

        from sglang.srt.distributed.device_communicators.pynccl import (
            PyNcclCommunicator,
        )

        self.quad_nccl = PyNcclCommunicator(group=self.quad_cpu, device=device)
        dist.barrier()
        self.pair_nccl = PyNcclCommunicator(group=self.pair_cpu, device=device)
        dist.barrier()

        self.quad_ca = None
        self.pair_ca = self._try_ca(self.pair_cpu, device)
        dist.barrier()
        if envs.SGLANG_DSV41_HIER_AR_CA.get():
            self.quad_ca = self._try_ca(self.quad_cpu, device)
            dist.barrier()

        # The quad reduce writes straight into the pair's staging buffer and the
        # pair reduce straight into the caller's tensor: one copy per AR, not four.
        self._ca_chain = None
        if self.quad_ca is not None and self.pair_ca is not None:
            self._ca_chain = (
                _staging_view(self.quad_ca, device),
                _staging_view(self.pair_ca, device),
            )

        self._push = None
        if self._ca_chain is not None and envs.SGLANG_DSV41_HIER_AR_PUSH.get():
            self._push = self._init_push(device)

        self._a2a_a = torch.empty(_MAX_HIER_BYTES, dtype=torch.uint8, device=device)
        self._a2a_b = torch.empty(_MAX_HIER_BYTES, dtype=torch.uint8, device=device)

        logger.info(
            "DSV4.1 2-step hier: quad_nccl=%s pair_nccl=%s quad_ca=%s pair_ca=%s "
            "push=%s max_bytes=%d",
            "on" if self.quad_nccl.available else "off",
            "on" if self.pair_nccl.available else "off",
            "on" if self.quad_ca is not None else "off",
            "on" if self.pair_ca is not None else "off",
            "on" if self._push is not None else "off",
            _MAX_HIER_BYTES,
        )

    def _init_push(self, device: torch.device):
        from sglang.kernels.ops.communication import sm70_hier_push_ar as push
        from sglang.srt.distributed.device_communicators.cuda_wrapper import (
            CudaRTLibrary,
        )
        from sglang.srt.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce,
        )

        lib = CudaRTLibrary()
        ptrs = []
        for group, nbytes in (
            (self.quad_cpu, push.quad_workspace_bytes(_PUSH_SLOT_BYTES)),
            (self.pair_cpu, push.pair_workspace_bytes(_PUSH_SLOT_BYTES)),
        ):
            group_ptrs = CustomAllreduce.create_shared_buffer(nbytes, group=group)
            own = group_ptrs[dist.get_rank(group=group)]
            lib.cudaMemset(ctypes.c_void_p(own), push.EMPTY_BYTE, nbytes)
            ptrs.append(group_ptrs)
        lib.cudaDeviceSynchronize()
        # No rank may push before every receive buffer holds the empty marker.
        dist.barrier()
        return push.Sm70HierPushAllReduce(
            ptrs[0],
            ptrs[1],
            _PUSH_SLOT_BYTES,
            dist.get_rank(group=self.quad_cpu),
            dist.get_rank(group=self.pair_cpu),
            device,
        )

    @staticmethod
    def _try_ca(group: ProcessGroup, device: torch.device):
        from sglang.srt.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce,
        )

        ca = CustomAllreduce(group=group, device=device, max_size=_MAX_HIER_BYTES)
        if getattr(ca, "disabled", True):
            return None
        return ca

    def should(self, tensor: torch.Tensor) -> bool:
        n = tensor.numel() * tensor.element_size()
        return (
            tensor.is_cuda
            and n <= _MAX_HIER_BYTES
            and n % 16 == 0
            and n > 0
        )

    def graph_capture_contexts(self):
        """Register staging-buffer IPC after CUDA-graph capture.

        Always copy through the pre-registered cudaMalloc buffer
        (see ``_stage_ca``). ``capture()`` then IPCs those buffers, not
        CUDA-graph pool pointers (relaunch99).
        """
        from contextlib import ExitStack, contextmanager

        @contextmanager
        def _stack():
            with ExitStack() as stack:
                if self.quad_ca is not None:
                    stack.enter_context(self.quad_ca.capture())
                if self.pair_ca is not None:
                    stack.enter_context(self.pair_ca.capture())
                yield

        return _stack()

    def _stage_ca(self, tensor: torch.Tensor, ca, dev_group: ProcessGroup) -> None:
        # Force the unregistered path: copy into the cudaMalloc IPC staging
        # buffer, then reduce. ``custom_all_reduce()`` uses registered=True
        # under capture, which records graph-pool pointers and dies at
        # cuPointerGetAttribute RANGE_START_ADDR on this box.
        if ca is not None and not getattr(ca, "disabled", True):
            if ca.should_custom_ar(tensor):
                out = ca._all_reduce_impl(tensor, registered=False)
                if out is not None:
                    tensor.copy_(out)
                    return
        dist.all_reduce(tensor, group=dev_group)

    def _stage_nccl(self, tensor: torch.Tensor, comm) -> None:
        with comm.change_state(enable=True):
            comm.all_reduce(tensor)

    def _pair_reduce_p2p(self, tensor: torch.Tensor) -> None:
        """2-rank NVLink hop: grouped send/recv + add. Not a second Tree AR.

        relaunch97 used NCCL Tree on the pair too (166 TREE_LL/tok, still ~86
        ms). Decode ARs are ~10 KiB; the extra Tree was launch tax, not hops.
        """
        nbytes = tensor.numel() * tensor.element_size()
        tmp = self._a2a_a[:nbytes].view(tensor.dtype).reshape(tensor.shape)
        peer = 1 - self.pair_nccl.rank
        with self.pair_nccl.change_state(enable=True):
            self.pair_nccl.group_start()
            self.pair_nccl.send(tensor, dst=peer)
            self.pair_nccl.recv(tmp, src=peer)
            self.pair_nccl.group_end()
        tensor.add_(tmp)

    def _reduce_ca_chain(self, tensor: torch.Tensor) -> bool:
        from sglang.srt.distributed.device_communicators import (
            custom_all_reduce_ops as ops,
        )

        quad, pair = self.quad_ca, self.pair_ca
        if quad.disabled or pair.disabled:
            return False
        if not (quad.should_custom_ar(tensor) and pair.should_custom_ar(tensor)):
            return False
        nbytes = tensor.numel() * tensor.element_size()
        quad_buf, pair_buf = (
            buf[:nbytes].view(tensor.dtype).view(tensor.shape)
            for buf in self._ca_chain
        )
        quad_buf.copy_(tensor)
        # A zero reg_buffer makes the op reduce from `inp` itself; the staging
        # buffers are registered at init (and recorded for IPC under capture).
        ops.all_reduce(quad._ptr, quad_buf, pair_buf, 0, 0)
        ops.all_reduce(pair._ptr, pair_buf, tensor, 0, 0)
        return True

    def reduce_inplace(self, tensor: torch.Tensor) -> bool:
        if not self.should(tensor):
            return False
        if self._push is not None and self._push.covers(tensor):
            self._push.all_reduce(tensor, tensor)
            return True
        if (
            self._ca_chain is not None
            and tensor.is_contiguous()
            and self._reduce_ca_chain(tensor)
        ):
            return True
        if self.quad_ca is not None:
            self._stage_ca(tensor, self.quad_ca, self.quad_dev)
        elif self.quad_nccl.available:
            self._stage_nccl(tensor, self.quad_nccl)
        else:
            return False
        # 2-rank bridge: custom-AR 1-stage (relaunch57 pair_ca was 0.8%).
        # NCCL SendRecv on this hop was 45 ms/tok (relaunch98).
        if self.pair_ca is not None:
            self._stage_ca(tensor, self.pair_ca, self.pair_dev)
        elif self.pair_nccl.available:
            self._pair_reduce_p2p(tensor)
        else:
            return False
        return True

    def all_to_all_single(self, output: torch.Tensor, input_: torch.Tensor) -> bool:
        """2-step all-to-all (quad 4-rank, then pair 2-rank). CUDA-graph safe."""
        if (
            not self.quad_nccl.available
            or not self.pair_nccl.available
            or output.shape != input_.shape
            or output.dtype != input_.dtype
            or not self.should(input_)
            or input_.numel() % _WORLD != 0
        ):
            return False
        nchunk = input_.numel() // _WORLD
        nbytes = input_.numel() * input_.element_size()
        send = self._a2a_a[:nbytes].view(input_.dtype).view(_WORLD, nchunk)
        recv = self._a2a_b[:nbytes].view(input_.dtype).view(_WORLD, nchunk)
        send.copy_(input_.reshape(_WORLD, nchunk))

        # Phase 1: [dest_quad, dest_local] -> [dest_local, dest_quad]
        qsend = recv.view(_QUAD, _PAIR, nchunk)
        qsend.copy_(send.view(_PAIR, _QUAD, nchunk).permute(1, 0, 2))
        qrecv = send.view(_QUAD, _PAIR, nchunk)
        with self.quad_nccl.change_state(enable=True):
            self.quad_nccl.all_to_all_single(qrecv.reshape(-1), qsend.reshape(-1))

        # Phase 2: [src_local, dest_quad] -> [dest_quad, src_local]
        psend = recv.view(_PAIR, _QUAD, nchunk)
        psend.copy_(qrecv.permute(1, 0, 2))
        precv = send.view(_PAIR, _QUAD, nchunk)
        with self.pair_nccl.change_state(enable=True):
            self.pair_nccl.all_to_all_single(precv.reshape(-1), psend.reshape(-1))

        output.reshape(_WORLD, nchunk).copy_(precv.reshape(_WORLD, nchunk))
        return True
