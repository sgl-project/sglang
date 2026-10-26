"""Symmetric-memory workspaces of the SP collective GEMMs.

The kernels are ``all_gather_matmul`` / ``matmul_reduce_scatter`` in
``sglang.kernels.ops.communication.sp_collective_gemm``; pass
:func:`sp_collective_gemm_workspace` as their ``workspace``. One workspace per
``(kind, group, width, dtype, device)`` lives for the process and grows by
reallocation outside CUDA-graph capture.
"""

import torch
import torch.distributed as dist

from sglang.kernels.ops.communication.sp_collective_gemm import _TILE_M, _TILE_N

_workspaces = {}


class SPCollectiveGemmWorkspace:
    def __init__(
        self,
        kind,
        group,
        rows,
        width,
        dtype,
        device,
        row_align=_TILE_M,
        flag_tile_mn=(_TILE_M, _TILE_N),
    ):
        import cutlass.torch as cutlass_torch
        import torch.distributed._symmetric_memory as symm_mem

        # Power of two >= row_align, so capacity stays a whole number of M tiles.
        self.capacity = max(row_align, 1 << (rows - 1).bit_length())
        self.world = dist.get_world_size(group)
        self.sms = torch.cuda.get_device_properties(device).multi_processor_count
        self.storage = symm_mem.empty(
            (self.capacity, width), dtype=dtype, device=device
        )
        self.handle = symm_mem.rendezvous(self.storage, group)
        if kind == "ag":
            self.gathered = torch.empty(
                (self.capacity * self.world, width), dtype=dtype, device=device
            )
            self.flags = torch.zeros(self.world, dtype=torch.int32, device=device)
            self.copy_stream = torch.cuda.Stream(device=device)
        else:
            if not self.handle.multicast_ptr:
                raise RuntimeError(
                    "CUTLASS GEMM-RS requires NVSwitch multicast mapping"
                )
            self.multicast = cutlass_torch.as_tensor(
                self.handle.multicast_ptr, self.storage.shape, dtype
            )
            num_flags = (
                self.capacity // flag_tile_mn[0] * (width // flag_tile_mn[1])
                + self.sms
                + 8
            )
            self.flags = symm_mem.empty((num_flags,), dtype=torch.int32, device=device)
            self.flags.zero_()
            self.flag_handle = symm_mem.rendezvous(self.flags, group)
            if not self.flag_handle.multicast_ptr:
                raise RuntimeError("CUTLASS GEMM-RS requires multicast flag storage")
            self.flag_multicast = cutlass_torch.as_tensor(
                self.flag_handle.multicast_ptr, self.flags.shape, self.flags.dtype
            )
            self.peer_flags = tuple(
                self.flag_handle.get_buffer(r, self.flags.shape, self.flags.dtype)
                for r in range(self.world)
            )
        self.handle.barrier(channel=0)


def sp_collective_gemm_workspace(
    kind, group, rows, width, x, row_align=_TILE_M, flag_tile_mn=(_TILE_M, _TILE_N)
):
    key = (kind, group, width, x.dtype, x.device)
    ws = _workspaces.get(key)
    if ws is None or rows > ws.capacity:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm up CUTLASS SP workspace before CUDA graph capture")
        if ws is not None:
            # Peer mappings must stay alive until all ranks stop reading them.
            ws.handle.barrier(channel=0)
            torch.cuda.current_stream().synchronize()
        ws = SPCollectiveGemmWorkspace(
            kind, group, rows, width, x.dtype, x.device, row_align, flag_tile_mn
        )
        _workspaces[key] = ws
    return ws
