from typing import List, Optional, Tuple

import torch


def _has_custom_ar() -> bool:
    """True if the custom all-reduce operators are compiled and registered.

    The custom/deterministic/quick all-reduce ops are CDNA-only and are omitted
    from the RDNA build (see sgl-kernel/setup_rocm.py). Probing the registered
    ops instead of the device arch avoids initializing CUDA at import time:
    eagerly calling torch.cuda.get_device_properties(0) here would force CUDA
    init on device 0, which breaks fork-based multiprocessing and pins the
    process to device 0 before CUDA_VISIBLE_DEVICES is honored. common_ops is
    loaded before this module, so the registry is already populated.
    """
    try:
        return hasattr(torch.ops.sgl_kernel, "init_custom_ar")
    except Exception:
        return False


# CDNA ROCm builds compile the custom all-reduce ops; RDNA builds omit them
# (multi-GPU all-reduce falls back to RCCL). CUDA always has them and is handled
# by the final `else` branch, so gate the ROCm wrappers on hip + op presence.
_HAS_CUSTOM_AR = torch.version.hip is not None and _has_custom_ar()

if _HAS_CUSTOM_AR:
    # ROCM custom allreduce
    def init_custom_ar(
        meta: torch.Tensor,
        rank_data: torch.Tensor,
        handles: List[str],
        offsets: List[int],
        rank: int,
        full_nvlink: bool,
    ) -> int:
        return torch.ops.sgl_kernel.init_custom_ar.default(
            meta, rank_data, handles, offsets, rank, full_nvlink
        )

    def all_reduce_reg(fa: int, inp: torch.Tensor, out: torch.Tensor) -> None:
        torch.ops.sgl_kernel.all_reduce_reg.default(fa, inp, out)

    def all_reduce_unreg(
        fa: int, inp: torch.Tensor, reg_buffer: torch.Tensor, out: torch.Tensor
    ) -> None:
        torch.ops.sgl_kernel.all_reduce_unreg.default(fa, inp, reg_buffer, out)

    def deterministic_all_reduce_reg(
        fa: int, inp: torch.Tensor, out: torch.Tensor
    ) -> None:
        torch.ops.sgl_kernel.deterministic_all_reduce_reg.default(fa, inp, out)

    def deterministic_all_reduce_unreg(
        fa: int, inp: torch.Tensor, reg_buffer: torch.Tensor, out: torch.Tensor
    ) -> None:
        torch.ops.sgl_kernel.deterministic_all_reduce_unreg.default(
            fa, inp, reg_buffer, out
        )

    def dispose(fa: int) -> None:
        torch.ops.sgl_kernel.dispose.default(fa)

    def meta_size() -> int:
        return torch.ops.sgl_kernel.meta_size.default()

    def register_buffer(
        fa: int, t: torch.Tensor, handles: List[str], offsets: List[int]
    ) -> None:
        return torch.ops.sgl_kernel.register_buffer.default(fa, t, handles, offsets)

    def get_graph_buffer_ipc_meta(fa: int) -> Tuple[torch.Tensor, List[int]]:
        return torch.ops.sgl_kernel.get_graph_buffer_ipc_meta.default(fa)

    def register_graph_buffers(
        fa: int, handles: List[str], offsets: List[List[int]]
    ) -> None:
        torch.ops.sgl_kernel.register_graph_buffers.default(fa, handles, offsets)

    def allocate_meta_buffer(size: int) -> torch.Tensor:
        return torch.ops.sgl_kernel.allocate_meta_buffer.default(size)

    def get_meta_buffer_ipc_handle(inp: torch.Tensor) -> torch.Tensor:
        return torch.ops.sgl_kernel.get_meta_buffer_ipc_handle.default(inp)

    # ROCM quick allreduce
    def init_custom_qr(
        rank: int, world_size: int, qr_max_size: Optional[int] = None
    ) -> int:
        return torch.ops.sgl_kernel.init_custom_qr.default(
            world_size, rank, qr_max_size
        )

    def qr_get_handle(fa: int) -> torch.Tensor:
        return torch.ops.sgl_kernel.qr_get_handle.default(fa)

    def qr_open_handles(fa: int, handles: list[torch.Tensor]) -> None:
        torch.ops.sgl_kernel.qr_open_handles.default(fa, handles)

    def qr_all_reduce(
        fa: int,
        profile: int,
        inp: torch.Tensor,
        out: torch.Tensor,
        cast_bf162half: bool,
    ) -> None:
        torch.ops.sgl_kernel.qr_all_reduce.default(
            fa, profile, inp, out, cast_bf162half
        )

    def qr_destroy(fa: int) -> None:
        torch.ops.sgl_kernel.qr_destroy.default(fa)

    def qr_max_size() -> int:
        return torch.ops.sgl_kernel.qr_max_size.default()

elif torch.version.hip is not None:
    # RDNA (e.g. gfx1151 / Strix Halo): the CDNA-only custom/deterministic/quick
    # all-reduce ops are not built, so no wrappers are exposed here. Multi-GPU
    # all-reduce falls back to RCCL; single-GPU never calls all-reduce.
    pass

else:

    def init_custom_ar(
        ipc_tensors: List[int], rank_data: torch.Tensor, rank: int, full_nvlink: bool
    ) -> int:
        return torch.ops.sgl_kernel.init_custom_ar.default(
            ipc_tensors, rank_data, rank, full_nvlink
        )

    def dispose(fa: int) -> None:
        torch.ops.sgl_kernel.dispose.default(fa)

    def all_reduce(
        fa: int,
        inp: torch.Tensor,
        out: torch.Tensor,
        reg_buffer: int,
        reg_buffer_sz_bytes: int,
    ) -> None:
        torch.ops.sgl_kernel.all_reduce.default(
            fa, inp, out, reg_buffer, reg_buffer_sz_bytes
        )

    def get_graph_buffer_ipc_meta(fa) -> Tuple[List[int], List[int]]:
        return torch.ops.sgl_kernel.get_graph_buffer_ipc_meta.default(fa)

    def register_buffer(fa: int, fake_ipc_ptrs: List[int]) -> None:
        return torch.ops.sgl_kernel.register_buffer.default(fa, fake_ipc_ptrs)

    def register_graph_buffers(
        fa: int, handles: List[List[int]], offsets: List[List[int]]
    ) -> None:
        torch.ops.sgl_kernel.register_graph_buffers.default(fa, handles, offsets)

    def meta_size() -> int:
        return torch.ops.sgl_kernel.meta_size.default()
