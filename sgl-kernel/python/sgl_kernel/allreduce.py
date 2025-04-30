from typing import List, Tuple

import torch

if torch.version.hip is not None:
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
