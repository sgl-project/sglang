from typing import List, Optional, Tuple

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

    # mscclpp
    def mscclpp_generate_unique_id() -> bytes:
        raise NotImplementedError()

    def mscclpp_init_context(
        unique_id: bytes,
        rank: int,
        world_size: int,
        scratch: torch.Tensor,
        put_buffer: torch.Tensor,
        nranks_per_node: int,
        rank_to_node: List[int],
        rank_to_ib: List[int],
        context_selection: int,
    ) -> int:
        raise NotImplementedError()

    def mscclpp_allreduce(
        context: int, inp: torch.Tensor, out: torch.Tensor, nthreads: int, nblocks: int
    ) -> None:
        raise NotImplementedError()

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

    def mscclpp_generate_unique_id() -> torch.Tensor:
        return torch.ops.sgl_kernel.mscclpp_generate_unique_id.default()

    def mscclpp_init_context(
        unique_id: torch.Tensor,
        rank: int,
        world_size: int,
        scratch: torch.Tensor,
        put_buffer: torch.Tensor,
        nranks_per_node: int,
        rank_to_node: List[int],
        rank_to_ib: List[int],
        context_selection: int,
    ) -> int:
        return torch.ops.sgl_kernel.mscclpp_init_context.default(
            unique_id,
            rank,
            world_size,
            scratch,
            put_buffer,
            nranks_per_node,
            rank_to_node,
            rank_to_ib,
            context_selection,
        )

    def mscclpp_allreduce(
        context: int, inp: torch.Tensor, out: torch.Tensor, nthreads: int, nblocks: int
    ) -> None:
        torch.ops.sgl_kernel.mscclpp_allreduce.default(
            context, inp, out, nthreads, nblocks
        )
