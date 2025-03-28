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
    # TRTLLM custom allreduce
    def init_custom_reduce(
        rank_id, num_devices, rank_data, buffers, tmp_buffers, barrier_in, barrier_out
    ):
        return torch.ops.sgl_kernel.init_custom_ar.default(
            rank_id,
            num_devices,
            rank_data,
            buffers,
            tmp_buffers,
            barrier_in,
            barrier_out,
        )

    def custom_dispose(fa):
        torch.ops.sgl_kernel.dispose.default(fa)

    def custom_reduce(fa, inp, out):
        torch.ops.sgl_kernel.all_reduce.default(fa, inp, out)

    def get_graph_buffer_ipc_meta(fa):
        return torch.ops.sgl_kernel.get_graph_buffer_ipc_meta.default(fa)

    def register_graph_buffers(fa, handles, offsets):
        torch.ops.sgl_kernel.register_graph_buffers.default(fa, handles, offsets)
