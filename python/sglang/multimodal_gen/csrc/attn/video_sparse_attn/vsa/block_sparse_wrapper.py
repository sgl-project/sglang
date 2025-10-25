import torch

try:
    from vsa_cuda import block_sparse_bwd, block_sparse_fwd
except ImportError:
    block_sparse_fwd = None
    block_sparse_bwd = None
from vsa.block_sparse_attn_triton import (
    triton_block_sparse_attn_backward,
    triton_block_sparse_attn_forward,
)

assert torch.__version__ >= "2.4.0", "VSA requires PyTorch 2.4.0 or higher"
from typing import Tuple

from vsa.index import map_to_index


@torch.library.custom_op(
    "vsa::block_sparse_attn_triton", mutates_args=(), device_types="cuda"
)
def block_sparse_attn_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_map: torch.Tensor,
    variable_block_sizes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    block_map = block_map.int()
    q2k_block_sparse_index, q2k_block_sparse_num = map_to_index(block_map)
    o, M = triton_block_sparse_attn_forward(
        q, k, v, q2k_block_sparse_index, q2k_block_sparse_num, variable_block_sizes
    )
    return o, M


@torch.library.register_fake("vsa::block_sparse_attn_triton")
def _block_sparse_attn_triton_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_map: torch.Tensor,
    variable_block_sizes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    o = torch.empty_like(q)
    M = torch.empty(
        (q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
    )
    return o, M


@torch.library.custom_op(
    "vsa::block_sparse_attn_backward_triton", mutates_args=(), device_types="cuda"
)
def block_sparse_attn_backward_triton(
    grad_output_padded: torch.Tensor,
    q_padded: torch.Tensor,
    k_padded: torch.Tensor,
    v_padded: torch.Tensor,
    o_padded: torch.Tensor,
    M: torch.Tensor,
    block_map: torch.Tensor,
    variable_block_sizes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grad_output_padded = grad_output_padded.contiguous()
    q2k_block_sparse_index, q2k_block_sparse_num = map_to_index(block_map)
    k2q_block_sparse_index, k2q_block_sparse_num = map_to_index(
        block_map.transpose(-1, -2)
    )
    dq, dk, dv = triton_block_sparse_attn_backward(
        grad_output_padded,
        q_padded,
        k_padded,
        v_padded,
        o_padded,
        M,
        q2k_block_sparse_index,
        q2k_block_sparse_num,
        k2q_block_sparse_index,
        k2q_block_sparse_num,
        variable_block_sizes,
    )
    return dq, dk, dv


@torch.library.register_fake("vsa::block_sparse_attn_backward_triton")
def _block_sparse_attn_backward_triton_fake(
    grad_output_padded: torch.Tensor,
    q_padded: torch.Tensor,
    k_padded: torch.Tensor,
    v_padded: torch.Tensor,
    o_padded: torch.Tensor,
    M: torch.Tensor,
    block_map: torch.Tensor,
    variable_block_sizes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grad_output_padded = grad_output_padded.contiguous()
    dq = torch.empty_like(grad_output_padded)
    dk = torch.empty_like(grad_output_padded)
    dv = torch.empty_like(grad_output_padded)
    return dq, dk, dv


def backward_triton(ctx, grad_output1, grad_output2):
    q_padded, k_padded, v_padded, o_padded, M, block_map, variable_block_sizes = (
        ctx.saved_tensors
    )
    dq, dk, dv = block_sparse_attn_backward_triton(
        grad_output1,
        q_padded,
        k_padded,
        v_padded,
        o_padded,
        M,
        block_map,
        variable_block_sizes,
    )
    return dq, dk, dv, None, None


def setup_context_triton(ctx, inputs, output):
    q_padded, k_padded, v_padded, block_map, variable_block_sizes = inputs
    o_padded, M = output
    ctx.save_for_backward(
        q_padded, k_padded, v_padded, o_padded, M, block_map, variable_block_sizes
    )


block_sparse_attn_triton.register_autograd(
    backward_triton, setup_context=setup_context_triton
)


major, minor = torch.cuda.get_device_capability(0)

if major == 9 and minor == 0:  # check if H100

    @torch.library.custom_op(
        "vsa::block_sparse_attn_SM90", mutates_args=(), device_types="cuda"
    )
    def block_sparse_attn_SM90(
        q_padded: torch.Tensor,
        k_padded: torch.Tensor,
        v_padded: torch.Tensor,
        block_map: torch.Tensor,
        variable_block_sizes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_padded = q_padded.contiguous()
        k_padded = k_padded.contiguous()
        v_padded = v_padded.contiguous()
        q2k_block_sparse_index, q2k_block_sparse_num = map_to_index(block_map)
        variable_block_sizes = variable_block_sizes.int()
        o_padded, lse_padded = block_sparse_fwd(
            q_padded,
            k_padded,
            v_padded,
            q2k_block_sparse_index,
            q2k_block_sparse_num,
            variable_block_sizes,
        )
        return o_padded, lse_padded

    @torch.library.register_fake("vsa::block_sparse_attn_SM90")
    def _block_sparse_attn_SM90_fake(
        q_padded: torch.Tensor,
        k_padded: torch.Tensor,
        v_padded: torch.Tensor,
        block_map: torch.Tensor,
        variable_block_sizes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_padded, k_padded, v_padded = [
            x.contiguous() for x in (q_padded, k_padded, v_padded)
        ]
        B, H, S, D = q_padded.shape
        o_padded = torch.empty_like(q_padded)
        lse_padded = torch.empty(
            (B, H, S, 1), device=q_padded.device, dtype=torch.float32
        )
        return o_padded, lse_padded

    @torch.library.custom_op(
        "vsa::block_sparse_attn_backward_SM90", mutates_args=(), device_types="cuda"
    )
    def block_sparse_attn_backward_SM90(
        grad_output_padded: torch.Tensor,
        q_padded: torch.Tensor,
        k_padded: torch.Tensor,
        v_padded: torch.Tensor,
        o_padded: torch.Tensor,
        lse_padded: torch.Tensor,
        block_map: torch.Tensor,
        variable_block_sizes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        grad_output_padded = grad_output_padded.contiguous()
        k2q_block_sparse_index, k2q_block_sparse_num = map_to_index(
            block_map.transpose(-1, -2)
        )
        grad_q_padded, grad_k_padded, grad_v_padded = block_sparse_bwd(
            q_padded,
            k_padded,
            v_padded,
            o_padded,
            lse_padded,
            grad_output_padded,
            k2q_block_sparse_index,
            k2q_block_sparse_num,
            variable_block_sizes,
        )
        grad_q_padded = grad_q_padded.to(grad_output_padded.dtype)
        grad_k_padded = grad_k_padded.to(grad_output_padded.dtype)
        grad_v_padded = grad_v_padded.to(grad_output_padded.dtype)
        return grad_q_padded, grad_k_padded, grad_v_padded

    @torch.library.register_fake("vsa::block_sparse_attn_backward_SM90")
    def _block_sparse_attn_backward_SM90_fake(
        grad_output_padded: torch.Tensor,
        q_padded: torch.Tensor,
        k_padded: torch.Tensor,
        v_padded: torch.Tensor,
        o_padded: torch.Tensor,
        lse_padded: torch.Tensor,
        block_map: torch.Tensor,
        variable_block_sizes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        torch._check(grad_output_padded.dtype == torch.bfloat16)
        torch._check(lse_padded.dtype == torch.float32)
        grad_output_padded = grad_output_padded.contiguous()
        dq = torch.empty_like(grad_output_padded)
        dk = torch.empty_like(grad_output_padded)
        dv = torch.empty_like(grad_output_padded)
        return dq, dk, dv

    def backward_SM90(ctx, grad_output1, grad_output2):
        (
            q_padded,
            k_padded,
            v_padded,
            o_padded,
            lse_padded,
            block_map,
            variable_block_sizes,
        ) = ctx.saved_tensors
        dq, dk, dv = block_sparse_attn_backward_SM90(
            grad_output1,
            q_padded,
            k_padded,
            v_padded,
            o_padded,
            lse_padded,
            block_map,
            variable_block_sizes,
        )
        return dq, dk, dv, None, None

    def setup_context_SM90(ctx, inputs, output):
        q_padded, k_padded, v_padded, block_map, variable_block_sizes = inputs
        o_padded, lse_padded = output
        ctx.save_for_backward(
            q_padded,
            k_padded,
            v_padded,
            o_padded,
            lse_padded,
            block_map,
            variable_block_sizes,
        )

    block_sparse_attn_SM90.register_autograd(
        backward_SM90, setup_context=setup_context_SM90
    )
