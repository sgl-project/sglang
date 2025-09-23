from __future__ import annotations

import math
from typing import Optional, Union

import torch
import triton
import triton.language as tl

from sglang.srt.custom_op import CustomOp
from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8, 16, 32]
        for num_stages in [2, 3, 4]
    ],
    key=["N"],
)
@triton.jit
def layer_norm_gate_fwd_kernel(
    X,  # pointer to the input  # (M, G, N)
    O,  # pointer to the gate  # (M, G, N)
    Y,  # pointer to the output  # (M, G, N)
    W,  # pointer to the weights  # (G, N)
    Rstd,  # pointer to the 1/std  # (M, G)
    N,  # number of columns in X
    stride_x0,
    stride_x1,
    stride_x2,
    stride_o0,
    stride_o1,
    stride_o2,
    stride_y0,
    stride_y1,
    stride_y2,
    stride_w0,
    stride_w1,
    stride_rstd0,
    stride_rstd1,
    eps,  # epsilon to avoid division by zero
    BLOCK_N: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    group = tl.program_id(1)
    X += row * stride_x0 + group * stride_x1
    Y += row * stride_y0 + group * stride_y1
    O += row * stride_o0 + group * stride_o1
    W += group * stride_w0
    # Compute mean and variance
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols * stride_x2, mask=cols < N, other=0.0).to(
        tl.float32
    )  # (BLOCK_N, )
    xbar = tl.where(cols < N, x, 0.0)  # (BLOCK_N, )
    var = tl.sum(xbar * xbar, axis=0) / N  # (1, )
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + row * stride_rstd0 + group * stride_rstd1, rstd)
    # Normalize and apply linear transformation
    mask = cols < N
    w = tl.load(W + cols * stride_w1, mask=mask).to(tl.float32)  # (BLOCK_N, )
    x_hat = x * rstd  # (BLOCK_N, )
    y = x_hat * w  # (BLOCK_N, )
    # Swish output gate
    o = tl.load(O + cols * stride_o2, mask=cols < N, other=0.0).to(tl.float32)
    y = y * tl.sigmoid(o)
    # Write output
    tl.store(Y + cols * stride_y2, y, mask=mask)


def rms_norm_gate_fwd(
    x: torch.Tensor,  # (M, group, hidden_size)  attn_output, rmsnorm
    o: torch.Tensor,  # (M, group, hidden_size)  gate, sigmoid
    weight: torch.Tensor,  # (group, hidden_size)
    eps: float,
):
    M, G, N = x.shape
    # allocate output
    y = torch.empty_like(x)
    rstd = torch.empty((M, G), dtype=torch.float, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps
    layer_norm_gate_fwd_kernel[(M, G)](
        x,
        o,
        y,
        weight,
        rstd,
        N,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        y.stride(0),
        y.stride(1),
        y.stride(2),
        weight.stride(0),
        weight.stride(1),
        rstd.stride(0),
        rstd.stride(1),
        eps,
        BLOCK_N,
    )
    # residual_out is None if residual is None and residual_dtype == input_dtype
    return y, rstd


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8, 16, 32]
        for num_stages in [2, 3, 4]
    ],
    key=["N"],
)
@triton.jit
def layer_norm_gate_bwd_kernel(
    X,  # pointer to the input  # (M, G, N)
    O,  # pointer to the gate  # (M, G, N)
    W,  # pointer to the weights  # (G, N)
    DY,  # pointer to the output gradient  # (M, G, N)
    DX,  # pointer to the input gradient  # (M, G, N)
    DO,  # pointer to the gate gradient  # (M, G, N)
    DW,  # pointer to the partial sum of weights gradient  # (sm_count, G, N)
    Rstd,  # pointer to the 1/std  # (M, G)
    stride_x0,
    stride_x1,
    stride_x2,
    stride_o0,
    stride_o1,
    stride_o2,
    stride_w0,
    stride_w1,
    stride_dy0,
    stride_dy1,
    stride_dy2,
    stride_dx0,
    stride_dx1,
    stride_dx2,
    stride_do0,
    stride_do1,
    stride_do2,
    stride_dw0,
    stride_dw1,
    stride_dw2,
    stride_rstd0,
    stride_rstd1,
    M,  # number of rows in X
    G,  # number of groups in X
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    rows_per_program,
    BLOCK_N: tl.constexpr,
):
    # Map the program id to the elements of X, DX, and DY it should compute.
    row_block_id = tl.program_id(0)
    group = tl.program_id(1)
    row_start = row_block_id * rows_per_program
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    X += row_start * stride_x0 + group * stride_x1
    O += row_start * stride_o0 + group * stride_o1
    DY += row_start * stride_dy0 + group * stride_dy1
    DX += row_start * stride_dx0 + group * stride_dx1
    DO += row_start * stride_do0 + group * stride_do1
    Rstd += group * stride_rstd1
    w = tl.load(W + group * stride_w0 + cols * stride_w1, mask=mask).to(tl.float32)
    dw = tl.zeros((BLOCK_N,), dtype=tl.float32)
    row_end = min((row_block_id + 1) * rows_per_program, M)
    for row in range(row_start, row_end):
        # Load data to SRAM
        x = tl.load(X + cols * stride_x2, mask=mask, other=0).to(
            tl.float32
        )  # (BLOCK_N, )
        o = tl.load(O + cols * stride_o2, mask=mask, other=0).to(
            tl.float32
        )  # (BLOCK_N, )
        dy = tl.load(DY + cols * stride_dy2, mask=mask, other=0).to(
            tl.float32
        )  # (BLOCK_N, )
        rstd = tl.load(Rstd + row * stride_rstd0)  # (1, )
        # Compute dx
        xhat = x * rstd  # (BLOCK_N, )
        xhat = tl.where(mask, xhat, 0.0)
        y = xhat * w
        sigmoid_o = tl.sigmoid(o)
        do = dy * y * sigmoid_o * (1 - sigmoid_o)  # (BLOCK_N, )
        dy = dy * sigmoid_o  # (BLOCK_N, )
        wdy = dy * w  # (BLOCK_N, )
        dw += dy * xhat  # (BLOCK_N, )
        c1 = tl.sum(xhat * wdy, axis=0) / N  # (1, )
        dx = (wdy - xhat * c1) * rstd  # (BLOCK_N, )
        tl.store(DX + cols * stride_dx2, dx, mask=mask)
        tl.store(DO + cols * stride_do2, do, mask=mask)
        X += stride_x0
        O += stride_o0
        DY += stride_dy0
        DX += stride_dx0
        DO += stride_do0
    tl.store(
        DW + row_block_id * stride_dw0 + group * stride_dw1 + cols * stride_dw2,
        dw,
        mask=mask,
    )


def layer_norm_bwd(
    dy: torch.Tensor,  # (M, group, hidden_size)
    x: torch.Tensor,  # (M, group, hidden_size)
    o: torch.Tensor,  # (M, group, hidden_size)
    weight: torch.Tensor,  # (group, hidden_size)
    eps: float,
    rstd: torch.Tensor,  # (M, group)
    x_dtype: torch.dtype = None,
):
    M, G, N = x.shape
    assert dy.shape == (M, G, N)
    assert weight.shape == (G, N)
    # allocate output
    dx = torch.empty_like(x)
    do = torch.empty_like(o)
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
    sm_count = M  # math.ceil(sm_count/G)
    dw = torch.empty((sm_count, G, N), dtype=torch.float32, device=weight.device)
    rows_per_program = math.ceil(M / sm_count)
    grid = (sm_count, G)
    layer_norm_gate_bwd_kernel[grid](
        x,  # (M, group, hidden_size)
        o,  # (M, group, hidden_size)
        weight,  # (group, hidden_size)
        dy,  # (M, group, hidden_size)
        dx,  # (M, group, hidden_size)
        do,  # (M, group, hidden_size)
        dw,  # (sm_count, group, hidden_size)
        rstd,  # (M, group)
        x.stride(0),
        x.stride(1),
        x.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        weight.stride(0),
        weight.stride(1),
        dy.stride(0),
        dy.stride(1),
        dy.stride(2),
        dx.stride(0),
        dx.stride(1),
        dx.stride(2),
        do.stride(0),
        do.stride(1),
        do.stride(2),
        dw.stride(0),
        dw.stride(1),
        dw.stride(2),
        rstd.stride(0),
        rstd.stride(1),
        M,
        G,
        N,
        eps,
        rows_per_program,
        BLOCK_N,
    )
    dw = dw.sum(0).to(weight.dtype)
    return dx, do, dw


class GroupRMSNormSigmoidGateFn(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        x,  # (*, group, hidden_size)  attn_output, rmsnorm
        o,  # (*, group, hidden_size)  gate, sigmoid
        weight,  # (group, hidden_size)
        eps=1e-6,
    ):
        x_shape_og = x.shape
        o_shape_og = o.shape
        # reshape input data into 2D tensor
        x = x.reshape(-1, x.shape[-2], x.shape[-1])  # (M, group, hidden_size)
        o = o.reshape(-1, x.shape[-2], x.shape[-1])  # (M, group, hidden_size)
        y, rstd = rms_norm_gate_fwd(x, o, weight, eps)
        ctx.save_for_backward(x, o, weight, rstd)
        ctx.x_shape_og = x_shape_og
        ctx.o_shape_og = o_shape_og
        ctx.eps = eps
        ctx.x_dtype = x.dtype
        y = y.reshape(x_shape_og)
        return y

    @staticmethod
    def backward(ctx, dy, *args):  # (*, group, hidden_size)
        x, o, weight, rstd = ctx.saved_tensors
        dy = dy.reshape(-1, dy.shape[-2], dy.shape[-1])  # (M, group, hidden_size)
        assert dy.shape == x.shape
        dx, do, dw = layer_norm_bwd(
            dy,
            x,
            o,
            weight,
            ctx.eps,
            rstd,
            x_dtype=ctx.x_dtype,
        )
        return (dx.reshape(ctx.x_shape_og), do.reshape(ctx.o_shape_og), dw, None)


def group_rms_norm_sigmoid_gate_fn(
    x,  # (*, group, hidden_size)
    o,  # (*, group, hidden_size)
    weight,  # (group, hidden_size)
    eps=1e-6,
):
    return GroupRMSNormSigmoidGateFn.apply(x, o, weight, eps)


class BailingMoEFusedGroupRMSNormSigmoidGate(CustomOp):
    name = "BailingMoEFusedGroupRMSNormSigmoidGate"

    def __init__(self, hidden_size, eps: float = 1e-6, group_norm_size: int = 1):
        super().__init__()
        self.tp_world = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.group_norm_size = group_norm_size
        assert (
            self.tp_world <= self.group_norm_size
        ), "tp_world must be less than or equal to group_norm_size"
        assert (
            self.group_norm_size % self.tp_world == 0
        ), "group_norm_size must be divisible by tp_world"
        self.linear_attn_norm_group_size_per_rank = (
            self.group_norm_size // self.tp_world
        )
        self.weight = torch.nn.Parameter(
            torch.ones(
                self.linear_attn_norm_group_size_per_rank,
                int(hidden_size / self.group_norm_size),
            )
        )
        self.weight.weight_loader = self.weight_loader
        self.variance_epsilon = eps

    @staticmethod
    def weight_loader(
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
    ) -> None:
        tp_world = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        shard_size = loaded_weight.shape[0] // tp_world
        shard = slice(tp_rank * shard_size, (tp_rank + 1) * shard_size)
        param.data.copy_(loaded_weight[shard].view(param.data.size(0), -1).contiguous())
        return

    def forward(self, x, gate):
        out = group_rms_norm_sigmoid_gate_fn(
            x.view(x.size(0), self.linear_attn_norm_group_size_per_rank, -1),
            gate.view(gate.size(0), self.linear_attn_norm_group_size_per_rank, -1),
            self.weight.data,
            eps=self.variance_epsilon,
        )
        return out.view(out.size(0), -1)


class BailingMoERMSNormTP(CustomOp):
    name = "BailingMoERMSNormTP"

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.tp_world = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.weight = torch.nn.Parameter(torch.ones(int(hidden_size / self.tp_world)))
        self.weight.requires_grad = False
        self.weight.weight_loader = self.weight_loader
        self.variance_epsilon = eps
        return

    @staticmethod
    def weight_loader(
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
    ) -> None:
        tp_world = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()

        shard_size = loaded_weight.shape[0] // tp_world
        shard = slice(tp_rank * shard_size, (tp_rank + 1) * shard_size)
        param.data.copy_(loaded_weight[shard])
        return

    def _forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True, dtype=torch.float32)
        if self.tp_world > 1:
            variance = tensor_model_parallel_all_reduce(variance) / self.tp_world
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x.to(orig_dtype) * self.weight
        return x

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        assert residual is None, "RMSNorm does not support residual connection."
        return self._forward(x)
