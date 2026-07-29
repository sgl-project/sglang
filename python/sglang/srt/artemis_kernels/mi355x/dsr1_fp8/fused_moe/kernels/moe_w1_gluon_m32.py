"""Expert-indexed W1 consuming route metadata prepared before the launch."""

from __future__ import annotations

import torch
import triton.experimental.gluon.language as gl
from triton.experimental import gluon

M = gl.constexpr(32)
H = gl.constexpr(7168)
I = gl.constexpr(256)
E = gl.constexpr(257)
BLOCK_M = gl.constexpr(32)
BLOCK_N = gl.constexpr(256)
BLOCK_K = gl.constexpr(128)
K_UNROLL = gl.constexpr(8)
NUM_WARPS = gl.constexpr(8)


@gluon.jit
def _load_preshuffled_w1_128x256(
    weight_ptr,
    expert_id,
    n_group_base,
    k_block,
    packed_layout: gl.constexpr,
    dot_b_layout: gl.constexpr,
):
    packed_k = gl.arange(0, BLOCK_K * 16, layout=gl.SliceLayout(0, packed_layout))
    packed_n = gl.arange(0, BLOCK_N // 16, layout=gl.SliceLayout(1, packed_layout))
    offsets = (
        expert_id * (2 * I * H)
        + (n_group_base + packed_n[:, None]) * (H * 16)
        + k_block * (BLOCK_K * 16)
        + packed_k[None, :]
    )
    packed = gl.amd.cdna4.buffer_load(weight_ptr, offsets, cache=".cg")
    logical = (
        packed.reshape(1, BLOCK_N // 16, BLOCK_K // 32, 2, 16, 16)
        .permute(0, 1, 4, 2, 3, 5)
        .reshape(BLOCK_N, BLOCK_K)
        .trans(1, 0)
    )
    return gl.convert_layout(logical, dot_b_layout, assert_trivial=True)


@gluon.jit
def _load_w1_group(
    x_ptr,
    w1_ptr,
    x_scale_ptr,
    w1_scale_ptr,
    token_ids,
    valid_rows,
    x_shared,
    x_scale_shared,
    expert_id,
    k_block,
    x_load_layout: gl.constexpr,
    mfma_layout: gl.constexpr,
    dot_a_layout: gl.constexpr,
    dot_b_layout: gl.constexpr,
    packed_weight_layout: gl.constexpr,
):
    row_mfma_layout: gl.constexpr = gl.SliceLayout(1, mfma_layout)
    k_offsets = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, x_load_layout))
    output_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, mfma_layout))
    x_offsets = token_ids[:, None] * H + k_block * BLOCK_K + k_offsets[None, :]
    x = gl.amd.cdna4.buffer_load(x_ptr, x_offsets, mask=valid_rows[:, None])
    x_scales = gl.amd.cdna4.buffer_load(
        x_scale_ptr,
        k_block * M + token_ids,
        mask=valid_rows,
    )
    x_shared.store(x)
    x_scale_shared.store(x_scales)
    x_dot = x_shared.load(layout=dot_a_layout)
    x_scale = x_scale_shared.load(layout=row_mfma_layout)
    gate = _load_preshuffled_w1_128x256(
        w1_ptr,
        expert_id,
        0,
        k_block,
        packed_layout=packed_weight_layout,
        dot_b_layout=dot_b_layout,
    )
    up = _load_preshuffled_w1_128x256(
        w1_ptr,
        expert_id,
        I // 16,
        k_block,
        packed_layout=packed_weight_layout,
        dot_b_layout=dot_b_layout,
    )
    gate_scale = gl.load(w1_scale_ptr + expert_id * 4 * 56 + (output_n // BLOCK_K) * 56 + k_block)
    up_scale = gl.load(w1_scale_ptr + expert_id * 4 * 56 + (2 + output_n // BLOCK_K) * 56 + k_block)
    return x_dot, x_scale, gate, up, gate_scale, up_scale


@gluon.jit
def _accumulate_w1_group(operands, gate_acc, up_acc, zero):
    x_dot, x_scale, gate, up, gate_scale, up_scale = operands
    gate_group = gl.amd.cdna4.mfma_scaled(
        x_dot,
        None,
        "e4m3",
        gate,
        None,
        "e4m3",
        zero,
    )
    up_group = gl.amd.cdna4.mfma_scaled(
        x_dot,
        None,
        "e4m3",
        up,
        None,
        "e4m3",
        zero,
    )
    gate_acc += gate_group * x_scale[:, None] * gate_scale[None, :]
    up_acc += up_group * x_scale[:, None] * up_scale[None, :]
    return gate_acc, up_acc


@gluon.jit
def _compute_w1_k8(
    x_ptr,
    w1_ptr,
    x_scale_ptr,
    w1_scale_ptr,
    token_shared,
    x_shared,
    x_scale_shared,
    activation_shared,
    expert_id,
    x_load_layout: gl.constexpr,
    mfma_layout: gl.constexpr,
    dot_a_layout: gl.constexpr,
    dot_b_layout: gl.constexpr,
    packed_weight_layout: gl.constexpr,
):
    """Compute and quantize the selected load-first K-unroll-8 W1 body."""
    row_load_layout: gl.constexpr = gl.SliceLayout(1, x_load_layout)
    token_ids = token_shared.load(layout=row_load_layout)
    valid_rows = token_ids < M
    gate_acc = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, layout=mfma_layout)
    up_acc = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, layout=mfma_layout)
    zero = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, layout=mfma_layout)

    for k_base in range(0, H // BLOCK_K, K_UNROLL):
        operands0 = _load_w1_group(
            x_ptr,
            w1_ptr,
            x_scale_ptr,
            w1_scale_ptr,
            token_ids,
            valid_rows,
            x_shared,
            x_scale_shared,
            expert_id,
            k_base,
            x_load_layout=x_load_layout,
            mfma_layout=mfma_layout,
            dot_a_layout=dot_a_layout,
            dot_b_layout=dot_b_layout,
            packed_weight_layout=packed_weight_layout,
        )
        operands1 = _load_w1_group(
            x_ptr,
            w1_ptr,
            x_scale_ptr,
            w1_scale_ptr,
            token_ids,
            valid_rows,
            x_shared,
            x_scale_shared,
            expert_id,
            k_base + 1,
            x_load_layout=x_load_layout,
            mfma_layout=mfma_layout,
            dot_a_layout=dot_a_layout,
            dot_b_layout=dot_b_layout,
            packed_weight_layout=packed_weight_layout,
        )
        operands2 = _load_w1_group(
            x_ptr,
            w1_ptr,
            x_scale_ptr,
            w1_scale_ptr,
            token_ids,
            valid_rows,
            x_shared,
            x_scale_shared,
            expert_id,
            k_base + 2,
            x_load_layout=x_load_layout,
            mfma_layout=mfma_layout,
            dot_a_layout=dot_a_layout,
            dot_b_layout=dot_b_layout,
            packed_weight_layout=packed_weight_layout,
        )
        operands3 = _load_w1_group(
            x_ptr,
            w1_ptr,
            x_scale_ptr,
            w1_scale_ptr,
            token_ids,
            valid_rows,
            x_shared,
            x_scale_shared,
            expert_id,
            k_base + 3,
            x_load_layout=x_load_layout,
            mfma_layout=mfma_layout,
            dot_a_layout=dot_a_layout,
            dot_b_layout=dot_b_layout,
            packed_weight_layout=packed_weight_layout,
        )
        operands4 = _load_w1_group(
            x_ptr,
            w1_ptr,
            x_scale_ptr,
            w1_scale_ptr,
            token_ids,
            valid_rows,
            x_shared,
            x_scale_shared,
            expert_id,
            k_base + 4,
            x_load_layout=x_load_layout,
            mfma_layout=mfma_layout,
            dot_a_layout=dot_a_layout,
            dot_b_layout=dot_b_layout,
            packed_weight_layout=packed_weight_layout,
        )
        operands5 = _load_w1_group(
            x_ptr,
            w1_ptr,
            x_scale_ptr,
            w1_scale_ptr,
            token_ids,
            valid_rows,
            x_shared,
            x_scale_shared,
            expert_id,
            k_base + 5,
            x_load_layout=x_load_layout,
            mfma_layout=mfma_layout,
            dot_a_layout=dot_a_layout,
            dot_b_layout=dot_b_layout,
            packed_weight_layout=packed_weight_layout,
        )
        operands6 = _load_w1_group(
            x_ptr,
            w1_ptr,
            x_scale_ptr,
            w1_scale_ptr,
            token_ids,
            valid_rows,
            x_shared,
            x_scale_shared,
            expert_id,
            k_base + 6,
            x_load_layout=x_load_layout,
            mfma_layout=mfma_layout,
            dot_a_layout=dot_a_layout,
            dot_b_layout=dot_b_layout,
            packed_weight_layout=packed_weight_layout,
        )
        operands7 = _load_w1_group(
            x_ptr,
            w1_ptr,
            x_scale_ptr,
            w1_scale_ptr,
            token_ids,
            valid_rows,
            x_shared,
            x_scale_shared,
            expert_id,
            k_base + 7,
            x_load_layout=x_load_layout,
            mfma_layout=mfma_layout,
            dot_a_layout=dot_a_layout,
            dot_b_layout=dot_b_layout,
            packed_weight_layout=packed_weight_layout,
        )
        gate_acc, up_acc = _accumulate_w1_group(operands0, gate_acc, up_acc, zero)
        gate_acc, up_acc = _accumulate_w1_group(operands1, gate_acc, up_acc, zero)
        gate_acc, up_acc = _accumulate_w1_group(operands2, gate_acc, up_acc, zero)
        gate_acc, up_acc = _accumulate_w1_group(operands3, gate_acc, up_acc, zero)
        gate_acc, up_acc = _accumulate_w1_group(operands4, gate_acc, up_acc, zero)
        gate_acc, up_acc = _accumulate_w1_group(operands5, gate_acc, up_acc, zero)
        gate_acc, up_acc = _accumulate_w1_group(operands6, gate_acc, up_acc, zero)
        gate_acc, up_acc = _accumulate_w1_group(operands7, gate_acc, up_acc, zero)

    activated = gate_acc * (1.0 / (1.0 + gl.exp(-gate_acc))) * up_acc
    activation_shared.store(activated)
    activated0 = activation_shared.slice(0, BLOCK_K, dim=1).load(layout=x_load_layout)
    activated1 = activation_shared.slice(BLOCK_K, BLOCK_K, dim=1).load(layout=x_load_layout)
    row_amax0 = gl.maximum(
        gl.max(gl.abs(activated0), axis=1, keep_dims=True),
        1.0e-10,
    )
    row_amax1 = gl.maximum(
        gl.max(gl.abs(activated1), axis=1, keep_dims=True),
        1.0e-10,
    )
    row_scale0 = row_amax0.to(gl.float32) * (1.0 / 448.0)
    row_scale1 = row_amax1.to(gl.float32) * (1.0 / 448.0)
    quantized0 = activated0 * (1.0 / row_scale0)
    quantized1 = activated1 * (1.0 / row_scale1)
    quantized0 = gl.minimum(gl.maximum(quantized0, -448.0), 448.0)
    quantized1 = gl.minimum(gl.maximum(quantized1, -448.0), 448.0)
    return (
        quantized0.to(x_ptr.type.element_ty),
        quantized1.to(x_ptr.type.element_ty),
        gl.reshape(row_scale0, (BLOCK_M,)),
        gl.reshape(row_scale1, (BLOCK_M,)),
    )


_WORKSPACES: dict[tuple[torch.device, int], tuple[torch.Tensor, torch.Tensor]] = {}


def _get_moe_workspaces_m32(
    x: torch.Tensor,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cached FP8 activation and FP32 scale workspaces.

    The buffers are keyed by device and expert count so their addresses remain
    stable across CUDA graph replays.  The selected M32 path stores two K128
    activation groups per expert.
    """
    key = (x.device, num_experts)
    workspaces = _WORKSPACES.get(key)
    if workspaces is None:
        workspaces = (
            torch.empty(
                (num_experts, 2, int(BLOCK_M), int(BLOCK_K)),
                dtype=x.dtype,
                device=x.device,
            ),
            torch.empty(
                (num_experts, 2, int(BLOCK_M)),
                dtype=torch.float32,
                device=x.device,
            ),
        )
        _WORKSPACES[key] = workspaces
    return workspaces


@gluon.jit
def _w1_precompacted_kernel(
    x_ptr,
    w1_ptr,
    x_scale_ptr,
    w1_scale_ptr,
    intermediate_workspace_ptr,
    intermediate_scale_workspace_ptr,
    route_tokens_ptr,
    route_counts_ptr,
):
    expert_id = gl.program_id(0)
    if gl.load(route_counts_ptr + expert_id) == 0:
        return

    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[32, 32, 64],
        transposed=True,
        warps_per_cta=[1, NUM_WARPS],
    )
    dot_a_layout: gl.constexpr = gl.DotOperandLayout(0, mfma_layout, 16)
    dot_b_layout: gl.constexpr = gl.DotOperandLayout(1, mfma_layout, 16)
    x_load_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[8, 8],
        warps_per_cta=[4, 2],
        order=[1, 0],
    )
    packed_weight_layout: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [0, 512], [0, 1024]],
        lane_bases=[[0, 16], [0, 32], [0, 64], [0, 128], [1, 0], [0, 256]],
        warp_bases=[[2, 0], [4, 0], [8, 0]],
        block_bases=[],
        shape=[BLOCK_N // 16, BLOCK_K * 16],
    )
    tile_shared_layout: gl.constexpr = gl.SwizzledSharedLayout(
        vec=16,
        per_phase=2,
        max_phase=8,
        order=[1, 0],
    )
    row_shared_layout: gl.constexpr = gl.SwizzledSharedLayout(
        vec=1,
        per_phase=1,
        max_phase=1,
        order=[0],
    )
    x_shared = gl.allocate_shared_memory(
        x_ptr.type.element_ty,
        [BLOCK_M, BLOCK_K],
        layout=tile_shared_layout,
    )
    x_scale_shared = gl.allocate_shared_memory(
        gl.float32,
        [BLOCK_M],
        layout=row_shared_layout,
    )
    activation_shared = gl.allocate_shared_memory(
        gl.float32,
        [BLOCK_M, BLOCK_N],
        layout=tile_shared_layout,
    )
    token_shared = gl.allocate_shared_memory(
        gl.int32,
        [BLOCK_M],
        layout=row_shared_layout,
    )
    row_load_layout: gl.constexpr = gl.SliceLayout(1, x_load_layout)
    rows = gl.arange(0, BLOCK_M, layout=row_load_layout)
    route_base = expert_id * BLOCK_M
    token_shared.store(gl.amd.cdna4.buffer_load(route_tokens_ptr, route_base + rows))

    inter0, inter1, scale0, scale1 = _compute_w1_k8(
        x_ptr,
        w1_ptr,
        x_scale_ptr,
        w1_scale_ptr,
        token_shared,
        x_shared,
        x_scale_shared,
        activation_shared,
        expert_id,
        x_load_layout=x_load_layout,
        mfma_layout=mfma_layout,
        dot_a_layout=dot_a_layout,
        dot_b_layout=dot_b_layout,
        packed_weight_layout=packed_weight_layout,
    )
    cols = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, x_load_layout))
    workspace_base = expert_id * 2 * BLOCK_M * BLOCK_K
    workspace_offsets = rows[:, None] * BLOCK_K + cols[None, :]
    gl.amd.cdna4.buffer_store(
        inter0,
        intermediate_workspace_ptr,
        workspace_base + workspace_offsets,
    )
    gl.amd.cdna4.buffer_store(
        inter1,
        intermediate_workspace_ptr,
        workspace_base + BLOCK_M * BLOCK_K + workspace_offsets,
    )
    scale0 = gl.convert_layout(scale0, row_load_layout)
    scale1 = gl.convert_layout(scale1, row_load_layout)
    scale_base = expert_id * 2 * BLOCK_M
    gl.amd.cdna4.buffer_store(
        scale0,
        intermediate_scale_workspace_ptr,
        scale_base + rows,
    )
    gl.amd.cdna4.buffer_store(
        scale1,
        intermediate_scale_workspace_ptr,
        scale_base + BLOCK_M + rows,
    )


def moe_w1_gluon_m32(
    x: torch.Tensor,
    w1: torch.Tensor,
    x_scale: torch.Tensor,
    w1_scale: torch.Tensor,
    route_tokens: torch.Tensor,
    route_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run exact K8 W1 using precomputed stable expert-indexed routes."""
    intermediate, intermediate_scale = _get_moe_workspaces_m32(x, int(E))
    _w1_precompacted_kernel[(int(E),)](
        x,
        w1,
        x_scale,
        w1_scale,
        intermediate,
        intermediate_scale,
        route_tokens,
        route_counts,
        num_warps=8,
        waves_per_eu=1,
    )
    return intermediate, intermediate_scale


__all__ = ["moe_w1_gluon_m32"]
