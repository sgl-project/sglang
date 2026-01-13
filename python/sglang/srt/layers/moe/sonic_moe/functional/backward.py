# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

from typing import Optional

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import torch
import triton
import triton.language as tl

from sglang.srt.layers.moe.sonic_moe.enums import (
    LIBRARY_NAME,
    TENSORMAP,
    ActivationType,
    is_glu,
)
from sglang.srt.layers.moe.sonic_moe.utils import (
    ceil_divide,
    convert_torch_tensor_to_cute_tensor,
    get_powers_of_2,
)

from .moe_config import (
    HopperWgmma_MoE_Down_proj_ActGrad_Bwd,
    HopperWgmma_MoE_Down_proj_WeightGrad_Bwd,
    HopperWgmma_MoE_Up_proj_ActGrad_Bwd,
    HopperWgmma_MoE_Up_proj_WeightGrad_Bwd,
)
from .reduction_over_k_gather import token_gather_and_sum_varlen_K_triton


def _get_autotune_configs_for_db2_and_ds() -> list[triton.Config]:
    configs = []
    for BLOCK_TK in get_powers_of_2(4, 32):
        configs.append(triton.Config({"BLOCK_TK": BLOCK_TK}, num_warps=8, num_stages=4))
    return configs


@triton.autotune(
    configs=_get_autotune_configs_for_db2_and_ds(),
    key=["H", "E"],
)
@triton.jit
def db2_and_ds_kernel(
    dout_ptr,  # (T, H)
    s_ptr,  # (TK,)
    new_ds_partial_ptr,  # (TK, n_h_blocks)
    old_ds_partial_ptr,  # (TK, OLD_DS_PARTIAL_N)
    b2_ptr,  # (E, H),
    db2_ptr,  # (E, H),
    x_gather_idx_ptr,  # (TK,), maps grouped -> token index
    s_scatter_idx_ptr,  # (TK,), maps grouped -> scatter index
    expert_offset_ptr,  # (E+1,), offsets in grouped layout
    H: tl.constexpr,
    E: tl.constexpr,
    OLD_DS_PARTIAL_N: tl.constexpr,
    BLOCK_H: tl.constexpr,  # Block size for H dimension
    BLOCK_TK: tl.constexpr,  # Block size for token dimension
    BLOCK_OLD_DS_PARTIAL_N: tl.constexpr,
):
    Eidx = tl.program_id(0)  # expert id
    Hidx = tl.program_id(1)  # h-block id
    NUM_H_BLOCKS: tl.constexpr = tl.num_programs(1)

    # Hidden dimension indices for this block
    h_offsets = Hidx * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offsets < H

    E_count_start = tl.load(expert_offset_ptr + Eidx)
    E_count_end = tl.load(expert_offset_ptr + Eidx + 1)
    n_tokens = E_count_end - E_count_start

    b2 = tl.load(b2_ptr + Eidx * H + h_offsets, mask=h_mask, other=0.0).to(tl.float32)

    db2_acc = tl.zeros([BLOCK_H], dtype=tl.float32)

    # Process tokens in blocks of BLOCK_TK
    for block_start in tl.range(0, n_tokens, BLOCK_TK):
        # Token offsets within this block
        tk_offsets = block_start + tl.arange(0, BLOCK_TK)
        tk_mask = tk_offsets < n_tokens
        tk_grouped = E_count_start + tk_offsets

        # Gather token indices: [BLOCK_TK]
        token_indices = tl.load(
            x_gather_idx_ptr + tk_grouped, mask=tk_mask, other=0
        ).to(tl.uint32)

        # Get scatter indices: [BLOCK_TK]
        scatter_indices = tl.load(
            s_scatter_idx_ptr + tk_grouped, mask=tk_mask, other=0
        ).to(tl.uint32)

        s = tl.load(s_ptr + scatter_indices, mask=tk_mask, other=0.0).to(tl.float32)

        # Gather dout: [BLOCK_TK, BLOCK_H]
        dout_offsets = token_indices[:, None] * H + h_offsets[None, :]
        dout_mask = tk_mask[:, None] & h_mask[None, :]
        dout = tl.load(dout_ptr + dout_offsets, mask=dout_mask, other=0.0).to(
            tl.float32
        )

        # Accumulate db2: sum over tokens of (dout * s)
        db2_acc += tl.sum(dout * s[:, None], axis=0)  # Sum over BLOCK_TK dimension

        # Compute ds: dot(dout, b2) for this H-block
        ds_partial = tl.sum(dout * b2[None, :], axis=1)  # [BLOCK_TK]

        # On first H-block, add old_ds_partial.sum(dim=1)
        if Hidx == 0:
            n_offsets = tl.arange(0, BLOCK_OLD_DS_PARTIAL_N)
            old_ds_partial_offsets = (
                scatter_indices[:, None] * OLD_DS_PARTIAL_N + n_offsets[None, :]
            )
            old_ds_partial_mask = tk_mask[:, None] & (
                n_offsets[None, :] < OLD_DS_PARTIAL_N
            )
            old_ds_partial_vals = tl.load(
                old_ds_partial_ptr + old_ds_partial_offsets,
                mask=old_ds_partial_mask,
                other=0.0,
            ).to(tl.float32)
            ds_partial += tl.sum(old_ds_partial_vals, axis=1)

        tl.store(
            new_ds_partial_ptr + scatter_indices * NUM_H_BLOCKS + Hidx,
            ds_partial,
            mask=tk_mask,
        )

    tl.store(db2_ptr + Eidx * H + h_offsets, db2_acc, mask=h_mask)


def _get_autotune_configs_for_db1() -> list[triton.Config]:
    configs = []
    for BLOCK_TK in get_powers_of_2(4, 128):
        for BLOCK_I in get_powers_of_2(64, 4096):
            if 4096 <= BLOCK_I * BLOCK_TK <= 16384:
                configs.append(
                    triton.Config(
                        {"BLOCK_I": BLOCK_I, "BLOCK_TK": BLOCK_TK},
                        num_warps=8,
                        num_stages=4,
                    )
                )
    return configs


def _prune_triton_autotune_config(configs, nargs, **kw):
    pruned_configs = []
    for c in configs:
        if c.kwargs["BLOCK_I"] <= triton.next_power_of_2(nargs["I"]):
            pruned_configs.append(c)
    return pruned_configs


@triton.autotune(
    configs=_get_autotune_configs_for_db1(),
    key=["I", "E"],
    prune_configs_by={"early_config_prune": _prune_triton_autotune_config},
)
@triton.jit
def db1_kernel(
    dz_ptr,  # (T, H)
    db1_ptr,  # (E, H),
    expert_offset_ptr,  # (E+1,), offsets in grouped layout
    I: tl.constexpr,
    E: tl.constexpr,
    BLOCK_I: tl.constexpr,  # Block size for H dimension
    BLOCK_TK: tl.constexpr,  # Block size for token dimension
):
    Eidx = tl.program_id(0)  # expert id

    E_count_start = tl.load(expert_offset_ptr + Eidx)
    E_count_end = tl.load(expert_offset_ptr + Eidx + 1)
    n_tokens = E_count_end - E_count_start

    NUM_I_BLOCKS: tl.constexpr = triton.cdiv(I, BLOCK_I)
    for Iidx in tl.static_range(0, NUM_I_BLOCKS, 1):
        i_offsets = Iidx * BLOCK_I + tl.arange(0, BLOCK_I)
        i_mask = i_offsets < I

        db1_acc = tl.zeros([BLOCK_I], dtype=tl.float32)

        # Process tokens in blocks of BLOCK_TK
        for block_start in tl.range(0, n_tokens, BLOCK_TK):
            # Token offsets within this block
            tk_offsets = block_start + tl.arange(0, BLOCK_TK)
            tk_mask = tk_offsets < n_tokens
            tk_grouped = E_count_start + tk_offsets

            dz_offsets = tk_grouped[:, None] * I + i_offsets[None, :]
            dz_mask = tk_mask[:, None] & i_mask[None, :]
            dz = tl.load(dz_ptr + dz_offsets, mask=dz_mask, other=0.0).to(tl.float32)

            db1_acc += tl.sum(dz, axis=0)  # Sum over BLOCK_TK dimension

        tl.store(db1_ptr + Eidx * I + i_offsets, db1_acc, mask=i_mask)


@triton.jit
def _colsum_smallN_kernel(
    y_ptr,  # *mut  T, shape [M]
    x_ptr,  # *const T, shape [M, N]
    stride_xm: tl.constexpr,
    stride_xn: tl.constexpr,  # strides of X
    stride_y: tl.constexpr,  # stride of Y (usually 1)
    N: tl.constexpr,  # sizes
    BLOCK_N: tl.constexpr,  # tile size along N
):
    row = tl.program_id(0)

    # assume BLOCK_N >= N
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    # Load a tile from the row; cast to fp32 for the reduction
    x = tl.load(x_ptr + row * stride_xm + offs * stride_xn, mask=mask, other=0).to(
        tl.float32
    )
    # Reduce this tile to a scalar and add
    acc = tl.sum(x, axis=0)

    # Store the row-sum (cast back to y dtype)
    tl.store(y_ptr + row * stride_y, acc)


@torch.library.custom_op(
    f"{LIBRARY_NAME}::_up_projection_backward",
    mutates_args={"dw1", "dx_expanded", "db1"},
)
def _up_projection_backward(
    x: torch.Tensor,
    w1: torch.Tensor,
    dx_expanded: torch.Tensor,
    dw1: torch.Tensor,
    dz: torch.Tensor,
    db1: torch.Tensor | None,
    expert_frequency_offset: torch.Tensor,
    expert_schedule_order: torch.Tensor,
    x_gather_idx: torch.Tensor,
    s_scatter_idx: torch.Tensor,
    is_glu_activation: bool,
    stream_id: int,
) -> None:
    I, H, E = w1.size()
    if is_glu_activation:
        I //= 2

    x = x.detach()

    # db1 computation, change it later
    if db1 is not None:
        db1_kernel[(E,)](
            dz, db1, expert_frequency_offset, (2 * I if is_glu_activation else I), E
        )

    mDz_trans = convert_torch_tensor_to_cute_tensor(dz.T, (1, 0), 0, 16, 8)
    mDw1_trans = convert_torch_tensor_to_cute_tensor(
        dw1.permute(1, 0, 2), (2, 1, 0), 0, 16, 8
    )

    mX_trans = convert_torch_tensor_to_cute_tensor(x.T, (1, 0), 0, 16, 8)
    mE_offset = convert_torch_tensor_to_cute_tensor(
        expert_frequency_offset, (0,), 0, 4, 1
    )
    mX_gather = convert_torch_tensor_to_cute_tensor(x_gather_idx, (0,), 0, 4, 1)
    mS_scatter = convert_torch_tensor_to_cute_tensor(s_scatter_idx, (0,), 0, 4, 1)
    mDz = convert_torch_tensor_to_cute_tensor(dz, (0, 1), 1, 16, 8)
    mDx_expanded = convert_torch_tensor_to_cute_tensor(dx_expanded, (0, 1), 1, 16, 8)
    mW1_trans = convert_torch_tensor_to_cute_tensor(
        w1.permute(1, 0, 2), (2, 1, 0), 0, 16, 8
    )

    if expert_schedule_order is None:
        mE_permute_order = None
    else:
        mE_permute_order = convert_torch_tensor_to_cute_tensor(
            expert_schedule_order, (0,), 0, 4, 1
        )
    current_stream = cuda.CUstream(stream_id)

    compile_dw1_key = ("dw1", E, H, I, is_glu_activation, x.dtype)
    if compile_dw1_key not in _up_projection_backward.compile_cache:
        dw1_module = HopperWgmma_MoE_Up_proj_WeightGrad_Bwd(E, H, I, is_glu_activation)
        tensormaps = [
            dw1_module.module.generate_tensormap(None, None, None) for _ in range(1)
        ]
        _up_projection_backward.compile_cache[compile_dw1_key] = cute.compile(
            dw1_module,
            mX_trans,
            mDz_trans,
            mDw1_trans,
            mE_offset,
            mX_gather,
            tensormaps,
            mE_permute_order,
            current_stream,
        )
        _up_projection_backward.compile_cache[f"dw1-{TENSORMAP}"] = tensormaps

    dw1_tensormaps = _up_projection_backward.compile_cache[f"dw1-{TENSORMAP}"]
    _up_projection_backward.compile_cache[compile_dw1_key](
        mX_trans,
        mDz_trans,
        mDw1_trans,
        mE_offset,
        mX_gather,
        dw1_tensormaps,
        mE_permute_order,
        current_stream,
    )

    compile_dx_key = ("dx", E, H, I, is_glu, x.dtype)
    if compile_dx_key not in _up_projection_backward.compile_cache:
        dx_module = HopperWgmma_MoE_Up_proj_ActGrad_Bwd(E, H, I, is_glu_activation)
        tensormaps = [
            dx_module.module.generate_tensormap(None, None, None) for _ in range(2)
        ]
        _up_projection_backward.compile_cache[compile_dx_key] = cute.compile(
            dx_module,
            mDz,
            mW1_trans,
            mDx_expanded,
            mE_offset,
            mX_gather,
            mS_scatter,
            tensormaps,
            mE_permute_order,
            current_stream,
        )
        _up_projection_backward.compile_cache[f"dx-{TENSORMAP}"] = tensormaps

    dx_tensormaps = _up_projection_backward.compile_cache[f"dx-{TENSORMAP}"]
    _up_projection_backward.compile_cache[compile_dx_key](
        mDz,
        mW1_trans,
        mDx_expanded,
        mE_offset,
        mX_gather,
        mS_scatter,
        dx_tensormaps,
        mE_permute_order,
        current_stream,
    )


_up_projection_backward.compile_cache = {}


@torch.library.custom_op(
    f"{LIBRARY_NAME}::_down_projection_backward",
    mutates_args={"dw2", "dz", "ds", "db2"},
)
def _down_projection_backward(
    dout: torch.Tensor,
    z: torch.Tensor,
    w2: torch.Tensor,
    dw2: torch.Tensor,
    dz: torch.Tensor,
    ds: torch.Tensor,
    b2: torch.Tensor | None,
    db2: torch.Tensor | None,
    topk_scores: torch.Tensor,
    expert_frequency_offset: torch.Tensor,
    expert_schedule_order: torch.Tensor,
    x_gather_idx: torch.Tensor,
    s_scatter_idx: torch.Tensor,
    stream_id: int,
    is_glu_activation: bool,
    activation_type: str,
) -> None:
    H, I, E = w2.size()
    TK = x_gather_idx.size(0)

    y1s = torch.empty(TK, I, dtype=z.dtype, device=z.device)

    dout = dout.detach()
    w2 = w2.detach()
    topk_scores = topk_scores.detach()

    mDout = convert_torch_tensor_to_cute_tensor(dout, (0, 1), 1, 16, 8)
    mDout_trans = convert_torch_tensor_to_cute_tensor(dout.T, (1, 0), 0, 16, 8)
    mDw2 = convert_torch_tensor_to_cute_tensor(dw2, (2, 0, 1), 1, 16, 8)
    mW2_trans = convert_torch_tensor_to_cute_tensor(
        w2.permute(1, 0, 2), (2, 1, 0), 0, 16, 8
    )
    mS = convert_torch_tensor_to_cute_tensor(topk_scores, (0,), 0, 4, 1)
    if is_glu_activation:
        mDz_kernel_input = convert_torch_tensor_to_cute_tensor(
            dz.view(torch.float32), (0, 1), 1, 16, 8
        )
        mZ_kernel_input = convert_torch_tensor_to_cute_tensor(
            z.view(torch.float32), (0, 1), 1, 16, 8
        )
    else:
        mDz_kernel_input = convert_torch_tensor_to_cute_tensor(
            dz.detach(), (0, 1), 1, 16, 8
        )
        mZ_kernel_input = convert_torch_tensor_to_cute_tensor(
            z.detach(), (0, 1), 1, 16, 8
        )

    mY1S = convert_torch_tensor_to_cute_tensor(y1s, (0, 1), 1, 16, 8)
    mY1S_trans = convert_torch_tensor_to_cute_tensor(y1s.T, (1, 0), 0, 16, 8)
    mE_offset = convert_torch_tensor_to_cute_tensor(
        expert_frequency_offset, (0,), 0, 4, 1
    )
    mX_gather = convert_torch_tensor_to_cute_tensor(x_gather_idx, (0,), 0, 4, 1)
    mS_scatter = convert_torch_tensor_to_cute_tensor(s_scatter_idx, (0,), 0, 4, 1)

    if expert_schedule_order is None:
        mE_permute_order = None
    else:
        mE_permute_order = convert_torch_tensor_to_cute_tensor(
            expert_schedule_order, (0,), 0, 4, 1
        )
    current_stream = cuda.CUstream(stream_id)
    ds_partial = None

    compile_dz_key = ("dz", E, H, I, z.dtype, activation_type)
    if compile_dz_key not in _down_projection_backward.compile_cache:
        # I don't know why but this sync appears to fix a mysterious initialization bug??
        torch.cuda.synchronize()
        dz_module = HopperWgmma_MoE_Down_proj_ActGrad_Bwd(
            E, H, I, ActivationType(activation_type)
        )
        tensormaps = [
            dz_module.module.generate_tensormap(None, None, None) for _ in range(3)
        ]

        ds_partial_N = max(ceil_divide(I, dz_module.module.tile_shape_mnk[1]), 1)
        ds_partial = torch.empty(
            TK, ds_partial_N, dtype=torch.float32, device=topk_scores.device
        )
        mDS_partial = convert_torch_tensor_to_cute_tensor(ds_partial, (0, 1), 1, 4, 1)

        _down_projection_backward.compile_cache["ds_partial_N"] = ds_partial_N
        _down_projection_backward.compile_cache[compile_dz_key] = cute.compile(
            dz_module,
            mDout,
            mW2_trans,
            mZ_kernel_input,
            mDz_kernel_input,
            mY1S,
            mS,
            mDS_partial,
            mE_offset,
            mX_gather,
            mS_scatter,
            tensormaps,
            mE_permute_order,
            current_stream,
        )
        _down_projection_backward.compile_cache[f"dz-{TENSORMAP}"] = tensormaps

    if ds_partial is None:
        ds_partial_N = _down_projection_backward.compile_cache["ds_partial_N"]
        ds_partial = torch.empty(
            TK, ds_partial_N, dtype=torch.float32, device=topk_scores.device
        )
        mDS_partial = convert_torch_tensor_to_cute_tensor(ds_partial, (0, 1), 1, 4, 1)

    dz_tensormaps = _down_projection_backward.compile_cache[f"dz-{TENSORMAP}"]
    _down_projection_backward.compile_cache[compile_dz_key](
        mDout,
        mW2_trans,
        mZ_kernel_input,
        mDz_kernel_input,
        mY1S,
        mS,
        mDS_partial,
        mE_offset,
        mX_gather,
        mS_scatter,
        dz_tensormaps,
        mE_permute_order,
        current_stream,
    )

    if db2 is None:
        # we don't need to update ds
        if ds_partial.size(1) == 1:
            ds.copy_(ds_partial.view(-1).to(dtype=ds.dtype))
        elif ds_partial.size(1) <= 32:
            ds.copy_(ds_partial.sum(dim=-1, dtype=ds.dtype))
        else:
            M, N = ds_partial.size()

            _colsum_smallN_kernel[M,](
                y_ptr=ds,
                x_ptr=ds_partial,
                stride_xm=ds_partial.stride(0),
                stride_xn=ds_partial.stride(1),
                stride_y=1,
                N=N,
                BLOCK_N=triton.next_power_of_2(N),
            )
    else:
        # db2 and ds update
        BLOCK_H = min(triton.next_power_of_2(H), 2048)
        NUM_H_BLOCKS = triton.cdiv(H, BLOCK_H)

        new_ds_partial = torch.empty(
            TK, NUM_H_BLOCKS, device=ds.device, dtype=torch.float32
        )

        db2_and_ds_kernel[(E, NUM_H_BLOCKS)](
            dout,
            topk_scores,
            new_ds_partial,
            ds_partial,
            b2,
            db2,
            x_gather_idx,
            s_scatter_idx,
            expert_frequency_offset,
            H,
            E,
            ds_partial_N,
            BLOCK_H=BLOCK_H,
            BLOCK_OLD_DS_PARTIAL_N=triton.next_power_of_2(ds_partial_N),
        )

        if NUM_H_BLOCKS == 1:
            ds.copy_(new_ds_partial.view(-1).to(dtype=ds.dtype))
        else:
            ds.copy_(new_ds_partial.sum(dim=-1, dtype=ds.dtype))

    compile_dw2_key = ("dw2", E, H, I, activation_type, dw2.dtype)
    if compile_dw2_key not in _down_projection_backward.compile_cache:
        dw2_module = HopperWgmma_MoE_Down_proj_WeightGrad_Bwd(E, H, I)
        tensormaps = [
            dw2_module.module.generate_tensormap(None, None, None) for _ in range(1)
        ]
        _down_projection_backward.compile_cache[compile_dw2_key] = cute.compile(
            dw2_module,
            mDout_trans,
            mY1S_trans,
            mDw2,
            mE_offset,
            mX_gather,
            tensormaps,
            mE_permute_order,
            current_stream,
        )
        _down_projection_backward.compile_cache[f"dw2-{TENSORMAP}"] = tensormaps

    dw2_tensormaps = _down_projection_backward.compile_cache[f"dw2-{TENSORMAP}"]
    _down_projection_backward.compile_cache[compile_dw2_key](
        mDout_trans,
        mY1S_trans,
        mDw2,
        mE_offset,
        mX_gather,
        dw2_tensormaps,
        mE_permute_order,
        current_stream,
    )


_down_projection_backward.compile_cache = {}


@torch.library.custom_op(
    f"{LIBRARY_NAME}::_token_broadcast_backward", mutates_args={"dx_reduced"}
)
def _token_broadcast_backward(
    dx_reduced: torch.Tensor,
    dx_expanded: torch.Tensor,
    s_reverse_scatter_idx: torch.Tensor,
    num_activated_expert_per_token_offset: torch.Tensor,
    varlen_K_max: int,
    H: int,
    is_varlen_K: bool,
) -> None:
    token_gather_and_sum_varlen_K_triton(
        dx_expanded,
        None,
        dx_reduced,
        s_reverse_scatter_idx,
        num_activated_expert_per_token_offset,
        dx_reduced.size(0),
        varlen_K_max,
        H,
        is_varlen_K,
    )


@triton.jit
def _softmax_bwd_scatter_small_kernel(
    dlogits_ptr,
    dlogits_full_ptr,
    score_ptr,
    dscore_ptr,
    idx_ptr,
    stride_dm: tl.constexpr,
    stride_dn: tl.constexpr,
    stride_sm: tl.constexpr,
    stride_sn: tl.constexpr,
    stride_gm: tl.constexpr,
    stride_gk: tl.constexpr,
    stride_im: tl.constexpr,
    stride_ik: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    dlogits_is_none: tl.constexpr,
):
    row = tl.program_id(axis=0)

    # tl.assume(K <= BLOCK_K)
    k_offs = tl.arange(0, BLOCK_K)
    k_mask = k_offs < K

    idx = tl.load(
        idx_ptr + row * stride_im + k_offs * stride_ik, mask=k_mask, other=0
    ).to(tl.int32)
    s_sel = tl.load(
        score_ptr + row * stride_sm + k_offs * stride_sn, mask=k_mask, other=0
    ).to(tl.float32)
    g_sel = tl.load(
        dscore_ptr + row * stride_gm + k_offs * stride_gk, mask=k_mask, other=0
    ).to(tl.float32)

    # dot = sum_j g_j * y_j over selected columns
    dot = tl.sum(g_sel * s_sel, axis=0)

    # scatter-only: dx[idx] += y_sel * (g_sel - dot)
    add_vals = s_sel * (g_sel - dot)

    indices = row * stride_dm + idx * stride_dn
    if not dlogits_is_none:
        add_vals += tl.load(dlogits_ptr + indices, mask=k_mask)
    tl.store(dlogits_full_ptr + indices, add_vals, mask=k_mask)


@torch.library.custom_op(
    f"{LIBRARY_NAME}::_softmax_topk_bwd", mutates_args={"dlogits_full"}
)
def _softmax_topk_bwd(
    dlogits_full: torch.Tensor,
    dlogits: Optional[torch.Tensor],
    dtopk_score: torch.Tensor,
    topk_router_score: torch.Tensor,
    topk_router_indices: torch.Tensor,
    K: int,
) -> None:
    T = dtopk_score.shape[0]

    _softmax_bwd_scatter_small_kernel[T,](
        dlogits,
        dlogits_full,
        topk_router_score,
        dtopk_score,
        topk_router_indices,
        dlogits_full.stride(0),
        dlogits_full.stride(1),
        topk_router_score.stride(0),
        topk_router_score.stride(1),
        dtopk_score.stride(0),
        dtopk_score.stride(1),
        topk_router_indices.stride(0),
        topk_router_indices.stride(1),
        K,
        triton.next_power_of_2(K),
        (dlogits is None),
    )


@triton.jit
def _topk_bwd_scatter_small_kernel(
    dlogits_full_ptr,
    dscore_ptr,
    idx_ptr,
    stride_dm: tl.constexpr,
    stride_dn: tl.constexpr,
    stride_gm: tl.constexpr,
    stride_gk: tl.constexpr,
    stride_im: tl.constexpr,
    stride_ik: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(axis=0)

    # tl.assume(K <= BLOCK_K)
    k_offs = tl.arange(0, BLOCK_K)
    k_mask = k_offs < K

    idx = tl.load(
        idx_ptr + row * stride_im + k_offs * stride_ik, mask=k_mask, other=0
    ).to(tl.int32)
    g_sel = tl.load(
        dscore_ptr + row * stride_gm + k_offs * stride_gk, mask=k_mask, other=0
    ).to(tl.float32)

    # scatter-only: dx[idx] += y_sel * (g_sel - dot)
    add_vals = g_sel

    indices = row * stride_dm + idx * stride_dn
    tl.store(dlogits_full_ptr + indices, add_vals, mask=k_mask)


@torch.library.custom_op(f"{LIBRARY_NAME}::_topk_bwd", mutates_args={"dlogits_full"})
def _topk_bwd(
    dlogits_full: torch.Tensor,
    dtopk_values: torch.Tensor,
    topk_indices: torch.Tensor,
    K: int,
) -> None:
    T = dtopk_values.shape[0]

    _topk_bwd_scatter_small_kernel[T,](
        dlogits_full,
        dtopk_values,
        topk_indices,
        dlogits_full.stride(0),
        dlogits_full.stride(1),
        dtopk_values.stride(0),
        dtopk_values.stride(1),
        topk_indices.stride(0),
        topk_indices.stride(1),
        K,
        triton.next_power_of_2(K),
    )
