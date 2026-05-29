"""MUSA TileLang causal conv1d forward kernel."""

import functools
from typing import List, Optional, Union

import tilelang
import tilelang.language as T
import torch

from sglang.srt.hardware_backend.musa.jit_kernel.tilelang.utils import (
    MUSA_COMMON_PASS_CONFIGS,
    MUSA_COMPILE_FLAGS,
    storage_window,
    tilelang_dtype,
)
from sglang.srt.layers.attention.mamba.causal_conv1d_triton import PAD_SLOT_ID
from sglang.srt.utils.custom_op import register_custom_op

_LOG2E = 1.4426950408889634

_CAUSAL_CONV1D_PASS_CONFIGS = dict(MUSA_COMMON_PASS_CONFIGS)
for _key, _value in (
    ("TL_ENABLE_LOWER_LDGSTG", True),
    ("TL_ENABLE_LOWER_LDGSTG_PREDICATED", True),
    ("TL_DISABLE_SAFE_COPY_PREDICATION", True),
    ("TL_DISABLE_SAFE_ROBUST_COPY_PREDICATION", True),
    ("TL_CONFIG_INDEX_BITWIDTH", 32),
):
    if hasattr(tilelang.PassConfigKey, _key):
        _CAUSAL_CONV1D_PASS_CONFIGS[getattr(tilelang.PassConfigKey, _key)] = _value


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


@functools.lru_cache(maxsize=64)
@tilelang.jit(
    out_idx=[],
    target="musa",
    pass_configs=_CAUSAL_CONV1D_PASS_CONFIGS,
    compile_flags=MUSA_COMPILE_FLAGS,
)
def _causal_conv1d_fwd_kernel(
    dtype: str,
    width: int,
    x_stride_dim: int,
    x_stride_token: int,
    w_stride_dim: int,
    w_stride_width: int,
    state_stride_seq: int,
    state_stride_dim: int,
    state_stride_token: int,
    o_stride_dim: int,
    o_stride_token: int,
    has_bias: bool,
    has_conv_states: bool,
    has_cache_indices: bool,
    has_initial_states: bool,
    use_pad_slot: bool,
    silu_activation: bool,
    block_m: int,
    block_n: int,
):
    x_numel = T.dynamic("x_numel")
    w_numel = T.dynamic("w_numel")
    bias_numel = T.dynamic("bias_numel")
    state_numel = T.dynamic("state_numel")
    cache_numel = T.dynamic("cache_numel")
    init_numel = T.dynamic("init_numel")
    query_numel = T.dynamic("query_numel")
    out_numel = T.dynamic("out_numel")
    state_len = width - 1

    @T.prim_func
    def sglang_musa_causal_conv1d_fwd(
        x: T.Tensor((x_numel,), dtype),
        weight: T.Tensor((w_numel,), dtype),
        bias: T.Tensor((bias_numel,), dtype),
        conv_states: T.Tensor((state_numel,), dtype),
        cache_indices: T.Tensor((cache_numel,), "int32"),
        has_initial_state: T.Tensor((init_numel,), "bool"),
        query_start_loc: T.Tensor((query_numel,), "int32"),
        out: T.Tensor((out_numel,), dtype),
        max_seq_len: T.int32,
        dim: T.int32,
        num_cache_lines: T.int32,
        pad_slot_id: T.int32,
    ):
        with T.Kernel(
            query_numel - 1,
            T.ceildiv(max_seq_len, block_m),
            T.ceildiv(dim, block_n),
            threads=block_n,
        ) as (seq_idx, chunk_idx, dim_block):
            tid = T.get_thread_binding()
            feat = dim_block * block_n + tid
            seq_start = T.alloc_var("int32")
            seq_end = T.alloc_var("int32")
            seq_len = T.alloc_var("int32")
            token_offset = T.alloc_var("int32")
            segment_len = T.alloc_var("int32")
            cache_idx = T.alloc_var("int32")
            load_init = T.alloc_var("bool")
            state_base = T.alloc_var("int32")
            x_base = T.alloc_var("int32")
            w_base = T.alloc_var("int32")
            out_base = T.alloc_var("int32")
            valid_seq = T.alloc_var("bool")
            col0 = T.alloc_var("float32")
            col1 = T.alloc_var("float32")
            col2 = T.alloc_var("float32")
            col3 = T.alloc_var("float32")
            w0 = T.alloc_var("float32")
            w1 = T.alloc_var("float32")
            w2 = T.alloc_var("float32")
            w3 = T.alloc_var("float32")
            w4 = T.alloc_var("float32")
            x_cur = T.alloc_var("float32")
            acc = T.alloc_var("float32")
            state_src = T.alloc_var("int32")
            state_cut = T.alloc_var("int32")

            seq_start = query_start_loc[seq_idx]
            seq_end = query_start_loc[seq_idx + 1]
            seq_len = seq_end - seq_start
            token_offset = chunk_idx * block_m
            segment_len = seq_len - token_offset
            if segment_len > block_m:
                segment_len = block_m

            cache_idx = seq_idx
            if has_cache_indices:
                cache_idx = cache_indices[seq_idx]
            valid_seq = segment_len > 0
            if use_pad_slot and cache_idx == pad_slot_id:
                valid_seq = False
            if has_conv_states and cache_idx >= num_cache_lines:
                valid_seq = False

            if valid_seq and feat < dim:
                x_base = seq_start * x_stride_token + feat * x_stride_dim
                w_base = feat * w_stride_dim
                out_base = seq_start * o_stride_token + feat * o_stride_dim
                state_base = cache_idx * state_stride_seq + feat * state_stride_dim

                col0 = 0.0
                col1 = 0.0
                col2 = 0.0
                col3 = 0.0
                load_init = False
                if has_initial_states:
                    load_init = has_initial_state[seq_idx]

                if chunk_idx == 0:
                    if has_conv_states and load_init:
                        if width >= 2:
                            col0 = T.Cast(
                                "float32",
                                conv_states[
                                    state_base + (state_len - 1) * state_stride_token
                                ],
                            )
                        if width >= 3:
                            col1 = col0
                            col0 = T.Cast(
                                "float32",
                                conv_states[
                                    state_base + (state_len - 2) * state_stride_token
                                ],
                            )
                        if width >= 4:
                            col2 = col1
                            col1 = T.Cast(
                                "float32",
                                conv_states[
                                    state_base + (state_len - 2) * state_stride_token
                                ],
                            )
                            col0 = T.Cast(
                                "float32",
                                conv_states[
                                    state_base + (state_len - 3) * state_stride_token
                                ],
                            )
                        if width >= 5:
                            col3 = col2
                            col2 = T.Cast(
                                "float32",
                                conv_states[
                                    state_base + (state_len - 2) * state_stride_token
                                ],
                            )
                            col1 = T.Cast(
                                "float32",
                                conv_states[
                                    state_base + (state_len - 3) * state_stride_token
                                ],
                            )
                            col0 = T.Cast(
                                "float32",
                                conv_states[
                                    state_base + (state_len - 4) * state_stride_token
                                ],
                            )

                    if has_conv_states:
                        state_cut = state_len - seq_len
                        for state_i in T.serial(state_len):
                            if state_len <= seq_len:
                                conv_states[
                                    state_base + state_i * state_stride_token
                                ] = x[
                                    x_base
                                    + (seq_len - state_len + state_i) * x_stride_token
                                ]
                            else:
                                if load_init and state_i < state_cut:
                                    state_src = state_i + seq_len
                                    conv_states[
                                        state_base + state_i * state_stride_token
                                    ] = conv_states[
                                        state_base + state_src * state_stride_token
                                    ]
                                elif state_i >= state_cut:
                                    conv_states[
                                        state_base + state_i * state_stride_token
                                    ] = x[
                                        x_base + (state_i - state_cut) * x_stride_token
                                    ]
                                else:
                                    conv_states[
                                        state_base + state_i * state_stride_token
                                    ] = T.Cast(dtype, 0.0)
                else:
                    if width >= 2:
                        col0 = T.Cast(
                            "float32",
                            x[x_base + (token_offset - 1) * x_stride_token],
                        )
                    if width >= 3:
                        col1 = col0
                        col0 = T.Cast(
                            "float32",
                            x[x_base + (token_offset - 2) * x_stride_token],
                        )
                    if width >= 4:
                        col2 = col1
                        col1 = T.Cast(
                            "float32",
                            x[x_base + (token_offset - 2) * x_stride_token],
                        )
                        col0 = T.Cast(
                            "float32",
                            x[x_base + (token_offset - 3) * x_stride_token],
                        )
                    if width >= 5:
                        col3 = col2
                        col2 = T.Cast(
                            "float32",
                            x[x_base + (token_offset - 2) * x_stride_token],
                        )
                        col1 = T.Cast(
                            "float32",
                            x[x_base + (token_offset - 3) * x_stride_token],
                        )
                        col0 = T.Cast(
                            "float32",
                            x[x_base + (token_offset - 4) * x_stride_token],
                        )

                w0 = T.Cast("float32", weight[w_base])
                w1 = T.Cast("float32", weight[w_base + w_stride_width])
                if width >= 3:
                    w2 = T.Cast("float32", weight[w_base + 2 * w_stride_width])
                if width >= 4:
                    w3 = T.Cast("float32", weight[w_base + 3 * w_stride_width])
                if width >= 5:
                    w4 = T.Cast("float32", weight[w_base + 4 * w_stride_width])

                for token_i in T.serial(block_m):
                    if token_i < segment_len:
                        x_cur = T.Cast(
                            "float32",
                            x[x_base + (token_offset + token_i) * x_stride_token],
                        )
                        acc = 0.0
                        if has_bias:
                            acc = T.Cast("float32", bias[feat])

                        if width == 2:
                            acc += col0 * w0 + x_cur * w1
                            col0 = x_cur
                        elif width == 3:
                            acc += col0 * w0 + col1 * w1 + x_cur * w2
                            col0 = col1
                            col1 = x_cur
                        elif width == 4:
                            acc += col0 * w0 + col1 * w1 + col2 * w2 + x_cur * w3
                            col0 = col1
                            col1 = col2
                            col2 = x_cur
                        else:
                            acc += (
                                col0 * w0
                                + col1 * w1
                                + col2 * w2
                                + col3 * w3
                                + x_cur * w4
                            )
                            col0 = col1
                            col1 = col2
                            col2 = col3
                            col3 = x_cur

                        if silu_activation:
                            acc = acc / (1.0 + T.exp2(-acc * _LOG2E))
                        out[out_base + (token_offset + token_i) * o_stride_token] = (
                            T.Cast(dtype, acc)
                        )

    return sglang_musa_causal_conv1d_fwd


_causal_conv1d_fwd_kernel.mode = "lazy"


@functools.lru_cache(maxsize=64)
@tilelang.jit(
    out_idx=[],
    target="musa",
    pass_configs=_CAUSAL_CONV1D_PASS_CONFIGS,
    compile_flags=MUSA_COMPILE_FLAGS,
)
def _causal_conv1d_fwd_3d_kernel(
    dtype: str,
    width: int,
    x_stride_batch: int,
    x_stride_dim: int,
    x_stride_token: int,
    w_stride_dim: int,
    w_stride_width: int,
    state_stride_seq: int,
    state_stride_dim: int,
    state_stride_token: int,
    o_stride_batch: int,
    o_stride_dim: int,
    o_stride_token: int,
    has_bias: bool,
    has_conv_states: bool,
    has_cache_indices: bool,
    has_initial_states: bool,
    use_pad_slot: bool,
    silu_activation: bool,
    block_m: int,
    block_n: int,
):
    x_numel = T.dynamic("x_numel")
    w_numel = T.dynamic("w_numel")
    bias_numel = T.dynamic("bias_numel")
    state_numel = T.dynamic("state_numel")
    cache_numel = T.dynamic("cache_numel")
    init_numel = T.dynamic("init_numel")
    out_numel = T.dynamic("out_numel")
    state_len = width - 1

    @T.prim_func
    def sglang_musa_causal_conv1d_fwd_3d(
        x: T.Tensor((x_numel,), dtype),
        weight: T.Tensor((w_numel,), dtype),
        bias: T.Tensor((bias_numel,), dtype),
        conv_states: T.Tensor((state_numel,), dtype),
        cache_indices: T.Tensor((cache_numel,), "int32"),
        has_initial_state: T.Tensor((init_numel,), "bool"),
        out: T.Tensor((out_numel,), dtype),
        batch_size: T.int32,
        dim: T.int32,
        seq_len: T.int32,
        num_cache_lines: T.int32,
        pad_slot_id: T.int32,
    ):
        with T.Kernel(
            batch_size,
            T.ceildiv(seq_len, block_m),
            T.ceildiv(dim, block_n),
            threads=block_n,
        ) as (batch_idx, chunk_idx, dim_block):
            tid = T.get_thread_binding()
            feat = dim_block * block_n + tid
            token_offset = T.alloc_var("int32")
            segment_len = T.alloc_var("int32")
            cache_idx = T.alloc_var("int32")
            load_init = T.alloc_var("bool")
            state_base = T.alloc_var("int32")
            x_base = T.alloc_var("int32")
            w_base = T.alloc_var("int32")
            out_base = T.alloc_var("int32")
            valid_seq = T.alloc_var("bool")
            col0 = T.alloc_var("float32")
            col1 = T.alloc_var("float32")
            col2 = T.alloc_var("float32")
            col3 = T.alloc_var("float32")
            w0 = T.alloc_var("float32")
            w1 = T.alloc_var("float32")
            w2 = T.alloc_var("float32")
            w3 = T.alloc_var("float32")
            w4 = T.alloc_var("float32")
            x_cur = T.alloc_var("float32")
            acc = T.alloc_var("float32")
            state_src = T.alloc_var("int32")
            state_cut = T.alloc_var("int32")

            token_offset = chunk_idx * block_m
            segment_len = seq_len - token_offset
            if segment_len > block_m:
                segment_len = block_m

            cache_idx = batch_idx
            if has_cache_indices:
                cache_idx = cache_indices[batch_idx]
            valid_seq = segment_len > 0
            if use_pad_slot and cache_idx == pad_slot_id:
                valid_seq = False
            if has_conv_states and cache_idx >= num_cache_lines:
                valid_seq = False

            if valid_seq and feat < dim:
                x_base = batch_idx * x_stride_batch + feat * x_stride_dim
                w_base = feat * w_stride_dim
                out_base = batch_idx * o_stride_batch + feat * o_stride_dim
                state_base = cache_idx * state_stride_seq + feat * state_stride_dim

                col0 = 0.0
                col1 = 0.0
                col2 = 0.0
                col3 = 0.0
                load_init = False
                if has_initial_states:
                    load_init = has_initial_state[batch_idx]

                if chunk_idx == 0:
                    if has_conv_states and load_init:
                        if width >= 2:
                            col0 = T.Cast(
                                "float32",
                                conv_states[
                                    state_base + (state_len - 1) * state_stride_token
                                ],
                            )
                        if width >= 3:
                            col1 = col0
                            col0 = T.Cast(
                                "float32",
                                conv_states[
                                    state_base + (state_len - 2) * state_stride_token
                                ],
                            )
                        if width >= 4:
                            col2 = col1
                            col1 = T.Cast(
                                "float32",
                                conv_states[
                                    state_base + (state_len - 2) * state_stride_token
                                ],
                            )
                            col0 = T.Cast(
                                "float32",
                                conv_states[
                                    state_base + (state_len - 3) * state_stride_token
                                ],
                            )
                        if width >= 5:
                            col3 = col2
                            col2 = T.Cast(
                                "float32",
                                conv_states[
                                    state_base + (state_len - 2) * state_stride_token
                                ],
                            )
                            col1 = T.Cast(
                                "float32",
                                conv_states[
                                    state_base + (state_len - 3) * state_stride_token
                                ],
                            )
                            col0 = T.Cast(
                                "float32",
                                conv_states[
                                    state_base + (state_len - 4) * state_stride_token
                                ],
                            )

                    if has_conv_states:
                        state_cut = state_len - seq_len
                        for state_i in T.serial(state_len):
                            if state_len <= seq_len:
                                conv_states[
                                    state_base + state_i * state_stride_token
                                ] = x[
                                    x_base
                                    + (seq_len - state_len + state_i) * x_stride_token
                                ]
                            else:
                                if load_init and state_i < state_cut:
                                    state_src = state_i + seq_len
                                    conv_states[
                                        state_base + state_i * state_stride_token
                                    ] = conv_states[
                                        state_base + state_src * state_stride_token
                                    ]
                                elif state_i >= state_cut:
                                    conv_states[
                                        state_base + state_i * state_stride_token
                                    ] = x[
                                        x_base + (state_i - state_cut) * x_stride_token
                                    ]
                                else:
                                    conv_states[
                                        state_base + state_i * state_stride_token
                                    ] = T.Cast(dtype, 0.0)
                else:
                    if width >= 2:
                        col0 = T.Cast(
                            "float32",
                            x[x_base + (token_offset - 1) * x_stride_token],
                        )
                    if width >= 3:
                        col1 = col0
                        col0 = T.Cast(
                            "float32",
                            x[x_base + (token_offset - 2) * x_stride_token],
                        )
                    if width >= 4:
                        col2 = col1
                        col1 = T.Cast(
                            "float32",
                            x[x_base + (token_offset - 2) * x_stride_token],
                        )
                        col0 = T.Cast(
                            "float32",
                            x[x_base + (token_offset - 3) * x_stride_token],
                        )
                    if width >= 5:
                        col3 = col2
                        col2 = T.Cast(
                            "float32",
                            x[x_base + (token_offset - 2) * x_stride_token],
                        )
                        col1 = T.Cast(
                            "float32",
                            x[x_base + (token_offset - 3) * x_stride_token],
                        )
                        col0 = T.Cast(
                            "float32",
                            x[x_base + (token_offset - 4) * x_stride_token],
                        )

                w0 = T.Cast("float32", weight[w_base])
                w1 = T.Cast("float32", weight[w_base + w_stride_width])
                if width >= 3:
                    w2 = T.Cast("float32", weight[w_base + 2 * w_stride_width])
                if width >= 4:
                    w3 = T.Cast("float32", weight[w_base + 3 * w_stride_width])
                if width >= 5:
                    w4 = T.Cast("float32", weight[w_base + 4 * w_stride_width])

                for token_i in T.serial(block_m):
                    if token_i < segment_len:
                        x_cur = T.Cast(
                            "float32",
                            x[x_base + (token_offset + token_i) * x_stride_token],
                        )
                        acc = 0.0
                        if has_bias:
                            acc = T.Cast("float32", bias[feat])

                        if width == 2:
                            acc += col0 * w0 + x_cur * w1
                            col0 = x_cur
                        elif width == 3:
                            acc += col0 * w0 + col1 * w1 + x_cur * w2
                            col0 = col1
                            col1 = x_cur
                        elif width == 4:
                            acc += col0 * w0 + col1 * w1 + col2 * w2 + x_cur * w3
                            col0 = col1
                            col1 = col2
                            col2 = x_cur
                        else:
                            acc += (
                                col0 * w0
                                + col1 * w1
                                + col2 * w2
                                + col3 * w3
                                + x_cur * w4
                            )
                            col0 = col1
                            col1 = col2
                            col2 = col3
                            col3 = x_cur

                        if silu_activation:
                            acc = acc / (1.0 + T.exp2(-acc * _LOG2E))
                        out[out_base + (token_offset + token_i) * o_stride_token] = (
                            T.Cast(dtype, acc)
                        )

    return sglang_musa_causal_conv1d_fwd_3d


_causal_conv1d_fwd_3d_kernel.mode = "lazy"


@functools.lru_cache(maxsize=64)
@tilelang.jit(
    out_idx=[],
    target="musa",
    pass_configs=_CAUSAL_CONV1D_PASS_CONFIGS,
    compile_flags=MUSA_COMPILE_FLAGS,
)
def _causal_conv1d_fwd_width4_vec_kernel(
    dtype: str,
    x_stride_token: int,
    w_stride_dim: int,
    state_stride_seq: int,
    state_stride_dim: int,
    state_stride_token: int,
    o_stride_token: int,
    has_bias: bool,
    has_conv_states: bool,
    has_cache_indices: bool,
    has_initial_states: bool,
    use_pad_slot: bool,
    silu_activation: bool,
    block_m: int,
    block_feats: int,
    vec_elems: int,
):
    x_numel = T.dynamic("x_numel")
    w_numel = T.dynamic("w_numel")
    bias_numel = T.dynamic("bias_numel")
    state_numel = T.dynamic("state_numel")
    cache_numel = T.dynamic("cache_numel")
    init_numel = T.dynamic("init_numel")
    query_numel = T.dynamic("query_numel")
    out_numel = T.dynamic("out_numel")
    num_threads = block_feats // vec_elems
    state_len = 3

    @T.prim_func
    def sglang_musa_causal_conv1d_fwd_width4_vec(
        x: T.Tensor((x_numel,), dtype),
        weight: T.Tensor((w_numel,), dtype),
        bias: T.Tensor((bias_numel,), dtype),
        conv_states: T.Tensor((state_numel,), dtype),
        cache_indices: T.Tensor((cache_numel,), "int32"),
        has_initial_state: T.Tensor((init_numel,), "bool"),
        query_start_loc: T.Tensor((query_numel,), "int32"),
        out: T.Tensor((out_numel,), dtype),
        max_seq_len: T.int32,
        dim: T.int32,
        num_cache_lines: T.int32,
        pad_slot_id: T.int32,
    ):
        with T.Kernel(
            query_numel - 1,
            T.ceildiv(max_seq_len, block_m),
            T.ceildiv(dim, block_feats),
            threads=num_threads,
        ) as (seq_idx, chunk_idx, dim_block):
            tid = T.get_thread_binding()
            feat_base = dim_block * block_feats + tid * vec_elems
            seq_start = T.alloc_var("int32")
            seq_end = T.alloc_var("int32")
            seq_len = T.alloc_var("int32")
            token_offset = T.alloc_var("int32")
            segment_len = T.alloc_var("int32")
            cache_idx = T.alloc_var("int32")
            load_init = T.alloc_var("bool")
            valid_seq = T.alloc_var("bool")
            state_cut = T.alloc_var("int32")
            state_base = T.alloc_local((vec_elems,), "int32")
            x_base = T.alloc_local((vec_elems,), "int32")
            w_base = T.alloc_local((vec_elems,), "int32")
            out_base = T.alloc_local((vec_elems,), "int32")
            col0 = T.alloc_local((vec_elems,), "float32")
            col1 = T.alloc_local((vec_elems,), "float32")
            col2 = T.alloc_local((vec_elems,), "float32")
            w0 = T.alloc_local((vec_elems,), "float32")
            w1 = T.alloc_local((vec_elems,), "float32")
            w2 = T.alloc_local((vec_elems,), "float32")
            w3 = T.alloc_local((vec_elems,), "float32")
            x_cur = T.alloc_local((vec_elems,), "float32")
            acc = T.alloc_local((vec_elems,), "float32")

            seq_start = query_start_loc[seq_idx]
            seq_end = query_start_loc[seq_idx + 1]
            seq_len = seq_end - seq_start
            token_offset = chunk_idx * block_m
            segment_len = seq_len - token_offset
            if segment_len > block_m:
                segment_len = block_m

            cache_idx = seq_idx
            if has_cache_indices:
                cache_idx = cache_indices[seq_idx]
            valid_seq = segment_len > 0
            if use_pad_slot and cache_idx == pad_slot_id:
                valid_seq = False
            if has_conv_states and cache_idx >= num_cache_lines:
                valid_seq = False
            load_init = False
            if has_initial_states:
                load_init = has_initial_state[seq_idx]

            if valid_seq:
                for v in T.vectorized(vec_elems):
                    feat = feat_base + v
                    col0[v] = 0.0
                    col1[v] = 0.0
                    col2[v] = 0.0
                    x_base[v] = seq_start * x_stride_token + feat
                    w_base[v] = feat * w_stride_dim
                    out_base[v] = seq_start * o_stride_token + feat
                    state_base[v] = (
                        cache_idx * state_stride_seq + feat * state_stride_dim
                    )
                    if feat < dim:
                        if chunk_idx == 0:
                            if has_conv_states and load_init:
                                col2[v] = T.Cast(
                                    "float32",
                                    conv_states[state_base[v] + 2 * state_stride_token],
                                )
                                col1[v] = T.Cast(
                                    "float32",
                                    conv_states[state_base[v] + state_stride_token],
                                )
                                col0[v] = T.Cast("float32", conv_states[state_base[v]])
                        else:
                            col2[v] = T.Cast(
                                "float32",
                                x[x_base[v] + (token_offset - 1) * x_stride_token],
                            )
                            col1[v] = T.Cast(
                                "float32",
                                x[x_base[v] + (token_offset - 2) * x_stride_token],
                            )
                            col0[v] = T.Cast(
                                "float32",
                                x[x_base[v] + (token_offset - 3) * x_stride_token],
                            )

                        w0[v] = T.Cast("float32", weight[w_base[v]])
                        w1[v] = T.Cast("float32", weight[w_base[v] + 1])
                        w2[v] = T.Cast("float32", weight[w_base[v] + 2])
                        w3[v] = T.Cast("float32", weight[w_base[v] + 3])

                if chunk_idx == 0 and has_conv_states:
                    state_cut = state_len - seq_len
                    for state_i in T.serial(state_len):
                        for v in T.vectorized(vec_elems):
                            feat = feat_base + v
                            if feat < dim:
                                if seq_len >= state_len:
                                    conv_states[
                                        state_base[v] + state_i * state_stride_token
                                    ] = x[
                                        x_base[v]
                                        + (seq_len - state_len + state_i)
                                        * x_stride_token
                                    ]
                                else:
                                    if load_init and state_i < state_cut:
                                        conv_states[
                                            state_base[v] + state_i * state_stride_token
                                        ] = conv_states[
                                            state_base[v]
                                            + (state_i + seq_len) * state_stride_token
                                        ]
                                    elif state_i >= state_cut:
                                        conv_states[
                                            state_base[v] + state_i * state_stride_token
                                        ] = x[
                                            x_base[v]
                                            + (state_i - state_cut) * x_stride_token
                                        ]
                                    else:
                                        conv_states[
                                            state_base[v] + state_i * state_stride_token
                                        ] = T.Cast(dtype, 0.0)

                for token_i in T.serial(block_m):
                    if token_i < segment_len:
                        for v in T.vectorized(vec_elems):
                            feat = feat_base + v
                            if feat < dim:
                                x_cur[v] = T.Cast(
                                    "float32",
                                    x[
                                        x_base[v]
                                        + (token_offset + token_i) * x_stride_token
                                    ],
                                )
                                acc[v] = 0.0
                                if has_bias:
                                    acc[v] = T.Cast("float32", bias[feat])
                                acc[v] += (
                                    col0[v] * w0[v]
                                    + col1[v] * w1[v]
                                    + col2[v] * w2[v]
                                    + x_cur[v] * w3[v]
                                )
                                col0[v] = col1[v]
                                col1[v] = col2[v]
                                col2[v] = x_cur[v]
                                if silu_activation:
                                    acc[v] = acc[v] / (1.0 + T.exp2(-acc[v] * _LOG2E))
                                out[
                                    out_base[v]
                                    + (token_offset + token_i) * o_stride_token
                                ] = T.Cast(dtype, acc[v])

    return sglang_musa_causal_conv1d_fwd_width4_vec


_causal_conv1d_fwd_width4_vec_kernel.mode = "lazy"


@functools.lru_cache(maxsize=64)
@tilelang.jit(
    out_idx=[],
    target="musa",
    pass_configs=_CAUSAL_CONV1D_PASS_CONFIGS,
    compile_flags=MUSA_COMPILE_FLAGS,
)
def _causal_conv1d_prefill_width4_kernel(
    dtype: str,
    x_stride_token: int,
    w_stride_dim: int,
    state_stride_seq: int,
    state_stride_dim: int,
    state_stride_token: int,
    o_stride_token: int,
    has_bias: bool,
    has_cache_indices: bool,
    has_initial_states: bool,
    use_pad_slot: bool,
    silu_activation: bool,
    block_m: int,
    block_n: int,
):
    x_numel = T.dynamic("x_numel")
    w_numel = T.dynamic("w_numel")
    bias_numel = T.dynamic("bias_numel")
    state_numel = T.dynamic("state_numel")
    cache_numel = T.dynamic("cache_numel")
    init_numel = T.dynamic("init_numel")
    query_numel = T.dynamic("query_numel")
    out_numel = T.dynamic("out_numel")

    @T.prim_func
    def sglang_musa_causal_conv1d_prefill_width4(
        x: T.Tensor((x_numel,), dtype),
        weight: T.Tensor((w_numel,), dtype),
        bias: T.Tensor((bias_numel,), dtype),
        conv_states: T.Tensor((state_numel,), dtype),
        cache_indices: T.Tensor((cache_numel,), "int32"),
        has_initial_state: T.Tensor((init_numel,), "bool"),
        query_start_loc: T.Tensor((query_numel,), "int32"),
        out: T.Tensor((out_numel,), dtype),
        max_seq_len: T.int32,
        dim: T.int32,
        num_cache_lines: T.int32,
        pad_slot_id: T.int32,
    ):
        with T.Kernel(
            query_numel - 1,
            T.ceildiv(max_seq_len, block_m),
            T.ceildiv(dim, block_n),
            threads=block_n,
        ) as (seq_idx, chunk_idx, dim_block):
            tid = T.get_thread_binding()
            feat = dim_block * block_n + tid
            seq_start = T.alloc_var("int32")
            seq_end = T.alloc_var("int32")
            seq_len = T.alloc_var("int32")
            token_offset = T.alloc_var("int32")
            segment_len = T.alloc_var("int32")
            cache_idx = T.alloc_var("int32")
            valid_seq = T.alloc_var("bool")
            load_init = T.alloc_var("bool")
            x_base = T.alloc_var("int32")
            w_base = T.alloc_var("int32")
            state_base = T.alloc_var("int32")
            out_base = T.alloc_var("int32")
            col0 = T.alloc_var("float32")
            col1 = T.alloc_var("float32")
            col2 = T.alloc_var("float32")
            w0 = T.alloc_var("float32")
            w1 = T.alloc_var("float32")
            w2 = T.alloc_var("float32")
            w3 = T.alloc_var("float32")
            x_cur = T.alloc_var("float32")
            acc = T.alloc_var("float32")

            seq_start = query_start_loc[seq_idx]
            seq_end = query_start_loc[seq_idx + 1]
            seq_len = seq_end - seq_start
            token_offset = chunk_idx * block_m
            segment_len = seq_len - token_offset
            if segment_len > block_m:
                segment_len = block_m

            cache_idx = seq_idx
            if has_cache_indices:
                cache_idx = cache_indices[seq_idx]
            valid_seq = segment_len > 0 and feat < dim
            if use_pad_slot and cache_idx == pad_slot_id:
                valid_seq = False
            if cache_idx >= num_cache_lines:
                valid_seq = False

            if valid_seq:
                x_base = seq_start * x_stride_token + feat
                w_base = feat * w_stride_dim
                out_base = seq_start * o_stride_token + feat
                state_base = cache_idx * state_stride_seq + feat * state_stride_dim

                col0 = 0.0
                col1 = 0.0
                col2 = 0.0
                load_init = False
                if has_initial_states:
                    load_init = has_initial_state[seq_idx]

                if chunk_idx == 0:
                    if load_init:
                        col0 = T.Cast("float32", conv_states[state_base])
                        col1 = T.Cast(
                            "float32",
                            conv_states[state_base + state_stride_token],
                        )
                        col2 = T.Cast(
                            "float32",
                            conv_states[state_base + 2 * state_stride_token],
                        )

                    if seq_len >= 3:
                        conv_states[state_base] = x[
                            x_base + (seq_len - 3) * x_stride_token
                        ]
                        conv_states[state_base + state_stride_token] = x[
                            x_base + (seq_len - 2) * x_stride_token
                        ]
                        conv_states[state_base + 2 * state_stride_token] = x[
                            x_base + (seq_len - 1) * x_stride_token
                        ]
                    else:
                        if seq_len == 1:
                            if load_init:
                                conv_states[state_base] = conv_states[
                                    state_base + state_stride_token
                                ]
                                conv_states[state_base + state_stride_token] = (
                                    conv_states[state_base + 2 * state_stride_token]
                                )
                            else:
                                conv_states[state_base] = T.Cast(dtype, 0.0)
                                conv_states[state_base + state_stride_token] = T.Cast(
                                    dtype, 0.0
                                )
                            conv_states[state_base + 2 * state_stride_token] = x[x_base]
                        else:
                            if load_init:
                                conv_states[state_base] = conv_states[
                                    state_base + 2 * state_stride_token
                                ]
                            else:
                                conv_states[state_base] = T.Cast(dtype, 0.0)
                            conv_states[state_base + state_stride_token] = x[x_base]
                            conv_states[state_base + 2 * state_stride_token] = x[
                                x_base + x_stride_token
                            ]
                else:
                    col2 = T.Cast(
                        "float32",
                        x[x_base + (token_offset - 1) * x_stride_token],
                    )
                    col1 = T.Cast(
                        "float32",
                        x[x_base + (token_offset - 2) * x_stride_token],
                    )
                    col0 = T.Cast(
                        "float32",
                        x[x_base + (token_offset - 3) * x_stride_token],
                    )

                w0 = T.Cast("float32", weight[w_base])
                w1 = T.Cast("float32", weight[w_base + 1])
                w2 = T.Cast("float32", weight[w_base + 2])
                w3 = T.Cast("float32", weight[w_base + 3])

                for token_i in T.serial(block_m):
                    if token_i < segment_len:
                        x_cur = T.Cast(
                            "float32",
                            x[x_base + (token_offset + token_i) * x_stride_token],
                        )
                        acc = 0.0
                        if has_bias:
                            acc = T.Cast("float32", bias[feat])
                        acc += col0 * w0 + col1 * w1 + col2 * w2 + x_cur * w3
                        col0 = col1
                        col1 = col2
                        col2 = x_cur
                        if silu_activation:
                            acc = acc / (1.0 + T.exp2(-acc * _LOG2E))
                        out[out_base + (token_offset + token_i) * o_stride_token] = (
                            T.Cast(dtype, acc)
                        )

    return sglang_musa_causal_conv1d_prefill_width4


_causal_conv1d_prefill_width4_kernel.mode = "lazy"


@functools.lru_cache(maxsize=64)
@tilelang.jit(
    out_idx=[],
    target="musa",
    pass_configs=_CAUSAL_CONV1D_PASS_CONFIGS,
    compile_flags=MUSA_COMPILE_FLAGS,
)
def _causal_conv1d_prefill_width4_body_kernel(
    dtype: str,
    x_stride_token: int,
    w_stride_dim: int,
    o_stride_token: int,
    has_bias: bool,
    has_cache_indices: bool,
    use_pad_slot: bool,
    silu_activation: bool,
    block_m: int,
    block_n: int,
):
    x_numel = T.dynamic("x_numel")
    w_numel = T.dynamic("w_numel")
    bias_numel = T.dynamic("bias_numel")
    cache_numel = T.dynamic("cache_numel")
    query_numel = T.dynamic("query_numel")
    out_numel = T.dynamic("out_numel")

    @T.prim_func
    def sglang_musa_causal_conv1d_prefill_width4_body(
        x: T.Tensor((x_numel,), dtype),
        weight: T.Tensor((w_numel,), dtype),
        bias: T.Tensor((bias_numel,), dtype),
        cache_indices: T.Tensor((cache_numel,), "int32"),
        query_start_loc: T.Tensor((query_numel,), "int32"),
        out: T.Tensor((out_numel,), dtype),
        max_seq_len: T.int32,
        dim: T.int32,
        num_cache_lines: T.int32,
        pad_slot_id: T.int32,
    ):
        with T.Kernel(
            query_numel - 1,
            T.ceildiv(max_seq_len - block_m, block_m),
            T.ceildiv(dim, block_n),
            threads=block_n,
        ) as (seq_idx, body_chunk_idx, dim_block):
            tid = T.get_thread_binding()
            feat = dim_block * block_n + tid
            seq_start = T.alloc_var("int32")
            seq_end = T.alloc_var("int32")
            seq_len = T.alloc_var("int32")
            token_offset = T.alloc_var("int32")
            segment_len = T.alloc_var("int32")
            cache_idx = T.alloc_var("int32")
            valid_seq = T.alloc_var("bool")
            x_base = T.alloc_var("int32")
            w_base = T.alloc_var("int32")
            out_base = T.alloc_var("int32")
            col0 = T.alloc_var("float32")
            col1 = T.alloc_var("float32")
            col2 = T.alloc_var("float32")
            w0 = T.alloc_var("float32")
            w1 = T.alloc_var("float32")
            w2 = T.alloc_var("float32")
            w3 = T.alloc_var("float32")
            x_cur = T.alloc_var("float32")
            acc = T.alloc_var("float32")

            seq_start = query_start_loc[seq_idx]
            seq_end = query_start_loc[seq_idx + 1]
            seq_len = seq_end - seq_start
            token_offset = (body_chunk_idx + 1) * block_m
            segment_len = seq_len - token_offset
            if segment_len > block_m:
                segment_len = block_m

            cache_idx = seq_idx
            if has_cache_indices:
                cache_idx = cache_indices[seq_idx]
            valid_seq = segment_len > 0 and feat < dim
            if use_pad_slot and cache_idx == pad_slot_id:
                valid_seq = False
            if cache_idx >= num_cache_lines:
                valid_seq = False

            if valid_seq:
                x_base = seq_start * x_stride_token + feat
                w_base = feat * w_stride_dim
                out_base = seq_start * o_stride_token + feat

                col2 = T.Cast(
                    "float32",
                    x[x_base + (token_offset - 1) * x_stride_token],
                )
                col1 = T.Cast(
                    "float32",
                    x[x_base + (token_offset - 2) * x_stride_token],
                )
                col0 = T.Cast(
                    "float32",
                    x[x_base + (token_offset - 3) * x_stride_token],
                )
                w0 = T.Cast("float32", weight[w_base])
                w1 = T.Cast("float32", weight[w_base + 1])
                w2 = T.Cast("float32", weight[w_base + 2])
                w3 = T.Cast("float32", weight[w_base + 3])

                if segment_len == block_m:
                    for token_i in T.serial(block_m):
                        x_cur = T.Cast(
                            "float32",
                            x[x_base + (token_offset + token_i) * x_stride_token],
                        )
                        acc = 0.0
                        if has_bias:
                            acc = T.Cast("float32", bias[feat])
                        acc += col0 * w0 + col1 * w1 + col2 * w2 + x_cur * w3
                        col0 = col1
                        col1 = col2
                        col2 = x_cur
                        if silu_activation:
                            acc = acc / (1.0 + T.exp2(-acc * _LOG2E))
                        out[out_base + (token_offset + token_i) * o_stride_token] = (
                            T.Cast(dtype, acc)
                        )
                else:
                    for token_i in T.serial(block_m):
                        if token_i < segment_len:
                            x_cur = T.Cast(
                                "float32",
                                x[x_base + (token_offset + token_i) * x_stride_token],
                            )
                            acc = 0.0
                            if has_bias:
                                acc = T.Cast("float32", bias[feat])
                            acc += col0 * w0 + col1 * w1 + col2 * w2 + x_cur * w3
                            col0 = col1
                            col1 = col2
                            col2 = x_cur
                            if silu_activation:
                                acc = acc / (1.0 + T.exp2(-acc * _LOG2E))
                            out[
                                out_base + (token_offset + token_i) * o_stride_token
                            ] = T.Cast(dtype, acc)

    return sglang_musa_causal_conv1d_prefill_width4_body


_causal_conv1d_prefill_width4_body_kernel.mode = "lazy"


@functools.lru_cache(maxsize=64)
@tilelang.jit(
    out_idx=[],
    target="musa",
    pass_configs=_CAUSAL_CONV1D_PASS_CONFIGS,
    compile_flags=MUSA_COMPILE_FLAGS,
)
def _causal_conv1d_decode_width4_batched_kernel(
    dtype: str,
    x_stride_token: int,
    w_stride_dim: int,
    state_stride_seq: int,
    state_stride_dim: int,
    state_stride_token: int,
    o_stride_token: int,
    has_bias: bool,
    has_cache_indices: bool,
    has_initial_states: bool,
    use_pad_slot: bool,
    silu_activation: bool,
    block_feats: int,
    batch_per_block: int,
):
    x_numel = T.dynamic("x_numel")
    w_numel = T.dynamic("w_numel")
    bias_numel = T.dynamic("bias_numel")
    state_numel = T.dynamic("state_numel")
    cache_numel = T.dynamic("cache_numel")
    init_numel = T.dynamic("init_numel")
    out_numel = T.dynamic("out_numel")
    num_threads = block_feats * batch_per_block

    @T.prim_func
    def sglang_musa_causal_conv1d_decode_width4_batched(
        x: T.Tensor((x_numel,), dtype),
        weight: T.Tensor((w_numel,), dtype),
        bias: T.Tensor((bias_numel,), dtype),
        conv_states: T.Tensor((state_numel,), dtype),
        cache_indices: T.Tensor((cache_numel,), "int32"),
        has_initial_state: T.Tensor((init_numel,), "bool"),
        out: T.Tensor((out_numel,), dtype),
        batch: T.int32,
        dim: T.int32,
        num_cache_lines: T.int32,
        pad_slot_id: T.int32,
    ):
        with T.Kernel(
            T.ceildiv(batch, batch_per_block),
            T.ceildiv(dim, block_feats),
            threads=num_threads,
        ) as (batch_block, dim_block):
            tid = T.get_thread_binding()
            batch_lane = tid // block_feats
            feat_lane = tid - batch_lane * block_feats
            seq_idx = batch_block * batch_per_block + batch_lane
            feat = dim_block * block_feats + feat_lane
            cache_idx = T.alloc_var("int32")
            valid = T.alloc_var("bool")
            load_init = T.alloc_var("bool")
            x_base = T.alloc_var("int32")
            w_base = T.alloc_var("int32")
            state_base = T.alloc_var("int32")
            col0 = T.alloc_var("float32")
            col1 = T.alloc_var("float32")
            col2 = T.alloc_var("float32")
            x_cur = T.alloc_var("float32")
            acc = T.alloc_var("float32")

            cache_idx = seq_idx
            if has_cache_indices and seq_idx < batch:
                cache_idx = cache_indices[seq_idx]
            valid = seq_idx < batch and feat < dim
            if use_pad_slot and cache_idx == pad_slot_id:
                valid = False
            if cache_idx >= num_cache_lines:
                valid = False

            if valid:
                load_init = False
                if has_initial_states:
                    load_init = has_initial_state[seq_idx]

                x_base = seq_idx * x_stride_token + feat
                w_base = feat * w_stride_dim
                state_base = cache_idx * state_stride_seq + feat * state_stride_dim
                x_cur = T.Cast("float32", x[x_base])
                col0 = 0.0
                col1 = 0.0
                col2 = 0.0
                if load_init:
                    col0 = T.Cast("float32", conv_states[state_base])
                    col1 = T.Cast(
                        "float32", conv_states[state_base + state_stride_token]
                    )
                    col2 = T.Cast(
                        "float32", conv_states[state_base + 2 * state_stride_token]
                    )

                acc = (
                    col0 * T.Cast("float32", weight[w_base])
                    + col1 * T.Cast("float32", weight[w_base + 1])
                    + col2 * T.Cast("float32", weight[w_base + 2])
                    + x_cur * T.Cast("float32", weight[w_base + 3])
                )
                if has_bias:
                    acc += T.Cast("float32", bias[feat])
                if silu_activation:
                    acc = acc / (1.0 + T.exp2(-acc * _LOG2E))

                out[seq_idx * o_stride_token + feat] = T.Cast(dtype, acc)
                if load_init:
                    conv_states[state_base] = conv_states[
                        state_base + state_stride_token
                    ]
                    conv_states[state_base + state_stride_token] = conv_states[
                        state_base + 2 * state_stride_token
                    ]
                else:
                    conv_states[state_base] = T.Cast(dtype, 0.0)
                    conv_states[state_base + state_stride_token] = T.Cast(dtype, 0.0)
                conv_states[state_base + 2 * state_stride_token] = T.Cast(dtype, x_cur)

    return sglang_musa_causal_conv1d_decode_width4_batched


_causal_conv1d_decode_width4_batched_kernel.mode = "lazy"


def _check_inputs(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    conv_states: Optional[torch.Tensor],
    query_start_loc: torch.Tensor,
    cache_indices: Optional[torch.Tensor],
    has_initial_state: Optional[torch.Tensor],
    activation: Optional[Union[str, bool]],
    seq_lens_cpu: List[int],
) -> tuple[int, int, int, int, str]:
    if isinstance(activation, bool) and activation:
        activation = "silu"
    if activation not in (None, "silu", "swish"):
        raise NotImplementedError("activation must be None, silu, or swish")
    if x.dim() != 2:
        raise ValueError(
            "TileLang causal_conv1d_fwd expects varlen x with shape (dim, total_tokens)"
        )
    if weight.dim() != 2:
        raise ValueError("weight must be a 2D tensor")
    if query_start_loc is None or query_start_loc.dim() != 1:
        raise ValueError("query_start_loc must be a 1D tensor")
    if query_start_loc.dtype is not torch.int32:
        raise TypeError("query_start_loc must be int32")
    dim, total_tokens = x.shape
    weight_dim, width = weight.shape
    if weight_dim != dim:
        raise ValueError("weight first dimension must match x dim")
    if width < 2 or width > 5:
        raise ValueError("TileLang causal_conv1d_fwd supports width in [2, 5]")
    if bias is not None and (bias.dim() != 1 or bias.numel() != dim):
        raise ValueError("bias must have shape (dim,)")
    if cache_indices is not None:
        if cache_indices.dim() != 1 or cache_indices.dtype is not torch.int32:
            raise TypeError("cache_indices must be a 1D int32 tensor")
        if cache_indices.numel() != query_start_loc.numel() - 1:
            raise ValueError("cache_indices length must match batch size")
    if has_initial_state is not None:
        if has_initial_state.dim() != 1 or has_initial_state.dtype is not torch.bool:
            raise TypeError("has_initial_state must be a 1D bool tensor")
        if has_initial_state.numel() != query_start_loc.numel() - 1:
            raise ValueError("has_initial_state length must match batch size")
        if conv_states is None:
            raise ValueError("has_initial_state requires conv_states")
    if conv_states is not None:
        if conv_states.dim() != 3:
            raise ValueError(
                "conv_states must have shape (num_cache_lines, dim, state_len)"
            )
        if conv_states.size(1) != dim or conv_states.size(2) < width - 1:
            raise ValueError("conv_states shape is incompatible with x/weight")
    if sum(seq_lens_cpu) != total_tokens:
        raise ValueError("seq_lens_cpu must sum to x.shape[1]")
    return (
        dim,
        total_tokens,
        width,
        max(seq_lens_cpu, default=0),
        tilelang_dtype(x.dtype),
    )


def _causal_conv1d_fwd_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    conv_states: Optional[torch.Tensor],
    query_start_loc: torch.Tensor,
    seq_lens_cpu: List[int],
    cache_indices: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    activation: Optional[Union[str, bool]] = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
) -> torch.Tensor:
    if isinstance(activation, bool) and activation:
        activation = "silu"
    dim, _total_tokens, width, max_seq_len, dtype = _check_inputs(
        x,
        weight,
        bias,
        conv_states,
        query_start_loc,
        cache_indices,
        has_initial_state,
        activation,
        seq_lens_cpu,
    )

    out = torch.empty_like(x)
    x_arg = storage_window(x)
    weight_arg = storage_window(weight)
    out_arg = storage_window(out)
    bias_arg = storage_window(bias) if bias is not None else x_arg
    conv_states_arg = storage_window(conv_states) if conv_states is not None else x_arg
    cache_indices_arg = cache_indices if cache_indices is not None else query_start_loc
    has_initial_state_arg = (
        has_initial_state
        if has_initial_state is not None
        else torch.empty((1,), dtype=torch.bool, device=x.device)
    )

    block_n = 256 if dim >= 256 else max(32, _next_power_of_2(dim))
    if max_seq_len == 1 and query_start_loc.numel() > 128 and dim >= 256:
        block_n = 128
    block_m = 8
    if width == 4 and max_seq_len >= 128 and query_start_loc.numel() > 2:
        block_m = 28
        block_n = 512
    num_cache_lines = conv_states.size(0) if conv_states is not None else 0
    state_stride_seq = conv_states.stride(0) if conv_states is not None else 0
    state_stride_dim = conv_states.stride(1) if conv_states is not None else 0
    state_stride_token = conv_states.stride(2) if conv_states is not None else 0

    if (
        width == 4
        and max_seq_len == 1
        and query_start_loc.numel() > 256
        and conv_states is not None
        and dim >= 4096
        and x.stride(0) == 1
        and out.stride(0) == 1
        and weight.stride(1) == 1
    ):
        _causal_conv1d_decode_width4_batched_kernel(
            dtype,
            int(x.stride(1)),
            int(weight.stride(0)),
            int(state_stride_seq),
            int(state_stride_dim),
            int(state_stride_token),
            int(out.stride(1)),
            bias is not None,
            cache_indices is not None,
            has_initial_state is not None,
            pad_slot_id is not None,
            activation in ("silu", "swish"),
            256,
            1,
        )(
            x_arg,
            weight_arg,
            bias_arg,
            conv_states_arg,
            cache_indices_arg,
            has_initial_state_arg,
            out_arg,
            int(query_start_loc.numel() - 1),
            int(dim),
            int(num_cache_lines),
            int(pad_slot_id if pad_slot_id is not None else PAD_SLOT_ID),
        )
        return out

    if (
        width == 4
        and max_seq_len >= 128
        and query_start_loc.numel() > 2
        and conv_states is not None
        and cache_indices is not None
        and x.stride(0) == 1
        and out.stride(0) == 1
        and weight.stride(1) == 1
    ):
        _causal_conv1d_prefill_width4_kernel(
            dtype,
            int(x.stride(1)),
            int(weight.stride(0)),
            int(state_stride_seq),
            int(state_stride_dim),
            int(state_stride_token),
            int(out.stride(1)),
            bias is not None,
            cache_indices is not None,
            has_initial_state is not None,
            pad_slot_id is not None,
            activation in ("silu", "swish"),
            int(block_m),
            int(block_n),
        )(
            x_arg,
            weight_arg,
            bias_arg,
            conv_states_arg,
            cache_indices_arg,
            has_initial_state_arg,
            query_start_loc,
            out_arg,
            int(block_m),
            int(dim),
            int(num_cache_lines),
            int(pad_slot_id if pad_slot_id is not None else PAD_SLOT_ID),
        )
        if max_seq_len > block_m:
            _causal_conv1d_prefill_width4_body_kernel(
                dtype,
                int(x.stride(1)),
                int(weight.stride(0)),
                int(out.stride(1)),
                bias is not None,
                cache_indices is not None,
                pad_slot_id is not None,
                activation in ("silu", "swish"),
                int(block_m),
                int(block_n),
            )(
                x_arg,
                weight_arg,
                bias_arg,
                cache_indices_arg,
                query_start_loc,
                out_arg,
                int(max_seq_len),
                int(dim),
                int(num_cache_lines),
                int(pad_slot_id if pad_slot_id is not None else PAD_SLOT_ID),
            )
        return out

    if (
        width == 4
        and dim >= 4096
        and max_seq_len == 1
        and query_start_loc.numel() == 2
        and x.stride(0) == 1
        and out.stride(0) == 1
        and weight.stride(1) == 1
    ):
        block_feats = 256
        vec_elems = 1
        _causal_conv1d_fwd_width4_vec_kernel(
            dtype,
            int(x.stride(1)),
            int(weight.stride(0)),
            int(state_stride_seq),
            int(state_stride_dim),
            int(state_stride_token),
            int(out.stride(1)),
            bias is not None,
            conv_states is not None,
            cache_indices is not None,
            has_initial_state is not None,
            pad_slot_id is not None,
            activation in ("silu", "swish"),
            int(block_m),
            int(block_feats),
            int(vec_elems),
        )(
            x_arg,
            weight_arg,
            bias_arg,
            conv_states_arg,
            cache_indices_arg,
            has_initial_state_arg,
            query_start_loc,
            out_arg,
            int(max_seq_len),
            int(dim),
            int(num_cache_lines),
            int(pad_slot_id if pad_slot_id is not None else PAD_SLOT_ID),
        )
        return out

    _causal_conv1d_fwd_kernel(
        dtype,
        int(width),
        int(x.stride(0)),
        int(x.stride(1)),
        int(weight.stride(0)),
        int(weight.stride(1)),
        int(state_stride_seq),
        int(state_stride_dim),
        int(state_stride_token),
        int(out.stride(0)),
        int(out.stride(1)),
        bias is not None,
        conv_states is not None,
        cache_indices is not None,
        has_initial_state is not None,
        pad_slot_id is not None,
        activation in ("silu", "swish"),
        int(block_m),
        int(block_n),
    )(
        x_arg,
        weight_arg,
        bias_arg,
        conv_states_arg,
        cache_indices_arg,
        has_initial_state_arg,
        query_start_loc,
        out_arg,
        int(max_seq_len),
        int(dim),
        int(num_cache_lines),
        int(pad_slot_id if pad_slot_id is not None else PAD_SLOT_ID),
    )
    return out


def _check_inputs_3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    conv_states: Optional[torch.Tensor],
    cache_indices: Optional[torch.Tensor],
    has_initial_state: Optional[torch.Tensor],
    activation: Optional[Union[str, bool]],
) -> tuple[int, int, int, int, str]:
    if isinstance(activation, bool) and activation:
        activation = "silu"
    if activation not in (None, "silu", "swish"):
        raise NotImplementedError("activation must be None, silu, or swish")
    if x.dim() != 3:
        raise ValueError(
            "TileLang causal_conv1d 3D path expects x with shape "
            "(batch, dim, seqlen)"
        )
    if weight.dim() != 2:
        raise ValueError("weight must be a 2D tensor")
    batch, dim, seq_len = x.shape
    weight_dim, width = weight.shape
    if weight_dim != dim:
        raise ValueError("weight first dimension must match x dim")
    if width < 2 or width > 5:
        raise ValueError("TileLang causal_conv1d_fwd supports width in [2, 5]")
    if bias is not None and (bias.dim() != 1 or bias.numel() != dim):
        raise ValueError("bias must have shape (dim,)")
    if cache_indices is not None:
        if cache_indices.dim() != 1 or cache_indices.dtype is not torch.int32:
            raise TypeError("cache_indices must be a 1D int32 tensor")
        if cache_indices.numel() != batch:
            raise ValueError("cache_indices length must match batch size")
    if has_initial_state is not None:
        if has_initial_state.dim() != 1 or has_initial_state.dtype is not torch.bool:
            raise TypeError("has_initial_state must be a 1D bool tensor")
        if has_initial_state.numel() != batch:
            raise ValueError("has_initial_state length must match batch size")
        if conv_states is None:
            raise ValueError("has_initial_state requires conv_states")
    if conv_states is not None:
        if conv_states.dim() != 3:
            raise ValueError(
                "conv_states must have shape (num_cache_lines, dim, state_len)"
            )
        if conv_states.size(1) != dim or conv_states.size(2) < width - 1:
            raise ValueError("conv_states shape is incompatible with x/weight")
    return batch, dim, seq_len, width, tilelang_dtype(x.dtype)


def _causal_conv1d_fwd_3d_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    conv_states: Optional[torch.Tensor],
    cache_indices: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    activation: Optional[Union[str, bool]] = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
) -> torch.Tensor:
    if isinstance(activation, bool):
        activation = "silu" if activation else None
    batch, dim, seq_len, width, dtype = _check_inputs_3d(
        x,
        weight,
        bias,
        conv_states,
        cache_indices,
        has_initial_state,
        activation,
    )

    out = torch.empty_like(x)
    x_arg = storage_window(x)
    weight_arg = storage_window(weight)
    out_arg = storage_window(out)
    bias_arg = storage_window(bias) if bias is not None else x_arg
    conv_states_arg = storage_window(conv_states) if conv_states is not None else x_arg
    cache_indices_arg = (
        cache_indices
        if cache_indices is not None
        else torch.empty((1,), dtype=torch.int32, device=x.device)
    )
    has_initial_state_arg = (
        has_initial_state
        if has_initial_state is not None
        else torch.empty((1,), dtype=torch.bool, device=x.device)
    )

    block_n = 256 if dim >= 256 else max(32, _next_power_of_2(dim))
    block_m = 8
    if width == 4 and seq_len >= 128 and batch > 1:
        block_m = 28
        block_n = 512

    num_cache_lines = conv_states.size(0) if conv_states is not None else 0
    state_stride_seq = conv_states.stride(0) if conv_states is not None else 0
    state_stride_dim = conv_states.stride(1) if conv_states is not None else 0
    state_stride_token = conv_states.stride(2) if conv_states is not None else 0

    _causal_conv1d_fwd_3d_kernel(
        dtype,
        int(width),
        int(x.stride(0)),
        int(x.stride(1)),
        int(x.stride(2)),
        int(weight.stride(0)),
        int(weight.stride(1)),
        int(state_stride_seq),
        int(state_stride_dim),
        int(state_stride_token),
        int(out.stride(0)),
        int(out.stride(1)),
        int(out.stride(2)),
        bias is not None,
        conv_states is not None,
        cache_indices is not None,
        has_initial_state is not None,
        pad_slot_id is not None,
        activation in ("silu", "swish"),
        int(block_m),
        int(block_n),
    )(
        x_arg,
        weight_arg,
        bias_arg,
        conv_states_arg,
        cache_indices_arg,
        has_initial_state_arg,
        out_arg,
        int(batch),
        int(dim),
        int(seq_len),
        int(num_cache_lines),
        int(pad_slot_id if pad_slot_id is not None else PAD_SLOT_ID),
    )
    return out


@register_custom_op(
    op_name="musa_causal_conv1d_fwd",
    mutates_args=["conv_states"],
)
def _causal_conv1d_fwd_custom(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    conv_states: Optional[torch.Tensor],
    query_start_loc: torch.Tensor,
    seq_lens_cpu: List[int],
    cache_indices: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
) -> torch.Tensor:
    return _causal_conv1d_fwd_impl(
        x,
        weight,
        bias,
        conv_states,
        query_start_loc,
        seq_lens_cpu,
        cache_indices,
        has_initial_state,
        activation,
        pad_slot_id,
    )


def causal_conv1d_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    conv_states: Optional[torch.Tensor],
    query_start_loc: torch.Tensor,
    seq_lens_cpu: List[int],
    cache_indices: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    activation: Optional[Union[str, bool]] = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
) -> torch.Tensor:
    if isinstance(activation, bool):
        activation = "silu" if activation else None
    return _causal_conv1d_fwd_impl(
        x,
        weight,
        bias,
        conv_states,
        query_start_loc,
        seq_lens_cpu,
        cache_indices,
        has_initial_state,
        activation,
        pad_slot_id,
    )


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    conv_states: Optional[torch.Tensor],
    query_start_loc: Optional[torch.Tensor] = None,
    seq_lens_cpu: Optional[List[int]] = None,
    cache_indices: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    activation: Optional[Union[str, bool]] = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
    **_: object,
) -> torch.Tensor:
    # The public MUSA entry eats both the extension-style fast path and the
    # varlen/stride path. 3D inputs use a native stride-aware kernel to avoid
    # transpose/reshape copies; 2D varlen inputs still use the existing
    # specialized/generic TileLang dispatch.
    if x.dim() == 3:
        return _causal_conv1d_fwd_3d_impl(
            x,
            weight,
            bias,
            conv_states,
            cache_indices,
            has_initial_state,
            activation,
            pad_slot_id,
        )

    if x.dim() != 2:
        raise ValueError(
            "TileLang causal_conv1d_fn expects x with shape "
            "(dim, total_tokens) or (batch, dim, seqlen)"
        )

    if query_start_loc is None:
        query_start_loc = torch.tensor(
            [0, x.shape[-1]], dtype=torch.int32, device=x.device
        )
    if seq_lens_cpu is None:
        seq_lens_cpu = (query_start_loc[1:] - query_start_loc[:-1]).cpu().tolist()

    return causal_conv1d_fwd(
        x,
        weight,
        bias,
        conv_states,
        query_start_loc,
        seq_lens_cpu,
        cache_indices,
        has_initial_state,
        activation,
        pad_slot_id,
    )
