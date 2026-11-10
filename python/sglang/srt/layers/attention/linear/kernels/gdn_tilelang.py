# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""Persistent SM70 GDN inference kernels implemented in TileLang.

The kernel is specialized for the Qwen3.5/Qwen3.8 inference contract:

* packed, row-major ``[token, q | k | v]`` projection output;
* ``K == V == 128`` and FP16 activations on SM70;
* variable-length packed batches;
* an indexed FP16/FP32 recurrent-state pool in ``[slot, Hv, V, K]`` order.

Each CTA owns a disjoint group of value columns and keeps that state shard in
registers while walking a sequence.  This exposes parallelism over value
columns without transposing the persistent state or materializing normalized
Q/K tensors.  The recurrence is the ordinary gated delta rule; this file is an
independent TileLang implementation, not a binding to FlashQLA.
"""

import os
from functools import lru_cache

import tilelang
import tilelang.language as T
import torch
from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}
if hasattr(tilelang.PassConfigKey, "TL_DISABLE_FAST_MATH"):
    _PASS_CONFIGS[tilelang.PassConfigKey.TL_DISABLE_FAST_MATH] = True
elif hasattr(tilelang.PassConfigKey, "TL_ENABLE_FAST_MATH"):
    _PASS_CONFIGS[tilelang.PassConfigKey.TL_ENABLE_FAST_MATH] = False

_KEY_DIM = 128
_VALUE_DIM = 128
_SOFTPLUS_THRESHOLD = 20.0
_THREADS = 128
_EXECUTION_BACKEND = "cython"


def _column_block(tokens: int, value_heads: int) -> int:
    """Select CTA column ownership for measured Qwen TP layouts.

    Short TP4 27B prefills need CTA count more than per-CTA reuse, while long
    prefills benefit from holding a wider state shard across the token loop.
    The choices intentionally remain small and auditable.
    """

    if value_heads == 12:
        return 32 if tokens <= 16 else 16
    if value_heads == 8:
        return 8 if tokens < 1024 else 32
    if value_heads in (16, 24, 32, 48):
        return 16
    return 8


@tilelang.jit(
    out_idx=[9],
    execution_backend=_EXECUTION_BACKEND,
    pass_configs=_PASS_CONFIGS,
)
def _packed_gdn_kernel(
    q_heads: int,
    value_heads: int,
    num_sequences: int,
    column_block: int,
    state_fp32: bool,
    threads: int = _THREADS,
):
    key_dim = _KEY_DIM
    value_dim = _VALUE_DIM
    qkv_dim = 2 * q_heads * key_dim + value_heads * value_dim
    nt = T.dynamic("nt")
    state_slots = T.dynamic("state_slots")
    heads_per_key = value_heads // q_heads
    state_dtype = T.float32 if state_fp32 else T.float16

    @T.prim_func
    def main(
        MixedQKV: T.Tensor([nt, qkv_dim], T.float16),
        GateA: T.Tensor([nt, value_heads], T.float16),
        GateB: T.Tensor([nt, value_heads], T.float16),
        ALog: T.Tensor([value_heads], T.float32),
        DtBias: T.Tensor([value_heads], T.float16),
        State: T.Tensor([state_slots, value_heads, value_dim, key_dim], state_dtype),
        StateIndices: T.Tensor([num_sequences], T.int32),
        CuSeqLens: T.Tensor([num_sequences + 1], T.int32),
        Scale: T.float32,
        Output: T.Tensor([nt, value_heads, value_dim], T.float16),
    ):
        with T.Kernel(
            T.ceildiv(value_dim, column_block),
            value_heads,
            num_sequences,
            threads=threads,
        ) as (value_tile, value_head, sequence):
            state = T.alloc_fragment([column_block, key_dim], T.float32)
            q = T.alloc_fragment([key_dim], T.float32)
            k = T.alloc_fragment([key_dim], T.float32)
            q_square = T.alloc_fragment([key_dim], T.float32)
            k_square = T.alloc_fragment([key_dim], T.float32)
            product = T.alloc_fragment([column_block, key_dim], T.float32)
            projected = T.alloc_fragment([column_block], T.float32)
            result = T.alloc_fragment([column_block], T.float32)
            q_norm = T.alloc_fragment([1], T.float32)
            k_norm = T.alloc_fragment([1], T.float32)

            state_slot = StateIndices[sequence]
            value_start = value_tile * column_block
            key_head = T.floordiv(value_head, heads_per_key)
            seq_start = CuSeqLens[sequence]
            seq_end = CuSeqLens[sequence + 1]

            if state_slot >= 0:
                for value_col, key_col in T.Parallel(column_block, key_dim):
                    if value_start + value_col < value_dim:
                        state[value_col, key_col] = State[
                            state_slot,
                            value_head,
                            value_start + value_col,
                            key_col,
                        ]
                    else:
                        state[value_col, key_col] = 0

                for local_token in T.serial(seq_end - seq_start):
                    token = seq_start + local_token
                    for key_col in T.Parallel(key_dim):
                        q[key_col] = T.cast(
                            MixedQKV[token, key_head * key_dim + key_col],
                            T.float32,
                        )
                        k[key_col] = T.cast(
                            MixedQKV[
                                token,
                                q_heads * key_dim + key_head * key_dim + key_col,
                            ],
                            T.float32,
                        )
                        q_square[key_col] = q[key_col] * q[key_col]
                        k_square[key_col] = k[key_col] * k[key_col]

                    T.reduce_sum(q_square, q_norm, dim=0)
                    T.reduce_sum(k_square, k_norm, dim=0)
                    for key_col in T.Parallel(key_dim):
                        q[key_col] = q[key_col] * Scale / T.sqrt(q_norm[0] + 1e-6)
                        k[key_col] = k[key_col] / T.sqrt(k_norm[0] + 1e-6)

                    gate_x = T.cast(GateA[token, value_head], T.float32) + T.cast(
                        DtBias[value_head], T.float32
                    )
                    softplus = T.if_then_else(
                        gate_x <= _SOFTPLUS_THRESHOLD,
                        T.log(1.0 + T.exp(gate_x)),
                        gate_x,
                    )
                    decay = T.exp(-T.exp(ALog[value_head]) * softplus)
                    beta = 1.0 / (
                        1.0 + T.exp(-T.cast(GateB[token, value_head], T.float32))
                    )

                    for value_col, key_col in T.Parallel(column_block, key_dim):
                        state[value_col, key_col] *= decay
                        product[value_col, key_col] = (
                            state[value_col, key_col] * k[key_col]
                        )
                    T.reduce_sum(product, projected, dim=1)

                    for value_col in T.Parallel(column_block):
                        value = T.if_then_else(
                            value_start + value_col < value_dim,
                            T.cast(
                                MixedQKV[
                                    token,
                                    2 * q_heads * key_dim
                                    + value_head * value_dim
                                    + value_start
                                    + value_col,
                                ],
                                T.float32,
                            ),
                            0,
                        )
                        projected[value_col] = (value - projected[value_col]) * beta

                    for value_col, key_col in T.Parallel(column_block, key_dim):
                        state[value_col, key_col] += projected[value_col] * k[key_col]
                        product[value_col, key_col] = (
                            state[value_col, key_col] * q[key_col]
                        )
                    T.reduce_sum(product, result, dim=1)

                    for value_col in T.Parallel(column_block):
                        if value_start + value_col < value_dim:
                            Output[token, value_head, value_start + value_col] = T.cast(
                                result[value_col], T.float16
                            )

                for value_col, key_col in T.Parallel(column_block, key_dim):
                    if value_start + value_col < value_dim:
                        State[
                            state_slot,
                            value_head,
                            value_start + value_col,
                            key_col,
                        ] = state[value_col, key_col]
            else:
                for local_token, value_col in T.Parallel(
                    seq_end - seq_start, column_block
                ):
                    if value_start + value_col < value_dim:
                        Output[
                            seq_start + local_token,
                            value_head,
                            value_start + value_col,
                        ] = 0

    return main


@lru_cache(maxsize=None)
def _get_packed_gdn_kernel(
    q_heads: int,
    value_heads: int,
    num_sequences: int,
    column_block: int,
    state_fp32: bool,
    threads: int = _THREADS,
):
    return _packed_gdn_kernel(
        q_heads=q_heads,
        value_heads=value_heads,
        num_sequences=num_sequences,
        column_block=column_block,
        state_fp32=state_fp32,
        threads=threads,
    )


def _run_packed_gdn(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> torch.Tensor:
    if mixed_qkv.device.type != "cuda":
        raise ValueError("TileLang GDN requires CUDA tensors.")
    if torch.cuda.get_device_capability(mixed_qkv.device) != (7, 0):
        raise ValueError("TileLang GDN is currently specialized for SM70.")
    if mixed_qkv.dtype != torch.float16:
        raise ValueError("TileLang GDN requires FP16 activations on SM70.")
    if ssm_states.dtype not in (torch.float16, torch.float32):
        raise ValueError("TileLang GDN requires an FP16 or FP32 recurrent-state pool.")
    # Unit inner stride is not enough: a column slice of a wider projection
    # (upstream's qwen3_5_gdn_prefill_projection_views) has stride(-1) == 1 but a
    # row stride wider than the row, and this kernel addresses row-major dense.
    # Checking only the inner stride let such a view through and read the wrong
    # elements -- wrong output, no error.
    if mixed_qkv.ndim != 2 or not mixed_qkv.is_contiguous():
        raise ValueError("mixed_qkv must be a contiguous packed 2D tensor.")
    if a.ndim != 2 or b.ndim != 2 or a.shape != b.shape:
        raise ValueError("a and b must have matching [tokens, value_heads] shapes.")
    if a.dtype != torch.float16 or b.dtype != torch.float16:
        raise ValueError("TileLang GDN requires FP16 gating inputs on SM70.")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("TileLang GDN requires contiguous packed gating inputs.")
    if ssm_states.ndim != 4:
        raise ValueError("ssm_states must have shape [slots, Hv, V, K].")

    value_heads, value_dim, key_dim = ssm_states.shape[-3:]
    if key_dim != _KEY_DIM or value_dim != _VALUE_DIM:
        raise ValueError("TileLang GDN currently supports K=V=128 only.")
    if a.shape != (mixed_qkv.shape[0], value_heads):
        raise ValueError("a/b shapes do not match mixed_qkv tokens and state heads.")
    qk_dim = mixed_qkv.shape[1] - value_heads * value_dim
    if qk_dim <= 0 or qk_dim % (2 * key_dim) != 0:
        raise ValueError("mixed_qkv does not contain a valid packed q/k/v layout.")
    q_heads = qk_dim // (2 * key_dim)
    if q_heads <= 0 or value_heads % q_heads != 0:
        raise ValueError("value_heads must be divisible by packed q_heads.")
    num_sequences = cache_indices.numel()
    if query_start_loc.numel() != num_sequences + 1:
        raise ValueError("query_start_loc must have one entry per sequence plus one.")
    if A_log.shape != (value_heads,) or dt_bias.shape != (value_heads,):
        raise ValueError("A_log/dt_bias must have one entry per value head.")

    # TileLang's global buffers are contiguous. Qwen3.8's ratio-3 projection
    # path already provides these packed tensors; keep a safe fallback for
    # other models without silently interpreting widened row strides.
    mixed_qkv = mixed_qkv.contiguous()
    cache_indices = cache_indices.contiguous().to(dtype=torch.int32)
    query_start_loc = query_start_loc.contiguous().to(dtype=torch.int32)
    tokens = mixed_qkv.shape[0]
    block_v = _column_block(tokens, value_heads)
    kernel = _get_packed_gdn_kernel(
        q_heads,
        value_heads,
        num_sequences,
        block_v,
        ssm_states.dtype == torch.float32,
    )
    return kernel(
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        ssm_states,
        cache_indices,
        query_start_loc,
        float(scale),
    )


class TileLangGDNKernel(LinearAttnKernelBase):
    """SM70 persistent GDN backend with packed prefill and decode."""

    supports_packed_decode = True
    supports_packed_extend = True
    supports_target_verify = False

    def packed_decode(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        out = _run_packed_gdn(
            mixed_qkv,
            a,
            b,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
        )
        return out.unsqueeze(0)

    def packed_extend(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        store_checkpoints: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, None, None]:
        if mixed_qkv.shape[0] < 256 and not store_checkpoints:
            out = _run_packed_gdn(
                mixed_qkv,
                a,
                b,
                A_log=A_log,
                dt_bias=dt_bias,
                scale=scale,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
            )
            return out.unsqueeze(0), None, None

        from sglang.srt.layers.attention.linear.kernels.gdn_chunked_tilelang import (
            packed_chunked_gdn_sm70,
            packed_recurrent_gdn_sm70,
        )

        value_heads, value_dim, key_dim = ssm_states.shape[-3:]
        qk_dim = mixed_qkv.shape[-1] - value_heads * value_dim
        q_heads = qk_dim // (2 * key_dim)
        direct_max_tokens = int(
            os.environ.get("SGLANG_V100_GDN_DIRECT_MAX_TOKENS", "448")
        )
        if direct_max_tokens < 0:
            raise ValueError("SGLANG_V100_GDN_DIRECT_MAX_TOKENS must be non-negative")
        packed_impl = (
            packed_recurrent_gdn_sm70
            if mixed_qkv.shape[0] <= direct_max_tokens
            else packed_chunked_gdn_sm70
        )
        out, checkpoints = packed_impl(
            mixed_qkv,
            a,
            b,
            q_heads=q_heads,
            value_heads=value_heads,
            a_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            state=ssm_states,
            state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            store_checkpoints=store_checkpoints,
        )
        return out, None, checkpoints

    def decode(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("TileLang GDN decode requires packed mixed_qkv.")

    def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        scale: float | None = None,
        store_checkpoints: bool = False,
        **kwargs,
    ) -> tuple:
        from sglang.srt.layers.attention.linear.kernels.gdn_chunked_tilelang import (
            chunked_gdn_sm70,
        )

        out, checkpoints = chunked_gdn_sm70(
            q,
            k,
            v,
            g,
            beta,
            scale=scale or k.shape[-1] ** -0.5,
            state=ssm_states,
            state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            store_checkpoints=store_checkpoints,
        )
        return out, None, checkpoints
