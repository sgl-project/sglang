# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2024, Tri Dao.
# Adapted from https://github.com/Dao-AILab/causal-conv1d/blob/main/causal_conv1d/causal_conv1d_interface.py
# and https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/ops/causal_conv1d.py
# Modified by NVIDIA: vectorized four-tap prefill with in-place continuation state; interface follows causal_conv1d_fn.
# Modified by NVIDIA: vectorize variable-length prefill over tokens and channels.

import torch
import triton
import triton.language as tl


@triton.jit
def _causal_conv1d_prefill_kernel(
    X,
    W,
    Bias,
    State,
    Indices,
    HasInitial,
    Cu,
    Table,
    Out,
    SXC: tl.constexpr,
    SXT: tl.constexpr,
    SWC: tl.constexpr,
    SWT: tl.constexpr,
    SSB: tl.constexpr,
    SSC: tl.constexpr,
    SST: tl.constexpr,
    C: tl.constexpr,
    BM: tl.constexpr,
    BC: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_INDICES: tl.constexpr,
    HAS_INITIAL: tl.constexpr,
    HAS_TABLE: tl.constexpr,
    SILU: tl.constexpr,
    PAD: tl.constexpr,
    STATE_MODE: tl.constexpr,
):
    if HAS_TABLE:
        seq = tl.load(Table + tl.program_id(0) * 2)
        chunk = tl.load(Table + tl.program_id(0) * 2 + 1)
    else:
        seq = tl.program_id(2)
        chunk = tl.program_id(1)
    if STATE_MODE == 1:
        chunk = tl.full((), 0, tl.int32)
    elif STATE_MODE == 2:
        chunk += 1
    if seq == PAD:
        return
    start = tl.load(Cu + seq).to(tl.int64)
    length = tl.load(Cu + seq + 1).to(tl.int64) - start
    if chunk * BM >= length:
        return
    slot = seq.to(tl.int64)
    if HAS_INDICES:
        slot = tl.load(Indices + seq).to(tl.int64)
    if slot == PAD:
        return
    channel_block = tl.program_id(2) if HAS_TABLE else tl.program_id(0)
    channels = channel_block * BC + tl.arange(0, BC)
    tokens = chunk.to(tl.int64) * BM + tl.arange(0, BM)
    mask = (tokens[:, None] < length) & (channels[None, :] < C)
    initial = False
    if HAS_INITIAL:
        initial = tl.load(HasInitial + seq).to(tl.int1)
    if HAS_BIAS:
        acc = tl.broadcast_to(
            tl.load(Bias + channels, channels < C, 0).to(tl.float32)[None, :],
            (BM, BC),
        )
    else:
        acc = tl.zeros((BM, BC), tl.float32)
    for tap in tl.static_range(4):
        source = tokens + tap - 3
        value = tl.load(
            X + (start + source[:, None]) * SXT + channels[None, :] * SXC,
            mask & (source[:, None] >= 0),
            0,
        )
        # Only chunk zero can read mutable history. Other chunks have their
        # complete halo in immutable X because BM is at least three.
        if STATE_MODE != 2 and chunk == 0:
            prior = tl.load(
                State
                + slot * SSB
                + channels[None, :] * SSC
                + (source[:, None] + 3) * SST,
                mask & (source[:, None] < 0) & initial,
                0,
            )
            value = tl.where(source[:, None] < 0, prior, value)
        weight = tl.load(W + channels * SWC + tap * SWT, channels < C, 0)
        acc += value * weight[None, :]
    if SILU:
        acc = acc / (1 + tl.exp(-acc))
    tl.store(Out + (start + tokens[:, None]) * C + channels[None, :], acc, mask)
    if STATE_MODE != 2 and chunk == 0:
        windows = tl.arange(0, 4)
        final_source = length + windows - 3
        state_mask = (windows[:, None] < 3) & (channels[None, :] < C)
        final_prior = tl.load(
            State
            + slot * SSB
            + channels[None, :] * SSC
            + (final_source[:, None] + 3) * SST,
            state_mask & (final_source[:, None] < 0) & initial,
            0,
        )
        final_current = tl.load(
            X + (start + final_source[:, None]) * SXT + channels[None, :] * SXC,
            state_mask & (final_source[:, None] >= 0),
            0,
        )
        final = tl.where(final_source[:, None] < 0, final_prior, final_current)
        # This CTA owns every history read and write for these channels.
        tl.debug_barrier()
        tl.store(
            State + slot * SSB + channels[None, :] * SSC + windows[:, None] * SST,
            final,
            state_mask,
        )


def causal_conv1d_prefill_vectorized(
    x,
    weight,
    bias,
    conv_states,
    query_start_loc,
    seq_lens_cpu,
    cache_indices=None,
    has_initial_state=None,
    activation="silu",
    pad_slot_id=-1,
    block_table=None,
    block_m=64,
):
    """Four-tap BF16 causal convolution with in-place continuation state.

    ``x`` is [C, total_tokens], including strided views of packed projections;
    the returned [C, total_tokens] view has contiguous token-major backing.
    The rectangular path uses ordered history-owner and history-free launches;
    a block table uses one launch. State is [slots, C, 3] and stores the last
    three *input* tokens. Empty
    sequences leave state unchanged; padded slots are skipped.

    Valid cache indices must be unique, and x must not alias state. An optional
    contiguous int32/int64 ``block_table`` lists (sequence, chunk) pairs in
    units of ``block_m``: every active pair must occur exactly once, including
    chunk zero for every nonempty active sequence. Padded rows use pad_slot_id.
    ``seq_lens_cpu`` is host metadata used only to size a rectangular grid.
    """
    if x.ndim != 2 or weight.shape != (x.shape[0], 4):
        raise ValueError("Expected x[C,T] and four-tap weight[C,4]")
    channels, total = x.shape
    if conv_states.ndim != 3 or conv_states.shape[1:] != (channels, 3):
        raise ValueError("Expected conv_states[slots,C,3]")
    if any(t.dtype != torch.bfloat16 for t in (x, weight, conv_states)):
        raise ValueError("Prefill convolution requires BF16 input, weight and state")
    if bias is not None and (bias.shape != (channels,) or bias.dtype != x.dtype):
        raise ValueError("Expected BF16 bias[C]")
    if activation not in (None, "silu", "swish") or block_m not in (8, 64):
        raise ValueError("Supported activation is SiLU or None; block_m is 8 or 64")
    if block_table is not None and (
        block_table.ndim != 2
        or block_table.shape[1] != 2
        or not block_table.is_contiguous()
        or block_table.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("Expected contiguous integer block_table[blocks,2]")
    out = torch.empty((total, channels), device=x.device, dtype=x.dtype)
    if total == 0:
        return out.transpose(0, 1)
    # Specialize the history owner separately: its live history values must
    # not inflate register usage in the history-free long-sequence tiles.
    tile_m = min(block_m, 32) if block_table is None else block_m
    block_n = 128 if tile_m == 32 else (256 if tile_m == 8 else 64)
    common = {
        "C": channels,
        "BM": tile_m,
        "BC": block_n,
        "HAS_BIAS": bias is not None,
        "HAS_INDICES": cache_indices is not None,
        "HAS_INITIAL": has_initial_state is not None,
        "HAS_TABLE": block_table is not None,
        "SILU": activation in ("silu", "swish"),
        "PAD": pad_slot_id,
        "num_warps": 4,
    }
    args = (
        x,
        weight,
        bias,
        conv_states,
        cache_indices,
        has_initial_state,
        query_start_loc,
        block_table,
        out,
        *x.stride(),
        *weight.stride(),
        *conv_states.stride(),
    )
    channel_blocks = triton.cdiv(channels, block_n)
    if block_table is not None:
        _causal_conv1d_prefill_kernel[(block_table.shape[0], 1, channel_blocks)](
            *args,
            STATE_MODE=0,
            **common,
        )
    else:
        batch, max_length = len(seq_lens_cpu), max(seq_lens_cpu)
        _causal_conv1d_prefill_kernel[(channel_blocks, 1, batch)](
            *args,
            STATE_MODE=1,
            **common,
        )
        if max_length > tile_m:
            # Stream order completes the history owner before these tiles;
            # their entire convolution halo comes from immutable X.
            _causal_conv1d_prefill_kernel[
                (channel_blocks, triton.cdiv(max_length, tile_m) - 1, batch)
            ](*args, STATE_MODE=2, **common)
    return out.transpose(0, 1)
