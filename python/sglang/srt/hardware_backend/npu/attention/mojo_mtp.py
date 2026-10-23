# SPDX-License-Identifier: Apache-2.0
"""Device-length paged GQA for static DSpark draft blocks.

Adapted from the Mojo MTP kernels in shengzhaotian/sglang@d1bf05bd30.
Enable with ASCEND_USE_MOJO_MTP=1. Setting
SGLANG_NPU_ATTN_BACKEND_NEEDS_CPU_SEQ_LENS=0 also allows the draft backend
to opt out of the scheduler's CPU length mirror.

``seq_lens`` contains the valid KV end, including the current query block.
``max_kv_len`` is only a scheduling bound; it never controls valid KV reads.
The query block and GQA heads share a matrix tile. Long contexts can split
the KV scan across cores and merge the partial softmax results.
"""

from functools import lru_cache

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_mtp_kernel(
    Q,
    K,
    V,
    O,
    SeqLens,
    CuQ,
    Pages,
    Acc,
    Lse,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kp,
    stride_kt,
    stride_vp,
    stride_vt,
    stride_ot,
    stride_oh,
    stride_od,
    stride_pb,
    stride_pp,
    scale,
    BATCH: tl.constexpr,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    DIM: tl.constexpr,
    PAGE: tl.constexpr,
    GAMMA: tl.constexpr,
    SPLITS: tl.constexpr,
    Q_TILE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    GROUP: tl.constexpr = HQ // HKV
    for task in range(tl.program_id(0), BATCH * HKV * SPLITS, tl.num_programs(0)):
        split = task % SPLITS
        kv_head = (task // SPLITS) % HKV
        batch = task // (SPLITS * HKV)
        kv_len = tl.load(SeqLens + batch)
        q_start = tl.load(CuQ + batch)
        chunk = tl.cdiv(tl.cdiv(kv_len, SPLITS), PAGE) * PAGE
        kv_start = split * chunk
        kv_end = tl.minimum(kv_start + chunk, kv_len)

        rows = tl.arange(0, Q_TILE)
        q_index = rows // GROUP
        q_head = kv_head * GROUP + rows % GROUP
        dims = tl.arange(0, BLOCK_D)
        q = tl.load(
            Q
            + (q_start + q_index[:, None]) * stride_qt
            + q_head[:, None] * stride_qh
            + dims[None, :] * stride_qd,
            mask=(q_index[:, None] < GAMMA) & (dims[None, :] < DIM),
            other=0.0,
        )
        maximum = tl.full((Q_TILE,), float("-inf"), tl.float32)
        denominator = tl.zeros((Q_TILE,), tl.float32)
        acc = tl.zeros((Q_TILE, BLOCK_D), tl.float32)

        for block in range(tl.cdiv(tl.maximum(kv_end - kv_start, 0), BLOCK_N)):
            start = kv_start + block * BLOCK_N
            count = tl.minimum(BLOCK_N, kv_end - start)
            physical_page = tl.load(
                Pages + batch * stride_pb + (start // PAGE) * stride_pp
            )
            in_page = start % PAGE
            k_ptr = tl.make_block_ptr(
                base=K
                + physical_page * stride_kp
                + kv_head * DIM
                + in_page * stride_kt,
                shape=(DIM, count),
                strides=(1, stride_kt),
                offsets=(0, 0),
                block_shape=(BLOCK_D, BLOCK_N),
                order=(0, 1),
            )
            v_ptr = tl.make_block_ptr(
                base=V
                + physical_page * stride_vp
                + kv_head * DIM
                + in_page * stride_vt,
                shape=(count, DIM),
                strides=(stride_vt, 1),
                offsets=(0, 0),
                block_shape=(BLOCK_N, BLOCK_D),
                order=(1, 0),
            )
            keys = tl.load(k_ptr, boundary_check=(0, 1), padding_option="zero")
            values = tl.load(v_ptr, boundary_check=(0, 1), padding_option="zero")
            scores = tl.dot(q, keys) * scale
            kv_positions = start + tl.arange(0, BLOCK_N)
            allowed = (kv_positions[None, :] < kv_end) & (q_index[:, None] < GAMMA)
            if CAUSAL:
                allowed = allowed & (
                    kv_positions[None, :] <= kv_len - GAMMA + q_index[:, None]
                )
            scores = tl.where(allowed, scores, float("-inf"))
            next_max = tl.maximum(maximum, tl.max(scores, 1))
            # A split can be empty for a causal query or a padded request.
            safe_max = tl.where(next_max == float("-inf"), 0.0, next_max)
            probability = tl.exp(scores - safe_max[:, None])
            correction = tl.exp(maximum - safe_max)
            denominator = denominator * correction + tl.sum(probability, 1)
            acc = acc * correction[:, None] + tl.dot(probability.to(keys.dtype), values)
            maximum = next_max

        safe_denominator = tl.where(denominator > 0, denominator, 1.0)
        acc = acc / safe_denominator[:, None]
        if SPLITS == 1:
            tl.store(
                O
                + (q_start + q_index[:, None]) * stride_ot
                + q_head[:, None] * stride_oh
                + dims[None, :] * stride_od,
                acc,
                mask=(q_index[:, None] < GAMMA) & (dims[None, :] < DIM),
            )
        else:
            tl.store(
                Acc + task * Q_TILE * DIM + rows[:, None] * DIM + dims[None, :],
                acc,
                mask=dims[None, :] < DIM,
            )
            lse = tl.where(
                denominator > 0, maximum + tl.log(safe_denominator), float("-inf")
            )
            tl.store(Lse + task * Q_TILE + rows, lse)


@triton.jit
def _reduce_mtp_kernel(
    Acc,
    Lse,
    O,
    CuQ,
    stride_ot,
    stride_oh,
    stride_od,
    BATCH: tl.constexpr,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    DIM: tl.constexpr,
    GAMMA: tl.constexpr,
    SPLITS: tl.constexpr,
    Q_TILE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    GROUP: tl.constexpr = HQ // HKV
    for task in range(tl.program_id(0), BATCH * HKV, tl.num_programs(0)):
        batch = task // HKV
        kv_head = task % HKV
        q_start = tl.load(CuQ + batch)
        rows = tl.arange(0, Q_TILE)
        dims = tl.arange(0, BLOCK_D)
        maximum = tl.full((Q_TILE,), float("-inf"), tl.float32)
        for split in tl.static_range(SPLITS):
            lse = tl.load(Lse + (task * SPLITS + split) * Q_TILE + rows)
            maximum = tl.maximum(maximum, lse)
        safe_max = tl.where(maximum == float("-inf"), 0.0, maximum)
        denominator = tl.zeros((Q_TILE,), tl.float32)
        result = tl.zeros((Q_TILE, BLOCK_D), tl.float32)
        for split in tl.static_range(SPLITS):
            workspace_task = task * SPLITS + split
            lse = tl.load(Lse + workspace_task * Q_TILE + rows)
            weight = tl.exp(lse - safe_max)
            partial = tl.load(
                Acc
                + workspace_task * Q_TILE * DIM
                + rows[:, None] * DIM
                + dims[None, :],
                mask=dims[None, :] < DIM,
                other=0.0,
            )
            result += weight[:, None] * partial
            denominator += weight
        result /= tl.where(denominator > 0, denominator, 1.0)[:, None]
        q_index = rows // GROUP
        q_head = kv_head * GROUP + rows % GROUP
        tl.store(
            O
            + (q_start + q_index[:, None]) * stride_ot
            + q_head[:, None] * stride_oh
            + dims[None, :] * stride_od,
            result,
            mask=(q_index[:, None] < GAMMA) & (dims[None, :] < DIM),
        )


@lru_cache(maxsize=1)
def _get_num_cores() -> tuple[int, int]:
    properties = triton.runtime.driver.active.utils.get_device_properties("npu")
    return properties["num_aicore"], properties["num_vectorcore"]


def mojo_paged_mtp_decode(
    *,
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_q_lens: torch.Tensor,
    block_tables: torch.Tensor,
    page_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    gamma: int,
    softmax_scale: float,
    max_kv_len: int,
    causal: bool = False,
) -> torch.Tensor:
    """Attend to actual device KV lengths; ``max_kv_len`` only sizes the work.

    Inputs use FIA's ND paged layout ``[pages, page_size, kv_heads * dim]``.
    Zero lengths denote padding, whose output is zero. All request rows have
    ``gamma`` queries; ragged verification and sliding windows are unsupported.
    """
    batch_size = seq_lens.numel()
    if (
        gamma <= 0
        or num_q_heads <= 0
        or num_kv_heads <= 0
        or num_q_heads % num_kv_heads
    ):
        raise ValueError(
            "Mojo MTP requires a positive query width and integral GQA groups"
        )
    if page_size < 16 or page_size & (page_size - 1):
        raise ValueError("Mojo MTP requires a power-of-two page size >= 16")
    if head_dim < 16 or head_dim > 256:
        raise ValueError("Mojo MTP requires a head dimension between 16 and 256")
    if query.shape != (batch_size * gamma, num_q_heads, head_dim):
        raise ValueError("Mojo MTP requires one fixed-width query block per request")
    expected_cache_tail = (page_size, num_kv_heads * head_dim)
    if k_cache.shape[1:] != expected_cache_tail or v_cache.shape != k_cache.shape:
        raise ValueError("Mojo MTP requires ND paged K/V with equal head dimensions")
    if k_cache.stride(-1) != 1 or v_cache.stride(-1) != 1:
        raise ValueError("Mojo MTP requires contiguous KV head dimensions")
    if seq_lens.dtype != torch.int32 or cu_q_lens.dtype != torch.int32:
        raise ValueError("Mojo MTP requires int32 device length tensors")
    if cu_q_lens.shape != (batch_size + 1,) or block_tables.shape[0] != batch_size:
        raise ValueError("Mojo MTP query offsets and page rows must match the batch")
    if query.dtype not in (torch.float16, torch.bfloat16) or not (
        query.dtype == k_cache.dtype == v_cache.dtype
    ):
        raise ValueError("Mojo MTP requires matching BF16 or FP16 Q/K/V")
    if any(
        t.device != query.device
        for t in (k_cache, v_cache, seq_lens, cu_q_lens, block_tables)
    ):
        raise ValueError("Mojo MTP inputs must reside on the same device")
    output = torch.empty_like(query)
    if not batch_size:
        return output
    cube_cores, vector_cores = _get_num_cores()
    group_size = num_q_heads // num_kv_heads
    tasks = batch_size * num_kv_heads
    splits = 1
    if tasks < cube_cores and max_kv_len >= (256 if group_size == 1 else 2048):
        splits = max(1, min(cube_cores // tasks, max_kv_len // 256))
    q_tile = max(16, triton.next_power_of_2(gamma * group_size))
    block_d = triton.next_power_of_2(head_dim)
    if splits > 1:
        acc = torch.empty(
            (tasks * splits, q_tile, head_dim), dtype=torch.float32, device=query.device
        )
        lse = torch.empty(
            (tasks * splits, q_tile), dtype=torch.float32, device=query.device
        )
    else:
        acc = lse = output  # Unused pointer arguments in the unsplit specialization.
    _paged_mtp_kernel[(cube_cores,)](
        query,
        k_cache,
        v_cache,
        output,
        seq_lens,
        cu_q_lens,
        block_tables,
        acc,
        lse,
        *query.stride(),
        k_cache.stride(0),
        k_cache.stride(1),
        v_cache.stride(0),
        v_cache.stride(1),
        *output.stride(),
        *block_tables.stride(),
        softmax_scale,
        BATCH=batch_size,
        HQ=num_q_heads,
        HKV=num_kv_heads,
        DIM=head_dim,
        PAGE=page_size,
        GAMMA=gamma,
        SPLITS=splits,
        Q_TILE=q_tile,
        BLOCK_D=block_d,
        BLOCK_N=min(128, page_size),
        CAUSAL=causal,
    )
    if splits > 1:
        _reduce_mtp_kernel[(vector_cores,)](
            acc,
            lse,
            output,
            cu_q_lens,
            *output.stride(),
            BATCH=batch_size,
            HQ=num_q_heads,
            HKV=num_kv_heads,
            DIM=head_dim,
            GAMMA=gamma,
            SPLITS=splits,
            Q_TILE=q_tile,
            BLOCK_D=block_d,
        )
    return output
