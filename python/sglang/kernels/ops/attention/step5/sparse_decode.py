# SPDX-License-Identifier: Apache-2.0
"""Step5 CuTe sparse decode over SGLang-owned native FULL KV storage."""

import functools
import math

import torch


def _add_periodic_work(sm_work, *, start, tasks, weight):
    if tasks <= 0:
        return
    full, remainder = divmod(tasks, len(sm_work))
    if full:
        increment = full * weight
        for index in range(len(sm_work)):
            sm_work[index] += increment
    for index in range(remainder):
        sm_work[(start + index) % len(sm_work)] += weight


def _split_plan_key(*, rows, sm_count, n_split4, n_split2):
    n_split1 = rows - n_split4 - n_split2
    sm_work = [0] * sm_count
    offset = 0
    _add_periodic_work(sm_work, start=offset, tasks=n_split4 * 4, weight=1)
    offset += n_split4 * 4
    _add_periodic_work(sm_work, start=offset, tasks=n_split2 * 2, weight=2)
    offset += n_split2 * 2
    _add_periodic_work(sm_work, start=offset, tasks=n_split1, weight=4)
    max_work = max(sm_work)
    work_items = rows + n_split2 + 3 * n_split4
    split_rows = n_split2 + n_split4
    return max_work, work_items, split_rows, n_split4


@functools.cache
def step5_sparse_decode_plan(rows: int, sm_count: int, max_split: int = 4):
    """Plan a fixed 1/2/4-way decode mix for one captured row capacity.

    This is the framework-independent host planner from the updated vLLM Step5
    reference. It minimizes the longest modeled SM wave first, then CTA and
    merge work. SGLang owns when to use it and the SM103 kernel execution.
    """
    if type(rows) is not int or rows <= 0:
        raise ValueError("Step5 decode rows must be a positive integer")
    if type(sm_count) is not int or sm_count <= 0:
        raise ValueError("Step5 decode SM count must be a positive integer")
    if max_split not in (1, 2, 4):
        raise ValueError("Step5 fixed decode max_split must be one of 1, 2, 4")
    if max_split == 1:
        return 0, 0, rows

    candidates = {(0, 0)}
    lower_bound = (4 * rows + sm_count - 1) // sm_count
    min_ctas = max(rows, (4 * rows + lower_bound - 1) // lower_bound)

    def add_candidates(work_items):
        if work_items < rows or work_items > 4 * rows:
            return
        n_split2 = work_items - rows
        if 0 <= n_split2 <= rows:
            candidates.add((0, n_split2))
        if max_split >= 4:
            low_split4 = max(0, (work_items - 2 * rows + 1) // 2)
            high_split4 = min(rows, (work_items - rows) // 3)
            for n_split4 in (
                low_split4,
                low_split4 + 1,
                (low_split4 + high_split4) // 2,
                high_split4 - 1,
                high_split4,
            ):
                if low_split4 <= n_split4 <= high_split4:
                    n_split2 = work_items - rows - 3 * n_split4
                    if 0 <= n_split2 <= rows - n_split4:
                        candidates.add((n_split4, n_split2))

    for base in (
        rows,
        min_ctas,
        sm_count,
        2 * sm_count,
        3 * sm_count,
        4 * sm_count,
        4 * rows,
    ):
        for delta in range(-32, 33):
            add_candidates(base + delta)
        for delta in range(-2 * sm_count, 2 * sm_count + 1, max(1, sm_count // 16)):
            add_candidates(base + delta)
    if max_split >= 4:
        candidates.add((rows, 0))
    candidates.add((0, rows))
    n_split4, n_split2 = min(
        candidates,
        key=lambda plan: _split_plan_key(
            rows=rows,
            sm_count=sm_count,
            n_split4=plan[0],
            n_split2=plan[1],
        ),
    )
    return n_split4, n_split2, rows + n_split2 + 3 * n_split4


def step5_sparse_decode_sm103_plan(rows: int, sm_count: int):
    """Choose the measured SM103 fixed-split admission policy.

    The reference planner models SM-wave balance, but the split merge and FP32
    workspace become slower than direct execution once the captured row count
    already fills an L20D's 148 SMs. Keep the reference mix below one SM wave
    and use one CTA per row at and above it.
    """
    if type(rows) is not int or rows <= 0:
        raise ValueError("Step5 decode rows must be a positive integer")
    if type(sm_count) is not int or sm_count <= 0:
        raise ValueError("Step5 decode SM count must be a positive integer")
    if rows >= sm_count:
        return 0, 0, rows
    return step5_sparse_decode_plan(rows, sm_count, max_split=4)


def step5_sparse_decode_partials(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    counts: torch.Tensor,
    packed_regions: torch.Tensor,
    causal_lengths: torch.Tensor,
    query_start_loc: torch.Tensor,
    valid_rows: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    n_split4: int = 0,
    n_split2: int = 0,
    dynamic_splits: bool = False,
    mtp_q_len: int = 1,
    softmax_scale: float = 1.0 / math.sqrt(192),
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> None:
    """Write BF16 final output or FP32 split partials with FP32 natural LSE.

    Q is [T,H,192], H in (8,16); native contiguous K/V are [slots,1,192], page64-aligned.
    The page16 views share storage, addresses and token numbering with native KV.
    Counts and causal lengths are int32 [T]; packed int64 regions [T,513/514/516]
    come from step5_pack_decode_regions. Counts outside valid_rows[0] must be
    zero, including padding within a batch. valid_rows is an int32 [1] device
    tensor, and query_start_loc contains monotone int32 request row boundaries.
    In ragged MTP mode each request contains at most mtp_q_len rows; boundaries
    cover all valid rows. These are producer contracts, not host data reads.

    Static output capacity is T+3*n_split4+n_split2; the first n_split4 rows use
    four partitions, the next n_split2 use two. Static padded outputs are zero
    with -inf LSE. Dynamic output reserves 16*T rows and compacts active work;
    unused dynamic capacity is neither initialized nor consumed by source merge.
    A single partition produces final attention. Otherwise the caller invokes
    the corresponding variable/dynamic natural-LSE merge.

    Caller owns disjoint metadata, partials and their lifetime/stream ordering.
    Warm compilation before Graph capture. The cache retains compiled functions
    only; this entry allocates no CUDA storage and performs no synchronization.
    """
    if q.device.type != "cuda" or torch.version.hip is not None:
        raise ValueError("Step5 sparse decode requires NVIDIA CUDA")
    if q.ndim != 3 or q.shape[1] not in (8, 16) or q.shape[2] != 192:
        raise ValueError("Step5 Q requires shape [T,8 or 16,192]")
    rows, heads, _ = q.shape
    if not 1 <= mtp_q_len <= 16:
        raise ValueError("Step5 mtp_q_len must be in [1,16]")
    if n_split4 < 0 or n_split2 < 0 or n_split4 + n_split2 > rows:
        raise ValueError("Invalid Step5 variable split plan")
    if dynamic_splits and (n_split4 or n_split2 or rows % mtp_q_len):
        raise ValueError(
            "Dynamic splits require no static plan and T divisible by mtp_q_len"
        )
    if not math.isfinite(softmax_scale) or softmax_scale <= 0:
        raise ValueError("Step5 softmax scale must be finite and positive")
    if k.ndim != 3 or tuple(k.shape[1:]) != (1, 192) or k.shape[0] % 64:
        raise ValueError("Step5 K requires page64-aligned native [slots,1,192]")
    if packed_regions.ndim != 2 or packed_regions.shape[1] not in (513, 514, 516):
        raise ValueError("Step5 packed regions require width 513/514/516")
    if query_start_loc.ndim != 1 or query_start_loc.shape[0] < 2:
        raise ValueError(
            "Step5 query_start_loc requires request boundaries [requests+1]"
        )
    work = rows * 16 if dynamic_splits else rows + 3 * n_split4 + n_split2
    if out.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("Step5 sparse decode output must be BF16 or FP32")
    if out.dtype == torch.float32 and work == rows:
        raise ValueError("FP32 decode output is reserved for split partials")
    kv_fp8 = k.dtype == torch.float8_e4m3fn
    if kv_fp8 != (k_scale is not None) or (k_scale is None) != (v_scale is None):
        raise ValueError("Step5 FP8 K/V require per-token scales")
    if kv_fp8:
        k_scale, v_scale = k_scale.view(-1), v_scale.view(-1)
    else:
        k_scale = v_scale = lse.new_empty(1)
    _validate_tensors(
        q.device,
        (
            ("Q", q, (rows, heads, 192), torch.bfloat16, 16),
            ("K", k, tuple(k.shape), k.dtype if kv_fp8 else torch.bfloat16, 16),
            ("V", v, tuple(k.shape), k.dtype, 16),
            ("k_scale", k_scale, (k.shape[0] if kv_fp8 else 1,), torch.float32, 4),
            ("v_scale", v_scale, (k.shape[0] if kv_fp8 else 1,), torch.float32, 4),
            ("counts", counts, (rows,), torch.int32, 4),
            (
                "regions",
                packed_regions,
                (rows, packed_regions.shape[1]),
                torch.int64,
                8,
            ),
            ("lengths", causal_lengths, (rows,), torch.int32, 4),
            (
                "query_start_loc",
                query_start_loc,
                tuple(query_start_loc.shape),
                torch.int32,
                4,
            ),
            ("valid_rows", valid_rows, (1,), torch.int32, 4),
            ("out", out, (work, heads, 192), out.dtype, 16),
            ("lse", lse, (work, heads), torch.float32, 4),
        ),
    )
    if rows:
        from sglang.kernels.ops.attention.step5.cute.cute_utils import device_cache_key
        from sglang.kernels.ops.attention.step5.cute.decode_dispatch import (
            compile_decode,
            device_sm_count,
        )

        device_key = device_cache_key(q.device)
        if device_key[2][0] == 9:
            _sparse_decode_sm90(
                q,
                k,
                v,
                device_key=device_key,
                packed_regions=packed_regions,
                counts=counts,
                causal_lengths=causal_lengths,
                query_start_loc=query_start_loc,
                valid_rows=valid_rows,
                out=out,
                lse=lse,
                softmax_scale=softmax_scale,
                n_split4=n_split4,
                n_split2=n_split2,
                dynamic_splits=dynamic_splits,
                mtp_q_len=mtp_q_len,
                k_scale=k_scale if kv_fp8 else None,
                v_scale=v_scale if kv_fp8 else None,
            )
            return
        compiled = compile_decode(
            packed_regions.shape[1],
            dynamic_splits,
            mtp_q_len,
            device_key,
            device_sm_count(device_key[1]),
            logical_q_heads=heads,
            output_fp32=out.dtype == torch.float32,
            kv_fp8=kv_fp8,
        )
        compiled(
            q,
            k.view(-1, 16, 1, 192),
            v.view(-1, 16, 1, 192),
            k_scale,
            v_scale,
            packed_regions,
            counts,
            causal_lengths,
            query_start_loc,
            valid_rows,
            out,
            lse,
            softmax_scale,
            n_split4,
            n_split2,
        )


def _sparse_decode_sm90(
    q,
    k,
    v,
    *,
    device_key,
    packed_regions,
    counts,
    causal_lengths,
    query_start_loc,
    valid_rows,
    out,
    lse,
    softmax_scale,
    n_split4,
    n_split2,
    dynamic_splits,
    mtp_q_len,
    k_scale,
    v_scale,
):
    from sglang.kernels.ops.attention.step5.cute.decode_dispatch import (
        compile_decode_sm90,
        device_sm_count,
    )

    compiled = compile_decode_sm90(
        packed_regions.shape[1],
        dynamic_splits,
        mtp_q_len,
        device_key,
        device_sm_count(device_key[1]),
        logical_q_heads=q.shape[1],
        output_fp32=out.dtype == torch.float32,
        kv_fp8=k_scale is not None,
    )
    compiled(
        q,
        k.view(-1, 16, 1, 192),
        v.view(-1, 16, 1, 192),
        k_scale,
        v_scale,
        packed_regions,
        counts,
        causal_lengths,
        query_start_loc,
        valid_rows,
        out,
        lse,
        softmax_scale,
        n_split4,
        n_split2,
    )


def _validate_tensors(device, tensors):
    for name, tensor, shape, dtype, alignment in tensors:
        if (
            tensor.device != device
            or tensor.dtype != dtype
            or tuple(tensor.shape) != shape
            or not tensor.is_contiguous()
            or tensor.data_ptr() % alignment
        ):
            raise ValueError(f"Step5 sparse decode invalid {name}")


def build_step5_sparse_decode_schedule(*, rows, request_rows, device_index):
    """Shape/device policy, evaluated once per forward/capture before layers."""
    from sglang.kernels.ops.attention.step5.cute.decode_dispatch import device_sm_count

    dynamic = request_rows == 1 and rows in (1, 4)
    split4 = split2 = 0
    if dynamic:
        work = 16 * rows
    else:
        split4, split2, work = step5_sparse_decode_sm103_plan(
            rows, device_sm_count(device_index)
        )
    return dict(
        decode_dynamic_splits=dynamic,
        decode_split4_rows=split4,
        decode_split2_rows=split2,
        decode_work_rows=work,
    )


def step5_sparse_decode_selected(
    q,
    k,
    v,
    *,
    packed,
    counts,
    causal_lengths,
    query_start_loc,
    valid_rows,
    out,
    lse,
    scale,
    dynamic_splits,
    n_split4,
    n_split2,
    work_rows,
    partial,
    partial_lse,
    head_gate=None,
    k_scale=None,
    v_scale=None,
):
    """Execute a precomputed split schedule and merge selected-region attention.

    Consumes packed regions, causal lengths and caller-owned scratch only. The
    caller prepares the schedule once before layer execution. No request state,
    ForwardBatch, speculative mode or indexer metadata is consulted here.
    """
    rows = q.shape[0]
    dynamic = dynamic_splits
    work = work_rows
    if head_gate is not None and (not dynamic or work == rows):
        raise ValueError("head gating requires the dynamic split merge")
    if work == rows:
        step5_sparse_decode_partials(
            q,
            k,
            v,
            counts=counts,
            packed_regions=packed,
            causal_lengths=causal_lengths,
            query_start_loc=query_start_loc,
            valid_rows=valid_rows,
            out=out,
            lse=lse,
            softmax_scale=scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )
        return

    # FP32 partial output removes the BF16 round-trip at every split boundary.
    # The final merge still writes the model's BF16 attention output and keeps
    # natural LSE in FP32. This narrows the arithmetic difference from the
    # unsplit path while retaining split scheduling.
    step5_sparse_decode_partials(
        q,
        k,
        v,
        counts=counts,
        packed_regions=packed,
        causal_lengths=causal_lengths,
        query_start_loc=query_start_loc,
        valid_rows=valid_rows,
        out=partial,
        lse=partial_lse,
        n_split4=n_split4,
        n_split2=n_split2,
        dynamic_splits=dynamic,
        mtp_q_len=rows if dynamic else 1,
        softmax_scale=scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    if dynamic:
        from sglang.kernels.ops.attention.step5.cute.splitkv_merge_sm90_gqa import (
            merge_dynamic_split_nat_lse_states_sm90_gqa,
        )

        merge_dynamic_split_nat_lse_states_sm90_gqa(
            partial,
            partial_lse,
            counts,
            out,
            lse,
            mtp_q_len=rows,
            **({"head_gate": head_gate} if head_gate is not None else {}),
        )
    else:
        from sglang.kernels.ops.attention.step5.cute.splitkv_merge_sm90_gqa import (
            merge_variable_split_nat_lse_states_sm90_gqa,
        )

        merge_variable_split_nat_lse_states_sm90_gqa(
            partial,
            partial_lse,
            out,
            lse,
            n_split4=n_split4,
            n_split2=n_split2,
        )
