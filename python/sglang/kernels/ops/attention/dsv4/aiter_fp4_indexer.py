from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, Union

import torch

if TYPE_CHECKING:
    from sglang.kernels.ops.attention.dsv4.compress import (
        CompressorDecodePlan,
        CompressorPrefillPlan,
    )


_Q_HEADS = 64
_Q_HEAD_DIM = 128
_ROPE_DIM = 64
_GROUP_SIZE = 32
_KV_BLOCK_SIZE = 64
_Q_SCALE_SHAPE = (1, 4, 16, 4)


def prepare_aiter_fp4_indexer_cos_sin(
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    expected_rope_width = _ROPE_DIM // 2
    if (
        freqs_cis.ndim != 2
        or freqs_cis.shape[0] == 0
        or freqs_cis.shape[1] != expected_rope_width
        or not freqs_cis.is_complex()
    ):
        raise ValueError(
            "AITER FP4 C4Indexer requires complex freqs_cis with shape "
            f"[max_position, {expected_rope_width}]; got shape "
            f"{tuple(freqs_cis.shape)} and dtype {freqs_cis.dtype}"
        )

    cos = freqs_cis.real.to(dtype=torch.bfloat16).contiguous()
    sin = freqs_cis.imag.to(dtype=torch.bfloat16).contiguous()
    return cos, sin


def _validate_cos_sin(
    cos: torch.Tensor, sin: torch.Tensor, device: torch.device
) -> None:
    expected_rope_width = _ROPE_DIM // 2
    for name, tensor in (("cos", cos), ("sin", sin)):
        if (
            tensor.ndim != 2
            or tensor.shape[0] == 0
            or tensor.shape[1] != expected_rope_width
        ):
            raise ValueError(
                "AITER FP4 C4Indexer requires precomputed cos/sin with shape "
                f"[max_position, {expected_rope_width}]; got {name} shape "
                f"{tuple(tensor.shape)}"
            )
        if tensor.dtype != torch.bfloat16:
            raise ValueError(
                "AITER FP4 C4Indexer requires precomputed cos/sin dtype "
                f"torch.bfloat16; got {name} dtype {tensor.dtype}"
            )
        if not tensor.is_contiguous():
            raise ValueError(f"AITER FP4 C4Indexer requires contiguous {name}")
        if tensor.device != device:
            raise ValueError(
                "AITER FP4 C4Indexer requires q/k and precomputed cos/sin on "
                f"the same device; got q/k on {device} and {name} on "
                f"{tensor.device}"
            )
    if cos.shape != sin.shape:
        raise ValueError(
            "AITER FP4 C4Indexer requires matching precomputed cos/sin shapes; "
            f"got cos {tuple(cos.shape)} and sin {tuple(sin.shape)}"
        )


def aiter_q_indexer_rope_hadamard_fp4_quant(
    q: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if q.ndim != 3 or tuple(q.shape[1:]) != (_Q_HEADS, _Q_HEAD_DIM):
        raise ValueError(
            "AITER FP4 C4Indexer requires q shape [T, 64, 128]; "
            f"got {tuple(q.shape)}"
        )
    if q.dtype != torch.bfloat16:
        raise ValueError(
            "AITER FP4 C4Indexer requires q dtype torch.bfloat16; " f"got {q.dtype}"
        )
    if not q.is_contiguous():
        raise ValueError("AITER FP4 C4Indexer requires contiguous q")

    num_tokens = q.shape[0]
    if positions.ndim != 1 or positions.shape[0] != num_tokens:
        raise ValueError(
            "AITER FP4 C4Indexer requires positions shape [T] matching q; "
            f"got positions {tuple(positions.shape)} for T={num_tokens}"
        )

    _validate_cos_sin(cos, sin, q.device)
    positions = positions.to(device=q.device, dtype=torch.int64).contiguous()

    # AITER is intentionally imported only after the HIP FP4 path is selected.
    import aiter

    q_fp4 = torch.empty(
        (num_tokens, _Q_HEADS, _Q_HEAD_DIM // 2),
        dtype=aiter.dtypes.fp4x2,
        device=q.device,
    )
    q_scale = torch.empty(
        (num_tokens, *_Q_SCALE_SHAPE), dtype=torch.uint8, device=q.device
    )
    aiter.rope_rotate_activation(
        q_fp4,
        q,
        cos,
        sin,
        positions,
        rope_dim=_ROPE_DIM,
        out_scale=q_scale,
        group_size=_GROUP_SIZE,
        shuffle_scale=True,
        do_rotate_act=True,
    )
    return q_fp4, q_scale


def aiter_fp4_paged_mqa_logits(
    *,
    q_fp4: torch.Tensor,
    q_scale: torch.Tensor,
    k_payload: torch.Tensor,
    k_scale: torch.Tensor,
    weights: torch.Tensor,
    page_table: torch.Tensor,
    c4_seq_lens: torch.Tensor,
    weight_scale: float,
    is_decode: bool,
) -> torch.Tensor:
    num_tokens = q_fp4.shape[0]
    if q_fp4.ndim != 3 or tuple(q_fp4.shape[1:]) != (_Q_HEADS, _Q_HEAD_DIM // 2):
        raise ValueError(
            "AITER FP4 C4Indexer logits requires q_fp4 shape [T, 64, 64]; "
            f"got {tuple(q_fp4.shape)}"
        )
    if q_scale.shape != (num_tokens, *_Q_SCALE_SHAPE):
        raise ValueError(
            "AITER FP4 C4Indexer logits requires q_scale shape "
            f"[T, 1, 4, 16, 4]; got {tuple(q_scale.shape)}"
        )
    if q_scale.dtype != torch.uint8:
        raise ValueError(
            "AITER FP4 C4Indexer logits requires uint8 Q scales; "
            f"got {q_scale.dtype}"
        )
    if tuple(k_payload.shape[1:]) != (1, 4, _KV_BLOCK_SIZE, 16):
        raise ValueError(
            "AITER FP4 C4Indexer logits requires K payload shape "
            f"[P, 1, 4, 64, 16]; got {tuple(k_payload.shape)}"
        )
    if k_payload.shape[0] == 0:
        raise ValueError(
            "AITER FP4 C4Indexer logits requires at least one K cache page"
        )
    if k_scale.shape != k_payload.shape[:-1] or k_scale.dtype != torch.uint8:
        raise ValueError(
            "AITER FP4 C4Indexer logits requires uint8 K scales with shape "
            f"[P, 1, 4, 64]; got shape {tuple(k_scale.shape)} and "
            f"dtype {k_scale.dtype}"
        )
    if weights.shape != (num_tokens, _Q_HEADS):
        raise ValueError(
            "AITER FP4 C4Indexer logits requires weights shape [T, 64]; "
            f"got {tuple(weights.shape)}"
        )
    if weights.dtype != torch.bfloat16 or not weights.is_contiguous():
        raise ValueError(
            "AITER FP4 C4Indexer logits requires contiguous bfloat16 weights"
        )
    if page_table.ndim != 2 or page_table.shape[0] != num_tokens:
        raise ValueError(
            "AITER FP4 C4Indexer logits requires row-expanded page_table "
            f"shape [T, max_blocks]; got {tuple(page_table.shape)}"
        )

    page_table = page_table.to(dtype=torch.int32).contiguous()
    logical_page_table_width = page_table.shape[1]
    padded_page_table_width = max(4, (logical_page_table_width + 3) // 4 * 4)
    padded_page_table = page_table.new_zeros(
        (num_tokens, padded_page_table_width), dtype=torch.int32
    )
    padded_page_table[:, :logical_page_table_width].copy_(page_table)
    c4_seq_lens = c4_seq_lens.reshape(-1).to(dtype=torch.int32).contiguous()
    if c4_seq_lens.shape != (num_tokens,):
        raise ValueError(
            "AITER FP4 C4Indexer logits requires row-expanded c4_seq_lens "
            f"shape [T]; got {tuple(c4_seq_lens.shape)}"
        )
    logical_max_seq_len = logical_page_table_width * _KV_BLOCK_SIZE
    max_seq_len = padded_page_table_width * _KV_BLOCK_SIZE

    # AITER FlyDSL is intentionally imported only after the HIP FP4 path is selected.
    from aiter.ops.flydsl import (
        flydsl_pa_mqa_logits_fp4,
        flydsl_pa_mqa_logits_fp4_prefill,
    )

    common_kwargs = {
        "weight_scale": weight_scale,
        "block_k": 256,
        "kv_block_size": _KV_BLOCK_SIZE,
        "num_warps": 4,
    }
    # FlyDSL models packed FP4 payloads as raw bytes. The AITER producer uses
    # torch's fp4x2 dtype for its pybind dispatch, so expose a zero-copy uint8
    # view at this boundary.
    q_payload = q_fp4.view(torch.uint8)
    k_payload_bytes = k_payload.view(torch.uint8)
    if is_decode:
        logits = flydsl_pa_mqa_logits_fp4(
            q_payload.reshape(num_tokens, 1, _Q_HEADS, _Q_HEAD_DIM // 2),
            q_scale.reshape(num_tokens, 1, *_Q_SCALE_SHAPE),
            k_payload_bytes,
            k_scale,
            padded_page_table,
            weights,
            c4_seq_lens,
            max_seq_len,
            next_n=1,
            parallel_unit_num=None,
            **common_kwargs,
        )
    else:
        row_to_batch = torch.arange(num_tokens, device=q_fp4.device, dtype=torch.int32)
        local_starts = torch.zeros(num_tokens, device=q_fp4.device, dtype=torch.int32)
        # This eager grid depends only on T, avoids sequence-value synchronization,
        # and guarantees at least one persistent unit for every query row.
        parallel_unit_num = max(512, num_tokens)
        logits = flydsl_pa_mqa_logits_fp4_prefill(
            q_payload,
            q_scale,
            k_payload_bytes,
            k_scale,
            padded_page_table,
            weights,
            row_to_batch,
            local_starts,
            c4_seq_lens,
            max_seq_len,
            parallel_unit_num=parallel_unit_num,
            **common_kwargs,
        )

    return logits[:, :logical_max_seq_len].contiguous()


def aiter_k_indexer_fp4_cache_write(
    *,
    k: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_epsilon: float,
    cos: torch.Tensor,
    sin: torch.Tensor,
    plan: Union[CompressorDecodePlan, CompressorPrefillPlan],
    out_loc: torch.Tensor,
    k_payload: torch.Tensor,
    k_scale: torch.Tensor,
) -> None:
    if k.ndim != 2 or k.shape[1] != _Q_HEAD_DIM:
        raise ValueError(
            "AITER FP4 C4Indexer requires k shape [N, 128]; " f"got {tuple(k.shape)}"
        )
    num_rows = k.shape[0]
    if plan.compress_ratio != 4 or plan[1].shape != (num_rows, 16):
        raise ValueError(
            "AITER FP4 C4Indexer requires a C4 plan with one 16-byte row "
            f"per K row; got ratio={plan.compress_ratio}, "
            f"plan shape={tuple(plan[1].shape)}, N={num_rows}"
        )
    if out_loc.ndim != 1:
        raise ValueError(
            "AITER FP4 C4Indexer requires one-dimensional out_loc; "
            f"got {tuple(out_loc.shape)}"
        )
    if norm_weight.shape != (_Q_HEAD_DIM,):
        raise ValueError(
            "AITER FP4 C4Indexer requires norm_weight shape [128]; "
            f"got {tuple(norm_weight.shape)}"
        )
    if tuple(k_payload.shape[1:]) != (1, 4, _KV_BLOCK_SIZE, 16):
        raise ValueError(
            "AITER FP4 C4Indexer requires payload shape [P, 1, 4, 64, 16]; "
            f"got {tuple(k_payload.shape)}"
        )
    if k_scale.shape != k_payload.shape[:-1]:
        raise ValueError(
            "AITER FP4 C4Indexer requires scale shape [P, 1, 4, 64]; "
            f"got {tuple(k_scale.shape)}"
        )
    if k_scale.dtype != torch.uint8:
        raise ValueError(
            "AITER FP4 C4Indexer requires uint8 K scales; " f"got {k_scale.dtype}"
        )
    _validate_cos_sin(cos, sin, k.device)
    if num_rows == 0:
        return

    plan_words = plan[1].view(torch.int32)
    seq_lens = plan_words[:, 0].to(torch.int64)
    positions = seq_lens - plan.compress_ratio
    position_in_range = (positions >= 0) & (positions < cos.shape[0])
    positions = torch.where(position_in_range, positions, torch.zeros_like(positions))
    valid = position_in_range & (seq_lens % plan.compress_ratio == 0)

    out_loc_i64 = out_loc.to(device=k.device, dtype=torch.int64)
    if plan.is_decode:
        if out_loc_i64.shape[0] != num_rows:
            raise ValueError(
                "AITER FP4 C4Indexer decode requires out_loc length N; "
                f"got {out_loc_i64.shape[0]} for N={num_rows}"
            )
        selected_slots = out_loc_i64
    elif out_loc_i64.shape[0] == 0:
        selected_slots = torch.full_like(seq_lens, -1)
        valid = torch.zeros_like(valid)
    else:
        ragged_ids = plan_words[:, 1].bitwise_and(0xFFFF).to(torch.int64)
        valid = valid & (ragged_ids < out_loc_i64.shape[0])
        safe_ragged_ids = ragged_ids.clamp(max=out_loc_i64.shape[0] - 1)
        selected_slots = out_loc_i64[safe_ragged_ids]

    slots = torch.where(valid, selected_slots, torch.full_like(selected_slots, -1))
    positions = positions.contiguous()
    slots = slots.contiguous()
    k_bf16 = k.to(dtype=torch.bfloat16).contiguous().view(num_rows, 1, _Q_HEAD_DIM)
    # Convert all 128 values per call so post-load weight mutations cannot stale a cache.
    norm_weight_bf16 = norm_weight.to(
        device=k.device, dtype=torch.bfloat16
    ).contiguous()

    # AITER is intentionally imported only after the HIP FP4 path is selected.
    import aiter

    aiter.rmsnorm_rope_rotate_activation_fp4quant_kvcache(
        k_payload,
        k_scale,
        k_bf16,
        norm_weight_bf16,
        cos,
        sin,
        positions,
        slots,
        norm_epsilon,
        rope_dim=_ROPE_DIM,
        kv_block_size=_KV_BLOCK_SIZE,
        group_size=_GROUP_SIZE,
        shuffle_scale=True,
        do_rotate_act=True,
    )
