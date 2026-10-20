"""Shared feature contract for ML latency prediction and offline training."""

import math
from collections.abc import Sequence

ML_FEATURE_NAMES = (
    "batch_size",
    "sum_extend",
    "max_extend",
    "min_extend",
    "sum_past",
    "max_past",
    "min_past",
    "sum_extend_x_past",
    "sum_extend_squared",
    "sum_past_squared",
    "sum_attn_flops",
    "sum_extend_x_max_past",
    "log1p_sum_past",
    "log1p_sum_attn_flops",
    "batch_size_x_sum_extend",
    "max_past_minus_min_past",
    "is_decode",
    "is_prefill",
)


def extract_ml_features(
    extend_lengths: Sequence[int], past_kv_lengths: Sequence[int]
) -> list[float]:
    """Build the ordered 18-feature vector consumed by ``MLTimePredictor``."""
    if not extend_lengths:
        raise ValueError("cannot extract ML features from an empty batch")
    if len(extend_lengths) != len(past_kv_lengths):
        raise ValueError("extend and past-KV length vectors must have equal size")
    if any(value < 0 for value in (*extend_lengths, *past_kv_lengths)):
        raise ValueError("token lengths must be non-negative")

    batch_size = len(extend_lengths)
    sum_extend = sum(extend_lengths)
    sum_past = sum(past_kv_lengths)
    sum_extend_x_past = sum(
        extend * past for extend, past in zip(extend_lengths, past_kv_lengths)
    )
    sum_extend_squared = sum(value * value for value in extend_lengths)
    sum_past_squared = sum(value * value for value in past_kv_lengths)
    sum_attn_flops = sum(
        extend * (past + extend / 2)
        for extend, past in zip(extend_lengths, past_kv_lengths)
    )
    max_extend = max(extend_lengths)
    max_past = max(past_kv_lengths)
    min_extend = min(extend_lengths)
    min_past = min(past_kv_lengths)

    return [
        batch_size,
        sum_extend,
        max_extend,
        min_extend,
        sum_past,
        max_past,
        min_past,
        sum_extend_x_past,
        sum_extend_squared,
        sum_past_squared,
        sum_attn_flops,
        sum_extend * max_past,
        math.log1p(sum_past),
        math.log1p(sum_attn_flops),
        batch_size * sum_extend,
        max_past - min_past,
        int(all(value == 1 for value in extend_lengths)),
        int(any(value > 1 for value in extend_lengths)),
    ]
