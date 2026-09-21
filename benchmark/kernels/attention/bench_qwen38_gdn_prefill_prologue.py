"""Qwen3.8 TP1 GDN prefill-prologue component and launch benchmark.

Run only on an isolated gfx950 device. This reports producer-only latency plus
the existing Q/K L2Norm suffix, then stops before the rest of the unchanged
chunked gated-delta scan.
"""

import argparse
import itertools
import json
from collections.abc import Callable

import torch
import triton

from sglang.kernels.ops.attention.fla.fused_gdn_gating import fused_gdn_gating
from sglang.kernels.ops.attention.fla.l2norm import l2norm_fwd
from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
    fused_qkvzba_split_reshape_cat_contiguous,
    qwen3_5_gdn_prefill_projection_views,
)
from sglang.kernels.ops.mamba.causal_conv1d_triton import (
    QWEN38_GDN_HEAD_DIM,
    QWEN38_GDN_NUM_QK_HEADS,
    QWEN38_GDN_NUM_V_HEADS,
    QWEN38_GDN_QKV_DIM,
    causal_conv1d_fn,
    qwen38_gdn_prefill_prologue,
)

QKVZ_DIM = QWEN38_GDN_QKV_DIM + QWEN38_GDN_NUM_V_HEADS * QWEN38_GDN_HEAD_DIM
BA_DIM = 2 * QWEN38_GDN_NUM_V_HEADS


def _sequence_lengths(tokens: int, num_sequences: int) -> list[int]:
    quotient, remainder = divmod(tokens, num_sequences)
    if quotient == 0:
        raise ValueError("tokens must be at least num_sequences")
    return [
        quotient + (sequence_idx < remainder) for sequence_idx in range(num_sequences)
    ]


def _cu_seqlens(lengths: list[int]) -> torch.Tensor:
    return torch.tensor(
        [0, *itertools.accumulate(lengths)],
        dtype=torch.int32,
        device="cuda",
    )


def _legacy_unpack(
    qkvz: torch.Tensor,
    ba: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q_dim = QWEN38_GDN_NUM_QK_HEADS * QWEN38_GDN_HEAD_DIM
    v_dim = QWEN38_GDN_NUM_V_HEADS * QWEN38_GDN_HEAD_DIM
    query, key, value, z = qkvz.split([q_dim, q_dim, v_dim, v_dim], dim=-1)
    raw_b, raw_a = ba.split(
        [QWEN38_GDN_NUM_V_HEADS, QWEN38_GDN_NUM_V_HEADS],
        dim=-1,
    )
    mixed_qkv = torch.cat(
        (query.reshape(query.shape[0], -1), key.reshape(key.shape[0], -1), value),
        dim=-1,
    )
    return (
        mixed_qkv,
        z.view(z.shape[0], QWEN38_GDN_NUM_V_HEADS, QWEN38_GDN_HEAD_DIM),
        raw_b.contiguous(),
        raw_a.contiguous(),
    )


def _materialize_qkv(
    mixed_qkv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = mixed_qkv.shape[0]
    qk_dim = QWEN38_GDN_NUM_QK_HEADS * QWEN38_GDN_HEAD_DIM
    q = (
        mixed_qkv[:, :qk_dim]
        .reshape(1, tokens, QWEN38_GDN_NUM_QK_HEADS, QWEN38_GDN_HEAD_DIM)
        .contiguous()
    )
    k = (
        mixed_qkv[:, qk_dim : 2 * qk_dim]
        .reshape(1, tokens, QWEN38_GDN_NUM_QK_HEADS, QWEN38_GDN_HEAD_DIM)
        .contiguous()
    )
    v = (
        mixed_qkv[:, 2 * qk_dim :]
        .reshape(1, tokens, QWEN38_GDN_NUM_V_HEADS, QWEN38_GDN_HEAD_DIM)
        .contiguous()
    )
    return q, k, v


def _bench(fn: Callable[[], object], warmup: int, repetitions: int) -> float:
    return float(
        triton.testing.do_bench(
            fn,
            warmup=warmup,
            rep=repetitions,
            return_mode="median",
        )
    )


def _count_device_events(fn: Callable[[], object]) -> tuple[int, list[str]]:
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    names = [
        event.name
        for event in prof.events()
        if getattr(event, "device_type", None) == torch.autograd.DeviceType.CUDA
    ]
    return len(names), names


def _run_shape(
    tokens: int,
    num_sequences: int,
    *,
    warmup: int,
    repetitions: int,
    profile_launches: bool,
) -> dict[str, object]:
    lengths = _sequence_lengths(tokens, num_sequences)
    qkvz = torch.randn(
        tokens,
        QKVZ_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    )
    ba = torch.randn(tokens, BA_DIM, dtype=torch.bfloat16, device="cuda")
    weight = (
        torch.randn(
            QWEN38_GDN_QKV_DIM,
            4,
            dtype=torch.bfloat16,
            device="cuda",
        )
        * 0.125
    )
    bias = (
        torch.randn(
            QWEN38_GDN_QKV_DIM,
            dtype=torch.bfloat16,
            device="cuda",
        )
        * 0.125
    )
    query_start_loc = _cu_seqlens(lengths)
    cache_indices = torch.arange(
        num_sequences,
        dtype=torch.int32,
        device="cuda",
    )
    has_initial_state = torch.tensor(
        [(idx & 1) != 0 for idx in range(num_sequences)],
        dtype=torch.bool,
        device="cuda",
    )
    A_log = torch.randn(
        QWEN38_GDN_NUM_V_HEADS,
        dtype=torch.float32,
        device="cuda",
    )
    dt_bias = torch.randn(
        QWEN38_GDN_NUM_V_HEADS,
        dtype=torch.bfloat16,
        device="cuda",
    )

    legacy_mixed, _, legacy_b, legacy_a = _legacy_unpack(qkvz, ba)
    legacy_conv_state = torch.randn(
        num_sequences,
        QWEN38_GDN_QKV_DIM,
        3,
        dtype=torch.bfloat16,
        device="cuda",
    )
    fused_conv_state = legacy_conv_state.clone()
    legacy_post_conv = causal_conv1d_fn(
        legacy_mixed.transpose(0, 1),
        weight,
        bias,
        conv_states=legacy_conv_state,
        query_start_loc=query_start_loc,
        seq_lens_cpu=lengths,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
    ).transpose(0, 1)
    legacy_q, legacy_k, _ = _materialize_qkv(legacy_post_conv)

    def legacy_producer_total():
        mixed_qkv, _, raw_b, raw_a = _legacy_unpack(qkvz, ba)
        post_conv = causal_conv1d_fn(
            mixed_qkv.transpose(0, 1),
            weight,
            bias,
            conv_states=legacy_conv_state,
            query_start_loc=query_start_loc,
            seq_lens_cpu=lengths,
            cache_indices=cache_indices,
            has_initial_state=has_initial_state,
        ).transpose(0, 1)
        q, k, v = _materialize_qkv(post_conv)
        return q, k, v, *fused_gdn_gating(A_log, raw_a, raw_b, dt_bias)

    def fused_producer_total():
        mixed_qkv, _, raw_b, raw_a = qwen3_5_gdn_prefill_projection_views(
            qkvz,
            ba,
            QWEN38_GDN_NUM_QK_HEADS,
            QWEN38_GDN_NUM_V_HEADS,
            QWEN38_GDN_HEAD_DIM,
            QWEN38_GDN_HEAD_DIM,
        )
        return qwen38_gdn_prefill_prologue(
            mixed_qkv.transpose(0, 1),
            weight,
            bias,
            fused_conv_state,
            query_start_loc,
            lengths,
            cache_indices,
            has_initial_state,
            A_log,
            raw_a,
            raw_b,
            dt_bias,
        )

    def legacy_with_qk_norm():
        q, k, v, g, beta = legacy_producer_total()
        return l2norm_fwd(q), l2norm_fwd(k), v, g, beta

    def fused_with_qk_norm():
        q, k, v, g, beta = fused_producer_total()
        return l2norm_fwd(q), l2norm_fwd(k), v, g, beta

    components = {
        "legacy_unpack": lambda: _legacy_unpack(qkvz, ba),
        "fused_unpack": lambda: fused_qkvzba_split_reshape_cat_contiguous(
            qkvz,
            ba,
            QWEN38_GDN_NUM_QK_HEADS,
            QWEN38_GDN_NUM_V_HEADS,
            QWEN38_GDN_HEAD_DIM,
            QWEN38_GDN_HEAD_DIM,
        ),
        "projection_views": lambda: qwen3_5_gdn_prefill_projection_views(
            qkvz,
            ba,
            QWEN38_GDN_NUM_QK_HEADS,
            QWEN38_GDN_NUM_V_HEADS,
            QWEN38_GDN_HEAD_DIM,
            QWEN38_GDN_HEAD_DIM,
        ),
        "causal_conv": lambda: causal_conv1d_fn(
            legacy_mixed.transpose(0, 1),
            weight,
            bias,
            conv_states=legacy_conv_state,
            query_start_loc=query_start_loc,
            seq_lens_cpu=lengths,
            cache_indices=cache_indices,
            has_initial_state=has_initial_state,
        ),
        "qkv_materialize": lambda: _materialize_qkv(legacy_post_conv),
        "q_l2norm": lambda: l2norm_fwd(legacy_q),
        "k_l2norm": lambda: l2norm_fwd(legacy_k),
        "gating": lambda: fused_gdn_gating(A_log, legacy_a, legacy_b, dt_bias),
        "legacy_producer_total": legacy_producer_total,
        "fused_producer_total": fused_producer_total,
        "legacy_with_qk_norm": legacy_with_qk_norm,
        "fused_with_qk_norm": fused_with_qk_norm,
    }
    latency_ms = {
        name: _bench(fn, warmup, repetitions) for name, fn in components.items()
    }
    launches = {
        "legacy_producer_expected": 8,
        "fused_producer_expected": 1,
        "legacy_with_qk_norm_expected": 10,
        "fused_with_qk_norm_expected": 3,
    }
    launch_names = {}
    if profile_launches:
        launches["legacy_producer_profiled"], launch_names["legacy_producer"] = (
            _count_device_events(legacy_producer_total)
        )
        launches["fused_producer_profiled"], launch_names["fused_producer"] = (
            _count_device_events(fused_producer_total)
        )
        (
            launches["legacy_with_qk_norm_profiled"],
            launch_names["legacy_with_qk_norm"],
        ) = _count_device_events(legacy_with_qk_norm)
        launches["fused_with_qk_norm_profiled"], launch_names["fused_with_qk_norm"] = (
            _count_device_events(fused_with_qk_norm)
        )
    return {
        "tokens": tokens,
        "num_sequences": num_sequences,
        "sequence_lengths": lengths,
        "latency_ms": latency_ms,
        "launches": launches,
        "launch_names": launch_names,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[8192, 10848, 13312, 16384],
    )
    parser.add_argument(
        "--num-sequences",
        type=int,
        nargs="+",
        default=[1, 4, 16],
    )
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--profile-launches", action="store_true")
    parser.add_argument("--output-json")
    args = parser.parse_args()

    if torch.version.hip is None:
        raise RuntimeError("This benchmark requires ROCm")
    results = [
        _run_shape(
            tokens,
            num_sequences,
            warmup=args.warmup,
            repetitions=args.repetitions,
            profile_launches=args.profile_launches,
        )
        for tokens in args.tokens
        for num_sequences in args.num_sequences
    ]
    payload = {
        "device": torch.cuda.get_device_name(),
        "arch": torch.cuda.get_device_properties(0).gcnArchName,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "results": results,
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as output:
            output.write(rendered)
            output.write("\n")


if __name__ == "__main__":
    main()
