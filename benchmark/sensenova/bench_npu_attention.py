"""Microbenchmark the NPU SDPA path used by SenseNova denoising."""

import argparse
import json
import math
import statistics
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-length", type=int, default=4096)
    parser.add_argument("--short-prefix-length", type=int, default=260)
    parser.add_argument("--long-prefix-length", type=int, default=286)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--atol", type=float, default=0.02)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument(
        "--native-fia",
        action="store_true",
        help="Also benchmark npu_fused_infer_attention_score with right-padded KV.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if not 0 < args.short_prefix_length < args.long_prefix_length:
        parser.error("prefix lengths must satisfy 0 < short < long")
    if args.query_length <= 0 or min(args.heads, args.kv_heads, args.head_dim) <= 0:
        parser.error("query length, heads, and head dimension must be positive")
    if args.heads % args.kv_heads:
        parser.error("heads must be divisible by kv-heads")
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("warmup must be nonnegative and iterations must be positive")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing result: {output}")

    import torch
    import torch_npu

    if not torch.npu.is_available():
        raise RuntimeError("No NPU is available")
    torch.npu.set_device(0)
    torch.manual_seed(0)
    torch.npu.manual_seed_all(0)

    device = torch.device("npu:0")
    dtype = torch.bfloat16
    batch_size = 2
    query_length = args.query_length
    key_length = args.long_prefix_length + query_length
    shape_q = (batch_size, args.heads, query_length, args.head_dim)
    shape_kv = (batch_size, args.kv_heads, key_length, args.head_dim)
    q = torch.randn(shape_q, device=device, dtype=dtype)
    k = torch.randn(shape_kv, device=device, dtype=dtype)
    v = torch.randn(shape_kv, device=device, dtype=dtype)

    all_true_mask = torch.ones(
        (batch_size, 1, query_length, key_length),
        device=device,
        dtype=torch.bool,
    )
    mixed_mask = all_true_mask.clone()
    mixed_mask[0, :, :, args.short_prefix_length : args.long_prefix_length] = False

    scale = 1.0 / math.sqrt(args.head_dim)

    def attention(q_, k_, v_, mask=None):
        if q_.shape[1] != k_.shape[1]:
            repeats = q_.shape[1] // k_.shape[1]
            k_ = k_.repeat_interleave(repeats, dim=1)
            v_ = v_.repeat_interleave(repeats, dim=1)
        return torch.nn.functional.scaled_dot_product_attention(
            q_,
            k_,
            v_,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
        )

    compact_k0 = torch.cat(
        (k[0:1, :, : args.short_prefix_length], k[0:1, :, args.long_prefix_length :]),
        dim=2,
    )
    compact_v0 = torch.cat(
        (v[0:1, :, : args.short_prefix_length], v[0:1, :, args.long_prefix_length :]),
        dim=2,
    )

    def two_compact_singletons():
        attention(q[0:1], compact_k0, compact_v0)
        attention(q[1:2], k[1:2], v[1:2])

    cases = {
        "b1_long_no_mask": lambda: attention(q[1:2], k[1:2], v[1:2]),
        "b2_no_mask": lambda: attention(q, k, v),
        "b2_all_true_mask": lambda: attention(q, k, v, all_true_mask),
        "b2_mixed_padding_mask": lambda: attention(q, k, v, mixed_mask),
        "two_compact_singletons": two_compact_singletons,
    }

    native_fia = None
    native_fia_left = None
    if args.native_fia:
        short_key_length = args.short_prefix_length + query_length
        right_padded_k = k.clone()
        right_padded_v = v.clone()
        right_padded_k[0, :, args.short_prefix_length : short_key_length].copy_(
            k[0, :, args.long_prefix_length :]
        )
        right_padded_v[0, :, args.short_prefix_length : short_key_length].copy_(
            v[0, :, args.long_prefix_length :]
        )
        right_padded_k[0, :, short_key_length:].zero_()
        right_padded_v[0, :, short_key_length:].zero_()
        actual_seq_lengths = [query_length] * batch_size
        actual_seq_lengths_kv = [
            short_key_length,
            args.long_prefix_length + query_length,
        ]

        def native_fia():
            result, _ = torch_npu.npu_fused_infer_attention_score(
                q,
                right_padded_k,
                right_padded_v,
                actual_seq_lengths=actual_seq_lengths,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                num_heads=args.heads,
                num_key_value_heads=args.kv_heads,
                scale=scale,
                input_layout="BNSD",
                sparse_mode=0,
            )
            return result

        cases["native_fia_b2_right_padded"] = native_fia

        left_padding = args.long_prefix_length - args.short_prefix_length
        left_padded_k = k.clone()
        left_padded_v = v.clone()
        left_padded_k[0, :, :left_padding].zero_()
        left_padded_v[0, :, :left_padding].zero_()
        left_padded_k[0, :, left_padding : args.long_prefix_length].copy_(
            k[0, :, : args.short_prefix_length]
        )
        left_padded_v[0, :, left_padding : args.long_prefix_length].copy_(
            v[0, :, : args.short_prefix_length]
        )
        kv_padding_size = torch.zeros(1, device=device, dtype=torch.int64)

        def native_fia_left():
            result, _ = torch_npu.npu_fused_infer_attention_score(
                q,
                left_padded_k,
                left_padded_v,
                actual_seq_lengths=actual_seq_lengths,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                kv_padding_size=kv_padding_size,
                num_heads=args.heads,
                num_key_value_heads=args.kv_heads,
                scale=scale,
                input_layout="BNSD",
                sparse_mode=0,
            )
            return result

        cases["native_fia_b2_left_padded"] = native_fia_left

    def measure(fn):
        for _ in range(args.warmup):
            fn()
        torch.npu.synchronize()
        samples = []
        for _ in range(args.iterations):
            start = time.perf_counter()
            fn()
            torch.npu.synchronize()
            samples.append((time.perf_counter() - start) * 1000)
        return {
            "median_ms": statistics.median(samples),
            "mean_ms": statistics.fmean(samples),
            "min_ms": min(samples),
            "max_ms": max(samples),
        }

    def error(reference, actual):
        diff = (reference.float() - actual.float()).abs()
        return {
            "max_abs": diff.max().item(),
            "mean_abs": diff.mean().item(),
            "allclose": torch.allclose(
                reference.float(), actual.float(), atol=args.atol, rtol=args.rtol
            ),
        }

    with torch.inference_mode():
        no_mask = attention(q, k, v)
        all_true = attention(q, k, v, all_true_mask)
        mixed = attention(q, k, v, mixed_mask)
        compact0 = attention(q[0:1], compact_k0, compact_v0)
        compact1 = attention(q[1:2], k[1:2], v[1:2])
        correctness = {
            "all_true_vs_no_mask": error(no_mask, all_true),
            "mixed_short_vs_compact": error(compact0, mixed[0:1]),
            "mixed_long_vs_compact": error(compact1, mixed[1:2]),
        }
        if native_fia is not None:
            native = native_fia()
            correctness["native_fia_short_vs_compact"] = error(compact0, native[0:1])
            correctness["native_fia_long_vs_compact"] = error(compact1, native[1:2])
            del native
        if native_fia_left is not None:
            native_left = native_fia_left()
            correctness["native_fia_left_short_vs_compact"] = error(
                compact0, native_left[0:1]
            )
            correctness["native_fia_left_long_vs_compact"] = error(
                compact1, native_left[1:2]
            )
            del native_left
        del no_mask, all_true, mixed, compact0, compact1

        measurements = {}
        for name, fn in cases.items():
            measurements[name] = measure(fn)
            print(json.dumps({"case": name, **measurements[name]}), flush=True)

    b1 = measurements["b1_long_no_mask"]["median_ms"]
    b2 = measurements["b2_no_mask"]["median_ms"]
    all_true = measurements["b2_all_true_mask"]["median_ms"]
    mixed = measurements["b2_mixed_padding_mask"]["median_ms"]
    compact = measurements["two_compact_singletons"]["median_ms"]
    derived = {
        "b2_throughput_speedup_vs_b1": 2 * b1 / b2,
        "b2_time_ratio_vs_b1": b2 / b1,
        "all_true_mask_overhead_pct": (all_true / b2 - 1) * 100,
        "mixed_mask_overhead_vs_no_mask_pct": (mixed / b2 - 1) * 100,
        "mixed_mask_overhead_vs_all_true_pct": (mixed / all_true - 1) * 100,
        "batched_mixed_speedup_vs_compact_singletons": compact / mixed,
    }
    if args.native_fia:
        native_right = measurements["native_fia_b2_right_padded"]["median_ms"]
        native_left = measurements["native_fia_b2_left_padded"]["median_ms"]
        derived.update(
            {
                "native_fia_right_speedup_vs_masked_sdpa": mixed / native_right,
                "native_fia_right_speedup_vs_unmasked_sdpa": b2 / native_right,
                "native_fia_right_b2_throughput_speedup_vs_b1": 2 * b1 / native_right,
                "native_fia_left_speedup_vs_masked_sdpa": mixed / native_left,
                "native_fia_left_speedup_vs_unmasked_sdpa": b2 / native_left,
                "native_fia_left_b2_throughput_speedup_vs_b1": 2 * b1 / native_left,
            }
        )
    result = {
        "config": {
            **vars(args),
            "dtype": str(dtype),
            "device": torch.npu.get_device_name(0),
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "q_shape": list(shape_q),
            "kv_shape": list(shape_kv),
        },
        "correctness": correctness,
        "measurements": measurements,
        "derived": derived,
    }
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"correctness": correctness, "derived": derived}, indent=2))
    print(f"Wrote {output}")

    if not all(item["allclose"] for item in correctness.values()):
        raise RuntimeError("Attention correctness check failed; see the JSON result")


if __name__ == "__main__":
    main()
