#!/usr/bin/env python3
"""Micro-simulator for concurrency=1 no-cache workloads using InferCast UMD."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from infercast.sdk import PerfDatabase, build_umd_static_model


def simulate_case(
    provider,
    *,
    isl: int,
    osl: int,
    runtime: dict,
) -> dict[str, float]:
    prefill_ms = float(
        provider.estimate_extend_forward_ms(
            batch_size=1,
            extend_len=isl,
            prefix_len=0,
            seq_imbalance_correction_scale=1.0,
            **runtime,
        )
    )
    decode_ms = [
        float(
            provider.estimate_decode_forward_ms(
                batch_size=1,
                history_len=isl + step,
                **runtime,
            )
        )
        for step in range(max(osl - 1, 0))
    ]
    tpot_ms = statistics.mean(decode_ms) if decode_ms else float("nan")
    duration_s = (prefill_ms + sum(decode_ms)) / 1000.0
    input_throughput = isl / duration_s if duration_s > 0 else 0.0
    return {
        "mean_ttft_ms": prefill_ms,
        "mean_tpot_ms": tpot_ms,
        "duration": duration_s,
        "input_throughput": input_throughput,
    }


def load_real(path: Path, concurrency: int = 1) -> dict:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if int(row.get("concurrency", -1)) == concurrency:
            return row
    raise ValueError(f"no concurrency={concurrency} row in {path}")


def pct_error(sim: float, real: float) -> float:
    if real == 0:
        return float("nan")
    return abs(sim - real) / real * 100.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems-root", type=Path, required=True)
    parser.add_argument("--system", default="mi355x")
    parser.add_argument("--version", default="0.5.17")
    parser.add_argument("--model-id", default="Qwen/Qwen3-32B-FP8")
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()

    db = PerfDatabase.open_fidb(
        args.system,
        "sglang",
        args.version,
        systems_root=str(args.systems_root),
        database_mode="SILICON",
    )
    provider = build_umd_static_model(db, args.model_id, backend="sglang")
    runtime = {
        "tp_size": args.tp_size,
        "pp_size": 1,
        "moe_tp_size": args.tp_size,
        "moe_ep_size": 1,
        "attention_dp_size": 1,
        "attn_kernel_impl": "cuda_graph",
        "attn_dtype": "bfloat16",
        "kv_cache_dtype": "fp8",
    }

    cases = [
        ("Qwen_512_512", 512, 512),
        ("Qwen_1024_512", 1024, 512),
        ("Qwen_1024_1024", 1024, 1024),
        ("Qwen_2048_1024", 2048, 1024),
    ]
    rows = []
    for name, isl, osl in cases:
        real_path = args.real_root / f"{name}-results.jsonl"
        if not real_path.is_file():
            real_path = args.real_root / name / "results.jsonl"
        real = load_real(real_path, args.concurrency)
        sim = simulate_case(provider, isl=isl, osl=osl, runtime=runtime)
        rows.append(
            {
                "case": name,
                "isl": isl,
                "osl": osl,
                "real_ttft_ms": real["ttft_ms_median"],
                "sim_ttft_ms": sim["mean_ttft_ms"],
                "ttft_err_pct": pct_error(sim["mean_ttft_ms"], real["ttft_ms_median"]),
                "real_tpot_ms": real["tpot_ms_median"],
                "sim_tpot_ms": sim["mean_tpot_ms"],
                "tpot_err_pct": pct_error(sim["mean_tpot_ms"], real["tpot_ms_median"]),
                "real_input_tps": real["out_tokens_total"] / real["wall_s"],
                "sim_input_tps": sim["input_throughput"],
                "input_tps_err_pct": pct_error(
                    sim["input_throughput"],
                    real["out_tokens_total"] / real["wall_s"],
                ),
                "real_duration_s": real["wall_s"],
                "sim_duration_s": sim["duration"],
                "duration_err_pct": pct_error(sim["duration"], real["wall_s"]),
            }
        )

    print(json.dumps(rows, indent=2))
    ttft_errs = [row["ttft_err_pct"] for row in rows]
    tpot_errs = [row["tpot_err_pct"] for row in rows]
    print(
        f"summary: ttft_err={min(ttft_errs):.2f}-{max(ttft_errs):.2f}% "
        f"tpot_err={min(tpot_errs):.2f}-{max(tpot_errs):.2f}%"
    )


if __name__ == "__main__":
    main()
