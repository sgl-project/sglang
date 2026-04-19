"""Sweep (seq_len, max_batch_size) -> % heter experts (K / total).

Uses vram_estimator to compute the BF16 expert budget for each combo and
prints a table. `seq_len` is treated as max_prompt_len + max_output_len
(KV reservation only uses the sum).
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import vram_estimator as vest


def _resolve_hf_path(path: str) -> str:
    m = re.match(r"^(.+)/hub/models--(.+?)--(.+?)/snapshots/[a-f0-9]+$", path)
    if m:
        os.environ.setdefault("HF_HOME", m.group(1))
        return f"{m.group(2)}/{m.group(3)}"
    return path


def _load_model_config(model_path: str):
    from transformers import AutoConfig
    return AutoConfig.from_pretrained(
        _resolve_hf_path(model_path), trust_remote_code=True, local_files_only=True
    )


def _parse_ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model_path",
        default="/data/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/"
                "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
    )
    ap.add_argument("--gpu_vram_gb", type=float, default=80.0)
    ap.add_argument(
        "--seq_lens", type=_parse_ints,
        default=[1024, 2048, 4096, 8192, 16384, 32768],
    )
    ap.add_argument(
        "--batch_sizes", type=_parse_ints,
        default=[1, 4, 8, 16, 32, 64, 128, 256],
    )
    ap.add_argument("--kv_reserve_frac", type=float, default=0.5)
    ap.add_argument("--headroom_gb", type=float, default=2.0)
    ap.add_argument("--headroom_frac", type=float, default=0.05)
    ap.add_argument("--prefill_budget_tokens", type=int, default=16384)
    ap.add_argument("--group_size", type=int, default=128)
    ap.add_argument(
        "--mode", choices=["pct", "k"], default="pct",
        help="Cell content: percent (x/total *100) or raw K.",
    )
    args = ap.parse_args()

    config = _load_model_config(args.model_path)
    total = config.num_experts * config.num_hidden_layers
    gpu_vram_bytes = int(args.gpu_vram_gb * (1 << 30))
    knobs = vest.BudgetKnobs(
        kv_reserve_frac=args.kv_reserve_frac,
        headroom_gb=args.headroom_gb,
        headroom_frac=args.headroom_frac,
        prefill_budget_tokens=args.prefill_budget_tokens,
        group_size=args.group_size,
    )

    # Header
    print(
        f"Model: {args.model_path.rsplit('/', 1)[-1] if '/' not in args.model_path else _resolve_hf_path(args.model_path)}  "
        f"L={config.num_hidden_layers} E={config.num_experts} total={total}  "
        f"VRAM={args.gpu_vram_gb:.0f}GB  kv_reserve_frac={args.kv_reserve_frac}"
    )
    print(f"Cell = K / {total} (% heter experts)" if args.mode == "pct"
          else f"Cell = K (out of {total})")
    print()

    col_w = 8
    header = "seq_len \\ bs".ljust(14) + "".join(
        f"{bs:>{col_w}}" for bs in args.batch_sizes
    )
    print(header)
    print("-" * len(header))

    for seq_len in args.seq_lens:
        row = f"{seq_len:<14}"
        for bs in args.batch_sizes:
            slo = vest.SLO(
                max_concurrency=bs,
                max_prompt_len=seq_len,
                max_output_len=0,
            )
            b = vest.compute_budget(config, gpu_vram_bytes, slo, knobs)
            if args.mode == "pct":
                pct = 100.0 * b.k_heter_experts / total
                cell = f"{pct:6.1f}%"
            else:
                cell = f"{b.k_heter_experts}"
            row += f"{cell:>{col_w}}"
        print(row)

    return 0


if __name__ == "__main__":
    sys.exit(main())
