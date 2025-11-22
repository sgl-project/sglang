"""
Helper script to launch python/sglang/auto_tune.py for multiple configs.

Fill in the model names below, then run:

    python scripts/auto_tune.py

Each config is constructed by zipping per-field lists (model, tp, ep, …) so
you can edit each dimension independently. Placeholders are skipped so you can
keep this file checked in without accidental runs.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List


MODELS: List[str] = [
    "deepseek-ai/DeepSeek-V3",
    "deepseek-ai/DeepSeek-R1",  
]
TP_SIZES: List[int] = [1, 2, 4, 8]
EP_SIZES: List[int] = [1, 2, 4, 8]
DTYPES: List[str] = ["fp8_w8a8", "int8_w8a8", "int8_w8a16"]
PER_CHANNEL_QUANT: List[bool] = [True, False]
BATCH_SIZES: List[int] = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 1536, 2048, 3072, 4096]
SEEDS: List[int] = [0]
DISABLE_SHARED_EXPERTS_FUSION: List[bool] = [True, False]
NUM_ITERS: List[int] = [10]


def build_cmd(
    *,
    model: str,
    tp: int,
    ep: int,
    dtype: str,
    per_channel_quant: bool,
    batch_size: int | None,
    seed: int,
    disable_shared_experts_fusion: bool,
    num_iters: int,
    auto_tune_py: Path,
) -> list[str]:
    cmd = [sys.executable, str(auto_tune_py)]
    cmd += ["--model", str(model)]
    cmd += ["--tp-size", str(tp)]
    cmd += ["--ep-size", str(ep)]
    cmd += ["--dtype", str(dtype)]
    if per_channel_quant:
        cmd.append("--per-channel-quant")
    if batch_size is not None:
        cmd += ["--batch-size", str(batch_size)]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if disable_shared_experts_fusion:
        cmd.append("--disable-shared-experts-fusion")
    cmd += ["--num-iters", str(num_iters)]
    return cmd


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    auto_tune_py = repo_root / "python" / "sglang" / "auto_tune.py"

    lengths = {len(MODELS), len(TP_SIZES), len(EP_SIZES), len(DTYPES), len(PER_CHANNEL_QUANT), len(BATCH_SIZES), len(SEEDS), len(DISABLE_SHARED_EXPERTS_FUSION), len(NUM_ITERS)}
    if len(lengths) != 1:
        raise SystemExit("All config lists must have the same length.")

    for idx, (model, tp, ep, dtype, pcq, batch, seed, disable_fusion, num_iters) in enumerate(
        zip(
            MODELS,
            TP_SIZES,
            EP_SIZES,
            DTYPES,
            PER_CHANNEL_QUANT,
            BATCH_SIZES,
            SEEDS,
            DISABLE_SHARED_EXPERTS_FUSION,
            NUM_ITERS,
        )
    ):
        if not model or str(model).startswith("FILL_ME_MODEL"):
            print(f"[skip] Config {idx} has placeholder model; edit MODELS first.")
            continue

        cmd = build_cmd(
            model=model,
            tp=tp,
            ep=ep,
            dtype=dtype,
            per_channel_quant=pcq,
            batch_size=batch,
            seed=seed,
            disable_shared_experts_fusion=disable_fusion,
            num_iters=num_iters,
            auto_tune_py=auto_tune_py,
        )

        print(f"[run ] Config {idx}: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"[fail] Config {idx} exited with {exc.returncode}")
            continue
        print(f"[done] Config {idx} completed\n")


if __name__ == "__main__":
    main()
