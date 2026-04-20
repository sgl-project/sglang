"""Generate per-mc heter (base) configs for the profile sweep.

For each mc in MC_LIST, computes the VRAM budget (KV reserve scales with
mc) and emits `configs/mc{mc}/{int4_only_experts, heter_config,
assignment_report}.json + mfs.txt` tailored to that SLO. Composite
scores are derived once from per-layer PPL + per-expert L2 sensitivity
and reused across mc levels.

KV sizing:
    * Default: worst-case envelope (max_prompt_len + max_output_len)
      scaled by ``BudgetKnobs.kv_reserve_frac``.
    * With ``--calib_json kv_calib/<task>.json`` (produced by
      ``run_calib.sh``): amortized ``μ + k·σ`` per request, using the
      task's measured ``total_len`` distribution. Much tighter and
      defensible for a fixed workload.

Pair with `gen_dyna_variants.py` (runtime dispatch variants per mc) and
`run_sweep.sh <task>` (efficiency or accuracy sweep).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
POLICY_DIR = THIS_DIR.parent / "policy" / "heter_assign"
SENS_DIR = THIS_DIR.parent / "sensitivity"

sys.path.insert(0, str(POLICY_DIR))
import vram_estimator as vest  # noqa: E402
from assign_experts import (  # noqa: E402
    _assign,
    _composite_scores,
    _load_model_config,
    _load_sensitivity_inputs,
    _write_outputs,
)

MC_LIST = [8, 16, 32, 64, 128, 256]

MODEL_PATH = (
    "/data/huggingface/hub/models--Qwen--Qwen3-30B-A3B/"
    "snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
)
INT4_CKPT = (
    "/data/huggingface/hub/models--Qwen--Qwen3-30B-A3B-GPTQ-Int4/"
    "snapshots/9b534e4318b7ebc3c961a839f13eb18b1833f441"
)
LAYER_SENS = SENS_DIR / "per_moe_layer" / "results" / "summary.json"
EXPERT_SENS = SENS_DIR / "per_expert" / "results" / "summary.json"

MAX_PROMPT_LEN = 2048
MAX_OUTPUT_LEN = 2048
GPU_INDEX = 0  # used only for torch.cuda.get_device_properties


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--task",
        help="Task name; when set, configs go to configs/<task>/mc{mc}/ "
             "instead of configs/mc{mc}/. Default keeps the flat layout.",
    )
    ap.add_argument(
        "--calib_json",
        type=Path,
        help="Path to a calib_kv.py output JSON (e.g. kv_calib/sharegpt.json). "
             "When its recommended_slo has mean_total_len + std_total_len, "
             "kv_bytes uses the amortized μ+k·σ branch.",
    )
    ap.add_argument(
        "--max_prompt_len", type=int, default=MAX_PROMPT_LEN,
        help="Worst-case envelope for prompt length (default %(default)d). "
             "Ignored when calib provides empirical stats.",
    )
    ap.add_argument(
        "--max_output_len", type=int, default=MAX_OUTPUT_LEN,
        help="Worst-case envelope for output length (default %(default)d). "
             "Ignored when calib provides empirical stats.",
    )
    return ap.parse_args()


def _load_calib(path: Path, logger: logging.Logger) -> dict:
    with open(path) as f:
        calib = json.load(f)
    rec = calib.get("recommended_slo", {}) or {}
    stats = {}
    if "mean_total_len" in rec and "std_total_len" in rec:
        stats["mean_total_len"] = float(rec["mean_total_len"])
        stats["std_total_len"] = float(rec["std_total_len"])
        stats["kv_headroom_sigmas"] = float(
            rec.get("kv_headroom_sigmas",
                    vest.BudgetKnobs.__dataclass_fields__[
                        "kv_headroom_sigmas"].default)
        )
        logger.info(
            "Calib loaded from %s: μ=%.1f σ=%.1f k=%.1f → per-req=%.1f tokens",
            path, stats["mean_total_len"], stats["std_total_len"],
            stats["kv_headroom_sigmas"],
            stats["mean_total_len"]
            + stats["kv_headroom_sigmas"] * stats["std_total_len"],
        )
    else:
        logger.warning(
            "Calib %s missing mean_total_len/std_total_len; falling back to "
            "worst-case KV.", path,
        )
    return stats


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logger = logging.getLogger(__name__)
    args = _parse_args()

    out_root = THIS_DIR / "configs"
    if args.task:
        out_root = out_root / args.task
    out_root.mkdir(parents=True, exist_ok=True)

    config = _load_model_config(MODEL_PATH)
    num_layers = config.num_hidden_layers
    num_experts = config.num_experts
    gpu_vram = torch.cuda.get_device_properties(GPU_INDEX).total_memory

    calib_stats: dict = {}
    knobs_kwargs: dict = {}
    if args.calib_json:
        calib_stats = _load_calib(args.calib_json, logger)
        if "kv_headroom_sigmas" in calib_stats:
            knobs_kwargs["kv_headroom_sigmas"] = calib_stats["kv_headroom_sigmas"]
    knobs = vest.BudgetKnobs(**knobs_kwargs)

    logger.info(
        "Model: H=%d I=%d L=%d num_experts=%d (%d total experts)",
        config.hidden_size, config.moe_intermediate_size,
        num_layers, num_experts, num_experts * num_layers,
    )

    ppl, l2 = _load_sensitivity_inputs(
        str(LAYER_SENS), str(EXPERT_SENS), num_layers, num_experts
    )
    scores = _composite_scores(ppl, l2, num_experts)

    logger.info("Sweeping max_concurrency ∈ %s (task=%s, out=%s)",
                MC_LIST, args.task or "<default>", out_root)
    summary = []
    for mc in MC_LIST:
        slo = vest.SLO(
            max_concurrency=mc,
            max_prompt_len=args.max_prompt_len,
            max_output_len=args.max_output_len,
            mean_total_len=calib_stats.get("mean_total_len"),
            std_total_len=calib_stats.get("std_total_len"),
        )
        budget = vest.compute_budget(config, gpu_vram, slo, knobs)
        logger.info("[mc=%d] VRAM breakdown:\n%s", mc, vest.format_budget(budget))

        k = budget.k_heter_experts
        if k == 0:
            logger.warning(
                "[mc=%d] bf16_budget=0; writing outputs with K=0 (all INT4).", mc
            )
        heter, int4_only = _assign(scores, k)

        mc_dir = out_root / f"mc{mc}"
        _write_outputs(
            out_dir=mc_dir,
            int4_only=int4_only,
            heter=heter,
            int4_checkpoint=INT4_CKPT,
            group_size=knobs.group_size,
            budget=budget,
            slo=slo,
            knobs=knobs,
            scores=scores,
            ppl=ppl,
            num_layers=num_layers,
        )
        logger.info(
            "[mc=%d] K=%d heter / %d int4-only → %s",
            mc, len(heter), len(int4_only), mc_dir,
        )
        summary.append((mc, len(heter), len(int4_only)))

    logger.info("Summary:")
    logger.info("  %-6s %-10s %-12s", "mc", "K_heter", "K_int4_only")
    for mc, kh, ki in summary:
        logger.info("  %-6d %-10d %-12d", mc, kh, ki)
    return 0


if __name__ == "__main__":
    sys.exit(main())
