"""Generate per-mc heter (base) configs for the experiments sweep.

For each mc in MC_LIST, computes the VRAM budget (KV reserve scales with
mc) and emits `data/configs/mc{mc}/{int4_only_experts, heter_config,
assignment_report}.json + mfs.txt` tailored to that SLO. Composite
scores are derived once from per-layer PPL + per-expert L2 sensitivity
and reused across mc levels.

KV sizing:
    * Default: worst-case envelope (max_prompt_len + max_output_len)
      scaled by ``BudgetKnobs.kv_reserve_frac``.
    * With ``--calib_json data/kv_calib/<task>.json`` (produced by
      ``pipeline/kv_calib/run_calib.sh``): amortized ``μ + k·σ`` per
      request, using the task's measured ``total_len`` distribution.
      Much tighter and defensible for a fixed workload.

Pair with `gen_dyna_variants.py` (runtime dispatch variants per mc) and
`pipeline/run_sweep.sh <task>` (efficiency or accuracy sweep).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = THIS_DIR.parent.parent
ROOT_DIR = EXPERIMENTS_DIR.parent
POLICY_DIR = ROOT_DIR / "policy" / "heter_assign"
SENS_DIR = ROOT_DIR / "legacy" / "sensitivity"

sys.path.insert(0, str(POLICY_DIR))
import vram_estimator as vest  # noqa: E402
from assign_experts import (  # noqa: E402
    _assign,
    _composite_scores,
    _load_hessian_scores,
    _load_model_config,
    _load_sensitivity_inputs,
    _write_outputs,
)

# Default sweep targets moderate-context / moderate-batch serving (gsm8k,
# sharegpt). For long-context workloads (RULER), KV dominates and even mc=8
# can blow the budget — override with e.g. MC_LIST="1 2 4 8 16 32 64" to
# explore the low-concurrency range where heter-moe still has runway.
_DEFAULT_MC_LIST = [8, 16, 32, 64, 128, 256]
MC_LIST = (
    [int(x) for x in os.environ["MC_LIST"].split()]
    if os.environ.get("MC_LIST") else _DEFAULT_MC_LIST
)

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
HESSIAN_SCORES = ROOT_DIR / "hessian" / "results" / "hessian_scores.json"

MAX_PROMPT_LEN = 2048
MAX_OUTPUT_LEN = 2048
GPU_INDEX = 0  # used only for torch.cuda.get_device_properties


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--task",
        help="Task name; when set, configs go to data/configs/<task>/mc{mc}/ "
             "instead of data/configs/mc{mc}/. Default keeps the flat layout.",
    )
    ap.add_argument(
        "--calib_json",
        type=Path,
        help="Path to a calib_kv.py output JSON "
             "(e.g. data/kv_calib/sharegpt/calib.json). "
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

    ranking = ap.add_mutually_exclusive_group()
    ranking.add_argument(
        "--hessian", dest="ranking", action="store_const", const="hessian",
        help="Rank experts by per-expert ½·dᵀHd from hessian_scores.json "
             "(default). Signed score: positive = INT4 hurts; negative = "
             "INT4 is neutral or helpful.",
    )
    ranking.add_argument(
        "--sensitivity", dest="ranking", action="store_const", const="sensitivity",
        help="Legacy ranking: max(0, ppl_increase[L]) × L2[L][E] / ΣL2[L]. "
             "Requires sensitivity summaries in --layer_sens / --expert_sens.",
    )
    ap.set_defaults(ranking="hessian")

    ap.add_argument(
        "--hessian_path", type=Path, default=HESSIAN_SCORES,
        help="Path to hessian_scores.json (only used when --hessian).",
    )
    ap.add_argument(
        "--dry_run", action="store_true",
        help="Print the artifacts that would be written per mc (importance "
             "summaries, K, |fo|_mean) but do not create any files.",
    )

    fo = ap.add_mutually_exclusive_group()
    fo.add_argument(
        "--fo_threshold", dest="fo_threshold", action="store_true",
        help="Cap K at the number of experts whose hessian_score exceeds the "
             "mean |first_order_score| (default). Experts below this noise "
             "floor are statistically indistinguishable from zero — spending "
             "BF16 budget on them doesn't improve accuracy.",
    )
    fo.add_argument(
        "--no_fo_threshold", dest="fo_threshold", action="store_false",
        help="Disable the |fo|-mean cap: fill the full VRAM-budget K with top "
             "scores even when most are near-noise-floor. Useful for A/B.",
    )
    ap.set_defaults(fo_threshold=True)

    ap.add_argument(
        "--layer_sens", type=Path, default=LAYER_SENS,
        help="Per-layer PPL sensitivity summary (only used when --sensitivity).",
    )
    ap.add_argument(
        "--expert_sens", type=Path, default=EXPERT_SENS,
        help="Per-expert L2 sensitivity summary (only used when --sensitivity).",
    )
    ap.add_argument(
        "--attention_num_bits", type=int, choices=(16, 4), default=16,
        help="Embedded into the produced heter_config.json. 4 = the runtime "
             "swaps every layer's self_attn.qkv_proj+o_proj to INT4 GPTQ-Marlin "
             "at server load (reusing the INT4 group's checkpoint). 16 (default) "
             "leaves attention BF16.",
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

    out_root = EXPERIMENTS_DIR / "data" / "configs"
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

    fo_mean: float | None = None
    fo_cap: int | None = None
    expert_importance: dict | None = None
    if args.ranking == "hessian":
        logger.info("Ranking policy: hessian (%s)", args.hessian_path)
        scores, first_order = _load_hessian_scores(
            str(args.hessian_path), num_layers, num_experts
        )
        ppl: dict = {}
        npos = sum(1 for v in scores.values() if v > 0)
        nneg = sum(1 for v in scores.values() if v < 0)
        logger.info(
            "  %d experts with positive hessian (BF16-critical), "
            "%d negative (INT4-preferred), %d zero",
            npos, nneg, len(scores) - npos - nneg,
        )

        fo_abs = [abs(v) for v in first_order.values()]
        fo_mean = sum(fo_abs) / len(fo_abs) if fo_abs else 0.0
        fo_cap = sum(1 for v in scores.values() if v > fo_mean)
        logger.info(
            "  |fo|_mean=%.3e → %d experts above noise floor "
            "(fo_threshold=%s)",
            fo_mean, fo_cap, args.fo_threshold,
        )

        expert_importance = {}
        fallback_layers: list[int] = []
        zero_counts: list[int] = []
        for L in range(num_layers):
            row = [
                (scores[(L, E)] if scores[(L, E)] > fo_mean else 0.0)
                for E in range(num_experts)
            ]
            zeros = sum(1 for v in row if v == 0.0)
            if zeros == num_experts:
                row = [1.0] * num_experts
                fallback_layers.append(L)
            expert_importance[L] = row
            zero_counts.append(zeros)
        above = [num_experts - z for z in zero_counts]
        logger.info(
            "  importance: per-layer #above fo_mean min/med/max="
            "%d/%d/%d; %d layers fell back to all-ones",
            min(above), sorted(above)[len(above) // 2], max(above),
            len(fallback_layers),
        )
    else:
        if args.fo_threshold and args.ranking != "hessian":
            logger.info(
                "fo_threshold has no effect with --sensitivity ranking; ignored."
            )
        logger.info(
            "Ranking policy: sensitivity (layer=%s expert=%s)",
            args.layer_sens, args.expert_sens,
        )
        ppl, l2 = _load_sensitivity_inputs(
            str(args.layer_sens), str(args.expert_sens),
            num_layers, num_experts,
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

        budget_k = budget.k_heter_experts
        if budget_k == 0:
            logger.warning(
                "[mc=%d] bf16_budget=0; writing outputs with K=0 (all INT4).", mc
            )

        fo_report: dict | None = None
        if args.ranking == "hessian" and args.fo_threshold:
            k = min(budget_k, fo_cap)
            fo_report = {
                "enabled": True,
                "fo_mean": fo_mean,
                "fo_cap": fo_cap,
                "budget_k": budget_k,
                "effective_k": k,
                "capped": k < budget_k,
            }
            if k < budget_k:
                logger.info(
                    "[mc=%d] capping K: budget=%d, |fo|-above=%d → K=%d",
                    mc, budget_k, fo_cap, k,
                )
        else:
            k = budget_k
            if args.ranking == "hessian":
                fo_report = {
                    "enabled": False,
                    "fo_mean": fo_mean,
                    "fo_cap": fo_cap,
                    "budget_k": budget_k,
                    "effective_k": k,
                    "capped": False,
                }

        heter, int4_only = _assign(scores, k)

        mc_dir = out_root / f"mc{mc}"
        if args.dry_run:
            planned = [
                "int4_only_experts.json",
                "heter_config.json",
                "assignment_report.json",
            ]
            if expert_importance is not None:
                planned.append("expert_importance.json")
            logger.info(
                "[mc=%d] DRY-RUN would write to %s: %s (K=%d heter / %d int4-only)",
                mc, mc_dir, ", ".join(planned), len(heter), len(int4_only),
            )
        else:
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
                ranking_policy=args.ranking,
                fo_threshold=fo_report,
                expert_importance=expert_importance,
                attention_num_bits=args.attention_num_bits,
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
