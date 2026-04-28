"""Generate per-mc heter (base) configs for the experiments sweep.

For each mc in MC_LIST, computes the VRAM budget (KV reserve scales with
mc) and emits ``data/configs/mc{mc}/{int4_only_experts, heter_config,
expert_importance, assignment_report}.json + mfs.txt`` tailored to that
SLO.

Three ranking policies select which (layer, expert) pairs go BF16:

  --hessian_importance (default)
      Top-K signed Hessian (½·dᵀHd) capped at the |first-order|-mean
      noise floor. Always uses the floor cap; experts below it are
      statistically indistinguishable from zero.

  --activation_frequency
      Pure routed-token-count ranking; top-K by token_count globally.
      No noise-floor cap — fills the full VRAM budget.

  --hybrid
      Top fo_cap by Hessian, then fill the leftover VRAM with the
      next-highest experts by token_count (drawn from the experts NOT
      already chosen by Hessian).

KV sizing (orthogonal to ranking):
    * Default: worst-case envelope (max_prompt_len + max_output_len)
      scaled by ``BudgetKnobs.kv_reserve_frac``.
    * With ``--calib_json data/kv_calib/<task>.json`` (produced by
      ``pipeline/kv_calib/run_calib.sh``): amortized ``μ + k·σ`` per
      request, using the task's measured ``total_len`` distribution.

Pair with ``gen_dyna_variants.py`` (runtime dispatch variants per mc) and
``pipeline/run_sweep.sh <task>`` (efficiency or accuracy sweep).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    _assign_hybrid,
    _load_hessian_scores,
    _load_model_config,
    _load_token_counts,
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
HESSIAN_SCORES = ROOT_DIR / "hessian" / "results" / "hessian_scores.json"
TOKEN_COUNTS = SENS_DIR / "per_expert" / "results" / "summary.json"

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
        help="Path to a calib_kv.py output JSON. When its recommended_slo "
             "has mean_total_len + std_total_len, kv_bytes uses the "
             "amortized μ+k·σ branch.",
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
        "--hessian_importance", dest="ranking",
        action="store_const", const="hessian_importance",
        help="(default) Rank by signed Hessian; cap K at experts above the "
             "|first-order|-mean noise floor.",
    )
    ranking.add_argument(
        "--activation_frequency", dest="ranking",
        action="store_const", const="activation_frequency",
        help="Rank purely by routed-token count; fill the full VRAM budget.",
    )
    ranking.add_argument(
        "--hybrid", dest="ranking",
        action="store_const", const="hybrid",
        help="Top fo_cap by Hessian, then fill the leftover VRAM (budget_k − "
             "fo_cap) with the highest-token-count experts among the rest.",
    )
    ap.set_defaults(ranking="hessian_importance")

    ap.add_argument(
        "--hessian_path", type=Path, default=HESSIAN_SCORES,
        help="hessian_scores.json (used by --hessian_importance and --hybrid).",
    )
    ap.add_argument(
        "--token_count_path", type=Path, default=TOKEN_COUNTS,
        help="per-expert summary with token_count (used by "
             "--activation_frequency and --hybrid).",
    )
    ap.add_argument(
        "--dry_run", action="store_true",
        help="Print the artifacts that would be written per mc (importance "
             "summaries, K, |fo|_mean) but do not create any files.",
    )

    ap.add_argument(
        "--attention_num_bits", type=int, choices=(16, 4), default=16,
        help="Embedded into the produced heter_config.json. 4 = the runtime "
             "swaps every layer's self_attn.qkv_proj+o_proj to INT4 GPTQ-Marlin "
             "at server load (reusing the INT4 group's checkpoint). 16 (default) "
             "leaves attention BF16.",
    )
    ap.add_argument(
        "--bf16_promotion_threshold", type=int, required=True,
        help="Required. Routed-token count above which the runtime promotes "
             "an expert to BF16 (universal, applies on top of any scoring "
             "policy). Profile-derived for the target GPU/kernel stack: e.g. "
             "Qwen3-30B-A3B GPTQ-Marlin INT4 vs BF16 crossover sits at ~72 "
             "on A100. Embedded as a top-level field in the produced "
             "heter_config.json.",
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


def _build_importance_hessian(
    scores: Dict[Tuple[int, int], float],
    fo_mean: float,
    num_layers: int,
    num_experts: int,
    logger: logging.Logger,
) -> Dict[int, List[float]]:
    """Per-layer importance row: hessian where above fo_mean, else 0.

    Layers with no above-floor experts fall back to all-ones (the runtime
    dispatch policy can't sort by an all-zero importance vector).
    """
    importance: Dict[int, List[float]] = {}
    fallback_layers: List[int] = []
    above_per_layer: List[int] = []
    for L in range(num_layers):
        row = [
            (scores[(L, E)] if scores[(L, E)] > fo_mean else 0.0)
            for E in range(num_experts)
        ]
        zeros = sum(1 for v in row if v == 0.0)
        if zeros == num_experts:
            row = [1.0] * num_experts
            fallback_layers.append(L)
        importance[L] = row
        above_per_layer.append(num_experts - zeros)
    logger.info(
        "  importance: per-layer #above fo_mean min/med/max=%d/%d/%d; "
        "%d layers fell back to all-ones",
        min(above_per_layer),
        sorted(above_per_layer)[len(above_per_layer) // 2],
        max(above_per_layer),
        len(fallback_layers),
    )
    return importance


def _build_importance_tokencount(
    token_counts: Dict[Tuple[int, int], int],
    num_layers: int,
    num_experts: int,
    logger: logging.Logger,
) -> Dict[int, List[float]]:
    """Per-layer importance row = token_count.

    Layers with all-zero token counts fall back to all-ones so runtime
    dispatch can still order experts within that layer.
    """
    importance: Dict[int, List[float]] = {}
    fallback_layers: List[int] = []
    for L in range(num_layers):
        row = [float(token_counts[(L, E)]) for E in range(num_experts)]
        if sum(row) == 0.0:
            row = [1.0] * num_experts
            fallback_layers.append(L)
        importance[L] = row
    logger.info(
        "  importance: token_count rows; %d layers fell back to all-ones",
        len(fallback_layers),
    )
    return importance


def _build_importance_hybrid(
    hessian_scores: Dict[Tuple[int, int], float],
    token_counts: Dict[Tuple[int, int], int],
    fo_mean: float,
    heter_pairs: List[Tuple[int, int]],
    kind_by_pair: Dict[Tuple[int, int], str],
    num_layers: int,
    num_experts: int,
    logger: logging.Logger,
) -> Dict[int, List[float]]:
    """Per-layer importance for hybrid: Hessian-chosen experts get their
    Hessian score; gap-fill experts get token_count rescaled per-layer to
    sit strictly below the smallest Hessian-chosen score in that layer
    (so Hessian-critical experts always win contested promotions); other
    experts get 0.

    A layer with neither Hessian-chosen nor gap-fill experts falls back
    to all-ones.
    """
    importance: Dict[int, List[float]] = {}
    fallback_layers: List[int] = []
    by_layer: Dict[int, List[Tuple[int, str]]] = {L: [] for L in range(num_layers)}
    for (L, E) in heter_pairs:
        by_layer[L].append((E, kind_by_pair[(L, E)]))

    for L in range(num_layers):
        row = [0.0] * num_experts
        hess_experts = [E for E, k in by_layer[L] if k == "hessian"]
        fill_experts = [E for E, k in by_layer[L] if k == "tokencount"]

        for E in hess_experts:
            row[E] = hessian_scores[(L, E)]

        if fill_experts:
            tc_max = max(token_counts[(L, E)] for E in fill_experts)
            if hess_experts:
                h_min = min(row[E] for E in hess_experts)
                # Strictly below the lowest Hessian-chosen value in this layer.
                ceiling = max(0.0, h_min) * 0.99
            else:
                ceiling = 1.0
            scale = ceiling / tc_max if tc_max > 0 else 0.0
            for E in fill_experts:
                row[E] = scale * token_counts[(L, E)]

        if not hess_experts and not fill_experts:
            row = [1.0] * num_experts
            fallback_layers.append(L)

        importance[L] = row

    logger.info(
        "  importance: hybrid rows; %d layers fell back to all-ones",
        len(fallback_layers),
    )
    return importance


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

    # Load whichever signals the chosen policy needs.
    hessian_scores: Optional[Dict[Tuple[int, int], float]] = None
    first_order: Optional[Dict[Tuple[int, int], float]] = None
    token_counts: Optional[Dict[Tuple[int, int], int]] = None
    fo_mean: Optional[float] = None
    fo_cap: Optional[int] = None

    if args.ranking in ("hessian_importance", "hybrid"):
        logger.info("Loading hessian scores: %s", args.hessian_path)
        hessian_scores, first_order = _load_hessian_scores(
            str(args.hessian_path), num_layers, num_experts
        )
        npos = sum(1 for v in hessian_scores.values() if v > 0)
        nneg = sum(1 for v in hessian_scores.values() if v < 0)
        logger.info(
            "  %d experts with positive hessian (BF16-critical), "
            "%d negative (INT4-preferred), %d zero",
            npos, nneg, len(hessian_scores) - npos - nneg,
        )
        fo_abs = [abs(v) for v in first_order.values()]
        fo_mean = sum(fo_abs) / len(fo_abs) if fo_abs else 0.0
        fo_cap = sum(1 for v in hessian_scores.values() if v > fo_mean)
        logger.info(
            "  |fo|_mean=%.3e → %d experts above noise floor",
            fo_mean, fo_cap,
        )

    if args.ranking in ("activation_frequency", "hybrid"):
        logger.info("Loading token counts: %s", args.token_count_path)
        token_counts = _load_token_counts(
            str(args.token_count_path), num_layers, num_experts
        )

    logger.info("Ranking policy: %s", args.ranking)
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

        # Policy dispatch: select pairs, build score-for-reporting, build
        # expert_importance, fill in policy-specific report fields.
        fo_report: Optional[dict] = None
        hybrid_split: Optional[dict] = None
        kind_by_pair: Optional[Dict[Tuple[int, int], str]] = None
        expert_importance: Dict[int, List[float]]

        if args.ranking == "hessian_importance":
            assert hessian_scores is not None and fo_cap is not None
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
            heter, int4_only = _assign(hessian_scores, k)
            score_by_pair = hessian_scores
            expert_importance = _build_importance_hessian(
                hessian_scores, fo_mean, num_layers, num_experts, logger
            )

        elif args.ranking == "activation_frequency":
            assert token_counts is not None
            tc_scores = {k: float(v) for k, v in token_counts.items()}
            k = budget_k
            heter, int4_only = _assign(tc_scores, k)
            score_by_pair = tc_scores
            expert_importance = _build_importance_tokencount(
                token_counts, num_layers, num_experts, logger
            )

        else:  # hybrid
            assert (
                hessian_scores is not None
                and token_counts is not None
                and fo_cap is not None
            )
            k_hess = min(budget_k, fo_cap)
            k_fill = max(0, budget_k - k_hess)
            logger.info(
                "[mc=%d] hybrid split: hessian=%d, tokencount_fill=%d "
                "(budget_k=%d, fo_cap=%d)",
                mc, k_hess, k_fill, budget_k, fo_cap,
            )
            heter, int4_only, kind_by_pair = _assign_hybrid(
                hessian_scores, token_counts, k_hess, k_fill
            )
            # For top_scores reporting: use the score that drove each pair's
            # selection. Hessian segment shows hessian; gap-fill shows
            # token_count. Units differ — kind_by_pair disambiguates.
            score_by_pair = {}
            for p in heter:
                if kind_by_pair[p] == "hessian":
                    score_by_pair[p] = hessian_scores[p]
                else:
                    score_by_pair[p] = float(token_counts[p])
            for p in int4_only:
                score_by_pair[p] = hessian_scores[p]
            hybrid_split = {
                "fo_mean": fo_mean,
                "fo_cap": fo_cap,
                "budget_k": budget_k,
                "k_hessian": k_hess,
                "k_tokencount_fill": k_fill,
            }
            expert_importance = _build_importance_hybrid(
                hessian_scores, token_counts, fo_mean,
                heter, kind_by_pair,
                num_layers, num_experts, logger,
            )

        mc_dir = out_root / f"mc{mc}"
        if args.dry_run:
            planned = [
                "int4_only_experts.json",
                "heter_config.json",
                "expert_importance.json",
                "assignment_report.json",
            ]
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
                scores=score_by_pair,
                num_layers=num_layers,
                bf16_promotion_threshold=args.bf16_promotion_threshold,
                ranking_policy=args.ranking,
                fo_threshold=fo_report,
                expert_importance=expert_importance,
                attention_num_bits=args.attention_num_bits,
                kind_by_pair=kind_by_pair,
                hybrid_split=hybrid_split,
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
