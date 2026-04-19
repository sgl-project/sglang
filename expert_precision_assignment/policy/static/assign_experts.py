"""Static expert precision assignment for HeterFusedMoE.

Given per-layer PPL sensitivity and per-expert L2 sensitivity plus an SLO
(max concurrency / prompt len / output len), pick which experts are heter
(dual BF16+INT4) vs INT4-only so the resulting deployment fits on one GPU.

Writes three files to --out_dir:
  int4_only_experts.json  -- per-layer expert-id lists for --heter-precision-config
  heter_config.json       -- complete config ready to pass to sglang.launch_server
  assignment_report.json  -- VRAM breakdown, K, per-layer counts, top scores

See docs/superpowers/specs/2026-04-19-static-expert-assignment-design.md.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

# Allow running from anywhere.
sys.path.insert(0, str(Path(__file__).parent))
import vram_estimator as vest

logger = logging.getLogger(__name__)


def _resolve_hf_path(path: str) -> str:
    """Accept either a raw HF repo id or a /hub/models--.../snapshots/... path."""
    m = re.match(r"^(.+)/hub/models--(.+?)--(.+?)/snapshots/[a-f0-9]+$", path)
    if m:
        os.environ.setdefault("HF_HOME", m.group(1))
        return f"{m.group(2)}/{m.group(3)}"
    return path


def _load_model_config(model_path: str):
    from transformers import AutoConfig
    hf_name = _resolve_hf_path(model_path)
    return AutoConfig.from_pretrained(
        hf_name, trust_remote_code=True, local_files_only=True
    )


def _load_sensitivity_inputs(
    layer_path: str,
    expert_path: str,
    num_layers: int,
    num_experts: int,
) -> Tuple[Dict[int, float], Dict[int, Dict[int, Tuple[float, int]]]]:
    """Return (ppl_increase_by_layer, l2_and_tokencount_by_layer_expert)."""
    with open(layer_path) as f:
        layer_d = json.load(f)
    with open(expert_path) as f:
        expert_d = json.load(f)

    ppl: Dict[int, float] = {}
    for k, v in layer_d["per_layer"].items():
        ppl[int(k)] = float(v["ppl_increase"])

    l2: Dict[int, Dict[int, Tuple[float, int]]] = {}
    for k, v in expert_d["per_layer"].items():
        L = int(k)
        row: Dict[int, Tuple[float, int]] = {}
        for ek, ev in v["experts"].items():
            row[int(ek)] = (float(ev["sensitivity"]), int(ev["token_count"]))
        l2[L] = row

    # Coverage validation — fail loud if anything is missing.
    missing_layers = [L for L in range(num_layers) if L not in ppl]
    if missing_layers:
        raise ValueError(
            f"layer_sensitivity missing layers: {missing_layers[:10]}"
            f"{' ...' if len(missing_layers) > 10 else ''}"
        )
    for L in range(num_layers):
        if L not in l2:
            raise ValueError(f"expert_sensitivity missing layer {L}")
        missing_experts = [E for E in range(num_experts) if E not in l2[L]]
        if missing_experts:
            raise ValueError(
                f"expert_sensitivity layer {L} missing experts: "
                f"{missing_experts[:10]}"
                f"{' ...' if len(missing_experts) > 10 else ''}"
            )
    return ppl, l2


def _composite_scores(
    ppl: Dict[int, float],
    l2: Dict[int, Dict[int, Tuple[float, int]]],
    num_experts: int,
) -> Dict[Tuple[int, int], float]:
    """score(L, E) = max(0, ppl_increase[L]) * l2[L][E] / sum_E l2[L]
    Forced to 0 when ppl<=0 or token_count==0."""
    scores: Dict[Tuple[int, int], float] = {}
    for L, experts in l2.items():
        layer_ppl = max(0.0, ppl[L])
        sum_l2 = sum(s for s, _ in experts.values())
        if sum_l2 <= 0 or layer_ppl <= 0:
            for E in range(num_experts):
                scores[(L, E)] = 0.0
            continue
        for E, (s, tc) in experts.items():
            scores[(L, E)] = 0.0 if tc == 0 else layer_ppl * s / sum_l2
    return scores


def _assign(
    scores: Dict[Tuple[int, int], float],
    K: int,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Return (heter_experts, int4_only_experts). Top-K by score → heter."""
    # Deterministic ordering: score DESC, layer ASC, expert ASC.
    ranked = sorted(
        scores.items(),
        key=lambda kv: (-kv[1], kv[0][0], kv[0][1]),
    )
    heter = [k for k, _ in ranked[:K]]
    int4_only = [k for k, _ in ranked[K:]]
    return heter, int4_only


def _write_outputs(
    out_dir: Path,
    int4_only: List[Tuple[int, int]],
    heter: List[Tuple[int, int]],
    int4_checkpoint: str,
    group_size: int,
    budget: vest.Budget,
    slo: vest.SLO,
    knobs: vest.BudgetKnobs,
    scores: Dict[Tuple[int, int], float],
    ppl: Dict[int, float],
    num_layers: int,
) -> Tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    # int4_only_experts.json: {layer_id: [expert_ids...]} (layer-id string,
    # expert ids sorted)
    by_layer: Dict[str, List[int]] = {}
    for L, E in int4_only:
        by_layer.setdefault(str(L), []).append(E)
    for k in by_layer:
        by_layer[k].sort()
    int4_path = out_dir / "int4_only_experts.json"
    with open(int4_path, "w") as f:
        json.dump(by_layer, f, indent=2)

    # heter_config.json: ready for --heter-precision-config.
    heter_config = {
        "groups": [
            {
                "name": "cold",
                "num_bits": 4,
                "group_size": group_size,
                "checkpoint": int4_checkpoint,
            },
            {"name": "hot", "num_bits": 16},
        ],
        "policy": "expert_batch",
        "policy_params": {"threshold": 128},
        "int4_only_experts_file": str(int4_path.resolve()),
    }
    config_path = out_dir / "heter_config.json"
    with open(config_path, "w") as f:
        json.dump(heter_config, f, indent=2)

    # assignment_report.json
    heter_by_layer: Dict[str, int] = {}
    int4_by_layer: Dict[str, int] = {}
    for L, _ in heter:
        heter_by_layer[str(L)] = heter_by_layer.get(str(L), 0) + 1
    for L, _ in int4_only:
        int4_by_layer[str(L)] = int4_by_layer.get(str(L), 0) + 1
    per_layer_counts = {
        str(L): {
            "heter": heter_by_layer.get(str(L), 0),
            "int4_only": int4_by_layer.get(str(L), 0),
        }
        for L in range(num_layers)
    }
    top_scores = [
        {"layer": L, "expert": E, "score": scores[(L, E)]}
        for (L, E) in heter[:20]
    ]
    zero_layers = sorted(L for L, v in ppl.items() if v <= 0)

    report = {
        "slo": {
            "max_concurrency": slo.max_concurrency,
            "max_prompt_len": slo.max_prompt_len,
            "max_output_len": slo.max_output_len,
        },
        "knobs": {
            "kv_reserve_frac": knobs.kv_reserve_frac,
            "headroom_gb": knobs.headroom_gb,
            "headroom_frac": knobs.headroom_frac,
            "prefill_budget_tokens": knobs.prefill_budget_tokens,
            "group_size": knobs.group_size,
        },
        "vram_breakdown_bytes": budget.as_breakdown(),
        "K_heter_experts": len(heter),
        "num_int4_only": len(int4_only),
        "per_layer_counts": per_layer_counts,
        "top_scores": top_scores,
        "zero_layers": zero_layers,
    }
    report_path = out_dir / "assignment_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    return int4_path, config_path, report_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layer_sensitivity", required=True)
    ap.add_argument("--expert_sensitivity", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--int4_checkpoint", required=True)
    ap.add_argument("--max_concurrency", type=int, required=True)
    ap.add_argument("--max_prompt_len", type=int, required=True)
    ap.add_argument("--max_output_len", type=int, required=True)
    ap.add_argument("--kv_reserve_frac", type=float, default=0.5)
    ap.add_argument("--headroom_gb", type=float, default=2.0)
    ap.add_argument("--headroom_frac", type=float, default=0.05)
    ap.add_argument("--prefill_budget_tokens", type=int, default=16384)
    ap.add_argument("--group_size", type=int, default=128)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--gpu_vram_bytes", type=int, default=-1,
                    help="Override detected GPU VRAM (for testing / planning).")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    config = _load_model_config(args.model_path)
    num_layers = config.num_hidden_layers
    num_experts = config.num_experts

    if args.gpu_vram_bytes > 0:
        gpu_vram = args.gpu_vram_bytes
    else:
        gpu_vram = torch.cuda.get_device_properties(args.gpu).total_memory

    slo = vest.SLO(
        max_concurrency=args.max_concurrency,
        max_prompt_len=args.max_prompt_len,
        max_output_len=args.max_output_len,
    )
    knobs = vest.BudgetKnobs(
        kv_reserve_frac=args.kv_reserve_frac,
        headroom_gb=args.headroom_gb,
        headroom_frac=args.headroom_frac,
        prefill_budget_tokens=args.prefill_budget_tokens,
        group_size=args.group_size,
    )
    budget = vest.compute_budget(config, gpu_vram, slo, knobs)

    logger.info(
        "Model: H=%d I=%d L=%d num_experts=%d  (%d total experts)",
        config.hidden_size, config.moe_intermediate_size,
        num_layers, num_experts, num_experts * num_layers,
    )
    logger.info("VRAM budget:\n%s", vest.format_budget(budget))

    if budget.k_heter_experts == 0:
        logger.error(
            "No BF16 budget available (bf16_budget_bytes=%d). "
            "Either relax SLO (lower max_concurrency / seq-len) or lower "
            "--kv_reserve_frac.",
            budget.bf16_budget,
        )
        # Still emit outputs so the operator can see the breakdown.

    ppl, l2 = _load_sensitivity_inputs(
        args.layer_sensitivity, args.expert_sensitivity, num_layers, num_experts
    )
    scores = _composite_scores(ppl, l2, num_experts)
    heter, int4_only = _assign(scores, budget.k_heter_experts)

    out_dir = Path(args.out_dir)
    int4_path, cfg_path, report_path = _write_outputs(
        out_dir=out_dir,
        int4_only=int4_only,
        heter=heter,
        int4_checkpoint=args.int4_checkpoint,
        group_size=args.group_size,
        budget=budget,
        slo=slo,
        knobs=knobs,
        scores=scores,
        ppl=ppl,
        num_layers=num_layers,
    )
    logger.info(
        "Wrote:\n  %s\n  %s\n  %s",
        int4_path, cfg_path, report_path,
    )
    logger.info(
        "Assigned %d heter / %d int4-only (of %d total)",
        len(heter), len(int4_only), num_experts * num_layers,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
