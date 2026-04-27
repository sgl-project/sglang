"""Static expert precision assignment helpers for HeterFusedMoE.

Pure library module: given a VRAM budget and per-(layer, expert) signals
(Hessian sensitivity and/or routed-token counts), pick which experts are
heter (dual BF16+INT4) vs INT4-only and write out the config files
sglang needs.

Three ranking policies are supported:
  * ``hessian_importance``: signed ½·dᵀHd from ``hessian/`` capped at the
    |first-order|-mean noise floor.
  * ``activation_frequency``: token_count from
    ``legacy/sensitivity/per_expert/`` (raw routed-token counts).
  * ``hybrid``: top fo_cap by Hessian, then fill the leftover VRAM with
    the next-highest experts by token_count.

Called by ``experiments/pipeline/gen_config/gen_heter_configs.py`` (the only entry point).
Writes to ``out_dir``:
  int4_only_experts.json  -- per-layer expert-id lists
  heter_config.json       -- ready for ``--heter-precision-config``
  assignment_report.json  -- VRAM breakdown, K, per-layer counts, top scores
  expert_importance.json  -- per-layer per-expert prior for runtime dispatch
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent))
import vram_estimator as vest  # noqa: E402


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


def _load_token_counts(
    summary_path: str,
    num_layers: int,
    num_experts: int,
) -> Dict[Tuple[int, int], int]:
    """Load per-(layer, expert) routed-token counts from the per-expert
    sensitivity summary at ``legacy/sensitivity/per_expert/results/summary.json``.

    The file's ``sensitivity`` column is unused here — only ``token_count``
    is consumed. Layers / experts missing from the file fail loud.
    """
    with open(summary_path) as f:
        d = json.load(f)
    per_layer = d.get("per_layer")
    if per_layer is None:
        raise ValueError(f"{summary_path}: missing 'per_layer'")

    tc: Dict[Tuple[int, int], int] = {}
    for L_str, layer_d in per_layer.items():
        L = int(L_str)
        for E_str, e_d in layer_d.get("experts", {}).items():
            tc[(L, int(E_str))] = int(e_d["token_count"])

    missing = [
        (L, E)
        for L in range(num_layers)
        for E in range(num_experts)
        if (L, E) not in tc
    ]
    if missing:
        raise ValueError(
            f"token_count missing {len(missing)} (layer, expert) pairs, "
            f"first 10: {missing[:10]}"
        )
    return tc


def _load_hessian_scores(
    hessian_path: str,
    num_layers: int,
    num_experts: int,
) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], float]]:
    """Load per-expert ½·dᵀHd and first-order scores from ``hessian_score.py``.

    Uses the signed Hessian score directly: positive = INT4 hurts end-to-end
    loss (expert should stay BF16); negative = INT4 is neutral or mildly
    helpful (safe to quantize). Ranking top-K by signed score DESC therefore
    naturally keeps the critical experts in BF16 and dumps loss-reducing or
    near-zero experts to INT4.

    Also returns per-expert first-order scores ``g·d`` — at a trained minimum
    g≈0 so ``|g·d|`` characterizes the MC noise floor; experts whose Hessian
    score is below ``mean(|g·d|)`` are statistically indistinguishable from
    zero and safe to leave INT4.
    """
    with open(hessian_path) as f:
        d = json.load(f)

    per_layer = d.get("per_layer")
    if per_layer is None:
        raise ValueError(f"{hessian_path}: missing 'per_layer'")

    scores: Dict[Tuple[int, int], float] = {}
    first_order: Dict[Tuple[int, int], float] = {}
    for L_str, layer_d in per_layer.items():
        L = int(L_str)
        experts = layer_d.get("experts", {})
        for E_str, e_d in experts.items():
            E = int(E_str)
            scores[(L, E)] = float(e_d["hessian_score"])
            first_order[(L, E)] = float(e_d["first_order_score"])

    missing = [
        (L, E)
        for L in range(num_layers)
        for E in range(num_experts)
        if (L, E) not in scores
    ]
    if missing:
        raise ValueError(
            f"hessian_scores missing {len(missing)} (layer, expert) pairs, "
            f"first 10: {missing[:10]}"
        )
    return scores, first_order


def _assign(
    scores: Dict[Tuple[int, int], float],
    K: int,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Return (heter_experts, int4_only_experts). Top-K by score → heter.

    Deterministic ordering: score DESC, layer ASC, expert ASC.
    """
    ranked = sorted(
        scores.items(),
        key=lambda kv: (-kv[1], kv[0][0], kv[0][1]),
    )
    heter = [k for k, _ in ranked[:K]]
    int4_only = [k for k, _ in ranked[K:]]
    return heter, int4_only


def _assign_hybrid(
    hessian_scores: Dict[Tuple[int, int], float],
    token_counts: Dict[Tuple[int, int], int],
    k_hess: int,
    k_fill: int,
) -> Tuple[
    List[Tuple[int, int]],
    List[Tuple[int, int]],
    Dict[Tuple[int, int], str],
]:
    """Two-stage selection: top ``k_hess`` by Hessian, then top ``k_fill``
    by token_count from the remainder.

    Returns (heter, int4_only, kind_by_pair) where kind_by_pair maps every
    selected pair to either ``"hessian"`` or ``"tokencount"``.
    """
    hess_top, rest = _assign(hessian_scores, k_hess)
    rest_token_counts = {k: float(token_counts[k]) for k in rest}
    fill_top, fill_rest = _assign(rest_token_counts, k_fill)

    heter = hess_top + fill_top
    int4_only = fill_rest
    kind_by_pair: Dict[Tuple[int, int], str] = {}
    for k in hess_top:
        kind_by_pair[k] = "hessian"
    for k in fill_top:
        kind_by_pair[k] = "tokencount"
    return heter, int4_only, kind_by_pair


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
    num_layers: int,
    bf16_promotion_threshold: int,
    ranking_policy: str,
    fo_threshold: Optional[dict] = None,
    expert_importance: Optional[Dict[int, List[float]]] = None,
    attention_num_bits: int = 16,
    kind_by_pair: Optional[Dict[Tuple[int, int], str]] = None,
    hybrid_split: Optional[dict] = None,
) -> Tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    by_layer: Dict[str, List[int]] = {}
    for L, E in int4_only:
        by_layer.setdefault(str(L), []).append(E)
    for k in by_layer:
        by_layer[k].sort()
    int4_path = out_dir / "int4_only_experts.json"
    with open(int4_path, "w") as f:
        json.dump(by_layer, f, indent=2)

    importance_path: Optional[Path] = None
    if expert_importance is not None:
        importance_path = out_dir / "expert_importance.json"
        # Keep keys as strings so the runtime loader can index by str(layer_id).
        by_layer_str = {
            str(L): [float(v) for v in row]
            for L, row in expert_importance.items()
        }
        with open(importance_path, "w") as f:
            json.dump(by_layer_str, f, indent=2)

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
        "policy_params": {},
        "bf16_promotion_threshold": int(bf16_promotion_threshold),
        "int4_only_experts_file": str(int4_path.resolve()),
        "attention_num_bits": attention_num_bits,
    }
    if importance_path is not None:
        heter_config["expert_importance_file"] = str(importance_path.resolve())
    config_path = out_dir / "heter_config.json"
    with open(config_path, "w") as f:
        json.dump(heter_config, f, indent=2)

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
    top_scores = []
    for (L, E) in heter[:20]:
        entry = {"layer": L, "expert": E, "score": scores[(L, E)]}
        if kind_by_pair is not None:
            entry["kind"] = kind_by_pair.get((L, E), "?")
        top_scores.append(entry)

    report = {
        "ranking_policy": ranking_policy,
        "fo_threshold": fo_threshold,
        "hybrid_split": hybrid_split,
        "slo": {
            "max_concurrency": slo.max_concurrency,
            "max_prompt_len": slo.max_prompt_len,
            "max_output_len": slo.max_output_len,
        },
        "knobs": {
            "kv_reserve_frac": knobs.kv_reserve_frac,
            "headroom_gb": knobs.headroom_gb,
            "headroom_frac": knobs.headroom_frac,
            "group_size": knobs.group_size,
            "chunked_prefill_size": knobs.chunked_prefill_size,
            "cuda_graph_max_bs": knobs.cuda_graph_max_bs,
            "tp_size": knobs.tp_size,
            "pp_size": knobs.pp_size,
            "num_piecewise_tokens": knobs.num_piecewise_tokens,
        },
        "vram_breakdown_bytes": budget.as_breakdown(),
        "K_heter_experts": len(heter),
        "num_int4_only": len(int4_only),
        "per_layer_counts": per_layer_counts,
        "top_scores": top_scores,
    }
    report_path = out_dir / "assignment_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    return int4_path, config_path, report_path
