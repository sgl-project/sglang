"""Static expert precision assignment helpers for HeterFusedMoE.

Pure library module: given per-layer PPL sensitivity and per-expert L2
sensitivity plus a VRAM budget, pick which experts are heter (dual
BF16+INT4) vs INT4-only and write out the three config files sglang
needs.

Called by ``profile/gen_heter_configs.py`` (the only entry point).
Writes to ``out_dir``:
  int4_only_experts.json  -- per-layer expert-id lists
  heter_config.json       -- ready for ``--heter-precision-config``
  assignment_report.json  -- VRAM breakdown, K, per-layer counts, top scores

See docs/superpowers/specs/2026-04-19-static-expert-assignment-design.md.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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

    by_layer: Dict[str, List[int]] = {}
    for L, E in int4_only:
        by_layer.setdefault(str(L), []).append(E)
    for k in by_layer:
        by_layer[k].sort()
    int4_path = out_dir / "int4_only_experts.json"
    with open(int4_path, "w") as f:
        json.dump(by_layer, f, indent=2)

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
        "zero_layers": zero_layers,
    }
    report_path = out_dir / "assignment_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    return int4_path, config_path, report_path
