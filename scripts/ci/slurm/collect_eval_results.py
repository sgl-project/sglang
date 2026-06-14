#!/usr/bin/env python3
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tabulate import tabulate

# Header strings — kept inline (vs imported from summarize.py) so this script
# stays self-contained.
MODEL = "Model"
HARDWARE = "Hardware"
FRAMEWORK = "Framework"
PRECISION = "Precision"
TP = "TP"
EP = "EP"
DP_ATTENTION = "DP Attention"
CONC = "Conc"
TASK = "Task"
SCORE = "Score"
EM_STRICT = "EM Strict"
EM_FLEXIBLE = "EM Flexible"
N_EFF = "N (eff)"
SPEC_DECODING = "Spec Decode"
PREFILL_TP = "Prefill TP"
PREFILL_EP = "Prefill EP"
PREFILL_DP_ATTN = "Prefill DP Attn"
PREFILL_WORKERS = "Prefill Workers"
DECODE_TP = "Decode TP"
DECODE_EP = "Decode EP"
DECODE_DP_ATTN = "Decode DP Attn"
DECODE_WORKERS = "Decode Workers"


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def find_eval_sets(root: Path) -> List[Path]:
    """Return directories that contain a meta_env.json (one set per job).

    Structure: eval_results/<artifact-name>/meta_env.json
    When download-artifact downloads a single artifact, files may be
    extracted flat into root (no subdirectory), so check root itself too.
    """
    out: List[Path] = []
    try:
        if (root / "meta_env.json").exists():
            out.append(root)
        for d in root.iterdir():
            if d.is_dir() and (d / "meta_env.json").exists():
                out.append(d)
    except Exception:
        pass
    return out


def detect_eval_jsons(d: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """Return (lm_eval_json, _) if present.

    Checks immediate directory for result JSONs.
    """
    immediate_jsons = list(d.glob("results*.json")) + [
        p for p in d.glob("*.json") if p.name != "meta_env.json"
    ]

    lm_path = None
    le_path = None

    for p in immediate_jsons:
        data = load_json(p)
        if not isinstance(data, dict):
            continue

        if "lm_eval_version" in data:
            if lm_path is None or p.stat().st_mtime > lm_path.stat().st_mtime:
                lm_path = p

    return lm_path, le_path


def extract_lm_metrics(json_path: Path) -> List[Dict[str, Any]]:
    """Extract metrics from lm-eval harness result JSON.

    Returns a list of metric dicts, one per task in the results.

    Uses explicit structure from the JSON file:
    - Task names from results keys
    - Metric name from configs.metric_list
    - Filter names from configs.filter_list
    - Values from results[task][metric,filter]
    """
    data = load_json(json_path) or {}
    results = data.get("results", {})
    configs = data.get("configs", {})

    if not results:
        return []

    extracted = []

    for task in results.keys():
        task_results = results[task]
        task_config = configs.get(task, {})

        metric_list = task_config.get("metric_list", [])
        base_metric = metric_list[0]["metric"] if metric_list else "exact_match"

        filter_list = task_config.get("filter_list", [])

        strict_val, strict_se = None, None
        flex_val, flex_se = None, None
        accuracy_val, accuracy_se = None, None

        def get_val_se(filter_name: str) -> Tuple[Optional[float], Optional[float]]:
            val_key = f"{base_metric},{filter_name}"
            se_key = f"{base_metric}_stderr,{filter_name}"
            return task_results.get(val_key), task_results.get(se_key)

        if not filter_list:
            if "acc" in task_results:
                accuracy_val = task_results.get("acc")
                accuracy_se = task_results.get("acc_stderr")
            else:
                strict_val = task_results.get(base_metric)
                strict_se = task_results.get(f"{base_metric}_stderr")
        else:
            for f in filter_list:
                fname = f["name"]
                if "strict" in fname:
                    strict_val, strict_se = get_val_se(fname)
                elif "flex" in fname or "extract" in fname:
                    flex_val, flex_se = get_val_se(fname)

        n_eff = data.get("n-samples", {}).get(task, {}).get("effective")

        model = data.get("model_name") or task_config.get("metadata", {}).get("model")

        extracted.append(
            {
                "task": task,
                "strict": strict_val,
                "strict_se": strict_se,
                "flex": flex_val,
                "flex_se": flex_se,
                "accuracy": accuracy_val,
                "accuracy_se": accuracy_se,
                "n_eff": n_eff,
                "model": model,
                "source": str(json_path),
            }
        )

    return extracted


def pct(x: Any) -> str:
    try:
        return f"{float(x) * 100:.2f}%"
    except Exception:
        return "N/A"


def se(x: Any) -> str:
    try:
        return f" ±{float(x) * 100:.2f}%"
    except Exception:
        return ""


def as_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def as_bool(x: Any, default: bool = False) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return default
    return str(x).lower() == "true"


def build_row(meta: Dict[str, Any], m: Dict[str, Any]) -> Dict[str, Any]:
    is_multinode = as_bool(meta.get("is_multinode"), False)
    prefill_tp = as_int(meta.get("prefill_tp", meta.get("tp", 1)), 1)
    prefill_ep = as_int(meta.get("prefill_ep", meta.get("ep", 1)), 1)
    prefill_num_workers = as_int(meta.get("prefill_num_workers", 1), 1)
    decode_tp = as_int(meta.get("decode_tp", meta.get("tp", 1)), 1)
    decode_ep = as_int(meta.get("decode_ep", meta.get("ep", 1)), 1)
    decode_num_workers = as_int(meta.get("decode_num_workers", 1), 1)
    prefill_dp_attention = meta.get("prefill_dp_attention")
    decode_dp_attention = meta.get("decode_dp_attention")
    dp_attention = meta.get("dp_attention", "none")

    if prefill_dp_attention is None:
        prefill_dp_attention = dp_attention
    if decode_dp_attention is None:
        decode_dp_attention = dp_attention

    if is_multinode:
        if prefill_dp_attention == decode_dp_attention:
            dp_attention = prefill_dp_attention
        else:
            dp_attention = (
                f"prefill={str(prefill_dp_attention).lower()},"
                f"decode={str(decode_dp_attention).lower()}"
            )

    row = {
        "is_multinode": is_multinode,
        "model_prefix": meta.get("model_prefix", "unknown"),
        "model": m.get("model") or meta.get("model", "unknown"),
        "hw": meta.get("hw", "unknown").upper(),
        "framework": meta.get("framework", "unknown").lower(),
        "precision": meta.get("precision", "unknown").lower(),
        "spec_decoding": meta.get("spec_decoding", "unknown"),
        "tp": as_int(meta.get("tp", prefill_tp), prefill_tp),
        "ep": as_int(meta.get("ep", prefill_ep), prefill_ep),
        "prefill_tp": prefill_tp,
        "prefill_ep": prefill_ep,
        "prefill_num_workers": prefill_num_workers,
        "decode_tp": decode_tp,
        "decode_ep": decode_ep,
        "decode_num_workers": decode_num_workers,
        "conc": as_int(meta.get("conc", 0), 0),
        "dp_attention": str(dp_attention).lower(),
        "prefill_dp_attention": str(prefill_dp_attention).lower(),
        "decode_dp_attention": str(decode_dp_attention).lower(),
        "task": m.get("task", "unknown"),
        "em_strict": m.get("strict"),
        "em_strict_se": m.get("strict_se"),
        "em_flexible": m.get("flex"),
        "em_flexible_se": m.get("flex_se"),
        "n_eff": m.get("n_eff"),
        "source": m.get("source"),
    }

    if m.get("strict") is not None:
        row["score"] = m.get("strict")
        row["score_name"] = "em_strict"
        row["score_se"] = m.get("strict_se")
    elif m.get("accuracy") is not None:
        row["score"] = m.get("accuracy")
        row["score_name"] = "accuracy"
        row["score_se"] = m.get("accuracy_se")
    else:
        row["score"] = None
        row["score_name"] = None
        row["score_se"] = None

    return row


def main():
    if len(sys.argv) < 3:
        print(
            "Usage: collect_eval_results.py <results_dir> <exp_name> "
            "[sort_by: model_prefix|hw]"
        )
        sys.exit(1)

    root = Path(sys.argv[1])
    exp_name = sys.argv[2]

    rows: List[Dict[str, Any]] = []
    for d in find_eval_sets(root):
        meta = load_json(d / "meta_env.json") or {}
        lm_path, le_path = detect_eval_jsons(d)

        if lm_path:
            metrics_list = extract_lm_metrics(lm_path)
        else:
            continue

        if not metrics_list:
            continue

        for m in metrics_list:
            row = build_row(meta, m)
            rows.append(row)

    single_node_rows = [r for r in rows if not r["is_multinode"]]
    multinode_rows = [r for r in rows if r["is_multinode"]]

    sort_by = sys.argv[3] if len(sys.argv) > 3 else "model_prefix"
    single_node_sort_key = (
        (
            lambda r: (
                r["hw"],
                r["framework"],
                r["precision"],
                r.get("spec_decoding", ""),
                r["tp"],
                r["ep"],
                r["conc"],
            )
        )
        if sort_by == "hw"
        else (
            lambda r: (
                r["model_prefix"],
                r["hw"],
                r["framework"],
                r["precision"],
                r.get("spec_decoding", ""),
                r["tp"],
                r["ep"],
                r["conc"],
            )
        )
    )
    multinode_sort_key = (
        (
            lambda r: (
                r["hw"],
                r["framework"],
                r["precision"],
                r.get("spec_decoding", ""),
                r["prefill_tp"],
                r["prefill_ep"],
                r["prefill_num_workers"],
                r["decode_tp"],
                r["decode_ep"],
                r["decode_num_workers"],
                r["conc"],
            )
        )
        if sort_by == "hw"
        else (
            lambda r: (
                r["model_prefix"],
                r["hw"],
                r["framework"],
                r["precision"],
                r.get("spec_decoding", ""),
                r["prefill_tp"],
                r["prefill_ep"],
                r["prefill_num_workers"],
                r["decode_tp"],
                r["decode_ep"],
                r["decode_num_workers"],
                r["conc"],
            )
        )
    )
    single_node_rows.sort(key=single_node_sort_key)
    multinode_rows.sort(key=multinode_sort_key)

    if not rows:
        print("> No eval results found to summarize.")
    else:
        MODEL_PREFIX = "Model Prefix"

        if single_node_rows:
            headers = [
                MODEL_PREFIX,
                HARDWARE,
                FRAMEWORK,
                PRECISION,
                SPEC_DECODING,
                TP,
                EP,
                CONC,
                DP_ATTENTION,
                TASK,
                SCORE,
                EM_STRICT,
                EM_FLEXIBLE,
                N_EFF,
                MODEL,
            ]
            table_rows = [
                [
                    r["model_prefix"],
                    r["hw"],
                    r["framework"].upper(),
                    r["precision"].upper(),
                    r["spec_decoding"],
                    r["tp"],
                    r["ep"],
                    r["conc"],
                    r["dp_attention"],
                    r["task"],
                    f"{pct(r['score'])}{se(r['score_se'])}",
                    f"{pct(r['em_strict'])}{se(r['em_strict_se'])}",
                    f"{pct(r['em_flexible'])}{se(r['em_flexible_se'])}",
                    r["n_eff"] or "",
                    r["model"],
                ]
                for r in single_node_rows
            ]
            print("### Single-Node Eval Results\n")
            print(tabulate(table_rows, headers=headers, tablefmt="github"))

        if multinode_rows:
            headers = [
                MODEL_PREFIX,
                HARDWARE,
                FRAMEWORK,
                PRECISION,
                SPEC_DECODING,
                PREFILL_TP,
                PREFILL_EP,
                PREFILL_DP_ATTN,
                PREFILL_WORKERS,
                DECODE_TP,
                DECODE_EP,
                DECODE_DP_ATTN,
                DECODE_WORKERS,
                CONC,
                TASK,
                SCORE,
                EM_STRICT,
                EM_FLEXIBLE,
                N_EFF,
                MODEL,
            ]
            table_rows = [
                [
                    r["model_prefix"],
                    r["hw"],
                    r["framework"].upper(),
                    r["precision"].upper(),
                    r["spec_decoding"],
                    r["prefill_tp"],
                    r["prefill_ep"],
                    r["prefill_dp_attention"],
                    r["prefill_num_workers"],
                    r["decode_tp"],
                    r["decode_ep"],
                    r["decode_dp_attention"],
                    r["decode_num_workers"],
                    r["conc"],
                    r["task"],
                    f"{pct(r['score'])}{se(r['score_se'])}",
                    f"{pct(r['em_strict'])}{se(r['em_strict_se'])}",
                    f"{pct(r['em_flexible'])}{se(r['em_flexible_se'])}",
                    r["n_eff"] or "",
                    r["model"],
                ]
                for r in multinode_rows
            ]
            if single_node_rows:
                print("\n")
            print("### Multi-Node Eval Results\n")
            print(tabulate(table_rows, headers=headers, tablefmt="github"))

    out_path = Path(f"agg_eval_{exp_name}.json")
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
