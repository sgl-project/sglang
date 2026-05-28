"""Pure-function scoring/rendering helpers shared by the per-model and matrix
SSM-dtype-accuracy summarizers. Kept free of I/O so it can be unit-tested.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence

SCORE_KEYS: Sequence[str] = ("score", "mean_score", "accuracy")


def extract_score(metrics: Mapping[str, object]) -> Optional[float]:
    """Pull the first present score-like field from a metrics dict.

    Mirrors the original inline behaviour: prefer "score", then "mean_score",
    then "accuracy"; return None when none are present (callers render N/A).
    """
    for key in SCORE_KEYS:
        if key in metrics:
            value = metrics[key]
            return value  # type: ignore[return-value]
    return None


def select_baseline_dtype(dtypes: Sequence[str]) -> str:
    """Baseline is float32 when present, otherwise the first dtype.

    Matches the original ordering used by both summarizers.
    """
    if not dtypes:
        raise ValueError("dtypes must be non-empty")
    return "float32" if "float32" in dtypes else dtypes[0]


def compute_deltas(
    scores: Mapping[tuple, Optional[float]],
    dtypes: Sequence[str],
    evals: Sequence[str],
    baseline_dtype: str,
    *,
    model: Optional[str] = None,
) -> list[dict]:
    """Return per-eval, per-non-baseline-dtype delta rows.

    ``scores`` is keyed by ``(dtype, eval)`` for per-model usage or
    ``(model, dtype, eval)`` for matrix usage; pass ``model`` to opt into the
    3-tuple form.
    """
    deltas: list[dict] = []
    for eval_name in evals:
        baseline_key = (
            (model, baseline_dtype, eval_name)
            if model is not None
            else (baseline_dtype, eval_name)
        )
        baseline = scores.get(baseline_key)
        for dtype in dtypes:
            if dtype == baseline_dtype:
                continue
            score_key = (
                (model, dtype, eval_name) if model is not None else (dtype, eval_name)
            )
            score = scores.get(score_key)
            delta = None if baseline is None or score is None else score - baseline
            deltas.append(
                {
                    "eval": eval_name,
                    "baseline_dtype": baseline_dtype,
                    "dtype": dtype,
                    "baseline_score": baseline,
                    "score": score,
                    "delta": delta,
                }
            )
    return deltas


def format_score(score: Optional[float]) -> str:
    return "N/A" if score is None else f"{score:.10f}"


def format_delta(delta: Optional[float]) -> str:
    return "N/A" if delta is None else f"{delta:+.10f}"


def render_score_table(
    scores: Mapping[tuple, Optional[float]],
    dtypes: Sequence[str],
    evals: Sequence[str],
    *,
    model: Optional[str] = None,
) -> list[str]:
    """Markdown ``| Eval | dtype1 | dtype2 | ...`` table lines."""
    lines = [
        "| Eval | " + " | ".join(dtypes) + " |",
        "|---" + "|---" * len(dtypes) + "|",
    ]
    for eval_name in evals:
        vals = []
        for dtype in dtypes:
            key = (model, dtype, eval_name) if model is not None else (dtype, eval_name)
            vals.append(format_score(scores.get(key)))
        lines.append(f"| {eval_name} | " + " | ".join(vals) + " |")
    return lines


def render_delta_table_per_model(deltas: Iterable[Mapping[str, object]]) -> list[str]:
    """Per-model delta table: ``| Eval | Baseline | Dtype | Delta |``."""
    lines = [
        "| Eval | Baseline | Dtype | Delta |",
        "|---|---:|---:|---:|",
    ]
    for item in deltas:
        lines.append(
            f"| {item['eval']} | {item['baseline_dtype']} | {item['dtype']} | {format_delta(item['delta'])} |"  # type: ignore[arg-type]
        )
    return lines


def render_delta_table_matrix(
    scores: Mapping[tuple, Optional[float]],
    model: str,
    dtypes: Sequence[str],
    evals: Sequence[str],
) -> list[str]:
    """Matrix delta table for one model: ``| Eval | Dtype | Delta vs float32 |``."""
    lines = [
        "| Eval | Dtype | Delta vs float32 |",
        "|---|---:|---:|",
    ]
    for eval_name in evals:
        baseline = scores.get((model, "float32", eval_name))
        for dtype in dtypes:
            if dtype == "float32":
                continue
            score = scores.get((model, dtype, eval_name))
            delta = None if baseline is None or score is None else score - baseline
            lines.append(f"| {eval_name} | {dtype} | {format_delta(delta)} |")
    return lines


def sort_dtypes_float32_first(dtypes: Iterable[str]) -> list[str]:
    """float32 first if present, then lexicographic. Matches matrix script."""
    return sorted(dtypes, key=lambda x: (x != "float32", x))
