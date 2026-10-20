"""Train MLTimePredictor bundles from synchronized SGLang schedule-batch JSONL."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sglang_simulator.time_predictor.ml_features import (
    ML_FEATURE_NAMES,
    extract_ml_features,
)
from sklearn.ensemble import HistGradientBoostingRegressor

MONOTONIC_FEATURES = {
    "batch_size",
    "sum_extend",
    "sum_past",
    "sum_extend_x_past",
    "sum_extend_squared",
    "sum_past_squared",
    "sum_attn_flops",
    "sum_extend_x_max_past",
    "log1p_sum_past",
    "log1p_sum_attn_flops",
    "batch_size_x_sum_extend",
}


def load_rows(roots, forward_modes, max_latency):
    files = set()
    for value in roots:
        root = Path(value).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(root)
        files.update(
            [root] if root.is_file() else root.rglob("TP0*.schedule_batch.jsonl")
        )
    if not files:
        raise ValueError(f"no schedule files found under {roots}")
    rows, skipped = [], Counter()
    for path in sorted(files):
        with path.open() as source:
            for number, line in enumerate(source, 1):
                try:
                    item = json.loads(line)
                    if not isinstance(item, dict):
                        raise ValueError("expected a JSON object")
                    if item.get("forward_mode") not in forward_modes:
                        skipped["mode"] += 1
                        continue
                    requests = item["request_infos"]
                    if not requests:
                        skipped["empty"] += 1
                        continue
                    latency = float(item["iter_latency"])
                    if not math.isfinite(latency) or not 0 < latency <= max_latency:
                        skipped["latency"] += 1
                        continue
                    extend = [r["extend_input_len"] for r in requests]
                    past = [r["prefix_indices_len"] for r in requests]
                    if any(type(v) is not int or v < 0 for v in extend + past):
                        raise ValueError("token lengths must be non-negative integers")
                    features = extract_ml_features(extend, past)
                except (KeyError, TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(f"{path}:{number}: {exc}") from exc
                rows.append((path.parent, features, latency))
    return rows, dict(skipped)


def split_rows(rows, seed):
    """Deterministic 50/50 split within each case (collection directory)."""
    by_case = defaultdict(list)
    for row in rows:
        by_case[row[0]].append(row)
    random = np.random.RandomState(seed)
    train, test = [], []
    for case in by_case.values():
        order = random.permutation(len(case))
        test_count = len(case) // 2
        test.extend(case[i] for i in order[:test_count])
        train.extend(case[i] for i in order[test_count:])
    return train, test


def as_arrays(rows):
    return np.asarray([r[1] for r in rows]), np.asarray([r[2] for r in rows])


def train_model(rows, loss, seed):
    model = HistGradientBoostingRegressor(
        loss="quantile" if loss == "p50" else "squared_error",
        quantile=0.5 if loss == "p50" else None,
        max_iter=400,
        learning_rate=0.04,
        max_leaf_nodes=63,
        l2_regularization=0.1,
        monotonic_cst=[int(name in MONOTONIC_FEATURES) for name in ML_FEATURE_NAMES],
        early_stopping=len(rows) > 1,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=seed,
    )
    return model.fit(*as_arrays(rows))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        action="append",
        required=True,
        help="Collection directory or JSONL file; repeatable.",
    )
    parser.add_argument(
        "--test-data-root",
        action="append",
        default=[],
        help="Independent evaluation data, excluded from fitting; repeatable.",
    )
    parser.add_argument(
        "--out-dir", required=True, help="New or empty output directory."
    )
    parser.add_argument(
        "--forward-modes",
        type=int,
        nargs="+",
        default=[1],
        help="SGLang ForwardMode values; default: EXTEND (1).",
    )
    parser.add_argument(
        "--max-latency",
        type=float,
        default=30.0,
        help="Largest retained latency, in seconds.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--loss", choices=("l2", "p50", "both"), default="both")
    args = parser.parse_args(argv)
    if not math.isfinite(args.max_latency) or args.max_latency <= 0:
        parser.error("--max-latency must be finite and positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.out_dir).expanduser()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"training output directory must be empty: {output}")
    rows, skipped = load_rows(args.data_root, args.forward_modes, args.max_latency)
    if not rows:
        raise ValueError(
            f"no usable training rows under {args.data_root}; skipped={skipped}"
        )
    if args.test_data_root:
        test, test_skipped = load_rows(
            args.test_data_root, args.forward_modes, args.max_latency
        )
        if not test:
            raise ValueError(
                f"no usable test rows under {args.test_data_root}; skipped={test_skipped}"
            )
        if {r[0] for r in rows} & {r[0] for r in test}:
            raise ValueError(
                "training and test roots contain overlapping collection cases"
            )
        train = rows
    else:
        train, test = split_rows(rows, args.seed)
        test_skipped = {}
    if not train or not test:
        raise ValueError(
            "need nonempty training and test data; provide a case with at least two usable rows or --test-data-root"
        )
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "seed": args.seed,
        "split": "independent" if args.test_data_root else "within_case_50_50",
        "train_rows": len(train),
        "test_rows": len(test),
        "final_fit_rows": len(rows),
        "skipped": skipped,
        "test_skipped": test_skipped,
        "metrics": {},
    }
    features, latency = as_arrays(test)
    for loss in ("l2", "p50") if args.loss == "both" else (args.loss,):
        model = train_model(train, loss, args.seed)
        errors = np.abs(model.predict(features) - latency)
        summary["metrics"][loss] = {
            "MAE_ms": float(errors.mean() * 1000),
            "MAPE_pct": float((errors / latency).mean() * 100),
        }  # codespell:ignore mape
        if not args.test_data_root:
            model = train_model(rows, loss, args.seed)
        # Runtime validates the ordered feature ABI before making predictions.
        bundle = {
            "model": model,
            "features": list(ML_FEATURE_NAMES),
            "sklearn_version": sklearn.__version__,
        }
        with (output / f"latency_model_hgbmono_{loss}.pkl").open("xb") as destination:
            joblib.dump(bundle, destination)
    with (output / "summary.json").open("x") as destination:
        json.dump(summary, destination, indent=2)
        destination.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
