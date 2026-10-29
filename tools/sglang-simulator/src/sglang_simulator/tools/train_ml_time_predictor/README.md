# Train an ML time predictor

Install the simulator package (`pip install ./tools/sglang-simulator` from a
source checkout, or install its wheel). The command and example data are included
in the package, so this works from any directory:

```bash
example=$(python -c "from importlib.resources import files; print(files('sglang_simulator.tools.train_ml_time_predictor') / 'example_steps.jsonl')")
sglang-simulator-train --data-root "$example" --out-dir /tmp/example-predictor
```

`example_steps.jsonl` contains collected EXTEND steps from a tiny, two-layer Qwen3
model with dummy weights on one A100, using repeated synthetic token sequences.
Only numeric feature inputs and measured latencies are retained. It demonstrates
the file format and training flow; it cannot produce a useful production predictor
or demonstrate prediction accuracy.

For real data, replace `--data-root` with a [collection](../collect_baseline/README.md)
directory. The trainer recursively reads `TP0*.schedule_batch.jsonl` to avoid
counting duplicate tensor-parallel ranks. Repeat `--data-root` for more cases, or
pass explicit files for a different rank selection. Use data collected for the
same target model, hardware and serving configuration.

The default filter keeps EXTEND (`forward_mode=1`) steps with nonempty batches and
finite latencies in `(0, 30]` seconds. Use `--forward-modes 1 2` to include DECODE
and `--max-latency` to change the upper bound. Malformed records fail with their
file and line; skipped row counts are included in the summary.

Training shares the simulator's ordered 18-feature definition and uses monotonic
HistGradientBoosting regressors. The new or empty output directory receives:

- `latency_model_hgbmono_l2.pkl`: squared-error model.
- `latency_model_hgbmono_p50.pkl`: median-quantile model.
- `summary.json`: seed, split sizes, skipped counts and held-out MAE (ms)/MAPE (%). <!-- codespell:ignore mape -->

Both models predict seconds. `l2` names the squared-error loss, not an L2 cache;
`p50` estimates the median. Compare them on independent data for your workload.

By default, each collection directory is split 50/50 with seed 42 (single-row
cases stay in training). Metrics evaluate this split; saved models are then
refitted on all input rows. This measures interpolation within collected cases.
Use repeatable `--test-data-root` arguments for independent evaluation cases;
those rows never enter fitting or the saved models. Same-directory overlaps are
rejected. Copied data is not detected: keep independent collections separate.
Choose one loss with `--loss l2` or `--loss p50` if needed.

```bash
sglang-simulator-train \
  --data-root /path/to/train-collection \
  --test-data-root /path/to/test-collection --out-dir /tmp/predictor
```

Point the simulator configuration at the selected bundle:

```json
{"predictor": {"name": "ml", "database_path": "/tmp/example-predictor/latency_model_hgbmono_l2.pkl"}}
```

Use the same scikit-learn version for training and inference, and load only
trusted joblib files. Before production use, validate the selected model on
independent step data and complete simulator workloads.
