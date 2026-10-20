# AR versus linear UNO performance on NPU

`run_perf.py` runs the offline math workload from
[PR #37667](https://github.com/sgl-project/sglang/pull/37667) using the existing
`run_math_eval.py` runner. It launches a fresh Engine process for every
configuration and compares AR against linear UNO. It supports NPU/Ascend and
CUDA/FA3; NPU tree mode is not implemented.

NPU UNO uses the internal `uno_ascend` LoRA backend. The published Qwen3 UNO
adapter is rank 128, beyond the general Ascend SGEMMV expand kernel's supported
ranks (8/16/32/64). This backend uses NPU-dispatched matrix multiplies for the
full adapter rank, including each fused QKV and gate/up slice; it keeps seed
rows base-only. It does not require FP8 or a rebuilt SGEMMV extension. CUDA
continues to use `uno_cublas`. Startup should report `Using uno_ascend as backend
of LoRA kernels` followed by `UNO LoRA initialized: backend=uno_ascend, rank=128`
for the published adapter. If it still reports `ascend`, check that the worker
imports the updated checkout.

The default sampling settings follow the PR: Qwen3-8B, BF16, context length
40960, maximum output 32768, temperature 1, top-k 50, top-p 0.95, seed 42.
GSM8K and MATH500 use one sample per problem; AIME uses ten. Dataset revisions,
chat templates, and grading are reused from `math_data.py` and `math_grader.py`.

Use an existing SGLang NPU environment with its matching `torch_npu` and
`sgl_kernel_npu`. The evaluation also requires `datasets`, `transformers`,
`huggingface_hub`, and `math_verify`. `--dry-run` does not load an Engine or
download models/data. Run commands from the repository root with Python 3.11+.

```bash
cd /Users/xk/codex/sglang
export MODEL_PATH=/models/Qwen3-8B
# Actual trained adapter directory containing adapter_config.json and weights.
export UNO_LORA_PATH=/models/uno-qwen3-8B/adapter
export DATA_ROOT=/data/uno-math
export RESULT_ROOT=/data/uno-results
```

Use the trained adapter compatible with the base checkpoint, such as the
`adapter/` subdirectory from `s-sahoo/uno-qwen3-8B`. A local directory avoids
ambiguity about the repository's adapter subdirectory. Dataset preparation
will download the pinned datasets unless they are already in `DATA_ROOT`.
Keep model/tokenizer revisions and all device/runtime settings fixed.

First inspect a small run without allocating an NPU:

```bash
python -m benchmark.uno.run_perf \
  --device npu --model-path "$MODEL_PATH" --uno-lora-path "$UNO_LORA_PATH" \
  --data-root "$DATA_ROOT" --output-dir "$RESULT_ROOT/smoke" \
  --benchmarks gsm8k --limit 8 --max-tokens 100 \
  --concurrencies 1 4 --widths 4 8 --graph-modes graph eager \
  --dry-run
```

Remove `--dry-run` to execute. This short run checks setup and produces heavily
length-capped reasoning answers; its accuracy is not a full math evaluation.

## Faster subset runs

Add `--quick` to use at most 64 randomly selected problems per dataset, one
completion per problem (including AIME), and at most 100 output tokens.
Explicit `--limit`, `--num-samples`, `--max-tokens`, and `--selection` override
these defaults. Without `--quick`, the original full-workload defaults remain.

```bash
python -m benchmark.uno.run_perf \
  --device npu --model-path "$MODEL_PATH" --uno-lora-path "$UNO_LORA_PATH" \
  --data-root "$DATA_ROOT" --output-dir "$RESULT_ROOT/quick" \
  --benchmarks gsm8k math500 aime25 \
  --concurrencies 1 64 --widths 8 --graph-modes graph \
  --quick --dataset-seed 42
```

This selects 64 GSM8K, 64 MATH500, and all 30 AIME25 problems. AR, UNO,
concurrency variants, and repeats use the same subset. Sampling is without
replacement, retains dataset order, and uses `--dataset-seed` independently of
the generation seed. `--limit N --selection random` also works without quick
mode, so output length and AIME sample count can be kept at their full defaults.
The underlying `run_math_eval` accepts the same selection and dataset-seed flags.
Results record selection settings and original dataset row indices.

For an even shorter initial check, use only GSM8K and fewer configurations:

```bash
python -m benchmark.uno.run_perf \
  --device npu --model-path "$MODEL_PATH" --uno-lora-path "$UNO_LORA_PATH" \
  --data-root "$DATA_ROOT" --output-dir "$RESULT_ROOT/quick-smoke" \
  --benchmarks gsm8k --concurrencies 1 8 --widths 8 --graph-modes graph \
  --quick --limit 16 --max-tokens 100
```

Quick mode does not shorten model loading or graph capture: every configuration
still starts a fresh engine. The three-dataset C=1/64 comparison starts 12
engines; the GSM8K C=1/8 check starts four. Thirty AIME completions cannot fill
C=64; the runner warns about underfilled concurrency. Use sufficient problems
or samples for a throughput study, rather than treating a small smoke run as
a saturated-concurrency benchmark. With 30 AIME problems, three samples give
90 completions, but increase runtime and still do not guarantee saturation.

Truncated reasoning and small subsets do not establish full-dataset accuracy
or production throughput. Inspect `length_capped_fraction` in the comparison
and increase `--max-tokens` for representative measurements. The 100-token
quick cap is a runtime bound, not an estimate of sufficient reasoning length.

Run the full PR-style AR and linear F=8 comparison at concurrency 1 and 64:

```bash
python -m benchmark.uno.run_perf \
  --device npu --model-path "$MODEL_PATH" --uno-lora-path "$UNO_LORA_PATH" \
  --data-root "$DATA_ROOT" --output-dir "$RESULT_ROOT/full" \
  --benchmarks gsm8k math500 aime25 \
  --concurrencies 1 64 --widths 8 --graph-modes graph \
  --context-length 40960 --max-tokens 32768 \
  --temperature 1 --top-k 50 --top-p 0.95 --random-seed 42 \
  --mem-fraction-static 0.8
```

The default is one run per configuration and no request warmup, matching the
PR's timing procedure. Lazy compilation or first-request costs inside
`Engine.generate` remain timed when warmup is zero. Full math evaluation can
take substantial time. For
steady-state measurements, add `--warmup-requests 8 --warmup-max-tokens 64`
and, for repeated observations, `--repeats 3`. Warmup and subsequent cache
flush are excluded from timing. Even repeats reverse the configuration order
to reduce systematic ordering effects. Warmup consumes RNG state; record it
when reproducing results.

To sweep width, concurrency, and decode graph settings, replace the relevant
options with:

```bash
--concurrencies 1 4 16 64 --widths 4 8 16 --graph-modes graph eager
```

Choose concurrency and context length that fit the device. `--base-gpu-id`
selects a logical device; the process inherits device visibility variables.
`--page-size` and `--mem-fraction-static` apply identically to AR and UNO when
specified. Runs are sequential, so AR and UNO do not contend for the NPU.
Use a new/empty output directory for each experiment. Engine failure or an
optional `--run-timeout SECONDS` stops the experiment; the runner cleans up
its own child process group. After every run, including a successful AR run,
the runner checks for live processes remaining in that run's private process
group. It sends SIGTERM, waits up to five seconds, then uses SIGKILL and waits
up to another 15 seconds if needed. A remaining live process stops the sweep
instead of launching another engine on the same device. Zombie processes are
excluded because they no longer hold device contexts. This check requires the
standard POSIX `ps` utility; it does not reset the NPU or terminate unrelated
process groups. Nonzero engine exits include the last 40 log lines in the
error. Completed results remain available. CPU-only lifecycle regressions can
be run with `python -m unittest benchmark.uno.test_run_perf -v`.

## Measurement and comparison

- Startup, graph capture, dataset preparation, tokenization, optional warmup,
  grading, and shutdown are outside the timed interval. The timer surrounds
  one blocking `Engine.generate` over the selected dataset. It includes
  prefill, decode, scheduling, and Engine output transfer; it is not isolated
  decode-kernel time.
- `C` is `max_running_requests`, a scheduler limit, not guaranteed live batch
  size. Small datasets, EOS, or memory pressure can reduce actual occupancy.
- Both AR and UNO use eager prefill. `graph` explicitly requests full decode
  graphs; `eager` disables decode graphs. The column records configuration,
  not a measured graph replay rate: inspect `engine.log` for capture and
  fallback behavior.
- NPU disables overlap scheduling for **both** modes because the current UNO
  implementation requires exact host sequence lengths. This is a controlled
  scheduling comparison, not a benchmark of each mode's independently tuned
  best deployment. CUDA retains default overlap scheduling.
- `tok/s = total output tokens / generation seconds`.
  `speedup = UNO tok/s / AR tok/s` within the same dataset, C, graph setting,
  and repeat. Token counts can differ between modes; compare accuracy and the
  fraction of length-capped outputs alongside throughput.
- UNO TPF follows the PR: `total output tokens / (2 * sum(spec_verify_ct))`.
  It counts both full-model passes per UNO cycle; the numerator includes the
  prefill seed, as in the PR. AR TPF is 1 by convention. TPF is not the
  throughput speedup and does not include prefill model forwards in its
  denominator. A UNO run with zero verify cycles is rejected as uninformative.
- `output_tok_s_per_configured_request` divides aggregate tok/s by configured
  C, following the PR. It is not measured per-request latency, TPOT, or ITL.
  This offline script does not report TTFT or P99 serving latency.
- Accuracy is graded using the shared math grader. An accuracy delta is
  descriptive, not proof of statistical equivalence. The PR discusses the
  top-k-first speculative versus joint AR top-k/top-p filtering distinction;
  identical 50/0.95 parameters do not prove identical distributions. Use
  `--top-p 1` to avoid that combined-filtering distinction, or
  `--temperature 0 --top-k 1 --top-p 1` for a greedy evaluation. Different
  forward shapes can still produce numerical differences. This runner does
  not enforce token-by-token greedy parity.

## Outputs

The root contains `plan.json`, `comparison.json`, `comparison.csv`, and
`comparison.md`. Each run directory contains the exact command, `engine.log`,
`summary.json`, `summary.md`, raw generations, grades, and scores. Comparison
rows include accuracy, output tokens/s, TPF, total generation seconds, speedup,
accuracy delta, and the fraction of outputs stopped at the length limit.
Repeated measurements remain separate rows.

For a single configuration, the existing runner now also accepts the NPU
options directly:

```bash
python -m benchmark.uno.run_math_eval \
  --device npu --attention-backend ascend --disable-overlap-schedule \
  --model-path "$MODEL_PATH" --uno-lora-path "$UNO_LORA_PATH" \
  --speculative-algorithm UNO --speculative-num-steps 1 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 8 \
  --benchmark gsm8k --max-running-requests 4 \
  --data-root "$DATA_ROOT" --output-dir "$RESULT_ROOT/single"
```
