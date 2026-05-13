# Branch vs Baseline Numerical-Divergence Test

Compares per-step logit/logprob distributions between a **record** run on
the **baseline** (typically `main`) and a **verify** run on a feature
branch. The verify run is forced (via the `forced_token_ids` debug
feature) to emit exactly the baseline's token sequence at every step, so
prefill and decode are both exercised on the branch with comparable
inputs and small kernel differences show up as small distribution drift
instead of catastrophic prefix divergence.

## Prerequisites

The test relies on the `forced_token_ids` debug feature in sglang core:
- `SamplingParams.forced_token_ids` / `forced_token_ids_path`
- `SamplingBatchInfo.prepare_forced_next_token_ids`
- `Sampler.forward` override step

These must be present on **both** branches under test.

Set `SGLANG_ENABLE_FORCED_TOKEN_IDS=1` in the server's environment for both
phases. Without the env var, `forced_token_ids` requests are rejected at
admission with no disk I/O.

## Workflow

Run from `sglang-source/test/manual/` (so `branch_compare` is importable
as a top-level package — `python -m test.manual.branch_compare.run`
collides with Python's stdlib `test` package and won't work):

```bash
cd sglang-source/test/manual

# === Phase 1: record baseline ===
git checkout main
SGLANG_ENABLE_FORCED_TOKEN_IDS=1 \
python -m branch_compare.run \
    --mode record \
    --artifact-dir /tmp/run1/main \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --tp-size 8 \
    --eval-name gpqa --num-examples 16 \
    --max-new-tokens 1024 \
    --topk-logprobs 128

# === Phase 2: verify on branch ===
git checkout my-feature-branch
SGLANG_ENABLE_FORCED_TOKEN_IDS=1 \
python -m branch_compare.run \
    --mode verify \
    --artifact-dir /tmp/run1/branch \
    --record-dir   /tmp/run1/main \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --tp-size 8 \
    --max-new-tokens 1024 \
    --topk-logprobs 128

# === Compare ===
python -m branch_compare.compare \
    --record-dir /tmp/run1/main \
    --branch-dir /tmp/run1/branch \
    --out-dir    /tmp/run1
```

`run.py` accepts every argument that `python -m sglang.launch_server`
accepts (via `ServerArgs.add_cli_args`), so pass any production server
config (`--quantization`, `--attention-backend`, `--mem-fraction-static`,
`--enable-deterministic-inference`, etc.) through directly.

If `--base-url` is set, the script attaches to an already-running server
instead of launching one. The user is responsible for ensuring the server
config matches between phases.

```bash
# Terminal A: launch the server once (e.g. slow multi-node config) and
# leave it running.
SGLANG_ENABLE_FORCED_TOKEN_IDS=1 \
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --tp-size 8 --port 30000

# Terminal B: point both phases at the running server. Server args are
# advisory in attach mode (recorded into meta.json but not used to launch).
cd sglang-source/test/manual
python -m branch_compare.run \
    --mode record \
    --artifact-dir /tmp/run1/main \
    --base-url http://127.0.0.1:30000 \
    --eval-name gpqa --num-examples 16 \
    --max-new-tokens 1024 --topk-logprobs 128

# then restart the server on the feature branch and:
python -m branch_compare.run \
    --mode verify \
    --artifact-dir /tmp/run1/branch \
    --record-dir   /tmp/run1/main \
    --base-url http://127.0.0.1:30000 \
    --max-new-tokens 1024 --topk-logprobs 128
```

## Output

In `--out-dir`:
- `summary.json` — aggregate cosine / abs-diff / rel-diff (mean, p50, p90, p99, min, max), per-prompt rollup, list of any steps where verify-side `output_ids` deviated from record-side.
- `histograms.txt` — ASCII bin charts (cosine, abs-diff, rel-diff, common-top-K), also printed to stdout.

## Limitations

- Disagg and speculative decoding are WIP. Use agg servers without speculative decoding for now.
- The forced-tokens path-loader reads any path the (trusted) caller provides.
  Don't enable `SGLANG_ENABLE_FORCED_TOKEN_IDS=1` on a server exposed to
  untrusted clients.
- PD-disagg caps top-K logprobs at 128. The script defaults to 128 to stay
  safe; raise it for non-disagg deployments.
