# Branch vs Main Numerical-Divergence Test

Compares per-step logit/logprob distributions between a **record** run on
`main` and a **verify** run on a feature branch. The verify run is forced
(via the `forced_token_ids` debug feature) to emit exactly main's token
sequence at every step, so prefill and decode are both exercised on the
branch with comparable inputs and small kernel differences show up as
small distribution drift instead of catastrophic prefix divergence.

## Prerequisites

The test relies on the `forced_token_ids` debug feature in sglang core:
- `SamplingParams.forced_token_ids` / `forced_token_ids_path`
- `SamplingBatchInfo.prepare_forced_next_token_ids`
- `Sampler.forward` override step

These must be present on **both** branches under test (they land on `main`
first as a generic feature; the branch under test inherits them via merge).

Set `SGLANG_ENABLE_FORCED_TOKEN_IDS=1` in the server's environment for both
phases. Without the env var, `forced_token_ids` requests are rejected at
admission with no disk I/O.

## Workflow

```bash
# === Phase 1: record on main ===
git checkout main
SGLANG_ENABLE_FORCED_TOKEN_IDS=1 \
python -m test.manual.branch_compare.run \
    --mode record \
    --artifact-dir /tmp/run1/main \
    --model-path /models/llama3-70b \
    --tp-size 8 \
    --eval-name gpqa --num-examples 16 \
    --max-new-tokens 1024 \
    --topk-logprobs 128

# === Phase 2: verify on branch ===
git checkout my-feature-branch
SGLANG_ENABLE_FORCED_TOKEN_IDS=1 \
python -m test.manual.branch_compare.run \
    --mode verify \
    --artifact-dir /tmp/run1/branch \
    --record-dir   /tmp/run1/main \
    --model-path /models/llama3-70b \
    --tp-size 8 \
    --max-new-tokens 1024 \
    --topk-logprobs 128

# === Compare ===
python -m test.manual.branch_compare.compare \
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

## Output

In `--out-dir`:
- `summary.json` — aggregate cosine / abs-diff / rel-diff (mean, p50, p90, p99, min, max), per-prompt rollup, list of any steps where verify-side `output_ids` deviated from record-side.
- `histograms.txt` — ASCII bin charts (cosine, abs-diff, rel-diff, common-top-K), also printed to stdout.

## Limitations

- Disagg and speculative decoding are not yet wired through. Use single-mode
  TP-only servers for now.
- The forced-tokens path-loader reads any path the (trusted) caller provides.
  Don't enable `SGLANG_ENABLE_FORCED_TOKEN_IDS=1` on a server exposed to
  untrusted clients.
- PD-disagg caps top-K logprobs at 128. The script defaults to 128 to stay
  safe; raise it for non-disagg deployments.
