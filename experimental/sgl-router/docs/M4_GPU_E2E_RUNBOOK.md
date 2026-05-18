# M4 GPU E2E Runbook

This runbook describes how to run the sgl-router M4 acceptance tests on a
real-GPU box. The tests live at `experimental/sgl-router/e2e/m4_acceptance/`
and are skipped automatically on CPU-only hosts.

## What this validates

1. **`/metrics` Prometheus endpoint** is served and labelled per worker.
2. **Cache-aware-zmq convergence** — same-prefix requests land on the same
   prefill worker (≥8/10).
3. **PD-mode decode affinity** — chat responses carry `x-sgl-decode-url`.
4. **No-prefill failure mode** — frozen prefill workers → 503 with the
   `no_prefill_workers_available` error code.
5. **Stale-request expiry** — frozen worker → 504 with the
   `stale_request_expired` error code.

The full SGLang contract — `kv_events` block on `/server_info` — is
validated indirectly: the cache-aware-zmq convergence test only works
if the router successfully subscribes to each worker's kv-event publisher,
which only works if the worker advertised the resolved block at startup.

## Prerequisites

- 4× H200 GPUs (the convergence + affinity tests max out at 4 workers
  total; smaller suites need fewer — gpu_allocator skips tests that
  exceed the live GPU count).
- CUDA 12.4+ + nvidia-smi visible on the host.
- Disk: ~100 GB cache for Qwen3-0.6B + Llama-3.x weights.
- Build deps: Rust 1.82+, Python 3.10+ with sglang installed.

## Step 1 — Provision a GPU devbox

For RadixArk infra (other clusters: replace with your own tooling):

```bash
rx devbox acquire \
  --gpu h200 \
  --count 4 \
  --name kangyan-zhou-h200-4gpu-m4-e2e \
  --ttl 4h
```

The devbox typically takes ~3 minutes to come up. Note the SSH alias
(usually `<name>.devbox.rdxa`).

## Step 2 — Sync the worktree

From your local machine:

```bash
rsync -avz \
  --exclude target \
  --exclude .git \
  --exclude '**/__pycache__' \
  --exclude '**/.pytest_cache' \
  --exclude e2e_test \
  ~/sglang_workspace/feat/sgl-router-m1-http-tokenizer/ \
  kangyan-zhou-h200-4gpu-m4-e2e.devbox.rdxa:~/sgl-router-m4/
```

Total transfer is ~150 MB; takes 30–60s on a typical link.

## Step 3 — Build sgl-router (release)

On the devbox:

```bash
cd ~/sgl-router-m4/experimental/sgl-router
cargo build --release
```

First build is ~4–6 minutes; subsequent builds reuse the target cache and
take <30s. The binary lands at `target/release/sgl-router`.

## Step 4 — Install SGLang + pytest

On the devbox (inside an isolated Python env or container):

```bash
cd ~/sgl-router-m4
pip install -e python/
pip install pytest httpx
```

Validate with `python3 -c "import sglang; print(sglang.__version__)"`.

## Step 5 — Pre-warm the model cache

The first test run will download Qwen3-0.6B from HF; pre-warm to keep
each worker's cold-start under 30 seconds:

```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-0.6B')
"
```

## Step 6 — Run the M4 acceptance suite

```bash
cd ~/sgl-router-m4/experimental/sgl-router/e2e
pytest m4_acceptance/ -v -s --tb=short 2>&1 | tee ~/m4-e2e.log
```

Expected output: 4 tests, ~10 minutes wall-clock total (model loads
dominate). Each test is marked `@pytest.mark.real_gpu` so a missing
GPU is a clean skip rather than a confusing failure.

To run a single test (e.g. when debugging):

```bash
pytest m4_acceptance/test_cache_aware_zmq_convergence.py -v -s
```

## Step 7 — Capture `/metrics` snapshot

After the convergence test runs, grab the `/metrics` text for the
artifact log:

```bash
curl -sf http://127.0.0.1:<router_port>/metrics > ~/m4-metrics.txt
```

The port is dynamic per Gateway instance; the test logs print it as the
router starts.

Save both `~/m4-e2e.log` and `~/m4-metrics.txt` to the repo as
`experimental/sgl-router/M4_GPU_E2E_RESULTS.md` (pytest output) +
`experimental/sgl-router/M4_GPU_E2E_METRICS.txt` (raw metrics).

## Step 8 — Tear down

```bash
rx devbox release kangyan-zhou-h200-4gpu-m4-e2e
```

**Always** release the devbox even on failure — GPUs are expensive.

## Troubleshooting

### Worker doesn't come up within 10 minutes

1. Check `nvidia-smi` — is the GPU actually free?
2. Tail the worker stdout: `tail -f /tmp/sglang-worker-<port>.log`
   (the conftest captures stdout via `subprocess.PIPE`; redirect to a
   file by setting `SGL_E2E_KEEP_LOGS=1`).
3. Common: HF cache miss + slow network on the devbox. Pre-warm the
   cache before retrying.

### `cargo build` fails with `link.exe not found` or `cc not found`

The devbox image is missing build essentials. Install with:

```bash
sudo apt-get install -y build-essential pkg-config libssl-dev
```

### Convergence test fails at 6/10 instead of 8/10

The 8/10 threshold accounts for the timing race between request 1's
response and request 2's selection (the kv-event publisher has to flush
to the router before the policy can pin the second request). If you
see 6/10 reproducibly:

1. Check the ZMQ subscriber's lag — is it >100ms?
2. Inspect `sgl_router_overlap_blocks{model_id="..."}` — are the
   `le="0"` and `le="1"` buckets non-empty? That would mean the radix
   tree isn't seeing the prefix.
3. The most common cause is `kv_events_config` not being passed to
   `sglang.launch_server`. Confirm by curling
   `http://<worker>:<port>/server_info | jq .kv_events`: the block
   MUST be present (Patch 1 in this M4 stack).

### Stale-request test never gets a 504

The stale_request_timeout defaults to 5 minutes. To shorten for the
test, the `Gateway` config emits a `[active_load] stale_request_timeout_secs = 5`
block — verify this surfaced in the running router by checking
`/metrics` for `sgl_router_stale_requests_total` (the family is always
present; non-zero values mean the janitor fired).

## Cost ceiling

A 4×H200 devbox costs roughly $30–50/hour. Phase D should fit in
<2 hours of wall time:

- Acquire: 3 min
- Sync + build + install: 10 min
- Pre-warm cache: 5 min
- Test run (4 acceptance + 2 chat_completions smoke): 20 min
- Capture + release: 5 min

Total: ~45 minutes if everything works the first time, 90 minutes with
one failure / retry loop.
