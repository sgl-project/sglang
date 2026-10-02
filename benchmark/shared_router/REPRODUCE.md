# Reproduce DeepSeek V4.1 shared router fusion

These commands compare the same SGLang source with horizontal fusion disabled
and enabled. They preserve TP4/EP1, DSpark block 5 and the 8K/1K workload.
Accuracy uses real acceptance; performance uses synthetic acceptance length
3.51. Never score synthetic outputs. Full qualification of this public-main
port is still in progress; these commands are not a claim of completed results.

## Environment and dependencies

Run inside a ROCm environment on four allocated MI355X GPUs. Preserve scheduler
GPU isolation; do not hardcode physical device IDs. On shared nodes allocate
64 CPU cores and 1 TiB host memory, use core binding, distinct server ports and
per-run compiler caches. Record co-location because it can affect latency.
The launcher disables SGLang's whole-node CPU affinity map.

The public environment under qualification is the immutable container
`lmsysorg/sglang@sha256:0a9bd5e897359bd26544f88845b78b910657a543c8746f2485b1927de5d2c351`:
Python 3.12.3, Torch 2.11.0+rocm7.2, HIP 7.2.26015 and Triton 3.7.0.
Use your site's container launcher inside the four-GPU allocation; do not
launch an unrestricted host Docker container that bypasses scheduler isolation.

The earlier development environment uses Python 3.12.3, Torch 2.14.0+rocm7.14
(`08187d9e0fba026dc8217405802ab5381dc88d90`), HIP 7.14.60850,
Triton 3.8.0 and FlyDSL 0.3.2. The private image used for the initial paired
jobs has SHA256 `46c40e5b35494650acbfe2e40894a8dc771b055a7e04f5c7548d23b3baf1bcd0`.
That image is not a public reproduction dependency.

Public AITER revision `e7d2453f25e5aaeea72a3a40b284ba453e026624`, pinned in
the public SGLang ROCm Dockerfile, passes the ten GPU tests, 24 checkpoint
cases and graph microbenchmark in this public image. Its public CK submodule
is `af9e1d1f1ae347c22feeb08fd2d42645075e0c5d`. The matched C2 pair completed
full GSM8K (native 1284/fused 1287 of 1319, zero request errors) and three-repeat
timing (median P50 TPOT 2.994574/2.923689 ms). The #42055 incremental pair also
completed; see README.md. Both forward sweeps are complete; incremental C1
reverse remains in progress. This is not a claim that every PR gate is complete.

Inside that container, make the pinned public dependency visible instead of
silently using an image fork. No custom AITER changes are needed:

```bash
git clone https://github.com/ROCm/aiter.git aiter-public
git -C aiter-public checkout e7d2453f25e5aaeea72a3a40b284ba453e026624
git -C aiter-public submodule update --init --recursive
export SR_AITER_REPO="$(realpath aiter-public)"
export PYTHONPATH="$SR_AITER_REPO"
export PYTHONNOUSERSITE=1 GPU_ARCHS=gfx950 MAX_JOBS=8
export SR_CACHE_ROOT="$(mktemp -d /tmp/shared-router-cache.XXXXXX)"
export AITER_JIT_DIR="$SR_CACHE_ROOT/aiter"
export SGLANG_JIT_CACHE_DIR="$SR_CACHE_ROOT/sglang"
export TRITON_CACHE_DIR="$SR_CACHE_ROOT/triton"
export FLYDSL_CACHE_DIR="$SR_CACHE_ROOT/flydsl"
export FLYDSL_RUNTIME_CACHE_DIR="$SR_CACHE_ROOT/flydsl-runtime"
python3 -c 'import aiter, os; from pathlib import Path; assert Path(aiter.__file__).resolve().is_relative_to(Path(os.environ["SR_AITER_REPO"])); print(aiter.__file__)'
```

Use a fresh cache root per arm/allocation; keep it stable across that arm's
accuracy/performance phases and record its location. The launcher prepends this
SGLang checkout to `PYTHONPATH` and preserves these cache paths while removing
inherited `SGLANG_*`/`AITER_*` experimental settings. Public dependencies and
private-development results must not be mixed in a matched performance pair.

Use this PR source and record `git rev-parse HEAD`, `git status --porcelain`,
import paths and package versions for SGLang, AITER, Torch, Triton and FlyDSL.
The preserved measured branch and current-main port are different revisions;
do not substitute one branch's results for the other.

## Obtain the inputs

Download `deepseek-ai/DeepSeek-V4.1-Flash` at revision
`dba1be0a40aa45a94ad051997016db3960a90277`, including its official tokenizer and
encoding files. Set `MODEL` to that snapshot directory.

```bash
git clone https://github.com/openai/grade-school-math.git gsm8k
git -C gsm8k checkout 3101c7d5072418e28b9008a6636bde82a006892c
git clone https://github.com/SemiAnalysisAI/InferenceX.git InferenceX
git -C InferenceX checkout 127c84be90f1536a92c1bd62f7f8b3b071d615fe
export DATASET="$PWD/gsm8k/grade_school_math/data"
export INFERENCEX="$PWD/InferenceX"
sha256sum "$DATASET/train.jsonl" "$DATASET/test.jsonl"
```

Expected SHA256 values are
`17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465`
for train and `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`
for test. There are 7,473 train and 1,319 test examples.

The supplied InferenceX adapter uses the checkpoint's official V4.1 framing.
It changes client encoding only, not model serving. The `vllm` backend name
selects the compatible HTTP completions protocol against SGLang.

## Run correctness tests

From the SGLang repository root:

```bash
python3 benchmark/shared_router/test_reproduction.py -v
python3 test/registered/amd/test_shared_router_gfx950.py -v
python3 benchmark/shared_router/checkpoint_test.py \
  --model "$MODEL" --output checkpoint-numerics.json
python3 benchmark/shared_router/bench.py --output shared-router-micro.json
```

Checkpoint tests use real weights/scales and synthetic activations; the
microbenchmark uses hot operands. Neither substitutes for model accuracy or
end-to-end performance.

## Measure full GSM8K with real acceptance

Launch a server in one terminal, within the GPU allocation. Use fusion `0` for
baseline and `1` for candidate, sequentially, retaining each server log.

```bash
bash benchmark/shared_router/launch_server.sh "$MODEL" accuracy 0 30000 \
  > baseline-accuracy-server.log 2>&1
```

Once `/health` responds successfully, run the client in another terminal in
the same environment. Use C from 1, 2, 4, 8, 16 or 32 and a fresh output path:

```bash
python3 benchmark/shared_router/gsm8k_eval.py \
  --dataset "$DATASET" --model-path "$MODEL" \
  --base-url http://127.0.0.1:30000 --mode full --concurrency 2 \
  --thinking thinking --reasoning-effort 100 --max-tokens 8192 \
  --context-length 1048576 --seed 0 --output-dir baseline-c2-accuracy
```

Stop that server before launching the candidate. Repeat with fusion `1` and
different log/output paths. Train examples 0–4 provide five-shot chain of
thought. Temperature is zero, top-p one; scoring requires a final `####`
numeric answer. All 1,319 questions count, including capped, missing-final or
failed responses as incorrect. Require zero request errors. Report paired
correct counts and score difference. The local loss tolerance is 0.5 percentage
points, not an upstream standard or statistical equivalence test.

## Measure fixed acceptance performance

Restart the server in performance mode, initially with fusion disabled:

```bash
bash benchmark/shared_router/launch_server.sh "$MODEL" perf 0 30000 \
  > baseline-perf-server.log 2>&1
```

After readiness, run:

```bash
bash benchmark/shared_router/run_perf.sh "$MODEL" "$INFERENCEX" 2 \
  http://127.0.0.1:30000 baseline-c2-perf
```

This performs one unreported warmup run and three measured repetitions.
Each has 160 requests (320 at C32), 2×C internal warmup requests, random
8192/1024 token budgets, range ratio 0.8, seed zero, ignore-EOS and no prefix
cache. Realized lengths are recorded rather than assumed to be exactly 8K/1K.
Repeat sequentially with fusion `1` for the candidate. Do not run A and B
simultaneously on the same GPUs. Repeat C1/C2 in reverse candidate-first order.

The collector checks request errors, token budgets, identical repetition
manifests and verification counters consistent with AL3.51. It reports both
the median and minimum of the three per-repeat P50 TPOT/ITL/TTFT statistics,
plus output tokens/s. These are not pooled percentiles. For latency,
`100 * (candidate / baseline - 1)` is negative for improvement. Report all
three samples and compare A/B request manifests, not just the best sample.

## Collect untimed device traces

After the three performance measurements, leave the server running and issue
the following probe separately. `TRACE_DIR` must be an absolute, new directory
visible and writable by the server. It does not need to be the same machine as
the HTTP client. Do not include this request in performance statistics.

The staged probe below has completed at C1/C2/C4/C8. Do not extrapolate it to
C16/C32: during qualification a C16 server stalled when the profiler restarted
between prefill and decode. The single-start, steady-decode protocol below
passed both arms at C16 and C32 on both public-main and incremental #42055
sources.
For readiness detection, `spec_verify_calls_total` is not a live per-step
counter in the pinned source: it increments when a request finishes. It is
suitable for the completed-request AL audit, not for triggering a profiler
while all requests must remain active.

```bash
export BASE_URL=http://127.0.0.1:30000 CONC=2
export TRACE_DIR=/absolute/server-visible/path/candidate-c2-trace
python3 - <<'PY'
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

def post(endpoint, payload):
    request = urllib.request.Request(
        os.environ['BASE_URL'] + endpoint,
        json.dumps(payload).encode(), {'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=600) as response:
        return response.read().decode()

print(post('/start_profile', dict(
    output_dir=os.environ['TRACE_DIR'], num_steps=8, activities=['CPU', 'GPU'],
    with_stack=False, record_shapes=True, profile_by_stage=True,
    profile_prefix='shared-router')))
def generate(index):
    return post('/generate', dict(
        input_ids=[1000 + index] * 8192,
        sampling_params=dict(temperature=0, max_new_tokens=256, ignore_eos=True)))
with ThreadPoolExecutor(max_workers=int(os.environ['CONC'])) as pool:
    responses = list(pool.map(generate, range(int(os.environ['CONC']))))
print('Probe requests completed:', len(responses))
PY
```

Wait for four `*TP-<rank>-DECODE.trace.json.gz` files (ranks 0–3); export can lag
behind the requests. Retain the compressed files and SHA256 hashes. In each
trace, correlate GPU kernel events with CPU `hipGraphLaunch` via their
`args.correlation` IDs. For fused C1/C2, require both `shared_router_projection`
and `_shared_down_router_topk` in at least two complete target replays per rank,
each with 40 calls of each stage. A partially exported final replay is not used
as proof; an incomplete interior replay requires investigation. Baseline must
not contain these fused stages. High-C traces may legitimately use native
fallback or include fused small-M tails. Kernel-name counts alone do not prove
which constexpr specialization ran or provide unbiased performance timings.
The probe uses synthetic token IDs solely for dispatch attribution, not accuracy.

### C16/C32 single-start steady-decode probe

Use this instead of the staged probe above, after timing is finished. Start
all requests before enabling profiling, then wait for a full active batch,
an empty queue and live sequence lengths beyond prefill. The pinned source
reports full sequence lengths in `decode_sum_seq_lens`, not SWA cache lengths.
The qualification harness validated this protocol at C16 and C32 on both
public-main and incremental #42055 arms. This self-contained
command uses the same requests, trigger and profiler payload; its payload was
also checked with CPU mocks. It is not a new timing measurement or a claim
that this exact standalone snippet was used by the qualification jobs.

```bash
export BASE_URL=http://127.0.0.1:30000 CONC=16
export TRACE_DIR=/absolute/server-visible/path/candidate-c16-steady-trace
python3 - <<'PY'
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

base = os.environ['BASE_URL']
conc = int(os.environ['CONC'])
assert conc in (16, 32)

def post(endpoint, payload):
    request = urllib.request.Request(
        base + endpoint, json.dumps(payload).encode(),
        {'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=600) as response:
        return response.read().decode()

def metric(text, name):
    values = [float(line.split()[-1]) for line in text.splitlines()
              if line.startswith(name + '{') or line.startswith(name + ' ')]
    assert values, 'Missing metric: ' + name
    return sum(values)

def generate(index):
    return post('/generate', dict(
        input_ids=[1000 + index] * 8192,
        sampling_params=dict(temperature=0, max_new_tokens=1024, ignore_eos=True)))

with ThreadPoolExecutor(max_workers=conc) as pool:
    futures = [pool.submit(generate, i) for i in range(conc)]
    deadline = time.monotonic() + 90
    while True:
        with urllib.request.urlopen(base + '/metrics', timeout=10) as response:
            snapshot = response.read().decode()
        assert not any(f.done() for f in futures), 'Probe ended before trigger'
        if (metric(snapshot, 'sglang:num_running_reqs') == conc
                and metric(snapshot, 'sglang:num_queue_reqs') == 0
                and metric(snapshot, 'sglang:decode_sum_seq_lens')
                    >= conc * (8192 + 32)):
            print('Trigger metrics:', snapshot)
            break
        assert time.monotonic() < deadline, 'No full-concurrency decode trigger'
        time.sleep(0.05)
    receipt = post('/start_profile', dict(
        output_dir=os.environ['TRACE_DIR'], start_step=1, num_steps=8,
        activities=['CPU', 'GPU'], with_stack=False, record_shapes=True,
        profile_by_stage=False, profile_prefix='steady-decode'))
    assert receipt.strip() == 'Start profiling.', receipt
    responses = [json.loads(f.result()) for f in futures]
assert all(r['meta_info']['completion_tokens'] == 1024 for r in responses)
print('Probe requests completed:', len(responses))
PY
```

On the pinned source, `start_step=1` clamps to the next forward. Wait for four
`steady-decode-*-TP-<rank>.trace.json.gz` files, without the staged `-DECODE`
suffix, and apply the same raw-hash and graph-correlation checks described
above. Both public-main C16 and C32 arms exported 16 graph launches per rank
with correlated GPU kernels and no fused launches in steady decode. A timeout or exported trace
alone is not proof of valid full-batch attribution. Keep trigger metrics and
server logs; do not restart an active run merely because observation timed out.

## Qualification beyond timing

Keep raw JSON, request manifests, logs, counters, source/environment receipts
and per-rank graph traces. C1/C2 require both fused stages to execute in graph
replay on every TP rank. Larger C can legitimately fall back except for short
tails. A flag being set does not prove engagement. Trace collection must be
outside the timed repetitions. Full-model startup and retained/peak memory
must also be compared, including the 5.625 MiB derived down weight per eligible
layer/rank. Startup logs and retained storage do not establish process peak
memory. The probe above collects traces; their engagement/fallback interpretation
must still be reviewed rather than treating successful export as qualification.

To assess #42055, compare off/on on one common source containing its producer
quantization changes. To compare #42011's folded shared expert, use a common
source and common sorting policy across normal, horizontal and folded arms.
The folded path changes weight quantization and needs its own accuracy gate.
Do not credit sorting or producer changes to horizontal fusion.
