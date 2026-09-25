# Rawsystemone validation and benchmarks

This client harness targets an **already-running** SGLang server. It does not
start a second serving process, load a model into the endpoint, or train weights.
See [the implementation report](IMPLEMENTATION.md) for what was actually run.

## CPU contract tests

From the repository root, with Python 3.10+ and the server's Pydantic 2,
FastAPI, httpx, and msgspec dependencies installed:

```bash
python test/registered/unit/entrypoints/test_rawsystemone.py
```

The suite imports the standalone endpoint without CUDA dependencies. It tests
the actual scheduler row-construction/chunk-assembly methods in isolation,
including repeated IDs and the last valid vocabulary ID. Its gated dispatcher
tests establish concurrent submission, work-conserving refill, global bounds,
failure cleanup, and cancellation. They do not establish numerical GPU parity.

## Live parity and independent oracle

Use a Linux NVIDIA CUDA deployment with the dependencies required by this
checkout. Start with a small unquantized checkpoint, float32, one tensor/data
parallel rank, one tokenizer worker, and native radix caching. For example,
use `HuggingFaceTB/SmolLM2-135M-Instruct` (Llama architecture). Set
`MODEL_REVISION` to an immutable checkpoint commit; do not compare moving
revisions. A gated model may require the usual `HF_TOKEN`; this example is public.
The commands below are provided for hardware validation and were not run here.

```bash
export MODEL_PATH=HuggingFaceTB/SmolLM2-135M-Instruct
export MODEL_REVISION=YOUR_IMMUTABLE_MODEL_COMMIT
python -m sglang.launch_server \
  --model-path "$MODEL_PATH" --revision "$MODEL_REVISION" \
  --dtype float32 --attention-backend torch_native \
  --tp 1 --dp 1 --context-length 8192 --disable-cuda-graph \
  --rawsystemone-max-candidates-per-batch 16 \
  --rawsystemone-max-inflight-batches-per-request 2 \
  --log-level info > /tmp/rawsystemone-server.log 2>&1
```

In another terminal:

```bash
RAWSYSTEMONE_URL=http://127.0.0.1:30000 \
  python test/manual/test_rawsystemone.py
```

The test loads a **test-only CPU** float32 teacher-forcing model with the same
checkpoint/revision. The server's complete token IDs must exactly equal the
oracle's native prefix IDs followed by suffix IDs encoded without added special
tokens. It computes full-vocabulary
`log_softmax(logits[t-1])[token[t]]`, without the service's prefix bookkeeping.
It checks the fixed boundary, Unicode, whitespace, first-token coverage,
unequal lengths, prefix subtraction, stable softmax, duplicates, independence of
conditional sums, permutation, split batches,
cache flushing, schema errors, and legacy generation/chat/score endpoints.
Set `SGLANG_API_KEY` to test an authenticated server too.

Initial **provisional acceptance thresholds**, not empirically established
tolerances: same-backend absolute per-token/mean difference `2e-4`, independent
CPU float32 oracle `5e-4`, relative tolerance zero. The suite reports maximum
per-token and mean-score differences. Investigate failures before changing
`RAWSYSTEMONE_PARITY_ATOL` or `RAWSYSTEMONE_ORACLE_ATOL`; record any justified
backend/dtype-specific change. These thresholds have not been validated here.

Repeat after restarting the same server with:

- batch concurrency 1, 2, and 4; native single-batch and split-batch limits;
- `--disable-radix-cache`;
- `--chunked-prefill-size 256`, then the production chunk size;
- each intended tensor-parallel size on appropriate hardware;
- a no-BOS tokenizer checkpoint, such as a supported Qwen2 configuration,
  alongside the BOS behavior of the chosen Llama tokenizer.

Do not claim these modes are validated from a single configuration's result.
Disk/tensor/distributed/IPC update handling and cleanup are covered locally by
control-message and mocked lifecycle tests. Real HTTP disconnect/GPU cleanup,
overload under mixed traffic, and real weight-update parity remain required
deployment acceptance checks.

## Performance matrix

Install `httpx` on the benchmark client. Record the **server checkout** commit
and dirty patch, checkpoint and tokenizer revisions, GPU model/count, precision,
driver/CUDA versions, and server settings. The harness filters `/server_info`
through a safe allowlist rather than saving authentication fields.

```bash
python benchmark/rawsystemone/bench_rawsystemone.py \
  --url http://127.0.0.1:30000 \
  --server-commit YOUR_SERVER_COMMIT \
  --model-revision YOUR_IMMUTABLE_MODEL_COMMIT \
  --hardware 'GPU_MODEL x COUNT; DRIVER_VERSION; CUDA_VERSION' \
  --server-log /tmp/rawsystemone-server.log \
  --prefix-tokens 128 1024 4096 --options 2 8 32 96 128 \
  --suffix-tokens 1 4 16 --concurrency 1 2 4 \
  --warmups 2 --repetitions 20 --output /tmp/rawsystemone-results.json
```

Prefix/suffix target lengths are approximate; the output records actual native
candidate token lengths. Reduce lengths for smaller model context windows.
Run again with `--mixed-lengths`, `--cache cold` on an idle server, and
`--background-traffic 2` for concurrent ordinary generation. Cache-disabled
measurements require restarting with `--disable-radix-cache`. Do not flush the
cache on a shared production server. `--cache warm` means the server remains
warm; default rawsystemone salts still isolate KV entries between parents.
Within each parent, all branches reuse the same salt and common prefix.

The harness compares sequential full scoring, one native full-scoring batch,
split full-scoring batches with concurrency 1/2/4, and the optimized endpoint
with its current **server-configured** batch/concurrency limits. Restart with
optimized concurrency 1/2/4 and compare result files. Client-side reference
concurrency flags do not reconfigure the endpoint. Every configuration checks
weights against independently scored full token sequences, slicing out suffix
targets before computing conditional means and softmax. Reference token IDs come
from `/v1/tokenize` with the prefix/option special-token settings above. These
tokenization calls are outside the reference timing loop.

Output includes median/p95 HTTP latency, requests/candidates per second, logical
tokens, maximum score differences, and matching per-request server diagnostics
when `--server-log` is supplied. These times include HTTP transport and
aggregation; full-reference calls transfer more token-score data than the
normal endpoint. They are not isolated kernel timings. Add native profiler
traces for scheduler/device batch behavior and direct prefill-work measurements.

Acceptance requires native cache hits and lower redundant uncached-input work
on long shared prefixes, plus observed multiple in-flight submissions when
capacity permits. A latency result alone, mocked cache counts, or Python
`async` does not establish GPU prefix reuse or device overlap. No universal
speedup is assumed. The single-token gather fast path and optional LLM compiler
are deliberately outside this patch.
