# Testing Decode-Verify-Rollback

Use the launcher and parity client in this directory to test DVR startup, target
logprob parity, state lifecycle, and decode throughput. Target full attention
supports Triton and FlashAttention 3. GDN tests use Triton linear attention,
page size 64, and FP32 recurrent state. Radix may be enabled or disabled; GDN
recurrent checkpoints are published only at exact chunk boundaries.

## Launch

The launcher prints the expanded command and does not activate an environment,
select GPUs, or modify compiler caches.

```bash
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
export MODEL_PATH=/models/Qwen3.5-35B-A3B
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TP_SIZE=4
export ATTENTION_BACKEND=fa3
export MAX_RUNNING_REQUESTS=1
export CONTEXT_LENGTH=16384
export MAX_TOTAL_TOKENS=16384
export CUDA_GRAPH_BS="1"
export CUDA_GRAPH_MAX_BS_DECODE=1

SERVER_MODE=self test/manual/dvr/launch_server.sh
```

`SERVER_MODE` accepts `normal`, `deterministic`, `self`, `eagle`, and `dflash`.
EAGLE/MTP and DFlash require a draft checkpoint:

```bash
SERVER_MODE=eagle DRAFT_MODEL_PATH="$MODEL_PATH" DRAFT_TOKENS=4 \
  test/manual/dvr/launch_server.sh

SERVER_MODE=dflash DRAFT_MODEL_PATH=/models/Qwen3.5-35B-A3B-DFlash \
  test/manual/dvr/launch_server.sh
```

The DVR-EAGLE path currently supports full-attention draft models. The launcher
uses sync scheduling by default (`DISABLE_OVERLAP=1`), which is also the
recommended DVR performance configuration. Set `DISABLE_OVERLAP=0` only for a
matched overlap comparison. Use
`DISABLE_RADIX_CACHE=1` or `DISABLE_CUSTOM_ALL_REDUCE=1` only for controlled
comparisons.

## Target Logprob Parity

Use the same non-greedy sampling configuration for matched standard
(`SERVER_MODE=normal`), deterministic, and DVR runs. The defaults are
temperature 1.0, top-p 0.95, top-k 20, and sampling seed 2026. Set
`SGLANG_RETURN_ORIGINAL_LOGPROB=True` when comparing original selected-token
logprobs.

Start a DVR server and generate an artifact:

```bash
python3 test/manual/dvr/check_logprob_parity.py dvr \
  --artifact /results/qwen35_fa3_smoke.json \
  --prompt-length 64 \
  --check-radix-reuse
```

Restart the same target with `SERVER_MODE=deterministic`, without a speculative
algorithm, and replay the forced sequences:

```bash
python3 test/manual/dvr/check_logprob_parity.py det \
  --artifact /results/qwen35_fa3_smoke.json
```

The client checks the serving-side numerical contract by comparing
selected-token original logprobs in three places:

1. repeated deterministic target prefill;
2. DVR target decode versus DVR forced target prefill;
3. DVR target decode versus ordinary deterministic target prefill.

End-to-end trainer/generator parity requires a separate training-side check
against the same forced token sequence and numerical contract.

The smoke profile also runs an independent request without returned logprobs.

The release profile runs the same checks over at least 10,000 generated tokens
and rejects truncated responses:

```bash
python3 test/manual/dvr/check_logprob_parity.py dvr \
  --artifact /results/qwen35_fa3_release.json \
  --qualification-profile release \
  --logged-only \
  --prompt-length 64

# Restart with SERVER_MODE=deterministic first.
python3 test/manual/dvr/check_logprob_parity.py det \
  --artifact /results/qwen35_fa3_release.json \
  --qualification-profile release
```

`--logged-only` omits the no-logprob control from the long run. The target model,
revision, dtype, quantization, TP, attention backends, and page size must match
between the DVR and deterministic phases.

The smoke matrix covers prompt lengths 1/63/64/65, output lengths 65/512/1024,
logprob on/off, intended batch sizes, Radix on/off, generated-prefix reuse, and
request lifecycle checks. Every performance-qualified self/EAGLE/DFlash,
Triton/FA3, and sync/overlap configuration must also pass the 10,000-token
parity checks.

## Lifecycle

### Radix Reuse

`--check-radix-reuse` verifies that a 64-token prompt followed by 512 generated
tokens retains the legal 512-token GDN boundary and that continuation from the
cache hit passes both parity checks. It expects the default 64-token chunk unless
`--radix-chunk-size` is changed.

### Online Weight Update

Online DVR updates support only address-stable, shape-preserving in-place loads
with `recapture_cuda_graph=false` and `flush_cache=true`. CUDA Graph recapture is
not supported.

Validate an update by starting from a finite perturbed checkpoint, loading the
correct checkpoint, and requiring both a visible pre/post change and the normal
parity checks:

```bash
python3 test/manual/dvr/check_logprob_parity.py dvr \
  --artifact /results/qwen35_weight_update.json \
  --prompt-length 64 \
  --hot-update-checkpoint /models/Qwen3.5-35B-A3B

# Restart the correct checkpoint with SERVER_MODE=deterministic first.
python3 test/manual/dvr/check_logprob_parity.py det \
  --artifact /results/qwen35_weight_update.json
```

The startup checkpoint must preserve architecture, parameter names, shapes, and
dtypes. Perturb a finite weight value rather than introducing NaNs or malformed
tensors. The server log must contain no additional CUDA Graph capture during
the update.

### Request Parameters

DVR requests must leave `frequency_penalty`, `presence_penalty`,
`repetition_penalty`, and `min_new_tokens` at their default values. Grammar-
constrained decoding is also outside the initial request contract.

## Performance

Use fresh processes and keep the model, TP, graph, cache, dataset, concurrency,
and sampling settings identical. Measure decode throughput from streaming time
per output token (TPOT), excluding TTFT:

```text
TPOT = (request latency - TTFT) / (generated tokens - 1)
decode tokens/s = 1000 / TPOT_ms
```

Do not pass `--disable-stream`. Use at least 10,000 output tokens so TTFT, the
first speculative block, and request-finalization effects are negligible. Leave
`--return-logprob` unset for timed runs.

```bash
python3 -m sglang.benchmark.serving \
  --backend sglang \
  --host 127.0.0.1 \
  --port 30000 \
  --dataset-name random \
  --num-prompts 3 \
  --random-input-len 64 \
  --random-output-len 10000 \
  --random-range-ratio 1 \
  --max-concurrency 1 \
  --request-rate inf \
  --seed 42 \
  --temperature 1.0 \
  --top-p 0.95 \
  --extra-request-body \
    '{"sampling_params":{"temperature":1.0,"top_p":0.95,"top_k":20,"sampling_seed":2026,"max_new_tokens":10000,"ignore_eos":true}}' \
  --warmup-requests 1 \
  --output-details \
  --output-file /results/dvr_tpot.jsonl
```

Run one discarded full-length warmup followed by three measured requests.
Report median TPOT for standard decode, deterministic decode, and DVR together
with mean output per verification round and `det_tpot / dvr_tpot`. Reject a run
unless every request returns exactly 10,000 tokens and no timed run triggers
compilation.

Compare sync and overlap only with matching baselines. Sync is recommended: one
DVR transaction already amortizes host scheduling across a draft/verify block,
while generic overlap has little remaining CPU latency to hide and adds its own
FutureMap, event, and publish scheduling. For concurrency greater than one,
report both per-request TPOT and aggregate output throughput; `1000 / TPOT_ms`
is not aggregate serving throughput.

## Unit Tests

```bash
PYTHONPATH=python python3 -m pytest -q \
  test/registered/unit/layers/attention/dvr \
  test/registered/unit/speculative/dvr
```

Also run the focused shared-path tests changed by the PR, `git diff --check`,
Python compilation, and `bash -n test/manual/dvr/launch_server.sh`. Real-model
tests remain required for graph, collective, acceptance, and throughput
behavior.
