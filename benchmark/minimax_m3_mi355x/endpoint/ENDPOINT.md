# MiniMax-M3 external endpoint on 4x MI350X/MI355X: image + serving command

Everything below is on branch `M3-perf` of `github.com/kevin-mii/sglang` (this directory is `benchmark/minimax_m3_mi355x/endpoint/`).

## Image

`Dockerfile` overlays this branch's `python/sglang` tree, the aiter FlyDSL XCD-swizzle fix and the M3 tuned MoE table on
`lmsysorg/sglang-rocm:v0.5.19-rocm724-mi35x-20260911` (digest `sha256:e62c69db892add25bb6a82f50a61726895b0814014c360e8acdd61fabc93a5c6`;
sglang 822e73ccdd, an ancestor of this branch; aiter 4ad99832). The branch does not change `python/sglang/kernels/aot`, so the
base image's compiled sgl-kernel, aiter and Triton are exactly the stack these numbers were measured on.

```bash
bash make_context.sh /tmp/m3ctx            # build context from this checkout (13 MB)
bash /tmp/m3ctx/build_on_cpu_vm.sh /tmp/m3ctx   # rx CPU VM build + push to 976589843892.dkr.ecr.us-west-2.amazonaws.com/radixark/sglang-ext:<tag>
```

`build_on_cpu_vm.sh` follows the radixark `build-sglang-images-cpu-vm` skill: it needs `rx login --google` and the AWS `radixark`
profile on the machine that runs it (the GPU devbox has neither). The tag is `m3-mi355x-<date>-<sglang commit>`.

## Serving command

Models: `amd/MiniMax-M3-MXFP4` (quark MXFP4 checkpoint) and `Inferact/MiniMax-M3-EAGLE3-GQA`, as directories under `/models`.

There is no wrapper script — this is the whole thing. The image already sets `SGLANG_USE_AITER=1` and
`HIP_FORCE_DEV_KERNARG=1`, so only the non-default knobs are passed here:

```bash
docker run --rm -it --network host --device /dev/kfd --device /dev/dri --group-add video --ipc host --shm-size 64g \
  -v /models:/models \
  -e SGLANG_MINIMAX_M3_INDEX_TOPK_FREQ=1 \
  -e SGLANG_FORWARD_UNKNOWN_TOOLS=1 -e SGLANG_ENABLE_STRICT_MODEL_NAME=1 -e SGLANG_ENABLE_QUEUE_FULL_429=1 \
  -e SGLANG_M3_ALLOW_CUSTOM_AR=1 -e SGLANG_CUSTOM_AR_ONE_STAGE_MAX_BYTES=262144 -e ROCM_QUICK_REDUCE_QUANTIZATION=INT4 \
  -e SGLANG_TRITON_EXTEND_LONG_PREFIX=1 -e SGLANG_USE_AITER_EXTEND_LONG_PREFIX=1 \
  -e SGLANG_CHUNKED_PREFILL_FAIRNESS_RESERVE=0.5 -e SGLANG_TIMEOUT_KEEP_ALIVE=3600 -e NCCL_MIN_NCHANNELS=112 \
  <image> \
  sglang serve --model-path /models/MiniMax-M3-MXFP4 --served-model-name MiniMax-M3 --trust-remote-code \
    --tp-size 4 --host 0.0.0.0 --port 30000 --api-key <bearer key> \
    --kv-cache-dtype fp8_e4m3 --chunked-prefill-size 8192 --mem-fraction-static 0.9 \
    --speculative-algorithm EAGLE3 --speculative-draft-model-path /models/MiniMax-M3-EAGLE3-GQA \
    --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
    --speculative-attention-mode decode --triton-attention-num-kv-splits 64 \
    --cuda-graph-backend-prefill breakable --reasoning-parser minimax-m3 --tool-call-parser minimax-m3 \
    --enable-metrics --enable-cache-report --max-running-requests 48 --max-queued-requests 64 \
    --watchdog-timeout 3600
```

On a devbox started **from this image** (`rx devbox acquire --gpu mi350x --count 4 --image <image>`), drop the `docker run`
wrapper and run the `sglang serve` line with those variables exported.

**Acceptance-test shape:** delete the five `--speculative-*` flags. That is the `SPEC=none` configuration in the results
below — slower, but the only one measured to pass every vendor gate (aime25 non-stop 0.21% against the handbook's 0.5%).

**Omit `--api-key`** to run without auth (needed for `sglang.test.few_shot_gsm8k`, which sends no bearer token).

### Why each variable is there

| variable | default | why it is set |
|---|---|---|
| `SGLANG_MINIMAX_M3_INDEX_TOPK_FREQ=1` | `2` | **The one that matters.** At 2 or 4 the model drops the `="` of its tool-call tags past ~60K tokens. See the root-cause section below. |
| `SGLANG_FORWARD_UNKNOWN_TOOLS=1` | `False` | parse tool calls with no/unknown inventory (handbook) |
| `SGLANG_ENABLE_STRICT_MODEL_NAME=1` | `False` | unknown model name -> 404 (handbook) |
| `SGLANG_ENABLE_QUEUE_FULL_429=1` | `False` | full queue -> 429 rather than a stall (handbook) |
| `SGLANG_M3_ALLOW_CUSTOM_AR=1` | `False` | M3-on-ROCm disables custom all-reduce by default; this opt-in re-enables it so quick-reduce can accelerate the prefill all-reduce |
| `SGLANG_CUSTOM_AR_ONE_STAGE_MAX_BYTES=262144` | `0` | one-stage all-reduce threshold |
| `ROCM_QUICK_REDUCE_QUANTIZATION=INT4` | `NONE` | quick-reduce INT4 path |
| `SGLANG_TRITON_EXTEND_LONG_PREFIX=1` | `False` | long-prefix extend kernels |
| `SGLANG_USE_AITER_EXTEND_LONG_PREFIX=1` | `False` | aiter long-prefix extend |
| `SGLANG_CHUNKED_PREFILL_FAIRNESS_RESERVE=0.5` | `0.0` | decode fairness against chunked prefill |
| `SGLANG_TIMEOUT_KEEP_ALIVE=3600` | `5` | long generations must not be reaped |
| `NCCL_MIN_NCHANNELS=112` | unset | RCCL channel floor, inherited from the benchmark env (`../reproduce.sh`). **Not measured in isolation** -- unlike every other collective knob it has no row in `../OPTIMIZATIONS.md` section 3, and this config routes the hot all-reduces around RCCL anyway (custom AR at decode sizes, quick-reduce INT4 at >= 64 MB). Kept only because every measured run had it set. |

Dropped from the old `serve_endpoint.sh`, verified individually against `python/sglang/srt/environ.py`:
`SGLANG_USE_AITER` / `HIP_FORCE_DEV_KERNARG` (already `ENV` in the image);
`SGLANG_OPT_MINIMAX_M3_FP8_INDEX_CACHE` / `SGLANG_OPT_USE_MINIMAX_GLUON_PREFILL` (already default `True`);
`SGLANG_MINIMAX_OPT_USE_GLUON_PREFILL` / `SGLANG_ENABLE_TRITON_EXTEND_LONG_PREFIX` (**dead names** — read nowhere in the tree);
`HIP_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES` (redundant when the container is given exactly 4 GPUs; set them only when
carving 4 GPUs out of a larger box). `--tp` became `--tp-size`: `--tp` is not a declared alias and only resolved by
argparse prefix matching. `python -m sglang.launch_server` became `sglang serve`, which the module itself now recommends
in a `UserWarning`; it is a strict superset (same `load_plugins()`, same `kill_process_tree` cleanup, plus backend detection).

## Vendor quality check (MiniMax provider handbook 2026-08-15)

All runs: TP4 on 4x MI350X (gfx950) in this box, MXFP4 quark checkpoint, fp8 KV, the serving command above (mem 0.9), on the
branch head after the eval fixes from zcnrex/sglang `m3-deploy`. Full numbers and per-sample records: `/scratch/results/` on the box.

| Check | Result | Reference |
|---|---|---|
| m3_format_check text suite (161 cases) | 156 passed, 4 skipped, 1 xfailed, 0 failed (production config with an API key) | vendor suite; first run failed 5 (streamed `[SILENCE]` lost its `]`, no trace id on 4xx, unknown model accepted, two auth cases) -> fixed in this branch |
| Provider-Verifier pass@10 (102 prompts x 10) | Query-Success 100%, ToolCalls-Match 84.8%, Schema-Accuracy 98.0%, Error-Only-Reasoning 0%, Language-Following 100%, Scenario-Check 100% | official MiniMax-M3: 100 / 98.8 / 98.9 / 0 / 100 / 100 |
| aime25 pass@1 avg-of-16 (MiniMax harness prompt, T=1.0, top_p 0.95, 98304 max tokens) | EAGLE3: 85.42% +/- 4.19 (SEM 1.05), non-stop 1.25%, avg 16.5K tokens. No speculative decoding: 87.50% +/- 5.09 (SEM 1.27), non-stop 0.21%, avg 18.0K tokens | official endpoint 91.46% +/- 3.84 (SEM 0.96), non-stop 0.21%; handbook asks non-stop < 0.5% |
| GSM8K-500 (5-shot) | 0.862 (EAGLE3 and no-spec alike) | branch reference 0.85-0.89 |
| Overload / auth / model name | 200 concurrent 4K-token requests: 112 x 200 + 88 x 429; no key -> 401; unknown model -> 404 with `X-Request-Id` | handbook: failures at 120% load must be 429 |
| Steady decode, 24 streams over ~70K-token contexts | 2,475 tok/s aggregate (103 tok/s per stream), accept length 2.5 | handbook: P50 TPS > 60 at some load tier |

### What the ToolCalls-Match gap is

In the 15% of tool prompts that came back as plain `stop`, the model had emitted the call, but with the `name="` glue of the invoke tag
dropped or doubled (`<invoke name Read">`, `<invoke nameRead">`, `<invoke name name="Read">`, or a namespace token spliced into the
tag) after 60-80K tokens of context, and the strict parser dropped it. The parser now accepts those forms (commit `db61ee4589`; 38 of 40
raw invokes sampled from the failing prompts parse, the other two are unrecoverable). The handbook asks providers for exactly this
leniency. Whether the malformed tags themselves come from the MXFP4 checkpoint or from the speculative verify path is what the
per-config probe below measures.

### What the aime25 gap is

Speculative decoding costs at most ~2 points on the full set (within one SEM) but 14/112 on the seven hardest problems, and it triples the
non-stop rate (1.25% vs 0.21%, the handbook wants < 0.5%). Turning off the index top-k sharing across layers did not help (54/112 vs
50/112 on the hard set), so that optimization is not the cause. The engine without speculation still sits ~4 points under the official
endpoint, which points at the MXFP4 checkpoint; the MXFP8 checkpoint is probed below.

### Per-config probe: one verifier loop (102 prompts) + raw sampling of the four long-context tool prompts (6 samples each)

All with the lenient parser. "raw invokes" counts `<invoke ...>` tags in the model's raw text: well-formed `name="X">` vs anything else.

| Config | ToolCalls-Match | Schema-Acc | raw invokes well-formed / malformed | EAGLE3 accept length |
|---|---:|---:|---:|---:|
| MXFP4, EAGLE3 GQA, index top-k shared across 4 layers (ATOM's config) | 0.970 | 0.952 | 10 / 32 | 2.65 |
| same, no speculative decoding | 0.910 | 0.947 | 20 / 22 | - |
| same, EAGLE3 with `--speculative-use-rejection-sampling` | 0.970 | 0.939 | 19 / 25 | 2.67 |
| MXFP8 checkpoint, no speculative decoding | 0.920 | 0.883 | 20 / 32 | - |
| MXFP8 checkpoint, EAGLE3 | 0.950 | 0.925 | 19 / 39 | 2.68 |
| MXFP4, no spec, INT4 quick-reduce off | 0.950 | 0.950 | 14 / 24 | - |
| MXFP4, no spec, bf16 KV + bf16 index cache | 0.930 | 0.910 | 26 / 19 | - |
| MXFP4, no spec, Gluon sparse prefill off | 0.930 | 0.936 | 32 / 23 | - |
| MXFP4, no spec, all of the above off + index top-k share frequency 1 | 0.990 | 0.964 | 48 / 0 | - |
| MXFP4, no spec, **only** index top-k share frequency 1 | 0.970 | 0.976 | 48 / 0 | - |
| MXFP4, EAGLE3, index top-k share frequency 2 | 0.990 | 0.976 | 39 / 5 | 2.94 |
| **MXFP4, EAGLE3, index top-k share frequency 1 (production)** | **0.990** | **0.988** | **54 / 0** | **3.07** |

The Inferact MHA draft (`Inferact/MiniMax-M3-EAGLE3`, 64 KV heads) did not fit next to the target at mem-fraction 0.9 (its draft KV
cache is 16x the GQA one); not pursued. Single-loop match rates carry about +/-3 points of noise; the raw invoke counts are the signal.

**Root cause:** sharing the sparse-attention index top-k across 4 layers (`SGLANG_MINIMAX_M3_INDEX_TOPK_FREQ=4`, ATOM's
`index_topk_freq`) is fine on GSM8K and aime25 but corrupts structured output once the context passes ~60K tokens: the model drops
the `="` of its tool-call tags. Neither the checkpoint (MXFP4 vs MXFP8), speculative decoding, INT4 quick-reduce, the fp8 caches nor the
Gluon prefill is the cause. The endpoint therefore runs frequency 1 (the AgentX benchmark configs keep 4 for their published numbers).
Speculative decoding on top of frequency 4 made it worse (32/42 vs 22/42 malformed); at frequency 1 it is clean and accepts more.

### Final production config: results

The serving command above (EAGLE3 GQA 3 steps / 4 draft tokens, mem 0.9, `SGLANG_MINIMAX_M3_INDEX_TOPK_FREQ=1`, lenient tool parser).

| Check | Result | Reference |
|---|---|---|
| Provider-Verifier pass@10 (10 x 102 prompts) | Query-Success 100%, ToolCalls-Match 97.8% (loops 0.96-0.99), Trigger-Similarity 99.4%, Schema-Accuracy 99.4%, Error-Only-Reasoning 0%, Language-Following 100%, Scenario-Check 90% (the single scenario prompt missed in 1 of 10 loops) | official MiniMax-M3: 100 / 98.8 (loops 0.98-0.99) / - / 98.9 / 0 / 100 / 100; vendor thresholds: 100 / ~98 +/-1 / >=98 / >=98 / 0 / >=40 / 100 |
| aime25 pass@1 avg-of-16 | 86.46% +/- 4.94 (SEM 1.23), **non-stop 2.29%** (11/480 reached the 98304-token cap), avg 15.2K tokens. Without speculative decoding (`SPEC=none`): 87.50% +/- 5.09, non-stop 0.21% | official endpoint 91.46% +/- 3.84 (SEM 0.96), non-stop 0.21% |
| m3_format_check text suite | 154 passed, 4 skipped, 1 xfailed; the 2 failures are the API-key cases on an eval server launched without a key (the keyed production launch passed all 156 earlier the same day) | 0 failures required |
| Steady decode, 24 streams over ~70K-token contexts | 2,213 tok/s aggregate (92 tok/s per stream) | 2,475 tok/s with the top-k shared across 4 layers: the fix costs ~11% decode throughput at this context length |
| Overload / auth / model name | as above (429 / 401 / 404 + `X-Request-Id`), unchanged | |

## Recommendation

Two launch shapes of the same image, both with the index top-k share frequency at 1 and the lenient parser:

- **Acceptance-test shape (`SPEC=none`)**: passes every hard gate measured here: tool-call match ~97%, aime25 87.5% with non-stop 0.21%
  (< 0.5% required), format suite clean. Decode is ~2.9K tok/s aggregate at 48 streams on 4 GPUs, i.e. ~60 tok/s per stream at that
  concurrency; run the vendor's 60/80/100/120% load ladder to pick `MAXRUN` so P50 TPS stays above 60.
- **Throughput shape (default, EAGLE3)**: ~2.2K tok/s at 24 streams over 70K contexts and 5.5K tok/s at 48 short streams, tool-call
  match 97.8%, aime25 86.5%, but non-stop 1.3-2.3% on aime25, which fails the handbook's < 0.5%. The EAGLE3 verify path is the open
  item: on the seven hardest aime25 problems it scores 50/112 against 64/112 without speculation, and rejection sampling, the
  draft choice and the index top-k frequency do not change that, so the remaining suspect is the target-verify attention path for
  the MiniMax sparse layers on very long generations (`pr/m3-eagle3-chain-verify`).

Not adopted, and why: index top-k sharing (quality, above); fp4 MoE activations at decode and forced draft acceptance (lossy by
construction, see `../OPTIMIZATIONS.md` section 4); online PTPC-FP8 dense layers (quality-neutral but measured slower on this build);
DSpark (per the owner's decision, EAGLE3 stays); the MHA EAGLE3 draft (does not fit at mem 0.9).
