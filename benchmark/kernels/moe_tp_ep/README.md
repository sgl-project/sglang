# MoE tensor parallelism versus expert parallelism on Ascend

This benchmark compares `Qwen3MoeSparseMoeBlock` with identical checkpoint weights
and **the same global token workload**. It targets BF16 `Qwen3MoeForCausalLM`
checkpoints with `model.safetensors.index.json`, on a single Ascend NPU node.
The validated model is `Qwen/Qwen3-30B-A3B`; other model families are untested.

## Workload and timing

For a global input of `N` tokens on `P` devices:

- Pure TP uses `--backend none --ep-size 1`. Every rank receives the same complete
  input; each expert's intermediate dimension is split across the devices.
- TP+EP uses `--backend deepep --ep-size P`. Rank 0 generates one global input with
  a fixed CPU seed and broadcasts it. The ranks then take disjoint contiguous
  token slices. Each expert resides on one device (internal MoE TP is 1).

Uneven splits put one extra token on each of the first `N % P` ranks. When `N < P`,
some source ranks receive zero tokens and still participate in communication.
No padding tokens enter the module. The sum of EP input lengths is exactly `N`.

Measurements use `torch.inference_mode()` and eager execution. The device Event
interval includes the gate, Top-K, experts and final all-reduce/combine. Input
broadcast/slicing and output gathering are outside this interval. Each iteration
aligns ranks before timing and synchronizes after the forward. Results include the
maximum device time across ranks, each rank's samples and host elapsed time.
These are module measurements; they exclude attention, scheduler overhead and
conversions between attention and MoE input layouts.

## Environment

Use a SGLang checkout and DeepEP wheel built for the same CANN, PyTorch, architecture
and device generation. On the original test environment, DeepEP
`1.0.0+607b0002.cann.9.0.0.b250` cannot run the target SGLang Decode path because its
`low_latency_dispatch` lacks `topk_weights`. An older environment's timing is not
validation of the current checkout. Record the source commit, wheel hash and image
digest for each measurement when the cloud provider exposes it. If it does not,
record that limitation and retain the actual component versions and checksums.

The corrected [A3 results](analysis.md) use CANN 9.0.0, PyTorch 2.10.0+cpu,
torch-npu 2.10.0 and DeepEP `1.0.0+e145d906.cann.9.0.0.b250` from the official
[2026.9.0.post3 release](https://github.com/sgl-project/sgl-kernel-npu/releases/tag/2026.9.0.post3).
The wheel and release archive hashes are in
[the environment record](results/a3-2026-09-16/revision-environment.json).
When installing this wheel with `pip --no-deps --target "$DEPS"`, expose its
native extension at the target root, as in the upstream NPU Dockerfile:

```bash
ln -s "$DEPS/deep_ep/deep_ep_cpp.cpython-311-aarch64-linux-gnu.so" \
  "$DEPS/deep_ep_cpp.cpython-311-aarch64-linux-gnu.so"
export PYTHONPATH="$SGLANG_ROOT/python:$DEPS${PYTHONPATH:+:$PYTHONPATH}"
```

Source the CANN environment before extending `PYTHONPATH`; preserve its TBE path.
Verify both `deep_ep.__file__` and `deep_ep_cpp.__file__` resolve to this installation.

The DeepEP wheel provides its vendor operator package. Register it before launch:

```bash
export ASCEND_CUSTOM_OPP_PATH=$(python3 -c "import deep_ep, os; print(os.path.join(os.path.dirname(deep_ep.__file__), 'vendors', 'hwcomputing'))")
export HCCL_BUFFSIZE=2048
export DEEPEP_HCCL_BUFFSIZE=2048
export DEEPEP_HYBRID_DEPLOYMENT=1
# Choose a free port range for your job on a shared host.
export HCCL_NPU_SOCKET_PORT_RANGE="24000-24031"
export DEEPEP_NORMAL_LONG_SEQ_ROUND=8
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=8192
export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=512
```

These are explicit experiment settings, not universal tuning recommendations.
Size DeepEP buffers for the largest **local** input and expert load, including
routing imbalance. Keep the settings fixed within a comparison. A failing case
exits nonzero; run capacity probes in separate processes because a device error
can invalidate subsequent collectives.

## Correctness first

Run from this directory. Replace `MODEL_PATH` with the local checkpoint directory:

```bash
MODEL_PATH=/path/to/Qwen3-30B-A3B
for phase in prefill decode; do
  torchrun --standalone --nproc_per_node=2 check_equivalence.py \
    --model-path "$MODEL_PATH" --backend none --ep-size 1 \
    --phase "$phase" --global-tokens 777 --save "tp_${phase}.pt"
  torchrun --standalone --nproc_per_node=2 check_equivalence.py \
    --model-path "$MODEL_PATH" --backend deepep --ep-size 2 \
    --phase "$phase" --global-tokens 777 --save "ep_${phase}.pt"
  python3 check_equivalence.py --compare "tp_${phase}.pt" "ep_${phase}.pt"
done
```

Repeat with `--global-tokens 1` to exercise an empty source rank, and with `4`,
`128` and `1001`. Use `--skew-experts 8` for biased routing; record observed expert
loads before calling any case an empty expert rank. The checker gathers every EP
output in token order and checks every TP replica. It rejects mismatched input or
weight hashes, shapes and non-finite outputs. The numerical criterion is maximum
absolute error divided by the TP output's maximum absolute magnitude, at most
`2e-2`. This is a block-level BF16 consistency check, not full-model accuracy.

CPU regression tests cover actual two-process broadcast/gather, uneven splits,
empty input ranks, mismatched inputs and deliberately corrupted rank 1 outputs:

```bash
python3 -m unittest discover -s . -p 'test_*.py' -v
```

The same regression suite is registered in `base-a-test-cpu` via
`test/registered/unit/bench/test_moe_tp_ep_benchmark.py`. Registration does not imply
that CI has run for a particular PR head.

## Performance and profiling

```bash
for backend in none deepep; do
  EP=1
  if [ "$backend" = deepep ]; then EP=2; fi
  torchrun --standalone --nproc_per_node=2 bench_moe_tp_ep.py \
    --model-path "$MODEL_PATH" --backend "$backend" --ep-size "$EP" \
    --phase prefill --global-tokens 128,512,2048,4096,8192,16384,32768 \
    --warmup 10 --iters 30 --repeat-id 1 --out "${backend}_prefill.json"
  torchrun --standalone --nproc_per_node=2 bench_moe_tp_ep.py \
    --model-path "$MODEL_PATH" --backend "$backend" --ep-size "$EP" \
    --phase decode --global-tokens 1,8,32,64,128,256,512,1001 \
    --warmup 10 --iters 30 --repeat-id 1 --out "${backend}_decode.json"
done
```

Repeat in independent processes, alternating the order of the two configurations.
JSON records the global and local sizes, input/weight hashes, environment,
script hashes, warmup count and all per-rank samples. Compare matched global input
hashes and settings; report process-level variation separately from within-run
samples. Do not label a difference statistically significant from a range alone.

For a profile, run a single global token count with `--profile-dir PATH` and
`--profile-iters 10`. Profiling follows the unprofiled measurement on the same
weights and inputs. Preserve all ranks' trace files. Operator duration sums can
include overlap and profiling overhead; inspect the timeline before attributing
an end-to-end difference to communication.

## Mapping to sequence length

For a single unchunked Prefill batch with `B` requests of input length `L`, the
global MoE input contains `B * L` tokens. Chunked prefill limits the tokens that
enter a particular forward, so longer requests can increase the number of forwards.
During ordinary Decode, `N` is the number of active requests, each contributing one
token. Context length changes attention work, but does not directly multiply MoE
input rows. Serving comparisons therefore must also record request length, batch
or concurrency, chunk size and the actual global token counts observed at MoE.

The earlier equal-**per-rank** comparison performed different global work in TP
and EP. Its ratios and communication attribution are withdrawn; use only corrected
measurements with the global workload definition above.

## Serving cross-check

`bench_serving.py` sends fixed-length synthetic token IDs to a local SGLang server.
It checks HTTP status and exact output token counts, excludes distinct warmup
requests, and saves every request's TTFT, TPOT and completion latency. TTFT starts
when the request is sent after acquiring a client concurrency slot. Throughput
includes the wall time for all measured requests.

Start a service with the same model, dtype and TP/EP settings as the module run.
For the serving experiment, fix `--chunked-prefill-size 8192`,
`--max-running-requests 4`, `--disable-cuda-graph`, `--disable-radix-cache`,
`--random-seed 20260922` and `--mem-fraction-static 0.8` for both configurations.
For example, launch pure TP in a separate terminal after setting the environment
above. For TP+EP, change `EP=2` and `BACKEND=deepep`:

```bash
EP=1
BACKEND=none
python3 -m sglang.launch_server --model-path "$MODEL_PATH" \
  --tp-size 2 --ep-size "$EP" --moe-a2a-backend "$BACKEND" \
  --deepep-mode auto --attention-backend ascend --dtype bfloat16 \
  --mem-fraction-static 0.8 --disable-cuda-graph --disable-radix-cache \
  --chunked-prefill-size 8192 --max-running-requests 4 \
  --random-seed 20260922 --host 127.0.0.1 --port 30198
```

Wait for the service's `/health` endpoint to return HTTP 200. Then, from this directory:

```bash
for length in 1024 8192 16384; do
  python3 bench_serving.py --port 30198 --input-len "$length" \
    --output-len 64 --requests 16 --concurrency 4 --seed 20260923 \
    --out "serving_${length}.json"
done
```

Repeat with three independently started services per configuration, alternating
configuration order between repeats and using matching prompt seeds in each pair.

For a separate layout observation run, add this directory to `PYTHONPATH` and
launch the service with a `--forward-hooks` JSON specification whose
`target_modules` is `["model.layers.0.mlp"]`, `hook_factory` is
`"observe_tokens:make_hook"`, and `config.output_dir` points to the desired output
directory. The hook records the observed phase, global valid-token count (when
provided by SGLang), local input rows and runtime batch size for every distinct
shape. Before each client run, write `{"input_length": 1024}` (with the actual
length) to `config.output_dir/request_shape.json` to label the observations.
Disable this hook for performance measurements.
Runtime batch size and local rows can include serving padding; use the global
valid-token count when interpreting the workload.

`inspect_routing.py` separately captures the actual Top-K IDs for a module input
and checks that global expert assignments total `global_tokens * topk`. Use the
same model, phase, seed and global token count as the profiled point.

Use `summarize.py RESULT_DIR --out SUMMARY.json` for the module sweep,
`summarize_serving.py RESULT_DIR --out SUMMARY.json` for serving files named
`none_r1_in1024.json` / `deepep_r1_in1024.json`, and
`summarize_profile.py TRACE_DIR --iters 10 --out SUMMARY.json` for all ranks'
profiling results. The profile parser separates HCCL operation spans from their
enclosed AICPU kernels to avoid summing both levels.
