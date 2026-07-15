# DeepSeek V4 with NVFP4 KV Cache on SM90

```{warning}
This is an experimental customer-validation configuration for the official
DeepSeek V4 Flash and Pro FP4 checkpoints. The SM90 fused decode kernel has
kernel, CUDA-graph, and end-to-end coverage, but NVFP4 is lossy. Complete
model-level accuracy validation before production deployment.
```

This guide serves DeepSeek V4 on one node with eight SM90 GPUs and TP8. The
model keeps its checkpoint-defined weight format, while the attention KV cache
uses the NVFP4 recipe:

- the 448-dimensional NoPE vector is stored as 224 bytes of packed E2M1 with
  28 bytes of E4M3 block-16 scales;
- the 64-dimensional RoPE vector remains in BF16 and occupies 128 bytes;
- each cached token therefore uses a fixed 380-byte attention row;
- decode reads the NVFP4 SWA and C4/C128 caches directly in a fused SM90
  FlashMLA kernel;
- with the default `SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=true`, extend and
  prefill requests materialize selected KV rows to BF16 before calling
  `flash_mla_sparse_fwd`. Prefill is not a raw-NVFP4 fused kernel.

The public attention backend remains `dsv4`. Do not select `flashmla_kv`,
`flashmla_sparse`, FA3, or NSA as a separate top-level backend for this path.

## Supported configuration

The native operator requires CUDA compute capability 9.0 and can run on H100,
H200, or H20. Model memory capacity is a separate constraint:

| Model | Q heads | Index top-k | Main layers | Initial validation hardware |
|---|---:|---:|---:|---|
| DeepSeek V4 Flash | 64 | 512 | 43 | 8 SM90 GPUs with enough checkpoint memory |
| DeepSeek V4 Pro | 128 | 1024 | 61 | 8 GPUs with approximately 141 GB each |

DeepSeek V4 Pro is approximately 805 GB at the tested revision and used about
131 GB per GPU after model load, KV allocation, and graph capture in the TP8
validation configuration. An 80 GB H100 is suitable for kernel testing and can
serve the smaller Flash model, but it is not a no-offload Pro TP8 target. The
initial Pro validation uses `--cpu-offload-gb 0`; CPU offload is not a supported
substitute in this guide.

Both variants use the following attention layout:

- QK/V dimensions: 512/512, with `448 NoPE + 64 RoPE`;
- user-visible page size: 256;
- internal physical pages: SWA 256, C4 64, and C128 2;
- layer compression ratios: C0, C4, and C128, selected from model metadata;
- one independent FP32 global scale for each primary or extra NVFP4 cache.

Use TP8, EP1, and PP1 for the first validation. Do not enable speculative
decoding/MTP, HiSparse, or prefill context parallelism. Multi-node serving,
two-batch overlap, disaggregation, and pipeline parallelism are outside the
scope of this guide.

The model checkpoint and the KV format are independent. The official Flash and
Pro checkpoints use checkpoint-defined FP4/MXFP4 expert weights; NVFP4 here
refers only to the KV cache. Do not add `--quantization fp8`, and do not infer
the KV recipe from the weight format. Keep `--load-format auto` and use the
Marlin MoE runner for the validated official FP4 checkpoints.

## Define the environment

Set all installation, model, cache, and result locations through environment
variables. The examples below intentionally contain no machine-specific paths.

```bash
: "${DELIVERY_REPOSITORY_URL:?set DELIVERY_REPOSITORY_URL}"
: "${DELIVERY_COMMIT:?set DELIVERY_COMMIT}"
: "${WORK_ROOT:?set WORK_ROOT}"
: "${MODEL_ROOT:?set MODEL_ROOT}"
: "${MODEL_REVISION:?set the approved model revision}"
: "${CUDA_HOME:?set CUDA_HOME}"

export SGLANG_ROOT="${WORK_ROOT}/sglang"
export VENV_PATH="${WORK_ROOT}/venv"
export CACHE_ROOT="${WORK_ROOT}/kernel-cache"
export RESULT_DIR="${WORK_ROOT}/dsv4-nvfp4-results"

export MODEL_VARIANT="Pro"  # Flash or Pro
export MODEL_ID="deepseek-ai/DeepSeek-V4-${MODEL_VARIANT}"
export MODEL_PATH="${MODEL_ROOT}/DeepSeek-V4-${MODEL_VARIANT}"
export SERVED_MODEL_NAME="deepseek-v4-${MODEL_VARIANT,,}"

export SERVER_HOST="0.0.0.0"
export CLIENT_HOST="127.0.0.1"
export SERVER_PORT="30000"
export TP_SIZE="8"
export MAX_JOBS="4"

export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"
export CUDA_CACHE_PATH="${CACHE_ROOT}/cuda"
export TORCH_EXTENSIONS_DIR="${CACHE_ROOT}/torch-extensions"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export PYTHONUNBUFFERED="1"
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP="1"

mkdir -p "${MODEL_ROOT}" "${CACHE_ROOT}" "${RESULT_DIR}"
```

Do not set `SGLANG_OPT_FP8_WO_A_GEMM=1` for this SM90 setup. That optimization
requires SM100 or newer and is automatically disabled on Hopper.

If a container is used, bind the directories represented by `SGLANG_ROOT`,
`MODEL_ROOT`, `CACHE_ROOT`, and `RESULT_DIR`, and expose all eight GPUs. Keep
the compiler and JIT caches across launches because the first startup compiles
MHC, DeepGEMM, Triton, and CUDA-graph shapes.

## Build the delivery commit

The Python package and `sglang-kernel` wheel must come from the same delivery
commit. A stale wheel does not contain `dsv4_sparse_decode_fwd_nvfp4`, and the
backend deliberately fails at startup instead of silently selecting a slower
fallback.

CUDA 12.5 or newer is required by the FlashMLA build. CUDA 12.8 and CUDA 13.x
are the recommended Hopper toolchains; CUDA 13.0 was used for the reference
end-to-end validation.

```bash
git clone --recursive "${DELIVERY_REPOSITORY_URL}" "${SGLANG_ROOT}"
git -C "${SGLANG_ROOT}" checkout "${DELIVERY_COMMIT}"
git -C "${SGLANG_ROOT}" submodule update --init --recursive

python3 -m venv "${VENV_PATH}"
source "${VENV_PATH}/bin/activate"

cd "${SGLANG_ROOT}"
python3 -m pip install --upgrade pip uv cmake ninja scikit-build-core
python3 -m pip install -e "python[all]"
make -C sgl-kernel build MAX_JOBS="${MAX_JOBS}"
```

For an overlay-style delivery, set `SGL_KERNEL_STAGE` to the staged package and
prepend it to `PYTHONPATH`. Do not mix a staged Python package with `.so` files
from another commit.

```bash
: "${SGL_KERNEL_STAGE:?set SGL_KERNEL_STAGE for an overlay installation}"
export PYTHONPATH="${SGL_KERNEL_STAGE}:${SGLANG_ROOT}/python:${PYTHONPATH:-}"
```

Skip the overlay block when the source-built wheel is installed directly.

## Download a pinned checkpoint

Download the model before reserving GPUs and keep the approved revision fixed
throughout correctness and performance validation.

```bash
hf download "${MODEL_ID}" \
  --revision "${MODEL_REVISION}" \
  --local-dir "${MODEL_PATH}"
```

Changing to an FP8-repacked checkpoint or another model revision changes the
weight path and requires a new accuracy and end-to-end validation. This guide
does not treat such a checkpoint as equivalent to the official FP4 model.

## Preflight the node and native operator

Run the following after entering the serving environment and installing the
source-built kernel:

```bash
python3 - <<'PY'
import importlib.metadata as md
import os

import torch
import sgl_kernel
from sgl_kernel import flashmla_ops  # noqa: F401; registers torch ops

tp_size = int(os.environ["TP_SIZE"])
assert torch.cuda.device_count() == tp_size, torch.cuda.device_count()
capabilities = [
    torch.cuda.get_device_capability(index) for index in range(tp_size)
]
assert all(capability == (9, 0) for capability in capabilities), capabilities
assert hasattr(torch.ops.sgl_kernel, "dsv4_sparse_decode_fwd_nvfp4")

print("sglang", md.version("sglang"))
print("sglang-kernel", md.version("sglang-kernel"))
print("torch", torch.__version__, "CUDA", torch.version.cuda)
print("sgl_kernel package", sgl_kernel.__file__)
print("SM90 DeepSeek V4 NVFP4 fused decode operator is available")
PY
```

Also verify that the checkpoint advertises the supported DeepSeek V4 shape:

```bash
python3 - <<'PY'
import os
from transformers import AutoConfig

root = AutoConfig.from_pretrained(
    os.environ["MODEL_PATH"], trust_remote_code=True
)
text = getattr(root, "text_config", root)
architectures = getattr(root, "architectures", None) or getattr(
    text, "architectures", []
)

assert "DeepseekV4ForCausalLM" in architectures, architectures
assert text.head_dim - text.qk_rope_head_dim == 448
assert text.qk_rope_head_dim == 64
assert text.head_dim == 512
assert getattr(text, "v_head_dim", text.head_dim) == 512
assert text.num_attention_heads in (64, 128)

print(
    "layers=", text.num_hidden_layers,
    "heads=", text.num_attention_heads,
    "index_topk=", text.index_topk,
)
PY
```

If the native-op check fails, rebuild and reinstall `sglang-kernel` before
launching the model. A test run in which every CUDA case is skipped does not
validate the extension.

## Launch DeepSeek V4

The values below reproduce the conservative TP8 validation setup. Flash can
usually use a lower static-memory fraction than Pro.

```bash
if [[ "${MODEL_VARIANT}" == "Pro" ]]; then
  export MEM_FRACTION_STATIC="0.92"
else
  export MEM_FRACTION_STATIC="0.85"
fi

export CONTEXT_LENGTH="65536"
export MAX_TOTAL_TOKENS="33792"
export MAX_RUNNING_REQUESTS="8"
export CHUNKED_PREFILL_SIZE="8192"
export MAX_PREFILL_TOKENS="16384"
export SWA_FULL_TOKENS_RATIO="0.3"
export CUDA_GRAPH_MAX_BS="8"
export KV_LABEL="nvfp4"
export SERVER_LOG="${RESULT_DIR}/server-nvfp4.log"

python3 -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust-remote-code \
  --host "${SERVER_HOST}" \
  --port "${SERVER_PORT}" \
  --tp-size "${TP_SIZE}" \
  --ep-size 1 \
  --pp-size 1 \
  --dtype auto \
  --load-format auto \
  --attention-backend dsv4 \
  --moe-runner-backend marlin \
  --moe-a2a-backend none \
  --kv-cache-dtype fp4_e2m1 \
  --fp4-kv-cache-recipe nvfp4 \
  --page-size 256 \
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}" \
  --max-prefill-tokens "${MAX_PREFILL_TOKENS}" \
  --context-length "${CONTEXT_LENGTH}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --max-total-tokens "${MAX_TOTAL_TOKENS}" \
  --max-running-requests "${MAX_RUNNING_REQUESTS}" \
  --swa-full-tokens-ratio "${SWA_FULL_TOKENS_RATIO}" \
  --cuda-graph-backend-decode full \
  --cuda-graph-max-bs-decode "${CUDA_GRAPH_MAX_BS}" \
  --cpu-offload-gb 0 \
  --random-seed 42 \
  --watchdog-timeout 3600 \
  2>&1 | tee "${SERVER_LOG}"
```

Do not copy the GLM-5.2 flags `--attention-backend nsa`,
`--dsa-prefill-backend flashmla_sparse`, or
`--dsa-decode-backend flashmla_kv`. DeepSeek V4 uses `dsv4`, which internally
dispatches normal decode to the fused NVFP4 operator. With the validated
default sparse-prefill setting, extend and prefill use the BF16 workspace path.

Prefill CUDA graphs are disabled automatically because large sparse prefill
uses a mutable BF16 dequantization workspace. Decode CUDA graphs remain
enabled. Do not add a global CUDA-graph disable flag, and do not needlessly add
`--disable-prefill-cuda-graph`; keeping the automatic transition visible in
the log verifies that the expected delivery code is active.

Cold startup of Pro can take 50 minutes or more on shared storage. Allow a
readiness timeout of at least 60 minutes, or 75 minutes on a cold filesystem.
The log can remain at `Execute dequant fp8 wo_a` while a lazy weight iterator
materializes the rest of the checkpoint; continuing CPU time and storage I/O
indicate progress. This is model loading, not NVFP4 KV initialization.

## Required startup and runtime markers

Wait for the health endpoint, then save the resolved server information:

```bash
export BASE_URL="http://${CLIENT_HOST}:${SERVER_PORT}"
export READINESS_TIMEOUT_SEC="${READINESS_TIMEOUT_SEC:-4500}"

readiness_deadline=$((SECONDS + READINESS_TIMEOUT_SEC))
until curl --fail --silent --show-error --max-time 5 \
  "${BASE_URL}/health" > /dev/null; do
  if (( SECONDS >= readiness_deadline )); then
    echo "Server was not ready after ${READINESS_TIMEOUT_SEC} seconds" >&2
    exit 1
  fi
  sleep 10
done

curl --fail --silent --show-error "${BASE_URL}/server_info" \
  > "${RESULT_DIR}/server-info-nvfp4.json"
```

The startup log must show the intended format and fused path:

```text
kv_cache_dtype='fp4_e2m1'
fp4_kv_cache_recipe='nvfp4'
Use dsv4 attention backend for DeepseekV4ForCausalLM, setting page_size to 256.
Initialized DeepSeek V4 NVFP4 global scales for N/N local layers.
DeepSeek V4 NVFP4 decode uses fused SM90 FlashMLA dsv4_sparse_decode_fwd_nvfp4
Disable prefill CUDA graph because cuda_graph_config resolved prefill.backend='disabled'
Capture target decode CUDA graph end.
The server is fired up and ready to roll!
```

For the tested checkpoints, `N/N` is `43/43` for Flash and `61/61` for Pro.
The log also reports `DSV4 memory calculation` and `DSV4 pool sizes`; preserve
those lines when comparing capacity.

During a long request, the expected phase behavior is:

```text
Prefill batch ... cuda graph: False
Decode batch ... cuda graph: True
```

The warning that FP4 E2M1 KV cache can reduce accuracy is expected. The log
must not contain a traceback, CUDA error, illegal or misaligned memory access,
OOM, NaN, assertion failure, or segmentation fault.

```bash
FATAL_PATTERN="Traceback|CUDA error|illegal memory access|misaligned address"
FATAL_PATTERN+="|out of memory|NaN|assertion failure|segmentation fault"
if rg -n "${FATAL_PATTERN}" "${SERVER_LOG}"; then
  echo "Fatal pattern found in ${SERVER_LOG}" >&2
  exit 1
fi
echo "No fatal pattern found in ${SERVER_LOG}"
```

No matching line and a zero exit status from the complete check are expected.

## Smoke tests

Run a short request, a request that crosses the logical 128-token SWA window,
and a 32K request that exercises chunked prefill and long-context decode.
`random-ids` with `--tokenize-prompt` controls token lengths exactly.

```bash
run_case() {
  local name="$1"
  local input_len="$2"
  local output_len="$3"

  python3 -m sglang.benchmark.serving \
    --backend sglang \
    --base-url "${BASE_URL}" \
    --model "${SERVED_MODEL_NAME}" \
    --tokenizer "${MODEL_PATH}" \
    --dataset-name random-ids \
    --tokenize-prompt \
    --random-input-len "${input_len}" \
    --random-output-len "${output_len}" \
    --random-range-ratio 1 \
    --num-prompts 1 \
    --max-concurrency 1 \
    --request-rate inf \
    --seed 42 \
    --warmup-requests 0 \
    --flush-cache \
    --output-details \
    --disable-tqdm \
    --ready-check-timeout-sec 60 \
    --output-file "${RESULT_DIR}/${KV_LABEL}-${name}.jsonl"
}

run_case short-16x32 16 32
run_case cross-swa-256x256 256 256
run_case chunked-prefill-32k-x512 32768 512
```

All requests must return successfully with the requested output count, finite
log probabilities when requested, and no new fatal log line. For a 32K input
with `CHUNKED_PREFILL_SIZE=8192`, four eager prefill chunks followed by
CUDA-graph decode are expected.

## Compare against FP8 KV cache

Always run an equal-capacity FP8 control before attributing an end-to-end
change to NVFP4. Stop the NVFP4 server, keep every model and server parameter
unchanged, replace:

```text
--kv-cache-dtype fp4_e2m1
--fp4-kv-cache-recipe nvfp4
```

with:

```text
--kv-cache-dtype fp8_e4m3
```

Set `KV_LABEL=fp8` and use a distinct FP8 server log before running the control.
Do not pass an FP4 recipe to the FP8 control. Run the same smoke sequence on
both services before measuring performance, then use identical tokenized
prompts, seed, token pool, graph batches, and request order.

```bash
export KV_LABEL="fp8"
export SERVER_LOG="${RESULT_DIR}/server-fp8.log"
```

For a compact directional comparison, use these two workloads:

```bash
# Long-context B1
python3 -m sglang.benchmark.serving \
  --backend sglang --base-url "${BASE_URL}" \
  --model "${SERVED_MODEL_NAME}" --tokenizer "${MODEL_PATH}" \
  --dataset-name random-ids --tokenize-prompt \
  --random-input-len 32768 --random-output-len 512 \
  --random-range-ratio 1 --num-prompts 1 --max-concurrency 1 \
  --request-rate inf --seed 42 --warmup-requests 1 --flush-cache \
  --output-details --disable-tqdm \
  --output-file "${RESULT_DIR}/${KV_LABEL}-b1-32k-x512.jsonl"

# True B4 decode within the conservative SWA pool
python3 -m sglang.benchmark.serving \
  --backend sglang --base-url "${BASE_URL}" \
  --model "${SERVED_MODEL_NAME}" --tokenizer "${MODEL_PATH}" \
  --dataset-name random-ids --tokenize-prompt \
  --random-input-len 1024 --random-output-len 1024 \
  --random-range-ratio 1 --num-prompts 4 --max-concurrency 4 \
  --request-rate inf --seed 42 --warmup-requests 1 --flush-cache \
  --output-details --disable-tqdm \
  --output-file "${RESULT_DIR}/${KV_LABEL}-b4-1k-x1024.jsonl"
```

Report TTFT, TPOT/ITL, end-to-end latency, output tokens per second, request
failures, and the actual server-side admission. With the conservative
`MAX_TOTAL_TOKENS=33792` and `SWA_FULL_TOKENS_RATIO=0.3`, four 4K-input,
1024-output requests do not form a true server B4 batch: the SWA pool admits
two requests and queues two. Check `#running-req` and `#queue-req` in the log
instead of relying only on client concurrency.

For equal-capacity performance, keep `MAX_TOTAL_TOKENS` identical. For a
separate capacity comparison, record the automatic `full_token` estimate from
`DSV4 memory calculation` for each format, or launch each format without an
explicit `--max-total-tokens`. Do not mix the extra NVFP4 capacity into an
equal-capacity speedup claim.

## Reference performance results

The following results were collected on 2026-07-14 from a development build.
They are directional validation data, not a production performance guarantee.
The final delivery commit should be benchmarked again if its source diff or
native binary changes.

The end-to-end setup used DeepSeek V4 Pro on eight H20-3e GPUs with 143771 MiB
each and TP8. Both launches used the same hybrid checkpoint and weight
path: its general quantization configuration is FP8, its routed experts are
FP4/MXFP4, and Marlin executes the experts. In the tables below, **FP8 means
FP8 KV cache**, not a change to model weights.

Common settings were CUDA 13.0, `dsv4`, page size 256, context length 65536,
`mem-fraction-static=0.92`, `swa-full-tokens-ratio=0.3`, and full decode CUDA
graphs through batch size 8. Both formats used the same explicit 33792-token
full pool, tokenized random prompts, seed 42, one warmup request, a cache flush,
and one measured round. The resulting physical pools were identical:
full/SWA/C4/C128 = 33792/9984/8448/264.

The FP8 checkpoint did not provide a dedicated KV scale, so the server logged
an FP8 KV scale of 1.0. These measurements compare serving performance and
capacity only; they do not establish FP8/NVFP4 accuracy equivalence.

### Equal-capacity end-to-end comparison

Positive throughput deltas mean NVFP4 was faster. Positive latency deltas mean
NVFP4 was slower.

| Workload | Server admission | FP8 output tok/s | NVFP4 output tok/s | Delta |
|---|---:|---:|---:|---:|
| 32K input, 512 output | B1 | 33.6252 | 33.1909 | -1.29% |
| 4K input, 1024 output, requested C4 | B2, two waves | 121.0805 | 119.6632 | -1.17% |
| 1K input, 1024 output | B4 | 231.8020 | 225.3721 | -2.77% |

| Workload | Metric | FP8 (ms) | NVFP4 (ms) | NVFP4 delta |
|---|---|---:|---:|---:|
| 32K input, 512 output, B1 | Mean E2E | 15209.84 | 15409.37 | +1.31% |
| | Median TTFT | 8471.46 | 8576.22 | +1.24% |
| | Mean TPOT / ITL | 13.1867 / 13.1866 | 13.3721 / 13.3721 | +1.41% / +1.41% |
| 4K input, 1024 output, effective B2 | Mean E2E | 25312.41 | 25608.95 | +1.17% |
| | Median TTFT | 10671.89 | 10813.58 | +1.33% |
| | p99 TTFT | 19181.48 | 19440.38 | +1.35% |
| | Mean TPOT / ITL | 14.4027 / 14.4027 | 14.5431 / 14.5431 | +0.97% / +0.97% |
| 1K input, 1024 output, B4 | Mean E2E | 17651.69 | 18155.25 | +2.85% |
| | Median TTFT | 1404.49 | 1417.34 | +0.91% |
| | p99 TTFT | 1404.74 | 1417.66 | +0.92% |
| | Mean TPOT / ITL | 15.9556 / 15.9556 | 16.4272 / 16.4271 | +2.96% / +2.96% |

The 32K B1 and effective-B2 differences are inside a +/-2% noise band. The
true-B4 case directionally lost 2.77% output throughput and added 2.96% TPOT,
but one round cannot establish a stable regression. The equal-capacity run
therefore shows no NVFP4 speed advantage.

The TP0 checkpoint-load timer reported 2885.83 seconds for FP8 and 2966.89
seconds for NVFP4, a 2.81% difference. This is a single cold-load observation
that includes checkpoint I/O and lazy initialization; it should not be
attributed to the KV format alone or interpreted as full time-to-readiness.

### KV capacity comparison

With the same 1.49 GB budget available to the DSV4 pool profiler:

| Metric | FP8 | NVFP4 | NVFP4 change |
|---|---:|---:|---:|
| Estimated bytes per full token | 21958.64 | 16646.03 | -24.19% |
| Automatic full-token capacity | 64512 | 84992 | +31.75% (1.3175x) |
| Automatic SWA pool | 19200 | 25344 | +32.00% |
| Automatic C4 pool | 16128 | 21248 | +31.75% |
| Automatic C128 pool | 504 | 664 | +31.75% |

The equal-capacity performance runs deliberately capped both formats at 33792
full tokens. A capacity-throughput run that uses the additional NVFP4 entries
to admit more concurrent or longer requests has not yet been completed. Do not
infer a throughput increase from the 1.3175x capacity ratio alone. Equivalent
FP8 end-to-end data for the Flash checkpoint has also not been collected.

### Fused-kernel microbenchmark

The fused NVFP4 operator was also compared with the previous NVFP4 path that
dequantized the complete padded primary and extra widths into a BF16 workspace
and then called `flash_mla_sparse_fwd`. This is **not an FP8 comparison** and
is not model end-to-end latency.

Each run used one H100 80 GB GPU; the table combines two same-class single-GPU
allocations. It used cold L2, 20 warmup iterations, 100 timed iterations per
round, and three rounds. Every round-median CV was below 2%, and the maximum
output difference was 1.22e-4. Cache construction and quantization, persistent
workspace allocation, and scheduler-metadata creation were outside the timed
region and amortized.

| Context | Variant | B1 speedup | B12 speedup | B17 speedup |
|---|---|---:|---:|---:|
| 32K | Flash C4 | 9.926x | 9.065x | 10.791x |
| 32K | Flash C128 | 11.754x | 15.997x | 18.595x |
| 32K | Pro C4 | 9.929x | 10.624x | 12.413x |
| 32K | Pro C128 | 11.116x | 14.786x | 17.141x |
| 128K full top-k | Flash C128 | 9.815x | 12.147x | 14.539x |
| 128K full top-k | Pro C128 | 9.591x | 10.651x | 12.303x |

The large kernel-only improvement removes the old explicit dequantization
fallback. It does not imply a similar model-level gain over FP8 because model
execution, communication, MoE, scheduling, and prefill remain outside this
single-layer measurement.

## Record the environment

Preserve enough information to reproduce a customer result:

```bash
{
  git -C "${SGLANG_ROOT}" rev-parse HEAD
  git -C "${SGLANG_ROOT}" diff --binary | sha256sum
  nvcc --version
  nvidia-smi \
    --query-gpu=name,memory.total,driver_version,clocks.sm,clocks.mem \
    --format=csv,noheader
  python3 - <<'PY'
import importlib.metadata as md
import torch

print("sglang", md.version("sglang"))
print("sglang-kernel", md.version("sglang-kernel"))
print("torch", torch.__version__)
print("torch CUDA", torch.version.cuda)
print("NCCL", torch.cuda.nccl.version())
PY
} | tee "${RESULT_DIR}/environment.txt"
```

Also save the full launch command, model ID and revision, container identifier
if applicable, server log, `/server_info`, benchmark JSONL files, and hashes of
the installed `flashmla_ops` and other staged binaries.

## Developer tests

Run the host integration test first:

```bash
cd "${SGLANG_ROOT}"
python3 -m pytest -q \
  test/registered/unit/mem_cache/test_dsv4_nvfp4_host_unit.py
```

Then run the native-kernel and DSV4 backend suites on an SM90 GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m pytest -q \
  sgl-kernel/tests/test_flashmla_dsv4_nvfp4.py

CUDA_VISIBLE_DEVICES=0 python3 -m pytest -q \
  test/registered/attention/unittests/dsv4/test_deepseek_v4.py
```

The GLM SM90 NVFP4 regression should also remain green because both operators
share the same FlashMLA extension:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m pytest -q \
  sgl-kernel/tests/test_flashmla_nvfp4.py
```

Repeat the native-op preflight after installing or replacing a wheel.

## Current limitations and common mistakes

- The native DeepSeek V4 NVFP4 operator is SM90-only and supports the fixed
  448-NoPE, 64-RoPE, V512, H64/H128 shapes described above.
- `--page-size 256` is mandatory. Do not copy the GLM-5.2 page size of 64.
- `--fp4-kv-cache-recipe nvfp4` is mandatory for `fp4_e2m1`; the MXFP4 recipe
  is a different cache format.
- Missing native operators indicate a stale or mismatched `sglang-kernel`
  build. There is no silent production fallback for decode.
- Decode is fused and CUDA-graph compatible. With the validated default,
  extend and sparse prefill use a BF16 workspace and automatically disable only
  prefill CUDA graphs.
- `SWA_FULL_TOKENS_RATIO` sizes the SWA pool relative to the full pool; it is
  not a general KV occupancy percentage. A small value can limit admission
  while the full pool still has free entries.
- Keep `CHUNKED_PREFILL_SIZE=8192` for initial long-context validation to bound
  the mutable prefill workspace.
- Speculative decoding/MTP, HiSparse, and prefill context parallelism are not
  supported by this NVFP4 path.
- CPU offload, PP, multi-node serving, TBO, and disaggregation are not covered
  by this validation recipe.
- NVFP4 is lossy. A successful smoke or kernel test does not replace
  model-level accuracy evaluation.
