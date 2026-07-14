# GLM-5.2 with NVFP4 KV Cache on SM90

```{warning}
This is an experimental customer-validation configuration. The SM90 NVFP4
kernel and its GLM-5.2 shapes have correctness coverage, but a full GLM-5.2
server and accuracy run is still required in the delivery environment. Do not
treat results from a different GLM checkpoint as GLM-5.2 validation.
```

This guide serves the FP8 checkpoint `zai-org/GLM-5.2-FP8` on one node with
eight SM90 GPUs and TP8. The latent DSA KV cache uses the NVFP4 recipe:

- the 512-dimensional latent vector is stored as packed E2M1 with E4M3
  block-16 scales and one FP32 scale per layer;
- the 64-dimensional RoPE vector remains in BF16;
- each token occupies a fixed 416-byte main-cache row;
- decode uses a FlashMLA kernel that dequantizes the selected KV rows inside
  the kernel;
- prefill materializes the selected cached prefix to BF16 before calling
  `flashmla_sparse`. It is not a native raw-NVFP4 prefill kernel.

## Supported configuration

Use the following configuration for the first validation:

- one node with 8 CUDA GPUs of compute capability 9.0;
- TP8, EP1, and no pipeline parallelism;
- CUDA 13.0 and a source build of `sglang-kernel` from the delivery commit;
- `GlmMoeDsaForCausalLM` with `kv_lora_rank=512`, RoPE dimension 64,
  Q dimension 576, V dimension 512, and DSA top-k 2048;
- page size 64;
- no speculative decoding/MTP, HiSparse, two-batch overlap, context
  parallelism, or prefill/decode disaggregation during the initial test.

The NVFP4 DSA path accepts only `--fp4-kv-cache-recipe nvfp4`; it is not the
MXFP4/E8M0 cache format. Pipeline parallelism and two-batch overlap are
rejected when GLM-5.2 index-topk sharing is active.

## Build the delivery commit

The installed `sglang-kernel` wheel must be rebuilt from the same delivery
commit as the Python package. An older wheel does not contain the SM90 NVFP4
FlashMLA operator.

```bash
git clone --recursive <delivery-repository-url> sglang
cd sglang
git checkout <delivery-commit>

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip uv cmake ninja scikit-build-core
python3 -m pip install -e "python[all]"
make -C sgl-kernel build MAX_JOBS=4
```

For a repeatable customer handoff, prefer a container or a set of wheels built
and tested together. Preserve the compiler/JIT caches between launches; the
first launch also compiles DeepGEMM and Triton kernels.

## Download the pinned model

The checkpoint is approximately 756 GB. Download it before reserving GPUs and
keep the revision fixed during validation.

```bash
export MODEL_PATH="${PWD}/models/GLM-5.2-FP8"

hf download zai-org/GLM-5.2-FP8 \
  --revision ba978f7d347eaf65d22f1a86833408afdb953541 \
  --local-dir "${MODEL_PATH}"
```

If the model is already available, set `MODEL_PATH` to that directory instead.

## Preflight the installation

Run this check on the serving node after installing the source-built kernel:

```bash
python3 - <<'PY'
import torch
import sgl_kernel.flash_mla  # registers flashmla_ops

assert torch.cuda.device_count() == 8, torch.cuda.device_count()
capabilities = [torch.cuda.get_device_capability(i) for i in range(8)]
assert all(capability == (9, 0) for capability in capabilities), capabilities
assert hasattr(torch.ops.sgl_kernel, "fwd_kvcache_mla_nvfp4")
print("SM90 NVFP4 FlashMLA operator is available")
PY
```

## Launch GLM-5.2

The following is the conservative customer-validation command. In particular,
`--dsa-decode-backend flashmla_kv` is required to select the native NVFP4
decode kernel. Without that explicit flag, NVFP4 uses the conservative FA3
fallback by default.

```bash
export SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD=0

python3 -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --served-model-name glm-5.2 \
  --quantization fp8 \
  --dtype bfloat16 \
  --tp-size 8 \
  --ep-size 1 \
  --trust-remote-code \
  --attention-backend nsa \
  --dsa-prefill-backend flashmla_sparse \
  --dsa-decode-backend flashmla_kv \
  --kv-cache-dtype fp4_e2m1 \
  --fp4-kv-cache-recipe nvfp4 \
  --page-size 64 \
  --chunked-prefill-size 8192 \
  --max-prefill-tokens 16384 \
  --context-length 131072 \
  --mem-fraction-static 0.85 \
  --max-running-requests 32 \
  --disable-radix-cache \
  --cuda-graph-max-bs 32 \
  --cuda-graph-bs 1 2 4 8 16 32 \
  --moe-a2a-backend none \
  --disable-custom-all-reduce \
  --disable-flashinfer-autotune \
  --random-seed 42 \
  --watchdog-timeout 3600 \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --host 0.0.0.0 \
  --port 30000 \
  2>&1 | tee glm52_nvfp4_server.log
```

The first startup can be much slower than later starts because it loads the
checkpoint, compiles kernels, and captures CUDA graphs. Allow a readiness
timeout of at least 30 minutes when storage and caches are cold.

## Validation gates

First confirm that the server is ready:

```bash
curl --fail --silent --show-error http://127.0.0.1:30000/v1/models
```

The startup log must identify the intended cache and attention paths. Check it
for the following values and for the absence of a traceback, OOM, illegal
memory access, or NaN report:

```text
kv_cache_dtype='fp4_e2m1'
fp4_kv_cache_recipe='nvfp4'
dsa_prefill_backend='flashmla_sparse'
dsa_decode_backend='flashmla_kv'
Initialized DSA NVFP4 global scales
```

Run both a short request and a request that crosses the 8K chunked-prefill
boundary:

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --base-url http://127.0.0.1:30000 \
  --dataset-name random \
  --random-input-len 4096 \
  --random-output-len 16 \
  --random-range-ratio 1 \
  --num-prompts 1 \
  --max-concurrency 1 \
  --seed 42

python3 -m sglang.bench_serving \
  --backend sglang \
  --base-url http://127.0.0.1:30000 \
  --dataset-name random \
  --random-input-len 9216 \
  --random-output-len 16 \
  --random-range-ratio 1 \
  --num-prompts 1 \
  --max-concurrency 1 \
  --seed 42
```

For customer acceptance, also compare a representative workload against
`--kv-cache-dtype fp8_e4m3` with every other server and request parameter held
constant. Report TTFT, inter-token latency, throughput, maximum KV token
capacity, failed requests, and any accuracy delta. Do not infer GLM-5.2
accuracy or performance from another GLM checkpoint.

## Developer tests

Run the host integration test first:

```bash
python3 -m pytest -q \
  test/registered/unit/mem_cache/test_dsa_nvfp4_host_unit.py
```

Then run the codec and native-kernel tests on an SM90 GPU:

```bash
python3 -m pytest -q \
  python/sglang/jit_kernel/tests/test_dsa_nvfp4_k_cache.py

CUDA_VISIBLE_DEVICES=0 python3 -m pytest -q \
  sgl-kernel/tests/test_flashmla_nvfp4.py
```

Before accepting the GPU result, repeat the preflight operator check. A test
suite in which all GPU cases were skipped does not validate the installed
extension.

## Current limitations

- The native fused kernel is SM90-only and supports the fixed DSA shape and
  cache layout described above.
- Native NVFP4 support is decode-only; prefill uses a BF16 materialization
  path.
- FA3 is a correctness fallback, not the native NVFP4 decode measurement.
- The initial customer validation excludes speculative decoding/MTP, PP,
  HiSparse, two-batch overlap, context parallelism, multi-node serving, and
  prefill/decode disaggregation.
- NVFP4 is lossy. Complete model-level accuracy validation before production
  deployment.
