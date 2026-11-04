# CPU prefix reuse with complete maintenance responses

Put the shared maintenance manual before the changing observation:

```text
shared manual + current observation -> /v1/completions -> full text + cache usage
```

This example sends three synthetic tickets and records complete model text, HTTP time to
first text, full-response time, reported cache hits and simple format/label checks.
It does not send commands to hardware. Incorrect outputs remain in the JSONL file.
An optional outer Markdown fence is normalized for schema/label checks; the raw
text is preserved and `raw_json_valid` separately reports whether it was unfenced
JSON. No fields or labels are repaired. Chat serialization explicitly requests
`return_dict=False` so the prompt contains integer token IDs across tokenizer API
versions, rather than a `BatchEncoding` object.

The client is CPU independent. The following fallback launch was verified on
SGLang v0.5.13 and main `8e37c6cf` with the
[optional CPU startup fix](https://github.com/sgl-project/sglang/pull/44064),
Linux x86-64 with AVX2 and no AMX, CPU-only Torch, and
Qwen2.5-Coder-1.5B-Instruct. It uses FP32 `torch_native` attention; it is not a
benchmark of optimized Xeon AMX or W8A8 kernels. Prepare a server using the
[CPU installation guide](https://docs.sglang.io/docs/hardware-platforms/cpu_server).
The launch below assumes the model and its tokenizer are already downloaded.

For the verified AVX2 fallback, use the `v0.5.13` CPU dependency manifest
(`python/pyproject_cpu.toml`) with Python 3.12 and the PyTorch CPU wheel index
(`https://download.pytorch.org/whl/cpu`). The tested versions are Torch
`2.12.0+cpu`, Transformers `5.8.1` and httpx `0.28.1`. This native FP32 path did
not require the optimized CPU extension. On an unpatched main, optional CPU
kernel imports can still require `sgl_kernel` during startup; the linked fix
restores native serving without that extension. The optimized x86 CPU extension
targets AVX-512/AMX and should be installed on supported hardware following the
guide. AOT-only features still require it; this example uses ordinary FP32
serving and radix cache.

```bash
export SGLANG_USE_CPU_ENGINE=1
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
# Choose four available CPUs on your host; check taskset -pc $$ first.
export SGLANG_CPU_OMP_THREADS_BIND=0,2,4,6
taskset -c 0,2,4,6 python -m sglang.launch_server \
  --model-path /path/to/pinned-model \
  --served-model-name Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --device cpu --dtype float32 --attention-backend torch_native \
  --disable-overlap-schedule --disable-cuda-graph --tp 1 \
  --max-running-requests 1 --max-total-tokens 4096 --context-length 2048 \
  --enable-cache-report --host 127.0.0.1 --port 30000
```

After startup warmup finishes (`The server is fired up and ready to roll!`):

```bash
python examples/runtime/cpu_prefix_reuse.py \
  --model-path /path/to/the-same-pinned-model \
  --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --cache-report-enabled --output /tmp/cpu-prefix-on.jsonl
```

Each stdout/JSONL row includes `status`, `manual_prefix_tokens`, `cached_tokens`,
`ttft_s`, `full_s`, `completion_tokens`, `text` and output checks. Expect a cold
first request followed by cache hits; the cache-off run should report zeros.
Request failures are retained with `status="error"` and cause exit code 1.
Model format or prediction errors remain evaluation results and do not change
the exit code.

For a cache-off comparison, stop that server and start a fresh one with identical
options plus `--disable-radix-cache`. Rerun the client with a different output path.
Keep the tokenizer/model revision, input hashes, affinity and output limit equal.
Compare the first cold request separately from the subsequent two requests.
This short example demonstrates functionality; repeated orders and more tickets
are needed for a performance conclusion. Full-response time includes generated
work, so check completion-token counts and prediction quality alongside latency.

Cache reporting must be enabled on the server before passing
`--cache-report-enabled` to the client. In the tested OpenAI-compatible endpoint,
final usage can omit `prompt_tokens_details` when a hit is zero; this is recorded
as zero only under that explicit flag. Without it, a missing hit remains `null`.
An absent final usage with reporting enabled, an invalid hit count, or a truncated
stream is an error. Shared-prefix length comes from the final serialized token IDs,
not the manual's character count. A reported hit can include additional common
tokens beyond the manual. Changing the early prompt prevents reuse.

Hardware and software versions should accompany any benchmark result. The
fallback is deliberately small and serial and does not imply hard real-time
control, general industrial diagnostic accuracy, or MCU model deployment.

Client regression tests require no server or model weights and are registered
in the CPU CI suite:

```bash
python test/registered/unit/examples/test_cpu_prefix_reuse.py -f
```
