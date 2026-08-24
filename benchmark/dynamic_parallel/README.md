# Dynamic attention-parallel benchmark

This benchmark compares TP, prefill CP, DCP, and runtime-selected execution on
the same deterministic request grid. It records latency, cache hits, selected
mode metrics, generated token IDs, and output-token logprobs in JSONL.

Run it from the repository root:

```bash
python -m sglang.benchmark.dynamic_parallel \
  --model-path deepseek-ai/DeepSeek-V3.1 \
  --modes tp,prefill_cp,dcp,dynamic \
  --batch-sizes 1,8,32 \
  --input-lengths 1024,4096,16384,32768 \
  --prefix-hit-ratios 0,0.5,0.9 \
  --output-length 32 \
  --repeats 3 \
  --mode-server-args-json '{"dcp":"--dcp-comm-backend a2a"}' \
  --strict-parity \
  --result-file dynamic_parallel.jsonl
```

Each grid point flushes the cache. For nonzero prefix-hit ratios, the driver
first primes the exact shared prefix, then submits one batched `/generate`
request. Put `tp` first when comparing several modes so later records receive a
parity result against the matching TP record.

Useful ROCm overrides:

```bash
--attention-backend aiter \
--server-env-json '{"SGLANG_JIT_DEEPGEMM_PRECOMPILE":"0"}' \
--extra-server-args '--page-size 1 --disable-piecewise-cuda-graph'
```

The `dynamic` mode expects the runtime selector flags implemented by this
feature branch. `--dynamic-include-dcp` additionally enables decode DCP.
