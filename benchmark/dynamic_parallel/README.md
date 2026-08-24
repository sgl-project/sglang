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
  --mode-server-args-json '{"dcp":"--dcp-comm-backend a2a","dynamic":"--dynamic-attn-parallel-min-prefill-tokens 8192"}' \
  --server-env-json '{"SGLANG_JIT_DEEPGEMM_PRECOMPILE":"0","SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK":"32768","MORI_SHMEM_HEAP_SIZE":"16G"}' \
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
feature branch. Dynamic prefill remains TP unless
`--dynamic-attn-parallel-min-prefill-tokens` is supplied through the dynamic
mode's server arguments; choose it from a measured crossover grid.
`--dynamic-include-dcp` additionally enables decode DCP over replicated KV,
which preserves the CP prefill path. Compact striped KV is an
explicit experiment; add `--dynamic-striped-min-context 8192` to assign prompts
at or above that length to the striped pool. The harness also disables radix
cache for this opt-in path because cross-residency eviction accounting is not
yet supported. Striped prefill currently uses the full-prefix DCP assembly path
and should be benchmarked separately.

For MLA CP with MoRI, set
`SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK >= --chunked-prefill-size`.
Large values also need a larger `MORI_SHMEM_HEAP_SIZE`; 16 GiB is sufficient
for the 32K-token example above on MI355X.
