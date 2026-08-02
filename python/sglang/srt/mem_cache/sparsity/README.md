# HBM-resident KV-cache sparsity

This package contains the common policy/controller/adaptor contracts for
post-hoc sparse attention. The initial runtime keeps every KV entry allocated
in HBM and changes only the subset visible to attention. It does not free,
compact, migrate, or swap KV-cache storage.

## Initial supported path

- NVIDIA CUDA, decoder-only MHA/GQA text models
- FA3 prefill and decode backends
- eager execution (CUDA Graph is disabled while this path is experimental)
- normal prefill followed by non-speculative decode
- page-granular StreamingLLM-style sink + recent visibility

Other backends, speculative decoding, local/hybrid attention, PD
disaggregation, DP/CP attention, physical eviction, and hierarchical placement
are rejected explicitly.

The implementation keeps logical policy output independent from physical KV
locations:

1. `SparsityPolicy` returns request-relative logical page indices.
2. `KVSparsityController` applies request/step/layer lifecycle rules.
3. `HBMResidentPlacement` translates logical pages through `req_to_token`.
4. `FlashAttentionVisibilityAdaptor` rewrites and later restores FA3 metadata.

During prefill, the controller publishes lifecycle contexts without changing
attention visibility. This lets a later KV-derived policy build auxiliary
representations without adding another model-forward hook.

## Launch

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-4B \
  --attention-backend fa3 \
  --enable-kv-cache-sparsity \
  --kv-cache-sparsity-config '{
    "policy": "streaming_llm",
    "backend": "fa3",
    "min_sparse_tokens": 4096,
    "policy_config": {
      "sink_pages": 4,
      "recent_pages": 2048
    }
  }'
```

The runtime `--page-size` and optional JSON `page_size` must match. With the
default page size of one, `sink_pages` and `recent_pages` are token counts.

## Validation

Run the CPU policy/controller/adaptor conformance tests with:

```bash
python -m pytest -q \
  test/registered/unit/mem_cache/test_kv_sparsity_framework.py
```

For an end-to-end dense-path parity smoke test, first set
`min_sparse_tokens` above the tested context length. Then lower it below the
context length and benchmark a genuinely sparse budget. Use the same FA3 eager
configuration for the dense baseline so CUDA Graph does not confound the
comparison.

```bash
python -m sglang.benchmark.serving \
  --backend sglang \
  --base-url http://127.0.0.1:30000 \
  --dataset-name random-ids \
  --random-input-len 8192 \
  --random-output-len 256 \
  --num-prompts 64 \
  --max-concurrency 16 \
  --warmup-requests 4 \
  --tokenize-prompt
```

Launch the dense comparison with the same `--attention-backend fa3` and
`--cuda-graph-config '{"decode":{"backend":"disabled"},"prefill":{"backend":"disabled"}}'`
settings used by the sparse path.

Report input/output throughput, inter-token latency, the exact sparse budget,
GPU type, dtype, batch/concurrency, and whether prefix caching was warm. This
mode reduces attention work and memory traffic but does not increase KV-cache
capacity because the complete cache remains resident.
