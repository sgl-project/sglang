# Unified Engram storage with Mooncake

EngramStore provides one storage abstraction for N-gram/hash embeddings and
direct token-indexed per-layer embeddings across the supported model families.
Models provide row IDs and interpret returned bytes; storage ownership and
lookup are shared. Its table contract supports local and remote backends; this
SGLang integration currently selects local storage.

`--engram-backend mooncake` stores lookup tables in host memory owned by
Mooncake. `--engram-local-dir` selects the snapshot directory (default:
`/dev/shm/sglang-engram`). Install a Mooncake build with `USE_CUDA=ON` that
provides `mooncake.engram.EngramStore` and its private CUDA extension.

```bash
python -m sglang.launch_server \
  --model-path /models/gemma-3n-E2B-it \
  --engram-backend mooncake \
  --engram-local-dir /dev/shm/sglang-engram
```

The default `native` backend preserves the original model implementation.
This integration covers:

| Model family | Stored table | Interpretation in SGLang |
| --- | --- | --- |
| DeepSeek V4.1 | TP-local FP8 weights and E8M0 scales packed per row | Block dequantization; existing TP/DP/CP collectives |
| Qwen4-Exp / Qwen3.8-Flash-Next | TP-local N-gram / PLE rows | Original table scale, convolution and request history |
| LongCat Ngram | TP-local over-embedding rows | Original hash, projection and token history |
| Gemma3n / Gemma4 | Per-layer input embedding rows | Original scaling, reshape, projection and gating |

Hashing, tokenizer behavior, weights names and collective placement remain in
their native model implementations. The adapters preserve the existing
checkpoint loading attributes. Mooncake receives only row IDs and bytes.

## Lifetime and memory

The first eager lookup publishes an immutable snapshot of the loaded CPU
parameters. Each table and rank gets an anonymous mapping (`persistent=False`),
so different model instances cannot accidentally reuse stale weights. Native
CPU parameters are retained; budget host memory for both them and the snapshot,
plus temporary packing for DeepSeek's weights and scales. CUDA registration
pins the snapshot pages. Disk-backed snapshots still require enough RAM for
these pinned pages.

Warm up all tables before CUDA graph capture. Lookup runs on the current CUDA
stream, including Qwen's existing prefetch stream. Normal stream waits and
graph dependencies provide completion ordering. No asynchronous request
object is exposed. Destroy graphs before releasing their tables.

Tables are immutable after publication: restart the engine to change these
weights. This initial adapter does not support live weight updates or remote
Mooncake storage. Anonymous mappings have no named snapshot files and are
released on process exit, including when the engine kills its workers.

## Tests

Run `pytest -q test/manual/test_mooncake_engram.py` on a CUDA host with the
matching Mooncake build. It compares raw lookup, table scaling, block scaling,
shard masking and CUDA graph replay against native kernels and tensor references. End-to-end
tests additionally need the original model checkpoints.
