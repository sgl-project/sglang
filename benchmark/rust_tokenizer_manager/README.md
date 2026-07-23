# Rust vs Python tokenizer manager: multimodal concurrency sweep

`bench_mm_ab.py` compares the two SGLang frontend stacks on an image workload
by sweeping client concurrency and plotting TTFT and ITL:

- **python arm** (`SGLANG_RUST_SERVER=0`): FastAPI api-server +
  `TokenizerManager` + `DetokenizerManager` (separate processes, zmq IPC).
- **rust arm** (`SGLANG_RUST_SERVER=1`): the embedded Rust server
  (`rust/sglang-server`) running api-server / tokenizer manager / detokenizer
  as Rust threads inside the scheduler process. Multimodal preprocessing still
  runs in Python (Rust mm workers calling `MmProcessorHost`, same
  `mm_processor` stack), so the A/B isolates the frontend / tokenize / IPC /
  detokenize differences.

Each arm is launched once with identical server args. Because every level
replays the same seeded images, both caches that could short-circuit repeats
are disabled: the radix/KV prefix cache (`--disable-radix-cache`) and the
vision-embedding LRU (`SGLANG_VLM_CACHE_SIZE_MB=0`). Every
concurrency level replays the same seeded dataset (built once and cached)
through the native `/generate` endpoint — the Rust api-server does not serve
`/v1/chat/completions`. Images travel as base64 data URIs in `image_data`;
note the Rust api-server needed its axum 2 MB `DefaultBodyLimit` disabled for
multi-MB image payloads (`rust/sglang-server/src/api_server/mod.rs`).

Requires the `sglang_server` PyO3 module
(`cd rust/sglang-server && maturin develop --release`).

## Usage

```bash
python benchmark/rust_tokenizer_manager/bench_mm_ab.py --gpu 1

# smaller/faster
python benchmark/rust_tokenizer_manager/bench_mm_ab.py --gpu 1 \
    --concurrencies 1 16 128 --num-prompts 256 --image-resolution 360p

# image-count sweep (fixed concurrency; x-axis = image count)
python benchmark/rust_tokenizer_manager/bench_mm_ab.py --gpu 1 \
    --concurrencies 64 --num-prompts 1024 --image-counts 1 2 3 4 --output-len 256
```

Defaults: `Qwen/Qwen3.5-0.8B`, 1x720p image + 128 text tokens in / 64 out per
request, concurrency 1 → 1024. Each level runs
`min(num_prompts, max(128, 32*concurrency))` requests (2048 cap by default):
enough samples for steady state everywhere without serializing thousands of
~200 ms requests at low concurrency, which would take minutes per level.

## Output

Written to `--output-dir` (default `results/<timestamp>/`):

- `sweep.png` — TTFT and ITL vs concurrency (log-log; mean solid, p99 dashed,
  one color per arm), plus a per-arm table on stdout
- `raw.json` — per-arm `{concurrency: {mean/p99 ttft/itl, request_throughput}}`
- `{arm}.jsonl`, `server_{arm}.log` — bench_serving lines and server logs
