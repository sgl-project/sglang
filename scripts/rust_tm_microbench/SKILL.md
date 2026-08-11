---
name: rust-tm-ttft-waterfall
description: Reproduce the rust tokenizer-manager TTFT stage waterfall — per-request latency breakdown from HTTP arrival through tokenize, scheduler queue, GPU prefill, and first-token return. Use when profiling rust-TM TTFT, attributing TTFT to pipeline stages, or rerunning the breakdown on a new model or commit.
---

# Rust TM TTFT waterfall

Measures, per request, where client TTFT goes on the `SGLANG_RUST_SERVER=1`
path: 12 rust stamps + the scheduler's `SchedulerReqTimeStats` + client stamps,
all `CLOCK_MONOTONIC` (identical clock for rust, python, and a same-host client
on Linux), joined by rid. Segments telescope exactly to client TTFT — a
non-zero `residual` row means a stamp is missing or double-counted.

Requires the stamp instrumentation on this branch
(`rust/sglang-server/src/ttft_stamp.rs` + call sites; scheduler line in
`output_streamer.py`). The stamps are log lines — no wire-format changes.

## 1. Build + install the rust TM

```bash
cd rust && cargo build --release -p sglang-server -j4
cp target/release/libsglang_server.so \
   ../python/sglang/srt/server/_core.cpython-312-*.so   # match the local python ABI tag
```

The profile overrides `CARGO_PROFILE_RELEASE_{DEBUG=2,STRIP=none,
FORCE_FRAME_POINTERS=true,LTO=thin}` are only needed if you also want
perf/flamegraphs; the waterfall itself needs a plain release build.

## 2. Launch

```bash
SGLANG_RUST_SERVER=1 PYTHONPATH=<repo>/python \
python3 -m sglang.launch_server --model-path Qwen/Qwen3.5-0.8B \
  --disable-radix-cache --stream-interval 1 --port 30800 > server.log 2>&1 &
```

- Keep `server.log` — it receives the stamp lines.
- `--disable-radix-cache` keeps every prefill real (prefix hits fake TTFT).
- 256K inputs on Qwen: add `--context-length 266240` and
  `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`.
- Hybrid-GDN models (Qwen3.5) on Blackwell may need
  `--linear-attn-decode-backend cutedsl` (triton decode kernel warmup crash);
  decode-kernel choice does not affect TTFT.

## 3. Measure (c=1 sequential)

```bash
nvidia-smi -i 0 -lgc 2032,2032     # lock clocks: c=1 DVFS parking adds 30-200ms noise
for L in 256 1024 16384 262144; do
  python3 scripts/rust_tm_microbench/driver.py --port 30800 \
    --input-len $L --num-requests 32 --warmup 3 --out in${L}.jsonl
done
nvidia-smi -i 0 -rgc
```

`--prompt-cache prompts_${L}.json` skips rebuilding random prompts on reruns
(256K prompts take ~1 min to build). The driver picks unique rids, so multiple
runs can share one server log.

## 4. Join into the waterfall

```bash
python3 scripts/rust_tm_microbench/postprocess.py \
  --server-log server.log --driver in*.jsonl --out breakdown.json
```

Prints mean ms per segment per input length. Verify before trusting anything:
all requests report complete stamp chains, and `residual` is ~0.000.

## 5. Teardown

```bash
kill $(lsof -ti :30800)    # NOT pkill -f: that pattern matches your own shell
```

## Reading the result

- `tokenize` (tok_start→tok_done) and `prefill_forward` (fwd_entry→prefill_fin)
  are normally the only segments above 1 ms; everything else is fixed
  µs-scale hand-off cost.
- `ring_wait_drain` grows to 1-2 ms on TP>1 (scheduler pickup + broadcast).
- Segment boundaries and their meaning are listed in `SEGMENTS` at the top of
  `postprocess.py`.
