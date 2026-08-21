---
name: rust-tm-flamegraph
description: Collect a CPU flamegraph of the rust tokenizer-manager threads on a live SGLANG_RUST_SERVER=1 server (perf attach + FlameGraph render). Use when investigating where the rust TM burns CPU inside a stage (e.g. tokenize) — for per-request stage latency use the rust-tm-ttft-waterfall skill instead.
---

# Rust TM flamegraph

Attaches perf to the named rust threads of a live scheduler process while
`driver.py` generates load, and renders an SVG. Reference workload below:
Qwen3.5-35B-A3B-FP8 with 16K inputs (tokenize ≈ 26 ms/request, so the
tokenizer thread accumulates plenty of samples in a 64-request burst).

## 1. Build with profiler-friendly symbols

The stock release profile strips symbols and omits frame pointers — perf would
show hex addresses. Override via env (no Cargo.toml edit):

```bash
cd rust
CARGO_PROFILE_RELEASE_DEBUG=2 CARGO_PROFILE_RELEASE_STRIP=none \
CARGO_PROFILE_RELEASE_FORCE_FRAME_POINTERS=true CARGO_PROFILE_RELEASE_LTO=thin \
  cargo build --release -p sglang-server -j4
cp target/release/libsglang_server.so \
   ../python/sglang/srt/server/_core.cpython-312-*.so   # match the local ABI tag
```

## 2. Launch the server

```bash
SGLANG_RUST_SERVER=1 PYTHONPATH=<repo>/python \
python3 -m sglang.launch_server --model-path Qwen/Qwen3.5-35B-A3B-FP8 --tp 2 \
  --disable-radix-cache --stream-interval 1 --port 30800 > /tmp/server.log 2>&1 &
grep -m1 "fired up and ready" <(tail -f /tmp/server.log)
```

(Hybrid-GDN Qwen models on Blackwell may also need
`--linear-attn-decode-backend cutedsl` — triton decode-kernel warmup crash.)

## 3. Collect

```bash
bash scripts/rust_tm_microbench/collect_flamegraph.sh 30800 \
  Qwen/Qwen3.5-35B-A3B-FP8 16384 64 rust_tm_in16384.svg
```

The script finds the scheduler pid, resolves the rust thread ids
(`tokenizer-*`, `tm-ingress-*`, `tm-egress-*`, `detokenizer-*`, `api-runtime`,
`tokio-rt-worker`), records at 999 Hz with frame-pointer unwinding while the
driver runs, and renders via github.com/brendangregg/FlameGraph (auto-cloned
to /tmp). Open the SVG in a browser: click to zoom, Search to highlight.

## Pitfalls

- `perf record -o` (and the script's workdir) must be LOCAL disk. An output
  file on a network mount fails every perf mode with EFAULT
  `"failed to write perf data: Bad address"` — it looks exactly like a
  sandbox restriction and is not.
- `cargo flamegraph --pid <scheduler pid>` also works (whole process, DWARF
  unwinding, Ctrl-C to stop) but cannot select threads and produces huge
  perf.data — keep captures under ~30 s.
- Cycle sampling only sees ON-CPU time: parked threads are invisible, so idle
  hand-off waits never appear. Expect the tokenizer thread to dominate
  (~98% of rust-TM CPU); that matches the waterfall. For latency attribution
  (queues, GPU forward), use the rust-tm-ttft-waterfall skill.
