---
name: diffusion-perf
description: Deprecated alias (merged into diffusion-kernel).
user-invocable: false
allowed-tools: Bash, Read
argument-hint: <model-path> [--prompt "..."] [--baseline baseline.json]
---

# Diffusion Performance Measurement

This skill has been merged into the canonical docs under `diffusion-kernel`:

- `../diffusion-kernel/diffusion-benchmark-and-profile.md` → **Perf dump & before/after compare**

Follow that document as the single source of truth:

- Always run `sglang generate ... --warmup --perf-dump-path <file>.json`
- Use `python python/sglang/multimodal_gen/benchmarks/compare_perf.py <baseline.json> <new.json>` to generate a PR-ready comparison table
