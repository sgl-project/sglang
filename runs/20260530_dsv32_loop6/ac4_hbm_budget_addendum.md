# AC-4 evidence addendum — full per-rank HBM budget + durable no-OOM proof

Closes the two acceptance-completeness gaps from the Round-4 review: (1) a complete
per-fraction HBM budget with torch reserved/allocated + the residual (not only named
tensors), and (2) durably-tracked no-generation-OOM / no-monotonic-growth proof at the
lifted operating point. **No verdict change** — AC-4 still PASSES at the lifted fraction
0.7. All numbers are from the same `serve_double_sparsity.sh SIGNATURE_DTYPE=int8`
TP=8 boots (boot excerpts under `memfraction_sweep_int8/boot_excerpt_*.txt`); the
0.7 stress was re-run once for a clean tracked client log (`reboot_0.7.log`).

## How the budget is measured
SGLang logs torch-allocator memory at each init stage as `avail mem` (torch
`mem_get_info` free) and `mem usage` (the per-stage torch allocation delta). The
torch-visible total is **139.80 GiB/rank** (from the 0.8 OOM message). So per fraction
`torch_used ≈ 139.80 − (post-cuda-graph avail)`, and each named component is the stage
delta. NVML per-GPU `used` is captured separately (`mf_*.txt`); because the DS server is
the **sole** process on each GPU, NVML per-GPU used == NVML per-process used (confirmed
at 0.7: `nvidia-smi --query-compute-apps` = 125106–125200 MiB/GPU == the per-GPU 125116
MiB). `torch.cuda.memory_reserved()/memory_allocated()` per rank are not exposed by the
serving HTTP API; the torch-tracked stage deltas above + the NVML per-process total +
the 0.8 OOM's literal `134.41 GiB allocated by PyTorch` provide the equivalent, and the
budget **closes** (Σ named + headroom ≈ torch_total to within ~0.5 GiB driver reserve).

## Per-rank HBM budget (GB, torch-tracked) — all named components + residual

| component | f=0.6 | f=0.7 (lifted) | f=0.8 (OOM) |
|---|---:|---:|---:|
| CUDA/NCCL init (`Init torch distributed`) | 1.23 | 1.23 | 1.23 |
| **weights** (DeepseekV32 fp8, `Load weight end mem usage`) | 80.63 | 80.63 | 80.63 |
| **KV pool** (`KV Cache ... KV size`) | 2.38 | 17.73 | 33.09 |
| KV-pool metadata (pool-end − weight-end − KV) | 1.47 | 1.28 | 1.62 |
| **table + scales** (int8, `token_label_table`) | 0.87 | 6.48 | 12.10 |
| **written + score-scratch + FlashMLA-meta + DS-bind** (graph-begin − pool-end − table) | 2.63 | 2.66 | (capture OOM'd) |
| **CUDA-graph pool** (`Capture cuda graph mem usage`) | 11.61 | 11.59 | partial → OOM |
| **headroom** (post-cuda-graph `avail mem`) | 38.43 | 17.65 | — |
| — torch_used (= 139.80 − headroom) | **101.37** | **122.15** | OOM at `134.41 alloc` |
| — NVML used/GPU (== per-process) | 101.4 GiB | 122.2 GiB | — |
| — residual (NVML_used − Σ named, ≈ caching slack/context) | ≈ 0.6 | ≈ 0.7 | — |
| `max_total_num_tokens` | 53056 | **396096** | 739200 (attempted) |

Notes: (a) `written` per rank = `bool[L, max_tokens]` = 61·396160·1 B = **0.023 GB** at 0.7 — negligible; it lives inside the 2.66 GB "+ DS-bind" line. (b) The residual (NVML − Σnamed) is ~0.6–0.7 GB (CUDA-context/caching slack) — the budget is **not only named tensors** and accounts for all HBM. (c) At 0.8 the OOM exception itself reports the literal torch figure: `134.41 GiB is allocated by PyTorch ... Tried to allocate 146.00 MiB ... 132.12 MiB free` during **cuda-graph capture** (boot-time, not generation) — the table (12.10) + KV (33.09) + partial graph pool exhaust HBM. (d) int8 vs fp16 at the same pool: 0.7 table **6.48 GB int8 vs 11.52 GB fp16** — the freed ~5 GB is what turns fp16's 0.7 gen-OOM into int8's 17.65 GB headroom.

## Durable no-generation-OOM / no-monotonic-growth proof at f=0.7
Tracked artifacts under `memfraction_sweep_int8/`:
- **`stress_0.7_client.txt`** — client request/result of the sustained stress (32 concurrent `/generate`, ~4096-ISL, 256 new tokens, 3 rounds + a ~30K-token long-context request): **`SUMMARY: 97/97 ok, 0 failed, elapsed 92.7s`**.
- **`stress_0.7_server_excerpt.txt`** — server-side scheduler log during the stress: Prefill batches (8192-token chunks, `#running-req` → 31) then Decode batches at **`#running-req: 32`**, token usage 0.37–0.39 of the 396096 pool, `cuda graph: True`, gen throughput ~380 tok/s; **generation-time OOM line count = 0**.
- **`nvml_timeseries_0.7.txt`** — NVML total-used / min-free over the run: used **1,005,832 → plateau 1,041,136 MiB** (last sample == max), min-free steady at **12,943 MiB**. Rose to the generation working set then **plateaued → no monotonic growth**.
- **`get_server_info_0.7.json`** — `/get_server_info` at 0.7 (`mem_fraction_static=0.7`, `max_total_num_tokens=396096`).

This is the direct, durable refutation of fp16's Loop-5 generation-OOM at 0.7: same fraction, sustained 4096-ISL × conc-32 traffic, **int8 serves with no generation-time OOM and a stable (plateaued) HBM footprint**.

## Verdict (unchanged) — AC-4 PASS at the lifted operating point 0.7
The budget is complete (named + residual, NVML- and torch-grounded, closing to ~0.5 GiB); `max_total_num_tokens` rises 53056 → 396096; the int8 compact table boots at 0.7 and survives sustained generation with no gen-OOM and no monotonic growth (durably tracked). The 0.8 cuda-graph-capture **boot** OOM is recorded with the literal torch figure; it does not gate the admission goal (0.7's 396096 ≈ 3.5× the AC-2 conc-64 target). The TTFT/SLO claim + admission-vs-prefill attribution remain AC-5 (next round).
