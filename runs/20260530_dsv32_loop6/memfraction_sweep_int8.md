# AC-4 — mem-fraction lift + no-OOM validation (int8 compact table)

**Question:** does the int8 compact `TokenLabelTable` let DS boot at a higher
`mem_fraction_static` *without* the generation-time OOM fp16 hit at 0.7, restoring
admitted KV capacity? Real single-node TP=8 H200, `serve_double_sparsity.sh`
`SIGNATURE_DTYPE=int8`, Loop-5 mask, Option B (fp8 KV, page 64, flashmla_kv,
overlap+piecewise-cuda-graph disabled).

## Mem-fraction sweep — full HBM budget (per rank unless noted)

| `f` | int8 table (sig+scales) | KV pool | `max_total_num_tokens` | mem-pool-end avail (torch) | post-cuda-graph avail (torch) | NVML used/GPU (idle) | NVML min-free (idle) | result |
|---:|---:|---:|---:|---:|---:|---:|---:|:--|
| 0.6 | 0.87 GB | 2.38 GB (53056 tok) | **53056** | 53.72 GB | 38.34 GB | 101.4 GiB (103842 MiB) | 39.0 GiB | serves |
| 0.7 | 6.48 GB | 17.73 GB (396096 tok) | **396096** | 38.33 GB | 17.56 GB | 122.2 GiB (125116 MiB) | 17.5 GiB | **serves + sustained-gen no-OOM** |
| 0.8 | 12.10 GB | 33.09 GB (739200 tok) | 739200 (attempted) | 22.68 GB | — (OOM) | — | — | **boot OOM (cuda-graph capture)** |

- Weights: `Load weight begin avail=137.97 GB` → ~84 GB/rank after load (the rest of NVML-used is KV + table + cuda-graph pool + CUDA-context/NCCL residual).
- cuda-graph pool: `Capture cuda graph end mem usage = 11.61 GB` (0.6) / `11.59 GB` (0.7) — a fixed ~11.6 GB boot cost (52 Option-B batch sizes).
- `max_total_num_tokens` **rises monotonically** with `f`: 53056 → 396096 → 739200, exactly the admission lift the footprint reduction was for.
- int8 vs fp16 table at the SAME pool: 0.7 → **6.48 GB int8 vs 11.52 GB fp16** (0.5625×); 0.6 → 0.87 vs 1.55 GB. The freed ~5 GB at 0.7 is what converts fp16's gen-OOM into int8's no-OOM headroom.
- Full per-fraction captures (NVML per-GPU, `/get_server_info`, log lines): `memfraction_sweep_int8/mf_{0.6,0.7,0.8}.txt` + `boot_excerpt_*.txt`.

## No-generation-OOM validation at the lifted fraction (0.7)
Sustained stress: **32 concurrent** `/generate`, ~4096-ISL prompts, 256 new tokens, **3 rounds** + one ~30K-token long-context request → **97/97 requests OK, 0 failed** (94 s). **No generation-time OOM** in the server log.

NVML time series during the run (`nvml_timeseries_0.7.csv`, 29 samples / ~90 s): total-used **1,001,106 → plateau 1,049,104 MiB**; min-free **17,947 → 11,947 MiB**. Memory rose ~48 GB to the generation-activation working set then **plateaued** (last sample == max) — **no monotonic growth**. This is the direct refutation of the fp16 failure mode: **fp16 at 0.7 generation-OOM'd in Loop-5; int8 at 0.7 serves a sustained 4096-ISL × conc-32 load with no OOM and a stable footprint.**

## The 0.8 ceiling (honest, AC-2-framed)
At `f=0.8` the int8 path **boot-OOMs during CUDA-graph capture** (not generation): the int8 table (12.10 GB) + KV pool (33.09 GB) leave only 22.68 GB pool-end headroom, and the ~11.6 GB cuda-graph capture pool + the table do not co-fit — verbatim: `torch.OutOfMemoryError: ... Tried to allocate 146.00 MiB. GPU 3 ... 132.12 MiB free ... Capture cuda graph failed`. This is the AC-2 "larger-pool feedback" (raising `f` grows the pool *and* the table). It is **not** a generation-time OOM and **not** the AC-4 negative test — the achieved no-OOM operating point is **0.7**.

Per the verified AC-2 budget, the real target is *enough admitted KV capacity with headroom to admit conc-64*, **not** `f=0.8` as a number, and the plan calls 0.7 "acceptable as a more conservative first step." **0.7's `max_total=396096` exceeds the AC-2 conc-64 admission target (~95K min / ~114K @20% margin) by ≈3.5×**, so the spine's admission goal is met with large margin at 0.7. Reaching 0.8 would require trimming the fixed Option-B cuda-graph batch set (a productionization pass explicitly out of scope) and is unnecessary for the admission goal; page-level/two-stage escalation is therefore **not** triggered (int8 met the goal).

## Verdict — PASS (lifted operating point = 0.7, int8 compact table)
`max_total_num_tokens` rises with `f` (53056 → 396096); the int8 compact table boots at the lifted fraction (0.7) and **survives sustained 4096-ISL × conc-32 generation with no generation-time OOM and no monotonic memory growth**; the full HBM budget (NVML + torch) is recorded; `/get_server_info` captured (`mf_0.7.txt`). The footprint→admission lift the loop exists for is validated on hardware. The 0.8 cuda-graph-capture boot-OOM is recorded transparently; it does not gate the admission goal (met ≈3.5× over at 0.7).

## Caveat (for AC-5)
This validates admission *capacity* + no-OOM. Whether the restored admission actually moves **P99 TTFT < 22 s** at conc 16/32/64 — and whether conc-64 TTFT is admission-bound vs prefill-bound — is the AC-5 client-SLO benchmark (next round), which must carry the measured admission-wait vs prefill-compute attribution.
