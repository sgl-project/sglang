# TokenLabelTable Footprint Feasibility Budget

Units below are GiB when derived from code/log formulas; SGLang logs label these as GB.

## Empirical Basis

TokenLabelTable signature footprint per rank:

```text
table_bytes_fp16 =
  num_layers_local * max_tokens * num_heads_local * label_dim * elem_size

V3.2 TP=8:
  L = 61
  H_local = 128 / 8 = 16
  label_dim = 16
  elem_size = 2 bytes
  max_tokens = max_total_num_tokens + page_size(64)

=> table_bytes_fp16 = 31,232 * max_tokens
=> table_GiB_fp16 = 2.9087067e-5 * max_tokens
```

The `written` tensor is `bool[L, max_tokens]` and is small relative to signatures.

Real anchors:

| anchor | status | max_total_num_tokens | table T | fp16 table | observed headroom |
|---|---:|---:|---:|---:|---:|
| A: `mem_fraction_static=0.6` | serves | 53,056 | 53,120 | 1.545 GiB | memory-pool-end 37.78 GiB, after-table ≈ 36.23 GiB |
| B: `mem_fraction_static≈0.70` | boots, gen OOM | 396,096 | 396,160 | 11.523 GiB | runtime headroom ≈ 12.29 GiB; later 248 MiB alloc fails |
| C: `mem_fraction_static=0.897` | boot OOM | 1,072,000 | 1,072,064 | 31.183 GiB | memory-pool-end 7.20 GiB, table alloc fails |

Raw KV cost from Anchor C:

```text
47.99 GiB / 1,072,000 tokens = 4.4767e-5 GiB/token
                              ≈ 46.94 KiB/token
```

The table is allocated after weights and KV pool allocation. It does not reduce the static budget directly; it competes with runtime/generation headroom. Raising `mem_fraction_static` increases the KV pool, which increases `max_total_num_tokens`, which increases TokenLabelTable bytes. This is the fixed point.

Approximate A/C static fit, using 139.80 GiB/rank:

```text
f(pool) ≈ (81.718 + 4.0749e-5 * pool) / 139.80
```

This maps 95K tokens to `f≈0.612`, 114K to `f≈0.620`, 396K to `f≈0.700`, and 1.072M to `f=0.897`. Anchor B’s stated `f≈0.77-0.8` at 396K is not consistent with that fit, so pool size and measured headroom are treated as authoritative; exact `mem_fraction_static` must be swept on hardware.

Approximate pre-table headroom fit from A/C:

```text
H_pre(pool) ≈ 39.372 - 3.0011e-5 * pool GiB
H_after_table(pool, lever) = H_pre(pool) - table_bytes(pool, lever)
```

## Admission Target

Anchor A admits 35.7 requests at nominal concurrency 64 with `max_total_num_tokens=53,056`.

```text
effective pool / admitted request = 53,056 / 35.7
                                  = 1,486.16 tokens/request

minimum pool for 64 admitted requests = 1,486.16 * 64
                                      = 95,114 tokens

20% margin target = 95,114 * 1.20
                  = 114,137 tokens
```

The admission target is therefore 95K tokens minimum, with 114K tokens as the working target. `f≈0.625` is a practical sweep point because the A/C fit predicts ≈139K tokens, or ≈93 admitted-request equivalents, while still far below the known 396K-token OOM region.

## Lever Budget

| lever | storage ratio vs fp16 token table | table @95K pool | table @114K pool | `f` needed for 95K / 114K | predicted after-table headroom @114K | B-pool cross-check @396K | predicted conc-64 admission |
|---|---:|---:|---:|---:|---:|---:|---:|
| fp16 table + raise `mem_fraction_static` | 1.0000x | 2.768 GiB | 3.322 GiB | ≈0.612 / ≈0.620 | ≈32.63 GiB | actual ≈12.29 GiB and gen OOM; model ≈15.96 GiB | 64 admitted at 95K; 20% margin at 114K |
| int8 same `label_dim` + fp16 scale | 0.5625x | 1.557 GiB | 1.869 GiB | ≈0.612 / ≈0.620 | ≈34.08 GiB | actual-adjusted ≈17.33 GiB; model ≈21.00 GiB | 64 admitted at 95K; 20% margin at 114K |
| page-level / two-stage fp16 | ≈1/64x | ≈0.043 GiB | ≈0.052 GiB | ≈0.612 / ≈0.620 | ≈35.89 GiB | actual-adjusted ≈23.63 GiB; model ≈27.30 GiB | 64 admitted at 95K; 20% margin at 114K |

Int8 math:

```text
fp16 vector per (layer, slot, head) = 16 dims * 2 bytes = 32 bytes
int8 signature = 16 dims * 1 byte = 16 bytes
scale = 1 fp16 per vector = 2 bytes
int8+scale = 18 / 32 = 0.5625x fp16
net table reduction = 1 / 0.5625 = 1.78x smaller
```

At the 114K working pool, int8 saves:

```text
3.322 - 1.869 = 1.453 GiB/rank
```

At the 396K B-pool, int8 saves:

```text
11.523 - 6.482 = 5.041 GiB/rank
```

Page-level math:

```text
one signature per 64-token page => approximately fp16_token_table / 64
```

At the 114K working pool, page-level saves ≈3.270 GiB/rank vs fp16 token-level. At the 396K B-pool, it saves ≈11.343 GiB/rank. This is structurally stronger, but it changes selector granularity and must be held to NIAH non-regression rather than bitwise or top-k equivalence.

## Binding Lever Decision

**Binding decision (authoritative for the implementation): build int8 same-`label_dim` as the compact `TokenLabelTable` path next.** int8 is predicted sufficient to restore nominal conc-64 admission with generation headroom at the admission target; the prediction includes the per-(layer, slot, head) scale-storage overhead and the larger-pool feedback (raising `mem_fraction_static` grows the pool, which grows the table). Page-level/two-stage is **not** built now; it is reserved strictly as an escalation if hardware shows int8 is insufficient (or a later requirement to operate at much larger pools).

This is the binding lever for the footprint implementation and it is **not** conditional on any prior no-code experiment. The footprint reduction is the principled fix to the root cause — the per-rank table is oversized relative to HBM headroom — and int8 reduces it directly (≈1.78×). The compact path is flag-gated with fp16 as the default until hardware validates it.

### fp16 lower-`mem_fraction` window — optional instrumentation only (does NOT gate the int8 build)

For completeness the budget records that, on paper, an fp16 table at a smaller `mem_fraction_static` bump also reaches the admission target:

```text
95K pool:  f≈0.612, after-table headroom ≈33.75 GiB
114K pool: f≈0.620, after-table headroom ≈32.63 GiB
f≈0.625:   pool≈139K, after-table headroom ≈31.17 GiB
```

The known generation OOM sits at `mem_fraction_static≈0.70` (the much larger ≈396K-token pool with an 11.5 GiB fp16 table), so an fp16 window around `f≈0.612-0.650` is not ruled out on paper. **However, this is a fragile, secondary observation, not a deployment lever and not a precondition for the int8 build:** it leaves the table at full fp16 size, so it depends on allocator residual / fragmentation margins and does not scale to the larger pools that 64K servability and conc-64 robustness need. It may, at most, be logged as one extra fp16 data point *during* the int8 compact-table mem-fraction sweep (the hardware mem-lift validation), purely for comparison. It must never replace, gate, or precede building the int8 compact table.

## Selection Equivalence Gate

Primary binding metric for int8:

```text
top-k overlap@2048 >= 0.99 vs fp16 baseline
```

Measure on synthetic V3.2-shaped selector inputs with the same query/channel-mask/table contents, comparing int8-dequantized scoring against fp16 scoring. Failure is overlap below 0.99.

Recorded-only diagnostics:

- selected-token recall
- score-error distribution
- rank displacement around the 2048 cutoff
- NIAH recall trend

## Minimum-Reversible-Opt-In Justification

The compact TokenLabelTable path is justified only to recover DS admission headroom; it is flag-gated, fp16 remains the default, and DSA remains the production default.

## Caveats and Hardware Confirmation

This is a predicted HBM budget, not a TTFT guarantee. Confirmation requires the mem-fraction sweep, full HBM accounting through NVML plus `torch` reserved/allocated residuals, and no-OOM long-generate traffic at the target workload.

Even after admission is fixed, conc-64 P99 TTFT may become prefill-compute-bound for 4096-ISL x 64 traffic. The downstream client-SLO benchmark must attribute TTFT into admission wait vs prefill compute before declaring the SLO resolved or missed.

## Provenance & Verification

This budget was authored as an independent expert analysis and integrated after verification against the real source and Loop-5 hardware logs:

- Table formula confirmed in `python/sglang/srt/layers/attention/double_sparsity/token_label_table.py` (`bytes_per_rank` / `estimate_hbm_bytes` = `num_layers_local * max_tokens * num_heads_local * label_dim * elem_size`).
- All three anchors are verbatim from `runs/20260528_dsv32_mvp/` boot logs (`token_label_table: 1.55 GB/rank L=61 T=53120`; `11.52 GB/rank T=396160`; the `Tried to allocate 31.18 GiB ... 7.20 GiB free` table-alloc OOM at `mem_fraction_static=0.897`).
- **Anchor B mem-fraction correction:** the source label said Anchor B `~=0.77-0.8`, but the budget's pool->`f` fit maps the 396K pool to `f≈0.70`, which matches the recorded Loop-5 finding "0.7 OOMs during generation". The Anchor B label has been corrected to `mem_fraction_static≈0.70`; the exact value is resolved by the hardware sweep regardless.
- **Binding-decision correction (Round-0 review):** an earlier draft of the Binding Lever Decision recommended a no-code fp16 lower-`mem_fraction` sweep *before* building the compact table, which conflicted with the committed plan (int8 is the chosen compaction lever; the compact-table mem-fraction validation must use the compact path). The decision now reads unambiguously: **build int8 same-`label_dim` for the footprint reduction next**; the fp16 lower-`f` window is optional comparison instrumentation only and must not gate, replace, or precede the int8 build.
- The linear A/C headroom fit overestimates real headroom by ~3.7 GiB at the 396K anchor (model ≈15.96 vs actual ≈12.29 GiB), so the hardware sweep — with full NVML + `torch.cuda.memory_reserved/allocated` accounting — is authoritative for the no-OOM determination, exactly as the caveat states.
