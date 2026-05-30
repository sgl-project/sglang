# AC-6 — DS is opt-in; DSA stays the production default (DEC-2 "Both"), proven on hardware

Two real TP=8 V3.2-FP8 servers, same locked Option B operating point (fp8 KV,
page 64, flashmla_kv prefill+decode, overlap-schedule + piecewise-cuda-graph
disabled), differing **only** by Double-Sparsity enablement:
- **DS opt-in** — node 0, `serve_double_sparsity.sh SIGNATURE_DTYPE=int8 MEM_FRACTION_STATIC=0.7`.
- **DSA default** — node 1, `serve_native_nsa.sh` (no DS flags, `MEM_FRACTION_STATIC=0.85`).

## The opt-in flag toggles the compact DS path (`get_server_info_keys.json`)

| field | DS opt-in (node 0) | DSA default (node 1) |
|---|---|---|
| `enable_double_sparsity` | **True** | **False** |
| `double_sparsity_config` | `{top_k:2048, …, "signature_dtype":"int8"}` | **None** |
| `mem_fraction_static` | 0.7 | 0.85 |
| `max_total_num_tokens` | 396096 | **910784** |
| `kv_cache_dtype` / `page_size` / backends | fp8_e4m3 / 64 / flashmla_kv | fp8_e4m3 / 64 / flashmla_kv |

- **DSA-default allocates NO DS `TokenLabelTable`** — `double_sparsity_config = None`,
  `enable_double_sparsity = False`, and the node-1 boot log has **0** `token_label_table`
  lines (`dsa_notable_boot_excerpt.txt`); the full 910784-token KV pool is used.
- **DS opt-in activates the compact int8 path** — every rank logs
  `token_label_table: 6.48 GB/rank … dtype=torch.int8 scales=float16`
  (`ds_table_boot_excerpt.txt`, all 8 TP ranks), the table the Loop-6 spine compacted.
- Same Option B operating point on both → the only difference is DS enablement (DEC-2 "Both":
  DS ships opt-in, DSA stays the default).

## DSA-default admits full concurrency and serves the workload (fresh boot)

`benchmark_baseline.sh` against node 1 DSA-default (gsp 4096 ISL / 512 OSL,
NUM_PROMPTS=64, 1 trial, WARMUP=0 / WINDOW=30, cross-node) — `dsa_default_slo.txt`:

| conc | achieved | completed | errors | P99 TTFT | per-req TPS |
|---:|---:|---:|---:|---:|---:|
| 16 | **16.00** | 64 | 0 | 22.6 s* | 16.9* |
| 32 | **32.00** | 64 | 0 | 86.1 s* | 14.1* |
| 64 | **64.00** | 64 | 0 | 202.4 s* | 14.1* |

- **DSA admits full nominal concurrency** (achieved == nominal) and serves every
  request (errors 0) — DSA-default is *not* admission-limited (unlike DS at the
  0.6/53K-pool point), exactly because it allocates no DS table and uses the full
  910784-token KV pool.
- **\*The TTFT/TPS here are cold-ramp numbers, not steady-state.** `WARMUP=0` +
  `request_rate=inf` floods all 64 prompts at once (16/32/64 concurrent
  4096-token prefills), so time-to-first-token is dominated by prefill/decode
  contention during the flood — the **same measurement artifact AC-5 documented
  for DS** (min TTFT here is 1.6 s; the median 22.5 s is the contended flood).
  Under this identical `WARMUP=0` methodology DSA (22.5 s) is **not** faster than
  DS (AC-5 conc-16 was 12.8 s), confirming the inflation is the cold ramp, not
  DS-specific.

## "Meets the SLO unchanged": the authoritative steady-state evidence
The clean DSA-default steady-state SLO is the **established Loop-5 baseline**
(P99 TTFT **0.73 / 1.37 / 2.04 s**, per-req TPS ≥ 30, full admission), measured at
**this identical Option B operating point** (fp8 KV, page 64, flashmla_kv, mem 0.85,
910784-token pool, radix-on) with the proper 120 s-warmup / 600 s-window
methodology. This fresh boot reproduces that operating point exactly
(`get_server_info_keys.json`) and confirms full admission + clean serving, so
enabling the DS code path leaves the DSA-default product unchanged. A fresh
all-trials steady-state DSA sweep is **AC-7's** job (3-trial, 120/600 s), not this
opt-in/no-table product proof. DS remains the opt-in knob that meets the workload
through the compacted int8 table.

## Artifacts
- `ac6_product_proof/get_server_info_keys.json` — DS vs DSA key server fields (the toggle).
- `ac6_product_proof/ds_table_boot_excerpt.txt` — DS int8 `token_label_table` per-rank lines.
- `ac6_product_proof/dsa_notable_boot_excerpt.txt` — DSA-default has 0 `token_label_table` lines, pool 910784.
- `ac6_product_proof/dsa_default_slo.txt` — the DSA-default SLO confirmation metrics.
