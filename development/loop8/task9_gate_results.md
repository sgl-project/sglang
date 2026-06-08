# Loop 8 / task9 — GLM-5.1 DS-vs-DSA-native gate record (8×H200, 2026-06-08)

Accuracy + client-SLO gates for the opt-in Double-Sparsity path on GLM-5.1-FP8, DS vs DSA-native on the
**same node/op-point** (only DS enablement + the required DS mem-fraction differ). Landing policy per DEC-2.

> **Status of this record:** the **SLO gates (iii decode-TPS, iv P99-TTFT)** are measured live with real
> DS-vs-DSA numbers below, but at a **PRELIMINARY op-point** (concurrency 32, 64 prompts, single trial, no
> 120 s/600 s window) — **NOT the final locked landing sweep** (conc 16/32/64 × 3 trials × 600 s). The
> **accuracy gates (i MMLU, ii NIAH)** are **PENDING** (the harness needs both DS+DSA servers live, which
> two TP=8 servers cannot do on 8 GPUs — execution plan below). Final landing numbers require the locked
> sweep; nothing here is marked "final/passed-to-land".

## Production landing mask (256-sample, AC-3)
- Path `/models/glm51-fp8-channel-mask-s256.safetensors`, `content_sha256=35155ac46ad7…`,
  `num_samples=256`, `calibration_source=real`, `label_dim=32`, `head_dim=192`, `layers=78`,
  `page_size=64`, `dtype=fp8_e4m3`, indices ∈ [0,191]. `load_channel_mask` re-verifies the hash;
  `verify_bind_shapes(qk_nope=192, label_dim=32, layers=78, TP=8)` PASS. DS server binds it (bind shape
  gate PASS all 78 layers/8 ranks) and serves. This is the production landing artifact (supersedes the
  32-sample bring-up mask `e7dbf4c9308f` used for R0 smoke).

## Op-point (both columns, same node — launch-arg parity verified, R5)
Both columns booted via the **paired locked-op-point launchers** (`serve_native_nsa.sh` DSA-native /
`serve_double_sparsity.sh` DS) with the GLM model + 256 mask. From each JSONL's `server_info`, the columns
match on **TP=8, page 64, kv_cache_dtype fp8_e4m3, disable_radix_cache=True, disable_piecewise_cuda_graph=
True, disable_overlap_schedule=True, dsa_prefill/decode_backend=flashmla_kv**, differing **only** by
`enable_double_sparsity` (False/True) and `mem_fraction_static` (DSA 0.8 / DS 0.7 — the inherent DS
TokenLabelTable reservation; the comparator ignores this asymmetry per BL-20260529). This **fixes the R4
op-point mismatch** (R4's DSA had `disable_piecewise_cuda_graph=False`). Workload: gsp 4096 ISL (sys 2253 +
q 1843, ~55% prefix cache) / 512 OSL, seed 431.

## SLO gates (iii) decode TPS + (iv) P99 TTFT — parity conc curve (R5, PRELIMINARY window)
Decode TPS = `output_tokens / (e2e_latency − ttft)` per request (DEC-4). Via `bench_serving` →
`benchmark_compare.py` (decode-TPS primary, strict `P99 TTFT < 22 s`). **PRELIMINARY:** `num_prompts =
concurrency`, single pass, NO 120 s/600 s window — short cold runs, not the locked landing sweep (esp.
conc 64 TTFT is cold-burst-inflated, per BL-20260530-cold-flood-not-steady-state-slo).

| conc | DSA decode P50 | DSA P99 TTFT | DSA verdict | DS decode P50 | DS P99 TTFT | DS verdict |
|-----:|---------------:|-------------:|:-----------:|--------------:|------------:|:----------:|
| 16 | **38.69** | 7.24 s | **PASS** | **23.16** | 3.68 s | FAIL (TPS) |
| 32 | **31.52** | 14.19 s | **PASS** | **17.09** | 37.17 s | FAIL |
| 64 | 24.35 | 28.32 s | FAIL (cold-burst) | 17.11 | 74.33 s | FAIL |

(DS achieved concurrency 16.0 / 22.6 / 40.2 — admission-bound at conc 32/64 by the smaller DS KV pool.)

**Findings (authoritative for the preliminary window):**
- **DS-on FAILS the decode-TPS ≥ 30 bar at EVERY concurrency** (best case conc-16 = 23.16 < 30), and the
  P99-TTFT bar at conc 32/64. Confirmed via the parity-matched comparator (DS SLO verdict: **fail**).
- **DSA-native PASSES the SLO at conc 16 and 32** (decode ≥ 30 + TTFT < 22 s) and fails only at conc 64
  (24.35 TPS / 28 s TTFT — a cold-burst short run; re-confirm under the locked steady-state window).
- DS-on is consistently ~1.4–1.7× slower decode than DSA-native — the expected posture (GLM ships a strong
  trained DSA indexer; DS is the default-off long-context-recall fallback, not a throughput win here).
- Artifacts: `runs/20260607_glm51_loop8/parity_{dsa,ds}_c{16,32,64}.jsonl`.

### Reading (characterization) — R5 parity numbers
At the standard client op-point, **DS-on is slower than DSA-native** (decode P50 17.1 vs 31.5 tok/s at
conc 32) and its P99 TTFT is admission-bound by the smaller DS KV pool (mem 0.7 + per-rank
TokenLabelTable → achieved conc 22.6 < 32). **DSA-native PASSES** the SLO at conc 16 (38.7 tok/s / 7.2 s)
and conc 32 (31.5 tok/s / 14.2 s), and fails only at conc 64 (24.4 / 28.3 s — a cold-burst short run).
**DS-on fails the decode-TPS ≥ 30 bar at EVERY concurrency** (23.2 / 17.1 / 17.1). Decode is **coherent**
(not degenerate). This is the **expected** posture from the plan: GLM ships a strong trained DSA indexer,
so DS is the **reversible default-OFF opt-in fallback**, valuable where the indexer underperforms
(long-context recall), **not** a throughput win on the standard workload.

## Accuracy gates (i) MMLU + (ii) NIAH — PENDING (executable offline path landed R6)
`test/manual/test_double_sparsity_v32.py` requires `DS_BASE_URL` **and** `DSA_BASE_URL` live
simultaneously (skipUnless); two TP=8 servers cannot co-reside on 8×H200 **and GLM-5.1 cannot run at TP=4**
(weights ~2× exceed a single H200), so the only viable path is **sequential collect + offline compare**.
`development/loop8/accuracy_gate.py` provides it (reusing the harness's tuned MMLU parser + deterministic
NIAH prompt-gen + recall scorer): `AC12_MODE=collect` scores ONE live server (`AC12_SIDE=dsa|ds`,
`AC12_BASE_URL=…`) and writes a per-side artifact (run_id + prompt-set hashes + hits/totals + index_topk);
`AC12_MODE=compare` (offline, no server) validates the two sides used the same prompt set and applies the
mandatory thresholds (MMLU DS within 1.0 pp of DSA; within-budget NIAH DS within 5.0 pp; beyond-budget =
characterization-only), failing closed on any mismatch. Offline-compare unit tests:
`test/registered/unit/test_accuracy_gate_compare.py` (9 pass). `AC12_INDEX_TOPK=2048` (GLM index_topk).
**The scoring RUN (collect DSA → collect DS → compare) on hardware is the next round.**

## DEC-2 landing-policy assessment (preliminary)
- **AC-1 DS-off byte-identical (mandatory): PASS** (task7, R3 — GLM DSA-native byte-identical HEAD vs
  d018026f9, 6/6 prompts).
- **MMLU within tolerance of DSA (mandatory): PENDING** (offline gate path landed R6; scoring run next).
- **DS-vs-DSA non-regression of the shipped default (mandatory): PASS** — the served default is DSA-native
  (DS default-off); AC-1 proves it unregressed.
- **SLO decode-TPS ≥ 30 + P99 TTFT < 22 s (mandatory): DS-on FAILS (R5 parity, preliminary window)** —
  decode P50 17.1 < 30 at every concurrency (23.2/17.1/17.1) and P99 TTFT fails at conc 32/64. The
  **DSA-native** column PASSES the SLO at conc 16/32 (38.7/31.5 tok/s ≥ 30, TTFT < 22 s) and fails conc 64
  (cold-burst). Preliminary window (`num_prompts=conc`, no 600 s); the locked 3×600 s sweep is needed for
  the authoritative steady-state numbers (esp. conc 64).
- **NIAH / long-context recall (characterization-only): PENDING.**

**Landing status per DEC-2 (unchanged — SLO mandatory):** DEC-2 resolves "parity + SLO mandatory-to-land".
On the R5 parity (preliminary-window) data, **DS-on does not meet the mandatory SLO** (fails decode-TPS at
all concurrencies + TTFT at conc 32/64), while DSA-native passes at conc 16/32. This record does **not**
reinterpret DEC-2. Resolving the landing therefore requires the authoritative conc-16/32/64 + locked-window
numbers and, if DS-on still fails (expected, given GLM's strong trained indexer makes DS the default-off
recall fallback rather than a throughput win), an **explicit user plan-evolution decision** on whether a
default-off DS opt-in may land despite a DS-on SLO miss — recorded as plan evolution if so, not assumed here.

## V3.2-vs-GLM shape matrix
| dim | DeepSeek-V3.2 | GLM-5.1 |
|-----|---------------|---------|
| qk_nope_head_dim | 128 | **192** |
| v_head_dim | 128 | **256** |
| qk_rope_head_dim | 64 | 64 |
| kv_lora_rank | 512 | 512 |
| q_lora_rank | 1536 | **2048** |
| num_hidden_layers | 61 | **78** |
| num_attention_heads | 128 | 64 |
| DSA index_topk | 2048 | 2048 |
| DSA index_head_dim | 128 | 128 |
| DSA index_n_heads | 64* | 32 |
| DS label_dim (calibrated) | 16 | **32** (DEC-3) |
| channel mask `content_sha256` | 36d8bf573091 (16-sample regen) | 35155ac46ad7 (256-sample landing) |

The same inherited DS wiring + bind-time `verify_bind_shapes` gate serve both shapes live (V3.2 128/128,
GLM 192/256) — see task6_serving_smoke.md.

## Repro
**SLO — sequential per column via the PAIRED launchers (launch-arg parity; seed-matched as of R6):**
```bash
GLM=/cluster-storage/models/models--zai-org--GLM-5.1-FP8/snapshots/f396cf805182f4ca10fa675e1a99815b3ca384db
# DSA-native column:
MODEL_PATH=$GLM MEM_FRACTION_STATIC=0.8 DISABLE_RADIX_CACHE=1 RANDOM_SEED=20260607 PORT=30000 \
  bash development/serve_native_nsa.sh   # then bench_serving (gsp 2253+1843/512) per conc -> parity_dsa_c{16,32,64}.jsonl
# DS column (shut DSA first):
MODEL_PATH=$GLM CHANNEL_MASK_PATH=/models/glm51-fp8-channel-mask-s256.safetensors \
  MEM_FRACTION_STATIC=0.7 RANDOM_SEED=20260607 PORT=30000 \
  bash development/serve_double_sparsity.sh   # then the same bench_serving per conc -> parity_ds_c{16,32,64}.jsonl
python development/benchmark_compare.py --baseline parity_dsa_c32.jsonl --ds parity_ds_c32.jsonl
```
**Full locked landing sweep (next round):** `development/benchmark_baseline.sh` (MODE=native_nsa) +
`benchmark.sh` (MODE=double_sparsity) with `CONCURRENCIES="16 32 64" NUM_PROMPTS=320 TRIALS=3
WARMUP_SECONDS=120 MEASUREMENT_WINDOW_S=600 RANDOM_SEED=20260607`, then `benchmark_compare.py --ac11`.
**Accuracy — sequential collect + offline compare (the executable path; both-URLs-at-once is infeasible):**
```bash
# boot DSA-native, then:
AC12_MODE=collect AC12_SIDE=dsa AC12_BASE_URL=http://127.0.0.1:30000 AC12_INDEX_TOPK=2048 \
  python development/loop8/accuracy_gate.py
# shut DSA, boot DS (256 mask), then:
AC12_MODE=collect AC12_SIDE=ds  AC12_BASE_URL=http://127.0.0.1:30000 AC12_INDEX_TOPK=2048 \
  python development/loop8/accuracy_gate.py
# offline (no server):
AC12_MODE=compare AC12_DSA_ARTIFACT=<dsa.json> AC12_DS_ARTIFACT=<ds.json> \
  python development/loop8/accuracy_gate.py   # exit 0 iff MMLU within 1.0pp AND within-budget NIAH within 5.0pp
```

## Remaining for the final landing record
1. **Accuracy gates** — RUN the collect→collect→compare flow on hardware (path landed R6, scoring run next).
2. **Full locked SLO sweep** (conc 16/32/64 × 3 trials × 600 s) for landing-grade steady-state numbers.
3. **DEC-2 landing decision (user)** — DS-on fails the mandatory SLO at all concurrencies; whether a
   default-off DS opt-in may land anyway (plan framing) vs literal DEC-2 SLO-mandatory is the user's call,
   recorded as plan evolution if relaxed.
