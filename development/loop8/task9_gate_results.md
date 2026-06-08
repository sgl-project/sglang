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

### Reading (characterization)
At the standard client op-point, **DS-on is slower than DSA-native** (decode 17 vs 30 tok/s) and its P99
TTFT (43 s) is admission-bound by the smaller DS KV pool (mem 0.7 + per-rank TokenLabelTable → achieved
conc 26.5 < 32). DSA-native nearly meets the bar (decode P50 29.63, marginally under 30 at conc 32;
P99 TTFT 15.6 s < 22 s — passes TTFT). Decode is **coherent** (not degenerate). This is the **expected**
posture from the plan: GLM ships a strong trained DSA indexer, so DS is the **reversible default-OFF
opt-in fallback**, valuable where the indexer underperforms (long-context recall), **not** a throughput
win on the standard workload.

## Accuracy gates (i) MMLU + (ii) NIAH — PENDING
`test/manual/test_double_sparsity_v32.py` requires `DS_BASE_URL` **and** `DSA_BASE_URL` live
simultaneously (skipUnless), but two TP=8 servers cannot co-reside on 8×H200. **Execution plan:** run the
two columns sequentially and reconcile offline (boot DSA → score MMLU/NIAH → persist; boot DS → score →
persist; compare), OR run both at TP=4 (4+4 GPUs) for the accuracy comparison only (accuracy is
op-point-insensitive vs the SLO). `AC12_INDEX_TOPK=2048` (GLM index_topk = 2048); NIAH within-budget
(≤2048 tokens) is the fair recall gate; beyond-budget (4K/16K/64K) is characterization-only.

## DEC-2 landing-policy assessment (preliminary)
- **AC-1 DS-off byte-identical (mandatory): PASS** (task7, R3 — GLM DSA-native byte-identical HEAD vs
  d018026f9, 6/6 prompts).
- **MMLU within tolerance of DSA (mandatory): PENDING** (accuracy gate not yet run).
- **DS-vs-DSA non-regression of the shipped default (mandatory): PASS** — the served default is DSA-native
  (DS default-off); AC-1 proves it unregressed.
- **SLO decode-TPS ≥ 30 + P99 TTFT < 22 s (mandatory): DS-on FAILS (preliminary, conc 32)** — decode P50
  17.12 < 30 and P99 TTFT 43.4 s ≥ 22 s. Note the **DSA-native** column ALSO does **not** pass the
  decode-TPS bar at conc 32 (P50 29.63 < 30); it passes only TTFT (15.6 s < 22 s). Both are preliminary
  (conc 32, 64 prompts, no locked window); the conc-16/32/64 curve + locked sweep are needed for the
  authoritative numbers.
- **NIAH / long-context recall (characterization-only): PENDING.**

**Landing status per DEC-2 (unchanged — SLO mandatory):** DEC-2 resolves "parity + SLO mandatory-to-land".
On the preliminary conc-32 data, **DS-on does not meet the mandatory SLO** (fails both decode-TPS and P99
TTFT), and even DSA-native is marginally under the 30-TPS bar at conc 32. This record does **not**
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
**SLO (per column, sequential on the same node):**
```bash
# DSA column: boot GLM DS-off (mem 0.8), then:
python -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 30000 \
  --dataset-name generated-shared-prefix --gsp-num-groups 1 --gsp-prompts-per-group 64 \
  --gsp-system-prompt-len 2253 --gsp-question-len 1843 --gsp-output-len 512 --gsp-range-ratio 1.0 \
  --num-prompts 64 --max-concurrency 32 --seed 431 --output-file gate_dsa_c32.jsonl --output-details
# DS column: boot GLM DS-on with the 256 mask (mem 0.7) via development/serve_double_sparsity.sh
#   (MODEL_PATH=<glm snapshot> CHANNEL_MASK_PATH=/models/glm51-fp8-channel-mask-s256.safetensors), then
#   the same bench_serving line → gate_ds_c32.jsonl.
python development/benchmark_compare.py --baseline gate_dsa_c32.jsonl --ds gate_ds_c32.jsonl
```
**Full locked landing sweep (next):** `development/benchmark_baseline.sh` (MODE=native_nsa) + `benchmark.sh`
(MODE=double_sparsity) with conc 16/32/64, TRIALS=3, WARMUP_SECONDS=120, MEASUREMENT_WINDOW_S=600, then
`benchmark_compare.py --ac11`. **Accuracy:** `DS_BASE_URL=… DSA_BASE_URL=… AC12_INDEX_TOPK=2048
PYTHONPATH=python python -m pytest test/manual/test_double_sparsity_v32.py -v` (sequential per the plan above).

## Remaining for the final landing record
1. Accuracy gates (MMLU + NIAH within/beyond budget), DS vs DSA.
2. Full locked SLO sweep (conc 16/32/64 × 3 trials × 600 s) for landing-grade decode-TPS + P99 TTFT.
3. Resolve the DEC-2 SLO-applies-to-default-vs-DS-on landing question with the user.
