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

## Op-point (both columns, same node)
Model `zai-org/GLM-5.1 (FP8)` (snapshot f396cf805…), TP=8, page 64, fp8 e4m3 KV, CUDA graph ON, radix
OFF, seed 431, gsp workload 4096 ISL (sys 2253 + q 1843, ~55% prefix cache) / 512 OSL, conc 32, 64 prompts.
- **DSA-native column:** DS off, `mem-fraction-static 0.8` (no DS table); `attention_backend='dsa'`.
- **DS column:** `--enable-double-sparsity`, 256 mask, top_k 2048, `mem-fraction-static 0.7` (DS
  TokenLabelTable overhead — the documented DS-vs-DSA mem-fraction asymmetry; the comparator ignores it).

## SLO gates (iii) decode TPS + (iv) P99 TTFT — PRELIMINARY (conc 32)
Decode TPS = `output_tokens / (e2e_latency − ttft)` per request (DEC-4). Via `bench_serving` →
`benchmark_compare.py` (decode-TPS primary, strict `P99 TTFT < 22 s`):

| Metric | DSA-native | DS (256 mask) |
|--------|-----------:|--------------:|
| Per-request decode tok/s **P50** | **29.63** | **17.12** |
| decode tok/s mean / P10 / min | 28.56 / 22.04 / 16.99 | 17.76 / 15.56 / 14.12 |
| **P99 TTFT (s)** | **15.55** | **43.37** |
| TTFT P50 (s) | 14.21 | 6.66 |
| TPOT P50 / P99 (ms) | 33.81 / 57.11 | 58.53 / 70.20 |
| Output throughput (tok/s) | 516.2 | 351.7 |
| Achieved concurrency | 31.99 | 26.46 (admission-bound) |
| Benchmark duration (s) | 63.5 | 93.2 |

**`benchmark_compare.py` DS SLO verdict (decode P50 ≥ 30, P99 TTFT < 22 s): FAIL.**
Artifacts: `runs/20260607_glm51_loop8/gate_dsa_c32.jsonl`, `gate_ds_c32.jsonl`.

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
- **SLO decode-TPS ≥ 30 + P99 TTFT < 22 s (mandatory): DS-on FAILS at conc 32 (preliminary)**; the
  DSA-native default nearly meets it (passes TTFT; TPS 29.63 at conc 32 — re-measure conc 16 for headroom).
- **NIAH / long-context recall (characterization-only): PENDING.**

**Open landing question (flagged for the user / DEC-2 call):** the client SLO is met by the **DSA-native
default** (which is what ships); **DS-on does not meet it on the standard workload** and is intended as a
default-off recall fallback. Whether that satisfies "SLO mandatory-to-land" (DEC-2) depends on reading the
bar as applying to the shipped default (→ land DS default-off, DSA meets SLO) vs to the DS-on path itself
(→ DS-on would not pass at this op-point). The plan's framing favors the former (DS = reversible opt-in).
**Not resolved here** — recorded for the landing decision.

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
