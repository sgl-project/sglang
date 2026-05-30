# Real-mask NIAH Non-Regression — int8 compact DS vs fp16 Loop-5 DS baseline (AC-3.1 / DEC-8)

**Question (DEC-8):** does the int8 compact `TokenLabelTable` regress real DS needle
recall vs the fp16 Loop-5 DS baseline on the actual calibrated Loop-5 channel mask?
DS recall is already weak (75/5/0 vs DSA 100), so there is no headroom to lose.

## Setup (real TP=8 hardware, single node + cross-node DSA)
- **DS (int8 compact)** — node 0 (`h200-10-220-51-16`), port 30000, TP=8, `mem_fraction_static=0.6`,
  `SIGNATURE_DTYPE=int8`, Loop-5 mask `/models/dsv32-fp8-channel-mask.safetensors`
  (sha `7b3207cae888`, L=61 H=128 label_dim=16). Boot proof (`ds_int8_boot_proof.txt`):
  `token_label_table: 0.87 GB/rank ... dtype=torch.int8 scales=float16` (vs fp16's 1.55 GB —
  the **0.5625×** reduction confirmed on hardware) and `double_sparsity_config='{...,"signature_dtype": "int8"}'`.
  Decode is coherent (raw `/generate` "The capital of France is" → " Paris.").
- **DSA (live reference)** — node 1 (`h200-10-220-51-5`), port 30001, TP=8, `mem_fraction_static=0.85`,
  `serve_native_nsa.sh` (the model's native trained sparse attention). Reached cross-node from node 0.
- **Harness:** `test/manual/test_double_sparsity_v32.py -k niah`, `AC12_NIAH_NUM_PROMPTS=20`,
  `DS_BASE_URL=http://127.0.0.1:30000 DSA_BASE_URL=http://10.220.51.5:30001`.
  Result: **2 passed, 2 skipped, 5 subtests passed** (308 s). Per-length artifacts in
  `real_mask_niah_int8/ac12_niah_*.json`; pytest log `real_mask_niah_int8/niah_pytest.log`.

## Result — NON-REGRESSION PASS at every length

| length | int8 DS recall (now) | fp16 Loop-5 DS baseline | live DSA | int8 ≥ fp16? | served |
|---:|---:|---:|---:|:--:|:--:|
| 1024 words (within budget) | **100%** | 100% | 100% | ✅ | 20/20 |
| 1536 words (within budget) | **100%** | 100% | 100% | ✅ | 20/20 |
| 4K | **85%** | 75% | 100% | ✅ (+10pp) | 20/20 |
| 16K | **5%** | 5% | 100% | ✅ (=) | 20/20 |
| 64K | **0%** (unservable) | 0% (unservable) | 100% | ✅ (=) | 0/20 |

- fp16 Loop-5 DS baseline: `runs/20260528_dsv32_mvp/ac12_results/ac12_niah_*.json`.
- Pass rule (pre-declared, from the review): `int8_ds_recall ≥ fp16_ds_recall` at every comparable
  length **and** no new DS unservable error where the fp16 baseline served. Both hold.

## Verdict — PASS
The int8 compact table **does not regress** real-mask DS needle recall vs the fp16 Loop-5 DS
baseline: it **matches** at 1024/1536/16K, **matches** the 64K admission limit (both unservable at
mem-0.6, identical pool 53056), and is **+10 pp at 4K** (85% vs 75% — within the ±5 pp per-needle
granularity at 20 prompts; not a regression). This is consistent with the synthetic selection gate
(top-k overlap@2048 ≥ 0.99) and the dense=100% property: int8 changes which 2048 tokens are
selected only at the boundary, which is recall-neutral here. The live DSA reference is 100% at every
length, confirming the paired cross-node setup is sound. Combined with the decode-scoring microbench
(int8 overhead ≪ the 33.9→30 TPS budget), the compact path is recall-neutral **and** TPS-neutral.

## Caveat
The 64K case is an admission limit (prompt 69970 tok > DS pool 53056), identical for int8 and fp16 at
mem-0.6 — it is not a recall difference. Lifting the pool (AC-4 mem-fraction sweep with the int8
table) is the next round; this NIAH was run at the Loop-5 mem-0.6 operating point so the comparison to
the fp16 Loop-5 baseline is apples-to-apples.
