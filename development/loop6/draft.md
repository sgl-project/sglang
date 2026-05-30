# Loop 6 Draft — Make Double Sparsity Shippable for the Client SLO on V3.2

## Objective

Make **Double Sparsity (DS) itself** pass the immediate client SLO in
`development/CLIENT_SLOS.md` on the production workload, and decide whether to
invest further in DS on a model that already ships a native sparse indexer.

The one thing that, done, makes this loop a success:

> **DS serves the client workload (4096 ISL / 512 OSL / conc 16–64 / ~55% cache)
> at `P99 TTFT < 22 s` AND `≥ 30 TPS/req`, on real hardware, at the locked Option B
> operating point — measured as an absolute pass/fail against the client SLO, not a
> DS-vs-DSA ratio.**

Everything else in this loop is either a small hardening, a strategic decision, or
explicitly-gated recall R&D.

## Why this loop exists

Loop 5 shipped the smoke MVP + the loop4-compatible MVP and closed AC-0/1/1.1/4/6/8/9/10/1b/Q,
plus AC-11 (executed, recorded directional miss) and AC-12 (MET under the user-authorized
DS-fair re-scope). DS demonstrably serves V3.2 FP8 at TP=8, page 64, fp8 KV, radix-on.

But Loop 5 graded "MVP" against **DS-vs-DSA parity** (the internal loop-4 bar). Re-graded
against the **client's** bar (`CLIENT_SLOS.md`), the picture is sharper and the remaining gap
is small:

| Client SLO (immediate) | Target | DS measured (Loop 5 AC-11, conc 16/32/64) | Verdict |
|---|---|---|---|
| Per-request throughput | **≥ 30 TPS/req** | 34.0 / 33.9 / 33.9 tok/s (p50) | ✅ already MET |
| Tail latency | **P99 TTFT < 22 s** | 57.7 / 132.9 / 292.0 s | ❌ MISS (hard) |
| Model / knobs | V3.2 FP8, TP, CUDA graphs, radix | all enabled and recorded | ✅ MET |
| Workload | 4096 ISL / 512 OSL / conc 16–64 / ~55% cache | `benchmark.sh` shape matches exactly | ✅ MET |

**The single client-facing blocker is P99 TTFT, and it is not a speed problem** — DS per-request
generation already beats the 30 TPS SLO. It is an **admission/queue** problem: DS reserves a
per-rank `TokenLabelTable` (~8 GB/rank, fp16) on top of the ~84 GB/rank V3.2 FP8 weights, so it
must run at `mem_fraction_static=0.6` (DSA runs at 0.85). The small KV pool admits only
**14.5 / 24.6 / 35.7** of the nominal 16 / 32 / 64 concurrency, so requests queue and TTFT
explodes. Raising mem past 0.6 currently **OOMs DS during generation**.

➡️ **Shrinking the `TokenLabelTable` footprint is the lever that converts DS from "fails the
client SLO" to "shippable."** That is the spine of this loop.

A second, strategic finding governs how far past the SLO to invest: **DS cannot beat V3.2's
native DSA on long-context recall** — the shared `flashmla_kv` decode kernel hard-caps DS
selection at the model's `index_topk=2048`, and DS's offline channel-mask selector is inferior to
V3.2's *trained* DSA indexer at that budget (Loop-5 NIAH 4K/16K/64K = 75% / 5% / 0% vs DSA 100%).
DS's value is clearest on models **without** a trained sparse indexer. Decide this before spending
GPU-hours on recall R&D (DEC-1).

## How this loop differs from Loop 5

Loop 5's failure mode was building CPU scaffolding instead of *running* code; its discipline was
"every round drops a hardware artifact." Loop 6 is a **research loop with a decision gate**:

1. **Answer the strategic gate (DEC-1) first.** It gates only the expensive Tier-2 recall R&D —
   the Tier-1 engineering wins pay off regardless. Do not spend rounds on a learned selector or a
   kernel variant until the gate is decided.
2. **The engineering wins are the safe mainline.** The footprint → admission → SLO chain is the
   spine. A code-only round (e.g. the footprint change + unit tests) is fine **if** the next round
   validates it on hardware; two code-only rounds in a row with no hardware artifact is a stall.
3. **Do NOT re-litigate the DS-fair AC-12 re-scope.** It was DECIDED in Loop 5 (user-authorized).
   Loop 6 may *characterize* beyond-budget recall further, but the gate definition is settled.

**Anchor:** `loop6-base` at the Loop-5 final commit (`989975625` or later) on
`dev/double-sparsity-standalone`. The Loop-5 mask `/models/dsv32-fp8-channel-mask.safetensors`
already exists — reuse it; regenerate only if a recipe field changes. The no-env-override radix-on
path is done.

---

## Strategic decision gate (DEC-1 — decide FIRST, gates Tier 2)

**Is Double Sparsity worth pursuing past the engineering wins on a model that already ships a
trained native sparse indexer (DSA)?**

- On V3.2, DS is capped at the native `index_topk=2048` budget by the *shared* decode kernel AND
  uses an inferior *offline* selector — so it cannot match (let alone beat) DSA long-context recall
  at the shared budget.
- DS's value proposition is clearer on models WITHOUT a trained sparse indexer (relevant to the
  deferred GLM-5.1 and 128k requirements).
- This gate determines whether **Tier 2 (recall R&D)** is in scope at all. Capture the answer as a
  `DEC-1` decision doc dropped under `runs/<date>_dsv32_loop6/ds_on_v32_decision.md`.

If the gate **closes** Tier 2, that is a **legitimate Loop 6 outcome** — the loop ships the Tier-1
client-SLO MVP and explicitly records "recall R&D not pursued on V3.2 because DSA dominates at the
shared budget; revisit on a no-native-indexer model." A closed gate is not a stall.

---

## Scope — IN

### Tier 1 — engineering wins (the client-SLO spine; pay off regardless of DEC-1)

1. **TokenLabelTable footprint reduction** *(handoff #2 — THE client-SLO blocker)*.
   Shrink the per-rank `TokenLabelTable`
   (`python/sglang/srt/layers/attention/double_sparsity/token_label_table.py`, ~8 GB/rank fp16
   today) so DS can serve at a higher `mem_fraction_static` without the generation-time OOM seen at
   0.7. Candidate levers (pick the minimum that works; do not over-engineer): int8-symmetric
   signatures with per-layer/slot/head scales applied at scoring, a narrower `label_dim`, or a
   tighter slot model. **Selection numerics must be preserved** — the change must not regress DS
   selection/recall. Keep fp16 as the default behind a flag until the compact path has unit + hardware
   evidence.

2. **mem_fraction lift + no-OOM validation** *(handoff #2)*. With the smaller table, boot DS at a
   higher `mem_fraction_static` (target decided in DEC-4; e.g. 0.8) and prove `max_total_num_tokens`
   rises with **no generation-time OOM** under sustained long `/generate`.

3. **⭐ Direct client-SLO validation** *(NEW — the loop's done-criterion)*. Run the **full** client
   workload (`benchmark.sh` at `NUM_PROMPTS=320`, conc 16 / 32 / 64) and assert **absolute
   `P99 TTFT < 22 s` and `≥ 30 TPS/req`** for DS. This is the artifact that says "DS is shippable for
   the client." (Loop 5's AC-11 used `NUM_PROMPTS=64` and reported only DS/DSA ratios — insufficient
   for an absolute SLO claim.)

4. **AC-11 directional re-sweep at the lifted mem fraction** *(handoff #3, DEC-7 from Loop 5)*.
   Re-run the 3-trial DS+DSA sweep (conc 16/32/64, 120 s warmup / 600 s window) at the new operating
   point; confirm DS achieved concurrency now tracks nominal; update `ac11_analysis.md` verdict.
   (Expected to improve once admission is fixed; the DS-vs-DSA *parity* miss at conc 16/32 shares the
   same root cause as the TTFT miss.)

5. **64K servability** *(handoff #2; also unblocks the deferred 128k requirement)*. At the lifted
   mem fraction, confirm a 64K-context `/generate` no longer returns HTTP 400 (or document the new
   admission ceiling). This is a *servability* win; 64K recall accuracy is separate (Tier 2).

6. **AC-12 within-budget gate from real token counts** *(handoff #4 / Codex queued #1)*. Change the
   harness to assert `within_budget` from the **actual** `usage.prompt_tokens` (or tokenized chat
   length), not the 1024/1536 **word-count** proxy. Rename `length_tokens`→`length_words` or add an
   `input_tokens` field. Must **not** change the DECIDED DS-fair gate definition; re-run the gate and
   diff against the word-count proxy to show it was safe (or correct it).

### Tier 2 — DS long-context recall R&D (GATED on DEC-1; expect to defer to a dedicated loop)

7. **DS long-context recall** *(handoff #1)* — **only if DEC-1 opens it.** A `flashmla_kv` decode-kernel
   variant accepting `top_k > index_topk` (today asserts `indices.shape[-1] == dsa_index_topk` in
   `dsa_backend.py`) AND/OR a query-aware / learned DS selector that places the needle in the 2048
   budget. Measure NIAH 4K/16K/64K recall delta vs the Loop-5 baseline DS 75% / 5% / 0%. This is
   GPU- and engineering-heavy — a new kernel variant or a distilled selector is not a one-round task.

## Scope — OUT

- **No new fixture/scaffolding code.** Loop 5's harnesses, comparator, and serve/bench scripts are
  the tools; use them, don't rebuild them.
- **Do NOT re-open the AC-12 DS-fair re-scope.** Settled in Loop 5. Beyond-budget recall may be
  *characterized* further but the gate definition is fixed.
- **Tier 2 recall R&D is OUT unless DEC-1 explicitly opens it.**
- **Deferred client requirements are OUT of this loop:** 128k ISL / 1024 OSL, nvfp4/mxfp4 weights,
  the performant-knobs × DS matrix (DP Attention, MTP/EAGLE, EP, explicit/mixed chunked prefill,
  overlap scheduling, piecewise CUDA graph), and GLM-5.1. They are their own downstream loops.
- **Productionization passes are OUT** unless trivially in the way: page-size flexibility beyond 64,
  removing dev-only `SGLANG_DS_*` env gates, CI registration of the manual gates, upstreaming/PR
  hygiene, multi-node TP scaling. Tracked in `development/roadmap.md` §5 for Loops 6–8.

---

## Acceptance criteria (draft — gen-plan will formalize positive/negative tests)

Each AC drops an artifact under `runs/<date>_dsv32_loop6/` (e.g. `runs/20260530_dsv32_loop6/`).

- **AC-L6-0 (DEC-1 strategic gate)** — *analyze*. A decision doc `ds_on_v32_decision.md` records:
  pursue Tier-2 recall R&D on V3.2 or not, with the `index_topk`/shared-kernel/selector rationale,
  and the explicit consequence for Tier 2. **Positive:** the doc states a decision and its Tier-2
  consequence. **Negative:** starting any Tier-2 R&D before this doc exists is out of order.

- **AC-L6-1 (TokenLabelTable footprint)** — *coding*. The per-rank table memory is reduced by the
  target factor, and DS selection numerics are preserved. **Positive:** a unit test shows the compact
  table's selected-token set matches the fp16 baseline within tolerance on a synthetic shape, and a
  measured per-rank byte count drops by the target. **Negative:** any selection divergence beyond
  tolerance, or the compact path becoming the default before hardware validation, fails.

- **AC-L6-2 (mem-fraction lift, no OOM)** — *hwrun*. DS boots at the higher `mem_fraction_static`
  (DEC-4 target) and survives a sustained long `/generate`. **Positive:** a mem-fraction sweep log
  (0.6 → … → target) shows `max_total_num_tokens` rising and a long-context generation completing
  with **no generation-time OOM**; `/get_server_info` recorded. **Negative:** a generation-time OOM
  at the target mem fraction fails (the table is still too big — iterate or reconsider the admission
  model).

- **AC-L6-3 (⭐ client-SLO validation — the done-criterion)** — *hwrun*. Full client workload
  (`benchmark.sh`, `NUM_PROMPTS=320`, conc 16 / 32 / 64). **Positive:** DS **absolute
  `P99 TTFT < 22 s`** AND **`≥ 30 TPS/req`** at conc 16 and 64; a `client_slo_report.md` records the
  absolute numbers vs the SLO with valid `.meta.json` sidecars. **Negative:** any conc with
  `P99 TTFT ≥ 22 s` or `< 30 TPS/req` fails the client-SLO MVP claim (record it as a follow-up with
  the admission/compute breakdown).

- **AC-L6-4 (AC-11 directional re-sweep)** — *hwrun*. 3-trial DS+DSA sweep at the lifted operating
  point, radix-on both sides, per-side `mem_fraction_static` consistency enforced. **Positive:** DS
  achieved concurrency tracks nominal (≈100%); comparator emits an updated TPS/TTFT summary and
  `ac11_analysis.md` is refreshed from the new artifacts. **Negative:** a sweep that hides
  queue-dominated admission (achieved ≪ nominal without disclosure) is invalid.

- **AC-L6-5 (64K servability)** — *hwrun*. **Positive:** a ~70K-token `/generate` returns 200 (no
  HTTP 400) at the lifted mem fraction, with the served `max_total_num_tokens` recorded; OR a
  documented new admission ceiling if 64K still doesn't fit. **Negative:** silently re-recording the
  Loop-5 HTTP 400 without the lifted-mem retry fails.

- **AC-L6-6 (AC-12 within-budget from real token counts)** — *coding*. **Positive:** the harness
  records `usage.prompt_tokens` per NIAH prompt and asserts `within_budget` from it; the re-run gate
  still PASSES (DS-fair definition unchanged) and a diff shows the word-count proxy was safe (or is
  corrected). **Negative:** any change that alters the DS-fair gate thresholds/definition fails.

- **AC-L6-7 (Tier 2 recall R&D — GATED)** — *coding/hwrun*, **only if AC-L6-0 opened it.**
  **Positive:** a selector or `top_k > index_topk` kernel-variant change with a NIAH 4K/16K/64K
  recall delta artifact showing movement vs DS 75% / 5% / 0%, and the TPS/TTFT cost recorded.
  **Negative:** starting this before AC-L6-0, or letting it block the Tier-1 spine, fails the loop
  discipline.

---

## Pending user decisions (resolve in gen-plan discussion mode)

- **DEC-1 — Strategic gate (above).** Pursue Tier-2 recall R&D on V3.2, or cap at Tier-1 engineering
  wins? Gates AC-L6-7 and the downstream 128k/GLM framing. *(PENDING — decide early.)*
- **DEC-2 — "Shippable" definition.** Is the deliverable "DS meets the client SLO *itself*," or "DS
  available as an opt-in knob while DSA is the default that meets the SLO"? (DSA already meets both
  SLOs trivially.) *(PENDING.)*
- **DEC-3 — TTFT target source.** Confirm the client SLO is **absolute `P99 TTFT < 22 s`** at the
  client workload (not a DS-vs-DSA ratio), validated at full `NUM_PROMPTS=320`. *(PENDING.)*
- **DEC-4 — Footprint approach + target mem fraction.** Which compaction lever (int8 signatures /
  narrower `label_dim` / tighter slot model), the target `mem_fraction_static` to validate
  (0.7 / 0.8?), and the OOM-safety bar. *(PENDING.)*
- **DEC-5 — Deployment topology.** Single-node TP=8 vs multi-node for the client deliverable.
  *(PENDING — affects how the SLO is validated.)*

---

## Hardware

See `CLUSTER.md` (2-node 8×H200; node 0 local `h200-10-220-51-16`, node 1
`h200-10-220-51-5` via `ssh double-sparsity` / `rx devbox exec double-sparsity --no-tmux --rank 1`).
DSv3.2 FP8 weights at `/cluster-storage/models/deepseek-ai/DeepSeek-V3.2`. Loop-5 mask already on
disk at `/models/dsv32-fp8-channel-mask.safetensors` (regenerate only if a recipe field changes).
Default ports: workers 30001, router 30000, prometheus 29000.

---

## Critical path (concrete commands)

All runs use the **Option B operating point** encapsulated by the `serve_*.sh` / `benchmark*.sh`
scripts (TP=8, fp8 KV, page 64, `flashmla_kv` prefill+decode, overlap-schedule + piecewise-cuda-graph
disabled, radix-on via the config-bound fixture state). `mem_fraction_static` is the lever Loop 6
deliberately moves. Don't hand-roll `launch_server`.

```bash
# 0. Sanity
nvidia-smi --query-gpu=index,name,memory.free --format=csv
ls -la /models/dsv32-fp8-channel-mask.safetensors        # reuse Loop-5's mask

# 0a. DEC-1 strategic gate — write runs/<date>_dsv32_loop6/ds_on_v32_decision.md FIRST.

# 1. TokenLabelTable footprint (handoff #2) — edit
#    python/sglang/srt/layers/attention/double_sparsity/token_label_table.py
#    (+ token_label_write.py quantize-on-write, selection_kernel.py apply-scales),
#    then prove selection equivalence + reduced bytes:
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q

# 2. mem-fraction lift + no-OOM sweep (handoff #2)
for MF in 0.6 0.7 0.8; do
  MEM_FRACTION_STATIC=$MF bash development/serve_double_sparsity.sh \
    2>&1 | tee development/logs/ds_memfrac_${MF}_$(date +%Y%m%d-%H%M%S).log
  # check /get_server_info -> max_total_num_tokens; fire a long /generate to flush OOM
done

# 3. 64K servability probe at the lifted mem fraction (handoff #2)
curl -s -X POST http://127.0.0.1:30000/generate -H 'Content-Type: application/json' \
  -d @development/loop6/probe_64k.json | python -c "import sys,json;print(json.load(sys.stdin).get('meta_info',{}))"

# 4. ⭐ DIRECT CLIENT-SLO VALIDATION — full workload, absolute P99 TTFT < 22s, >= 30 TPS/req
NUM_PROMPTS=320 MEM_FRACTION_STATIC=<lifted> \
MODE=double_sparsity CONCURRENCIES="16 32 64" bash development/benchmark.sh
#   -> write runs/<date>_dsv32_loop6/client_slo_report.md with the ABSOLUTE numbers vs the SLO.

# 5. AC-11 directional re-sweep at the lifted mem fraction (handoff #3, DEC-7)
TRIALS=3 WARMUP_SECONDS=120 MEASUREMENT_WINDOW_S=600 MEM_FRACTION_STATIC=<lifted> \
MODE=native_nsa CONCURRENCIES="16 32 64" bash development/benchmark_baseline.sh
TRIALS=3 WARMUP_SECONDS=120 MEASUREMENT_WINDOW_S=600 MEM_FRACTION_STATIC=<lifted> \
MODE=double_sparsity CONCURRENCIES="16 32 64" bash development/benchmark.sh
python development/benchmark_compare.py --ac11 \
  --baseline development/results/native_nsa_gsp_isl4096_osl512_c64_t3.jsonl \
  --ds       development/results/double_sparsity_gsp_isl4096_osl512_c64_t3.jsonl \
  --output   runs/$(date +%Y%m%d)_dsv32_loop6/ac11_resweep.md

# 6. AC-12 within-budget gate from REAL token counts (handoff #4) — edit the harness, then re-run:
DS_BASE_URL=http://127.0.0.1:30000 DSA_BASE_URL=http://127.0.0.1:30001 \
  pytest test/manual/test_double_sparsity_v32.py -v

# 7. (Tier 2, ONLY if DEC-1 opened it) recall R&D
#    flashmla_kv asserts indices.shape[-1] == dsa_index_topk in
#    python/sglang/srt/layers/attention/dsa_backend.py — that is the hard cap to relax for a
#    top_k > index_topk variant. A learned selector instead reshapes the offline channel mask.
#    Measure NIAH 4K/16K/64K recall delta vs DS 75/5/0.
```

> **Killing servers between bench runs:** `pkill -f sglang_router` does NOT catch the Rust process
> (renamed to `sglang::router`). Use `pkill -f 'sglang::router'` (or match the worker
> `python -m sglang.launch_server` pattern) so the old router doesn't hold the port across the DS↔DSA swap.

---

## Acceptance evidence — what "Loop 6 done" looks like

A directory `runs/<date>_dsv32_loop6/` containing:

- `ds_on_v32_decision.md` — the DEC-1 strategic-gate decision.
- The TokenLabelTable footprint change with a per-rank byte-count measurement + selection-equivalence
  unit-test result.
- A `mem_fraction_static` sweep log showing higher `max_total_num_tokens` and **no generation-time OOM**.
- **`client_slo_report.md`** — the headline: DS absolute `P99 TTFT` and `TPS/req` at the full client
  workload vs the `< 22 s` / `≥ 30 TPS` SLO, with bench JSONLs + `.meta.json` sidecars.
- `ac11_resweep.md` — the refreshed DS-vs-DSA directional verdict at the lifted operating point.
- A 64K servability result (served, or a documented admission ceiling).
- The AC-12 within-budget re-run asserting from real token counts + a diff vs the word-count proxy.
- If DEC-1 opened Tier 2: a NIAH 4K/16K/64K recall-delta artifact vs DS 75% / 5% / 0%.

**Tier-1 done (client-SLO MVP):** "DS now serves the client workload at `P99 TTFT < 22 s` and
`≥ 30 TPS/req` after the TokenLabelTable footprint reduction lifted `mem_fraction_static` and
restored full admission; 64K is servable (or characterized); the AC-12 within-budget gate asserts
from actual token counts; the strategic gate on Tier-2 recall R&D is decided."

**Tier-1 + Tier-2 done (only if DEC-1 opened it):** the above plus "DS recall at the
widened/learned budget moved from the Loop-5 baseline DS 75% / 5% / 0%, and the TPS/TTFT cost is recorded."

---

## Risks + likely failure modes

1. **Footprint shrink regresses selection (int8 quant of signatures).** The compact table could
   change the top-k selection vs fp16 and drop recall/quality. Mitigation: a selection-equivalence
   unit test against the fp16 baseline; keep fp16 the default behind a flag until hardware-validated;
   re-run AC-Q / AC-12-within-budget after the change.
2. **mem-fraction lift still OOMs at the target.** The table size may not be the *only* lever — the
   admission model itself may need adjustment. Mitigation: read the OOM verbatim; if footprint alone
   doesn't reach the target after ~2 rounds, reconsider the admission/KV-budget model (see Step 8
   stagnation signals in `development/loop6/runbook.md`).
3. **P99 TTFT still > 22 s at full nominal concurrency even after admission is fixed.** If at conc 64
   the bottleneck shifts from admission-queue to prefill compute (4096 ISL × 64), TTFT may still miss.
   Mitigation: AC-L6-3 must break down admission-wait vs prefill-compute; if prefill-bound, that's a
   different (chunked-prefill / scheduling) follow-up, surfaced honestly rather than hidden.
4. **DS-vs-DSA TPS parity at conc 16/32 still misses** even with full admission. The client SLO is
   **absolute 30 TPS/req (met)**, so this is secondary — record it as a DEC-7 directional follow-up,
   don't let it block the client-SLO MVP claim.
5. **Tier-2 recall R&D gets started before DEC-1.** The expensive trap. Mitigation: AC-L6-0 is a hard
   prerequisite; a closed gate is a legitimate outcome, not a stall.

## Loop-runner notes

- One mainline objective per round, taken from the Tier-1 spine first (gate → footprint → mem lift →
  client-SLO validation → AC-11 re-sweep → 64K → AC-12 token-count); Tier 2 only after the gate opens.
- A code-only round (footprint change + unit tests; AC-12 harness edit) is acceptable **if** the next
  round validates on hardware. Two code-only rounds in a row with no `runs/<date>_dsv32_loop6/`
  artifact is a stall.
- Reuse the Loop-5 mask, serve/bench scripts, comparator, and quality harnesses — don't rebuild them.
- Implementation code/comments must NOT contain plan-process markers (`AC-`, `DEC-`, "Tier",
  "Option B", "Round N") — use behavior-based naming; those markers live in this plan doc only.
- Push to `jimmy` at every round boundary (cluster pre-emptions).
</content>
