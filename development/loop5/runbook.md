# Loop 5 — New Session Runbook

Run these in order in a **fresh `claude` session**. The summary at the bottom of the prior context will not be present in the new session; the artifacts on disk (this runbook + `development/loop5/draft.md` + `CLUSTER.md`) are the handoff.

**Loop 5 goal in one sentence:** stop building CPU-only scaffolding and actually *run* the Loop-4 Double Sparsity code on the 2-node H200 cluster — generate the missing channel mask, boot DS-on V3.2 FP8 at the Option B operating point, capture a DS+DSA benchmark pair and a paired quality smoke, then (loop4-compatible tier) flip radix-on and close the AC-11 comparator + AC-12 full quality gate.

Authoritative scope: `development/loop5/draft.md` (290 lines, two deliverable tiers).
Hardware map: `CLUSTER.md` (2-node 8×H200 layout + node-1 access).

> **Why this loop is different from Loop 4.** Loop 4 built deep code-tier scaffolding (comparator gauntlet, bench_serving timing path, M3-B fixtures, AC-12 harness) but never executed against hardware. The remaining ACs are all hardware-gated, and the loop drifted by adding more fixture code instead of running the code that exists. Loop 5's mainline objective every round is **the next concrete command from the critical path**, and every round must drop an **artifact** into `runs/<date>_dsv32_mvp/`. A round that produced only code and no artifact stalled.
>
> **The actual root blocker:** `/models/dsv32-fp8-channel-mask.safetensors` does not exist on disk. Generating it (Step 1 of the critical path) unblocks every DS-on AC.
>
> **The one allowed code patch:** the Round 38 AC-10 producer bug (`_write_token_labels` does not accept `forward_batch` but the capture branch references it and hides the failure). Patch it before any radix-on / default-cookbook parity claim. Everything else in the Loop-4 code stays as-is unless a specific bench failure forces a fix.

---

## Two deliverable tiers (keep these distinct)

1. **Smoke MVP** — DS-on V3.2 FP8 serves real requests on H200, produces non-trivial DS selection, has one DS bench JSON + one DSA bench JSON, and passes the paired 20-prompt quality smoke. Single-trial benches are allowed here.
2. **Loop4-compatible MVP** — the smoke milestone **plus**: AC-10 radix flipped on for the final run, AC-11 3-trial comparator (conc 16/32/64, 120s warmup, 600s window, medians), AC-6 CUDA-graph status recorded, AC-1b chunked-prefill probed, AC-12 full NIAH 4K/16K/64K + MMLU 5-shot.

If AC-10 / AC-11 / AC-12 are missing, the result is a useful **smoke milestone**, not the minimal viable working version requested by Loop 4. Do not conflate the two in any round summary.

---

## Phase 0 — Pre-session sanity

Run these in the **current session** (i.e. before opening the new `claude` session). All commands assume CWD `/sgl-workspace/sglang`.

```bash
# 1. Verify clean tree on the right branch
git status                       # should show only the loop5 runbook (until committed)
git branch --show-current        # should be dev/double-sparsity-standalone
git log --oneline -1             # head should be 1f6ae4cae (or later) — the loop5-draft backup commit

# 2. Sanity-check the loop5 draft + cluster map are present
test -f development/loop5/draft.md
test -f CLUSTER.md

# 3. Confirm the code Loop 5 is going to RUN actually exists on disk
test -f development/serve_double_sparsity.sh
test -f development/serve_native_nsa.sh
test -f development/benchmark.sh
test -f development/benchmark_baseline.sh
test -f development/benchmark_compare.py
test -f python/sglang/srt/layers/attention/double_sparsity/calibrate.py
test -f python/sglang/srt/layers/attention/double_sparsity/channel_mask.py
test -f test/manual/test_dsv32_quality_smoke.py
test -f test/manual/test_double_sparsity_v32.py

# 4. Confirm the root blocker is still missing (expected: file NOT found)
ls -la /models/dsv32-fp8-channel-mask.safetensors || echo "MASK MISSING — this is the Loop 5 root blocker"

# 5. (Re)create the project humanize config. `.humanize/` is gitignored
#    (.gitignore line 282: `.humanize*`), so this file lives on disk only.
#    The new session must run this same block before invoking any /humanize: command.
mkdir -p .humanize
cat > .humanize/config.json <<'EOF'
{
  "codex_model": "gpt-5.5",
  "codex_effort": "xhigh",
  "bitlesson_model": "sonnet",
  "agent_teams": false,
  "alternative_plan_language": "",
  "gen_plan_mode": "discussion"
}
EOF

# 6. Stage and commit the runbook (still untracked).
#    .humanize/config.json is NOT staged — .gitignore covers it.
git add development/loop5/runbook.md
git commit -m "[Sparsity] Loop-5: runbook"

# 7. Anchor loop5-base AT the runbook-committed head
git rev-parse --verify loop5-base 2>/dev/null || git branch loop5-base HEAD
git rev-parse loop5-base                                   # record this SHA in the round 0 summary

# 8. Push everything to jimmy so rank-1 (and the new session) sees the same state
git push jimmy dev/double-sparsity-standalone
git push jimmy loop5-base
```

> **Cross-node sync (only if you need rank-1 in sync before the DSA baseline / multi-node runs):**
> Node-1 access is documented in `CLUSTER.md` — either `ssh double-sparsity` or `rx devbox exec double-sparsity --no-tmux --rank 1 -- <cmd>`.
> ```bash
> rx devbox exec double-sparsity --no-tmux --rank 1 -- bash -lc 'cd /sgl-workspace/sglang && git fetch jimmy && git checkout dev/double-sparsity-standalone && git reset --hard jimmy/dev/double-sparsity-standalone && git branch -f loop5-base jimmy/loop5-base'
> ```
> Note `git reset --hard` won't copy gitignored files, so rank-1 needs the `.humanize/config.json` block (Step 5) re-run manually if it will invoke any humanize command.

---

## Phase 1 — Open a fresh Claude Code session

Close the current session (or use a new terminal). Then:

```bash
cd /sgl-workspace/sglang
claude                          # fresh session
```

In the new session, the first thing to confirm: `cat development/loop5/runbook.md` exists AND `cat .humanize/config.json` shows the gpt-5.5 / xhigh / sonnet config from Phase 0 Step 5. Also run `nvidia-smi --query-gpu=index,name,memory.free --format=csv` to confirm 8 GPUs are visible and free before doing anything hardware-bound. If the config is missing (e.g. rank-1 was synced via `git reset --hard`), rerun the `cat > .humanize/config.json <<'EOF' ...` block from Phase 0.

---

## Phase 2 — Inside the new session: plan → refine → loop

### Humanize config in effect (project `.humanize/config.json`)

```json
{
  "codex_model":  "gpt-5.5",
  "codex_effort": "xhigh",
  "bitlesson_model": "sonnet",
  "agent_teams": false,
  "alternative_plan_language": "",
  "gen_plan_mode": "discussion"
}
```

That config is the source of truth for every command below — `gen-plan`, `refine-plan`, `start-rlcr-loop`, and the per-task `bitlesson-selector` all read from this hierarchy (project config overrides user-global). The runbook still re-passes `--codex-model gpt-5.5:xhigh` and `--discussion` on the CLI for visibility.

### Step 1 — Generate the plan from the draft (`gen-plan`, discussion mode)

```
/humanize:gen-plan --input development/loop5/draft.md --output development/loop5/plan.md --discussion
```

What happens:
1. Codex first-pass analysis (gpt-5.5:xhigh) emits `CORE_RISKS`, `MISSING_REQUIREMENTS`, `TECHNICAL_GAPS`, `ALTERNATIVE_DIRECTIONS`, `QUESTIONS_FOR_USER`, `CANDIDATE_CRITERIA`.
2. Claude builds candidate plan v1 from the draft + Codex v1.
3. Up to **3 convergence rounds** with a second Codex pass. Stops on no `REQUIRED_CHANGES` and no high-impact `DISAGREE`.
4. In discussion mode, every unresolved `needs_user_decision` becomes an `AskUserQuestion`. The decisions that matter most for Loop 5:
   - **Tier gate:** is the round-end target the *smoke MVP* or the *loop4-compatible MVP*? (These have different evidence bars.)
   - **Quantitative gates as hard vs trend:** quality smoke (prefix-match ≥ 0.80, ROUGE-L ≥ 0.85, NIAH-mini 4/5), AC-11 comparator (DS TPS within 5% of DSA, DS P99 TTFT ≤ 1.10× DSA), AC-11 sweep shape (conc 16/32/64, 3 trials, 120s warmup, 600s window).
   - **Calibrate fallback:** is TP=2 (`--tp 2 --gpus 0,1`) an acceptable auto-fallback if TP=1 OOMs, or stop-and-ask?
5. Final `plan.md` is written with AC-X format, task tags (`coding`/`analyze`/`hwrun`), and a `## Pending User Decisions` section for any `DEC-N` still PENDING.

> **Keep the plan thin.** Loop 5 is an *execution* plan, not an architecture plan. The only `coding` task is the Round 38 AC-10 producer-bug fix; everything else is `hwrun`/`analyze` (calibrate, boot, bench, compare, quality). If gen-plan starts proposing new fixture code or refactors, that is exactly the Loop-4 drift this loop exists to avoid — push back in the discussion answers.

`--auto-start-rlcr-if-converged` is **intentionally omitted**. We want a hard checkpoint before the first hardware run.

### Step 2 (recommended) — Add critique comments before the first refine pass

Use **two voices**: Pensieve (Linus-style architectural critique) + Codex (independent cross-review). Comment markers must use `<comment>...</comment>` (refine-plan understands `<comment>`, `<cmt>`, and `CMT:`/`ENDCMT`).

```
Ask pensieve to review @development/loop5/plan.md and check for any code smell, software architecture issues, and — most importantly — whether the plan keeps Loop 5 on its execution rails. How would Linus Torvalds react? Focus on: (1) whether the Round 38 AC-10 producer-bug fix in dsa_backend.py::_write_token_labels is scoped to the minimum change, (2) whether any AC sneaks in new fixture/scaffolding code instead of running existing code, (3) whether the smoke vs loop4-compatible tiers stay cleanly separated, (4) whether "every round produces an artifact in runs/<date>_dsv32_mvp/" is enforceable. Structure your critiques by adding comments to the file with <comment>CRITIQUE</comment>.
```

Then:

```
/humanize:ask-codex Do you agree with these Linus-style comments in @development/loop5/plan.md? Add additional critiques on:
- whether the AC-10 producer fix (add forward_batch param to _write_token_labels + thread it through extend/decode/TRT-LLM call sites) is complete and matches the Round 38 regression requirement
- whether the single-trial smoke benches are clearly fenced off from the AC-11 3-trial comparator evidence
- whether the radix-on flip (remove --disable-radix-cache from the DS launcher) is correctly gated behind a passing producer-capture proof
- whether the calibrate OOM fallback to TP=2 is safe given the mask must validate at shape [L, H, 16], fp8_e4m3, head_dim=128, page_size=64, label_dim=16
- whether the quality smoke gates (prefix-match 0.80 / ROUGE-L 0.85 / NIAH-mini 4/5) are the right bar before spending GPU-hours on the full AC-12 gate
Structure each critique as <comment>CRITIQUE</comment>.
```

Optional but recommended: the one code change (producer-bug fix) sits on a hot path that already "hides the failure," so it's worth stress-testing before round 0 of RLCR.

### Step 3 — First refine pass (`refine-plan`, discussion, **NEW output file**)

```
/humanize:refine-plan \
  --input development/loop5/plan.md \
  --output development/loop5/refined_plan_v1.md \
  --discussion
```

Outputs:
- `development/loop5/refined_plan_v1.md` — comment-free, refined version
- `.humanize/plan_qa/plan-qa.md` — comment ledger (every `CMT-N`: classification, disposition, answer/research/edits)

If `## Pending User Decisions` in `refined_plan_v1.md` is non-empty after this pass, you have a choice (see Step 4).

### Step 4 — Decide whether to do another refine round

**Skip a second pass when:**
- `## Pending User Decisions` is empty in `refined_plan_v1.md`
- The convergence status (last paragraph) is `converged`
- The Linus + Codex critiques produced ≤ ~5 comments total and they were all `answered`/`applied`/`resolved` in `plan-qa.md`

**Do a second pass when:**
- New disagreements surfaced during the first refine that weren't in the original draft
- The first refine's QA shows `deferred` items you now have a position on
- The tier gate (smoke vs loop4-compatible) is still ambiguous in any AC

For round 2 (optional):

```
# (a) Read refined_plan_v1.md, add a fresh round of <comment> blocks.
# (b) Refine to a v2 file:

/humanize:refine-plan \
  --input development/loop5/refined_plan_v1.md \
  --output development/loop5/refined_plan_v2.md \
  --discussion
```

If you go to v2, the file you hand to `start-rlcr-loop` becomes `refined_plan_v2.md` (etc.). The runbook commits **every version** so each is recoverable.

> **Stop criterion for the refine loop:** `## Pending User Decisions` empty AND convergence `converged` AND no further `<comment>` blocks added. Don't loop refine-plan for the sake of looping.

Commit each refined version as you go:

```bash
git add development/loop5/plan.md development/loop5/refined_plan_v*.md .humanize/plan_qa/
git commit -m "[Sparsity] Loop-5: plan + refined_plan (v1..vN) + QA ledger"
git push jimmy dev/double-sparsity-standalone
```

### Step 5 — Start the RLCR loop

Final input to `start-rlcr-loop` is whichever `refined_plan_vN.md` survived the refine rounds.

```
/humanize:start-rlcr-loop \
  --plan-file development/loop5/refined_plan_vN.md \
  --codex-model gpt-5.5:xhigh \
  --base-branch loop5-base \
  --yolo
```

Flag rationale:
- **`--codex-model gpt-5.5:xhigh`** — explicit on the CLI even though `.humanize/config.json` already sets it; makes the round-0 summary's `codex_model:` line unambiguous.
- **`--base-branch loop5-base`** — **YES, use this.** Without it, Codex review at end-of-loop diffs against `main` and sees all of Loop 1–4's code + design docs. With `loop5-base`, the review focuses only on what Loop 5 changed (which should be ~one producer-bug fix plus run artifacts). Same reason Loop 4's runbook used `loop4-base`.
- **`--yolo`** — skips the plan-understanding quiz. Justified because you co-authored every AC via discussion-mode gen-plan + refine-plan, and the bitlesson lessons still load per-task (`bitlesson_model: sonnet` runs every iteration regardless of `--yolo`). **Do not also pass `--skip-quiz`** — `--yolo` is the superset.

Flags **intentionally not used**: `--auto-start-rlcr-if-converged`, `--skip-impl`, `--push-every-round` (push manually at round boundaries), `--track-plan-file`, `--claude-answer-codex`, `--agent-teams`, `--privacy`.

### Step 6 — During the loop: hardware-execution rules (this is the whole point of Loop 5)

**Unit tests are necessary but not sufficient, and CPU fixtures are explicitly NOT progress this loop.** Every round must:
1. Advance the **next concrete command from the critical path** (see cheatsheet below) — one mainline objective per round, no multi-day fixture refactors.
2. Produce an **artifact** under `runs/<date>_dsv32_mvp/`. A round that produced only code changes and no artifact **stalled** — say so in the summary.

Per-step hardware anchor (each maps to a critical-path step):

| Step | AC | Artifact in `runs/<date>_dsv32_mvp/` |
|------|----|--------------------------------------|
| 0a — producer-bug fix | AC-10 (producer) | Round-38 producer regression passes; `SGLANG_DS_RADIX_FIXTURE_CAPTURE=1` `/generate` returns non-empty `meta_info["double_sparsity_radix_capture"]` (per-token `per_token_slot_sha`, `per_layer_written_all_true=True`) |
| 1 — channel mask | AC-4 | `calibrate.log` + `/models/dsv32-fp8-channel-mask.safetensors` (NOT committed) validated: shape `[L,H,16]`, `dtype=fp8_e4m3`, `head_dim=128`, `page_size=64`, `label_dim=16` |
| 2 — DS boot smoke | AC-1 | `serve_double_sparsity.log` (no crash, validator accepted, 8 GPUs) + one `/generate` response + non-empty capture meta_info |
| 3 — DSA baseline bench | AC-9 | `native_nsa_*c{16,32,64}_t1.jsonl` + `.meta.json` sidecars |
| 4 — DS bench | AC-8 | `double_sparsity_*c{16,32,64}_t1.jsonl` + `.meta.json` sidecars |
| 5 — comparator | AC-7/AC-8 | `mvp_compare.md` (TPS, TTFT, no-op detector) |
| 6 — quality smoke | AC-8 | `dsv32_quality_smoke_*.json` (prefix-match / ROUGE-L / NIAH-mini) |
| 7 — radix-on flip | AC-10 | DS launcher no longer passes `--disable-radix-cache`; radix-on boot log + producer-capture proof |
| 8 — final comparator | AC-11 | 3-trial DSA+DS sweep JSONLs + pass/fail summary (DS TPS within 5% of DSA, P99 TTFT ≤ 1.10×) |
| 9 — full quality gate | AC-12 | NIAH 4K/16K/64K + MMLU 5-shot artifacts + pass/fail summary |
| — | AC-6 / AC-1b | CUDA-graph status recorded; chunked-prefill probe result recorded (if it fails, disable on BOTH DS and DSA for apples-to-apples) |

If a round closes a step using only a unit test or a CPU fixture, the next round must add the hardware artifact before moving on. Codex review at round end should flag this; if it doesn't, the `bitlesson-selector` lessons will.

### Step 7 — Done criterion

**Smoke MVP done** — `runs/<date>_dsv32_mvp/` contains: `calibrate.log` + mask validation, `serve_*.log` for both DS and DSA boots, branch+SHA, full `/get_server_info` args, six bench JSONLs (`native_nsa_*` + `double_sparsity_*` at c16/32/64, single trial) with `.meta.json` sidecars, `mvp_compare.md`, and `dsv32_quality_smoke_*.json` passing (prefix-match ≥ 0.80, ROUGE-L ≥ 0.85, NIAH-mini ≥ 4/5). Narrative: "DS-on V3.2 FP8 serves at the locked Option B operating point, side-by-side with DSA at conc 16/32/64, quality smoke passes on 20 paired prompts."

**Loop4-compatible MVP done** — the smoke bundle **plus**: radix-on DS+DSA launch evidence, AC-11 3-trial comparator artifacts with a green summary (DS TPS within 5% of DSA, DS P99 TTFT ≤ 1.10× DSA), AC-6 CUDA-graph status recorded, AC-1b chunked-prefill probe recorded, and AC-12 NIAH 4K/16K/64K + MMLU 5-shot artifacts passing. Narrative adds: "The final run used matching production knobs including radix cache enabled; AC-11 comparator and AC-12 quality gates are complete."

### Step 8 — Stagnation signals (cancel-the-loop checklist)

Loop 5's failure mode is the same drift that killed Loop 4's hardware progress: adding fixture code instead of running code. Cancel manually with `/humanize:cancel-rlcr-loop` and re-scope if:
- **A round produced only code changes and no artifact in `runs/<date>_dsv32_mvp/`.** That is the definition of a stall this loop.
- **Two consecutive rounds add fixture/scaffolding code without a hardware run.** Hard stop — this is the exact Loop-4 drift.
- **The channel mask (Step 1) is not on disk after round 2.** It's the root blocker; if calibrate keeps failing, the mitigation (TP=2 fallback, or read the OOM verbatim) isn't being applied.
- **DS boot smoke (Step 2) fails the validator on something other than the known `--disable-radix-cache` DEC-2 path after 2 rounds.** Read the validator error verbatim; don't burn rounds guessing.
- **Codex review emits `[P0]` markers about the AC-10 producer fix being incomplete** after it was claimed done.

To cancel cleanly:

```bash
/humanize:cancel-rlcr-loop
git status                       # check what's local-only
git diff loop5-base HEAD         # what changed since the anchor
```

### Step 9 — Cleanup if you abort

```bash
git checkout dev/double-sparsity-standalone
git reset --hard loop5-base      # discards Loop 5 code, keeps draft+runbook (inside loop5-base)
rm -rf .humanize/rlcr/<loop5-timestamp>
```

If you also want to drop the refined plans (to rerun gen-plan from scratch):

```bash
git rm development/loop5/plan.md development/loop5/refined_plan_*.md
rm -rf .humanize/plan_qa
git commit -m "[Sparsity] Loop-5: reset planning artifacts"
```

> The generated mask `/models/dsv32-fp8-channel-mask.safetensors` and the `runs/<date>_dsv32_mvp/` artifacts are NOT tracked by git, so a `reset --hard` leaves them in place. Delete them by hand only if you want a truly clean re-calibrate.

---

## Phase 3 — Branch-state map after this runbook commits

After **Phase 0** runs:

```
dev/double-sparsity-standalone @ HEAD-after-runbook
  └─ loop5-base (anchor at the same commit)
       └─ contains: loop5/draft.md, loop5/runbook.md, CLUSTER.md, all Loop-4 code
       └─ does NOT contain: loop5/plan.md, refined_plan_*.md, the channel mask, any run artifacts

jimmy/dev/double-sparsity-standalone   (pushed)
jimmy/loop5-base                       (pushed)
```

> **`.humanize/config.json` is not in any of the above** — it's gitignored (`.gitignore:282`). Phase 0 Step 5 re-creates it on disk; rank-1 sync recreates it manually after `git reset --hard`.

After **Phase 2 Step 5** kicks off RLCR:

```
dev/double-sparsity-standalone moves forward with R0, R1, ... commits
loop5-base stays pinned at the runbook-committed head
codex review at end of loop diffs HEAD vs loop5-base
```

---

## Critical-path cheatsheet (mirrored from `draft.md`)

All runs use the **Option B operating point**, which the `serve_*.sh` / `benchmark*.sh` scripts already encapsulate (TP=8, `kv_cache_dtype=fp8_e4m3`, `page_size=64`, `flashmla_kv` prefill+decode backends, overlap-schedule + piecewise-cuda-graph disabled). Don't hand-roll `launch_server` — use the scripts so the knobs stay locked and matched between DS and DSA.

```bash
# 0. Sanity
nvidia-smi --query-gpu=index,name,memory.free --format=csv

# 0a. Before any radix-on claim: patch the Round 38 AC-10 producer bug
#     (_write_token_labels in python/sglang/srt/layers/attention/dsa_backend.py:1501
#      must accept forward_batch; thread the live forward_batch through the extend,
#      decode, and TRT-LLM call sites; publish radix capture only when forward_batch
#      is present AND mode is extend; add the Round-38 producer regression).
#     Verify:
SGLANG_DS_RADIX_FIXTURE_CAPTURE=1 bash development/serve_double_sparsity.sh &
# (wait for /health on :30000)
curl -s -X POST http://127.0.0.1:30000/generate -H 'Content-Type: application/json' \
  -d '{"text":"Hello from DS","sampling_params":{"temperature":0.0,"max_new_tokens":32}}' \
  | python -c "import sys,json; r=json.load(sys.stdin); print(r['text'][:200]); print('capture:', bool(r.get('meta_info',{}).get('double_sparsity_radix_capture')))"

# 1. Channel mask (~15-30 min, single GPU). Root blocker.
mkdir -p /models
python -m sglang.srt.layers.attention.double_sparsity.calibrate \
    --model /cluster-storage/models/deepseek-ai/DeepSeek-V3.2 \
    --dtype bfloat16 --kv-cache-dtype fp8_e4m3 --tp 1 \
    --output /models/dsv32-fp8-channel-mask.safetensors \
    --label-dim 16 --page-size 64 --num-samples 256 --block-size 512 --seed 42 \
    -v 2>&1 | tee development/logs/calibrate_$(date +%Y%m%d-%H%M%S).log
#   OOM mitigation: --tp 2 --gpus 0,1

# 2. Validate mask artifact
python -c "
from sglang.srt.layers.attention.double_sparsity.channel_mask import load_channel_mask
m = load_channel_mask('/models/dsv32-fp8-channel-mask.safetensors')
print(f'dtype={m.dtype} head_dim={m.head_dim} page_size={m.page_size} label_dim={m.label_dim}')
print(f'channel_selection.shape={tuple(m.channel_selection.shape)}')
print(f'content_sha256[:12]={m.content_sha256[:12]}')
"

# 3. DSA baseline bench (boot serve_native_nsa.sh first, ~10-30 min)
MODE=native_nsa CONCURRENCIES="16 32 64" bash development/benchmark_baseline.sh
# (kill DSA server; boot DS server)

# 4. DS bench
MODE=double_sparsity CONCURRENCIES="16 32 64" bash development/benchmark.sh

# 5. Two-column comparator (single trial, smoke tier)
python development/benchmark_compare.py \
  --baseline development/results/native_nsa_gsp_isl4096_osl512_c64_t1.jsonl \
  --ds       development/results/double_sparsity_gsp_isl4096_osl512_c64_t1.jsonl \
  --output development/results/mvp_compare.md

# 6. Quality smoke (both servers up on different ports)
DS_BASE_URL=http://127.0.0.1:30000 DSA_BASE_URL=http://127.0.0.1:30001 \
  pytest test/manual/test_dsv32_quality_smoke.py -v

# 7. Final loop4-compatible comparator, AFTER AC-10 radix flip
#    (DS launcher no longer passes --disable-radix-cache)
TRIALS=3 WARMUP_SECONDS=120 MEASUREMENT_WINDOW_S=600 \
MODE=native_nsa CONCURRENCIES="16 32 64" bash development/benchmark_baseline.sh
TRIALS=3 WARMUP_SECONDS=120 MEASUREMENT_WINDOW_S=600 \
MODE=double_sparsity CONCURRENCIES="16 32 64" bash development/benchmark.sh

# 8. Full quality gate (AC-12)
DS_BASE_URL=http://127.0.0.1:30000 DSA_BASE_URL=http://127.0.0.1:30001 \
  pytest test/manual/test_double_sparsity_v32.py -v
```

> **Killing servers between bench runs:** `pkill -f sglang_router` does NOT catch the Rust process — it was renamed to `sglang::router`. Use `pkill -f 'sglang::router'` (or match on the worker `python -m sglang.launch_server` pattern) so the old router doesn't hold the port across the DS↔DSA swap.

---

## Files of interest (quick re-derivation)

- **Draft (authoritative scope):** `development/loop5/draft.md`
- **Hardware map:** `CLUSTER.md` (node 0 `h200-10-220-51-16` local; node 1 `h200-10-220-51-5` via `ssh double-sparsity` / `rx devbox exec double-sparsity --no-tmux --rank 1`)
- **Loop-4 plan artifacts (reference for code already built):** `development/loop4/plan.md`, `development/loop4/refined_plan_v1.md`, `development/loop4/RUNBOOK.md`
- **Calibrate entrypoint:** `python/sglang/srt/layers/attention/double_sparsity/calibrate.py`
- **Mask loader:** `python/sglang/srt/layers/attention/double_sparsity/channel_mask.py::load_channel_mask`
- **Round 38 producer-bug site:** `python/sglang/srt/layers/attention/dsa_backend.py::_write_token_labels` (def at ~line 1501; call sites ~1664, ~1863, ~2387) + `python/sglang/srt/models/deepseek_v2.py` (~line 2073)
- **Radix capture probe:** `python/sglang/srt/layers/attention/double_sparsity/radix_fixture_capture.py` (env `SGLANG_DS_RADIX_FIXTURE_CAPTURE=1`)
- **Serve scripts:** `development/serve_double_sparsity.sh`, `development/serve_native_nsa.sh`
- **Bench harness:** `development/benchmark.sh`, `development/benchmark_baseline.sh`, `development/benchmark_compare.py`
- **Quality tests:** `test/manual/test_dsv32_quality_smoke.py` (smoke, 20 prompts), `test/manual/test_double_sparsity_v32.py` (full AC-12 gate)
- **Model weights:** `/cluster-storage/models/deepseek-ai/DeepSeek-V3.2`
- **Mask output (NOT committed):** `/models/dsv32-fp8-channel-mask.safetensors`
- **Acceptance evidence dir:** `runs/<date>_dsv32_mvp/` (e.g. `runs/20260528_dsv32_mvp/`)
