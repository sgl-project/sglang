# Loop 4 — New Session Runbook

Run these in order in a **fresh `claude` session**. The summary at the bottom of the prior context will not be present in the new session; the artifacts on disk (this runbook + `development/loop4/draft.md` + `development/past_implementations/study/07-mvp-proposed-architecture.md`) are the handoff.

**Loop 4 goal in one sentence:** rotate the existing Double Sparsity package to token-level signatures at `page_size=64`, bring up an end-to-end V3.2 FP8 `bench_serving` run with DS-on at the Option B operating point on 8×H200, and (stretch) commit a comparator row vs DSA-on.

Authoritative scope: `development/loop4/draft.md` (233 lines, 14 ACs).
Authoritative design intent: `development/past_implementations/study/07-mvp-proposed-architecture.md` (874 lines).

---

## Phase 0 — Pre-session sanity

Run these in the **current session** (i.e. before opening the new `claude` session). All commands assume CWD `/sgl-workspace/sglang`.

```bash
# 1. Verify clean tree on the right branch
git status                       # should show only the loop4 runbook/CLIENT_SLOS (until committed)
git branch --show-current        # should be dev/double-sparsity-standalone
git log --oneline -1             # current head should be 4dc7957ef (or later) — the §07 rename commit

# 2. Sanity-check the §07 doc + draft are in sync
test -f development/past_implementations/study/07-mvp-proposed-architecture.md
test -f development/loop4/draft.md
grep -c '07-mvp-proposed-architecture.md' development/loop4/draft.md   # expect 5

# 3. (Re)create the project humanize config. `.humanize/` is gitignored on purpose
#    (rlcr/ and plan_qa/ are session artifacts), so this file lives on disk only.
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

# 4. Stage and commit the runbook + CLIENT_SLOS (still untracked).
#    .humanize/config.json is NOT staged — .gitignore line 282 (`.humanize*`) covers it.
git add development/loop4/RUNBOOK.md development/CLIENT_SLOS.md
git commit -m "[Sparsity] Loop-4: runbook + client SLOs"

# 5. Anchor loop4-base AT the runbook-committed head
git rev-parse --verify loop4-base 2>/dev/null || git branch loop4-base HEAD
git rev-parse loop4-base                                   # record this SHA in the round 0 summary

# 6. Push everything to jimmy so rank-1 (and the new session) sees the same state
git push jimmy dev/double-sparsity-standalone
git push jimmy loop4-base
```

> **Cross-node sync (optional, only if you need rank-1 in sync before AC-5 / AC-8):**
> ```bash
> rx devbox run --rank 1 -- bash -lc 'cd /sgl-workspace/sglang && git fetch jimmy && git checkout dev/double-sparsity-standalone && git reset --hard jimmy/dev/double-sparsity-standalone && git branch -f loop4-base jimmy/loop4-base'
> ```

---

## Phase 1 — Open a fresh Claude Code session

Close the current session (or use a new terminal). Then:

```bash
cd /sgl-workspace/sglang
claude                          # fresh session
```

In the new session, the first thing to confirm: `cat development/loop4/RUNBOOK.md` exists AND `cat .humanize/config.json` shows the gpt-5.5 / xhigh / sonnet config from Phase 0 Step 3. If the config is missing (e.g. because rank-1 was synced via `git reset --hard` which won't copy gitignored files), rerun the `cat > .humanize/config.json <<'EOF' ...` block from Phase 0 before proceeding.

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
/humanize:gen-plan --input development/loop4/draft.md --output development/loop4/plan.md --discussion
```

What happens:
1. Codex first-pass analysis (gpt-5.5:xhigh) emits `CORE_RISKS`, `MISSING_REQUIREMENTS`, `TECHNICAL_GAPS`, `ALTERNATIVE_DIRECTIONS`, `QUESTIONS_FOR_USER`, `CANDIDATE_CRITERIA`.
2. Claude builds candidate plan v1 from the draft + Codex v1.
3. Up to **3 convergence rounds** with a second Codex pass (claude ↔ codex). Stops on no `REQUIRED_CHANGES` and no high-impact `DISAGREE`.
4. In discussion mode, every unresolved `needs_user_decision` becomes an `AskUserQuestion` you must answer. Quantitative metrics in the draft (30 TPS, 22s P99 TTFT, 5 pp NIAH-Δ, 1 pp MMLU-Δ, 8192 token chunk, etc.) will each be confirmed as **hard requirement vs trend**.
5. Final `plan.md` is written with AC-X format, task tags (`coding`/`analyze`), and a `## Pending User Decisions` section for any `DEC-N` items still PENDING.

> **Discussion is on purpose** — Loop 3 failed in part because it skipped this gate. Answer the questions even if Claude proposes a sensible default; the answers are what `bitlesson-selector` and the goal tracker key on later.

`--auto-start-rlcr-if-converged` is **intentionally omitted**. We want a hard checkpoint before any code is touched.

### Step 2 (recommended) — Add critique comments before the first refine pass

Use **two voices**: Pensieve (Linus-style architectural critique) + Codex (independent cross-review). Comment markers must use `<comment>...</comment>` (refine-plan understands `<comment>`, `<cmt>`, and `CMT:`/`ENDCMT`).

```
Ask pensieve to review @development/loop4/plan.md and check for any code smell, software architecture issues. How would Linus Torvalds react to this plan? Focus on the AC-0 architecture rotation (page → token), the M1/M2 hooks at nsa_backend.py, the M3 multi-process TP harness, and the AC-1b chunked-prefill probe. Structure your critiques by adding comments to the file with <comment>CRITIQUE</comment>.
```

Then:

```
/humanize:ask-codex Do you agree with these Linus-style comments in @development/loop4/plan.md? Add additional critiques on:
- whether AC-0 understates the rename blast radius across deepseek_v2.py / nsa_backend.py / 150 unit tests
- whether AC-1b probe correctly captures all chunked-prefill chunk-boundary cases
- whether AC-5 multi-process TP test on a single H200 node is sufficient evidence for TP=8 production
- whether the AC-8 lightweight quality smoke (20 prompts, ROUGE-L ≥ 0.85) is too lenient
- whether explicit `--disable-radix-cache` until AC-10 actually neutralizes radix interference in AC-8
Structure each critique as <comment>CRITIQUE</comment>.
```

This is optional but **strongly recommended**: the §07 doc was rotated mid-planning (page→token), so the plan has rotation seams worth stress-testing before round 0 of RLCR.

### Step 3 — First refine pass (`refine-plan`, discussion, **NEW output file**)

```
/humanize:refine-plan \
  --input development/loop4/plan.md \
  --output development/loop4/refined_plan_v1.md \
  --discussion
```

Outputs:
- `development/loop4/refined_plan_v1.md` — comment-free, refined version
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
- You want to inject a new round of Linus/Codex critique against `refined_plan_v1.md` (the rotation seams are real and you might want a second look)

For round 2 (optional):

```
# (a) Read refined_plan_v1.md, add a fresh round of <comment> blocks.
# (b) Refine in-place or to a v2 file:

/humanize:refine-plan \
  --input development/loop4/refined_plan_v1.md \
  --output development/loop4/refined_plan_v2.md \
  --discussion
```

If you go to v2, the file you hand to `start-rlcr-loop` becomes `refined_plan_v2.md` (or `refined_plan_v3.md`, etc.). The runbook commits **every version** so each is recoverable.

> **Stop criterion for the refine loop:** `## Pending User Decisions` is empty AND convergence is `converged` AND no further `<comment>` blocks added. Don't loop refine-plan for the sake of looping — pending decisions block `start-rlcr-loop` from being clean.

Commit each refined version as you go:

```bash
git add development/loop4/plan.md development/loop4/refined_plan_v*.md .humanize/plan_qa/
git commit -m "[Sparsity] Loop-4: plan + refined_plan (v1..vN) + QA ledger"
git push jimmy dev/double-sparsity-standalone
```

### Step 5 — Start the RLCR loop

Final input to `start-rlcr-loop` is whichever `refined_plan_vN.md` survived the refine rounds.

```
/humanize:start-rlcr-loop \
  --plan-file development/loop4/refined_plan_vN.md \
  --codex-model gpt-5.5:xhigh \
  --base-branch loop4-base \
  --yolo
```

Flag rationale:
- **`--codex-model gpt-5.5:xhigh`** — explicit on the CLI even though `.humanize/config.json` already sets it; makes the round-0 summary's `codex_model:` line unambiguous and avoids any inherited environment override.
- **`--base-branch loop4-base`** — **YES, use this.** Without it, Codex review at end-of-loop diffs against `main` and will see all of Loop 1–2's ~3,887 LOC plus the design docs in `development/past_implementations/study/`. With `loop4-base`, the review focuses only on what Loop 4 actually changed. This is the same reason Loop 3's RUNBOOK used `--base-branch loop3-base`.
- **`--yolo`** — skips the plan-understanding quiz. Justified because:
  1. You wrote (or co-authored, via discussion-mode gen-plan and refine-plan) every AC.
  2. Two refine-plan passes already substituted for the quiz's purpose.
  3. The bitlesson lessons (`BL-20260520-*`) are still loaded per-task because `bitlesson_model: sonnet` runs every iteration regardless of `--yolo`.

  **Do not also pass `--skip-quiz`.** `--yolo` is the superset.

Flags **intentionally not used**:
- `--auto-start-rlcr-if-converged` (not on this command — only on `gen-plan`; reiterated here so the runbook's intent is clear: no auto-start anywhere in Loop 4).
- `--skip-impl` (we are doing the implementation).
- `--push-every-round` (push manually at round boundaries; cheaper visibility cost than 14 force-pushes).
- `--track-plan-file`, `--claude-answer-codex`, `--agent-teams`, `--privacy` — none needed for this loop.

### Step 6 — During the loop: hardware verification rules

Carry forward from Loop 3's RUNBOOK: **unit tests are necessary but not sufficient.** Every AC ends with at least one of:
- A real forward pass against the V3.2 FP8 model on H200, OR
- A `bench_serving` run with TPS / TTFT / TPOT numbers committed to the round summary.

If a round closes an AC using only unit tests, the next round must add the hardware step before moving on. Codex review at the end of that round should flag this — and if it doesn't, the `bitlesson-selector` will (`BL-20260520-read-fields-before-abort-mutation` etc.).

The per-AC hardware anchor:

| AC | Hardware evidence |
|----|-------------------|
| AC-0 | 150 renamed unit tests pass; `python -c "from sglang.srt.layers.attention.double_sparsity import TokenLabelTable, retrieve_topk"` |
| AC-1 | Forward-pass hook test reads populated slots at `out_cache_loc` |
| AC-1b | One M3 run at `chunked_prefill_size=4096`; per-token byte-equality assert |
| AC-2 | 2× slot-budget request run; KV pool slot count unchanged after |
| AC-3 | Multi-request batch with known cross-request boundaries; kernel-level inspection |
| AC-4 | Real V3.2 calibration writes `/models/dsv32-fp8-channel-mask.safetensors` (NOT committed); `load_channel_mask` accepts it |
| AC-5 | `torch.multiprocessing.spawn` TP=2 harness; bit-equal `selected_token_indices` |
| AC-6 | Real V3.2 conc=64 capture + 100-step replay at Option B operating point |
| AC-7 | Forward-pass test confirms selector not invoked at prefill below threshold; decode after invokes it |
| AC-8 | `bench_serving` JSON + lightweight quality smoke (20 prompts) committed |
| AC-9..12 | Stretch: real DSA + DS runs, comparator JSON, NIAH/MMLU JSON |
| AC-13 | All 150 unit tests pass after AC-0 rename |

### Step 7 — Done criterion

Phase A: the round summary that closes AC-8 contains a `bench_serving` table like:

```
| config       | conc | TPS  | P50 TTFT | P99 TTFT | P50 ITL | dense_fallback | sparsity_rate |
|--------------|------|------|----------|----------|---------|----------------|---------------|
| DS-on        | 16   | ...  | ...      | ...      | ...     | 0              | ...           |
| DS-on        | 32   | ...  | ...      | ...      | ...     | 0              | ...           |
| DS-on        | 64   | ...  | ...      | ...      | ...     | 0              | ...           |
```

…with DS-on not crashing, `selected_tokens.shape[1] < total_seq_len` on ≥ 90 % of decode steps, `dense_fallback_total == 0`, and the 20-prompt quality smoke passing (prefix-match ≥ 80 %, mean ROUGE-L ≥ 0.85, NIAH-mini ≥ 4/5).

Phase B (stretch, only if Phase A closes by ≤ round 8): comparator emits a green row at conc=64: **DS-on TPS ≥ DSA-on TPS** AND P99 TTFT ≤ DSA × 1.10, with NIAH-Δ ≤ 5 pp and MMLU-Δ ≤ 1 pp.

### Step 8 — Stagnation signals (cancel-the-loop checklist)

Apply the Loop 2 lesson: if 2 consecutive rounds open more gaps than they close, stop manually with `/humanize:cancel-rlcr-loop` and re-scope.

Specific cancel signals for Loop 4:
- **AC-0 not closed by round 2.** The rotation is load-bearing for everything else; if the rename takes longer than 2 rounds, the estimate is wrong and Phase B must be cut.
- **AC-1 hook fires but slots stay default after 2 round-end summaries claim otherwise.** Means the `bind_runtime_data` wiring isn't reaching the new shape; debug, don't burn rounds.
- **Day 3 closes < 2 ACs with > 2 new gaps open.** Hard stop.
- **Codex review at round end emits `[P0]` markers about the architecture rotation itself.** Means the §13 rotation is being relitigated mid-loop; that's a scope-creep death spiral.

To cancel cleanly:

```bash
/humanize:cancel-rlcr-loop
git status                       # check what's local-only
git diff loop4-base HEAD         # what changed since the anchor
```

### Step 9 — Cleanup if you abort

```bash
git checkout dev/double-sparsity-standalone
git reset --hard loop4-base      # discards Loop 4 code, keeps the draft+runbook (they're inside loop4-base)
rm -rf .humanize/rlcr/<loop4-timestamp>
```

If you also want to drop the refined plans (e.g. to rerun gen-plan from scratch):

```bash
git rm development/loop4/plan.md development/loop4/refined_plan_*.md
rm -rf .humanize/plan_qa
git commit -m "[Sparsity] Loop-4: reset planning artifacts"
```

---

## Phase 3 — Branch-state map after this runbook commits

After **Phase 0** runs:

```
dev/double-sparsity-standalone @ HEAD-after-runbook
  └─ loop4-base (anchor at the same commit)
       └─ contains: 07-mvp-proposed-architecture.md, loop4/draft.md, loop4/RUNBOOK.md, .humanize/config.json, CLIENT_SLOS.md
       └─ does NOT contain: plan.md, refined_plan_*.md, any code change

jimmy/dev/double-sparsity-standalone   (pushed)
jimmy/loop4-base                       (pushed)
```

> **`.humanize/config.json` is not in any of the above.** It's gitignored. Phase 0 Step 3 re-creates it on disk; rank-1 sync recreates it manually after `git reset --hard`. Everything else flows from the merged config (project → user → default).

After **Phase 2 Step 5** kicks off RLCR:

```
dev/double-sparsity-standalone moves forward with R0, R1, ... commits
loop4-base stays pinned at the runbook-committed head
codex review at end of loop diffs HEAD vs loop4-base
```

---

## Operating-point cheatsheet (mirrored from `draft.md`)

For all of Phase A + Phase B, both DSA baseline and DS runs use the Option B operating point:

```bash
# DSA baseline (Phase B AC-9):
python -m sglang.launch_server \
  --model deepseek-ai/DeepSeek-V3.2 \
  --tp 8 \
  --kv-cache-dtype fp8_e4m3 \
  --dsa-prefill-backend flashmla_kv \
  --dsa-decode-backend flashmla_kv \
  --disable-overlap-schedule \
  --disable-piecewise-cuda-graph \
  --page-size 64 \
  --trust-remote-code

# DS (Phase A AC-8 / Phase B AC-11):
python -m sglang.launch_server \
  --model deepseek-ai/DeepSeek-V3.2 \
  --tp 8 \
  --kv-cache-dtype fp8_e4m3 \
  --dsa-prefill-backend flashmla_kv \
  --dsa-decode-backend flashmla_kv \
  --disable-overlap-schedule \
  --disable-piecewise-cuda-graph \
  --page-size 64 \
  --trust-remote-code \
  --enable-double-sparsity \
  --double-sparsity-config '{"top_k":2048,"page_size":64,"channel_mask_path":"/models/dsv32-fp8-channel-mask.safetensors","device_buffer_size":4096}'
  # Phase A: also pass --disable-radix-cache (until AC-10 flips it).
  # Phase B (after AC-10): remove --disable-radix-cache; radix cache ON.
```

Chunked prefill (conditional on AC-1b outcome):
- Probe **passes** → keep H200 auto-default (`chunked_prefill_size=8192`) on both sides.
- Probe **fails** → append `--chunked-prefill-size -1` to **both** launch commands. Explicit support becomes Loop 5.

---

## Files of interest (quick re-derivation)

- **Draft:** `development/loop4/draft.md`
- **Design doc:** `development/past_implementations/study/07-mvp-proposed-architecture.md`
- **Client SLOs:** `development/CLIENT_SLOS.md`
- **DS package (subject to AC-0 rename):** `python/sglang/srt/layers/attention/double_sparsity/`
- **NSA hook sites (AC-1):** `python/sglang/srt/layers/attention/nsa_backend.py` (search `set_mla_kv_buffer`)
- **DS attention hook:** `python/sglang/srt/models/deepseek_v2.py::DeepseekV2AttentionMLA._select_topk_indices` (~line 2060)
- **Existing unit tests:** `test/registered/unit/layers/attention/test_double_sparsity_unit.py` (150 tests, shape updates expected)
- **Bench harness:** `development/serve_double_sparsity.sh`, `development/benchmark.sh`, `development/benchmark_compare.py`
