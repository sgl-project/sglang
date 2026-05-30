# Loop 6 — New Session Runbook

Run these in order in a **fresh `claude` session**. The summary at the bottom of the prior context will not be present in the new session; the artifacts on disk (this runbook + `development/loop6/draft.md` + `CLUSTER.md` + the Loop-5 handoff `runs/20260528_dsv32_mvp/next_loop_issues.md`) are the handoff.

**Loop 6 goal in one sentence:** Loop 5 closed the smoke + loop4-compatible MVP and AC-10/AC-11/AC-12 (AC-12 under the DS-fair re-scope); Loop 6 takes the carried-over R&D from `next_loop_issues.md` — **first answer the strategic gate (is DS worth pursuing on a DSA-native model like V3.2?), then bank the engineering wins that pay off regardless (TokenLabelTable footprint → 64K servability + AC-11 admission, AC-12 within-budget gate from real token counts), and only open the recall R&D if the gate says yes.**

Authoritative scope: `development/loop6/draft.md` — **currently a DUMMY stub.** It must be fleshed out from the Loop-5 handoff before `gen-plan` (see Phase 0 Step 2).
Loop-5 handoff (the source list): `runs/20260528_dsv32_mvp/next_loop_issues.md`.
Hardware map: `CLUSTER.md` (2-node 8×H200 layout + node-1 access).

> **Why this loop is different from Loop 5.** Loop 5's failure mode was building CPU scaffolding instead of *running* code; its discipline was "every round drops a hardware artifact." Loop 6 is a **research loop with a decision gate**, so its discipline is different:
> 1. **Answer the strategic question first.** `next_loop_issues.md` #5 asks whether DS is even worthwhile on V3.2 — it is capped at the native `index_topk=2048` budget by the *shared* decode kernel AND uses an inferior offline selector, so it cannot match DSA long-context recall at the shared budget. The recall R&D (item #1) is GPU- and engineering-expensive; **do not spend rounds on it until the gate is decided.** Capture the decision as a `DEC-N`.
> 2. **The engineering wins are the safe mainline.** Items #2 (TokenLabelTable footprint), #3 (AC-11 re-sweep), and #4 (AC-12 within-budget from real token counts) pay off regardless of the strategic answer. They are the spine of the loop.
> 3. **Do NOT re-litigate the AC-12 DS-fair re-scope.** It was DECIDED in Loop 5 (user-authorized, option (b)); see `runs/20260528_dsv32_mvp/ac12_analysis.md`. Loop 6 may *characterize* beyond-budget recall further, but the gate definition is settled.
>
> **What changed since Loop 5's root blocker:** `/models/dsv32-fp8-channel-mask.safetensors` already exists on disk (Loop 5 generated it). Regenerate it **only** if a recipe field changes; otherwise reuse it. DS boots radix-on via the config-bound fixture state file (no env override) — that path is done.

---

## Two deliverable tiers (keep these distinct)

1. **Tier 1 — engineering wins (no recall R&D, pays off regardless of the strategic answer):**
   - **#4 AC-12 within-budget gate from actual token counts** — record `usage.prompt_tokens` and assert `within_budget` from that, not the word-count proxy.
   - **#2 TokenLabelTable footprint** — shrink the per-rank `TokenLabelTable` so DS serves at a higher `mem_fraction_static` without the generation-time OOM seen at 0.7; this unblocks 64K servability and the AC-11 admission gap.
   - **#3 AC-11 directional TTFT re-sweep** — re-run the 3-trial sweep vs DSA *after* the footprint work lifts DS effective concurrency.
2. **Tier 2 — recall R&D (GATED on the strategic decision):**
   - **#1 DS long-context recall** — a query-aware / learned selector that places the needle in the 2048 budget, and/or a `flashmla_kv` decode-kernel variant that accepts `top_k > index_topk` (the kernel today asserts `indices.shape[-1] == dsa_index_topk`, hard-capping DS at 2048).

If Tier 2 is skipped because the strategic gate closed it, that is a **legitimate Loop 6 outcome** — say so explicitly in the round summary; do not treat a closed gate as a stall.

---

## Phase 0 — Pre-session sanity

Run these in the **current session** (i.e. before opening the new `claude` session). All commands assume CWD `/sgl-workspace/sglang`.

```bash
# 1. Verify clean tree on the right branch
git status                       # should show only the loop6 dir (until committed)
git branch --show-current        # should be dev/double-sparsity-standalone
git log --oneline -1             # head should be 989975625 (Loop-5 final) or later

# 2. FLESH OUT THE DRAFT. development/loop6/draft.md is a DUMMY stub.
#    Open it, fill Objective / strategic gate / scope / ACs, seeding from the
#    Loop-5 handoff. Do NOT run gen-plan against the stub.
test -f development/loop6/draft.md
test -f runs/20260528_dsv32_mvp/next_loop_issues.md   # the source list
test -f CLUSTER.md

# 3. Confirm the Loop-5 outputs Loop 6 builds on are present
ls -la /models/dsv32-fp8-channel-mask.safetensors      # should EXIST (Loop 5 made it)
test -f development/serve_double_sparsity.sh
test -f development/serve_native_nsa.sh
test -f development/benchmark.sh
test -f development/benchmark_baseline.sh
test -f development/benchmark_compare.py
test -f python/sglang/srt/layers/attention/double_sparsity/token_label_table.py
test -f test/manual/test_double_sparsity_v32.py
ls runs/20260528_dsv32_mvp/ac11_analysis.md runs/20260528_dsv32_mvp/ac12_analysis.md runs/20260528_dsv32_mvp/evidence_bundle.md

# 4. (Re)create the project humanize config. `.humanize/` is gitignored
#    (.gitignore: `.humanize*`), so this file lives on disk only. The new
#    session must run this same block before invoking any /humanize: command.
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

# 5. Stage and commit the runbook + (real) draft.
#    .humanize/config.json is NOT staged — .gitignore covers it.
git add development/loop6/runbook.md development/loop6/draft.md
git commit -m "[Sparsity] Loop-6: runbook + draft"

# 6. Anchor loop6-base AT the committed head
git rev-parse --verify loop6-base 2>/dev/null || git branch loop6-base HEAD
git rev-parse loop6-base                                   # record this SHA in the round 0 summary

# 7. Push everything to jimmy so rank-1 (and the new session) sees the same state
git push jimmy dev/double-sparsity-standalone
git push jimmy loop6-base
```

> **Cross-node sync (only if you need rank-1 in sync before the cross-node AC-12 / multi-node runs):**
> Node-1 access is documented in `CLUSTER.md` — either `ssh double-sparsity` or `rx devbox exec double-sparsity --no-tmux --rank 1 -- <cmd>`.
> ```bash
> rx devbox exec double-sparsity --no-tmux --rank 1 -- bash -lc 'cd /sgl-workspace/sglang && git fetch jimmy && git checkout dev/double-sparsity-standalone && git reset --hard jimmy/dev/double-sparsity-standalone && git branch -f loop6-base jimmy/loop6-base'
> ```
> Note `git reset --hard` won't copy gitignored files, so rank-1 needs the `.humanize/config.json` block (Step 4) re-run manually if it will invoke any humanize command.

---

## Phase 1 — Open a fresh Claude Code session

Close the current session (or use a new terminal). Then:

```bash
cd /sgl-workspace/sglang
claude                          # fresh session
```

In the new session, the first thing to confirm: `cat development/loop6/runbook.md` exists, `development/loop6/draft.md` is **no longer the dummy stub** (you fleshed it out in Phase 0 Step 2), AND `cat .humanize/config.json` shows the gpt-5.5 / xhigh / sonnet config from Phase 0 Step 4. Also run `nvidia-smi --query-gpu=index,name,memory.free --format=csv` to confirm 8 GPUs are visible and free before anything hardware-bound. If the config is missing (e.g. rank-1 was synced via `git reset --hard`), rerun the `cat > .humanize/config.json <<'EOF' ...` block from Phase 0.

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
/humanize:gen-plan --input development/loop6/draft.md --output development/loop6/plan.md --discussion
```

What happens:
1. Codex first-pass analysis (gpt-5.5:xhigh) emits `CORE_RISKS`, `MISSING_REQUIREMENTS`, `TECHNICAL_GAPS`, `ALTERNATIVE_DIRECTIONS`, `QUESTIONS_FOR_USER`, `CANDIDATE_CRITERIA`.
2. Claude builds candidate plan v1 from the draft + Codex v1.
3. Up to **3 convergence rounds** with a second Codex pass. Stops on no `REQUIRED_CHANGES` and no high-impact `DISAGREE`.
4. In discussion mode, every unresolved `needs_user_decision` becomes an `AskUserQuestion`. The decisions that matter most for Loop 6:
   - **Strategic gate (DEC):** does Loop 6 pursue DS recall R&D on V3.2 (Tier 2) at all, or stop at the engineering wins (Tier 1)? This gates the most expensive ACs — resolve it early.
   - **TokenLabelTable footprint approach:** shrink the per-rank table in place vs. a different admission model; and the target `mem_fraction_static` to validate (e.g. 0.7 / 0.8) and the OOM-safety bar.
   - **AC-11 re-sweep gates:** the same shape as Loop 5 (conc 16/32/64, 3 trials, 120s warmup, 600s window, medians) and whether the directional TTFT target is hard or trend after the footprint lift.
   - **AC-12 token-count assertion:** confirm `usage.prompt_tokens` (or equivalent) is the authority for `within_budget`, replacing the word-count proxy — and that this does not change the DECIDED DS-fair gate definition.
5. Final `plan.md` is written with AC-X format, task tags (`coding`/`analyze`/`hwrun`), and a `## Pending User Decisions` section for any `DEC-N` still PENDING.

> **Keep the plan honest about cost.** Tier 2 (recall R&D) is genuinely large — a learned selector or a new `flashmla_kv` kernel variant is not a one-round task. If gen-plan tries to fold deep kernel work into the mainline before the strategic gate is decided, that is exactly the over-reach to push back on in the discussion answers. Tier 1 is the spine; Tier 2 is conditional.

`--auto-start-rlcr-if-converged` is **intentionally omitted**. We want a hard checkpoint before committing GPU-hours.

### Step 2 (recommended) — Add critique comments before the first refine pass

Use **two voices**: Pensieve (Linus-style architectural critique) + Codex (independent cross-review). Comment markers must use `<comment>...</comment>` (refine-plan understands `<comment>`, `<cmt>`, and `CMT:`/`ENDCMT`).

```
Ask pensieve to review @development/loop6/plan.md and check for any code smell, software architecture issues, and — most importantly — whether the plan stays on its Loop-6 rails. How would Linus Torvalds react? Focus on: (1) whether the strategic gate (is DS worth pursuing on a DSA-native model) is decided BEFORE any recall-R&D AC consumes GPU-hours, (2) whether the TokenLabelTable footprint change is scoped to the minimum that lifts mem_fraction without a generation-time OOM, (3) whether Tier 1 (engineering) and Tier 2 (recall R&D) stay cleanly separated, (4) whether the plan accidentally re-opens the DECIDED AC-12 DS-fair re-scope. Structure your critiques by adding comments to the file with <comment>CRITIQUE</comment>.
```

Then:

```
/humanize:ask-codex Do you agree with these Linus-style comments in @development/loop6/plan.md? Add additional critiques on:
- whether the TokenLabelTable footprint shrink (token_label_table.py) is the right lever for both 64K servability and the AC-11 admission gap, or whether they are actually separable
- whether the recall-R&D split (learned selector vs a flashmla_kv decode-kernel variant accepting top_k > index_topk) is correctly fenced behind the strategic decision and not started speculatively
- whether the AC-12 within-budget assertion from usage.prompt_tokens is a safe, behavior-preserving harness touch that does NOT change the DECIDED DS-fair gate
- whether the AC-11 re-sweep is correctly gated behind a passing footprint lift (no point re-sweeping at the old mem fraction)
Structure each critique as <comment>CRITIQUE</comment>.
```

### Step 3 — First refine pass (`refine-plan`, discussion, **NEW output file**)

```
/humanize:refine-plan \
  --input development/loop6/plan.md \
  --output development/loop6/refined_plan_v1.md \
  --discussion
```

Outputs:
- `development/loop6/refined_plan_v1.md` — comment-free, refined version
- `.humanize/plan_qa/plan-qa.md` — comment ledger (every `CMT-N`: classification, disposition, answer/research/edits)

If `## Pending User Decisions` in `refined_plan_v1.md` is non-empty after this pass, you have a choice (see Step 4).

### Step 4 — Decide whether to do another refine round

**Skip a second pass when:**
- `## Pending User Decisions` is empty in `refined_plan_v1.md` (in particular the strategic gate is resolved)
- The convergence status (last paragraph) is `converged`
- The Linus + Codex critiques produced ≤ ~5 comments total and they were all `answered`/`applied`/`resolved` in `plan-qa.md`

**Do a second pass when:**
- The strategic gate is still PENDING (you cannot scope Tier 2 without it)
- New disagreements surfaced during the first refine that weren't in the original draft
- The first refine's QA shows `deferred` items you now have a position on

For round 2 (optional):

```
# (a) Read refined_plan_v1.md, add a fresh round of <comment> blocks.
# (b) Refine to a v2 file:

/humanize:refine-plan \
  --input development/loop6/refined_plan_v1.md \
  --output development/loop6/refined_plan_v2.md \
  --discussion
```

If you go to v2, the file you hand to `start-rlcr-loop` becomes `refined_plan_v2.md` (etc.). The runbook commits **every version** so each is recoverable.

> **Stop criterion for the refine loop:** strategic gate resolved AND `## Pending User Decisions` empty AND convergence `converged` AND no further `<comment>` blocks added. Don't loop refine-plan for the sake of looping.

Commit each refined version as you go:

```bash
git add development/loop6/plan.md development/loop6/refined_plan_v*.md .humanize/plan_qa/
git commit -m "[Sparsity] Loop-6: plan + refined_plan (v1..vN) + QA ledger"
git push jimmy dev/double-sparsity-standalone
```

### Step 5 — Start the RLCR loop

Final input to `start-rlcr-loop` is whichever `refined_plan_vN.md` survived the refine rounds.

```
/humanize:start-rlcr-loop \
  --plan-file development/loop6/refined_plan_vN.md \
  --codex-model gpt-5.5:xhigh \
  --base-branch loop6-base \
  --yolo
```

Flag rationale:
- **`--codex-model gpt-5.5:xhigh`** — explicit on the CLI even though `.humanize/config.json` already sets it; makes the round-0 summary's `codex_model:` line unambiguous.
- **`--base-branch loop6-base`** — **YES, use this.** Without it, Codex review at end-of-loop diffs against `main` and sees all of Loop 1–5's code + design docs. With `loop6-base`, the review focuses only on what Loop 6 changed. Same reason every prior loop pinned its own `loopN-base`.
- **`--yolo`** — skips the plan-understanding quiz. Justified because you co-authored every AC via discussion-mode gen-plan + refine-plan, and the bitlesson lessons still load per-task (`bitlesson_model: sonnet` runs every iteration regardless of `--yolo`). **Do not also pass `--skip-quiz`** — `--yolo` is the superset.

Flags **intentionally not used**: `--auto-start-rlcr-if-converged`, `--skip-impl`, `--push-every-round` (push manually at round boundaries — but see the auto-memory note below), `--track-plan-file`, `--claude-answer-codex`, `--agent-teams`, `--privacy`.

> **Push between rounds.** Per the standing preference, `git push jimmy` at every round boundary so cluster pre-emptions don't lose a round's work — whether or not `--push-every-round` is set.

### Step 6 — During the loop: how Loop 6 measures progress

Loop 6 is **not** Loop 5's "every round must drop a hardware artifact" loop — some Tier 1 work (the AC-12 token-count assertion, the TokenLabelTable change itself) is code + unit-test first, then validated on hardware. The progress rule is:
1. Advance **one mainline objective per round**, taken from the Tier-1 spine first (gate → footprint → AC-11 re-sweep → AC-12 token-count), Tier-2 only after the gate opens.
2. When a step is **hardware-validated**, drop the artifact under `runs/<date>_dsv32_loop6/`. A code-only round is fine *if* the next round validates it on hardware; two code-only rounds in a row with no hardware validation is a stall (see Step 8).

Per-step anchor (each maps to a `next_loop_issues.md` item):

| Step | Handoff item | Type | Artifact in `runs/<date>_dsv32_loop6/` |
|------|--------------|------|----------------------------------------|
| 0 — strategic gate | #5 | analyze/decision | `ds_on_v32_decision.md` (DEC-N: pursue Tier 2 or not, with the index_topk/selector rationale) |
| 1 — AC-12 token-count assert | #4 | coding | harness records `usage.prompt_tokens`; `within_budget` asserted from it; re-run gate artifact + diff vs the word-count proxy |
| 2 — TokenLabelTable footprint | #2 | coding/hwrun | `token_label_table` change + a `mem_fraction_static` sweep log (e.g. 0.6→0.7→0.8) showing `max_total_num_tokens` rising with **no generation-time OOM** |
| 3 — 64K servability | #2 | hwrun | a 64K-context `/generate` that no longer returns HTTP 400 at the lifted mem fraction (or a documented why-not) |
| 4 — AC-11 re-sweep | #3 (DEC-7) | hwrun/analyze | 3-trial DSA+DS sweep JSONLs at the lifted mem fraction + updated `ac11_analysis.md` verdict |
| 5 — recall R&D (Tier 2, gated) | #1 | coding/hwrun | ONLY if Step 0 opened it: selector or kernel-variant change + a needle-recall delta artifact |

### Step 7 — Done criterion

**Tier 1 done** — `runs/<date>_dsv32_loop6/` contains: the strategic-gate decision doc, the AC-12 within-budget gate re-run asserting from real token counts (with a diff showing the word-count proxy was safe or was wrong), the TokenLabelTable footprint change with a mem-fraction sweep showing higher `max_total_num_tokens` and no generation-time OOM, a 64K servability result, and an AC-11 re-sweep at the lifted mem fraction with an updated verdict. Narrative: "DS now serves at a higher mem fraction without OOM, 64K is servable (or characterized), the AC-11 admission gap is re-measured, and the AC-12 within-budget gate asserts from actual token counts. The strategic gate on Tier-2 recall R&D is decided."

**Tier 1 + Tier 2 done** (only if the strategic gate opened Tier 2) — the Tier-1 bundle **plus**: a query-aware/learned-selector change and/or a `flashmla_kv` decode-kernel variant accepting `top_k > index_topk`, with a needle-recall delta artifact (4K/16K/64K) showing the movement vs the Loop-5 baseline of DS 75%/5%/0%. Narrative adds: "DS recall at the widened/learned budget moved from the Loop-5 baseline; the cost in TPS/TTFT is recorded."

### Step 8 — Stagnation signals (cancel-the-loop checklist)

Loop 6's failure modes are different from Loop 5's. Cancel manually with `/humanize:cancel-rlcr-loop` and re-scope if:
- **The strategic gate (Step 0) is still undecided after round 2 yet Tier-2 recall R&D rounds are being attempted.** Decide the gate before spending GPU-hours on a learned selector or kernel variant.
- **Two consecutive code-only rounds with no hardware validation.** Tier 1's footprint/64K/AC-11 work is hardware-gated; code without a `runs/<date>_dsv32_loop6/` artifact after a second round is a stall.
- **The TokenLabelTable footprint change keeps re-introducing the generation-time OOM at the target mem fraction after 2 rounds.** Read the OOM verbatim and consider whether the admission model (not just the table size) is the real lever.
- **A round re-opens the DECIDED AC-12 DS-fair re-scope** instead of just characterizing beyond-budget recall — that decision is settled; don't burn rounds re-arguing it.
- **Codex review emits `[P0]` markers** about the footprint change corrupting selection/labels, or the token-count assertion changing the gate definition.

To cancel cleanly:

```bash
/humanize:cancel-rlcr-loop
git status                       # check what's local-only
git diff loop6-base HEAD         # what changed since the anchor
```

### Step 9 — Cleanup if you abort

```bash
git checkout dev/double-sparsity-standalone
git reset --hard loop6-base      # discards Loop 6 code, keeps draft+runbook (inside loop6-base)
rm -rf .humanize/rlcr/<loop6-timestamp>
```

If you also want to drop the refined plans (to rerun gen-plan from scratch):

```bash
git rm development/loop6/plan.md development/loop6/refined_plan_*.md
rm -rf .humanize/plan_qa
git commit -m "[Sparsity] Loop-6: reset planning artifacts"
```

> The existing mask `/models/dsv32-fp8-channel-mask.safetensors` and the `runs/<date>_dsv32_loop6/` artifacts are NOT tracked by git, so a `reset --hard` leaves them in place. Delete run artifacts by hand only if you want a truly clean re-run; **keep the mask** unless a recipe field changed.

---

## Phase 3 — Branch-state map after this runbook commits

After **Phase 0** runs:

```
dev/double-sparsity-standalone @ HEAD-after-loop6-commit
  └─ loop6-base (anchor at the same commit)
       └─ contains: loop6/draft.md, loop6/runbook.md, CLUSTER.md, all Loop 1–5 code + run artifacts
       └─ does NOT contain: loop6/plan.md, refined_plan_*.md, any Loop-6 code change, any Loop-6 run artifacts

jimmy/dev/double-sparsity-standalone   (pushed)
jimmy/loop6-base                       (pushed)
```

> **`.humanize/config.json` is not in any of the above** — it's gitignored (`.gitignore`: `.humanize*`). Phase 0 Step 4 re-creates it on disk; rank-1 sync recreates it manually after `git reset --hard`.

After **Phase 2 Step 5** kicks off RLCR:

```
dev/double-sparsity-standalone moves forward with R0, R1, ... commits
loop6-base stays pinned at the runbook+draft commit
codex review at end of loop diffs HEAD vs loop6-base
```

---

## Critical-path cheatsheet (mirrored from `draft.md` once it is fleshed out)

All runs use the **Option B operating point**, which the `serve_*.sh` / `benchmark*.sh` scripts already encapsulate (TP=8, `kv_cache_dtype=fp8_e4m3`, `page_size=64`, `flashmla_kv` prefill+decode backends, overlap-schedule + piecewise-cuda-graph disabled, radix cache enabled via the config-bound fixture state file). Don't hand-roll `launch_server` — use the scripts so knobs stay locked and matched between DS and DSA. The `mem_fraction_static` lever is the one Loop 6 deliberately moves (Step 2 above).

```bash
# 0. Sanity
nvidia-smi --query-gpu=index,name,memory.free --format=csv
ls -la /models/dsv32-fp8-channel-mask.safetensors    # reuse Loop-5's mask

# 1. AC-12 within-budget gate from REAL token counts (handoff #4)
#    Edit test/manual/test_double_sparsity_v32.py so within_budget is asserted
#    from usage.prompt_tokens (not the 1024/1536 word-count proxy), then re-run:
DS_BASE_URL=http://127.0.0.1:30000 DSA_BASE_URL=http://127.0.0.1:30001 \
  pytest test/manual/test_double_sparsity_v32.py -v

# 2. TokenLabelTable footprint (handoff #2) — shrink the per-rank table, then
#    sweep mem_fraction_static and confirm max_total_num_tokens rises with NO
#    generation-time OOM (the failure seen at 0.7 in Loop 5).
#    Footprint code: python/sglang/srt/layers/attention/double_sparsity/token_label_table.py
for MF in 0.6 0.7 0.8; do
  MEM_FRACTION_STATIC=$MF bash development/serve_double_sparsity.sh \
    2>&1 | tee development/logs/ds_memfrac_${MF}_$(date +%Y%m%d-%H%M%S).log
  # check /get_server_info -> max_total_num_tokens; fire a long /generate to flush OOM
done

# 3. 64K servability probe at the lifted mem fraction
curl -s -X POST http://127.0.0.1:30000/generate -H 'Content-Type: application/json' \
  -d @development/loop6/probe_64k.json | python -c "import sys,json;print(json.load(sys.stdin).get('meta_info',{}))"

# 4. AC-11 directional TTFT re-sweep (handoff #3, DEC-7) at the lifted mem fraction
TRIALS=3 WARMUP_SECONDS=120 MEASUREMENT_WINDOW_S=600 \
MODE=native_nsa CONCURRENCIES="16 32 64" bash development/benchmark_baseline.sh
TRIALS=3 WARMUP_SECONDS=120 MEASUREMENT_WINDOW_S=600 \
MODE=double_sparsity CONCURRENCIES="16 32 64" bash development/benchmark.sh
python development/benchmark_compare.py \
  --baseline development/results/native_nsa_gsp_isl4096_osl512_c64_t3.jsonl \
  --ds       development/results/double_sparsity_gsp_isl4096_osl512_c64_t3.jsonl \
  --output   runs/$(date +%Y%m%d)_dsv32_loop6/ac11_resweep.md

# 5. (Tier 2, ONLY if the strategic gate opened it) recall R&D
#    flashmla_kv kernel asserts indices.shape[-1] == dsa_index_topk in
#    python/sglang/srt/layers/attention/dsa_backend.py — that is the hard cap to
#    relax for a top_k > index_topk variant. A learned selector instead reshapes
#    the offline channel mask. Measure NIAH 4K/16K/64K recall delta vs DS 75/5/0.
```

> **Killing servers between bench runs:** `pkill -f sglang_router` does NOT catch the Rust process — it was renamed to `sglang::router`. Use `pkill -f 'sglang::router'` (or match on the worker `python -m sglang.launch_server` pattern) so the old router doesn't hold the port across the DS↔DSA swap.

---

## Files of interest (quick re-derivation)

- **Draft (authoritative scope — flesh out the DUMMY first):** `development/loop6/draft.md`
- **Loop-5 handoff (the source list for this loop):** `runs/20260528_dsv32_mvp/next_loop_issues.md`
- **Hardware map:** `CLUSTER.md` (node 0 `h200-10-220-51-16` local; node 1 `h200-10-220-51-5` via `ssh double-sparsity` / `rx devbox exec double-sparsity --no-tmux --rank 1`)
- **Loop-5 analyses (baselines Loop 6 moves):** `runs/20260528_dsv32_mvp/ac11_analysis.md`, `ac12_analysis.md`, `evidence_bundle.md`
- **TokenLabelTable footprint (handoff #2):** `python/sglang/srt/layers/attention/double_sparsity/token_label_table.py` (also referenced from `deepseek_v2.py`, `double_sparsity/__init__.py`, `double_sparsity/selector.py`)
- **flashmla_kv `index_topk` hard cap (handoff #1, Tier 2):** the `indices.shape[-1] == dsa_index_topk` assert in `python/sglang/srt/layers/attention/dsa_backend.py`
- **AC-12 within-budget harness (handoff #4):** `test/manual/test_double_sparsity_v32.py` (the `within_budget` / `INDEX_TOPK` / `prompt_tokens` logic)
- **Mask loader / recipe:** `python/sglang/srt/layers/attention/double_sparsity/channel_mask.py::load_channel_mask`, `python/sglang/srt/layers/attention/double_sparsity/calibrate.py`
- **Serve scripts:** `development/serve_double_sparsity.sh`, `development/serve_native_nsa.sh`
- **Bench harness:** `development/benchmark.sh`, `development/benchmark_baseline.sh`, `development/benchmark_compare.py`
- **Quality tests:** `test/manual/test_dsv32_quality_smoke.py` (smoke, 20 prompts), `test/manual/test_double_sparsity_v32.py` (full AC-12 gate)
- **Model weights:** `/cluster-storage/models/deepseek-ai/DeepSeek-V3.2`
- **Mask (already on disk, NOT committed):** `/models/dsv32-fp8-channel-mask.safetensors`
- **Acceptance evidence dir (new for this loop):** `runs/<date>_dsv32_loop6/` (e.g. `runs/20260530_dsv32_loop6/`)
