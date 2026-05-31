# Loop 7 — New Session Runbook

Run these in order in a **fresh `claude` session**. The summary at the bottom of the prior context will not be present in the new session; the artifacts on disk (this runbook + `development/loop7/draft.md` + the as-built doc `development/past_implementations/study/08-current-system-architecture.md` + `CLUSTER.md`) are the handoff.

**Loop 7 goal in one sentence:** Loop 6 landed the full Tier-1 engineering spine (compact int8 `TokenLabelTable` → `mem_fraction_static`=0.7 → admission → conc-16 P99 TTFT 13.13 s < 22 at full context, AC-5 directional) and **deferred the DS long-context recall R&D to its own loop**; Loop 7 **opens that gated Tier-2 work** — make DS *competitive on recall* (today NIAH 4K/16K/64K = **75% / 5% / 0%** vs DSA 100% at the same 2048 budget) via a query-aware/learned selector and/or a `flashmla_kv` decode-kernel variant that accepts `top_k > index_topk`.

Authoritative scope: `development/loop7/draft.md` — **already fleshed out (not a dummy stub).** Review/tighten it before `gen-plan`; you do not need to write it from scratch.
As-built system state (read FIRST): `development/past_implementations/study/08-current-system-architecture.md`.
Strategic gate (the OPEN gate that authorizes this loop): `runs/20260530_dsv32_loop6/ds_on_v32_decision.md`.
Recall characterization (the baseline this loop moves): `runs/20260528_dsv32_mvp/ac12_analysis.md` (+ `ac12_results/`).
Hardware map: `CLUSTER.md` (2-node 8×H200 layout + node-1 access).

> **Why this loop is different from Loop 6.** Loop 6 was a Tier-1 *engineering* loop with a decision gate; its spine was admission/TTFT and it closed AC-5 **directional** (DEC-3). Loop 7 is a **research loop whose prerequisite is already satisfied** — the strategic gate `ds_on_v32_decision.md` resolved to *pursue* recall R&D **after** the Tier-1 spine landed, and it has. So:
> 1. **The gate is OPEN; the question is now "which lever, how far," not "whether."** Do not re-litigate the gate — build on it.
> 2. **Recall is the mainline; do not regress the Tier-1 spine.** The lifted **DS int8 / mem 0.7 / radix-on / TP=8** operating point and the directional AC-5 result are the baseline. Every Loop-7 change is measured against "did recall move *without* breaking the spine."
> 3. **Cheap probe before heavy build.** The learned/query-aware selector (no kernel change) is the low-cost lever to try first; the adjustable-`top_k` decode kernel is the high-ceiling/high-cost lever, gated behind evidence that 2048 itself (not selector quality) is the wall.
> 4. **A new kernel must be a NEW opt-in path.** The default DSA decode asserts `indices.shape[-1] == dsa_index_topk == 2048`; relaxing it for DS must not weaken that assert on the DSA-default path (the Loop-6 non-regression product property must hold).
>
> **What changed since Loop 6's close:** the compact int8 table, the scale-aware write/score path, the `SIGNATURE_DTYPE` launcher flag, the radix-fixture dtype fail-closed, the R17 score early-exit, and the R23 deterministic tie-break are all **landed** (see the as-built doc §3–§5). The mask `/models/dsv32-fp8-channel-mask.safetensors` already exists; regenerate only if a recipe field changes.

---

## Two deliverable tiers (keep these distinct)

This loop is entirely **Tier 2** in roadmap terms, but it has its own cheap-vs-heavy split — keep them separate so one can land or be cut without entangling the other:

1. **Probe (cheap, no kernel change) — try first:**
   - **Tier-2.B — learned / query-aware DS selector.** Place the needle inside the existing 2048 budget better than the offline channel-mask projection (or pull top-p/Twilight forward). Stays within the locked ABI; flag-gated with the channel-mask as default; held to **NIAH non-regression**, not bitwise equivalence.
2. **Build (heavy, kernel ABI work) — only if the probe shows 2048 itself is the wall:**
   - **Tier-2.A — adjustable-`top_k` sparse decode kernel.** A `flashmla_kv`-style decode kernel that relaxes `indices.shape[-1] == dsa_index_topk` as a **new opt-in DS path** so DS can spend a larger budget (e.g. 4096/8192). CUDA-graph-safe, zero-alloc under replay, R23 deterministic tie-break carried over, default DSA assert untouched.
3. **Secondary engineering scope (separable):**
   - **Tier-2.C — 128k servability.** KV-budget/admission to *serve* 128k (extends Loop-6's 64K servability). Separate from recall; decide in `gen-plan` whether it belongs in this loop or its own.

If the heavy kernel (Tier-2.A) is **not** opened because the selector probe (Tier-2.B) already moves recall, or because the evidence says a wider budget won't help, that is a **legitimate Loop 7 outcome** — say so explicitly in the round summary; do not treat a closed sub-gate as a stall.

---

## Phase 0 — Pre-session sanity

Run these in the **current session** (i.e. before opening the new `claude` session). All commands assume CWD `/sgl-workspace/sglang`.

```bash
# 1. Verify clean tree on the right branch
git status                       # should show only the loop7 dir (until committed)
git branch --show-current        # should be dev/double-sparsity-standalone
git log --oneline -3             # head should include the loop7 draft + this runbook commit

# 2. The draft is REAL (not a dummy). Review/tighten it; do NOT regenerate from scratch.
test -s development/loop7/draft.md                                          # non-empty
test -f development/past_implementations/study/08-current-system-architecture.md  # as-built doc
test -f runs/20260530_dsv32_loop6/ds_on_v32_decision.md                     # the OPEN gate
test -f runs/20260528_dsv32_mvp/ac12_analysis.md                            # recall baseline
test -f CLUSTER.md

# 3. Confirm the Loop-6 outputs Loop 7 builds on are present
ls -la /models/dsv32-fp8-channel-mask.safetensors      # should EXIST (Loop 5 made it; reuse)
test -f development/serve_double_sparsity.sh
test -f development/serve_native_nsa.sh
test -f development/benchmark.sh
test -f python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py   # the selector (Tier-2.B)
test -f python/sglang/srt/layers/attention/double_sparsity/token_label_write.py  # channel-mask projection
test -f python/sglang/srt/layers/attention/dsa_backend.py                        # the index_topk assert (Tier-2.A)
test -f test/manual/test_double_sparsity_v32.py                                  # NIAH gate

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

# 5. Stage and commit the runbook (+ draft if not already committed).
#    .humanize/config.json is NOT staged — .gitignore covers it.
git add development/loop7/runbook.md development/loop7/draft.md
git commit -m "[Sparsity] Loop-7: runbook + draft"

# 6. Anchor loop7-base AT the committed head
git rev-parse --verify loop7-base 2>/dev/null || git branch loop7-base HEAD
git rev-parse loop7-base                                   # record this SHA in the round 0 summary

# 7. Push everything to jimmy so rank-1 (and the new session) sees the same state
git push jimmy dev/double-sparsity-standalone
git push jimmy loop7-base
```

> **Cross-node sync (only if you need rank-1 in sync before a cross-node recall / multi-node run):**
> Node-1 access is documented in `CLUSTER.md` — either `ssh double-sparsity` or `rx devbox exec double-sparsity --no-tmux --rank 1 -- <cmd>`.
> ```bash
> rx devbox exec double-sparsity --no-tmux --rank 1 -- bash -lc 'cd /sgl-workspace/sglang && git fetch jimmy && git checkout dev/double-sparsity-standalone && git reset --hard jimmy/dev/double-sparsity-standalone && git branch -f loop7-base jimmy/loop7-base'
> ```
> Note `git reset --hard` won't copy gitignored files, so rank-1 needs the `.humanize/config.json` block (Step 4) re-run manually if it will invoke any humanize command.

---

## Phase 1 — Open a fresh Claude Code session

Close the current session (or use a new terminal). Then:

```bash
cd /sgl-workspace/sglang
claude                          # fresh session
```

In the new session, the first thing to confirm: `cat development/loop7/runbook.md` exists, `development/loop7/draft.md` is the **real fleshed-out draft**, you have read `development/past_implementations/study/08-current-system-architecture.md` (the as-built state), AND `cat .humanize/config.json` shows the gpt-5.5 / xhigh / sonnet config from Phase 0 Step 4. Also run `nvidia-smi --query-gpu=index,name,memory.free --format=csv` to confirm 8 GPUs are visible and free before anything hardware-bound. If the config is missing (e.g. rank-1 was synced via `git reset --hard`), rerun the `cat > .humanize/config.json <<'EOF' ...` block from Phase 0.

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
/humanize:gen-plan --input development/loop7/draft.md --output development/loop7/plan.md --discussion
```

What happens:
1. Codex first-pass analysis (gpt-5.5:xhigh) emits `CORE_RISKS`, `MISSING_REQUIREMENTS`, `TECHNICAL_GAPS`, `ALTERNATIVE_DIRECTIONS`, `QUESTIONS_FOR_USER`, `CANDIDATE_CRITERIA`.
2. Claude builds candidate plan v1 from the draft + Codex v1.
3. Up to **3 convergence rounds** with a second Codex pass. Stops on no `REQUIRED_CHANGES` and no high-impact `DISAGREE`.
4. In discussion mode, every unresolved `needs_user_decision` becomes an `AskUserQuestion`. The decisions that matter most for Loop 7:
   - **Which lever leads:** the learned/query-aware selector (Tier-2.B, cheap, no kernel) as the first probe vs jumping to the adjustable-`top_k` kernel (Tier-2.A, heavy). Recommend B-as-probe, A-if-needed.
   - **Recall gate hardness:** a strict recall target vs a DEC-3-style *recorded directional uplift* vs the DS baseline 75/5/0 (a recorded+characterized result is the realistic MVP floor; a strict target is the stretch).
   - **Kernel ABI scope (if Tier-2.A opens):** the new variant must be a NEW opt-in path; the default DSA `indices.shape[-1] == dsa_index_topk` assert stays intact. Confirm zero-alloc-under-graph and the R23 tie-break contract carry over.
   - **128k servability (Tier-2.C):** in this loop or its own? It is engineering, not recall R&D.
5. Final `plan.md` is written with AC-X format, task tags (`coding`/`analyze`/`hwrun`), and a `## Pending User Decisions` section for any `DEC-N` still PENDING.

> **Keep the plan honest about cost.** The adjustable-`top_k` decode kernel (Tier-2.A) is genuinely large — a CUDA-graph-safe sparse-attention decode kernel with its own fixed-shape ABI is not a one-round task. If gen-plan tries to fold deep kernel work into the mainline before the cheap selector probe (Tier-2.B) has shown whether the 2048 cap (not selector quality) is the wall, that is exactly the over-reach to push back on in the discussion answers.

`--auto-start-rlcr-if-converged` is **intentionally omitted**. We want a hard checkpoint before committing GPU-hours.

### Step 2 (recommended) — Add critique comments before the first refine pass

Use **two voices**: Pensieve (Linus-style architectural critique) + Codex (independent cross-review). Comment markers must use `<comment>...</comment>` (refine-plan understands `<comment>`, `<cmt>`, and `CMT:`/`ENDCMT`).

> **These two review commands are written as GENERALIZED, reusable templates.** The bracketed `[PLAN FILE]` and the `### Loop-specific focus` block are the only parts you change per loop. The general review lenses (sequencing, minimum lever, clean separation, don't-re-open-decided-scope, cost honesty) apply to *every* loop; the loop-specific block instantiates them for the loop at hand. The instantiation for **Loop 7** is filled in below each template.

#### 2a — Generalized Pensieve review (template)

```
Ask pensieve to review @[PLAN FILE] for code smells, software-architecture issues, and — most importantly — whether the plan stays on its rails for THIS loop's stated objective. How would Linus Torvalds react? Apply these general review lenses (they hold for any loop):
1. SEQUENCING — is every expensive or irreversible step (GPU-hours, a new kernel/ABI, a data-format or schema change) gated behind the decision or cheap proof that justifies it, rather than started speculatively?
2. MINIMUM LEVER — is each change scoped to the smallest reversible edit that achieves the objective, with the existing default path preserved and the new path opt-in / flag-gated?
3. CLEAN SEPARATION — are independent workstreams (cheap probe vs heavy build; engineering vs research; recall vs servability) kept separable so one can land, or be cut, without entangling the other?
4. DON'T RE-OPEN DECIDED SCOPE — does the plan accidentally re-litigate a prior loop's DECIDED decision (gate, re-scope, operating point) instead of building on it?
5. COST HONESTY — does the plan state the true cost of its single biggest item and avoid folding deep work into the mainline before its prerequisite is met?
6. EVIDENCE — is every acceptance claim tied to a durable, recomputable artifact (raw arrays + a fail-closed verifier), not a stored derived number?
Then apply THIS loop's specific focus (below) as extra lenses.
### Loop-specific focus
[3-5 bullets naming the concrete traps for this loop]
Structure your critiques by adding comments to the file with <comment>CRITIQUE</comment>.
```

**Loop-7 instantiation** — `[PLAN FILE]` = `@development/loop7/plan.md`; `### Loop-specific focus`:
- Is the heavy adjustable-`top_k` decode kernel (Tier-2.A) gated behind a learned/query-aware selector probe (Tier-2.B) result, not started first?
- If Tier-2.A is in scope, is it a NEW opt-in DS decode path that leaves the default DSA `indices.shape[-1] == dsa_index_topk` assert untouched (Loop-6 non-regression product property must hold)?
- Does the recall gate avoid demanding a strict pass where a *recorded directional uplift* vs the 75/5/0 baseline is the realistic MVP (DEC-3 precedent), while still being falsifiable?
- Does the plan keep the lifted DS int8 / mem 0.7 / directional-AC-5 spine as an untouched baseline (no Tier-1 regression)?
- Is 128k servability (Tier-2.C) kept separable from recall R&D (own AC, can be cut)?

#### 2b — Generalized ask-codex review (template)

```
/humanize:ask-codex Do you agree with these Linus-style comments in @[PLAN FILE]? Add additional, independent critiques. Apply the same general lenses (sequencing / minimum-lever / clean-separation / don't-re-open-decided-scope / cost-honesty / durable-evidence) and then THIS loop's specific risks:
[3-5 bullets — the same loop-specific focus as 2a, phrased as questions Codex can falsify against the repo]
For each, say whether the plan's current ACs/tasks already cover it; if not, what concrete AC or task is missing. Structure each critique as <comment>CRITIQUE</comment>.
```

**Loop-7 instantiation** — `[PLAN FILE]` = `@development/loop7/plan.md`; loop-specific risks:
- Whether the learned/query-aware selector (Tier-2.B) is correctly the first, cheaper lever and the `flashmla_kv` `top_k > index_topk` kernel variant (Tier-2.A) is fenced behind evidence that the 2048 cap — not selector quality — limits recall.
- Whether the new decode kernel (if opened) is provably CUDA-graph-safe and zero-alloc under replay, carries the R23 deterministic tie-break, and does NOT relax the assert on the DSA-default path.
- Whether the recall acceptance is measured as a NIAH 4K/16K/64K delta vs DS 75/5/0 on real hardware with a durable artifact, not a one-shot anecdote.
- Whether the TPS/TTFT cost of a wider budget is recorded (a larger `top_k` is more decode work — the spine must not silently regress).
- Whether 128k servability is a separable AC with its own pass/fail, not entangled with recall.

### Step 3 — First refine pass (`refine-plan`, discussion, **NEW output file**)

```
/humanize:refine-plan \
  --input development/loop7/plan.md \
  --output development/loop7/refined_plan_v1.md \
  --discussion
```

Outputs:
- `development/loop7/refined_plan_v1.md` — comment-free, refined version
- `.humanize/plan_qa/plan-qa.md` — comment ledger (every `CMT-N`: classification, disposition, answer/research/edits)

If `## Pending User Decisions` in `refined_plan_v1.md` is non-empty after this pass, you have a choice (see Step 4).

### Step 4 — Decide whether to do another refine round

**Skip a second pass when:**
- `## Pending User Decisions` is empty in `refined_plan_v1.md` (in particular the lever order — selector-probe vs kernel — and the recall-gate hardness are resolved)
- The convergence status (last paragraph) is `converged`
- The Linus + Codex critiques produced ≤ ~5 comments total and they were all `answered`/`applied`/`resolved` in `plan-qa.md`

**Do a second pass when:**
- The lever order or the recall-gate hardness is still PENDING (you cannot scope the kernel work without it)
- New disagreements surfaced during the first refine that weren't in the original draft
- The first refine's QA shows `deferred` items you now have a position on

For round 2 (optional):

```
# (a) Read refined_plan_v1.md, add a fresh round of <comment> blocks (reuse the generalized templates in Step 2).
# (b) Refine to a v2 file:

/humanize:refine-plan \
  --input development/loop7/refined_plan_v1.md \
  --output development/loop7/refined_plan_v2.md \
  --discussion
```

If you go to v2, the file you hand to `start-rlcr-loop` becomes `refined_plan_v2.md` (etc.). The runbook commits **every version** so each is recoverable.

> **Stop criterion for the refine loop:** lever order resolved AND recall-gate hardness resolved AND `## Pending User Decisions` empty AND convergence `converged` AND no further `<comment>` blocks added. Don't loop refine-plan for the sake of looping.

Commit each refined version as you go:

```bash
git add development/loop7/plan.md development/loop7/refined_plan_v*.md .humanize/plan_qa/
git commit -m "[Sparsity] Loop-7: plan + refined_plan (v1..vN) + QA ledger"
git push jimmy dev/double-sparsity-standalone
```

### Step 5 — Start the RLCR loop

Final input to `start-rlcr-loop` is whichever `refined_plan_vN.md` survived the refine rounds.

```
/humanize:start-rlcr-loop \
  --plan-file development/loop7/refined_plan_vN.md \
  --codex-model gpt-5.5:xhigh \
  --base-branch loop7-base \
  --yolo
```

Flag rationale:
- **`--codex-model gpt-5.5:xhigh`** — explicit on the CLI even though `.humanize/config.json` already sets it; makes the round-0 summary's `codex_model:` line unambiguous.
- **`--base-branch loop7-base`** — **YES, use this.** Without it, Codex review at end-of-loop diffs against `main` and sees all of Loop 1–6's code + design docs. With `loop7-base`, the review focuses only on what Loop 7 changed. Same reason every prior loop pinned its own `loopN-base`.
- **`--yolo`** — skips the plan-understanding quiz. Justified because you co-authored every AC via discussion-mode gen-plan + refine-plan, and the bitlesson lessons still load per-task (`bitlesson_model: sonnet` runs every iteration regardless of `--yolo`). **Do not also pass `--skip-quiz`** — `--yolo` is the superset.

Flags **intentionally not used**: `--auto-start-rlcr-if-converged`, `--skip-impl`, `--push-every-round` (push manually at round boundaries — but see the note below), `--track-plan-file`, `--claude-answer-codex`, `--agent-teams`, `--privacy`.

> **Push between rounds.** Per the standing preference, `git push jimmy` at every round boundary so cluster pre-emptions don't lose a round's work — whether or not `--push-every-round` is set.

### Step 6 — During the loop: how Loop 7 measures progress

Loop 7 is a **research loop**: some rounds are code + unit-test first (the selector reshape, the kernel skeleton), then validated on hardware by a recall measurement. The progress rule:
1. Advance **one mainline objective per round**, taken cheap-lever-first: baseline recall measurement → selector probe (Tier-2.B) → (only if needed) kernel variant (Tier-2.A) → optional 128k servability (Tier-2.C).
2. When a step is **hardware-validated**, drop the artifact under `runs/<date>_dsv32_loop7/`. A code-only round is fine *if* the next round validates it on hardware; two code-only rounds in a row with no recall/servability artifact is a stall (see Step 8).

Per-step anchor:

| Step | Lever | Type | Artifact in `runs/<date>_dsv32_loop7/` |
|------|-------|------|----------------------------------------|
| 0 — recall baseline | — | hwrun | NIAH 4K/16K/64K DS-vs-DSA at the lifted op-point, reproducing DS 75/5/0 (the number this loop moves) |
| 1 — selector probe | Tier-2.B | coding/hwrun | learned/query-aware selector (flag-gated, channel-mask default) + a NIAH recall delta vs 75/5/0; non-regression on within-budget cases |
| 2 — kernel variant (gated) | Tier-2.A | coding/hwrun | ONLY if Step 1 shows the 2048 cap is the wall: new opt-in `flashmla_kv` decode path accepting `top_k > index_topk`, CUDA-graph-safe/zero-alloc, default DSA assert untouched + a recall delta at the wider budget + the TPS/TTFT cost |
| 3 — 128k servability (optional) | Tier-2.C | hwrun | a 128k-context `/generate` that serves at the lifted op-point (or a documented ceiling) |

### Step 7 — Done criterion

**Recall MVP done** — `runs/<date>_dsv32_loop7/` contains: the recall baseline (DS 75/5/0 reproduced), and at least one lever (Tier-2.B selector and/or Tier-2.A kernel) with a **NIAH 4K/16K/64K recall delta** vs that baseline on real hardware, captured as a durable artifact, with the **Tier-1 spine unregressed** (the lifted DS int8 / mem 0.7 point still serves; the DSA-default product property holds). Narrative: "DS long-context recall moved from the Loop-6 baseline of 75/5/0 via the selector/kernel lever; the TPS/TTFT cost is recorded; the default DSA path is unchanged." Per DEC-3 precedent a recorded+attributed *directional* uplift is accepted MVP progress; a strict recall target is the stretch.

**Recall + 128k done** (only if Tier-2.C was in scope) — the recall bundle **plus** a 128k servability result.

### Step 8 — Stagnation signals (cancel-the-loop checklist)

Cancel manually with `/humanize:cancel-rlcr-loop` and re-scope if:
- **The heavy kernel (Tier-2.A) is being built before the cheap selector probe (Tier-2.B) has shown the 2048 cap — not selector quality — is the limit.** Run the probe first; it may move recall without any kernel work.
- **Two consecutive code-only rounds with no recall/servability artifact** under `runs/<date>_dsv32_loop7/`. Research code without a measured recall delta after a second round is a stall.
- **The recall lever keeps failing to move NIAH at 16K/64K after 2 rounds.** Read whether the bottleneck is selection placement (selector) or budget size (kernel) and switch lever rather than iterating the wrong one.
- **A round regresses the Tier-1 spine** (the lifted DS int8 / mem 0.7 point no longer serves, or the DSA-default product property breaks) — that is a Loop-6 invariant; stop and fix before any more recall work.
- **A round re-opens a DECIDED decision** (the strategic gate, the operating point, the AC-12 DS-fair re-scope) instead of building on it.
- **Codex review emits `[P0]` markers** about the new kernel weakening the DSA-default `dsa_index_topk` assert, breaking CUDA-graph zero-alloc, or the selector corrupting selection/labels.

To cancel cleanly:

```bash
/humanize:cancel-rlcr-loop
git status                       # check what's local-only
git diff loop7-base HEAD         # what changed since the anchor
```

### Step 9 — Cleanup if you abort

```bash
git checkout dev/double-sparsity-standalone
git reset --hard loop7-base      # discards Loop 7 code, keeps draft+runbook (inside loop7-base)
rm -rf .humanize/rlcr/<loop7-timestamp>
```

If you also want to drop the refined plans (to rerun gen-plan from scratch):

```bash
git rm development/loop7/plan.md development/loop7/refined_plan_*.md
rm -rf .humanize/plan_qa
git commit -m "[Sparsity] Loop-7: reset planning artifacts"
```

> The existing mask `/models/dsv32-fp8-channel-mask.safetensors` and the `runs/<date>_dsv32_loop7/` artifacts are NOT tracked by git, so a `reset --hard` leaves them in place. Delete run artifacts by hand only if you want a truly clean re-run; **keep the mask** unless a recipe field changed.

---

## Phase 3 — Branch-state map after this runbook commits

After **Phase 0** runs:

```
dev/double-sparsity-standalone @ HEAD-after-loop7-commit
  └─ loop7-base (anchor at the same commit)
       └─ contains: loop7/draft.md, loop7/runbook.md, the as-built 08 doc, CLUSTER.md, all Loop 1–6 code + run artifacts
       └─ does NOT contain: loop7/plan.md, refined_plan_*.md, any Loop-7 code change, any Loop-7 run artifacts

jimmy/dev/double-sparsity-standalone   (pushed)
jimmy/loop7-base                       (pushed)
```

> **`.humanize/config.json` is not in any of the above** — it's gitignored (`.gitignore`: `.humanize*`). Phase 0 Step 4 re-creates it on disk; rank-1 sync recreates it manually after `git reset --hard`.

After **Phase 2 Step 5** kicks off RLCR:

```
dev/double-sparsity-standalone moves forward with R0, R1, ... commits
loop7-base stays pinned at the runbook+draft commit
codex review at end of loop diffs HEAD vs loop7-base
```

---

## Critical-path cheatsheet (mirrored from `draft.md`)

All runs use the **Option B operating point at the lifted DS int8 / mem 0.7 point**, which the `serve_*.sh` / `benchmark*.sh` scripts already encapsulate (TP=8, `kv_cache_dtype=fp8_e4m3`, `page_size=64`, `flashmla_kv` prefill+decode, overlap-schedule + piecewise-cuda-graph disabled, radix-on via the config-bound fixture, `SIGNATURE_DTYPE=int8`). Don't hand-roll `launch_server` — use the scripts so knobs stay locked and matched between DS and DSA. Loop 7 deliberately moves the **selector / `top_k` budget**, not the mem fraction.

```bash
# 0. Sanity
nvidia-smi --query-gpu=index,name,memory.free --format=csv
ls -la /models/dsv32-fp8-channel-mask.safetensors    # reuse Loop-5/6's mask

# 0b. Recall BASELINE — reproduce DS 75/5/0 vs DSA 100 at the lifted op-point (the number this loop moves)
SIGNATURE_DTYPE=int8 MEM_FRACTION_STATIC=0.7 bash development/serve_double_sparsity.sh &  # DS, port 30000
bash development/serve_native_nsa.sh &                                                     # DSA, port 30001
DS_BASE_URL=http://127.0.0.1:30000 DSA_BASE_URL=http://127.0.0.1:30001 \
  pytest test/manual/test_double_sparsity_v32.py -v -k niah        # NIAH 4K/16K/64K both sides

# 1. Selector probe (Tier-2.B) — learned/query-aware selector, flag-gated, channel-mask default.
#    Selector path: python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py
#                   python/sglang/srt/layers/attention/double_sparsity/token_label_write.py (channel-mask projection)
#    Re-run the NIAH gate above; record the recall delta vs 75/5/0 under runs/<date>_dsv32_loop7/.

# 2. (Tier-2.A, ONLY if the probe shows 2048 is the wall) adjustable-top_k decode kernel.
#    The hard cap to relax (as a NEW opt-in DS path, NOT on the DSA-default path):
#      indices.shape[-1] == self.dsa_index_topk   in
#      python/sglang/srt/layers/attention/dsa_backend.py (_forward_flashmla_kv)
#    Must stay CUDA-graph-safe + zero-alloc under replay; carry the R23 deterministic tie-break.
#    Measure NIAH recall delta at the wider budget AND the per-req TPS/TTFT cost.

# 3. (Tier-2.C, optional) 128k servability probe at the lifted op-point
#    (build a 128k prompt analogous to development/loop6/probe_64k.json):
curl -s -X POST http://127.0.0.1:30000/generate -H 'Content-Type: application/json' \
  -d @runs/<date>_dsv32_loop7/probe_128k.json | python -c "import sys,json;print(json.load(sys.stdin).get('meta_info',{}))"
```

> **Killing servers between bench runs:** `pkill -f sglang_router` does NOT catch the Rust process — it was renamed to `sglang::router`. Use `pkill -f 'sglang::router'` (or match on the worker `python -m sglang.launch_server` pattern) so the old router doesn't hold the port across the DS↔DSA swap.

> **Theory over pragmatism (standing preference):** prefer the theoretically correct adjustable-budget / learned-selector design over a cheap hack even at higher engineering cost; do not justify an inferior selector/kernel choice by engineering cost alone.

---

## Files of interest (quick re-derivation)

- **Draft (authoritative scope — already real):** `development/loop7/draft.md`
- **As-built system state (read FIRST):** `development/past_implementations/study/08-current-system-architecture.md`
- **The OPEN strategic gate (authorizes this loop):** `runs/20260530_dsv32_loop6/ds_on_v32_decision.md`
- **Recall baseline this loop moves:** `runs/20260528_dsv32_mvp/ac12_analysis.md` (+ `ac12_results/`)
- **Loop-6 process record:** `.humanize/rlcr/2026-05-30_06-27-19/` (goal-tracker, round summaries); roadmap `development/roadmap.md` §6 Loop 7
- **Hardware map:** `CLUSTER.md` (node 0 local; node 1 via `ssh double-sparsity` / `rx devbox exec double-sparsity --no-tmux --rank 1`)
- **The selector to improve (Tier-2.B):** `python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py`, `token_label_write.py`
- **flashmla_kv `index_topk` hard cap (Tier-2.A):** the `indices.shape[-1] == dsa_index_topk` assert in `python/sglang/srt/layers/attention/dsa_backend.py`
- **NIAH / recall harness:** `test/manual/test_double_sparsity_v32.py`; smoke `test/manual/test_dsv32_quality_smoke.py`
- **Mask loader / recipe:** `python/sglang/srt/layers/attention/double_sparsity/channel_mask.py::load_channel_mask`, `…/double_sparsity/calibrate.py`
- **Serve scripts:** `development/serve_double_sparsity.sh`, `development/serve_native_nsa.sh`
- **Bench harness:** `development/benchmark.sh`, `development/benchmark_baseline.sh`, `development/benchmark_compare.py`
- **Model weights:** `/cluster-storage/models/deepseek-ai/DeepSeek-V3.2`
- **Mask (already on disk, NOT committed):** `/models/dsv32-fp8-channel-mask.safetensors`
- **Acceptance evidence dir (new for this loop):** `runs/<date>_dsv32_loop7/` (e.g. `runs/20260601_dsv32_loop7/`)
