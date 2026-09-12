---
name: sglang-cherrypick
description: Trigger the bot-cherry-pick workflow for a batch of merged PRs onto a release branch and monitor each run to completion. Use when an SGLang release manager asks to cherry-pick a list of PRs to a release branch.
---

# SGLang Cherry-Pick

Trigger `.github/workflows/bot-cherry-pick.yml` for each PR in a list, then monitor the resulting workflow runs and report per-PR success/failure with links to the created cherry-pick PRs (or the failure reason).

## Slash Command

`/sglang-cherrypick <target_branch> <pr1> [pr2 pr3 ...]`

Examples:
- `/sglang-cherrypick release/v0.5.7 25956 25958 25987`
- `/sglang-cherrypick release/v0.5.7 25956,25958,25987` (comma-separated also accepted)

## Arguments

- **`target_branch`** (required): Release branch in the form `release/vX.Y` or `release/vX.Y.Z`. Must already exist on `origin` (i.e., `sgl-project/sglang`).
- **`pr_numbers`** (required, one or more): Merged PR numbers to cherry-pick. Each must be a positive integer.

## Repository

Always targets the upstream repo `sgl-project/sglang`. The workflow's job guard (`if: github.repository == 'sgl-project/sglang'`) means triggering it on a fork is a no-op.

## Workflow

### Step 1 — Validate arguments

Fail fast before triggering anything.

```bash
# target branch shape (matches the workflow's own validator)
[[ "$TARGET_BRANCH" =~ ^release/v[0-9]+\.[0-9]+(\.[0-9]+)?$ ]] || die "Invalid target_branch"

# branch exists on upstream
gh api "repos/sgl-project/sglang/branches/$TARGET_BRANCH" --jq '.name' >/dev/null || die "Branch not found"

# each PR is numeric, exists, MERGED, and has a recorded merge commit
declare -A PR_TO_SHA=()
declare -A PR_TO_TITLE=()
for PR in "${PRS[@]}"; do
  [[ "$PR" =~ ^[0-9]+$ ]] || die "PR '$PR' is not a positive integer"
  PR_JSON=$(gh pr view "$PR" --repo sgl-project/sglang --json state,mergeCommit,title) \
    || die "PR #$PR not found"
  STATE=$(jq -r .state <<<"$PR_JSON")
  [[ "$STATE" == "MERGED" ]] || die "PR #$PR is not MERGED (state=$STATE)"
  SHA=$(jq -r '.mergeCommit.oid // empty' <<<"$PR_JSON")
  [[ -n "$SHA" ]] || die "PR #$PR has no merge commit recorded"
  PR_TO_SHA[$PR]="$SHA"
  PR_TO_TITLE[$PR]=$(jq -r .title <<<"$PR_JSON")
done
```

Report any failures and **stop** — do not trigger partial batches.

### Step 2 — Pre-flight: list changed files and detect conflicts locally

Before dispatching any workflow, simulate each cherry-pick locally with `git merge-tree` to (a) show the user which files would change and (b) catch conflicts before paying for a CI run. `git merge-tree --write-tree` is a side-effect-free 3-way merge — it touches neither the working tree nor any ref.

**2a. Locate the upstream remote (`sgl-project/sglang`).** Both the dual-remote (`upstream` + `origin` fork) and single-remote setups need to work.

```bash
UPSTREAM_REMOTE=$(git remote -v \
  | awk '$2 ~ /[:\/]sgl-project\/sglang(\.git)?$/ && $3 == "(fetch)" {print $1; exit}')
[[ -n "$UPSTREAM_REMOTE" ]] || die "No remote points to sgl-project/sglang"
```

**2b. Fetch the target branch and each PR's merge commit.** Fetch the commits by SHA (in case they're not on a ref the user has locally) and the target branch in one call.

```bash
git fetch "$UPSTREAM_REMOTE" "$TARGET_BRANCH" "${PR_TO_SHA[@]}" --quiet \
  || die "Failed to fetch from $UPSTREAM_REMOTE"

TARGET_REF="refs/remotes/$UPSTREAM_REMOTE/$TARGET_BRANCH"
```

**2c. Index existing cherry-pick PRs on the target branch.** One `gh pr list` call gets every cherry-pick PR ever filed against this branch (any state). For each input PR, we cross-reference by the title suffix `(#<PR>)` that the bot workflow always uses.

```bash
# Fetch all cherry-pick PRs against this branch (any state), then bucket by
# source-PR number using the title pattern "(#<source_pr>)".
EXISTING_CP_JSON=$(gh pr list --repo sgl-project/sglang \
  --base "$TARGET_BRANCH" \
  --label cherry-pick \
  --state all \
  --limit 200 \
  --json number,title,url,state)

declare -A PR_TO_EXISTING_CP=()   # source_pr -> JSON array of existing cherry-pick PRs
for PR in "${PRS[@]}"; do
  PR_TO_EXISTING_CP[$PR]=$(jq -c \
    "[.[] | select(.title | contains(\"(#${PR})\"))]" <<<"$EXISTING_CP_JSON")
done
```

For each input PR, classify the existing cherry-picks:

- **`MERGED`** present → the cherry-pick already landed. **Skip** this PR in Step 3.
- **`OPEN`** present (and no `MERGED`) → a previous dispatch is still in flight. **Warn**, ask the user whether to skip or re-dispatch, but default to **skip** (re-dispatching creates a duplicate).
- Only `CLOSED` (no merged, no open) → previous attempts were abandoned; safe to re-dispatch.
- Empty → no prior attempt; proceed normally.

**2d. For each PR, run `git merge-tree` and diff the result.** The semantics of `cherry-pick` are: 3-way-merge with base = parent of source commit, ours = target tip, theirs = source commit. For merge commits the workflow uses `-m 1`, which means base = **first** parent — `${SHA}^` resolves to `${SHA}^1` for both regular and merge commits, so one form covers both.

```bash
declare -A PR_TO_CONFLICTS=()
declare -A PR_TO_FILES=()

for PR in "${PRS[@]}"; do
  SHA="${PR_TO_SHA[$PR]}"

  # --write-tree: print the resulting tree SHA on success
  # Exit 0 = clean merge; exit 1 = conflicts
  if MERGE_OUT=$(git merge-tree --write-tree \
                   --merge-base="${SHA}^" \
                   "$TARGET_REF" "$SHA" 2>&1); then
    RESULT_TREE=$(head -1 <<<"$MERGE_OUT")
    PR_TO_CONFLICTS[$PR]=""
    # Show files that actually differ between target tip and the merged tree.
    # This is more accurate than `git show --name-status $SHA` because it
    # accounts for changes already present on the release branch. As a
    # side-effect, an already-cherry-picked commit shows up here as "0 files".
    PR_TO_FILES[$PR]=$(git diff --name-status "$TARGET_REF" "$RESULT_TREE")
  else
    # Conflict output format (git ≥2.40): first line is the (partial) tree,
    # remaining lines list conflicted paths and informational messages.
    # We just capture and surface it; user decides what to do.
    PR_TO_CONFLICTS[$PR]="$MERGE_OUT"
    PR_TO_FILES[$PR]=$(git show --name-status --format= "$SHA" 2>/dev/null)
  fi
done
```

**2e. Print a pre-flight report.** One table summarizing each PR, followed by per-PR file lists. The "Prior cherry-pick" column uses the classification from 2c.

```markdown
## Cherry-Pick Pre-Flight — `release/vX.Y.Z`

| PR     | Title                    | Merge SHA | Prior cherry-pick    | Conflicts    | # files |
|--------|--------------------------|-----------|----------------------|--------------|---------|
| #25733 | [Bug] Fix V4-Pro NaN ... | 79ea30d1  | ✅ merged as #26063  | clean        | 0       |
| #25562 | [bugfix] Fix wrong ...   | b19052c9  | none                 | **CONFLICT** | —       |
| #25585 | [Bugfix] Fix missing ... | 86c6c77f  | none                 | clean        | 2       |

### Files (PR #25585 — clean)
M  python/sglang/srt/layers/communicator.py
M  python/sglang/srt/models/deepseek_v4.py

### Conflict detail (PR #25562)
<merge-tree output: conflicted paths and reasons>
```

**2f. Gate before dispatching.** Stop and report if any PR is in either of these states:

- `git merge-tree` reports a **conflict** — the workflow would just fail; let the user fix or remove that PR.
- Already has a **MERGED** cherry-pick PR on the target branch — re-dispatching would create a redundant PR. Skip it (or, if the user really wants a re-run, they can pass an explicit override list).
- Has an **OPEN** cherry-pick PR with no merged one — default to skipping with a warning; surface the open PR's URL so the user can review/merge/close it before re-dispatching.

Only PRs that are **clean** AND have **no merged-or-open** prior cherry-pick should proceed to Step 3.

As a sanity check, a clean pre-flight that shows **0 files changed** is the structural signature of "this commit is already on the branch" — if you see it without an existing merged cherry-pick PR being detected (rare, e.g. the original PR was force-merged onto the release branch directly), surface that too and skip the dispatch.

### Step 3 — Dispatch each PR's workflow run

`gh workflow run` (gh ≥2.45) prints the dispatched run's URL on stdout — parse it directly. Fall back to the snapshot/diff polling only if the URL isn't returned (older gh).

```bash
# Snapshot once up front in case we need the fallback path.
mapfile -t SEEN < <(gh run list \
  --workflow=bot-cherry-pick.yml \
  --repo sgl-project/sglang \
  --limit 50 \
  --json databaseId --jq '.[].databaseId')

declare -A PR_TO_RUN=()  # pr_number -> run_id

for PR in "${PRS[@]}"; do
  DISPATCH_OUT=$(gh workflow run bot-cherry-pick.yml \
    --repo sgl-project/sglang \
    -f pr_number="$PR" \
    -f target_branch="$TARGET_BRANCH" 2>&1) || { echo "$DISPATCH_OUT"; die "dispatch failed for PR #$PR"; }

  # Preferred path: gh prints the run URL like
  #   https://github.com/sgl-project/sglang/actions/runs/26275460359
  RUN_URL=$(grep -oE 'https://github.com/[^[:space:]]+/actions/runs/[0-9]+' \
              <<<"$DISPATCH_OUT" | head -1)
  RUN_ID="${RUN_URL##*/}"

  # Fallback for older gh that doesn't print the URL: poll the runs list,
  # filter to workflow_dispatch events we haven't seen yet.
  if [[ -z "$RUN_ID" ]]; then
    for _ in $(seq 1 30); do
      sleep 2
      CANDIDATE=$(gh run list \
        --workflow=bot-cherry-pick.yml \
        --repo sgl-project/sglang \
        --limit 10 \
        --json databaseId,event \
        --jq '[.[] | select(.event=="workflow_dispatch") | .databaseId] | .[0]')
      if [[ -n "$CANDIDATE" ]] \
         && ! printf '%s\n' "${SEEN[@]}" | grep -qx "$CANDIDATE" \
         && ! printf '%s\n' "${PR_TO_RUN[@]}" | grep -qx "$CANDIDATE"; then
        RUN_ID="$CANDIDATE"
        break
      fi
    done
  fi

  if [[ -z "$RUN_ID" ]]; then
    echo "::warning::No new workflow run detected for PR #$PR within 60s"
    PR_TO_RUN[$PR]="UNKNOWN"
  else
    PR_TO_RUN[$PR]="$RUN_ID"
  fi
done
```

**Notes:**
- The workflow has `concurrency: cherry-pick-${{ target_branch }}` with `cancel-in-progress: false`. So multiple dispatches against the same target branch **queue serially**, not in parallel. That's fine — we batch the triggers and the GitHub side serializes execution.
- `gh workflow run` is fire-and-forget; the dispatched run shows up in `gh run list` within a few seconds.

### Step 4 — Monitor each run to completion

Use `gh run watch` per run id, sequentially (since they execute serially anyway).

```bash
for PR in "${PRS[@]}"; do
  RUN_ID="${PR_TO_RUN[$PR]}"
  [[ "$RUN_ID" == "UNKNOWN" ]] && continue

  gh run watch "$RUN_ID" \
    --repo sgl-project/sglang \
    --exit-status \
    --interval 15 \
    >/dev/null 2>&1 || true   # we read conclusion below; don't abort the loop on fail
done
```

`gh run watch` blocks until the run completes. Use `--interval 15` to be polite on rate limits.

### Step 5 — Collect outcomes per PR

For each PR, fetch the run conclusion and (if successful) the URL of the created cherry-pick PR.

```bash
for PR in "${PRS[@]}"; do
  RUN_ID="${PR_TO_RUN[$PR]}"

  if [[ "$RUN_ID" == "UNKNOWN" ]]; then
    echo "PR #$PR: UNKNOWN (no run found)"
    continue
  fi

  CONCLUSION=$(gh run view "$RUN_ID" --repo sgl-project/sglang \
    --json conclusion,status,url \
    --jq '"\(.status) \(.conclusion) \(.url)"')

  STATUS=$(awk '{print $1}' <<<"$CONCLUSION")
  RESULT=$(awk '{print $2}' <<<"$CONCLUSION")
  RUN_URL=$(awk '{print $3}' <<<"$CONCLUSION")

  if [[ "$RESULT" == "success" ]]; then
    # Find the cherry-pick PR created by this run. Title format from the workflow:
    #   "[Cherry-pick to <BRANCH>] <ORIG TITLE> (#<PR>)"
    CP_PR=$(gh pr list --repo sgl-project/sglang \
      --base "$TARGET_BRANCH" \
      --label cherry-pick \
      --state all \
      --limit 30 \
      --json number,title,url,createdAt \
      --jq "[.[] | select(.title | contains(\"(#${PR})\"))][0]")

    CP_URL=$(jq -r '.url // "N/A"' <<<"$CP_PR")
    CP_NUM=$(jq -r '.number // "?"' <<<"$CP_PR")
    echo "PR #$PR  -> SUCCESS  cherry-pick PR #$CP_NUM ($CP_URL)  [run: $RUN_URL]"
  else
    # Failure: pull the cherry-pick step's last error line so the user sees why.
    REASON=$(gh run view "$RUN_ID" --repo sgl-project/sglang --log-failed 2>/dev/null \
      | grep -m1 -E "::error::" \
      | sed -E 's/^[^:]*::error::?//' \
      || echo "(see run logs)")
    echo "PR #$PR  -> $RESULT  reason: $REASON  [run: $RUN_URL]"
  fi
done
```

### Step 6 — Final summary

Print one table sorted by input order:

```markdown
## Cherry-Pick Batch Summary — `release/v0.5.7`

| PR  | Status   | Cherry-pick PR | Run | Notes |
|-----|----------|----------------|-----|-------|
| #25956 | success | #26031 | run/12345 | — |
| #25958 | failure | —      | run/12346 | Cherry-pick of <SHA> onto release/v0.5.7 failed due to conflicts |
| #25987 | success | #26032 | run/12347 | — |

**Totals:** N succeeded, M failed, K unknown.
```

For any **failure**, suggest the manual fallback from the workflow's own error message:

> Resolve locally: `git checkout release/v0.5.7 && git cherry-pick <SHA>`, fix conflicts, push a branch, and open the PR by hand.

## Common Failure Modes

| Symptom | Cause | Action |
|---------|-------|--------|
| `PR #X is not merged (state=OPEN)` | PR not yet merged | Wait for merge or pass `--commit-sha` (not supported by this slash command) |
| `Target branch '...' does not exist` | Typo or branch not cut yet | Confirm branch name; release manager may not have cut it |
| `Cherry-pick of <SHA> onto <BRANCH> failed due to conflicts` | Code drift on release branch (should already have been caught in Step 2 pre-flight) | Do it manually as instructed above |
| Pre-flight `git merge-tree` reports a conflict | Same as above, caught locally before any CI run | Remove that PR from the batch and resolve manually |
| `No remote points to sgl-project/sglang` | Skill invoked from a checkout that only has a fork remote | Add the upstream remote: `git remote add upstream https://github.com/sgl-project/sglang.git` |
| Pre-flight `git merge-tree` errors with `unknown option` | git < 2.38 | Upgrade git, or run the skill on a machine with a modern git |
| Pre-flight reports `Prior cherry-pick: merged as #N` | The PR has already been cherry-picked and merged onto this release branch | Skip this PR — re-dispatching would create a duplicate PR. Verify #N is the right one before removing from the list. |
| Pre-flight reports `Prior cherry-pick: OPEN as #N` | A previous dispatch is still in flight (PR not yet merged or closed) | Default: skip and ask the user to land or close #N first. Re-dispatching creates a parallel duplicate that needs to be cleaned up afterwards. |
| Pre-flight is clean but `# files = 0` and no prior cherry-pick PR was found | Commit landed on the release branch by direct merge (not via the bot), or via a rebase that rewrote the SHA | Skip the dispatch — the change is already there. Surface this anomaly so the user knows the bot wasn't the source. |
| Multiple runs but only one detected | Two dispatches landed in the same `gh run list` poll cycle | Re-run for the missing PR, or look up its run by hand: `gh run list --workflow=bot-cherry-pick.yml --event workflow_dispatch -L 20` |
| `403` on `gh workflow run` | Missing `actions:write` on the token | Use a token that has workflow dispatch rights on `sgl-project/sglang` |

## Notes

- The skill **never modifies** the workflow file. It only dispatches it.
- The skill operates on the upstream repo only (`sgl-project/sglang`); the user's fork is irrelevant here.
- Per-branch concurrency means picking 20 PRs to the same release branch will take ~20× the runtime of one. There is no parallelism to be gained client-side. If the user batches across **different** target branches, those run concurrently.
- Do not skip the merged-state precheck — the workflow will reject unmerged PRs, but we want a single batched validation report up front rather than N individual workflow failures.
- The skill should be invoked with `gh auth status` already passing; if not, surface the auth error and stop.
