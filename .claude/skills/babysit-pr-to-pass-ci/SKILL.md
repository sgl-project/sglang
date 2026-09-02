---
name: babysit-pr-to-pass-ci
description: Start and persistently pursue a goal to babysit an SGLang pull request until selected GitHub Actions workflows pass on the latest PR head. Use when asked to monitor, babysit, retry, or fix PR CI for lint.yml, pr-test.yml, pr-test-extra.yml, AMD, or other named workflows; classify failures as PR-related versus flaky or infrastructural, auto-fix and push only small clean fixes, rerun failed jobs only up to 10 times, and ignore unselected workflows.
---

# Babysit a PR Until Selected CI Passes

Drive only the selected workflows to green on the PR's latest head commit. Keep working across turns by starting a durable goal; do not stop merely because a run was dispatched, a retry began, or a fix was pushed.

## Invocation and arguments

Accept:

```text
$babysit-pr-to-pass-ci [PR_NUMBER|PR_URL] [WORKFLOW ...]
$babysit-pr-to-pass-ci [PR_NUMBER|PR_URL] --only WORKFLOW [WORKFLOW ...]
```

- If the PR is omitted, resolve the PR associated with the current branch.
- If workflows are omitted, select `lint.yml` and `pr-test.yml`.
- Without `--only`, add named workflows to the two defaults.
- With `--only`, monitor exactly the named workflows.
- Accept a workflow basename such as `pr-test-extra.yml`, a repository path such as `.github/workflows/pr-test-extra.yml`, or an unambiguous workflow display name.
- Treat `extra` as `pr-test-extra.yml`.
- Accept natural-language equivalents, such as “PR 12345 plus extra and AMD.” Resolve “AMD” to an exact workflow file from the user's wording or repository context; do not guess when multiple AMD workflows fit.

Examples:

```text
$babysit-pr-to-pass-ci
$babysit-pr-to-pass-ci 12345
$babysit-pr-to-pass-ci https://github.com/sgl-project/sglang/pull/12345 pr-test-extra.yml
$babysit-pr-to-pass-ci 12345 --only pr-test-amd-rocm720.yml
```

## Start or continue the durable goal

Treat explicit invocation of this skill as authorization to create a goal and perform the scoped CI actions below.

1. Resolve the PR and exact workflow set first.
2. Inspect the current goal.
3. If no unfinished goal exists, create one immediately using the product's goal mechanism. Include the resolved PR URL, selected workflow files, and this stopping condition: every selected workflow has a successful run for the PR's latest head SHA.
4. Include these constraints in the goal objective:
   - Ignore every unselected workflow.
   - Diagnose each selected-workflow failure as PR-related, unrelated/infrastructure, or still uncertain.
   - Implement, validate, commit, and push a PR-related fix only when it is clean, non-tricky, and at most 100 changed lines.
   - Stop for user review before editing when the proposed fix is tricky, dirty, or over 100 changed lines.
   - Retry unrelated failures at most 10 times per workflow and head SHA, using failed-jobs-only reruns.
   - Never rerun a full workflow merely to recover failed tests.
5. If the active goal already represents this same PR and workflow set, continue it instead of creating another goal.
6. If a different unfinished goal is active, do not replace it silently. Report the conflict and ask the user to pause or clear it.

Do not mark the goal complete until the completion check at the end of this skill succeeds.

## Preflight

1. Verify GitHub authentication and resolve the repository without changing remote configuration.
2. Resolve the PR and record at least:
   - PR number and URL
   - open/draft state
   - base branch and base SHA
   - head branch and head SHA
   - head repository owner/name, including whether it is a fork
   - labels
   - whether maintainers may modify the branch
3. Require an open PR. If it is draft and a selected workflow is blocked by the draft gate, stop and ask the user to mark it ready; do not change draft state automatically.
4. Normalize every selected workflow to an existing file under `.github/workflows/`. Fail fast on ambiguous or nonexistent names.
5. Read each selected workflow's triggers and gates. Do not assume all optional workflows behave like `pr-test.yml`.
6. Inspect `git status`. Preserve all user changes. If a fix becomes necessary and the current checkout is dirty, on a different branch, or otherwise unsafe, use an isolated worktree for the PR head.

Known SGLang gates:

- `pr-test.yml` requires the `run-ci` label for relevant PR changes. If its existing gate job failed because the label was absent, add `run-ci` when authorized and rerun failed jobs in that existing run; adding the label alone does not necessarily create a new `pr-test.yml` run.
- `pr-test-extra.yml` requires both `run-ci` and `run-ci-extra`. It listens for relevant label events, so adding a missing label may create a fresh run. Prefer that fresh current-head run; rerun failed jobs in an existing current run only when needed.
- AMD-extra workflows may use the same `run-ci` plus `run-ci-extra` gate. Verify the actual selected file.
- Treat draft, cooldown, maintenance, and missing-label failures as gate/infrastructure conditions, not code regressions. Fix only the safe gate condition in scope; wait for cooldown or maintenance windows rather than burning retries immediately.

Adding the known CI opt-in labels is within scope. Do not add unrelated labels, mark the PR ready, alter reviewers, merge, or mutate other PR metadata.

## Track only current-head runs

Use both PR checks and workflow-run data so workflow names, run IDs, attempts, jobs, and logs can be correlated. Prefer workflow file IDs over display names when filtering runs.

For each selected workflow:

1. Select the newest run associated with the current PR head SHA.
2. Ignore runs for older head SHAs, including green runs made stale by a new push.
3. Treat `queued`, `requested`, `waiting`, `pending`, and `in_progress` as active; keep monitoring.
4. Treat a workflow as passing only when its current-head run concludes `success`.
5. If no current-head run exists, wait briefly for event delivery, then inspect the workflow's path filters, event triggers, and label gates.
6. Trigger only through a mechanism explicitly supported by that workflow. Never invent `workflow_dispatch` inputs or dispatch a workflow against the default branch when the intent is to test the PR head.
7. If a selected workflow cannot safely be triggered for the PR head, report the exact blocker and request direction.

After any user push or skill-authored push, re-read the PR head SHA, discard stale run selections, and start tracking the new head. Reset retry counters for the new head SHA.

## Monitor persistently

- Maintain a compact ledger containing workflow file, head SHA, run ID, attempt, status, conclusion, skill-initiated retry count, and latest failure signature.
- Poll at reasonable intervals, normally 30–60 seconds while runs are active. Give concise progress updates during long waits.
- Re-check the PR head SHA on each monitoring cycle and before every rerun or push.
- Ignore failures, cancellations, and pending checks from unselected workflows. Do not diagnose, retry, fix, or wait for them.
- Do not declare success while a selected workflow is absent, stale, pending, skipped, cancelled, neutral, or failed.

## Diagnose a selected-workflow failure

1. Inspect the failed jobs and failed-step logs. In `pr-test.yml`, distinguish the root failure from cascade failures caused by `check-pr-test-health`, wait jobs, or final aggregation.
2. Record the exact failing test/check, error text, affected runner or hardware, and relevant log URL.
3. Compare the failure against the PR's changed files and diff.
4. Classify it:

   **PR-related** when evidence ties the failure to changed code, tests, dependencies, configuration, generated files, or behavior introduced by the PR. A reproducible failure on the PR that does not occur on the base is strong evidence.

   **Unrelated/infrastructure** when evidence points to a transient network or download error, runner loss, GPU/driver instability, service outage, pre-existing main-branch failure, known flaky threshold, unrelated test area, or another environmental cause.

   **Uncertain** when evidence is insufficient. Do not label a failure flaky merely because it might be intermittent. Narrowly reproduce it, compare recent main/scheduled CI, inspect relevant code, or otherwise gather enough evidence to choose a category.

5. When several root failures exist, classify each one. Address PR-related failures before spending retries on unrelated cascades.

## Handle a PR-related failure

Estimate the proposed fix before editing.

A fix is eligible for automatic execution only when all are true:

- The fix is well understood, localized, and clean.
- The fix is not tricky, hacky, speculative, or a broad workaround.
- The fix changes at most 100 lines, counting additions plus deletions in the fix patch, not the pre-existing PR diff.
- The fix does not require a broad API or architecture redesign, risky dependency migration, large generated-file rewrite, or unrelated cleanup.
- A focused validation is available.

If any condition fails, stop before editing. Report the diagnosis, proposed approach, expected files, estimated size, risks, and validation plan, then wait for user review.

For an eligible fix:

1. Work from the latest PR head in a clean checkout or isolated worktree.
2. Make only the necessary change and add or update focused tests when appropriate.
3. Run the smallest meaningful local validation. For lint failures, reproduce the specific hook/check before considering broader lint validation.
4. Review the resulting patch and count changed lines. If it becomes over 100 lines or turns tricky/dirty, do not commit or push it; stop for review and clearly identify the uncommitted skill-authored changes.
5. Commit only skill-authored files. Never use `git add .`, rewrite user commits, amend without request, or include pre-existing local changes.
6. Push a new commit non-forcibly to the exact PR head repository and branch. Never force-push. If permission or fork routing is unclear, stop and report the push command/target that needs authorization.
7. Resolve the new PR head SHA and return to current-head monitoring. A push is progress, not completion.

## Handle an unrelated or infrastructure failure

Retry only the failed jobs in the current selected-workflow run:

```bash
gh run rerun <RUN_ID> --failed --repo <OWNER/REPO>
```

Rules:

- Never use a full-run rerun as a substitute for `--failed`.
- Never rerun a stale run after the PR head SHA changes or a newer current-head run exists.
- Count only successfully dispatched skill-initiated reruns.
- Allow at most 10 such reruns per selected workflow and head SHA.
- Wait for each new attempt to finish, inspect its result, and update the failure signature before deciding on another retry.
- If the failure changes or becomes plausibly PR-related, return to diagnosis instead of blindly consuming all retries.
- If 10 retries are exhausted, stop. Report the attempts, recurring signatures, job/run links, and evidence that the failure is unrelated or still uncertain. Do not exceed the cap without a new explicit user instruction.

If GitHub cannot rerun only the failed portion of the selected run, stop and ask the user rather than rerunning the full workflow.

## Completion check

Before completing the goal:

1. Fetch the PR head SHA again.
2. For every selected workflow, verify that its newest applicable run is for that exact SHA and concludes `success`.
3. Verify no selected workflow is still pending and no success was invalidated by a newer push.
4. Ignore all unselected workflows even if they are failing.
5. Mark the goal complete only after all selected workflows satisfy these checks.

Report the final PR URL, head SHA, selected workflows, passing run links, any commits pushed, and retry counts. Keep the report scoped to the selected workflows.
