# Triggering PPU E2E cases on a PR

PPU PR E2E cases are selected through PR comments, not `test-*` labels.
The selection is saved for the PR and reused on subsequent commits until a new
command replaces or clears it. The `run-ci` label separately authorizes hardware
usage; an E2E command does not add that label or bypass other CI checks.

## Quick start

1. Ensure the PR is open and not a draft. The
   [PPU E2E Commands workflow](../../../.github/workflows/ppu-e2e-commands.yml)
   and its [handler](pr_e2e_commands.py) must already be present on the repository's
   **default branch**. Adding them only to the PR branch does not enable commands.
2. A human maintainer with repository `write`, `maintain`, or `admin` permission
   posts a **new, standalone, single-line comment** in the PR conversation:

   ```text
   /run-ppu-e2e test-qwen
   ```

   Copy only the command, without Markdown fences or explanatory text. Do not use
   an inline review comment, edit an existing comment, or post as a bot.
3. A human maintainer manually adds the `run-ci` label in the GitHub UI if it is
   not already present. For E2E authorization, a bot-applied label is not accepted;
   remove it and have a maintainer apply it manually. If GitHub requires approval
   for a fork PR's Actions run, approve that separately.
4. Check the bot's `PPU E2E:` reply and the **CI Pipeline** run. The
   **PPU E2E Selection** job lists the selected executions. Registration validation
   and the current run's CPU Shortlist must pass before Full CI allocates hardware.

With `test-qwen`, the default E2E coverage is the catalog's single-node Qwen
Answer cases and Accuracy smoke cases. There is no extra E2E selection when no
command has been accepted; ordinary path-selected PR CI remains separate.

## Selectors and coverage

Use one or more model/format selectors after `/run-ppu-e2e`:

| Type | Accepted arguments | Meaning |
| --- | --- | --- |
| Model family | `test-qwen`, `test-deepseek`, `test-glm`, `test-kimi`, `test-minimax` | Select cases belonging to those families. |
| Format | `test-w4a8`, `test-w8a8`, `test-bf16`, `test-fp8`, `test-mxfp4` | Select matching formats across all families. `test-fp8` includes FP8 and FP8 channelwise; `test-mxfp4` maps to MXFP4-FP8. `test-w4a8` is accepted but currently matches no executions (W4A8 left the nightly line), so the request fails closed. |
| All models/formats | `test-all` | Select every family/format, initially with default coverage. It does not enable every coverage modifier. |

**Selectors use union (OR), not intersection (AND).** For example,
`test-qwen test-glm` selects all Qwen cases plus all GLM cases; it does not mean
"Qwen AND GLM". Duplicate executions are removed. Commands cannot select
an arbitrary file path, suite name, catalog entry ID, or shell command.

Without modifiers, coverage is single-node Answer plus Accuracy smoke. Add any
of these modifiers to expand it:

| Modifier | Additional coverage |
| --- | --- |
| `test-accuracy-full` | Full Accuracy evaluation, including its required checkpoint-matched smoke dependency. |
| `test-perf` | Single-node performance cases; multi-node performance also requires `test-multinode`. |
| `test-multinode` | Multi-node Answer cases, plus multi-node performance when combined with `test-perf`. |
| `test-pd` | Two-node prefill/decode-disaggregated performance cases; does not require `test-perf` or `test-multinode`. |

Modifiers require a model/format selector or `test-all`. If a requested modifier
matches no case, or any selected entry is blocked, the selection fails rather
than silently dropping the requested coverage. `test-functional` is recognized
but currently blocked because the required functional nightly suites are absent.

### Comment examples

Each row is a separate, complete PR comment. Each new command **replaces** the
previous selection rather than adding to it.

| Goal | Comment |
| --- | --- |
| Default Qwen coverage | `/run-ppu-e2e test-qwen` |
| BF16 across all families | `/run-ppu-e2e test-bf16` |
| Qwen OR GLM | `/run-ppu-e2e test-qwen test-glm` |
| Qwen default plus full Accuracy | `/run-ppu-e2e test-qwen test-accuracy-full` |
| Qwen default plus single-node performance | `/run-ppu-e2e test-qwen test-perf` |
| Qwen default plus multi-node Answer | `/run-ppu-e2e test-qwen test-multinode` |
| Qwen default plus single-/multi-node performance and multi-node Answer | `/run-ppu-e2e test-qwen test-perf test-multinode` |
| Qwen default plus PD performance | `/run-ppu-e2e test-qwen test-pd` |
| Clear persistent E2E selection | `/clear-ppu-e2e` |

`/clear-ppu-e2e` takes no arguments. It does not remove `run-ci`, cancel active
work, or disable ordinary PR CI.

## Preview exact cases without hardware

From the repository root, use Python 3.12 (the CI version). These commands do not
require PPU hardware, GitHub credentials, or an SGLang installation:

```bash
python3 scripts/ci/ppu/select_pr_e2e.py validate
python3 scripts/ci/ppu/select_pr_e2e.py preview --selectors test-qwen
python3 scripts/ci/ppu/select_pr_e2e.py preview --selectors test-qwen test-glm
python3 scripts/ci/ppu/select_pr_e2e.py preview --selectors test-qwen test-perf test-multinode
```

`preview` prints `selected` execution IDs and `errors`; a nonzero exit status
means the selection is blocked or invalid. This is only a local preview, not a
submission to CI. Inspect [pr_e2e_catalog.json](pr_e2e_catalog.json) for each ID's
suite, test files, config, topology, time budget, and source provenance. Preview
before requesting broad or expensive coverage; do not treat blocked entries as
skipped successes.

## Persistence, reruns, and results

- A new valid command is saved even when `run-ci` is absent. Later PR commits use
  the latest accepted selection while hardware authorization remains separate.
- If current-revision CI is active or queued, the command does not cancel it.
  After it finishes, the handler rechecks the latest request and requests
  **Re-run all jobs** when necessary. A superseded run cannot satisfy the latest
  E2E request.
- If an eligible completed CI run exists for the current PR revision, the handler
  requests a full rerun. If CI already consumed that request, it avoids a duplicate.
  To retry a failed submission or repeat the same selection, post a new command.
- If no eligible current-revision run/selection artifact exists, or the run is too
  old, follow the bot's reply and push a new commit to start fresh PR CI.
- For manual retries, use **Re-run all jobs**, not **Re-run failed jobs**. CPU
  evidence and selection plans must match the run, attempt, revision, and plan hash.
- Do not edit or delete accepted commands or bot receipts. Validation rechecks
  their identity; post a new run/clear command to change selection.

In **CI Pipeline**, inspect **PPU E2E Selection**, the E2E jobs under **Full CI**,
and **PPU E2E Gate**. Selected cases execute in ordered phases: default,
extended, then distributed, with one entry at a time per phase. Later phases
require earlier selected phases to succeed. The gate requires verified completion
of every selected ID; a saved command or successful worker launch alone is not a
passing result.

Run artifacts include:

- `ppu-e2e-plan-<run_id>-<attempt>`: `selection.json`, including selected entries
  and selection/authorization errors.
- `ppu-e2e-result-<run_id>-<attempt>-<entry>`: terminal `entry.json` result.
- `ppu-e2e-evidence-<run_id>-<attempt>-<entry>`: sanitized execution evidence.

If nothing starts, first check the default-branch deployment, command permissions
and syntax, bot reply, draft state, manually applied `run-ci`, fork approval,
selection errors, and any active/queued CI. Changing `test-*` labels or editing an
old comment is not an E2E trigger.
