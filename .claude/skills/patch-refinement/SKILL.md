---
name: patch-refinement
description: Interactively draft, refine, or review English Git commit messages for PPU patches and completed large-feature patch sets, regardless of whether the caller communicates in Chinese or English. Use when preparing a commit, amending a commit message, reviewing patch documentation, or checking that a commit records AI-generated, caller-reviewed Why and How modules; caller-provided dependencies or an explicit None; assertions; prerequisite commits; library-version changes; unit-test filenames detected from the commit or supplied by the caller; and caller-provided end-to-end test cases.
---

# Patch Refinement

Interactively produce an English commit message that lets engineers and AI automation understand why a patch exists, what it assumes, and how it should be validated. Ground every statement in the patch, repository history, commands, caller input, and test output; never invent rationale, dependencies, cases, or results.

## Interaction Contract

- Treat skill use as an interactive drafting and review process, not a one-shot transformation.
- Accept caller answers and feedback in Chinese or English. Match the caller's language in conversational prompts when useful, but always write the commit message itself in English.
- Translate caller-provided rationale corrections, dependencies, model descriptions, platform descriptions, and test cases into clear English without changing their meaning.
- Preserve literal commit IDs, model identifiers, code symbols, commands, arguments, environment-variable names and values, paths, dtypes, platform names, and numerical acceptance criteria.
- Collect missing information through focused follow-up prompts. Do not finalize until the caller has reviewed `Why` and `How`, supplied `Dependency` or explicitly confirmed `None`, the Unit module records UT filenames found in the commit or caller-supplied filenames or an explicit `None`, and the caller has supplied the required `Test case` entries.
- Show the updated English draft after incorporating caller feedback so the caller can continue reviewing it.
- After all required interaction is complete, produce one complete final commit-message candidate and explicitly ask the caller to confirm it. Completion of information collection is not itself confirmation.

## Workflow

1. Establish the unit being documented: one PPU patch, or one completed large-feature patch set.
2. Inspect the exact commit content and relevant history. For an existing commit, inspect its full diff and commit boundary; for a commit being prepared, inspect the staged diff and include unstaged changes only when the caller places them in scope. Inspect test output supplied by the user or available in the task context.
3. Generate the `Why` module in English from that commit evidence. Limit it to the problem or limitation; do not include the solution.
4. Generate the `How` module in English from the implementation. Limit it to the solution or implementation approach; do not repeat the problem. Use `- None` when the patch is simple, self-explanatory, and needs no additional implementation summary. Review the two drafts together and remove duplicated context, sentences, and explanations while retaining unavoidable domain identifiers. Present both as model-authored drafts, request caller review, and incorporate corrections before treating either as approved.
5. Check whether the caller supplied `Dependency` or explicitly confirmed that there is no dependency. If neither is available, prompt: `请提供该 commit 的 Dependency；如果没有依赖，请明确回复 None。` Do not infer that silence means no dependency.
6. Inspect the commit's changed-file list for added or modified UT files. If any are present, add their repository-relative filenames directly to `Test` → `Unit`, exactly as shown by Git, without asking the caller to repeat them. Do not add related but unchanged tests. If the commit contains no UT file, use caller-supplied UT filenames; if none were supplied, prompt: `commit 中未发现 UT 文件。请提供该 commit 的 UT 文件名；如果没有 UT，请明确回复 None。`
7. Check whether the caller supplied the required E2E test cases. If not, prompt exactly: `Test case表示该commit需要哪些case进行验证，例如精度测试、需要特定arg、环境变量的性能测试`. Do not invent the cases; request them before finalizing the message.
8. Draft a subject in `type(scope): subject` form. Require a non-empty scope and describe externally meaningful behavior rather than only an implementation detail.
9. Complete the four modules `Why`, `How`, `Dependency`, and `Test`, in that order, using `assets/commit-message-template.txt`.
10. Run `python3 scripts/validate_commit_msg.py <message-file>` and resolve every error.
11. Verify that no unresolved evidence request or placeholder remains. Present the complete English commit message in one fenced text block and ask the caller to confirm whether to use it.
12. If the caller requests changes, revise the message and repeat the final-confirmation step. Treat only an explicit confirmation as approval of the message. Do not create, amend, rebase, or push a commit unless the user separately requests that Git operation.

## Required Content

### Why

Always include the `Why` module and write its content as one or more bullet points. Require a concrete problem statement except for `chore` and `typo` commits.

- Generate `Why` from the actual commit diff, affected call paths, changed behavior, and relevant history. Do not require the caller to supply the initial wording.
- State directly what problem or limitation the commit is intended to solve.
- Use natural problem-focused bullets. Do not split the module into fixed `Problem` and `Impact if omitted` fields.
- Keep implementation choices, algorithms, changed functions, control flow, and solution details out of `Why`; place them in `How`.
- Describe only the problem context needed to understand why the commit exists. Do not restate filenames or code edits.
- For a large-feature patch set, describe the overall problem the feature addresses, not only the issue handled by the last patch in the set.
- Distinguish conclusions supported by the commit from uncertain intent. If critical intent cannot be derived, draft only what the evidence supports and ask a focused follow-up question.
- Submit the generated bullets to the caller for review. Incorporate their corrections without claiming the original inference was authoritative.

For `chore` or `typo`, keep the module and use an `Exemption` bullet to state why behavioral rationale does not apply.

### How

Always include the `How` module immediately after `Why`. Generate its initial English content from the implementation and submit it to the caller for review together with `Why`.

- Summarize only the solution: the key implementation approach, control or data flow, compatibility mechanism, or important design decision.
- Do not restate the problem, motivation, or consequences already captured in `Why`.
- Avoid paraphrasing a `Why` bullet with solution-oriented wording. Repeat only identifiers or domain terms that are necessary to make the solution precise.
- Explain the solution at the level needed to understand the patch without listing every changed file.
- When the patch is sufficiently simple, logically clear, and self-describing, write exactly `- None`.
- Do not append an explanation after `None`, and do not combine `None` with other bullets.
- Do not treat the model-generated `How` or `None` choice as approved until the caller confirms or edits it.

### Dependency

Always include the `Dependency` module and write its content as bullet points. Require the caller to provide its content. Do not generate dependencies from the diff or silently interpret missing input as no dependency.

When the caller explicitly confirms there is no dependency, write:

`- None`

Do not append an explanation, colon, or any other text after `None`.

When the caller provides dependencies, normalize them into applicable bullets without changing their meaning:

- Preconditions and assumptions.
- The assertion or guard that enforces each code-level assumption. If an enforceable assumption has no assertion, flag the patch as incomplete rather than hiding the gap in prose.
- Prerequisite commits, with stable identifiers when known.
- Dependency or library version changes and relevant compatibility constraints.

Inspect caller-provided code assumptions against the patch and identify their assertion or guard. If an enforceable assumption has no protection, flag the patch as incomplete instead of inventing a guard or deleting the dependency. Do not label ordinary files changed by the patch as dependencies.

### Test

Always require `Test`. Populate Unit from the commit when possible and require caller-provided E2E coverage requirements.

- Under `Unit`, list the repository-relative filenames of UT files added or modified by the commit, one file per bullet. Preserve each filename exactly as Git reports it.
- Automatically include every changed UT file found in the commit; do not require the caller to provide those filenames again.
- If no UT file is changed in the commit, list filenames manually supplied by the caller.
- Only when the commit contains no UT file and the caller explicitly confirms that there is no UT, write exactly `- None` under `Unit`.
- Do not include Unit status, commands, coverage descriptions, inferred related tests, or unchanged test files discovered elsewhere in the repository.
- Do not interpret the absence of a changed UT file or caller input as `None`; prompt the caller to provide filenames or explicitly confirm `None`.
- Always provide end-to-end test cases as model-keyed bullet items. Use the exact model name as each item title; create one item per affected or representative model.
- Under every model item, require the three subtitles `Model data type`, `Affected platforms`, and `Test case`.
- Treat `Test case` as the list of cases required to validate the commit, not as an execution-status report. Require the caller to provide it.
- Write each required case as a direct bullet describing the concrete validation scenario and its necessary arguments, environment variables, topology, inputs, or acceptance criteria when applicable.
- Do not require a case-type label such as `Accuracy:` or `Performance:`. Preserve one if the caller explicitly provides it, but do not add one automatically.
- Do not substitute `PASS`, `NOT RUN`, commands already executed, or inferred cases for caller-provided requirements.

## Review Standard

Reject or return a draft for revision when any of these conditions holds:

- The subject lacks a type, a non-empty scope, or a concrete summary.
- A non-`chore`/`typo` patch lacks `Why` or does not identify the problem the commit solves.
- `Why` is split into fixed `Problem` and `Impact if omitted` fields instead of direct problem-focused bullets.
- `Why` contains implementation or solution details that belong in `How`.
- The generated `Why` is not grounded in the exact commit content or is presented as caller-approved before review.
- `How` is missing, not grounded in the implementation, or presented as caller-approved before review.
- `How` repeats the problem or substantially duplicates wording and context from `Why` instead of describing only the solution.
- `How` uses `None` together with other text or bullets instead of exactly `- None`.
- `Dependency` was inferred by the model, or `None` was used without explicit caller confirmation.
- An assumption is documented but an enforceable code invariant has no corresponding assertion or guard.
- Test claims cannot be traced to commands, procedures, or supplied results.
- Unit omits a UT file added or modified by the commit.
- Unit contains inferred related tests, execution metadata, or anything other than commit-changed UT filenames, caller-provided UT filenames, or an explicitly confirmed `None` when the commit contains no UT file.
- An E2E model item is missing, vague, or omits `Model data type`, `Affected platforms`, or caller-provided `Test case` bullets.
- A patch set message describes only the final patch and not the completed feature's overall intent and validation boundary.
- The message does not contain exactly the four ordered modules `Why`, `How`, `Dependency`, and `Test`.
- Any module contains prose that is not formatted as a bullet point.
- A message is labeled final while required information, placeholders, or unresolved review comments remain.
- The interaction ends without presenting the complete final commit message and requesting explicit caller confirmation.

When caller-provided dependency information is unavailable, emit the Dependency prompt and do not fabricate it. When test evidence is unavailable, identify the gap honestly. When required E2E cases are unavailable, emit the required Test case prompt and do not fabricate them.

## Output

During information collection, label incomplete messages as drafts. After the interactive collection and review steps are complete, return one paste-ready English commit message in a fenced text block, with no placeholders or review annotations inside it, and explicitly ask the caller to confirm it. If the caller requests changes, return a revised complete message and ask again. Keep repository-specific identifiers, commands, model names, data types, platforms, and results exact.

Do not confuse message confirmation with authorization to mutate Git state. Confirming the text approves only the commit message unless the caller also asks to create or amend the commit.

Use `assets/commit-message-template.txt` as the canonical structure. The validator enforces the subject shape, all four ordered modules, and bullet-point formatting inside every module.
