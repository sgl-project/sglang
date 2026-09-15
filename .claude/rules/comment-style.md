---
paths:
  - "**/*.py"
  - "**/*.cu"
  - "**/*.cuh"
  - "**/*.cpp"
  - "**/*.h"
  - "**/*.rs"
---

# Comment Style

Applies to `#` and `//` comments, Python docstrings, and C/C++ Doxygen blocks
(`///`, `/** ... */`) -- Python, CUDA/C++, and Rust alike. Comments are reviewed
like code, with the burden of proof reversed: the author justifies a comment's
existence, not the reviewer its removal.

## Two modes

Writing a comment and editing someone else's are different decisions.

**Writing.** You still have the context that made the comment necessary, so the
judgment is reliable. Everything below `## Cleanup` is written for this case.

**Editing.** The judgment is unreliable and the error is one-way. A one-line
comment carries too little text to tell a restatement apart from the last anchor
for a cross-file fact -- deciding takes the whole function and its callers, more
context than the line costs. And the payoff is asymmetric: keeping a useless
one-liner costs a line of scroll, while deleting a load-bearing one deletes a
fact silently, inside a diff of two hundred deletions where no reviewer will
catch it. So: a bright line, not a judgment call.

> **A comment-only diff does not touch one-line comments.** It removes or
> condenses multi-line prose blocks. A one-liner is rewritten only for a reason
> of its own -- it is wrong, it is stale, or it is commented-out code -- never
> because the line below it says the same thing.

## Cleanup: what a comment-only diff may remove

- **Multi-line prose restating the code** -- the function name, the next line,
  the branch condition, or the loop body, written out as a paragraph.
- **Python `Args:` / `Returns:` blocks** on anything that is not a documented API
  surface -- see `## Documentation blocks` for the list of surfaces that keep
  them.
- **Multi-line history and rationale** -- what the code used to do, why the
  change was made, which approach was abandoned. Each of these has a home
  elsewhere (see `## Where other explanations live`).
- **Commented-out code**, at any length.

Condense rather than delete when a block carries one fact that is not
recoverable: keep that sentence, drop the enumeration around it.

## What a comment is for

State the fact a reader cannot see from here. Delete the comment and ask what it
costs to recover: a fact that lives in another file costs everything, a grouping
the names only half-encode costs a tedious reconstruction, a restated line costs
nothing.

- **Cross-boundary constraints.** Layout, field order, or call order shared with
  a CUDA kernel, the Rust router, an IPC schema, or another process. Nothing in
  the Python file shows the other side, so nothing else can warn the next editor.

  ```python
  # The PD wire schema must match on P and D even when only D runs spec decoding;
  # a seedless prefill writes the invalid sentinel.
  ```

- **Units and layout the name cannot carry.** Tokens vs reqs vs pages vs bytes
  vs slots, and tensor shape/dtype/layout. Encode it in the name first; comment
  only when the name is fixed by an existing interface.

  ```python
  # [num_tokens, num_kv_heads, head_dim], fp8 e4m3, page-major.
  ```

- **Where a magic number came from** -- not what it means. A hardware
  constraint, a measurement, or an admission that it was picked arbitrarily.
  The last one is the most valuable: it tells the next person the value is safe
  to change.

  ```python
  # Hardware constraint: TMA descriptors require 16B alignment.
  # Measured on fp8 GEMM; re-tune when the kernel changes.
  # Arbitrary; no evidence this is the right threshold.
  ```

- **Workarounds, anchored to a verifiable reference and a retirement
  condition.** An unanchored workaround is immortal -- nobody can prove it is
  safe to delete.

  ```python
  # Workaround for pytorch/pytorch#12345; drop once we require torch >= 2.9.
  # Temporary workaround: Event.wait() regresses TPOT on AMD MI355.
  ```
  (second line: `python/sglang/srt/managers/overlap_utils.py:471`)

- **Contracts that are a decision, not a mechanism** -- a sentinel's meaning, a
  deliberate omission from a list, an ordering that looks incidental.

  ```python
  # None means the request is excluded from the radix cache, not that lookup failed.
  ```

- **Structure the names only partly encode** -- a group boundary in a long flat
  block, a section split in a long body. `# ===== Helpers =====` above two
  functions costs nothing to see past; the same banner splitting a genuinely long
  flat module (see `python/sglang/srt/environ.py`) is the only statement of where
  one group ends.

Not worth writing, at any length: prose that restates the line below it,
`# Step 1:` / `# Step 2:` numbering over straight-line code (extract named
helpers if the flow needs numbers), the names of the callers (that is what grep
is for), and hedging -- "this should probably be revisited" is either a fact to
establish and state, or a `TODO(<owner>)`.

## Form and tags

- **One or two lines.** A genuinely intricate invariant may run longer; that is a
  rare exception, not a licence. Documentation blocks are the separate case
  below.
- **ASCII and English only.** No Unicode arrows, math symbols, or CJK.
- **Break at a clause boundary.** If a comment needs a second line, wrap after a
  semicolon or a comma -- never mid-phrase. A sentence that will not split
  cleanly is a sentence that should be shortened instead.
- **Attach the comment to the line it constrains**, not to the top of the
  function. Comments collected into a preamble are the ones that go stale.
- **You change the line, you own its comment.** Update it or delete it -- never
  leave it orphaned. A stale comment is worse than no comment.

Two tag spellings only. New code does not use `FIXME`, `XXX`, or `HACK`; existing
occurrences are grandfathered and get folded into these when the line is touched.

- `# NOTE:` -- a constraint or trap. Most of the time the prefix adds nothing;
  drop it and just state the fact.
- `# TODO(<gh-handle>):` -- planned work, **always** with an owner or an issue
  link. A bare `# TODO: fix this` is rejected in review; an unowned TODO is
  never retired. `TODO(perf)` and similar topic tags are acceptable when the work
  is a standing category rather than one person's task.

## Where other explanations live

Every explanation has exactly one home. A second copy in a comment drifts from
the first.

- **What the code used to do -> `git log`.** Comments describe the current
  state: not what was tried first, not which approach was abandoned. The
  exception is a past failure that is still a live constraint -- write it as the
  constraint, not as the story.

  ```python
  # Bad:  We used to call this before init_new but it broke CUDA graph capture.
  # Good: Must run after graph capture; capturing this allocation deadlocks.
  ```

- **Why the change was made -> the PR body.** Why now, what else was tried,
  which benchmark moved. "Now we also handle the case where ..." argues for a
  diff, and is written for a reviewer who is long gone.
- **Design rationale -> `docs/` or a module docstring. CLI semantics -> the
  `ServerArgs` help text. Test intent -> the test.**
- **Superseded code -> `git show`.**

Also out: review attribution ("as suggested in review") and our own PR numbers
used as a changelog. An upstream issue URL is different -- it is not a changelog
entry but a workaround's retirement condition, and the section above requires it.

What stays next to the line is why *this line* exists, written for a reader who
has neither the PR nor the discussion.

## Documentation blocks

Python docstrings and Doxygen blocks are part of the product on the surfaces
users, integrators, and tooling read, and noise everywhere else. Where warranted
they may run past two lines; every other rule in this file still applies.

- **Yes:** `Engine` and entrypoint public methods, `ServerArgs` help text, base
  classes third parties subclass (attention backends, quant methods, model
  extension points), and `sgl-kernel` op signatures -- where the shape/dtype
  contract is the documentation.
- **Yes:** exported C++ / CUDA entities, as Doxygen with `\brief` / `\param` /
  `\tparam` / `\return`. clangd renders these on hover, driven by
  `CommentFormat: Doxygen`, so the per-parameter enumeration is the
  caller-facing contract rather than filler.
- **Yes:** a bug-regression test, stating in one or two lines which black-box
  behavior must not come back -- the live constraint, not the incident. Root
  cause and repro belong in the issue and the PR
  (see `.claude/rules/unit-test-admission.md`).
- **No:** internal helpers, overrides, private methods. Never hand-write or
  generate Python `Args:` / `Returns:` blocks -- the signature and its type
  hints already carry the names and types.

  The one exception is the orchestration step. In the frozen classes
  (`Scheduler`, `TokenizerManager`, `ModelRunner` -- see
  `.claude/skills/large-class-style`), each `init_*` / `handle_*` / event-loop
  method is an override point a subclass picks from a list; its one-line
  docstring is the catalog entry that says what the step does, so it stays even
  when it reads close to the method name. A genuine private helper
  (`_validate_*`, `_normalize_*`) that no subclass overrides still gets nothing.
