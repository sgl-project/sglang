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

**The test.** Delete the comment. If someone familiar with the repo can recover
the fact from the surrounding code plus a grep, it stays deleted.

The next three sections are that test applied. A comment either carries a fact
you cannot see from here (**keep**), carries nothing the code does not already
carry (**delete**), or carries a real fact whose home is somewhere else
(**move**).

## Keep: facts you cannot see from here

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

## Delete: nothing the code does not already carry

- **Restating the code.** The function name, the next line, the branch
  condition, or the loop body in prose.
- **Step-by-step play-by-play.** `# Step 1: ...` / `# Step 2: ...` over
  straight-line code. If the flow needs numbering, extract named helpers.
- **Naming the callers.** That is what grep is for.
- **Section banners on short bodies.** `# ===== Helpers =====` above two
  functions. Banners are for genuinely long modules only
  (see `python/sglang/srt/environ.py`).
- **Hedging.** "This should probably be revisited." Either establish the fact
  and state it, or file it as `TODO(<owner>)`.

## Move: real facts that live somewhere else

Every explanation has exactly one home. A second copy in a comment drifts from
the first.

- **What the code used to do -> `git log`.** Comments describe the current
  state: not what was tried first, not which approach was abandoned.

  The exception is a past failure that is still a live constraint. That is not
  history -- but write it as the constraint, not as the story.

  ```python
  # Bad:  We used to call this before init_new but it broke CUDA graph capture.
  # Good: Must run after graph capture; capturing this allocation deadlocks.
  ```

- **Why the change was made -> the PR body.** Why now, what else was tried,
  which benchmark moved. "Now we also handle the case where ..." argues for a
  diff, and is written for a reviewer who is long gone.
- **Design rationale -> `docs/` or a module docstring. CLI semantics -> the
  `ServerArgs` help text. Test intent -> the test.**
- **Superseded code -> `git show`.** Never leave commented-out code behind.

Also out: review attribution ("as suggested in review") and our own PR numbers
used as a changelog. An upstream issue URL is different -- it is not a changelog
entry but a workaround's retirement condition, and the Keep section requires it.

What stays next to the line is why *this line* exists, written for a reader who
has neither the PR nor the discussion.

## Form

Governs `#` / `//` comments; the length and placement of documentation blocks
follow the section below. ASCII-and-English applies to both.

- **One or two lines.** A genuinely intricate invariant may run longer; that is
  a rare exception, not a licence.
- **ASCII and English only.** No Unicode arrows, math symbols, or CJK.
- **Break at a clause boundary.** If a comment needs a second line, wrap after a
  semicolon or a comma -- never mid-phrase. A sentence that will not split
  cleanly is a sentence that should be shortened instead.
- **Attach the comment to the line it constrains**, not to the top of the
  function. Comments collected into a preamble are the ones that go stale.
- **You change the line, you own its comment.** Update it or delete it -- never
  leave it orphaned. A stale comment is worse than no comment.

## Tags

Two spellings only. New code does not use `FIXME`, `XXX`, or `HACK`; existing
occurrences are grandfathered and get folded into these when the line is touched.

- `# NOTE:` -- a constraint or trap. Most of the time the prefix adds nothing;
  drop it and just state the fact.
- `# TODO(<gh-handle>):` -- planned work, **always** with an owner or an issue
  link. A bare `# TODO: fix this` is rejected in review; an unowned TODO is
  never retired. `TODO(perf)` and similar topic tags are acceptable when the work
  is a standing category rather than one person's task.

## API documentation blocks

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
