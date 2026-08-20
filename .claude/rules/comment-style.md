# Comment Style

Comments are reviewed like code, but with the burden of proof reversed: the
author must justify a comment's existence, not the reviewer its removal.

**The test.** Delete the comment. If someone familiar with the repo can recover
the fact from the surrounding code plus a grep, it stays deleted.

## Write these

A comment earns its place by carrying a fact the reader cannot see from here.

- **Cross-boundary constraints.** Layout, field order, or call order shared with
  a CUDA kernel, the Rust router, an IPC schema, or another process. Nothing in
  the Python file shows the other side, so nothing else can warn the next editor.

  ```python
  # The PD wire schema must match on P and D even when only D runs spec decoding;
  # a seedless prefill writes the invalid sentinel.
  ```

- **Units and layout that the name cannot carry.** Tokens vs reqs vs pages vs
  bytes vs slots, and tensor shape/dtype/layout. Encode it in the name first;
  comment only when the name is fixed by an existing interface.

  ```python
  # [num_tokens, num_kv_heads, head_dim], fp8 e4m3, page-major.
  ```

- **Where a magic number came from** -- not what it means. A hardware
  constraint, a measurement, or an admission that it was picked arbitrarily.
  The last one is the most valuable: it tells the next person the value is
  safe to change.

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

## Delete these

- **Restating the code.** The function name, the next line, the branch
  condition, or the loop body in prose.
- **Narrating the diff.** "Now we also handle the case where ..." describes a
  change, not the code. Written for a reviewer who is long gone.
- **Step-by-step play-by-play.** `# Step 1: ...` / `# Step 2: ...` over
  straight-line code. If the flow needs numbering, extract named helpers.
- **Naming the callers.** That is what grep is for.
- **Section banners on short bodies.** `# ===== Helpers =====` above two
  functions. Banners are for genuinely long modules only
  (see `python/sglang/srt/environ.py`).
- **Commented-out code, review attribution, PR numbers.** `git blame` and the
  PR carry all three.
- **Hedging.** "This should probably be revisited." Either establish the fact
  and state it, or file it as `TODO(<owner>)`.

## Form

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
  never retired. `TODO(perf)` and similar topic tags are acceptable when the
  work is a standing category rather than one person's task.

## Docstrings

Docstrings are part of the product on the surfaces users and integrators touch,
and noise everywhere else.

- **Yes:** `Engine` and entrypoint public methods, `ServerArgs` help text, base
  classes third parties subclass (attention backends, quant methods, model
  extension points), and `sgl-kernel` op signatures -- where the shape/dtype
  contract is the documentation.
- **No:** internal helpers, overrides, private methods. Never auto-generate
  `Args:` / `Returns:` blocks for a three-line function.

## Current state, not history

Comments describe what the code does now. No record of what it used to do, what
was tried first, or which approach was abandoned.

The exception is when a past failure is still a live constraint. Then it is not
history -- but write it as the constraint, not as the story.

```python
# Bad:  We used to call this before init_new but it broke CUDA graph capture.
# Good: Must run after graph capture; capturing this allocation deadlocks.
```

## Do not duplicate a source of truth

Design rationale lives in `docs/` or a module docstring; CLI semantics live in
the `ServerArgs` help text; test intent lives in the test. Copying any of them
into a comment creates a second copy that drifts. What stays next to the line is
why *this line* exists.
