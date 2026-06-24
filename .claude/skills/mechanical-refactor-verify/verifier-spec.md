# Move verifier — specification (source of truth)

This file defines the rule the move verifier enforces. It is the **single source of
truth**: `mechanical_refactor_verify_utils.py` implements exactly this rule and its
comments point here, the tests assert exactly this rule, and `SKILL.md` /
`verification-mode.md` describe it in prose. If any of them disagree with this file,
this file wins and the others are the bug.

## What the verifier certifies

`verify_move_commit(commit)` answers one yes/no question about a single commit:

> Is this commit a **pure relocation** — code moved from one place to another
> byte-for-byte, with no other change except **import statements**?

`CLEAN MOVE` = yes. `NEEDS REVIEW` = anything else.

The bar is deliberately strict. **Only a byte-for-byte move is a move.** Imports are
the one allowed non-moved change. A line that merely *looks* almost the same but
differs by a single byte is **not** a move and is reported for review.

## The rule, precisely

Given a commit (diffed against its first parent):

1. **Collect changed lines.** From the commit's patch, take the removed lines (`-`) and
   added lines (`+`), excluding diff/hunk headers. Renames and splits are **not**
   followed (`-M` is off), so a renamed or split file appears as its full content
   removed and re-added — which the move match below pairs up.

2. **Collect import lines.** For every file the commit touches, parse the *before*
   version (at `commit^`) and the *after* version (at `commit`) with Python's `ast`.
   A source line is an **import line** if it lies within the line span of an `Import`
   or `ImportFrom` node. This is structural, so both single-line imports and
   parenthesised multi-line imports (every member line) are recognised. Record the
   stripped text of each import line:
   - `imports_before` = import-line texts across all before-versions,
   - `imports_after` = import-line texts across all after-versions.
   A file that does not parse as Python contributes no import lines.

3. **Normalise.** For each changed line, `strip()` it (drop leading indentation and
   trailing whitespace) and discard blank lines. Indentation is ignored so a method
   body dedented to module level — or re-indented under another class — still matches.

4. **Match the move.** Let `R` and `A` be the multisets of normalised removed and added
   lines. The moved lines are the multiset intersection `R ∩ A`. The leftovers are
   `R_only = R − A` and `A_only = A − R`.

5. **Allow only imports as leftovers.** A leftover removed line is allowed iff its text
   is in `imports_before`; a leftover added line is allowed iff its text is in
   `imports_after`. Everything else is **to review**.

6. **Verdict.**
   - `CLEAN MOVE` iff there is at least one moved line and nothing is left to review.
   - Otherwise `NEEDS REVIEW`, listing every to-review line.
   - A commit with no removed and no added lines (a pure rename git records with no
     content change) has nothing to review and is reported clean as well.

## Allowed — and nothing else

- A line relocated **byte-for-byte modulo leading indentation**.
- **Import statements** — added, removed, or repathed; single-line or multi-line.

## Not allowed (→ NEEDS REVIEW)

Each of these leaves a non-import leftover, so the commit is not a pure move:

- A **call-site rewrite**: `self.foo(x)` → `foo(x)`, or `Old.foo(x)` → `New.foo(x)`.
  The old and new call lines are not byte-identical, so each is an unmatched leftover.
- A **decorator** added or dropped (`@staticmethod`, `@classmethod`).
- A **signature change**: `def f(self: T)` → `def f(self)`.
- A **constant re-derived** in the destination module: `_flag = compute_flag()`.
- A line a **formatter re-wrapped** during the move so it is no longer byte-identical.
- Anything that "looks almost the same but changed a little".

To certify a move that would otherwise carry one of these, push the non-move change
into a separate commit (see `prep-and-move.md`), or — when a formatter re-wraps lines —
use reproduce mode (see `reproduce-mode.md`) instead.

## What the verdict does and does not assert

- **Order-blind.** The match is on multisets, so a relocation that also reorders lines
  within the moved block still reads `CLEAN MOVE`. Cross-check order with
  `git show <commit> --color-moved=dimmed-zebra --color-moved-ws=allow-indentation-change`.
- **Indentation-blind by design.** Two lines that differ only in leading whitespace are
  the same line.
- It judges the **shape of the diff**, not intent. Because imports are the only
  tolerated leftover, a commit that changes any non-import line cannot read clean — but
  still confirm the commit's purpose from its subject before trusting the verdict.
- Import classification is by **text membership** in the parsed import set. A code line
  whose text coincidentally equals an import line elsewhere in the file would be treated
  as an import; this is rare and the strictness elsewhere makes it harmless.
