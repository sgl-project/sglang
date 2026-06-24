# Move verifier — specification (source of truth)

This file defines the rule the move verifier enforces. It is the **single source of
truth**: `mechanical_refactor_verify_utils.py` implements exactly this rule and its
comments point here, the tests assert exactly this rule, and `SKILL.md` /
`verification-mode.md` describe it in prose. If any of them disagree with this file,
this file wins and the others are the bug.

## What the verifier certifies

`verify_move_commit(commit)` answers one yes/no question about a single commit:

> Is this commit a **pure relocation** — code moved from one place to another
> byte-for-byte, with no change beyond a small, fixed set of **move artifacts**?

`CLEAN MOVE` = yes. `NEEDS REVIEW` = anything else.

The bar is strict. The moved **body is byte-for-byte**, and the only other changes
allowed are the deterministic side-effects of relocating code:

- **imports** — the symbol's home changed, so importers add / repath an import;
- a one-sided **`@staticmethod` / `@classmethod`** — a method became a free function;
- **requalifying a moved symbol's call sites** — its home changed, so `self.foo(x)`
  becomes `foo(x)`, or `Old.foo(x)` becomes `New.foo(x)`: same symbol, same arguments,
  only the qualifier differs.

Everything else is `NEEDS REVIEW`. A change to the body, to a call's **arguments**, to a
symbol that **did not move**, or a line that merely *looks* almost the same, is **not** a
move artifact.

## Why this split exists — prep is human-reviewed, move is machine-checked

A behaviour-preserving extraction is two commits:

- a **prep** commit — the in-place reshape (de-self a method, retype `self`). A human
  reviews it, so it must be **small** and contain **no relocation**: the code stays
  exactly where it is, so the diff is a handful of lines that are easy to eyeball.
- a **move** commit — the relocation, certified by this verifier.

The whitelist above is exactly what a relocation forces and a human should not have to
re-read. Whitelisting those artifacts is what lets the move be machine-checked; it is
**not** a licence to fold reshape work into the move. If the move's diff contains
anything outside the whitelist, it is not a pure move — the reshape leaked in, and it
belongs in prep (see `prep-and-move.md`).

## The rule, precisely

Given a commit (diffed against its first parent):

1. **Collect changed lines.** From the commit's patch, take the removed (`-`) and added
   (`+`) lines, excluding diff/hunk headers. Renames and splits are not followed (`-M`
   off), so a renamed or split file appears as its full content removed and re-added —
   which the match in step 4 pairs up.

2. **Collect import lines.** For every file the commit touches, parse the *before*
   version (at `commit^`) and the *after* version (at `commit`) with Python's `ast`. A
   line is an **import line** if it lies within an `Import` / `ImportFrom` node, so
   single-line and parenthesised multi-line imports (every member line) are both
   recognised. Record the stripped text as `imports_before` and `imports_after`. A file
   that does not parse as Python contributes no import lines.

3. **Normalise.** `strip()` each changed line (drop leading indentation and trailing
   whitespace) and discard blank lines. Indentation is ignored so a method body dedented
   to module level — or re-indented under another class — still matches.

4. **Match the move.** Let `R` and `A` be the multisets of normalised removed and added
   lines. The moved lines are `R ∩ A`; the leftovers are `R_only = R − A` and
   `A_only = A − R`. The names defined by a `def` / `class` line **within the moved
   lines** are the commit's **moved symbols**.

5. **Peel the whitelist off the leftovers.**
   - a leftover removed line in `imports_before` (or added line in `imports_after`) is an
     allowed **import**;
   - a leftover line that is exactly `@staticmethod` or `@classmethod` is an allowed
     **decorator**;
   - of what remains, drop every `Qualifier.` prefix that sits before a **moved symbol**
     and match removed against added again: a pair that now matches is an allowed
     **call-site requalification**. Only the qualifier is removed, so a call whose
     arguments also changed will not match.

6. **Verdict.** `CLEAN MOVE` iff there is at least one moved line and nothing is left to
   review. Otherwise `NEEDS REVIEW`, listing every leftover. A commit with no removed and
   no added lines (a pure rename git records with no content change) is clean too.

## Allowed — the whole whitelist

- A line relocated **byte-for-byte modulo leading indentation**.
- **Import statements** — added, removed, or repathed; single-line or multi-line.
- A one-sided **`@staticmethod` / `@classmethod`**.
- A **call-site requalification of a moved symbol** — removed and added lines that are
  identical after dropping a `Qualifier.` before a symbol defined in the relocated block.

## Not allowed (→ NEEDS REVIEW)

- Any change to a relocated line's **content** beyond its qualifier — an argument
  changed, an operator changed, internal spacing changed ("looks almost the same").
- A **call rewrite for a symbol that did not move** in this commit (no matching relocated
  `def` / `class`) — e.g. pointing a consumer at a different implementation. This is what
  keeps a consumer-only edit from passing as a move.
- A **signature change** on the moved definition itself (`def f(self: T)` → `def f(self)`):
  the `def` line is not byte-identical, so it is not even a moved symbol.
- A **constant re-derived** in the destination module (`_flag = compute_flag()`).
- A line a **formatter re-wrapped** during the move so it is no longer byte-identical
  (use reproduce mode, see `reproduce-mode.md`).

## What the verdict does and does not assert

- **Order-blind.** The match is on multisets, so a relocation that also reorders lines
  within the moved block still reads `CLEAN MOVE`. Cross-check order with
  `git show <commit> --color-moved=dimmed-zebra --color-moved-ws=allow-indentation-change`.
- **Indentation-blind by design.** Two lines that differ only in leading whitespace are
  the same line.
- **Requalification is scoped to symbols relocated in this commit.** A call rewrite is
  forgiven only for a symbol whose `def` / `class` line moved here, so a consumer-only
  call rewrite (no relocated definition) cannot pass as a move.
- It judges the **shape of the diff**, not intent. Confirm the commit's purpose from its
  subject before trusting a `CLEAN MOVE`.
- Import classification is by **text membership** in the parsed import set; a code line
  whose text coincidentally equals an import line elsewhere in the file would be treated
  as an import. This is rare and the strictness elsewhere makes it harmless.
