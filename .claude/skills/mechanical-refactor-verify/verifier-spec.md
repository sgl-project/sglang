# Move verifier — specification (source of truth)

This file defines the rule the move verifier enforces. It is the **single source of
truth**: `mechanical_refactor_verify_utils.py` implements exactly this rule and its comments point here, the
tests (`test_mechanical_refactor_verify_utils.py`) assert exactly this rule, and `SKILL.md` /
`verification-mode.md` describe it in prose. If any of them disagree with this file, this
file wins and the others are the bug.

## What the verifier certifies

`verify_move_commit(commit)` answers one yes/no question about a single commit:

> Is this commit a **pure relocation** — code moved from one place to another **in the
> same order**, allowing one **uniform indentation shift** of the whole block, with no
> change beyond a small fixed set of **move artifacts**?

`CLEAN MOVE` = yes. `NEEDS REVIEW` = anything else.

The bar is strict. The moved block's lines appear in the **same order** on both sides;
the whole block may be shifted by one **constant** indentation amount (the unavoidable
reindent of relocation); and the only other changes allowed are the deterministic
side-effects of relocating code:

- **imports** — the symbol's home changed, so importers add / repath an import;
- a one-sided **`@staticmethod` / `@classmethod`** — a method became a free function;
- a **`self` type annotation dropped** from the moved definition — relocating
  `@staticmethod def foo(self: Target)` into `Target` as the instance method
  `def foo(self)` drops the decorator and the now-redundant annotation on `self`;
- **requalifying a moved symbol's call sites** — `self.foo(x)` becomes `foo(x)`, or
  `Old.foo(x)` becomes `New.foo(x)`: same symbol, same arguments, only the qualifier
  differs;
- **lowering a moved method's call site** — a staticmethod call `Owner.method(receiver,
  rest)` becoming the instance-method call `receiver.method(rest)` (the receiver moves out
  of the argument list), compared on the whole call expression so a formatter's wrapping of
  the line is tolerated;
Everything else is `NEEDS REVIEW`: a **reorder**, a **non-uniform** indentation change, a
**trailing-whitespace** change, a **line merge/split**, a changed **argument**, a call
rewrite for a symbol that **did not move**, a **rename** of the moved symbol, or **any
new module-level scaffolding** the move introduces (a `logger`, a constant, a
`TYPE_CHECKING` guard). The last two are correct flags, not false positives: a move must be
a pure relocation, so a rename or new-module scaffolding belongs in the **prep** commit —
see `prep-and-move.md`.

**Blank lines are ignored.** A blank line never changes Python behavior, and separator
blank lines (PEP 8 spacing between definitions) legitimately collapse when code is split
across files or relocated, so a blank-line-only difference is tolerated.

## Why this split exists — prep is human-reviewed, move is machine-checked

A behaviour-preserving extraction is two commits:

- a **prep** commit — the in-place reshape (de-self a method, retype `self`). A human
  reviews it, so it must be **small** and contain **no relocation**: the code stays
  exactly where it is.
- a **move** commit — the relocation, certified by this verifier.

The whitelist above is exactly what a relocation forces and a human should not have to
re-read. Whitelisting those artifacts is what lets the move be machine-checked; it is
**not** a licence to fold reshape work into the move. If the move's diff contains anything
outside the whitelist, the reshape leaked in and belongs in prep (see `prep-and-move.md`).

## The rule, precisely

Given a commit (diffed against its first parent):

1. **Collect changed lines, per file.** From the patch take the removed (`-`) and added
   (`+`) lines, grouped by file. Within each file, **cancel** any line that appears
   byte-for-byte as **both** removed and added — that is git's own diff artifact (an
   unchanged line re-represented as remove + add when nearby lines change), not a
   relocation; a real relocation keeps its lines because they cross files. Aggregate the
   surviving removed and added lines in patch order. Renames/splits are not followed
   (`-M` off).

2. **Collect import lines.** For every file the commit touches, parse the *before* version
   (at `commit^`) and the *after* version (at `commit`) with Python's `ast`. A line is an
   **import line** if it lies within an `Import` / `ImportFrom` node, so single-line and
   parenthesised multi-line imports are both recognised.

3. **Peel the whitelist, preserving order.** From the surviving lines remove:
   - **imports** (a line whose stripped text is in the parsed import set);
   - **decorators** (a line that is exactly `@staticmethod` or `@classmethod`);
   - **call-site requalifications** — a removed line and an added line that become equal
     after dropping a `Qualifier.` prefix before a **moved symbol** (a qualifier must
     actually be present, so a verbatim body line is not consumed). The moved symbols are
     the `def` / `class` names that appear on both sides (ignoring indentation, and after
     the self-annotation drop, so a method relocated as `def foo(self: T)` -> `def foo(self)`
     still counts);
   - **call-site lowerings** — a call group starting at `Owner.method(receiver, ...)` whose
     canonical form (whitespace-collapsed, receiver moved to the front, redundant `= (...)`
     wrapper dropped) equals an added `receiver.method(...)` call group. A line-based diff
     cannot reassemble a call the formatter split across an `= (` line or whose closing
     bracket was unchanged context, so heavily reflowed call sites still need review (use
     reproduce mode).

4. **Compare the remaining block as an ordered signature.** A block's **signature** is its
   non-blank lines with the block's common leading indent removed, in order, and with a
   type annotation on any `self` parameter dropped (so `def foo(self: Target)` matches
   `def foo(self)`). The removed block and the added block match iff their signatures are
   **equal as sequences**. This absorbs a uniform indentation shift (the common prefix is
   removed) and the `self`-annotation drop, while preserving relative indentation, trailing
   whitespace, and order — so a reorder, a non-uniform indent change, a trailing-whitespace
   change, or a line merge makes the signatures differ.

5. **Verdict.** `CLEAN MOVE` iff the signatures match and at least one line relocated.
   Otherwise `NEEDS REVIEW`, printing the signature diff. A commit whose surviving removed
   and added lists are both empty (a pure rename git records with no content change) is
   clean too.

## Allowed — the whole whitelist

- A line **relocated in order**, modulo one **uniform** leading-indentation shift of the
  whole block.
- **Import statements** — added, removed, or repathed; single-line or multi-line.
- A one-sided **`@staticmethod` / `@classmethod`**.
- A **`self` type annotation dropped** from the moved definition.
- A **call-site requalification of a moved symbol**, or a **call-site lowering**
  (`Owner.method(receiver, rest)` → `receiver.method(rest)`) tolerant of a formatter's
  line wrapping.
- **Blank-line changes** — ignored.

## Not allowed (→ NEEDS REVIEW)

- A **reorder** of lines within the moved block.
- A **non-uniform** indentation change (one that is not a single shift of the whole block).
- A **trailing-whitespace** change, an internal-whitespace change, or a **line
  merge/split**.
- A **changed argument** in an otherwise-requalified call.
- A **call rewrite for a symbol that did not move** in this commit — e.g. pointing a
  consumer at a different implementation. This keeps a consumer-only edit from passing.
- A **signature change** other than dropping the `self` annotation — a real parameter's
  type, name, default, or position changing.
- A **constant re-derived** in the destination module (`_flag = compute_flag()`).
- A line a **formatter re-wrapped** so it is no longer identical (use reproduce mode, see
  `reproduce-mode.md`).

## What the verdict does and does not assert

- **Order-aware.** The block is compared as a sequence, so a relocation that reorders lines
  within the moved block reads `NEEDS REVIEW`.
- **Uniform-indentation-aware.** A whole-block shift by one constant amount is allowed; a
  non-uniform change is flagged, since it can change Python semantics.
- **Blank-line-insensitive** by design (see above).
- **Requalification is scoped to symbols relocated in this commit**, so a consumer-only
  call rewrite (no relocated definition) cannot pass as a move.
- It judges the **shape of the diff**, not intent. Confirm the commit's purpose from its
  subject before trusting a `CLEAN MOVE`.
- Import classification is by **text membership** in the parsed import set; a code line
  whose text coincidentally equals an import line elsewhere in the file would be treated as
  an import. This is rare and the strictness elsewhere makes it harmless.
