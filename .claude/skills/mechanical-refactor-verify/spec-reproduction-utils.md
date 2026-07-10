# Reproduction utils — specification (source of truth)

## 1. Scope

- Source of truth for `scripts/mechanical_refactor_reproduction_utils.py`: the
  **clean-move property** its primitives implement (§2), each primitive's contract (§3),
  and the byte-diff arbiter's semantics (§4).
- The module, its tests, and the guides defer to this file; on any disagreement, this
  file wins.
- Elsewhere: commit splitting → `guide-split.md`; producing a proof →
  `guide-construct-proof.md`; reading one → `guide-verify-proof.md`.

## 2. The property — a "clean move"

> A commit is a **clean move** iff every change it makes is code **relocated in the same
> order** — allowing one **uniform indentation shift** of the whole block — plus a small
> fixed set of **move artifacts**, and nothing else.

- Equivalently: the commit is reproducible by composing only the primitives of §3.
- The whitelist (§2.1) is exactly what they do; the not-allowed list (§2.2) is what they
  refuse, so it surfaces as a residual diff.

### 2.1 Allowed — the whole whitelist

- A line **relocated in order**, modulo one **uniform** leading-indentation shift of the
  whole block.
- **Defs/classes gathered from scattered positions** into a **new module**, each cut
  verbatim, assembled under an **audited authored header**:
    - the byte diff certifies the bodies; the header is reproduced from the target;
    - the header audit accepts only: imports, a docstring, a TYPE_CHECKING import block,
      a `logging.getLogger(__name__)` logger, or an unparse-equivalent copy of an
      assignment actually deleted from the source (`drop_assigns`, e.g.
      `_is_hip = is_hip()`);
    - every dropped assignment must reappear in the header — anything else raises instead
      of certifying.
- The **body of an extracted function** — an inline block relocated verbatim into a new
  def; the `def` signature, an optional `return`, and the replacing `call` are authored.
  Faithful **only** when the body moves unchanged; a de-self, control-flow restructure, or
  bookkeeping consolidation is semantic and goes in its own commit first.
- **Import statements** — added, removed, or repathed; single-line or parenthesised.
  Realised directly from the target (a wholly new module's statement verbatim, wrapping
  preserved); a new-module move may add `from __future__ import annotations`.
- A one-sided **`@staticmethod` / `@classmethod`** — method ↔ free function.
- A **`self` type annotation dropped** from the moved definition — relocating
  `@staticmethod def foo(self: Target)` into `Target` as `def foo(self)`.
- A **call-site requalification** — `Owner.foo(x)` → `foo(x)`: same symbol, same argument
  bytes, only the qualifier dropped. (An `Old.foo(x)` → `New.foo(x)` owner swap is not a
  primitive; it surfaces as a residual.)
- A **call-site lowering** — `Owner.method(receiver, rest)` → `receiver.method(rest)`:
  the receiver moves out of the argument list.
- **Deleting a source file the relocation emptied** — nothing left beyond a docstring,
  imports, or a `TYPE_CHECKING` block (`delete_file` refuses anything else).
- **Blank-line changes** — ignored (§2.3).

### 2.2 Not allowed — the commit is **not** a clean move

- A **reorder** of lines within the moved block.
- A **statement-level reorder** that relocates no definition — it changes evaluation
  order: a reshape a human must confirm, not a certifiable relocation.
- A **non-uniform** indentation change — it can change Python semantics.
- A **trailing-whitespace** change, an internal-whitespace change, or a **line
  merge/split**.
- A **changed argument** in an otherwise-requalified call.
- A **call rewrite for a symbol that did not move** in this commit.
- A **signature change** other than dropping the `self` annotation.
- A **rename** of the moved symbol (even a privacy flip `_foo` → `foo`).
- **Scaffolding or a constant authored into an existing module** — a logger, a module
  constant, a `TYPE_CHECKING` guard, a re-derived `_flag = compute_flag()`. (A *new*
  module's header is authored from the target, §2.1; an existing module's body is not a
  place to author fresh code.)
- A **changed body in an extracted function** — de-self, control-flow restructure, or a
  folded-in bookkeeping change: a semantic rewrite, not a relocation.

- Reshape work (rename, fresh scaffolding, statement reorder, changed extraction body)
  belongs in the prepare/postpare phases of `guide-split.md`.
- The proof reports it as a residual — never certifies it.

### 2.3 Blank lines are ignored

- A blank line never changes Python behavior; PEP 8 separator blanks legitimately collapse
  on relocation.
- The formatter normalises both the reproduced and target sides, so a blank-line-only
  difference cannot reach the byte diff.
- Assumption: the **target commit is itself pre-commit-clean** (true for any commit that
  passed this repo's hooks); a target that skipped the formatter can show blank-line
  residuals.

## 3. The faithful relocation primitives

- Each primitive does only a relocation-faithful edit — AST-located, spliced as original
  source text, never regenerated.
- Therefore a byte match after the formatter certifies the commit is *exactly* that
  relocation.

- `move_symbol(name, *, src, dst, into_class, from_class, dedent, drop_self_annotation,
  before, leave_delegate, delegate_name)`:
    - cuts a `def` (functions only; a class moves via the extract primitives) with its
      decorators; drops its own `@staticmethod`/`@classmethod`;
    - shifts indentation uniformly (negative `dedent` indents into a class);
    - pastes at a class end, at module level, or above the named sibling `before`;
    - same-named defs need `from_class`; an ambiguous name or missing anchor raises;
    - `leave_delegate` **authors** a forwarding stub in the source (original header + one
      `return self.<attr>.<name>(...)`, `await`ed for async) — audit it like any header.
- `extract_to_new_module(src, dst, *, symbols, future_import)`:
    - cuts the contiguous source tail: the moved defs/classes plus leading scaffolding
      (imports, TYPE_CHECKING guards, name-target assignments only);
    - an executable trailing statement stops the cut;
    - prepends `from __future__ import annotations` when the move adds it.
- `extract_symbols_to_new_module(src, dst, *, symbols, header, order, drop_assigns)`:
    - cuts the named defs/classes from **scattered** positions; assembles the new module
      under the audited `header` (§2.1);
    - `drop_assigns` deletes a relocated module-level constant from the source; a chained
      `A = B = 1` keeps the surviving bindings.
- `extract_function(src, dst, *, name, signature, body, body_indent, call, return_text,
  before, into_class)`:
    - cuts an inline `body` verbatim (must match at a line boundary);
    - re-indents under the authored `signature` — multi-line string interiors keep their
      exact bytes;
    - replaces the block with the authored `call`.
- `lower_call_sites(name, owner, *, paths)` — `Owner.m(receiver, rest)` →
  `receiver.m(rest)` by splicing the original argument bytes (literal spelling, comments,
  magic trailing comma survive); nested matching calls are all rewritten.
- `requalify_call_sites(name, owner, *, paths)` — `Owner.m(args)` → `m(args)`; only the
  qualifier span changes.
- `remove_import(rel, import_text, *, in_function)` — function-scoped or module-level;
  whole-statement match with token boundaries (`import os` cannot hit `import os.path`);
  removes exactly the matched import even on a semicolon-joined line.
- `remove_imported_name(rel, *, module, name, asname)` — drops one name from a
  `from m import a, b` (or a plain `import x`), realising a lost import directly (this
  repo's ruff has no F811). A name on its own line in an exploded, parenthesized import is
  deleted in place when **2+ names survive** (or the import carries comments): the parens,
  the magic trailing comma, and the comments are preserved and the formatter leaves it
  multi-line — a flat rebuild would drop the magic comma and collapse an import the target
  left multi-line. A **lone** surviving name with no comments collapses to a single line
  (the formatter does not keep one name exploded); a name sharing a line (a flat
  single-line import) is likewise rebuilt. Dropping the sole name removes the whole
  statement.
- `add_imported_name(rel, *, module, name, asname)` — the dual of `remove_imported_name`:
  adds one name to an existing `from module import a, b`. Use it (over `add_import`) when the
  target extends an existing line rather than adding a fresh statement — the sorter will not
  merge a new statement across an intervening non-import (e.g. a module-level assignment
  between two import blocks). An import carrying comments is refused (a rebuild would drop
  them); a name already present fails loudly.
- `add_import(rel, import_stmt)` — the import sorter places it; with no existing imports
  it lands below the module docstring.
- `add_typechecking_import(rel, import_stmt)` — appends inside the destination's
  `if TYPE_CHECKING:` block; the sorter orders it. A lone `pass` placeholder (the block's
  only statement) is dropped, since populating an empty block makes its placeholder redundant.
- `repath_import(rel, *, old_module, new_module, name)` — repaths a function-scoped
  `from old import … name …` (relative imports included) in place; module-level repaths
  fall out of add/remove + the sorter.
- `delete_file(path)` — deletes a source module the relocation emptied; refuses anything
  beyond a docstring, imports, or a `TYPE_CHECKING` block.

Cross-cutting guarantees:

- CRLF sources round-trip byte-for-byte; synthesized lines follow the file's newline
  style.
- Column arithmetic is UTF-8-byte-accurate; non-ASCII text does not shift a rewrite.

## 4. The arbiter — reproduce and byte-diff

`Repro.run()` (and the lower-level `verify_mechanical_refactor`):

- checks out the base commit in a throwaway worktree;
- replays the recorded primitives;
- runs the repo's pre-commit hooks on the changed files;
- byte-diffs against the target commit — an empty diff is the proof; a non-empty diff is
  returned as the residual, exactly what the relocation does not account for.

Properties:

- It runs the **real formatter**: a call split across an `= (` line, or a reflow leaving a
  closing bracket as context, reproduces exactly — no diff-shape heuristic to fool.
- Explicit tradeoff: whatever the **pre-commit hooks auto-fix is absorbed** on both sides
  (e.g. ruff's F401 removing a now-unused import). A hook-introduced change rides under a
  byte match, so the hook set is part of the trusted base.
