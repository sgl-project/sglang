# Reproduction utils — specification (source of truth)

This file is the source of truth for `scripts/mechanical_refactor_reproduction_utils.py`:
the **clean-move property** its primitives implement, each primitive's contract, and the
byte-diff arbiter's semantics. The module, its tests, and the guides defer to this file; if
any of them disagrees, this file wins. (How to split a change into commits:
`guide-split.md`. How to produce a proof: `guide-construct-proof.md`. How to read one:
`guide-verify-proof.md`.)

## 1. The property — a "clean move"

> A commit is a **clean move** iff every change it makes is code **relocated in the same
> order** — allowing one **uniform indentation shift** of the whole block — plus a small
> fixed set of **move artifacts**, and nothing else.

Equivalently: the commit is reproducible by composing only the faithful relocation
primitives (§2). The whitelist below is exactly what those primitives do; the not-allowed
list is what they refuse to do, so it surfaces as a residual diff.

### 1.1 Allowed — the whole whitelist

- A line **relocated in order**, modulo one **uniform** leading-indentation shift of the
  whole block (the unavoidable reindent of relocation).
- **Defs/classes gathered from scattered positions** in one source into a **new module**,
  each cut verbatim and assembled under an **authored header** — the module-level imports, a
  `logger`, an `if TYPE_CHECKING:` guard — reproduced from the target. The defs are the
  proven relocation (the byte diff certifies the bodies). The header is **audited**, not
  trusted: every header statement must be an import, a docstring, a TYPE_CHECKING import
  block, a `logging.getLogger(__name__)` logger, or an unparse-equivalent copy of an
  assignment actually deleted from the source (`drop_assigns`, e.g. `_is_hip = is_hip()`),
  and every dropped assignment must reappear in the header — anything else raises instead
  of certifying.
- The **body of an extracted function** — an inline block relocated verbatim into a new def,
  with the `def` **signature**, an optional `return`, and the **call** that replaces the block
  authored. The body is the proven relocation; the small interface is authored. This is
  faithful **only** when the body moves unchanged — a de-self, a control-flow restructure, or a
  bookkeeping consolidation is semantic and goes in its own commit first.
- **Import statements** — added, removed, or repathed; single-line or parenthesised
  multi-line. The symbol's home changed, so importers adjust. An import diff is always
  harmless, so it is realised directly from the target (a wholly new module's statement is
  reproduced verbatim, preserving its wrapping); a move to a new module may also add
  `from __future__ import annotations`.
- A one-sided **`@staticmethod` / `@classmethod`** — a method became a free function or
  vice versa.
- A **`self` type annotation dropped** from the moved definition — relocating
  `@staticmethod def foo(self: Target)` into `Target` as `def foo(self)` drops the
  decorator and the now-redundant `self` annotation.
- A **call-site requalification of a moved symbol** — `Owner.foo(x)` → `foo(x)`: same
  symbol, same argument bytes, only the qualifier dropped. (An `Old.foo(x)` → `New.foo(x)`
  owner swap is not a primitive; it surfaces as a residual.)
- A **call-site lowering of a moved method** — a staticmethod call
  `Owner.method(receiver, rest)` → the instance-method call `receiver.method(rest)` (the
  receiver moves out of the argument list).
- **Deleting a source file the relocation emptied** — after its defs moved out, a file
  holding nothing beyond a docstring, imports, or a `TYPE_CHECKING` block may be deleted
  (`delete_file` refuses anything else).
- **Blank-line changes** — ignored (§1.3): the formatter normalises them.

### 1.2 Not allowed — the commit is **not** a clean move

- A **reorder** of lines within the moved block.
- A **statement-level reorder** that relocates no definition — e.g. moving a call below
  other statements in a method body. It changes evaluation order, so it is a *reshape* a
  human must confirm, not a machine-certifiable relocation.
- A **non-uniform** indentation change (one that is not a single shift of the whole block) —
  it can change Python semantics.
- A **trailing-whitespace** change, an internal-whitespace change, or a **line merge/split**.
- A **changed argument** in an otherwise-requalified call.
- A **call rewrite for a symbol that did not move** in this commit (e.g. pointing a
  consumer at a different implementation).
- A **signature change** other than dropping the `self` annotation — a real parameter's
  type, name, default, or position changing.
- A **rename** of the moved symbol (even a privacy flip `_foo` → `foo`).
- **Scaffolding or a constant the move authors into an existing module** — a `logger`, a
  module constant, a `TYPE_CHECKING` guard, or a re-derived `_flag = compute_flag()`
  introduced into a module that already exists. (A *new* module's header is authored from the
  target, §1.1; an existing module's body is not a place to author fresh code.)
- A **changed body in an extracted function** — a de-self (`self.x` → a parameter), a
  control-flow restructure (an `if/elif/else` chain becoming early `return`s), or a
  bookkeeping consolidation folded into the extraction. The body is no longer the source's, so
  it is a semantic rewrite, not a relocation (it goes in its own commit).

A rename, fresh scaffolding, a statement reorder, or an extraction whose body changed is
reshape work; it belongs in the prepare/postpare phases of `guide-split.md`, not in the
move. The reproduce proof reports such a commit as a residual diff — it does not certify it.

### 1.3 Blank lines are ignored

- A blank line never changes Python behavior, and PEP 8 separator blanks legitimately
  collapse when code is split across files or relocated. The formatter normalises them on
  both the reproduced and target sides, so a blank-line-only difference cannot appear in the
  byte diff. This assumes the **target commit is itself pre-commit-clean** (true for any
  commit that passed this repo's hooks); a target that skipped the formatter can show
  blank-line residuals.

## 2. The faithful relocation primitives

Each primitive does only a relocation-faithful edit — AST-located, spliced as original
source text, never regenerated — so a byte match after the formatter certifies the commit
is *exactly* that relocation:

- `move_symbol(name, *, src, dst, into_class, from_class, dedent, drop_self_annotation,
  before, leave_delegate, delegate_name)` — cut a `def` (functions only; a class moves via
  the extract primitives) with its decorators, drop its own `@staticmethod`/`@classmethod`,
  shift the indentation uniformly (a negative `dedent` indents into a class), and paste it
  at a class end, at module level, or above the named sibling `before`. Same-named defs must
  be disambiguated with `from_class`; an ambiguous name or missing anchor raises.
  `leave_delegate` additionally **authors** a forwarding stub (the original header plus one
  `return self.<attr>.<name>(...)` line, `await`ed for an async def) in the source —
  authored code, so audit it like any header.
- `extract_to_new_module(src, dst, *, symbols, future_import)` — cut the contiguous tail of
  the source (the moved defs/classes plus the scaffolding that leads into them — imports,
  TYPE_CHECKING guards, name-target assignments only; an executable trailing statement stops
  the cut) and write it as a new module, prepending `from __future__ import annotations`
  when the move adds it.
- `extract_symbols_to_new_module(src, dst, *, symbols, header, order, drop_assigns)` — cut
  the named defs/classes from **scattered** positions (not just a tail) and assemble the new
  module under the audited `header` (§1.1); `drop_assigns` deletes a module-level constant
  that relocated into the header (a chained `A = B = 1` keeps the surviving bindings).
- `extract_function(src, dst, *, name, signature, body, body_indent, call, return_text,
  before, into_class)` — cut an inline `body` verbatim (it must match at a line boundary),
  re-indent it under the authored `signature` (multi-line string interiors keep their exact
  bytes), insert the def, and replace the block with the authored `call`. Faithful only when
  the body is unchanged.
- `lower_call_sites(name, owner, *, paths)` — `Owner.m(receiver, rest)` → `receiver.m(rest)`
  by splicing the original argument bytes (literal spelling, comments, and the magic
  trailing comma survive); nested matching calls are all rewritten.
- `requalify_call_sites(name, owner, *, paths)` — `Owner.m(args)` → `m(args)`; only the
  qualifier span changes.
- `remove_import(rel, import_text, *, in_function)` — function-scoped or module-level;
  matches whole statements with token boundaries (`import os` cannot hit `import os.path`)
  and removes exactly the matched import even on a semicolon-joined line.
- `remove_imported_name(rel, *, module, name, asname)` — drop a single name from a
  `from m import a, b` (or a plain `import x`), realising a lost import directly instead of
  relying on the formatter to prune it (this repo's ruff has no F811); an import that
  carries comments loses only the dropped alias's own line.
- `add_import(rel, import_stmt)` — the formatter's import sorter places it; in a file with
  no imports it lands below the module docstring.
- `add_typechecking_import(rel, import_stmt)` — append an import inside the destination's
  `if TYPE_CHECKING:` block (a moved annotation needs its type imported there); the sorter
  orders the block.
- `repath_import(rel, *, old_module, new_module, name)` — repath a function-scoped
  `from old import … name …` (relative imports included) to `from new import …` in place
  (module-level repaths fall out of add/remove + the sorter).
- `delete_file(path)` — delete a source module the relocation emptied; refuses a file still
  holding anything beyond a docstring, imports, or a `TYPE_CHECKING` block.

CRLF sources round-trip byte-for-byte (synthesized lines follow the file's newline style),
and column arithmetic is UTF-8-byte-accurate, so non-ASCII text does not shift a rewrite.

## 3. The arbiter — reproduce and byte-diff

`Repro.run()` (and the lower-level `verify_mechanical_refactor`) checks out the base commit
in a throwaway worktree, replays the recorded primitives, runs the repo's pre-commit hooks
on the changed files, and byte-diffs the result against the target commit. An empty diff is
the proof; a non-empty diff is returned as the residual — exactly what the relocation does
not account for.

- It runs the **real formatter**, so a call the formatter split across an `= (` line, or a
  reflow that left a closing bracket as unchanged context, is reproduced exactly — there is
  no diff-shape heuristic to fool.
- One tradeoff is explicit: whatever the repo's **pre-commit hooks auto-fix is absorbed**
  into both sides (e.g. ruff's F401 removing a now-unused import). A hook-introduced change
  rides under a byte match, so the hook set itself is part of the trusted base.
