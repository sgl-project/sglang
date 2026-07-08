# Mechanical-move proof — specification (source of truth)

## 1. What this file is

- This file is the **single source of truth** for *what the skill certifies* and *how*. The
  code (`scripts/mechanical_refactor_reproduce_utils.py`, `scripts/mechanical_refactor_generate_proof.py`),
  the tests, and the prose guides (`SKILL.md`, `guide.md`) all implement or describe this
  file. If any of them disagrees with it, **this file wins** and the others are the bug.
- There is **one property** being certified — *a commit is a pure relocation* — and **one
  proof** of it: **reproduce** (§3). Regenerate the move from the base commit with faithful
  AST primitives, run the formatter, and diff byte-for-byte against the target commit. An
  empty diff is a machine proof; any residual is a bundled non-move change surfaced for
  review.
- The proof is authoritative because it runs the **real formatter** and compares **bytes**:
  a commit a formatter rewrapped is reproduced exactly, and the proof is a small,
  self-contained Python script anyone can re-run and audit.

## 2. The property — a "clean move"

> A commit is a **clean move** iff every change it makes is code **relocated in the same
> order** — allowing one **uniform indentation shift** of the whole block — plus a small
> fixed set of **move artifacts**, and nothing else.

Equivalently: the commit is reproducible by composing only the faithful relocation
primitives (§3.3). The whitelist below is exactly what those primitives do; the not-allowed
list is what they refuse to do, so it surfaces as a residual diff.

### 2.1 Allowed — the whole whitelist

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
  authored. The body is the proven relocation; the small interface is authored (§2.4). This is
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
- **Blank-line changes** — ignored (§2.3): the formatter normalises them.

### 2.2 Not allowed — the commit is **not** a clean move

- A **reorder** of lines within the moved block.
- A **statement-level reorder** that relocates no definition — e.g. moving a call below
  other statements in a method body. It changes evaluation order, so it is a *reshape* a
  human must confirm, not a machine-certifiable relocation (§2.4, §3.4).
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
  target, §2.1; an existing module's body is not a place to author fresh code.)
- A **changed body in an extracted function** — a de-self (`self.x` → a parameter), a
  control-flow restructure (an `if/elif/else` chain becoming early `return`s), or a
  bookkeeping consolidation folded into the extraction. The body is no longer the source's, so
  it is a semantic rewrite, not a relocation (it goes in its own commit, §2.4).

A rename, fresh scaffolding, a statement reorder, or an extraction whose body changed is
reshape work, so it does not ride in the move (§2.4, `guide.md`). The
reproduce proof reports such a commit as a residual diff (or, when no definition relocated at
all, as unsupported) — it does not certify it.

### 2.3 Blank lines are ignored

- A blank line never changes Python behavior, and PEP 8 separator blanks legitimately
  collapse when code is split across files or relocated. The formatter normalises them on
  both the reproduced and target sides, so a blank-line-only difference cannot appear in the
  byte diff. This assumes the **target commit is itself pre-commit-clean** (true for any
  commit that passed this repo's hooks); a target that skipped the formatter can show
  blank-line residuals.

### 2.4 The three phases — prepare, move, postpare

A behaviour-preserving relocation is up to **three commits**, in this order:

- an **(optional) prepare** commit — a **minimal** in-place reshape that the relocation needs
  (de-self a method, retype `self`). A human reviews it, so it must be small and contain **no
  cross-file def relocation and no body relocation**: the code stays where it is.
- a **move** commit — the pure relocation, certified by this spec. This carries the **bulk**.
- an **(optional) postpare** commit — a **minimal** tail fixup the relocation cannot do
  mechanically: a module path inside a **string literal**, a doc reference. A human reviews it.

Both ends are optional and minimal; neither prepare nor postpare ever relocates a def across
files or moves a body. The whitelist in §2.1 is exactly what a relocation *forces*; it is
**not** a licence to fold reshape work into the move.

A semantic refactor is never a prepare, an extraction's bulk goes in the move, and a
new-module extraction needs no staging prep — the rationale, the case recipes, and the
anti-patterns live in `guide.md` Part 1; this spec only fixes the phase boundaries above.

## 3. The proof — reproduce: regenerate the move and byte-diff

Regenerate the move from the base commit with faithful AST primitives, run the formatter
(pre-commit: black / isort / ruff), and diff byte-for-byte against the target commit. An
empty diff proves the commit is exactly that relocation.

### 3.1 The auto-generated, self-contained Python script

- The generator is run over a commit range:

  ```
  python3 scripts/mechanical_refactor_generate_proof.py <base>..<tip> --match -move: --out DIR
  ```

- For each matched commit it:
    - **infers a recipe** from the commit's diff + before-state AST — which symbols moved
      (`src` → `dst`, into which class, or into a new module), which call sites were lowered
      or requalified, which imports were repathed, and the symmetric module-level import diff
      each file gained or lost;
    - **emits** a standalone `repro_scripts/<sha>.py` whose only import is
      `mechanical_refactor_reproduce_utils`;
    - **runs** it — checks out the base in a throwaway worktree, replays the primitives, runs
      `pre-commit`, and byte-diffs against the target;
    - records the verdict.
- The **product** is a self-contained folder, independently auditable without the skill:
    - `repro_scripts/<sha>.py` — one script per commit;
    - `output.log` and `output.html` — the verdicts;
    - a copy of `mechanical_refactor_reproduce_utils.py` — the only dependency.
- **Verdicts**:
    - `PASS` — byte-identical to the target; the commit is certified a clean move.
    - `RESIDUAL` — a non-empty diff; the residual is the bundled non-move change to review.
    - `UNSUPPORTED` — no definition relocated (a rename or statement-level reshape), or the
      recipe is not yet inferable (§3.4); review as prep, or hand-write the `Repro`.

### 3.2 Why it is a trustworthy proof

- It runs the **real formatter**, so a call the formatter split across an `= (` line, or a
  reflow that left a closing bracket as unchanged context, is reproduced exactly — there is
  no diff-shape heuristic to fool.
- The proof is the **few primitive calls** in the generated script; a reviewer audits those,
  not a hand-written transform.
- It is **self-contained and re-runnable** by anyone (a CI step, a PR reviewer), so the
  certification is not tied to the skill being installed.
- One tradeoff is explicit: whatever the repo's **pre-commit hooks auto-fix is absorbed**
  into both sides (e.g. ruff's F401 removing a now-unused import). A hook-introduced change
  rides under a PASS, so the hook set itself is part of the trusted base.

### 3.3 The faithful relocation primitives

Each primitive does only a **relocation-faithful, AST-driven** edit and never changes logic,
so a byte match after the formatter certifies the commit is *exactly* that relocation:

- `move_symbol` — cut a `def` (functions only; a class moves via the extract primitives)
  with its decorators, drop its own `@staticmethod`/`@classmethod`, shift the indentation
  uniformly, and paste it at a class end, at module level, or above a named sibling
  (`before=`). Same-named defs must be disambiguated with `from_class=`; an ambiguous or
  missing anchor raises. `leave_delegate=` additionally **authors** a forwarding stub
  (the original header plus one `return self.<attr>.<name>(...)` line) in the source —
  authored code, so audit it in the emitted script like any header.
- `extract_to_new_module` — cut the contiguous tail of the source (the moved defs plus the
  scaffolding that leads into them) and write it as a new module, prepending
  `from __future__ import annotations` when the move adds it.
- `extract_symbols_to_new_module` — cut the named defs/classes from **scattered** positions
  (not just a tail) and assemble the new module under an authored `header` reproduced from the
  target; `drop_assigns` deletes a module-level constant that relocated into the header.
- `extract_function` — cut an inline block verbatim, re-indent it under an authored
  `signature` (with an optional `return`), insert the def, and replace the block with an
  authored `call`. Faithful only when the body is unchanged.
- `lower_call_sites` — `Owner.m(receiver, rest)` → `receiver.m(rest)`.
- `requalify_call_sites` — `Owner.m(args)` → `m(args)`.
- `remove_import` — function-scoped or module-level, all occurrences, with the trailing
  blank.
- `remove_imported_name` — drop a single name from a `from m import a, b` (or a plain
  `import x`), realising a lost import directly instead of relying on the formatter to prune
  it (this repo's ruff has no F811).
- `add_import` — let the formatter's import sorter place it; a wholly new module's statement is
  added verbatim from the target so its wrapping is preserved.
- `add_typechecking_import` — append an import inside the destination's `if TYPE_CHECKING:`
  block (a moved annotation needs its type imported there); the sorter orders the block.
- `repath_import` — repath a function-scoped `from old import ... name ...` to
  `from new import ...` in place (module-level repaths fall out of add/remove + the sorter).
- `delete_file` — delete a source module the relocation emptied; refuses a file still
  holding anything beyond a docstring, imports, or a `TYPE_CHECKING` block.

### 3.4 What the proof does **not** certify

- A **rename** (even a privacy flip) and a **statement-level reorder** are not pure
  relocations; the generator infers no recipe and reports them `UNSUPPORTED`. They belong in
  prepare (§2.4) and are reviewed there, not machine-certified.
- A **new-module extract whose symbols are not all top-level in the source** — a method still
  inside a class — is reported `UNSUPPORTED`: prepare must lift it out (de-self) first.
- An **extract from multiple sources into one new file**, and an **inline-block
  extract-function**, are not auto-inferred — split so each new module is filled from a single
  source, or hand-write the `Repro` (compose `extract_function` for a disciplined extraction).
- A **non-mechanical reference update** the move cannot derive — a module path inside a string
  literal — surfaces as a residual; it is a one-line **postpare** (or prepare), not a move.

For an inference gap (not a property violation), write the `Repro` by hand (see
`guide.md`); the same byte-diff then certifies it.

## 4. What a verdict asserts (and does not)

- **Order-aware** — a relocation that reorders lines within the moved block leaves a
  residual.
- **Uniform-indentation-aware** — a whole-block shift by one constant amount reproduces
  cleanly; a non-uniform change leaves a residual, since it can change Python semantics.
- **Blank-line-insensitive** by design (§2.3): the formatter normalises both sides.
- **Requalification / lowering / repath is scoped to symbols relocated in this commit**, so a
  consumer-only call or import rewrite (no relocated definition) cannot reproduce as a move —
  it surfaces as a residual.
- It judges the **shape of a relocation**, not **intent** — a `PASS` says "this commit is
  exactly these relocations", not "this relocation was a good idea". Confirm the commit's
  purpose from its subject before trusting any clean verdict.
