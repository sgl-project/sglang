# Mechanical-move verifier — specification (source of truth)

## 1. What this file is

- This file is the **single source of truth** for *what the skill certifies* and *how*. The
  code (`mechanical_refactor_reproduce_utils.py`, `mechanical_refactor_reproduce_gen_utils.py`),
  the tests, and the prose guides (`SKILL.md`, `how-to-guide.md`, `mental-model-prep-and-move.md`) all
  implement or describe this file. If any of them disagrees with it, **this file wins** and
  the others are the bug.
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
- When the destination module is **new**, the relocated block may include the module
  **scaffolding** the prep commit staged in the source's tail — module-level imports, a
  `logger`, an `if TYPE_CHECKING:` guard, module constants — because the move *relocates*
  that scaffolding, it does not author it (§2.4).
- **Import statements** — added, removed, or repathed; single-line or parenthesised
  multi-line. The symbol's home changed, so importers adjust. A move to a new module may
  also add `from __future__ import annotations`.
- A one-sided **`@staticmethod` / `@classmethod`** — a method became a free function or
  vice versa.
- A **`self` type annotation dropped** from the moved definition — relocating
  `@staticmethod def foo(self: Target)` into `Target` as `def foo(self)` drops the
  decorator and the now-redundant `self` annotation.
- A **call-site requalification of a moved symbol** — `self.foo(x)` → `foo(x)`, or
  `Old.foo(x)` → `New.foo(x)`: same symbol, same arguments, only the qualifier differs.
- A **call-site lowering of a moved method** — a staticmethod call
  `Owner.method(receiver, rest)` → the instance-method call `receiver.method(rest)` (the
  receiver moves out of the argument list).
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
- **New module-level scaffolding the move authors fresh** — a `logger`, a module constant, a
  `TYPE_CHECKING` guard the move *introduces* (as opposed to relocates from the source tail,
  §2.1).
- A **constant re-derived** in the destination module (`_flag = compute_flag()`).

A rename, fresh scaffolding, or a statement reorder is reshape work, so it belongs in the
**prep** commit, not the move (§2.4, `mental-model-prep-and-move.md`). The reproduce proof reports such a
commit as a residual diff (or, when no definition relocated at all, as unsupported) — it does
not certify it.

### 2.3 Blank lines are ignored

- A blank line never changes Python behavior, and PEP 8 separator blanks legitimately
  collapse when code is split across files or relocated. The formatter normalises them on
  both the reproduced and target sides, so a blank-line-only difference cannot appear in the
  byte diff.

### 2.4 Why the bar is this strict — prep is human-reviewed, move is machine-checked

- A behaviour-preserving extraction is **two commits**:
    - a **prep** commit — the in-place reshape (de-self a method, retype `self`, stage the
      new module's scaffolding, rename, reorder statements). A human reviews it, so it must
      be **small** and contain **no cross-file relocation**: the code stays where it is.
    - a **move** commit — the pure relocation, certified by this spec.
- The whitelist in §2.1 is exactly what a relocation *forces* and a human should not have to
  re-read. It is **not** a licence to fold reshape work into the move. If the move's diff
  contains anything outside the whitelist, the reshape leaked in and belongs in prep.
- **Extracting to a new module** is staged this way: prep inlines the future module's whole
  body — scaffolding plus the def — as a trailing block in the source file (a human reviews
  that the body is unchanged and the scaffolding is right); the move then cuts that tail into
  the new file. The new file therefore contains only relocated lines plus `from __future__
  import annotations`, so it stays a clean move.

## 3. The proof — reproduce: regenerate the move and byte-diff

Regenerate the move from the base commit with faithful AST primitives, run the formatter
(pre-commit: black / isort / ruff), and diff byte-for-byte against the target commit. An
empty diff proves the commit is exactly that relocation.

### 3.1 The auto-generated, self-contained Python script

- The generator is run over a commit range:

  ```
  python3 mechanical_refactor_reproduce_gen_utils.py <base>..<tip> --match -move: --out DIR
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

### 3.3 The faithful relocation primitives

Each primitive does only a **relocation-faithful, AST-driven** edit and never changes logic,
so a byte match after the formatter certifies the commit is *exactly* that relocation:

- `move_symbol` — cut a `def`/`class` with its decorators, drop `@staticmethod` /
  `@classmethod`, dedent, paste at a class end or at module level.
- `extract_to_new_module` — cut the contiguous tail of the source (the moved defs plus the
  scaffolding that leads into them) and write it as a new module, prepending
  `from __future__ import annotations` when the move adds it. The formatter sorts the imports
  and normalises blank lines.
- `lower_call_sites` — `Owner.m(receiver, rest)` → `receiver.m(rest)`.
- `requalify_call_sites` — `Owner.m(args)` → `m(args)`.
- `remove_import` — function-scoped or module-level, all occurrences, with the trailing
  blank.
- `add_import` — let the formatter's import sorter place it.
- `add_typechecking_import` — append an import inside the destination's `if TYPE_CHECKING:`
  block (a moved annotation needs its type imported there); the sorter orders the block.
- `repath_import` — repath a function-scoped `from old import ... name ...` to
  `from new import ...` in place (module-level repaths fall out of add/remove + the sorter).

### 3.4 What the proof does **not** certify

- A **rename** (even a privacy flip) and a **statement-level reorder** are not pure
  relocations; the generator infers no recipe and reports them `UNSUPPORTED`. They belong in
  prep (§2.4) and are reviewed there, not machine-certified.
- A **new-file extract whose source is not a staged trailing block** — a method still inside
  the class, or a constant sitting far above the moved defs — is reported `UNSUPPORTED`:
  finish the prep so the whole module body sits together at the source tail first.
- An **extract from multiple sources into one new file** is not yet inferred — split it so
  each new module is filled from a single source, or hand-write the `Repro`.

For an inference gap (not a property violation), write the `Repro` by hand (see
`how-to-guide.md`); the same byte-diff then certifies it.

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
