# Mechanical-move verifier — specification (source of truth)

## 1. What this file is

- This file is the **single source of truth** for *what the skill certifies* and *how*.
  The code (`mechanical_refactor_verify_utils.py`, `mechanical_refactor_reproduce_utils.py`,
  `mechanical_refactor_reproduce_gen_utils.py`), the tests, and the prose guides
  (`SKILL.md`, `verification-mode.md`, `reproduce-mode.md`) all implement or describe this
  file. If any of them disagrees with it, **this file wins** and the others are the bug.
- There is **one property** being certified — *a commit is a pure relocation* — and **two
  proofs** of it:
    - **Proof A — reproduce** (§3): regenerate the move from the base commit and diff it
      byte-for-byte. This is the primary, authoritative proof and the one normally run,
      because it emits a self-contained Python script anyone can re-run and audit.
    - **Proof B — inspect** (§4): judge the shape of the commit's diff statically. Fast and
      dependency-free, but conservative — it declines to guess when a formatter rewrapped
      lines, deferring to Proof A.
- A commit is a **clean move** if **either** proof certifies it (§5).

## 2. The property — a "clean move"

> A commit is a **clean move** iff every change it makes is code **relocated in the same
> order** — allowing one **uniform indentation shift** of the whole block — plus a small
> fixed set of **move artifacts**, and nothing else.

### 2.1 Allowed — the whole whitelist

- A line **relocated in order**, modulo one **uniform** leading-indentation shift of the
  whole block (the unavoidable reindent of relocation).
- **Import statements** — added, removed, or repathed; single-line or parenthesised
  multi-line. The symbol's home changed, so importers adjust.
- A one-sided **`@staticmethod` / `@classmethod`** — a method became a free function or
  vice versa.
- A **`self` type annotation dropped** from the moved definition — relocating
  `@staticmethod def foo(self: Target)` into `Target` as `def foo(self)` drops the
  decorator and the now-redundant `self` annotation.
- A **call-site requalification of a moved symbol** — `self.foo(x)` → `foo(x)`, or
  `Old.foo(x)` → `New.foo(x)`: same symbol, same arguments, only the qualifier differs.
- A **call-site lowering of a moved method** — a staticmethod call
  `Owner.method(receiver, rest)` → the instance-method call `receiver.method(rest)` (the
  receiver moves out of the argument list), compared on the whole call expression so a
  formatter's wrapping is tolerated.
- **Blank-line changes** — ignored (§2.3).

### 2.2 Not allowed — the commit is **not** a clean move

- A **reorder** of lines within the moved block.
- A **non-uniform** indentation change (one that is not a single shift of the whole block) —
  it can change Python semantics.
- A **trailing-whitespace** change, an internal-whitespace change, or a **line merge/split**.
- A **changed argument** in an otherwise-requalified call.
- A **call rewrite for a symbol that did not move** in this commit (e.g. pointing a
  consumer at a different implementation) — keeps a consumer-only edit from passing.
- A **signature change** other than dropping the `self` annotation — a real parameter's
  type, name, default, or position changing.
- A **rename** of the moved symbol (even a privacy flip `_foo` → `foo`).
- **Any new module-level scaffolding** the move introduces — a `logger`, a module
  constant, a `TYPE_CHECKING` guard.
- A **constant re-derived** in the destination module (`_flag = compute_flag()`).

The last two are **correct flags, not false positives**: a rename or fresh scaffolding is
reshape work, so it belongs in the **prep** commit, not the move (§2.4, `prep-and-move.md`).

### 2.3 Blank lines are ignored

- A blank line never changes Python behavior, and PEP 8 separator blanks legitimately
  collapse when code is split across files or relocated, so a blank-line-only difference is
  tolerated by both proofs.

### 2.4 Why the bar is this strict — prep is human-reviewed, move is machine-checked

- A behaviour-preserving extraction is **two commits**:
    - a **prep** commit — the in-place reshape (de-self a method, retype `self`, add the new
      module's scaffolding, rename). A human reviews it, so it must be **small** and contain
      **no relocation**: the code stays exactly where it is.
    - a **move** commit — the pure relocation, certified by this spec.
- The whitelist in §2.1 is exactly what a relocation *forces* and a human should not have to
  re-read. It is **not** a licence to fold reshape work into the move. If the move's diff
  contains anything outside the whitelist, the reshape leaked in and belongs in prep.

## 3. Proof A (primary) — reproduce: regenerate the move and byte-diff

Instead of inspecting the diff, **regenerate** the move from the base commit with faithful
AST primitives, run the formatter, and diff byte-for-byte against the target commit. An
empty diff is a machine proof that the commit is exactly that relocation; any residual diff
is a bundled non-move change surfaced for review. This is the proof that handles a commit
the formatter rewrapped, which Proof B cannot.

### 3.1 The auto-generated, self-contained Python script

- The generator is run over a commit range:

  ```
  python3 mechanical_refactor_reproduce_gen_utils.py <base>..<tip> --match -move: --out DIR
  ```

- For each matched commit it:
    - **infers a recipe** from the commit's diff + before-state AST — which symbol moved
      (`src` → `dst`, into which class), which call sites were lowered or requalified, which
      local imports the move orphaned, which module imports the destination gained;
    - **emits** a standalone `repro_scripts/<sha>.py` whose only import is
      `mechanical_refactor_reproduce_utils`;
    - **runs** it — checks out the base in a throwaway worktree, replays the primitives,
      runs `pre-commit` (black / isort / ruff), and byte-diffs against the target;
    - records the verdict.
- The **product** is a self-contained folder, independently auditable without the skill:
    - `repro_scripts/<sha>.py` — one script per commit;
    - `output.log` and `output.html` — the verdicts;
    - a copy of `mechanical_refactor_reproduce_utils.py` — the only dependency.
- **Verdicts**:
    - `PASS` — byte-identical to the target; the commit is certified a clean move.
    - `RESIDUAL` — a non-empty diff; the residual is the bundled non-move change to review.
    - `UNSUPPORTED` — the recipe is not yet inferable (§3.4); hand-write the `Repro`.

### 3.2 Why it is the authoritative proof

- It runs the **real formatter**, so a call the formatter split across an `= (` line, or a
  reflow that left a closing bracket as unchanged context, is reproduced exactly — the case
  Proof B conservatively flags.
- The proof is the **few primitive calls** in the generated script; a reviewer audits those,
  not a hand-written transform.
- It is **self-contained and re-runnable** by anyone (a CI step, a PR reviewer), so the
  certification is not tied to the skill being installed.

### 3.3 The faithful relocation primitives

Each primitive does only a **relocation-faithful, AST-driven** edit and never changes logic,
so a byte match after the formatter certifies the commit is *exactly* that relocation:

- `move_symbol` — cut a `def`/`class` with its decorators, drop `@staticmethod` /
  `@classmethod`, dedent, paste at a class end or at module level.
- `lower_call_sites` — `Owner.m(receiver, rest)` → `receiver.m(rest)`.
- `requalify_call_sites` — `Owner.m(args)` → `m(args)`.
- `remove_import` — function-scoped, all occurrences, with the trailing blank.
- `add_import` — let the formatter's import sorter place it.

### 3.4 What the generator does **not** yet infer (reported `UNSUPPORTED`)

- a **new-file extract** — per `prep-and-move.md` the new module's scaffolding belongs in
  prep, so once the chain is split the move targets an existing module;
- a move whose **source is already a free function** — its callers cannot be inferred from
  the qualifier alone;
- a move that also **renames** — not a pure relocation.

For these, write the `Repro` by hand (see `reproduce-mode.md`).

## 4. Proof B (fast screen) — inspect: judge the diff's shape

A static, dependency-free check (no worktree, no formatter): `verify_move_commit(commit)`
answers one yes/no question from the commit's diff alone. Fast, so it is the first screen
over a stack of commits — but **conservative**: a commit the formatter rewrapped reads
`NEEDS REVIEW` here even when it is a faithful move. That is **not a false positive** — the
inspect path is declining to guess, and the commit should be settled by Proof A.

### 4.1 The verdict

- `CLEAN MOVE` — the property in §2 holds by the diff's shape.
- `NEEDS REVIEW` — anything else (the signature diff is printed).

### 4.2 The rule, precisely

Given a commit (diffed against its first parent):

1. **Collect changed lines, per file.** Take the removed (`-`) and added (`+`) lines, grouped
   by file. Within each file, **cancel** any line that appears byte-for-byte as **both**
   removed and added — that is git's own diff artifact (an unchanged line re-represented when
   nearby lines change), not a relocation; a real relocation keeps its lines because they
   cross files. Aggregate the survivors in patch order. Renames/splits are not followed
   (`-M` off).
2. **Collect import lines.** For every file the commit touches, parse the *before* version
   (at `commit^`) and the *after* version (at `commit`) with `ast`. A line is an **import
   line** if it lies within an `Import` / `ImportFrom` node, so single-line and parenthesised
   multi-line imports are both recognised.
3. **Peel the whitelist, preserving order.** From the surviving lines remove:
    - **imports** (a line whose stripped text is in the parsed import set);
    - **decorators** (a line that is exactly `@staticmethod` or `@classmethod`);
    - **call-site requalifications** — a removed and an added line that become equal after
      dropping a `Qualifier.` prefix before a **moved symbol** (a qualifier must actually be
      present, so a verbatim body line is not consumed). Moved symbols are the `def`/`class`
      names on both sides (ignoring indentation, after the `self`-annotation drop);
    - **call-site lowerings** — a call group starting at `Owner.method(receiver, ...)` whose
      canonical form (whitespace-collapsed, receiver moved to the front, redundant `= (...)`
      wrapper dropped) equals an added `receiver.method(...)` call group. A line-based diff
      cannot reassemble a call the formatter split across an `= (` line or whose closing
      bracket is unchanged context — those still read `NEEDS REVIEW`; settle them with
      Proof A.
4. **Compare the remaining block as an ordered signature.** A block's **signature** is its
   non-blank lines with the block's common leading indent removed, in order, with any `self`
   parameter's type annotation dropped. The removed and added blocks match iff their
   signatures are **equal as sequences**. This absorbs a uniform indentation shift and the
   `self`-annotation drop while preserving relative indentation, trailing whitespace, and
   order.
5. **Verdict.** `CLEAN MOVE` iff the signatures match and at least one line relocated;
   otherwise `NEEDS REVIEW`. A commit whose surviving removed and added lists are both empty
   (a pure rename git records with no content change) is clean too.

## 5. How the two proofs relate

- A commit is a **clean move** if **either** proof certifies it (Proof A `PASS` or Proof B
  `CLEAN MOVE`).
- **Proof B first, Proof A to settle.** Inspect is the cheap screen over a stack; whatever it
  flags `NEEDS REVIEW` is either a real bundled change or a formatter reflow — run Proof A to
  tell which (`PASS` = it was only reflow; `RESIDUAL` = a real change to review).
- **Disagreement resolves to Proof A.** Proof B is a heuristic on diff shape; Proof A runs the
  real formatter and compares bytes, so it is authoritative.
- Both certify the **same property** (§2). Neither judges **intent**: confirm the commit's
  purpose from its subject before trusting any clean verdict.

## 6. What a verdict does and does not assert

- **Order-aware** — a relocation that reorders lines within the moved block is not clean.
- **Uniform-indentation-aware** — a whole-block shift by one constant amount is allowed; a
  non-uniform change is flagged, since it can change Python semantics.
- **Blank-line-insensitive** by design (§2.3).
- **Requalification/lowering is scoped to symbols relocated in this commit**, so a
  consumer-only call rewrite (no relocated definition) cannot pass as a move.
- It judges the **shape of a relocation**, not **intent** — confirm the commit's purpose from
  its subject.
- Proof B classifies imports by **text membership** in the parsed import set; a code line
  whose text coincidentally equals an import line elsewhere would be treated as an import.
  This is rare, and the strictness elsewhere makes it harmless.
