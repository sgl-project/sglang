# Mechanical refactor guide: prep + move + proof

## Part 1 — Split the change: prepare, move, postpare

A "move a method/function from one place to another" change is really **two
operations with different correctness criteria**:

| Operation | What it does | How you check it |
|---|---|---|
| **Semantic reshape** | method → free function or method; `self.X` → a parameter, or `self` retyped to the target class; signature / typing change | behavior unchanged: lint + tests pass |
| **Physical move** | cut from the source, paste into the target, fix imports | the moved body is byte-identical, line for line, and the only other changes are move artifacts (imports, a dropped `@staticmethod`, requalified call sites) |

Put both in one commit and the two criteria contaminate each other: a single hunk then
contains the reshape **and** an indentation shift **and** a cross-file relocation, so
neither a human nor a tool can mechanically confirm "the body that landed is the body
that left" — you have to re-read the logic to be sure.

**Rule:** split a mechanical relocation into up to **three** commits — an optional
**prepare** (a minimal in-place reshape), the **move** (the pure relocation, certified by the
reproduce proof; see `SKILL.md`), and an optional **postpare** (a minimal tail fixup the move
cannot do mechanically, e.g. a module path inside a string literal). Both ends are minimal and
covered by tests; the move carries the bulk. See `spec.md` §2.4. ("prep" below is the
prepare phase.)

**A large semantic refactor is not one of these phases.** Consolidating bookkeeping,
deduplicating logic, restructuring control flow, or redesigning an API is its **own commit**,
reviewed for **equivalence** (tests or a written argument) — never folded into a prepare under
the "small reshape" label. Prepare is for the *minimal* reshape a relocation forces.

**Prep is human-reviewed, so it stays small and relocates nothing.** The code keeps its
place — prep only changes its *shape* (de-self a method, retype `self`) so a human can
eyeball the whole diff. The **move** does all the relocating and is machine-certified;
the verifier forgives only the artifacts a relocation forces (imports, a dropped
`@staticmethod`, requalifying the moved symbol's call sites). Never fold reshape work
into the move to make the verifier pass — if the move's diff has anything outside those
artifacts, the reshape leaked in and belongs back in prep.

The shape of the **prep** commit depends on where the code is going — to a
module-level function (**Case 1**) or onto a class (**Case 2**). The **move** commit is
the same idea in both: a pure relocation whose body is byte-identical.

### Case 1: method → free function (in a module)

#### Commit 1 — prep: de-self in place (no relocation)

Reshape the method **in its original file and position** so it no longer depends on
`self`. The body stays exactly where it is — this is a small, human-reviewed diff:

- `self.X` (read) → pass `X` in as a parameter.
- `self.X = v` (write) → `return v`; the caller assigns. (Or pass an explicit mutable
  object.)
- `self.other_method(...)` → prep that method in the same commit, or inject it as a
  `Callable` argument.
- once `self` is gone, mark it `@staticmethod`; the body **does not move**.
- call site: `self.foo(args)` → `TheClass.foo(args)` (class-qualified).

Qualifying the call site reflects the real fact that `foo` no longer needs an instance.
The decorator and this qualifier are the only relocation artifacts the next commit will
carry, and the verifier whitelists exactly those — so prep can stay this small.

**Check:** lint + tests pass; the diff is just the body reshape plus the call-site
qualifier — and nothing has moved.

#### Commit 2 — move: relocate to the module

Cut the `@staticmethod` block, paste it into the target module, and do only the minimal
sealing:

- drop `@staticmethod`, dedent to module level; the body is **unchanged, line for line**.
- source file: add the import of the moved symbol, and remove any now-unused imports.
- call site: `TheClass.foo(args)` → `foo(args)` (qualifier removed; args untouched).

**Check:** the body is byte-identical and the only other changes are move artifacts —
the dropped decorator, the import, and the requalified call site — so the reproduce proof
(`mechanical_refactor_generate_proof.py <commit>`) reports `PASS`. Cross-check with
`git show <commit> --color-moved=dimmed-zebra --color-moved-ws=allow-indentation-change`,
which marks the whole block as moved.

### Case 2: method → method on a class

Use when the goal is to pull **several methods and the fields they read/write** out
into a new (or existing) class. Here prep does **not** convert every `self.X` into a
parameter — instead it builds the class and switches the **type** of `self`, leaving
the body untouched.

#### Commit 1 — prep: build the class, retype `self`

1. Create the target class with the fields the moved methods touch (a frozen dataclass
   is simplest; drop `frozen` only if the methods mutate fields).
2. Wire an instance into the call path — by composition (`self.component = Target(...)`
   in the source ctor), by constructing it at the call site, or by temporarily holding
   both.
3. Retype each moved method as a `@staticmethod` whose parameter is still named `self`
   but **typed** as the target class — body unchanged:

   ```python
   class Source:
       component: Target

       @staticmethod
       def foo(self: Target) -> None:
           ...  # body still reads self.field_a / self.field_b
   ```

4. caller: `self.foo(...)` → `Source.foo(self.component, ...)`.

The trick is keeping the parameter named `self` and changing only its type. `self` is
an ordinary parameter name in Python, so every `self.X` in the body resolves against
the target class both statically and at runtime (the argument *is* a target-class
instance). Renaming the parameter would force rewriting every `self.X` and destroy the
"body unchanged across both commits" invariant.

**Prep stays minimal.** Do only what is needed to relocate the methods onto the class.
Fancier reshapes — signature redesign, extracting helpers, parameter objects,
mutate→return, method renames, splitting a method, dead-branch removal — belong in
**later, non-mechanical follow-up commits**, never in prep.

**Runtime-mutable state → inject a `Callable` getter (still in prep).** If a moved
method reads state on the source object that changes every step (counters, the current
batch, running stats), inject a `Callable[[], T]` getter into the target ctor and
rewrite the body `self.X` → `self.get_X()`. Do **not** thread it as a per-call keyword
argument, and do **not** reach back into the source object.

```python
class Target:
    def __init__(self, *, static_field, get_running_state: "Callable[[], State]"):
        self.static_field = static_field
        self.get_running_state = get_running_state

    @staticmethod
    def check(self: "Target") -> None:
        running = self.get_running_state()   # was self.running_state
        ...
```

```python
# source ctor
self.component = Target(
    static_field=...,
    get_running_state=lambda: self.running_state,
)
```

Per-call keyword arguments are rejected because they make every call site noisy, make
the component API non-self-contained, and force the caller to remember to thread the
state.

**Check:** lint + tests pass; the body is unchanged; types check (`self: Target`
matches the instance the caller passes).

#### Commit 2 — move: relocate into the class

Cut `foo` into the target class and drop `@staticmethod` (it becomes a normal instance
method); the body is **unchanged, line for line**:

- header `def foo(self: Target)` → `def foo(self)` (inside the class the type can be
  omitted).
- caller: `Source.foo(self.component, ...)` → `self.component.foo(...)`.

**Check:** the body is byte-identical, and the dropped `@staticmethod` and the
`def foo(self: Target)` → `def foo(self)` annotation drop are move artifacts. The caller
changes from `Source.foo(self.component, ...)` to `self.component.foo(...)` — the receiver
moves out of the argument list — which the reproduce proof replays with its `lower_call_sites`
primitive, so the whole commit reports `PASS` (`mechanical_refactor_generate_proof.py
<commit>`). The split still pays off: because prep left the body untouched, the move is a
clean cut/paste.

### Extracting to a new module: the move gathers scattered defs under an authored header

When the destination module does not exist yet, the **move gathers the defs straight from
wherever they sit** in the source — no prep is needed to stage them at the tail first. The
reproduce proof replays this with `extract_symbols_to_new_module`: each def/class is cut from
the source **verbatim** (the byte diff certifies the body), and the new file is assembled under
an **authored header** — the module-level imports, a `logger`, platform constants, an
`if TYPE_CHECKING:` block — reproduced from the target. The header is small authored
boilerplate, the same harmless category as an import; the defs are the proven relocation. A
module-level constant that moved into the header (e.g. `_is_hip = is_hip()`) is dropped from
the source too. So a pure new-module extraction is **one move commit, no prep** (the proof
reports `PASS`). The only thing that lands outside it is a non-mechanical reference the move
cannot derive — e.g. a module path inside a string literal — which is a one-line **postpare**.

If a symbol is **not top-level** in the source (a method still inside a class), prepare must
de-self it out first (Case 1); the proof reports `UNSUPPORTED` until then.

### Extract-function: the bulk goes in the move

Turning an inline block into a new function is an extraction, so its **bulk — the relocated
body — belongs in a certified move**, not buried in a prep. The `extract_function` primitive
cuts the inline block **verbatim** into the new def (the byte diff certifies the body) and
authors only the small interface: the `def` signature, an optional `return`, and the `call`
that replaces the block.

This is faithful **only when the body moves unchanged**. If the extraction also de-selfs
(`self.x` → a parameter), restructures control flow (an `if/elif/else` chain becoming early
`return`s), or folds in a bookkeeping change, those are **semantic** and must be a separate
commit reviewed for equivalence **first** — then the move relocates the now-unchanged body. An
extraction that rewrites the body *as* it extracts (the two entangled) is a semantic commit,
not a certifiable move; do not dress it up as one.

### A move never renames

The moved symbol keeps the **same name on both sides**. A rename — even a privacy flip
(`_foo` → `foo`) — is a separate single-purpose commit *before* the move (rename in place,
update call sites), so the move stays a same-named relocation. A move commit that also
renames cannot be machine-certified and must be split: rename first, then move.

### Anti-pattern: prep adds the body, move deletes it

If the prep commit **adds** a large block to the target file and the move commit
**deletes** the same block from the source, you have reversed the order. In the correct
order, prep leaves the body in the source (it only builds the target skeleton, retypes
the header, and qualifies the caller); the move does the cut/paste. The body should
appear and disappear exactly once — on the move side. Fix a reversed pair by pushing
the "add the body" work out of prep and into the move as a cut/paste.

### When NOT to split (single commit)

- Moving an **already** module-level free function → single move-only commit.
- Pure file rename / whole-file move → single commit.
- Trivial field deletion, or `getattr(obj, "x", ...)` → direct attribute access → single commit.
- A class-internal helper relocated next to another helper in the same module → single commit.

### Which actions are mechanical vs not

Boundary: everything needed to build the component correctly the first time is
mechanical; reshaping it *after* it exists is not.

| Action | Bucket |
|---|---|
| target class skeleton + ctor + fields | mechanical (prep) |
| `@dataclass(frozen=True, slots=True, kw_only=True)` decoration | mechanical (prep) |
| composition wiring (`self.component = Target(...)`) | mechanical (prep) |
| `Callable` getter injection for runtime-mutable state | mechanical (prep) |
| platform conditionals carried along with the body | mechanical (prep / move) |
| cross-file import path rewrites | mechanical (move) |
| field-ownership migration into the component ctor | mechanical (a single pre-step) |
| inlining an `init_*` method body into a ctor | mechanical (a single pre-step) |
| privacy flip (`_x` ↔ `x`) | mechanical (a single rename) |
| signature redesign (new kwargs, changed defaults, positional → kw-only) | **not** mechanical |
| body simplification / dead-branch removal / logic rewrite | **not** mechanical |
| semantic method rename | **not** mechanical |

The smaller prep is, the easier "behavior unchanged" is to confirm. More commits, each
small and independently reviewable, beats one big prep that mixes ten flavors of
semantic change. Review order follows commit order: prep → move → then any
non-mechanical reshapes as separate follow-up commits.

### Naming

A split relocation uses consecutive commits with reserved suffixes:

```
<id>-prepare: <subject>    # optional: minimal in-place reshape (de-self, or retype-self)
<id>-move: <subject>       # pure relocation, certified by the reproduce proof
<id>-postpare: <subject>   # optional: minimal tail fixup (e.g. a string-literal path)
```

The `<phase>:` form is what the range command's `--match -move:` regex keys on.

Both ends are optional and minimal. A large semantic refactor is a separate commit, not one of
these phases. Use a short kebab identifier for `<id>`.

## Part 2 — Prove the move commit is mechanical

The proof that a commit is a pure relocation: regenerate it from the base commit with
faithful AST primitives, run the formatter, and diff byte-for-byte against the target. An
empty diff is the proof. See `SKILL.md` for the principle and `spec.md` for exactly
what counts as a clean move.

### Auto-generate the reproduce script from a commit (primary path)

`mechanical_refactor_generate_proof.py` infers the recipe from a commit's diff and
before-state AST, then emits and runs a standalone, auditable reproduce script — no one
hand-writes it.

```bash
# one commit: print the inferred script and run it
python3 .claude/skills/mechanical-refactor-verify/scripts/mechanical_refactor_generate_proof.py <commit>

# a range: write a self-contained folder (repro_scripts/<sha>.py + output.log + output.html)
python3 .claude/skills/mechanical-refactor-verify/scripts/mechanical_refactor_generate_proof.py \
    <base>..<tip> --match -move: --out repro_out
```

A passing script is the proof; its few primitive calls are what a reviewer audits. The
inference covers:

- a **method moved onto an existing class** — call sites lowered (`Owner.m(recv, …)` →
  `recv.m(…)`), the orphaned local import removed;
- a **method moved to a module-level free function** — call sites requalified
  (`Owner.m(…)` → `m(…)`);
- a **free-function-source move to an existing module** — the call stays bare, callers
  repath their import (`repath_import` for a function-scoped import; a module-level repoint is
  realised as remove-old + add-new);
- a **new-module extract of scattered defs** — the defs are cut from wherever they sit
  (`extract_symbols_to_new_module`) and assembled under an authored header reproduced from the
  target (imports, a logger, constants, a `TYPE_CHECKING` block); a constant that relocated
  into the header is dropped from the source. A contiguous-tail source still uses
  `extract_to_new_module`.

The **module-level import diff is realised directly** from the target — gained names are added
(a wholly new module's statement verbatim, so its wrapping is kept), lost names removed with
`remove_imported_name`. This is deterministic and does not rely on the formatter pruning (this
repo's ruff has no F811).

It reports `UNSUPPORTED` (single-commit mode prints the verdict with the notes and exits
non-zero; range mode records it in `output.log`/`output.html`) — review as prepare, or
hand-write the `Repro` — for:

- a commit that relocates **no definition** — a rename (even a privacy flip `_foo` → `foo`)
  or a statement-level reorder; these are reshapes that belong in prepare;
- a **new-module extract whose symbols are not all top-level in the source** — a method still
  inside a class; prepare must de-self it out first;
- an **extract drawing from more than one source file** into one new module, and an
  **inline-block extract-function** — compose `extract_function` by hand for a disciplined
  extraction (the body must be unchanged; a de-self / restructure is a separate semantic
  commit).

### The `Repro` builder — relocation primitives

When the generator reports `UNSUPPORTED`, compose the transform from the same faithful
primitives by hand. Each does only a relocation-faithful edit -- AST-located, spliced as
original source text (it never changes logic), so a byte match after the formatter certifies the commit is exactly that relocation —
and a bundled change surfaces as a residual diff.

```python
import sys
from pathlib import Path

sys.path.append(".claude/skills/mechanical-refactor-verify/scripts")
from mechanical_refactor_reproduce_utils import Repro

r = Repro(base="<base_sha>", target="<commit>")
# Adapt call sites / repath imports BEFORE moving, so a call to a moved method from inside
# another moved method is lowered while still in the source and travels with the body.
r.lower_call_sites("update_weights_from_ipc", "ModelRunner", paths=["a.py", "b.py"])
r.remove_import("a.py", "from x import ModelRunner", in_function="update_weights_from_ipc")
r.move_symbol("update_weights_from_ipc", src="a.py", dst="dst.py", into_class="WeightUpdater", dedent=0)
r.add_import("dst.py", "import gc")
r.run()   # PASS = byte-identical; otherwise prints the residual
```

Primitives:

- `move_symbol(name, *, src, dst, into_class, from_class, dedent, drop_self_annotation,
  before, leave_delegate, delegate_name)` — cut a def with its decorators, drop its own
  `@staticmethod`/`@classmethod`, shift indentation (a negative `dedent` indents into a
  class), paste at a class end / module level / above the sibling `before`. Same-named defs
  need `from_class`; `leave_delegate` authors a forwarding stub in the source (audit it).
- `extract_to_new_module(src, dst, *, symbols, future_import)` — cut the contiguous tail of
  the source (the moved defs/classes plus the scaffolding that leads into them) and write it
  as a new module, prepending `from __future__ import annotations` when the move adds it.
- `extract_symbols_to_new_module(src, dst, *, symbols, header, order, drop_assigns)` — cut the
  named defs/classes from scattered positions and assemble the new module under an authored
  `header` reproduced from the target; `drop_assigns` deletes a relocated module-level constant
  from the source.
- `extract_function(src, dst, *, name, signature, body, body_indent, call, return_text, before,
  into_class)` — cut an inline `body` verbatim into a new `name` def under an authored
  `signature`/`return`, replacing the block with `call`. Faithful only when the body is
  unchanged; compose it by hand (no auto-inference).
- `lower_call_sites` — `Owner.m(receiver, rest)` → `receiver.m(rest)`.
- `requalify_call_sites` — `Owner.m(args)` → `m(args)`.
- `remove_import` — function-scoped or module-level; matches whole statements (token
  boundaries), removes exactly the matched import even on a semicolon-joined line.
- `delete_file` — delete a module the relocation emptied (refuses live code).
- `remove_imported_name(rel, *, module, name, asname)` — drop a single name from a
  `from m import a, b` (or a plain `import x`), realising a lost import directly.
- `add_import` — the formatter's import sorter places it.
- `add_typechecking_import(rel, import_stmt)` — append an import inside the destination's
  `if TYPE_CHECKING:` block (for a moved annotation's type).
- `repath_import(rel, *, old_module, new_module, name)` — repath a function-scoped
  `from old import … name …` to `from new import …` in place.

### A hand-written transform for a non-relocation mechanical change

For a whole-file split or rename — where no single symbol relocates — write a `transform()`
and call `verify_mechanical_refactor`; the scaffold (worktree, pre-commit on the changed
files, diff, reporting) lives in the skill's utils.

```python
import sys
from pathlib import Path

sys.path.append(".claude/skills/mechanical-refactor-verify/scripts")
from mechanical_refactor_reproduce_utils import verify_mechanical_refactor, git_add_and_commit

BASE_COMMIT = "<base_sha>"
TARGET_COMMIT = "<final_sha>"

def transform(dir_root: Path) -> None:
    source = dir_root / "path/to/source.py"
    lines = source.read_text().splitlines(keepends=True)
    for target_path, start, end in [("path/to/a.py", 1, 50), ("path/to/b.py", 51, 120)]:
        target = dir_root / target_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(lines[start - 1 : end]))
    source.unlink()
    git_add_and_commit("split source.py", cwd=str(dir_root))
    # A rename is just: for each file, write content.replace(OLD, NEW); commit.

if __name__ == "__main__":
    verify_mechanical_refactor(BASE_COMMIT, TARGET_COMMIT, transform)
```

### Make the proof re-runnable by reviewers

Share the whole `--out` folder (a gist, a PR attachment, a repo branch) — it is
self-contained: `repro_scripts/<sha>.py` plus the copied
`mechanical_refactor_reproduce_utils.py` it imports. A generated script resolves that module
relative to its own path, so share the folder, not one raw file (a
`python3 <(curl ...)` process substitution breaks the relative import). Include the
commands in the PR description:

````markdown
## Mechanical move — reproducible

```bash
# from the repo root, with the shared folder unpacked next to it
python3 <folder>/repro_scripts/<sha>.py   # PASS = byte-identical to this commit
```
````

A mechanical PR contains **only** mechanical changes (moves, splits, renames, import fixes,
formatting). Semantic changes go in a separate PR.
