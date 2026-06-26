# How to prove a commit is a mechanical move

The proof that a commit is a pure relocation: regenerate it from the base commit with
faithful AST primitives, run the formatter, and diff byte-for-byte against the target. An
empty diff is the proof. See `SKILL.md` for the principle and `verifier-spec.md` for exactly
what counts as a clean move.

## Auto-generate the reproduce script from a commit (primary path)

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

It reports `UNSUPPORTED` (review as prepare, or hand-write the `Repro`) for:

- a commit that relocates **no definition** — a rename (even a privacy flip `_foo` → `foo`)
  or a statement-level reorder; these are reshapes that belong in prepare;
- a **new-module extract whose symbols are not all top-level in the source** — a method still
  inside a class; prepare must de-self it out first;
- an **extract drawing from more than one source file** into one new module, and an
  **inline-block extract-function** — compose `extract_function` by hand for a disciplined
  extraction (the body must be unchanged; a de-self / restructure is a separate semantic
  commit).

## The `Repro` builder — relocation primitives

When the generator reports `UNSUPPORTED`, compose the transform from the same faithful
primitives by hand. Each does only a relocation-faithful, AST-driven edit (it never changes
logic), so a byte match after the formatter certifies the commit is exactly that relocation —
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

- `move_symbol` — cut a def with its decorators, drop `@staticmethod`/`@classmethod`, dedent,
  paste at the end of a class or at module level.
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
- `remove_import` — function-scoped or module-level, all occurrences, with the trailing blank.
- `remove_imported_name(rel, *, module, name, asname)` — drop a single name from a
  `from m import a, b` (or a plain `import x`), realising a lost import directly.
- `add_import` — the formatter's import sorter places it.
- `add_typechecking_import(rel, import_stmt)` — append an import inside the destination's
  `if TYPE_CHECKING:` block (for a moved annotation's type).
- `repath_import(rel, *, old_module, new_module, name)` — repath a function-scoped
  `from old import … name …` to `from new import …` in place.

## A hand-written transform for a non-relocation mechanical change

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

## Make the proof re-runnable by reviewers

Put the generated script (or your hand-written one) where a reviewer can run it — attached to
the PR description or a gist — and include the one-click command:

````markdown
## Mechanical move — reproducible

```bash
python3 <(curl -sL <raw_url>)   # PASS = byte-identical to this PR
```
````

A mechanical PR contains **only** mechanical changes (moves, splits, renames, import fixes,
formatting). Semantic changes go in a separate PR.
