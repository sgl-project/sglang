# Construct a proof for a move commit

How the author of a claimed-mechanical commit produces its proof: a runnable script that
regenerates the commit from its base with the faithful relocation primitives and byte-diffs
the result against it. What counts as a clean move — and each primitive's exact contract —
is `spec-reproduction-utils.md`; how to split the change so the move commit is provable is
`guide-split.md`; how a reviewer consumes the proof is `guide-verify-proof.md`.

## Auto-generate the reproduce script (primary path)

`mechanical_refactor_proof_generator.py` infers the recipe from a commit's diff and
before-state AST, then emits and runs a standalone, auditable reproduce script — no one
hand-writes it.

```bash
# one commit: print the inferred script and run it (non-zero exit unless PASS)
python3 .claude/skills/mechanical-refactor-verify/scripts/mechanical_refactor_proof_generator.py <commit>

# a range: write a self-contained folder
python3 .claude/skills/mechanical-refactor-verify/scripts/mechanical_refactor_proof_generator.py \
    <base>..<tip> --match -move: --out repro_out
```

The range product is a self-contained folder, auditable without the skill installed:
`repro_scripts/<sha>.py` (one script per commit), `output.log` + `output.html` (the
verdicts), and a copy of `mechanical_refactor_reproduction_utils.py` — the scripts' only
dependency.

The inference covers:

- a **method moved onto an existing class** — call sites lowered (`Owner.m(recv, …)` →
  `recv.m(…)`), the orphaned local import removed;
- a **method moved to a module-level free function** — call sites requalified
  (`Owner.m(…)` → `m(…)`);
- a **free-function-source move to an existing module** — the call stays bare, callers
  repath their import (`repath_import` for a function-scoped import; a module-level repoint is
  realised as remove-old + add-new);
- a **new-module extract of scattered defs** — the defs are cut from wherever they sit
  (`extract_symbols_to_new_module`) and assembled under the audited header; a constant that
  relocated into the header is dropped from the source. A contiguous-tail source still uses
  `extract_to_new_module`;
- a **source file the commit deletes** once its defs relocated (`delete_file`).

The **module-level import diff is realised directly** from the target — gained names are added
(a wholly new module's statement verbatim, so its wrapping is kept), lost names removed with
`remove_imported_name`. A commit that also touches non-Python files infers normally; the
non-Python diff is noted and left to the residual.

It reports `UNSUPPORTED` — review as prepare, or hand-write the `Repro` — for:

- a commit that relocates **no definition** — a rename (even a privacy flip `_foo` → `foo`)
  or a statement-level reorder; these are reshapes that belong in prepare;
- a **new-module extract whose symbols are not all top-level in the source** — a method still
  inside a class; prepare must de-self it out first;
- an **extract drawing from more than one source file** into one new module, and an
  **inline-block extract-function** — compose `extract_function` by hand for a disciplined
  extraction (the body must be unchanged; a de-self / restructure is a separate semantic
  commit).

## Hand-write the `Repro` when inference falls short

Compose the transform from the same primitives (contracts in `spec-reproduction-utils.md`
§2); the same byte-diff then certifies it.

```python
import sys
from pathlib import Path

sys.path.append(".claude/skills/mechanical-refactor-verify/scripts")
from mechanical_refactor_reproduction_utils import Repro

r = Repro(base="<base_sha>", target="<commit>")
# Adapt call sites / repath imports BEFORE moving, so a call to a moved method from inside
# another moved method is lowered while still in the source and travels with the body.
r.lower_call_sites("update_weights_from_ipc", "ModelRunner", paths=["a.py", "b.py"])
r.remove_import("a.py", "from x import ModelRunner", in_function="update_weights_from_ipc")
r.move_symbol("update_weights_from_ipc", src="a.py", dst="dst.py", into_class="WeightUpdater", dedent=0)
r.add_import("dst.py", "import gc")
r.run()   # PASS = byte-identical; otherwise prints the residual
```

## A hand-written transform for a non-relocation mechanical change

For a whole-file split or rename — where no single symbol relocates — write a `transform()`
and call `verify_mechanical_refactor`; the scaffold (worktree, pre-commit on the changed
files, diff, reporting) lives in the skill's utils.

```python
import sys
from pathlib import Path

sys.path.append(".claude/skills/mechanical-refactor-verify/scripts")
from mechanical_refactor_reproduction_utils import verify_mechanical_refactor, git_add_and_commit

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

## Publish the proof with the PR

Share the whole `--out` folder (a gist, a PR attachment, a repo branch) — it is
self-contained. A generated script resolves `mechanical_refactor_reproduction_utils`
relative to its own path, so share the folder, not one raw file (a `python3 <(curl ...)`
process substitution breaks the relative import). Include the commands in the PR
description:

````markdown
## Mechanical move — reproducible

```bash
# from the repo root, with the shared folder unpacked next to it
python3 <folder>/repro_scripts/<sha>.py   # PASS = byte-identical to this commit
```
````

A mechanical PR contains **only** mechanical changes (moves, splits, renames, import fixes,
formatting). Semantic changes go in a separate PR.
