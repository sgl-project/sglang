# Construct a proof

- Two levels, one chapter each: §1 constructs the **proof folder for a whole chain** (and
  publishes it with the PR); §2 constructs the proof for a **single commit** (the
  generator, the hand-written `Repro`, the hand-written transform).
- The property and primitive contracts: `spec-reproduction-utils.md`. Splitting the change
  so the move is provable: `guide-split.md`. Consuming the proof: `guide-verify-proof.md`.

## 1. Construct proofs for a whole chain

### 1.1 Generate the proof folder

```bash
# a range: write a self-contained folder
python3 .claude/skills/mechanical-refactor-verify/scripts/mechanical_refactor_proof_generator.py \
    <base>..<tip> --match -move: --out repro_out
```

- `mechanical_refactor_proof_generator.py` infers each recipe from a commit's diff and
  before-state AST.
- It emits and runs a standalone, auditable script per commit — no one hand-writes it.

### 1.2 The folder product

Self-contained, auditable without the skill installed:

- `repro_scripts/<sha>.py` — one script per commit;
- `output.log` + `output.html` — the verdicts;
- a copy of `mechanical_refactor_reproduction_utils.py` — the scripts' only dependency.

The folder is also the `--proof` input to the chain verifier
(`mechanical_refactor_reproduction_cli.py`, contract in `spec-reproduction-cli.md`).

### 1.3 Publish the proof with the PR

#### 1.3.1 What to share

- Share the scripts **plus** the copied `mechanical_refactor_reproduction_utils.py` — the
  scripts import it, so a lone raw file is not runnable.
- Never a `python3 <(curl ...)` one-liner: process substitution gives the script no real
  directory, so the import breaks.
- Flat layouts work: Python puts the script's own directory on `sys.path`, so the utils
  module can sit either next to the script or one level up (the `--out` layout).

#### 1.3.2 Author: create a gist

```bash
cd repro_out
gh gist create --desc "mechanical-move proof for PR #NNNN" \
    repro_scripts/*.py mechanical_refactor_reproduction_utils.py output.log
# prints https://gist.github.com/<user>/<gist_id> -- put it in the PR description
```

- `gh gist create` flattens paths — fine per §1.3.1.
- Alternatives: a PR attachment (zip the `--out` folder) or a branch holding it.

#### 1.3.3 Reviewer: download and re-run

```bash
gh gist clone <gist_id> /tmp/proof        # or: git clone https://gist.github.com/<gist_id>.git /tmp/proof
cd <repo-root>                            # the run resolves the repo from the cwd
python3 /tmp/proof/<sha>.py               # PASS = byte-identical to this commit
```

- Include exactly these commands in the PR description under a
  "Mechanical move — reproducible" heading.

#### 1.3.4 Keep the PR mechanical — in both directions

- A mechanical PR contains **only** mechanical changes (moves, splits, renames, import
  fixes, formatting). Semantic changes go in a separate PR.
- The dual holds too: a semantic (`non_mechanical_provable`) commit must not swallow a
  provable relocation to skip the proof — split it out and prove it
  (`guide-split.md` §2.2; property: `spec-reproduction-cli.md` §2.1).

## 2. Construct the proof for a single commit

### 2.1 What a proof is

- A runnable script that regenerates the commit from its base with the faithful relocation
  primitives and byte-diffs the result against it.

### 2.2 Auto-generate the script (primary path)

```bash
# one commit: print the inferred script and run it (non-zero exit unless PASS)
python3 .claude/skills/mechanical-refactor-verify/scripts/mechanical_refactor_proof_generator.py <commit>
```

#### 2.2.1 What the inference covers

- **Method → existing class**: call sites lowered (`Owner.m(recv, …)` → `recv.m(…)`), the
  orphaned local import removed.
- **Method → module-level free function**: call sites requalified (`Owner.m(…)` → `m(…)`).
- **Free function → existing module**: the call stays bare; callers repath their import
  (`repath_import` when function-scoped; module-level repoints realised as remove-old +
  add-new).
- **New-module extract of scattered defs**: `extract_symbols_to_new_module` under the
  audited header; a constant that relocated into the header is dropped from the source.
  A contiguous-tail source still uses `extract_to_new_module`.
- **A source file the commit deletes** once its defs relocated: `delete_file`.
- **The module-level import diff**, realised directly from the target: gained names added
  (a wholly new module's statement verbatim, wrapping kept, or one name folded into an
  existing `from module import …` with `add_imported_name`), lost names removed with
  `remove_imported_name`.
- Non-Python files in the commit do not block inference; their diff is noted and left to
  the residual.

#### 2.2.2 What it reports `UNSUPPORTED`

- Single-commit mode prints the verdict with notes and exits non-zero; range mode
  records it in the outputs.
- Review such a commit as prepare, or hand-write the `Repro` (§2.3).
- The cases:
    - **no definition relocated** — a rename (even a privacy flip `_foo` → `foo`) or a
      statement-level reorder; reshapes belong in prepare;
    - **a new-module extract whose symbols are not all top-level in the source** — a
      method still inside a class; prepare must de-self it out first;
    - **an extract drawing from more than one source file**, and an **inline-block
      extract-function** — compose `extract_function` by hand (the body must be unchanged;
      a de-self / restructure is a separate semantic commit).

### 2.3 Hand-write the `Repro` when inference falls short

- Compose the transform from the same primitives (`spec-reproduction-utils.md` §3).
- The same byte-diff then certifies it.

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

### 2.4 A hand-written transform for a non-relocation mechanical change

- For a whole-file split or rename — no single symbol relocates — write a `transform()`
  and call `verify_mechanical_refactor`.
- The scaffold (worktree, pre-commit, diff, reporting) lives in the skill's utils.

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
