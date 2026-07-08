# Verify a proof for a move commit

How the reviewer of a claimed-mechanical commit consumes its proof. The property being
certified and the primitives' contracts are `spec-reproduction-utils.md`; how the proof was
produced (and the folder layout it arrives in) is `guide-construct-proof.md`.

## 1. Re-run it

Run the shared script from the repo root:

```bash
python3 <folder>/repro_scripts/<sha>.py
```

The run is the proof — it replays the primitives from the base commit and byte-diffs
against the target in a throwaway worktree. Do not trust a pasted verdict you did not
re-run.

## 2. Read the verdict

- **PASS** — byte-identical to the target: the commit is exactly the relocations listed in
  the script, nothing else.
- **RESIDUAL** — a non-empty diff: the residual is precisely the bundled non-move change.
  Review it as semantic content; if it is a legitimate tail fixup (a string-literal module
  path, a doc reference), it belongs in a postpare commit, not in the move.
- **UNSUPPORTED** — the generator inferred no recipe (the cases are listed in
  `guide-construct-proof.md`). The commit is not thereby wrong — but it is not
  machine-certified: review it by hand as a prepare-style reshape, or ask the author for a
  hand-written `Repro`.

## 3. Audit the script's authored surfaces

A PASS certifies the relocated bytes; the small **authored** surfaces are reproduced from
the target and must be read by a human. In the script, look at:

- the `header=` of an `extract_symbols_to_new_module` call — the module audits it
  (imports / docstring / TYPE_CHECKING imports / logger / relocated `drop_assigns`
  copies only), so what remains to check is that the relocated assignments *should* move;
- a `leave_delegate=` on a `move_symbol` — the forwarding stub is authored code in the
  source file;
- the `signature=` / `return_text=` / `call=` of an `extract_function` — the new function's
  interface is authored, only its body is certified;
- the `drop_assigns=` list — each named constant leaves the source file.

## 4. Know what a PASS does and does not assert

- Requalification / lowering / repath in a script is tied to symbols the same script
  relocates; a consumer-only call or import rewrite with no relocated definition cannot
  reproduce as a move and would surface as a residual.
- Whatever the repo's pre-commit hooks auto-fix is absorbed on both sides
  (`spec-reproduction-utils.md` §3) — the hook set is part of what you are trusting.
- A PASS judges the **shape of a relocation**, not **intent**: it says "this commit is
  exactly these relocations", not "this relocation was a good idea". Confirm the commit's
  subject matches what the script actually moves before approving.

## Why the mechanism is trustworthy

- It runs the real formatter and compares bytes — there is no diff-shape heuristic to fool
  (`spec-reproduction-utils.md` §3).
- The proof is the few primitive calls in the script; auditing them (plus §3 above) is the
  whole human surface.
- The folder is self-contained and re-runnable by anyone — a CI step or a reviewer —
  without the skill installed.
