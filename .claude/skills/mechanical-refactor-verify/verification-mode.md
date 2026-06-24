# Mode B — Verify each mechanical commit in a mixed stack

Use when the work is a chain where each commit becomes its own PR, and only some
commits are mechanical (mixed with semantic ones). See `SKILL.md` for when to pick
this mode over reproduce mode. There is no reproduce script; you certify each
mechanical commit directly from its diff.

## Step 1: Classify each commit

A mechanical *relocation* (function move, file split, module extraction) vs a
*semantic* change (new logic, API / signature redesign, behavior change). Only the
relocations are certified here; semantic commits get ordinary human review.

## Step 2: Certify each relocation commit

```bash
python3 .claude/skills/mechanical-refactor-verify/mechanical_refactor_verify_utils.py move <commit>
```

It reports, for the commit's diff:

- how many lines were **relocated byte-for-byte** (indentation ignored, so a method
  becoming an indentation-shifted free function still counts);
- the **wiring** lines (the new import and the rewritten call sites);
- any **to review** lines — the only thing a human must read.

`CLEAN MOVE` means nothing needs review. Otherwise read the (usually tiny) to-review
set and confirm each line is an equivalent adaptation (for example a call site that
changed only its qualifier, or a constant re-derived in the new module). A line that
is a real behavior change means the commit is not a pure move — it should be a prep +
move pair (see below).

Optional eyeball cross-check of the same commit:

```bash
git show <commit> --color-moved=dimmed-zebra --color-moved-ws=allow-indentation-change
```

### What `CLEAN MOVE` does and does not assert

- It compares the deleted and added line sets (indentation ignored), so it certifies
  the body **did not change** during the move. It does **not** check line order within
  the moved block — the `--color-moved` cross-check above catches a reordered block.
- It judges the **shape of the diff**, not intent. A commit that is not really a move
  but whose few changed lines happen to look move-shaped can read `CLEAN MOVE`. Always
  confirm intent from the commit subject and the wiring lines before trusting it.

## Step 3: Semantic commits

These get ordinary human review — the verifier does not apply to them. Mixing
mechanical and semantic commits in one branch is fine: each commit is reviewed by the
method that fits it.

Extractions that cannot be a single pure move are split into a prep commit and a move
commit — see "Make extractions verifiable: split prep + move" in `SKILL.md` and the
full philosophy in `prep-and-move.md`.
