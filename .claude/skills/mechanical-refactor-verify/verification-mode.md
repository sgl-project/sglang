# Mode B — Verify each mechanical commit in a mixed stack

Use when the work is a chain where each commit becomes its own PR, and only some
commits are mechanical (mixed with semantic ones). See `SKILL.md` for when to pick
this mode over reproduce mode. There is no reproduce script; you certify each
mechanical commit directly from its diff.

The exact rule the verifier enforces is specified in `verifier-spec.md`; this page is
the workflow around it.

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
  body dedented to module level or re-indented under another class still counts);
- the **import** lines — the only non-relocated change a clean move may contain;
- any **to review** lines — everything else, the only thing a human must read.

`CLEAN MOVE` means every changed line was either relocated byte-for-byte or an import,
so nothing needs review. Otherwise each to-review line is a non-import change — a
rewritten call site, a dropped decorator, a re-derived constant, or a line that changed
by even one byte — which means the commit is not a pure move. It should be a prep + move
pair (see below), or, if a formatter re-wrapped the relocated lines, verified with
reproduce mode instead.

Optional eyeball cross-check of the same commit:

```bash
git show <commit> --color-moved=dimmed-zebra --color-moved-ws=allow-indentation-change
```

### What `CLEAN MOVE` does and does not assert

- It compares the deleted and added line multisets (indentation ignored), so it
  certifies the body **did not change** during the move. It does **not** check line
  order within the moved block — the `--color-moved` cross-check above catches a
  reordered block.
- It judges the **shape of the diff**, not intent. Because imports are the only
  tolerated leftover, a commit that changes any non-import line cannot read
  `CLEAN MOVE` — but still confirm the commit's purpose from its subject before trusting
  the verdict.

## Step 3: Semantic commits

These get ordinary human review — the verifier does not apply to them. Mixing
mechanical and semantic commits in one branch is fine: each commit is reviewed by the
method that fits it.

Extractions that cannot be a single pure move are split into a prep commit and a move
commit — see "Make extractions verifiable: split prep + move" in `SKILL.md` and the
full philosophy in `prep-and-move.md`.
