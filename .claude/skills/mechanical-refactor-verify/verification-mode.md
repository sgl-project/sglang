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
python3 .claude/skills/mechanical-refactor-verify/verify_move.py <commit>
```

It reports, for the commit's diff:

- how many lines were **relocated in order** (the whole block may be shifted by one
  uniform indentation amount, so a method body dedented to module level still counts);
- the **move artifacts** — imports, a dropped `@staticmethod`/`@classmethod`, and call
  sites requalified for a moved symbol — the only non-relocated changes a clean move may
  contain;
- if it is not a clean move, a **signature diff** of the moved block — the lines a human
  must read.

`CLEAN MOVE` means the moved block matched in order under a uniform indent shift and
every other change was one of those artifacts. Otherwise the block diff shows a change
outside the whitelist — a reorder, a non-uniform indent change, a trailing-whitespace
change, a line merge, a changed argument, a re-derived constant, or a call rewrite for a
symbol that did not move — which means the commit is not a pure move. It should be a prep
+ move pair (see below), or, if a formatter re-wrapped the relocated lines, verified with
reproduce mode instead.

Optional eyeball cross-check of the same commit:

```bash
git show <commit> --color-moved=dimmed-zebra --color-moved-ws=allow-indentation-change
```

### What `CLEAN MOVE` does and does not assert

- It compares the moved block as an **ordered sequence** with the block's common indent
  removed, so it certifies the body **did not change** and was **not reordered**, while
  allowing one **uniform** indentation shift. A non-uniform indent change, a
  trailing-whitespace change, or a line merge is flagged.
- **Blank lines are ignored** — they never change behavior and separator blanks collapse
  during a relocation.
- It judges the **shape of the diff**, not intent. The requalification artifact is
  scoped to symbols whose definition moved in this commit, so a consumer-only call
  rewrite cannot pass as a move — but still confirm the commit's purpose from its subject
  before trusting a `CLEAN MOVE`.

## Step 3: Semantic commits

These get ordinary human review — the verifier does not apply to them. Mixing
mechanical and semantic commits in one branch is fine: each commit is reviewed by the
method that fits it.

Extractions that cannot be a single pure move are split into a prep commit and a move
commit — see "Make extractions verifiable: split prep + move" in `SKILL.md` and the
full philosophy in `prep-and-move.md`.
