# Modify this skill: the engine, the generator, or the spec

- How to change `scripts/mechanical_refactor_reproduction_utils.py` (the proof engine),
  `scripts/mechanical_refactor_proof_generator.py` (the generator),
  `scripts/mechanical_refactor_reproduction_cli.py` (the chain verifier), or the specs —
  without silently weakening the proof.
- Read this **before** editing any file under this skill. The engine is trusted: a wrong
  primitive certifies a non-mechanical commit as clean, and every downstream reviewer
  believes it. Changes here carry a higher bar than ordinary code.

## 1. What has which bar

| File | Role | Bar |
|---|---|---|
| `spec-reproduction-utils.md` | **normative** source of truth (SKILL.md §2): the clean-move property, the whitelist / not-allowed lists, each primitive's contract, the arbiter | any behavior change lands here first |
| `spec-reproduction-cli.md` | **normative** source of truth for the chain verifier: the word rule, the proof obligation, the report, the exit codes | any behavior change lands here first |
| `mechanical_refactor_reproduction_utils.py` | **trusted engine**: the relocation primitives + the arbiter | highest — byte-faithfulness proven by tests |
| `mechanical_refactor_proof_generator.py` | **convenience**: infers a recipe from a diff | lower — may report `RESIDUAL`/`UNSUPPORTED` without compromising trust, but still tested |
| `mechanical_refactor_reproduction_cli.py` | **gatekeeper**: walks a chain, runs the proofs, reports | high — a false chain PASS certifies an unproven commit; classification, resolution, and PASS-criterion behavior proven by tests |
| `guide-*.md`, `SKILL.md` | workflow + file map | kept in sync, never describe behavior the code lacks |

## 2. Cardinal rule — the spec leads, code follows

- `spec-reproduction-utils.md` wins over every other file (SKILL.md §2). Code serves the
  spec, not the reverse.
- A behavior change to a primitive, to §2.1/§2.2 (what counts as a clean move), or to the
  §4 arbiter **must edit the spec in the same commit as the code**. Code and spec never
  diverge across commits.
    - New primitive → add its contract to §3.
    - Changed clean-move boundary → edit §2.1 (allowed) / §2.2 (not allowed).
    - Changed reproduce/diff behavior → edit §4.
- If you discover code and spec already disagree, the spec is authoritative: fix the code
  to match — or, if the spec itself is wrong, change it **deliberately** in one commit with
  the reasoning, not as a silent side effect.

## 3. The faithfulness invariant — never break this

- Every primitive relocates **original source bytes**: AST-located, spliced as the source
  text that was there, **never regenerated**. A byte match after the formatter is the
  *entire* proof — the moment a primitive regenerates instead of splicing, the proof is
  worthless (it can no longer distinguish "moved" from "rewritten to look moved").
- A new or changed primitive **must**:
    - locate its target through the AST, not by string search over source;
    - splice the original bytes (interiors of multi-line strings, comments, a magic
      trailing comma, semicolon-joined statements all survive verbatim);
    - preserve the file's newline style (CRLF round-trips) and UTF-8-byte-accurate columns.
- Do **not** add a primitive that normalizes, reflows, or reformats. Formatting belongs to
  the pre-commit pass in the §4 arbiter, applied to **both** sides; primitives do
  relocation only.
- When the generator cannot infer a move, the answer is a hand-written `Repro`
  (guide-construct-proof §2.3) — **not** loosening a primitive to make it fit.

## 4. Testing rules — the hard bar

- The engine is trusted, so an untested change to it is not acceptable. "It ran once" is
  not a test.
- Run the **full** suite and keep it green:

  ```bash
  cd scripts && uv run --with pytest --python 3.12 python -m pytest tests/ -q
  ```

  Baseline at the time of writing: **188 passed**. Your change must leave the count at or
  above baseline — never delete a case to make the suite pass.
- Layout mirrors the modules; put your test where it belongs:
    - `tests/reproduction_utils/` — one `test_<primitive>.py` per engine primitive.
    - `tests/proof_generator/` — the inference layer (`test_infer_*`, `test_script_and_diff`).
    - `tests/reproduction_cli/` — the chain verifier (classification, proof discovery,
      chain walking, the report).
- A new or changed **primitive** requires, in its `test_<primitive>.py`:
    - a **byte-exact** assertion on the resulting file (compare full bytes, not "contains");
    - at least one **adversarial** case where a regenerating implementation would differ
      from splicing — a comment mid-body, a magic trailing comma, odd indentation, a
      semicolon-joined import, non-ASCII text, or a CRLF file — asserting the original
      bytes survive;
    - the **raise paths** the spec promises: ambiguous anchor raises, missing anchor
      raises, wrong/absent `from_class` raises.
- A change to the **generator** requires a `tests/proof_generator/` case that runs it on a
  synthetic commit and asserts `PASS`, plus one non-move / bundled-change case that asserts
  `RESIDUAL` or `UNSUPPORTED` — so a future regression that makes it "pass" a dirty commit
  is caught.
- A change to the **chain verifier** requires a `tests/reproduction_cli/` case asserting a
  verified chain passes, plus one asserting the broken shape it guards (an unclassified
  commit, a missing proof, a failing proof) still fails — so a regression cannot silently
  green a dirty chain.
- The engine stays **self-contained**: `mechanical_refactor_reproduction_utils.py` imports
  only `git` (via subprocess) and the standard library. Do not add a third-party dependency
  to it.

## 5. Before you commit — checklist

- [ ] `spec-reproduction-utils.md` / `spec-reproduction-cli.md` edited in this same
      commit (if any behavior changed).
- [ ] Full pytest suite green; case count ≥ prior baseline.
- [ ] New/changed primitive has: byte-exact test + ≥1 adversarial (regeneration-would-differ)
      case + the raise-path tests.
- [ ] Generator change has a `PASS` test **and** a `RESIDUAL`/`UNSUPPORTED` test.
- [ ] `SKILL.md` §3 file map and the relevant `guide-*.md` updated for any new file or
      workflow change.
- [ ] `mechanical_refactor_reproduction_utils.py` still imports only git + stdlib.
