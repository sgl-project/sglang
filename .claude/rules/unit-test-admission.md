---
paths:
  - "test/**/*.py"
---

# Unit Test Admission Criteria

Every test case must have a concrete answer to: "what future diff would turn
this case red?" If the only answer is "editing the test itself", delete it.

A new unit test case must fall into one of these categories:

1. **Bug regression.** Guards a bug that actually happened (CI failure, issue,
   incident). Before committing, verify the case fails on the pre-fix code and
   passes on the fix. Describe the bug mechanism in the docstring in black-box
   terms. For concurrency bugs, reproduce the exact interleaving
   deterministically (`create_task` + `sleep(0)` scheduling); do not rely on
   probabilistic stress -- a stress loop that cannot hit the bug even on the
   buggy code has zero guard value.

2. **Derived property.** Pins down a conclusion that required reasoning to
   establish -- boundary/alignment math, invariants, protocol semantics (FIFO
   fairness, idempotency, round-trip). Protects against "looks equivalent"
   rewrites that silently break the derivation.

3. **Critical-path bookkeeping.** Defends conventions that are easy to break by
   forgetting to sync -- registry completeness, field lifecycle, serialization
   compatibility. Enumerating assertions are fine here; the guarded failure
   mode is "someone extended X without updating Y". Example: the ratchet tests
   (`test/registered/unit/test_module_state_ratchet.py`).

Not admissible:

- Happy-path tautologies that re-assert what the implementation trivially does.
- Mirror tests that restate the implementation logic as assertions.
- Probabilistic stress that cannot reproduce the failure it claims to guard.

One strong case beats several weak ones: each additional case must guard a
distinct failure mode. Ask "which bug escapes if I delete this case?" -- no
answer means delete it.

Test mechanics (placement, CI registration, fixtures) live in
[`write-sglang-test`](../skills/write-sglang-test/SKILL.md).
