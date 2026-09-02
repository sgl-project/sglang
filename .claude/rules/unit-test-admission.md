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

**Distinguishing test — does deletion leave a silent-failure path?** A case
that *looks* like a tautology/mirror is still admissible when it guards a
failure mode no other case covers. The criterion is not "is the code under
test simple?" but "would some regression pass every remaining test if this
case were deleted?"

Keep (bookkeeping, not mirror) when the assertion guards one of:

- An **external-source literal** — a value copied from an outside spec
  (OTel semantic conventions, a protocol field name, a vendor API shape).
  Deleting it removes the only guard against silently copying the spec wrong.
  Example: `assertEqual(SpanAttributes.GEN_AI_LATENCY_E2E, "gen_ai.latency.e2e")`
  stays — the string is dictated by the OTel spec, not by this repo's code.
- A **completeness / negative-branch contract** — "all builtins are
  registered", "a non-matching id does *not* trigger", "the default is
  applied when the input is absent". Even if the code is a one-liner, the
  failure mode is "someone added X without updating Y" or "a predicate
  degraded to always-true". Example: `test_abort_non_matching_rid` (asserts
  an unmatched rid is *not* aborted) stays because no positive-match test
  covers the no-op branch.

Delete (true mirror/tautology) when the assertion merely echoes an
**isolated** implementation output — changing it breaks nothing outside the
line itself, so the test has no independent guard value. Example:
`assertEqual(MixedPrecisionConfig.get_min_capability(),
Fp4Config.get_min_capability())` goes — the source body is literally
`return Fp4Config.get_min_capability()`, and flipping it is an isolated
change that every dependent test catches anyway.

One strong case beats several weak ones: each additional case must guard a
distinct failure mode. Ask "which bug escapes if I delete this case?" -- no
answer means delete it.

Test mechanics (placement, CI registration, fixtures) live in
[`write-sglang-test`](../skills/write-sglang-test/SKILL.md).
