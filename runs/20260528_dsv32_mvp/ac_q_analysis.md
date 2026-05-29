# AC-Q paired quality smoke — result + analysis (Round 5)

Single-node sequential run (DEC-2): DSA references captured first (DSA up), then DS
compared (DS up), via `test/manual/test_dsv32_quality_smoke.py capture|compare`.
Generation path: `/v1/chat/completions` (chat template applied), temperature 0,
`max_tokens` 256 (smoke) / 16 (NIAH). Artifact: `dsv32_quality_smoke.json`; references:
`dsa_quality_refs.json`.

## Gate verdict

| Gate | Value | Threshold | Verdict |
|------|-------|-----------|---------|
| prefix_match_rate | 0.80 | ≥ 0.80 | PASS |
| **mean_rouge_l** | **0.726** | **≥ 0.85** | **FAIL** |
| niah_mini_recall | 5/5 | ≥ 4/5 | PASS |
| first_8_tokens_divergence | 0 | == 0 | PASS |

**AC-Q overall: FAIL** (the plan makes any single gate below threshold a hard fail).

## Why ROUGE-L missed — benign long-generation drift, not a correctness regression

Per-prompt ROUGE-L is bimodal: **median = 1.000**, mean = 0.726, min = 0.10.

- **13/20 prompts score ≥ 0.89, and 11 of those are exactly 1.00** — every short
  factual answer reproduces DSA verbatim: `William Shakespeare`, `Au`, `Jupiter`,
  `Tokyo`, `Central Processing Unit`, `Pacific Ocean`, `Leonardo da Vinci`, `1024`,
  `299792458`, `2`, `Le chat est sur le tapis.`
- **NIAH-mini recall is 5/5** — DS recalls every needle (ZEBRA-7, MARLIN-42, ORCHID-99,
  GLACIER-13, PHARAOH-88).
- The 7 sub-0.5 prompts are all **open-ended / explanatory** (list primes, complete the
  sequence, moon-landing explanation, boiling-point explanation, 17×23 worked steps, SI
  unit, hexagon). On these, DS and DSA agree on the answer and the first tokens
  (prefix-match and first-8 gates pass), then diverge in wording and length — DS often
  elaborates much longer (e.g. SI-unit prompt: DSA 73 chars vs DS 945 chars; 17×23: DSA
  182 vs DS 890). This is the expected behavior of greedy (temperature-0) decoding under
  two different attention numerics: once a single token differs after the shared prefix,
  the continuations cascade apart. ROUGE-L over a 256-token free-form generation is highly
  sensitive to that cascade and to length mismatch, even when both outputs are correct.

One minor DS-side artifact was observed (prompt 10: DS rendered `\( \times 23 \)`,
dropping the `17` from the LaTeX that DSA kept) — cosmetic, did not affect the numeric
answer. No factual errors were observed in DS outputs.

## Conclusion

DS-on V3.2 reproduces DSA's **answers** faithfully (exact short answers, full NIAH recall,
matching prefixes, no first-8 divergence). The ROUGE-L gate fails only because of benign
temperature-0 decode drift on long free-form explanations, not a quality regression. The
immutable AC defines mean_rouge_l ≥ 0.85 as a hard gate, so AC-Q is recorded as **not met**;
the divergence analysis is surfaced for reconciliation rather than masked, and the
threshold/measurement is NOT altered unilaterally.
