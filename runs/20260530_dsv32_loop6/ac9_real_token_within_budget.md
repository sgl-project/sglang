# AC-9 — within-budget gate asserted from real `usage.prompt_tokens` (live re-run)

The within-budget NIAH gate now decides `within_budget` from the **server-measured
tokenized input length** (`usage.prompt_tokens`), not the 1024/1536 **word-count**
proxy — failing closed if a served prompt omits usage. The **DS-fair gate
definition is unchanged** (`INDEX_TOPK = 2048`, 5 pp DS-vs-DSA recall tolerance,
1024/1536-word within-budget lengths, hard recall gate).

## Harness change (commit `d6e884aa9`)
- `_generate` returns `(text, prompt_tokens)`, reading `usage.prompt_tokens` (chat
  `/v1/chat/completions`) or `meta_info.prompt_tokens` (raw `/generate`); `_as_int_or_none` coerces.
- `_GenAttempt` carries `prompt_tokens`; `_summarize_prompt_tokens` reports
  `(max-over-served input_tokens, usage_missing)` — `usage_missing` is the **fail-closed** signal.
- `_niah_record` computes `within_budget` from the real `input_tokens` (→ `None` /
  fail-closed when usage is missing), and records `input_tokens`, `dsa_input_tokens`,
  `usage_missing`, and the old `within_budget_wordcount_proxy` for the safety diff.
- `test_niah_within_budget` asserts the within-budget premise from real tokens
  (usage present **and** `input_tokens <= INDEX_TOPK`), failing closed otherwise.
- The misleading word-count knob `length_tokens` was renamed `length_words`.

Local dry-run (mock responses) confirmed the parsing + fail-closed logic before hardware.

## Live re-run (real TP=8 hardware, DS int8 @ 0.7 node 0 + DSA-default node 1)
`DS_BASE_URL=http://localhost:30000 DSA_BASE_URL=http://10.220.51.5:30000`
`python3 -m pytest test/manual/test_double_sparsity_v32.py -k within_budget` → **1 passed, 2 subtests passed (26.5 s)**.

| length_words | **input_tokens** (real, `usage.prompt_tokens`) | `within_budget` (real) | word-count proxy | DS recall | DSA recall | Δ | verdict |
|---:|---:|:--:|:--:|---:|---:|---:|:--:|
| 1024 | **1128** | ✅ True (≤ 2048) | True | 100.0% | 100.0% | 0.0 pp | PASS |
| 1536 | **1678** | ✅ True (≤ 2048) | True | 100.0% | 100.0% | 0.0 pp | PASS |

- `usage_missing = False` at both lengths (usage present on every served prompt — not fail-closed).
- DS `input_tokens` == DSA `input_tokens` (same tokenizer): 1128 / 1678.

## Was the word-count proxy safe? (the required diff)
**Yes.** At both within-budget lengths the real tokenized length (1128 / 1678) is below
`INDEX_TOPK = 2048`, and the real-token `within_budget` **matches** the old
word-count proxy (both True) — recorded per-length as `within_budget` vs
`within_budget_wordcount_proxy` in `ac9_within_budget/ac12_niah_{1024,1536}_*.json`.
So the 1024/1536-word within-budget set was a safe proxy; no length needed to move
to the characterization set. The gate now asserts this from real tokens and would
**fail closed** if a future server omitted usage or if a length's real tokens
exceeded the budget.

## Artifacts
- `ac9_within_budget/ac12_niah_1024_20260530T115708Z.json`, `ac12_niah_1536_20260530T115720Z.json`
  (per-length: `input_tokens`, `dsa_input_tokens`, `usage_missing`, `within_budget`,
  `within_budget_wordcount_proxy`, recall, verdict).
