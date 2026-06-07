# Loop 8 / task7 — AC-1: GLM-5.1 DS-OFF byte-identical token-IDs (live, 8×H200, 2026-06-07)

DSA-native (Double Sparsity disabled) GLM-5.1 decode produces **byte-identical output token IDs** before
vs after all Loop-8 changes, under one fixed tuple — the do-not-regress-the-shipped-model gate (AC-1).
This is a **live** before/after comparison (two server boots), not a code-level argument.

## Verification tuple (AC-1.1)
- **Model:** `/cluster-storage/models/models--zai-org--GLM-5.1-FP8/snapshots/f396cf805182f4ca10fa675e1a99815b3ca384db`
  (`zai-org/GLM-5.1 (FP8)`, `glm_moe_dsa`).
- **Runtime:** TP=8, `kv-cache-dtype fp8_e4m3`, `page-size 64`, `attention_backend='dsa'`,
  CUDA graph ON, radix cache OFF, `disable-overlap-schedule`, `mem-fraction-static 0.8`,
  `--random-seed 20260607`.
- **Sampling:** greedy (`temperature=0`), `max_new_tokens=48`.
- **Prompt set (6):** capital-of-France, first-five-primes, transformer-one-sentence,
  translate-to-French, 17×23, haiku-about-the-ocean (see `capture_ds_off_token_ids.py`).
- **DS state:** `--enable-double-sparsity` ABSENT → `enable_double_sparsity=False`,
  `double_sparsity_config=None`; **0 `double_sparsity bind` lines** in either server log (DS fully inert
  when not requested — AC-1 negative).

## Commits compared
- **Baseline (before Loop-8 source changes):** `d018026f9` = `0063839c2^` (parent of the first Loop-8
  source commit). Run from a git worktree at `/tmp/sglang-baseline` with `PYTHONPATH` pointed at it;
  confirmed `import sglang` resolves to the worktree and its `deepseek_v2.py` has **no** `verify_bind_shapes`
  (pre-gate code).
- **Candidate (after all Loop-8 changes):** HEAD `2eaf3d942`.

## Result — BYTE-IDENTICAL
All 6 prompts produced **equal** 48-token output-ID sequences between baseline and candidate:

| Prompt | tokens | verdict |
|--------|--------|---------|
| "The capital of France is" | 48 | MATCH |
| "List the first five prime numbers:" | 48 | MATCH |
| "Explain in one sentence what a transformer neural network is." | 48 | MATCH |
| "Translate to French: Good morning, how are you?" | 48 | MATCH |
| "Q: What is 17 multiplied by 23? A:" | 48 | MATCH |
| "Write a haiku about the ocean." | 48 | MATCH |

**VERDICT: BYTE-IDENTICAL** (AC-1 PASS). This is consistent with the code: every Loop-8 shared-hook change
is gated under `if use_double_sparsity` (or read only at the DS bind site), so the DS-off native-DSA
forward is unchanged — and the live token-IDs confirm it.

## Artifacts / repro
- Token-ID captures: `runs/20260607_glm51_loop8/glm_dsoff_head_ids.json`,
  `runs/20260607_glm51_loop8/glm_dsoff_baseline_ids.json`.
- Server logs: `runs/20260607_glm51_loop8/glm_dsoff_head.log`,
  `runs/20260607_glm51_loop8/glm_dsoff_baseline.log`.
- Capture tool: `development/loop8/capture_ds_off_token_ids.py` (greedy, `return_logprob=True` →
  `meta_info.output_token_logprobs[i][1]`).
- Repro: boot the launch line above at HEAD, run the capture tool; `git worktree add /tmp/sglang-baseline
  d018026f9`, boot the same launch line with `PYTHONPATH=/tmp/sglang-baseline/python`, run the capture
  tool; diff the two JSON files.

`is_deepseek_dsa(GLM-5.1)` remains True (the server selected `attention_backend='dsa'` unchanged with DS
absent), satisfying the AC-1 positive "DSA path selected unchanged".
