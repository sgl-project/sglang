"""Shared sglang server pool config used by every SWA e2e canary test.

Centralized so that re-tuning the SWA workload only requires touching one place.

Workload assumptions (mirror what every SWA e2e test sends):

- ``model_mode = "swa"`` → google/gemma-4-E2B-it, SWA window = 1024 tokens.
- ``send_parallel_requests(n=8)`` → 8 concurrent reqs, each with ~7K-token prompt
  (``_LONG_PROMPT_BODY`` after BPE), max_new_tokens=2048.

Per-pool sizing constraints
---------------------------

The SWA model has two KV pools (see ``sglang.srt.mem_cache.swa_memory_pool``):

- FULL pool: ``size = max-total-tokens``. Holds every token for FULL-attention layers
  for the lifetime of the req. Must fit all in-flight req contents + decode tail.
- SWA pool: ``size = max-total-tokens * swa-full-tokens-ratio``. Holds only the
  in-window tokens for SWA layers (older tokens are freed via ``free_swa`` once
  they slide past the window).

Required to avoid server deadlock:

  SWA pool ≥ max concurrent in-flight SWA tokens
           = n_reqs * SWA_window
           = 8 * 1024 = 8192

Required to avoid FULL-pool preempt under our 8 × 7K workload:

  FULL pool ≥ n_reqs * (prompt_tokens + max_new_tokens)
            = 8 * (7000 + 2048) = 72_384
  → max-total-tokens ≥ 72_384

Required to force SWA pool slot REUSE (so swa_full_idx_divergence > 0):

  SWA pool < total SWA writes during prefill
          = n_reqs * prompt_tokens
          = 8 * 7000 = 56_000
  → ratio < 56_000 / max-total-tokens

Chosen values
-------------

- ``--max-total-tokens=81920`` → FULL pool = 81920 (≥ 72_384 needed, with headroom).
- ``--swa-full-tokens-ratio=0.2`` → SWA pool = 16384.
    - ≥ 8192 in-flight → no deadlock.
    - < 56_000 total writes → free-list recycled during prefill → divergence > 0.
    - 2× the in-flight floor → tolerates allocator transient peaks.

If divergence still measures 0 (e.g. allocator's free-list ordering keeps mapping
identity by coincidence), tighten ratio toward 0.15 → SWA pool 12288. Below that,
in-flight headroom shrinks and bursty schedulers can deadlock.
"""

from __future__ import annotations

from typing import Final

SWA_POOL_SERVER_ARGS: Final[tuple[str, ...]] = (
    "--max-total-tokens",
    "81920",
    "--swa-full-tokens-ratio",
    "0.2",
)
