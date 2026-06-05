"""Draft kv_indices overflow probe for EAGLE topk>1 spec decoding.

An undersized draft kv_indices row (e.g. a cuda-graph buffer missing the topk
factor) is written out of bounds and *silently* corrupts memory, so "server
stayed alive" doesn't catch it. This probe drives sequences into the overflow
regime; the deterministic detector is the size invariant in
FlashInferMultiStepDraftBackend.common_template.
"""

import requests


def run_draft_kv_overflow_test(
    base_url: str,
    page_size: int,
    num_steps: int,
    max_len: int = 1536,
):
    if page_size > 1:
        # Sweep the holey page boundary (last_page_len in [page-num_steps+1, page-1],
        # num_new_pages>=2) at every page multiple, so required row length grows with seq.
        lo = max(1, page_size - num_steps + 1)
        residues = list(range(lo, page_size))
        lens = sorted(
            {
                base + r
                for base in range(page_size, max_len, page_size)
                for r in residues
                if base + r <= max_len
            }
        )
    else:
        # page_size == 1: no holey boundary, but topk * seq_len still overflows a
        # narrow row at long sequences. Sweep increasing lengths.
        lens = list(range(256, max_len + 1, 256))

    assert lens, "no probe lengths generated -- check page_size/num_steps/max_len"

    for n in lens:
        res = requests.post(
            base_url + "/generate",
            json={
                "input_ids": [13] * n,
                "sampling_params": {
                    "max_new_tokens": num_steps + 4,
                    "temperature": 0,
                },
            },
            timeout=120,
        )
        # Non-200 / dropped connection = the scheduler hit the size invariant or a
        # device assert, i.e. the bug.
        assert res.status_code == 200, (
            f"draft kv-indices overflow probe failed at input_len={n}: "
            f"status={res.status_code} body={res.text[:300]}"
        )

    # Scheduler must still be healthy after the sweep.
    assert (
        requests.get(base_url + "/health_generate", timeout=30).status_code == 200
    ), "scheduler unhealthy after draft kv-indices overflow probe"
