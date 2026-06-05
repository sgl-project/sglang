"""Draft-buffer overflow probe for EAGLE/EAGLE3 topk>1 spec decoding.

Background: the multi-step draft attention packs ``topk`` per-branch sequences
(prefix + draft) into each row of its ``kv_indices`` buffer, so the row must hold
``topk * seq_len``. If the buffer is allocated without the ``topk`` factor (e.g. a
cuda-graph buffer sized ``max_bs * max_context_len`` instead of
``max_bs * topk * max_context_len``) the index-building kernel writes out of bounds
once ``topk * seq_len`` exceeds the row width. That corruption is usually SILENT --
it stomps a neighboring kv-index row (wrong attention, still in-vocab tokens, no
crash) and only sometimes lands on the int64 draft token list, producing an
out-of-vocab token id that crashes the target verify embedding gather. A test that
only checks "server stayed alive" therefore misses most of the damage.

This probe drives draft sequences into the overflow regime (long sequences, and
for page_size>1 the page-boundary residues where num_new_pages>=2). The reliable
detector is the in-kernel size invariant in
``FlashInferMultiStepDraftBackend.common_template`` (it fails fast and
deterministically when a row is too narrow). This kit's job is to make sure some
request actually reaches that regime so the invariant has something to catch.
"""

import requests


def run_draft_kv_overflow_test(
    base_url: str,
    page_size: int,
    num_steps: int,
    max_len: int = 1536,
):
    if page_size > 1:
        # last_page_len in [page-num_steps+1, page-1] -> num_new_pages >= 2 (the
        # holey boundary). Sweep that band at every page multiple up to max_len so
        # the per-row required length keeps growing with seq_len.
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
        # 200 = ok. A scheduler-side size-invariant failure or device assert kills
        # the worker -> the request errors / the connection drops, which surfaces
        # here as a non-200 or a RequestException (caught by the test runner).
        assert res.status_code == 200, (
            f"draft kv-indices overflow probe failed at input_len={n}: "
            f"status={res.status_code} body={res.text[:300]}"
        )

    # Scheduler must still be healthy after the sweep.
    assert (
        requests.get(base_url + "/health_generate", timeout=30).status_code == 200
    ), "scheduler unhealthy after draft kv-indices overflow probe"
