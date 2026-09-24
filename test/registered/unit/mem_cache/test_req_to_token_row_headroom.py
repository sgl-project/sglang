"""The req_to_token row headroom must cover the decode reserve.

`get_req_to_token_extra_context_len` is the one place that sizes the headroom
the req_to_token row keeps beyond `model_config.context_len`. The decode step
over-allocates KV by `get_alloc_reserve_per_decode()`, rounded up to the
allocator's page, and `assign_req_to_token_pool_func` writes the row out to
that length -- so an under-sized headroom spills into the neighbouring row.

Ground truth here is `page_aligned_decode_alloc_lens`, the same function the
decode paths use (eagle_utils / dflash_info_v2 / uno_info), so the assertion
compares the headroom against what an actual decode step asks for rather than
against a restatement of the formula.
"""

import unittest

from sglang.srt.environ import envs
from sglang.srt.mem_cache.allocation_sizing import (
    get_alloc_page_size,
    get_alloc_reserve_per_decode,
    get_req_to_token_extra_context_len,
    page_aligned_decode_alloc_lens,
)
from sglang.srt.model_executor.pool_configurator import compute_swa_request_cap
from sglang.srt.runtime_context import get_context, get_parallel, get_spec
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

CONTEXT_LEN = 4096

# (speculative_algorithm, speculative_num_steps, speculative_eagle_topk,
#  speculative_num_draft_tokens)
SPEC_CONFIGS = [
    ("DSPARK", 5, 1, 17),
    ("EAGLE", 5, 1, 8),
    ("EAGLE", 3, 2, 6),
]
PAGE_SIZES = [2, 8, 64, 256]


class _KV:
    def __init__(self, committed_len: int):
        self.kv_committed_len = committed_len
        self.kv_allocated_len = committed_len


class _Req:
    def __init__(self, committed_len: int):
        self.kv = _KV(committed_len)


def _worst_case_growth() -> int:
    """Largest `nxt - committed` a decode step can ask for.

    Swept over a window of committed lengths wide enough to cover every
    residue modulo the page, since the page rounding is what makes the growth
    exceed the raw reserve.
    """
    page_size = get_alloc_page_size()
    reserve = get_alloc_reserve_per_decode()
    return max(
        page_aligned_decode_alloc_lens(
            [_Req(committed_len)], reserve=reserve, page_size=page_size
        )[1][0]
        - committed_len
        for committed_len in range(CONTEXT_LEN - 2 * page_size, CONTEXT_LEN + 1)
    )


class TestReqToTokenRowHeadroom(CustomTestCase):
    def test_headroom_covers_page_rounded_decode_reserve(self):
        """page_size > 1 rounds the reserve up by up to page_size - 1.

        Only the page>1 / UNO branch is asserted: at page_size == 1 a non-UNO
        algorithm's headroom does not cover the reserve on main today, which is
        tracked separately (#33579) and not what this test locks.
        """
        for algo, steps, topk, tokens in SPEC_CONFIGS:
            for page_size in PAGE_SIZES:
                with (
                    get_context().override_server_args(
                        speculative_algorithm=algo,
                        speculative_num_steps=steps,
                        speculative_eagle_topk=topk,
                        speculative_num_draft_tokens=tokens,
                        page_size=page_size,
                    ),
                    get_parallel().override(attn_dcp_size=1),
                ):
                    extra = get_req_to_token_extra_context_len()
                    needed = _worst_case_growth()
                    self.assertGreaterEqual(
                        extra,
                        needed,
                        f"{algo} steps={steps} topk={topk} tokens={tokens} "
                        f"page_size={page_size}: row headroom {extra} < decode "
                        f"reserve {needed}",
                    )

    def test_uno_headroom_covers_reserve_at_page_size_one(self):
        """UNO's double buffer applies at every page size."""
        with (
            get_context().override_server_args(
                speculative_algorithm="UNO",
                speculative_num_steps=5,
                speculative_num_draft_tokens=8,
                page_size=1,
            ),
            get_parallel().override(attn_dcp_size=1),
        ):
            self.assertGreaterEqual(
                get_req_to_token_extra_context_len(),
                _worst_case_growth(),
            )

    def test_swa_request_cap_uses_the_shared_decode_reserve(self):
        """`compute_swa_request_cap` sizes the whole SWA pool from a worst-case
        per-request footprint, so its spec-v2 branch has to be the same reserve
        the allocator takes.

        Compared against the identical config with spec off: the window,
        eviction and chunked-prefill terms cancel, leaving the reserve swap
        (page_size -> get_alloc_reserve_per_decode) plus the eviction padding's
        draft-token term.
        """
        eviction_interval = max(1, envs.SGLANG_SWA_EVICTION_INTERVAL.get())
        num_reqs = 8
        shared = dict(
            max_running_requests=num_reqs,
            chunked_prefill_size=8192,
            disable_overlap_schedule=False,
            page_size=1,
        )
        with (
            get_context().override_server_args(speculative_algorithm=None, **shared),
            get_parallel().override(attn_dcp_size=1),
        ):
            self.assertIsNone(get_spec().speculative_algorithm)
            cap_plain = compute_swa_request_cap(page_size=1, window=128, attn_dp_size=1)

        for algo, steps, topk, tokens in SPEC_CONFIGS:
            with (
                get_context().override_server_args(
                    speculative_algorithm=algo,
                    speculative_num_steps=steps,
                    speculative_eagle_topk=topk,
                    speculative_num_draft_tokens=tokens,
                    **shared,
                ),
                get_parallel().override(attn_dcp_size=1),
            ):
                reserve = get_alloc_reserve_per_decode()
                cap_spec = compute_swa_request_cap(
                    page_size=1, window=128, attn_dp_size=1
                )
                # Per request: the eviction padding's draft-token term grows
                # from 1 to `tokens`, and decode_alloc swaps page_size for the
                # reserve. Both scale by num_reqs.
                per_request = eviction_interval * (tokens - 1) + (reserve - 1)
                self.assertEqual(cap_spec - cap_plain, num_reqs * per_request)


if __name__ == "__main__":
    unittest.main()
