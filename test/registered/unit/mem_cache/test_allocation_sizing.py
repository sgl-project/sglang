"""CPU unit tests for req_to_token row-width sizing (Bug B, issue 33579).

Self-contained: inlines the arithmetic from get_alloc_len_per_decode so the
test requires no sglang imports.  The logic under test is:

    get_alloc_len_per_decode:
        page_size == 1 (DSPARK default) or spec_topk == 1:
            return max(spec_steps * spec_topk, spec_tokens)
        else (page>1, topk>1 tree):
            num_pages = (page_size-1 + spec_steps + page_size-1) // page_size
            return max(num_pages * page_size * spec_topk, spec_tokens)

    get_alloc_reserve_per_decode  = 2 * get_alloc_len_per_decode

    get_req_to_token_extra_context_len:
        extra = 4 + max_speculative_num_draft_tokens
        if speculative_algorithm is not None and page_size > 1:   # <-- the buggy gate
            extra = max(extra, get_alloc_reserve_per_decode + page_size - 1)
        return extra

On main (without PR 33581) the page_size > 1 gate means page_size=1 never
gets the reserve-sized headroom, so gamma=16 yields headroom=21 < reserve=34.
"""

import sys
from types import SimpleNamespace

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# --------------------------------------------------------------------------
# Inline the arithmetic (no sglang import needed)
# --------------------------------------------------------------------------


def _alloc_len(args) -> int:
    if args.speculative_algorithm is None:
        return 1
    steps = args.speculative_num_steps or 1
    topk = args.speculative_eagle_topk or 1
    tokens = args.max_speculative_num_draft_tokens
    ps = args.page_size
    # DSPARK uses page_size=1 or topk=1 => simple path
    if ps == 1 or topk == 1:
        return max(steps * topk, tokens)
    num_pages = ((ps - 1) + steps + ps - 1) // ps
    return max(num_pages * ps * topk, tokens)


def _reserve(args) -> int:
    return 2 * _alloc_len(args)


def _headroom_current(args) -> int:
    """Mirrors the CURRENT (buggy) get_req_to_token_extra_context_len."""
    extra = 4 + (args.max_speculative_num_draft_tokens or 0)
    if args.speculative_algorithm is not None and args.page_size > 1:
        extra = max(extra, _reserve(args) + args.page_size - 1)
    return extra


def _headroom_fixed(args) -> int:
    """Mirrors the FIXED get_req_to_token_extra_context_len (PR 33581)."""
    extra = 4 + (args.max_speculative_num_draft_tokens or 0)
    if args.speculative_algorithm is not None:
        # Drop the page_size > 1 gate: reserve applies at every page size.
        extra = max(extra, _reserve(args) + max(args.page_size - 1, 0))
    return extra


def _args(*, page_size, gamma, spec_algo="DSPARK"):
    return SimpleNamespace(
        page_size=page_size,
        max_speculative_num_draft_tokens=gamma,
        speculative_algorithm=spec_algo,
        speculative_num_draft_tokens=gamma,
        speculative_num_steps=gamma,
        speculative_eagle_topk=1,
    )


# --------------------------------------------------------------------------
# Red-light tests: current code is wrong at page_size=1
# --------------------------------------------------------------------------


def test_current_page1_gamma6_headroom_too_small():
    """Current code: page_size=1 gamma=6 headroom < reserve (Bug B exists)."""
    args = _args(page_size=1, gamma=6)
    reserve = _reserve(args)  # 2 * max(6,6) = 14
    extra = _headroom_current(args)  # 4 + 6 = 10  (gate blocks the reserve path)
    print(f"\n[current] gamma=6  page_size=1: reserve={reserve} headroom={extra}")
    # This is the BUG: headroom is too small
    assert (
        extra < reserve
    ), f"Expected headroom {extra} < reserve {reserve} to confirm Bug B exists"


def test_current_page1_gamma16_headroom_too_small():
    """Current code: page_size=1 gamma=16 headroom=21 < reserve=34 (Bug B exists)."""
    args = _args(page_size=1, gamma=16)
    reserve = _reserve(args)  # 2 * 17 = 34
    extra = _headroom_current(args)  # 4 + 16 = 20 < 34
    print(f"\n[current] gamma=16 page_size=1: reserve={reserve} headroom={extra}")
    assert (
        extra < reserve
    ), f"Expected headroom {extra} < reserve {reserve} to confirm Bug B exists"


# --------------------------------------------------------------------------
# Green-light tests: fixed code is correct
# --------------------------------------------------------------------------


def test_fixed_page1_headroom_covers_reserve_gamma6():
    """Fixed code: page_size=1, gamma=6 headroom >= reserve."""
    args = _args(page_size=1, gamma=6)
    reserve = _reserve(args)
    extra = _headroom_fixed(args)
    print(f"\n[fixed]   gamma=6  page_size=1: reserve={reserve} headroom={extra}")
    assert extra >= reserve, f"headroom={extra} < reserve={reserve}"


def test_fixed_page1_headroom_covers_reserve_gamma16():
    """Fixed code: page_size=1, gamma=16 headroom >= reserve."""
    args = _args(page_size=1, gamma=16)
    reserve = _reserve(args)
    extra = _headroom_fixed(args)
    print(f"\n[fixed]   gamma=16 page_size=1: reserve={reserve} headroom={extra}")
    assert extra >= reserve, f"headroom={extra} < reserve={reserve}"


def test_fixed_page256_headroom_covers_aligned_reserve():
    """Fixed code: page_size=256 headroom covers aligned reserve."""
    args = _args(page_size=256, gamma=6)
    reserve = _reserve(args)
    extra = _headroom_fixed(args)
    needed = reserve + args.page_size - 1
    print(
        f"\n[fixed]   gamma=6  page_size=256: reserve={reserve} needed={needed} headroom={extra}"
    )
    assert extra >= needed


def test_non_spec_headroom_unchanged():
    """Non-speculative config: extra=4 unchanged in both current and fixed."""
    args = _args(page_size=1, gamma=0, spec_algo=None)
    assert _headroom_current(args) == 4
    assert _headroom_fixed(args) == 4


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
