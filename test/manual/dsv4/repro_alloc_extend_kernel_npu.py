"""Discriminating test: sgl_kernel_npu alloc_extend_kernel vs pytorch reference.

The incident's corrupted req_to_token rows ([mf-stage], seg=[129024,134020))
are written from out_cache_loc produced by this kernel when the chunk closes
<200 pages. Compare against the repo's own reference on the exact shapes.
"""

import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, "python")

from sgl_kernel_npu.mem_cache.allocator import alloc_extend_kernel  # noqa: E402
from sglang.srt.utils import next_power_of_2  # noqa: E402

sys.path.insert(0, "/home/w00889861/dsv4/sgl-kernel-npu/tests/python/sgl_kernel_npu")
from test_alloc_extend_slot import alloc_extend_pytorch  # noqa: E402

DEV = "npu:0"
PAGE = 128
POOL_PAGES = 3544


def run_case(name, pre, seq, last_loc_val, bs=1):
    free_pages = torch.arange(1, 1 + POOL_PAGES, dtype=torch.int64, device=DEV)
    pre_t = torch.tensor([pre] * bs, dtype=torch.int64, device=DEV)
    seq_t = torch.tensor([seq] * bs, dtype=torch.int64, device=DEV)
    last_t = torch.tensor([last_loc_val] * bs, dtype=torch.int64, device=DEV)
    extend_tokens = (seq - pre) * bs

    want, _ = alloc_extend_pytorch(
        pre_t.cpu(), seq_t.cpu(), last_t.cpu(), free_pages.cpu(), PAGE,
        (seq_t - pre_t).cpu(),
    )
    want = want.to(DEV)

    got = torch.empty((extend_tokens,), dtype=torch.int32, device=DEV)
    alloc_extend_kernel[(bs,)](
        pre_t, seq_t, last_t, free_pages, got,
        next_power_of_2(bs), PAGE, next_power_of_2(extend_tokens),
    )
    torch.npu.synchronize()

    want64 = want.to(torch.int64)
    got64 = got.to(torch.int64)
    exact = torch.equal(want64, got64)
    in_range = bool((got64 >= 0).all() and (got64 < POOL_PAGES * PAGE).all())
    uniq = bool(len(torch.unique(got64)) == got64.numel())
    bad = (want64 != got64).nonzero().flatten()
    print(
        f"[{'OK ' if exact and in_range else 'BAD'}] {name}: exact={exact} "
        f"in_range={in_range} unique={uniq} mismatches={bad.numel()}"
        + (f" first_bad_at={bad[:4].tolist()} got={got64[bad[:4]].tolist()} want={want64[bad[:4]].tolist()}" if bad.numel() else "")
    )
    return exact and in_range and uniq


def main():
    ok = True
    # Incident shapes: final small chunk of a 134K request under CP.
    ok &= run_case("incident-final-chunk", 129024, 134020, 129023)
    ok &= run_case("incident-alt", 129024, 134036, 129023)
    ok &= run_case("cold-chunk", 0, 5120, -1)
    ok &= run_case("mid-chunk", 65536, 71680, 65535)
    # Sweep new-page counts around the 200-page kernel cutoff.
    for pages in (1, 2, 8, 39, 40, 64, 100, 150):
        seq = 129024 + pages * PAGE
        ok &= run_case(f"pages-{pages}", 129024, seq, 129023)
    print("RESULT:", "ALL-MATCH" if ok else "KERNEL BROKEN (reproduced)")


if __name__ == "__main__":
    main()
