"""Discriminating test: write_req_to_token_pool_triton vs CPU reference on NPU.

The corrupted row segments ([mf-stage], req_to_token[15, 129024:134020)) are
written by this kernel's extend half. Compare against the exact per-req CPU
loop from allocation.py on the incident shapes, including the pinned
non_blocking pointer-array copy the triton path performs.
"""

import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, "python")

from sglang.kernels.ops.memory.common import (  # noqa: E402
    write_req_to_token_pool_triton,
)

DEV = "npu:0"
MAX_CTX = 140_000
POOL_SLOTS = 453_000


def run_case(name, bs, prefixes, seqs):
    rows = []
    prefix_tensors = []
    for i in range(bs):
        row = torch.zeros(MAX_CTX, dtype=torch.int32, device=DEV)
        pref = torch.randint(1, POOL_SLOTS, (prefixes[i],), dtype=torch.int64, device=DEV)
        row[: prefixes[i]] = pref.to(torch.int32)
        rows.append(row)
        prefix_tensors.append(pref)

    req_to_token = torch.stack(rows)
    pre_t = torch.tensor(prefixes, dtype=torch.int64, device=DEV)
    seq_t = torch.tensor(seqs, dtype=torch.int64, device=DEV)
    ext_t = seq_t - pre_t
    extend_total = int(ext_t.sum().item())
    out_cache_loc = torch.randint(
        1, POOL_SLOTS, (extend_total,), dtype=torch.int64, device=DEV
    )
    req_idx = torch.arange(bs, dtype=torch.int64, device=DEV)

    # Mirror allocation.py's launch, including the pinned async pointer array.
    prefix_pointers = torch.tensor(
        [t.data_ptr() for t in prefix_tensors],
        dtype=torch.uint64,
        pin_memory=True,
    ).to(DEV, non_blocking=True)

    write_req_to_token_pool_triton[(bs,)](
        req_to_token, req_idx, prefix_pointers, pre_t, seq_t, ext_t,
        out_cache_loc, req_to_token.stride(0),
    )
    torch.npu.synchronize()

    ok = True
    pt = 0
    for i in range(bs):
        row = req_to_token[i]
        if not torch.equal(row[: prefixes[i]], prefix_tensors[i].to(torch.int32)):
            print(f"[BAD] {name}: req{i} PREFIX region corrupted")
            ok = False
        ext = out_cache_loc[pt : pt + int(ext_t[i])]
        got = row[prefixes[i] : seqs[i]]
        if not torch.equal(got, ext.to(torch.int32)):
            diff = (got != ext.to(torch.int32)).nonzero().flatten()
            print(
                f"[BAD] {name}: req{i} EXTEND region corrupted "
                f"{diff.numel()}/{ext.numel()} at rel {diff[:4].tolist()} "
                f"got={got[diff[:4]].tolist()} want={ext[diff[:4]].tolist()}"
            )
            ok = False
        pt += int(ext_t[i])
    if ok:
        print(f"[OK ] {name}: bs={bs} prefixes={prefixes}")
    return ok


def main():
    ok = True
    ok &= run_case("incident-final-chunk", 1, [129024], [134020])
    ok &= run_case("incident-alt", 1, [129024], [134036])
    ok &= run_case("cold-first-chunk", 1, [0], [5120])
    ok &= run_case("mid-chunk", 1, [65536], [71680])
    ok &= run_case("radix-hit", 1, [118784], [126976])
    # Repeat the incident shape: a race needs iterations to surface.
    for t in range(20):
        ok &= run_case(f"incident-repeat-{t}", 1, [129024], [134020])
    print("RESULT:", "ALL-MATCH" if ok else "WRITER BROKEN (reproduced)")


if __name__ == "__main__":
    main()
