"""Verify assign_draft_cache_locs_contiguous (triton) vs CPU reference on NPU.

Suspected second NPU-triton miscompile: after target prefill, draft() calls
this kernel to copy draft cache locs from req_to_token[row, seq_len:...]. A
miscompiled load address yields float-bit garbage locs; the subsequent draft
store_cache then scatters bf16 KV far outside the small draft pool, clobbering
req_to_token (the [mf-stage] corruption).
"""

import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, "python")

from sglang.kernels.ops.speculative.cache_locs import (  # noqa: E402
    assign_draft_cache_locs_contiguous,
)

DEV = "npu:0"
POOL_LEN = 140_000
POOL_SLOTS = 453_000
TOPK, STEPS = 1, 3


def cpu_ref(rows, req_idx, seq_lens, bs):
    out = torch.empty((bs * TOPK * STEPS,), dtype=torch.int64, device=DEV)
    for i in range(bs):
        out[i * TOPK * STEPS : (i + 1) * TOPK * STEPS] = rows[
            req_idx[i], seq_lens[i] : seq_lens[i] + TOPK * STEPS
        ].to(torch.int64)
    return out


def run_case(name, bs, seq_len):
    req_to_token = torch.randint(
        1, POOL_SLOTS, (bs + 8, POOL_LEN), dtype=torch.int32, device=DEV
    )
    seq_lens = torch.tensor([seq_len] * bs, dtype=torch.int64, device=DEV)
    req_idx = torch.arange(bs, dtype=torch.int64, device=DEV)

    want = cpu_ref(req_to_token, req_idx, seq_lens, bs)

    got = torch.empty((bs * TOPK * STEPS,), dtype=torch.int64, device=DEV)
    assign_draft_cache_locs_contiguous[(bs,)](
        req_idx, req_to_token, seq_lens, got, POOL_LEN, TOPK, STEPS
    )
    torch.npu.synchronize()

    exact = torch.equal(want, got)
    print(
        f"[{'OK ' if exact else 'BAD'}] {name}: bs={bs} seq={seq_len} "
        f"want={want.tolist()[:6]} got={got.tolist()[:6]}"
    )
    return exact


def main():
    ok = True
    ok &= run_case("incident", 1, 129024)
    ok &= run_case("cold", 1, 5120)
    ok &= run_case("mid", 1, 65536)
    for bs in (1, 2, 4, 8):
        ok &= run_case(f"bs{bs}", bs, 118784)
    print("RESULT:", "ALL-MATCH" if ok else "KERNEL BROKEN (reproduced)")


if __name__ == "__main__":
    main()
