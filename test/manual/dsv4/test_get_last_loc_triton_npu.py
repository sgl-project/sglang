"""Discriminating test: get_last_loc_triton vs get_last_loc_torch on NPU.

Reproduces the suspected NPU triton miscompilation with the exact shapes
from the 2026-08-21 DSV4 prefill-CP failure (bs=1, prefix 129024, int32
req_to_token). A mismatch here proves the root cause on a single device;
a clean sweep means the hypothesis is wrong and the hunt reopens.
"""

import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, "python")

from sglang.kernels.ops.memory.common import get_last_loc_triton  # noqa: E402
from sglang.srt.mem_cache.allocation import get_last_loc_torch  # noqa: E402

DEV = "npu:0"
POOL_SLOTS = 453_000  # full pool size in tokens (3544 pages x 128)
MAX_CTX = 140_000

torch.manual_seed(0)


def run_case(name, bs, prefixes, req_slots=None):
    req_to_token = torch.randint(
        1, POOL_SLOTS, (bs, MAX_CTX), dtype=torch.int32, device=DEV
    )
    idx = (
        torch.tensor(req_slots, dtype=torch.int64, device=DEV)
        if req_slots
        else torch.arange(bs, dtype=torch.int64, device=DEV)
    )
    pre = torch.tensor(prefixes, dtype=torch.int64, device=DEV)

    want = get_last_loc_torch(req_to_token, idx, pre).to(torch.int64)
    got = get_last_loc_triton(req_to_token, idx, pre).to(torch.int64)
    ok = torch.equal(want, got)
    print(
        f"[{'OK ' if ok else 'BAD'}] {name}: bs={bs} prefixes={prefixes} "
        f"want={want.tolist()[:6]} got={got.tolist()[:6]}"
    )
    return ok


def main():
    all_ok = True
    # Exact failing shapes from the incident log.
    all_ok &= run_case("incident-final-chunk", 1, [129024])
    all_ok &= run_case("incident-cold-chunk", 1, [7040])
    all_ok &= run_case("incident-radix-hit", 1, [118784])
    all_ok &= run_case("incident-slot15", 1, [129024], req_slots=[15])
    # Sweep batch specializations (next_power_of_2 constexpr).
    for bs in (1, 2, 3, 4, 8, 16):
        prefixes = [118784 + 8192 * i for i in range(bs)]
        all_ok &= run_case(f"sweep-bs{bs}", bs, prefixes)
    # Prefix 0 rows (fresh reqs) and mixed.
    all_ok &= run_case("mixed-zero-prefix", 3, [0, 118784, 0])

    # Randomized stress on the incident shape family.
    for trial in range(50):
        bs = int(torch.randint(1, 5, (1,)).item())
        prefixes = [
            int(torch.randint(0, MAX_CTX // 128, (1,)).item()) * 128 for _ in range(bs)
        ]
        all_ok &= run_case(f"rand-{trial}", bs, prefixes)

    print("RESULT:", "ALL-MATCH (hypothesis NOT confirmed)" if all_ok else "MISMATCH FOUND (miscompile reproduced)")


if __name__ == "__main__":
    main()
