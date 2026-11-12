# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""torchrun check: the DFlash draft all-reduce site's poll deadline (SGLANG_OPT_DFLASH_DRAFT_AR_SITE,
speculative/dflash_draft_ar_site.py) stops the waiting rank instead of letting it spin when a peer is late past it.

    torchrun --nproc-per-node 4 test/registered/spec/dflash/dflash_draft_ar_site_deadline_tp.py --late-ms 100 --deadline-ms 2000
    torchrun --nproc-per-node 4 test/registered/spec/dflash/dflash_draft_ar_site_deadline_tp.py --late-ms 3000 --deadline-ms 500

Rank 0 calls one 8-row site at once; every other rank sleeps --late-ms first. Expected, fixed before any run of this file:
  late < deadline: every rank finishes, every rank holds the same bits, and every mailbox is the sentinel again.
  late > deadline: rank 0's site traps (a CUDA error at its synchronize) no earlier than the deadline and within
    deadline + 2 s of the launch. Printed, not judged: each late rank reads rank 0's own slot in rank 0's mailbox through
    the symmetric-memory peer mapping before its own call (still published = rank 0 stopped before its re-arm), and
    whether its own late site then finishes.
A trap leaves rank 0's CUDA context unusable, so in that mode no rank calls a collective after the site: each rank prints
one DEADLINE_RESULT line and exits with 1 on a failed expectation.
"""

import argparse
import json
import os
import time
import types

import torch
import torch.distributed as dist

from sglang.srt.runtime_context import publish
from sglang.srt.server_args import ServerArgs

ROWS, BLOCK = 8, 8
TRAP_SLACK_S = 2.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--late-ms", type=int, required=True)
    ap.add_argument("--deadline-ms", type=int, required=True)
    a = ap.parse_args()
    local_rank, world = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    publish(ServerArgs(model_path="dummy"), role="tokenizer")
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()

    from sglang.srt.layers.layernorm import RMSNorm
    from sglang.srt.models.dflash import DFlashGroupedConv
    from sglang.srt.speculative import dflash_draft_ar_site as m

    deadline_ns = a.deadline_ms * 1_000_000
    expect_trap = a.late_ms > a.deadline_ms
    group = types.SimpleNamespace(world_size=world, rank_in_group=rank, device_group=dist.group.WORLD)
    sites = m.DraftArSites(group, device, deadline_ns)
    conv = DFlashGroupedConv(m.HIDDEN, BLOCK, m.TAPS, m.GROUP).to(device=device, dtype=torch.bfloat16)
    norm = RMSNorm(m.HIDDEN, eps=1e-6).to(device=device, dtype=torch.bfloat16)
    g = torch.Generator(device=device).manual_seed(11 + rank)
    partial = torch.randn(ROWS, m.HIDDEN, generator=g, device=device).to(torch.bfloat16)
    shared = torch.Generator(device=device).manual_seed(3)
    projection = (0.3 * torch.randn(ROWS, 2 * m.TAPS * m.GROUPS, generator=shared, device=device)).to(torch.bfloat16)
    residual = torch.randn(ROWS, m.HIDDEN, generator=shared, device=device).to(torch.bfloat16)
    m.compile_site(ROWS, world, BLOCK, deadline_ns)
    torch.cuda.synchronize()
    dist.barrier()

    result = {"rank": rank, "late_ms": a.late_ms if rank else 0, "deadline_ms": a.deadline_ms, "expect_trap": expect_trap}
    ok = True
    if rank != 0:
        time.sleep(a.late_ms / 1000)
        if expect_trap:
            try:
                peer = sites.mailbox.handle.get_buffer(0, (2, world, m.CAP, m.HIDDEN), torch.bfloat16)
                words = peer[0, 0, :ROWS].contiguous().view(torch.int32)
                result["rank0_own_slot_published_words"] = int((words != m.SENTINEL).sum())
                result["rank0_own_slot_words"] = words.numel()
            except Exception as e:  # printed only: the mapping's state after a remote trap is not part of the protocol
                result["rank0_slot_read_error"] = f"{type(e).__name__}: {str(e)[:200]}"
    t0 = time.time()
    try:
        out = sites.run(0, partial, projection, conv, residual, norm)
        torch.cuda.synchronize()
        result["finished"], result["finite"] = True, bool(torch.isfinite(out).all())
    except Exception as e:
        result["finished"], result["error"] = False, f"{type(e).__name__}: {str(e)[:300]}"
    result["wall_s"] = round(time.time() - t0, 4)

    if rank == 0 and expect_trap:
        trapped = not result["finished"]
        in_window = a.deadline_ms / 1000 <= result["wall_s"] <= a.deadline_ms / 1000 + TRAP_SLACK_S
        ok = trapped and in_window
        result["verdict"] = "PASS" if ok else ("FAIL: no trap" if not trapped else "FAIL: trap outside [deadline, deadline + 2 s]")
        print("DEADLINE_RESULT " + json.dumps(result), flush=True)
        time.sleep((a.late_ms + 3000) / 1000)  # keep this rank's memory mapped while the late ranks run
        os._exit(0 if ok else 1)
    if expect_trap:
        result["verdict"] = "printed only"
        print("DEADLINE_RESULT " + json.dumps(result), flush=True)
        os._exit(0)
    ok = result["finished"] and result["finite"]
    dist.barrier()
    outs = [torch.empty_like(out) for _ in range(world)]
    dist.all_gather(outs, out)
    res = [torch.empty_like(residual) for _ in range(world)]
    dist.all_gather(res, residual)
    result["ranks_identical"] = all(torch.equal(o.view(torch.int16), out.view(torch.int16)) for o in outs) and all(
        torch.equal(r.view(torch.int16), residual.view(torch.int16)) for r in res
    )
    torch.cuda.synchronize()
    result["rearmed"] = bool((sites.mailbox.buffer.view(torch.int32) == m.SENTINEL).all())
    ok = ok and result["ranks_identical"] and result["rearmed"]
    result["verdict"] = "PASS" if ok else "FAIL"
    print("DEADLINE_RESULT " + json.dumps(result), flush=True)
    dist.barrier()
    dist.destroy_process_group()
    os._exit(0 if ok else 1)


if __name__ == "__main__":
    main()
