# SPDX-License-Identifier: Apache-2.0
"""Fail-stop of the guarded waits (RFC #43949 section 8) at every named wait site of DeepSeek-V4.1's launches:
a stall injected on one rank, at the first launch after a warm-up step (early) or three launches later (late), stops
every rank within the deadline plus a margin; the stalled wait is the injected one, and each rank's stop reason and
site stay readable after the processes are gone; and no rank completes a step without every peer's data for it.

Two modes. ``launch``: a step is the launch alone; at the waits after the stalled rank's last send (ffn, epoch) its
peers have its data and finish the launch. ``step``: a step is the FFN launch followed by a TP collective, as every
serving step ends with one (the logits gather); then no rank finishes the stalled step at any site, so no token of
it can be sent. A rank that waits in the collective is stopped once the stalled rank has stopped, as SGLang stops
the process tree.

Run directly (no torchrun): python3 test_guard.py [--tp 4] [--sites ...] [--after 0,3] [--mode launch|step]."""

import argparse
import json
import queue
import socket
import sys
import time

import torch
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=1800, suite="nightly-amd-8-gpu-mi35x", nightly=True)

DEADLINE_S = 6.0
MARGIN_S = 90.0  # process exit after the GPU fault, without core dumps
TREE_KILL_S = (
    30.0  # how long a peer may wait in the collective after the stalled rank stopped
)
ROWS = 8
WARM = 1  # normal steps first, so every kernel is loaded before the stalled one
K1_SITES = ("front_flags", "front_qkvp")
# the waits after a rank's last send of a launch: in launch mode its peers already have its data and finish
LAST_EXCHANGE = ("ffn", "epoch")
STALLED_RANK = 1


def _no_core():
    """Make this process non-dumpable (prctl PR_SET_DUMPABLE, 0), so a GPU fault writes no core whatever the host
    core_pattern does; the ranks fault on purpose and this node pipes cores to apport on a shared disk."""
    import ctypes

    ctypes.CDLL(None, use_errno=True).prctl(4, 0, 0, 0, 0)  # PR_SET_DUMPABLE = 4


def _port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def rank_main(rank, tp, port, site, after, kind, mode, handle, out):
    import torch.distributed as dist
    from aiter.ops.flydsl.quick_allreduce_int4_ipc import UncachedIpcHeap
    from test_dsv41_ffn import make_weights, mono_weights, seam_inputs

    from sglang.experimental.monokernels.hip.common.guard import sites
    from sglang.experimental.monokernels.hip.deepseek_v41.gfx950.layer import (
        WAIT_SITES,
    )
    from sglang.experimental.monokernels.hip.deepseek_v41.gfx950.runner import (
        DSV41MonoLayer,
    )

    _no_core()  # the ranks fault on purpose; no core to the shared disk
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=tp
    )
    nccl = dist.new_group(backend="nccl") if mode == "step" else None
    mirror = UncachedIpcHeap.open_mem_handle(handle) + 64 * rank
    fault = (site, after + WARM) if rank == STALLED_RANK else None
    r = DSV41MonoLayer(tp, rank, dist.group.WORLD, dev, DEADLINE_S, fault, mirror)
    r.allocate([ROWS])
    r.build([ROWS], k1=kind == "k1")
    w, _, _ = make_weights(tp, rank, dev, seed=0)
    mw = mono_weights(w)
    del w
    part, res, post, comb, pre = seam_inputs(ROWS, tp, rank, dev)
    k1 = r.kernel(ROWS, "k1") if kind == "k1" else None
    x = torch.zeros(ROWS, 5120, dtype=torch.bfloat16, device=dev)
    pos = torch.arange(ROWS, dtype=torch.int64, device=dev)
    swa = torch.zeros(64, 64 * 584, dtype=torch.uint8, device=dev)
    logits = torch.ones(ROWS, 256, device=dev)
    zeros = {n: torch.zeros(s, dtype=dt, device=dev) for n, s, dt in (
        ("hc", (24, 4 * 5120), torch.float32), ("v", (8192,), torch.float32),
        ("bf", (8192,), torch.bfloat16), ("w8", (8192, 5120), torch.float8_e4m3fn), ("s8", (256, 160), torch.uint8),
    )}  # fmt: skip
    out.put(("sites", rank, list(sites()), list(WAIT_SITES)))
    if nccl is not None:
        dist.all_reduce(logits, group=nccl)  # NCCL's own setup before the timed steps
        torch.cuda.synchronize()
    for step in range(WARM + after + 3):
        out.put(("start", rank, step, time.time()))
        if k1 is not None:
            # the attention launch alone: K1's waits are inside one GPU, so the FFN launch after it carries the stop
            # to the peers; K1's numerics are not checked here
            k1(
                res.data_ptr(), x.data_ptr(), post.data_ptr(), comb.data_ptr(), pre.data_ptr(),
                zeros["hc"].data_ptr(), zeros["v"].data_ptr(), zeros["v"].data_ptr(), zeros["bf"].data_ptr(),
                r.res_mid.data_ptr(), r.post_a.data_ptr(), r.comb_a.data_ptr(), r.pre_a.data_ptr(),
                zeros["w8"].data_ptr(), zeros["s8"].data_ptr(), zeros["bf"].data_ptr(), zeros["bf"].data_ptr(),
                zeros["w8"].data_ptr(), zeros["s8"].data_ptr(), zeros["v"].data_ptr(), pos.data_ptr(),
                pos.data_ptr(), swa.data_ptr(), swa.stride(0), 64, r.q.data_ptr(), r.scratch(ROWS).data_ptr(),
                r.epoch.data_ptr(), stream=torch.cuda.current_stream(),
            )  # fmt: skip
        r.ffn(mw, part, res, post, comb, pre, layer_id=step)
        if nccl is not None:
            # the step's closing collective, behind the launch on this rank's stream
            dist.all_reduce(logits, group=nccl)
        torch.cuda.synchronize()
        out.put(("done", rank, step, time.time()))
    out.put(("finished", rank))


def _attempt(tp, site, after, kind, mode):
    from aiter.ops.flydsl.quick_allreduce_int4_ipc import UncachedIpcHeap

    from sglang.experimental.monokernels.hip.common import guard as g
    from sglang.experimental.monokernels.hip.deepseek_v41.gfx950.layer import (  # noqa: F401
        WAIT_SITES,
    )

    torch.cuda.set_device(0)
    buf = UncachedIpcHeap.alloc_uncached(64 * tp)
    view = torch.as_tensor(
        type("B", (), {"__cuda_array_interface__": {"shape": (64 * tp,), "typestr": "|u1", "data": (buf, False), "version": 3}})(),
        device="cuda:0",
    )  # fmt: skip
    view.zero_()
    torch.cuda.synchronize()
    handle = UncachedIpcHeap.get_mem_handle_bytes(buf)
    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    port = _port()
    procs = [
        ctx.Process(
            target=rank_main, args=(r, tp, port, site, after, kind, mode, handle, out)
        )
        for r in range(tp)
    ]
    for p in procs:
        p.start()
    table, done, stall_at, stalled_gone = {}, {}, None, None
    deadline = time.time() + 900
    stalled_step = WARM + after
    tree_killed = []
    while any(p.is_alive() for p in procs) and time.time() < deadline:
        if (
            stall_at is None and any(p.exitcode not in (None, 0) for p in procs)
        ):  # a rank died before the stalled step: setup failed (a port taken, for example)
            break
        if procs[STALLED_RANK].exitcode is not None and stalled_gone is None:
            stalled_gone = time.time()
        if (
            mode == "step"
            and stalled_gone is not None
            and time.time() - stalled_gone > TREE_KILL_S
        ):
            tree_killed = [r for r, p in enumerate(procs) if p.is_alive()]
            break
        try:
            msg = out.get(timeout=2)
        except queue.Empty:
            continue
        if msg[0] == "sites":
            table[msg[1]] = (msg[2], msg[3])
        elif msg[0] == "start" and msg[1] == STALLED_RANK and msg[2] == stalled_step:
            stall_at = msg[3]
        elif msg[0] == "done":
            done.setdefault(msg[1], []).append(msg[2])
    setup_failed = stall_at is None
    alive = [r for r, p in enumerate(procs) if p.is_alive() and r not in tree_killed]
    for p in procs:
        if p.is_alive():
            p.kill()
        p.join(timeout=30)
    t_end = time.time() if not tree_killed else stalled_gone
    words = view.view(torch.int32).view(tp, 16)[:, :3].tolist()
    UncachedIpcHeap.free_device_mem(buf)
    same_sites = all(
        list(s) == list(g.sites()) and list(w) == list(s) for s, w in table.values()
    )
    reasons = {
        r: g.decode_reason(w[g.MIRROR_DEADLINE] or w[g.MIRROR_PEER])
        for r, w in enumerate(words)
    }
    armed = g.site_name(words[STALLED_RANK][g.MIRROR_ARMED])
    completed = sorted(r for r, steps in done.items() if stalled_step in steps)
    stop_s = (t_end - stall_at) if stall_at else None
    known = set(g.sites())
    peers_may_finish = mode == "launch" and site in LAST_EXCHANGE
    ok = (
        not alive
        and same_sites
        and stall_at is not None
        and stop_s <= DEADLINE_S + MARGIN_S
        and armed == site
        and reasons[STALLED_RANK].get("reason") == "deadline"
        and all(
            reasons[r].get("reason") in ("deadline", "peer stopped")
            and reasons[r].get("site") in known
            for r in range(tp)
            if r not in tree_killed
        )
        and STALLED_RANK not in completed
        and (not completed or peers_may_finish)
    )
    return setup_failed, ok, {
        "tp": tp, "site": site, "after": after, "kind": kind, "mode": mode,
        "stop_s": round(stop_s, 1) if stop_s else None, "armed": armed, "alive_at_timeout": alive,
        "stopped_by_tree_kill": tree_killed, "reasons": reasons, "completed_stalled_step": completed,
        "same_site_ids": same_sites, "exitcodes": [p.exitcode for p in procs],
    }  # fmt: skip


def run_case(tp, site, after, kind, mode):
    """One case; once more on a fresh port when a rank died before the stalled step."""
    for attempt in (1, 2):
        setup_failed, ok, info = _attempt(tp, site, after, kind, mode)
        info["attempt"] = attempt
        if not setup_failed:
            break
    return ok, info


def main():
    from sglang.experimental.monokernels.hip.deepseek_v41.gfx950.layer import (
        WAIT_SITES,
    )

    _no_core()  # the parent spawns the ranks; make it non-dumpable too
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--sites", default=",".join(WAIT_SITES))
    ap.add_argument("--after", default="0,3")
    ap.add_argument("--mode", default="launch", choices=("launch", "step"))
    args = ap.parse_args()
    fails = 0
    for site in args.sites.split(","):
        if site not in WAIT_SITES:
            raise SystemExit(f"unknown wait site {site!r}; sites: {WAIT_SITES}")
        for after in (int(a) for a in args.after.split(",")):
            kind = "k1" if site in K1_SITES else "ffn"
            ok, info = run_case(args.tp, site, after, kind, args.mode)
            fails += not ok
            print(("PASS " if ok else "FAIL ") + json.dumps(info), flush=True)
    print("ALL PASS" if not fails else f"FAILED {fails}", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
