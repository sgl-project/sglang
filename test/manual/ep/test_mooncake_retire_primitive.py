#!/usr/bin/env python3
"""Retire-primitive smoke test on a Mooncake cohort of 4.

Validates the ``try_retire_ranks`` + ``retire_barrier`` +
``retiree_local_cleanup`` primitives without booting a full sglang
scheduler / model. Runs as a raw torchrun-style contract check.

Protocol:
    1. Spawn 4 worker processes wired as a Mooncake-backend WORLD group.
    2. Boot ``ElasticEPStateManager`` with ``max_ep_size=4``.
    3. All 4 ranks call ``retire_barrier()`` then ``try_retire_ranks([3])``.
    4. Rank 3 calls ``retiree_local_cleanup()`` and ``sys.exit(0)``.
    5. Survivors (rank 0/1/2) issue an ``all_reduce`` on the SGLang WORLD
       GroupCoordinator's device-group and assert the result matches the
       expected K=3 cohort sum (i.e. Mooncake honored the flipped mask).

Usage (single node, 4 GPUs, requires Mooncake IB):
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
        MOONCAKE_IB_DEVICE=mlx5_0 \
        python test/manual/ep/test_mooncake_retire_primitive.py

The all_reduce check exercises two properties of the Mooncake mask:

* Mooncake C++ tolerance of the retiree's exit: survivors' post-flip
  ``all_reduce`` must succeed and not hang after rank 3 exits.
* Dynamic (not enqueue-latched) mask reads: survivors' first collective
  after the flip must return the K=3 cohort sum.
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import socket
import sys
import time
from typing import Optional

# Test parameters (kept as module constants so all ranks agree).
COHORT_SIZE = 4
RETIREE_RANK = COHORT_SIZE - 1
MASTER_ADDR_DEFAULT = "127.0.0.1"


def _pick_master_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _init_mooncake_world(rank: int, world_size: int, master_addr: str, master_port: int, ib_device: Optional[str]) -> None:
    """Bring up torch.distributed WORLD + sglang._WORLD on Mooncake backend.

    Mirrors what :func:`sglang.srt.distributed.parallel_state.init_distributed_
    environment` + :func:`init_world_group` do at server boot, minus the
    model-parallel scaffolding we don't need for this contract test.
    """
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    if ib_device is not None:
        os.environ.setdefault("MOONCAKE_IB_DEVICE", ib_device)

    import torch

    from sglang.srt.distributed import parallel_state
    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        init_world_group,
    )

    torch.cuda.set_device(rank)

    distributed_init_method = f"tcp://{master_addr}:{master_port}"

    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=rank,
        distributed_init_method=distributed_init_method,
        backend="mooncake",
        max_world_size=COHORT_SIZE,
    )

    parallel_state._WORLD = init_world_group(
        [list(range(world_size))],
        local_rank=rank,
        backend="mooncake",
        max_world_size=COHORT_SIZE,
    )


def _init_elastic_state(rank: int) -> None:
    from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager
    from sglang.srt.server_args import ServerArgs

    server_args = ServerArgs(
        model_path="/tmp/none",  # unused; state manager only reads flags.
        elastic_ep_backend="mooncake",
        max_ep_size=COHORT_SIZE,
    )
    ElasticEPStateManager.init(server_args)

    inst = ElasticEPStateManager.instance()
    assert inst is not None, "ElasticEPStateManager.init returned None"
    assert inst.effective_ep_size == COHORT_SIZE, (
        f"expected effective_ep_size={COHORT_SIZE}, got {inst.effective_ep_size}"
    )
    assert int(inst.active_ranks.sum().item()) == COHORT_SIZE, (
        f"[rank {rank}] initial active_ranks not fully populated: {inst.active_ranks.tolist()}"
    )


def _worker(rank: int, world_size: int, master_addr: str, master_port: int, ib_device: Optional[str], result_queue: multiprocessing.Queue) -> None:
    try:
        _init_mooncake_world(rank, world_size, master_addr, master_port, ib_device)
        _init_elastic_state(rank)

        from sglang.srt.distributed.parallel_state import get_world_group
        from sglang.srt.elastic_ep.elastic_ep import (
            ElasticEPStateManager,
            retire_barrier,
            retiree_local_cleanup,
            try_retire_ranks,
        )

        import torch

        world = get_world_group()
        pre_mask = ElasticEPStateManager.instance().active_ranks.tolist()

        # Simulate the control-plane sequence that the scheduler tick
        # loop drives in production: everyone hits the barrier, then
        # everyone flips the retiree bit, then retirees exit.
        ElasticEPStateManager.begin_scale(COHORT_SIZE - 1)
        ElasticEPStateManager.mark_draining()
        retire_barrier()

        ElasticEPStateManager.mark_retiring()
        ok = try_retire_ranks([RETIREE_RANK])
        assert ok, f"[rank {rank}] try_retire_ranks returned False"

        post_mask = ElasticEPStateManager.instance().active_ranks.tolist()
        assert post_mask[RETIREE_RANK] == 0, (
            f"[rank {rank}] active_ranks[{RETIREE_RANK}] not flipped: {post_mask}"
        )
        assert sum(post_mask) == COHORT_SIZE - 1, (
            f"[rank {rank}] wrong active count post-flip: {post_mask}"
        )

        if rank == RETIREE_RANK:
            retiree_local_cleanup()
            result_queue.put(("retiree_ok", rank, pre_mask, post_mask))
            sys.exit(0)

        # Give the retiree a moment to actually exit before we post
        # another collective on WORLD. Mooncake's peer watchdog needs
        # observing the exit to update its peer-state cache.
        time.sleep(2.0)

        # Survivor path: assert dist.all_reduce over the SGLang WORLD
        # coordinator's Mooncake device group returns the K=3-cohort sum.
        # Each survivor contributes its own global rank; expected sum is
        # 0 + 1 + 2 = 3. If Mooncake didn't honor the mask flip, this
        # will either hang forever (retiree gone from process table but
        # still expected) or return 0+1+2+3=6 (mask latched at construct
        # time and never updated).
        payload = torch.tensor([float(rank)], device=torch.cuda.current_device())
        torch.distributed.all_reduce(payload, group=world.device_group)
        got = float(payload.item())
        expected = float(sum(r for r in range(COHORT_SIZE) if r != RETIREE_RANK))
        assert got == expected, (
            f"[rank {rank}] all_reduce sum mismatch: got={got} expected={expected} "
            f"post_mask={post_mask}"
        )

        # Second collective to make sure the state is durable across ticks.
        payload.fill_(1.0)
        torch.distributed.all_reduce(payload, group=world.device_group)
        got2 = float(payload.item())
        assert got2 == float(COHORT_SIZE - 1), (
            f"[rank {rank}] second all_reduce mismatch: got={got2} expected={COHORT_SIZE - 1}"
        )

        # Simulate reconfig phase + commit so we exercise the full FSM.
        ElasticEPStateManager.mark_reconfiguring()
        ElasticEPStateManager.commit_scale()
        final_phase = ElasticEPStateManager.get_scale_phase()
        assert final_phase == "serving_shrunk", (
            f"[rank {rank}] expected serving_shrunk after commit, got {final_phase}"
        )
        assert ElasticEPStateManager.get_effective_ep_size() == COHORT_SIZE - 1

        result_queue.put(("survivor_ok", rank, pre_mask, post_mask))
        sys.exit(0)

    except Exception as exc:  # noqa: BLE001 - propagate everything to parent
        import traceback

        result_queue.put(("error", rank, repr(exc), traceback.format_exc()))
        sys.exit(1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ib-device", default=os.environ.get("MOONCAKE_IB_DEVICE"))
    parser.add_argument("--master-addr", default=MASTER_ADDR_DEFAULT)
    args = parser.parse_args()

    if args.ib_device is None:
        print(
            "WARNING: no IB device specified (--ib-device or MOONCAKE_IB_DEVICE).",
            file=sys.stderr,
        )

    ctx = multiprocessing.get_context("spawn")
    result_queue: multiprocessing.Queue = ctx.Queue()
    master_port = _pick_master_port()
    print(f"[main] master port {master_port}, cohort={COHORT_SIZE}, retiree={RETIREE_RANK}")

    procs = []
    for rank in range(COHORT_SIZE):
        p = ctx.Process(
            target=_worker,
            args=(rank, COHORT_SIZE, args.master_addr, master_port, args.ib_device, result_queue),
        )
        p.start()
        procs.append(p)

    deadline = time.monotonic() + 180.0
    results = {}
    while time.monotonic() < deadline and len(results) < COHORT_SIZE:
        try:
            tag, rank, *rest = result_queue.get(timeout=5.0)
        except Exception:
            continue
        results[rank] = (tag, rest)

    ok = True
    for p in procs:
        p.join(timeout=30.0)
        if p.exitcode not in (0, None):
            print(f"[main] rank {p.pid} exit={p.exitcode}", file=sys.stderr)
            ok = False
        if p.is_alive():
            print(f"[main] rank {p.pid} still alive, terminating", file=sys.stderr)
            p.terminate()
            p.join(timeout=5.0)
            ok = False

    for rank in range(COHORT_SIZE):
        if rank not in results:
            print(f"[main] rank {rank} did not report a result", file=sys.stderr)
            ok = False
            continue
        tag, rest = results[rank]
        print(f"[main] rank {rank}: {tag}")
        if tag == "error":
            print(f"  {rest[0]}\n{rest[1]}", file=sys.stderr)
            ok = False

    if ok:
        print("retire primitive: PASS")
        return 0
    print("retire primitive: FAIL", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
