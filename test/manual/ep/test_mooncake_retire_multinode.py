#!/usr/bin/env python3
"""Multi-node retire-primitive contract test.

Multi-node counterpart to ``test_mooncake_retire_primitive.py``.
Exercises ONLY the mask-flip layer (``try_retire_ranks`` +
``retire_barrier`` + ``retiree_local_cleanup``) across two physical
nodes with no sglang scheduler / model / MoE dispatcher in the loop.

One process per GPU is launched by the multi-node driver script.
Each rank reads ``SLURM_PROCID`` / ``SLURM_NTASKS`` / ``SLURM_LOCALID``
(or the ``RANK`` / ``WORLD_SIZE`` / ``LOCAL_RANK`` torchrun equivalents),
brings up a Mooncake WORLD, and validates:

  1. retire_barrier() completes across nodes.
  2. try_retire_ranks() flips the mask in every rank's local tensor.
  3. Retirees ``os._exit(0)``, survivors keep serving collectives.
  4. Post-flip all_reduce returns the survivor-only sum -- proving
     Mooncake honors the mask across the RDMA fabric.

Exit code 0 on PASS, non-zero on FAIL.
"""

from __future__ import annotations

import os
import socket
import sys
import time
from typing import Optional


def _env_int(name: str, default: Optional[int] = None) -> int:
    val = os.environ.get(name)
    if val is None or val == "":
        if default is None:
            raise RuntimeError(f"missing required env {name}")
        return default
    return int(val)


def _resolve_rank_topology() -> tuple[int, int, int, str, int]:
    """Return (rank, world_size, local_rank, master_addr, master_port).

    Accepts Slurm ``SLURM_PROCID`` / ``SLURM_NTASKS`` / ``SLURM_LOCALID``
    or the equivalent ``RANK`` / ``WORLD_SIZE`` / ``LOCAL_RANK`` set by
    torchrun.
    """
    if "SLURM_PROCID" in os.environ and "RANK" not in os.environ:
        rank = _env_int("SLURM_PROCID")
        world_size = _env_int("SLURM_NTASKS")
        local_rank = _env_int("SLURM_LOCALID")
    else:
        rank = _env_int("RANK")
        world_size = _env_int("WORLD_SIZE")
        local_rank = _env_int("LOCAL_RANK", 0)

    master_addr = os.environ.get("MASTER_ADDR")
    if master_addr is None:
        raise RuntimeError("MASTER_ADDR must be set (node hosting global rank 0)")
    master_port = _env_int("MASTER_PORT")
    return rank, world_size, local_rank, master_addr, master_port


def _init_mooncake_world(
    rank: int,
    world_size: int,
    local_rank: int,
    master_addr: str,
    master_port: int,
    max_world_size: int,
    ib_device: Optional[str],
) -> None:
    """Minimal WORLD bring-up over the Mooncake backend for the primitive
    contract test -- no model-parallel subgroups, no scheduler wiring."""
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)
    if ib_device is not None:
        os.environ.setdefault("MOONCAKE_IB_DEVICE", ib_device)

    import torch

    from sglang.srt.distributed import parallel_state
    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        init_world_group,
    )

    torch.cuda.set_device(local_rank)

    distributed_init_method = f"tcp://{master_addr}:{master_port}"
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method=distributed_init_method,
        backend="mooncake",
        max_world_size=max_world_size,
    )
    parallel_state._WORLD = init_world_group(
        list(range(world_size)),
        local_rank=local_rank,
        backend="mooncake",
    )


def _init_elastic_state(cohort_size: int) -> None:
    """Build ``ElasticEPStateManager`` state directly, bypassing
    ``ServerArgs`` (which would try to load a HF model config)."""
    from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager

    inst = ElasticEPStateManager._build_state(ep_size=cohort_size, device=None)
    inst.effective_ep_size = cohort_size
    inst.original_ep_size = cohort_size
    ElasticEPStateManager._instance = inst

    assert inst.active_ranks.numel() == cohort_size, (
        f"[bootstrap] active_ranks size {inst.active_ranks.numel()} != "
        f"cohort {cohort_size}"
    )
    assert int(inst.active_ranks.sum().item()) == cohort_size, (
        "[bootstrap] active_ranks not fully populated at init: "
        f"{inst.active_ranks.tolist()}"
    )


def _log(rank: int, hostname: str, *fields: object) -> None:
    parts = " ".join(str(f) for f in fields)
    print(f"[rank {rank}@{hostname}] {parts}", flush=True)


def main() -> int:
    rank, world_size, local_rank, master_addr, master_port = _resolve_rank_topology()

    # Retire the last ``num_retirees`` global ranks -- on a 2-node
    # 8-rank layout this leaves both retirees on node 1, mirroring
    # the MC05 shrink pattern.
    num_retirees = _env_int("MSMN_NUM_RETIREES", 2)
    retirees = list(range(world_size - num_retirees, world_size))
    cohort_size = world_size

    hostname = socket.gethostname()
    ib_device = os.environ.get("MOONCAKE_IB_DEVICE") or os.environ.get(
        "MSMN_IB_DEVICE"
    )

    _log(rank, hostname, f"boot world={world_size} local={local_rank} "
         f"master={master_addr}:{master_port} ib={ib_device} "
         f"retirees={retirees}")

    try:
        _init_mooncake_world(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            master_addr=master_addr,
            master_port=master_port,
            max_world_size=cohort_size,
            ib_device=ib_device,
        )
    except Exception as exc:  # noqa: BLE001
        import traceback

        _log(rank, hostname, "ERROR during mooncake world init:", repr(exc))
        traceback.print_exc()
        return 2

    try:
        _init_elastic_state(cohort_size)
    except Exception as exc:  # noqa: BLE001
        import traceback

        _log(rank, hostname, "ERROR during elastic state init:", repr(exc))
        traceback.print_exc()
        return 3

    import torch

    from sglang.srt.distributed.parallel_state import get_world_group
    from sglang.srt.elastic_ep.elastic_ep import (
        ElasticEPStateManager,
        retire_barrier,
        retiree_local_cleanup,
        try_retire_ranks,
    )

    world = get_world_group()
    pre_mask = ElasticEPStateManager.instance().active_ranks.tolist()
    _log(rank, hostname, f"pre-flip active_ranks={pre_mask}")

    try:
        # Pre-flip sanity: every rank sums its own rank id -> 0+1+...+N-1.
        payload = torch.tensor([float(rank)], device=torch.cuda.current_device())
        torch.distributed.all_reduce(payload, group=world.device_group)
        pre_sum = float(payload.item())
        expected_pre = float(sum(range(cohort_size)))
        assert pre_sum == expected_pre, (
            f"[rank {rank}] pre-flip all_reduce mismatch: got {pre_sum} "
            f"expected {expected_pre}"
        )
        _log(rank, hostname, f"pre-flip all_reduce OK sum={pre_sum}")

        # Collective barrier before the mask flip; survivors + retirees.
        ElasticEPStateManager.begin_scale(cohort_size - num_retirees)
        ElasticEPStateManager.mark_draining()
        retire_barrier()
        _log(rank, hostname, "retire_barrier OK")

        # Flip active_ranks[retirees]=0 on every launch-time group.
        ElasticEPStateManager.mark_retiring()
        ok = try_retire_ranks(retirees)
        assert ok, f"[rank {rank}] try_retire_ranks returned False"

        post_mask = ElasticEPStateManager.instance().active_ranks.tolist()
        for r in retirees:
            assert post_mask[r] == 0, (
                f"[rank {rank}] active_ranks[{r}] not flipped: {post_mask}"
            )
        assert sum(post_mask) == cohort_size - num_retirees
        _log(rank, hostname, f"post-flip active_ranks={post_mask}")

        # Retirees exit with os._exit to avoid Python atexit handlers
        # dragging Mooncake's C++ destructors into a live CUDA context.
        if rank in retirees:
            retiree_local_cleanup()
            _log(rank, hostname, "retiree_local_cleanup OK, exiting")
            os._exit(0)

        # Give Mooncake's peer-state watchdog a chance to observe the
        # cross-node retiree TCP disconnect before the next collective.
        time.sleep(3.0)

        # Post-flip: Mooncake must skip retirees, even across nodes.
        payload.fill_(float(rank))
        torch.distributed.all_reduce(payload, group=world.device_group)
        post_sum = float(payload.item())
        expected_post = float(sum(r for r in range(cohort_size) if r not in retirees))
        assert post_sum == expected_post, (
            f"[rank {rank}] survivor all_reduce mismatch: got {post_sum} "
            f"expected {expected_post} (retirees={retirees})"
        )
        _log(rank, hostname, f"post-flip all_reduce OK sum={post_sum}")

        # Durability check on a second survivor collective.
        payload.fill_(1.0)
        torch.distributed.all_reduce(payload, group=world.device_group)
        second_sum = float(payload.item())
        assert second_sum == float(cohort_size - num_retirees), (
            f"[rank {rank}] second survivor all_reduce mismatch: got "
            f"{second_sum} expected {cohort_size - num_retirees}"
        )
        _log(rank, hostname, f"second all_reduce OK sum={second_sum}")

        # FSM close-out so survivors land in ``serving_shrunk``.
        ElasticEPStateManager.mark_reconfiguring()
        ElasticEPStateManager.commit_scale()
        final_phase = ElasticEPStateManager.get_scale_phase()
        assert final_phase == "serving_shrunk", (
            f"[rank {rank}] expected serving_shrunk, got {final_phase}"
        )
        assert ElasticEPStateManager.get_effective_ep_size() == (
            cohort_size - num_retirees
        )
        _log(rank, hostname, f"commit_scale OK phase={final_phase}")

    except Exception as exc:  # noqa: BLE001
        import traceback

        _log(rank, hostname, "ERROR:", repr(exc))
        traceback.print_exc()
        return 4

    _log(rank, hostname, "PASS")
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
