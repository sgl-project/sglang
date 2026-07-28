"""Manual eight-process ROCm SharedEP VMM feasibility probe.

Run on one 8-GPU ROCm 7.2 node:

    torchrun --standalone --nproc-per-node=8 \
      test/manual/ep/test_shared_ep_vmm_rocm.py

This exercises only the VMM runtime; no SGLang server or model is started.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import os
import sys
import types
from pathlib import Path

import torch
import torch.distributed as dist

from sglang.srt.distributed.device_communicators import vmm_utils

# Avoid importing SharedEP's compute backend: this probe intentionally exercises
# only the memory runtime.
_SHARED_EP_PACKAGE = "sglang.srt.layers.moe.shared_ep"
if _SHARED_EP_PACKAGE not in sys.modules:
    package = types.ModuleType(_SHARED_EP_PACKAGE)
    srt_root = Path(vmm_utils.__file__).resolve().parents[2]
    package.__path__ = [str(srt_root / "layers" / "moe" / "shared_ep")]
    sys.modules[_SHARED_EP_PACKAGE] = package
_vmm = importlib.import_module(f"{_SHARED_EP_PACKAGE}.vmm")
allocate_rank_major_vmm = _vmm.allocate_rank_major_vmm


def _publish_stage_error(stage: str, local_error: BaseException | None) -> None:
    world_size = dist.get_world_size()
    errors: list[str | None] = [None] * world_size
    text = (
        None if local_error is None else f"{type(local_error).__name__}: {local_error}"
    )
    dist.all_gather_object(errors, text)
    failures = [
        f"rank {failed_rank}: {error}"
        for failed_rank, error in enumerate(errors)
        if error is not None
    ]
    if failures:
        raise RuntimeError(f"{stage} failed; " + "; ".join(failures))


def _open_fd_count() -> int:
    return len(os.listdir("/proc/self/fd"))


def _exercise_mapping(logical_bytes: int) -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)

    capability = vmm_utils.query_vmm_capability(device)
    capability_error = (
        None
        if capability.supported and capability.platform == "rocm"
        else RuntimeError(f"unexpected SharedEP VMM capability result: {capability}")
    )
    _publish_stage_error("ROCm VMM capability probe", capability_error)

    # Warm DLPack and teardown before measuring descriptor cleanup.
    warmup_error = None
    try:
        warmup = allocate_rank_major_vmm(
            cpu_group=dist.group.WORLD,
            device=device,
            logical_rank_bytes=world_size,
        )
        warmup.close()
        del warmup
        gc.collect()
        torch.cuda.synchronize()
    except BaseException as error:
        warmup_error = error
    _publish_stage_error("VMM warmup and teardown", warmup_error)
    dist.barrier()
    baseline_fds = _open_fd_count()

    allocation = None
    allocation_error = None
    try:
        allocation = allocate_rank_major_vmm(
            cpu_group=dist.group.WORLD,
            device=device,
            logical_rank_bytes=logical_bytes,
        )
    except BaseException as error:
        allocation_error = error
    _publish_stage_error("rank-major allocation", allocation_error)
    assert allocation is not None

    publish_error = None
    try:
        # Every owner first publishes a distinct local pattern.
        allocation.local_storage.fill_(rank + 1)
        torch.cuda.synchronize()
    except BaseException as error:
        publish_error = error
    _publish_stage_error("local pattern publication", publish_error)
    dist.barrier()

    read_error = None
    try:
        for owner in range(world_size):
            segment = allocation.global_storage.narrow(
                0,
                allocation.rank_offset(owner),
                logical_bytes,
            )
            if not torch.all(segment == owner + 1).item():
                raise AssertionError(
                    f"rank {rank} read the wrong initial pattern from owner {owner}"
                )
        del segment
    except BaseException as error:
        read_error = error
    _publish_stage_error("all-rank initial reads", read_error)

    write_error = None
    try:
        # Each writer updates a disjoint byte in every owner's physical segment.
        remote_offsets = torch.tensor(
            [allocation.rank_offset(owner) + rank for owner in range(world_size)],
            dtype=torch.int64,
            device=device,
        )
        allocation.global_storage[remote_offsets] = 128 + rank
        torch.cuda.synchronize()
    except BaseException as error:
        write_error = error
    _publish_stage_error("all-rank remote writes", write_error)
    dist.barrier()

    verify_error = None
    try:
        expected = torch.arange(
            128,
            128 + world_size,
            dtype=torch.uint8,
            device=device,
        )
        if not torch.equal(allocation.local_storage[:world_size], expected):
            raise AssertionError(
                f"owner rank {rank} did not observe every remote writer"
            )
        for owner in range(world_size):
            observed = allocation.global_storage.narrow(
                0,
                allocation.rank_offset(owner),
                world_size,
            )
            if not torch.equal(observed, expected):
                raise AssertionError(
                    f"rank {rank} read the wrong remote-write vector "
                    f"from owner {owner}"
                )
        del observed, expected, remote_offsets
        torch.cuda.synchronize()
    except BaseException as error:
        verify_error = error
    _publish_stage_error("remote-write visibility validation", verify_error)

    cleanup_error = None
    try:
        assert allocation is not None
        allocation.close()
        allocation.close()
        if allocation._base_va != 0:
            raise AssertionError("allocation retained its virtual address after close")
        if allocation.local_storage.numel() or allocation.global_storage.numel():
            raise AssertionError("allocation retained tensor views after close")
        del allocation
        gc.collect()
        torch.cuda.synchronize()
    except BaseException as error:
        cleanup_error = error
    dist.barrier()
    final_fds = _open_fd_count()
    if cleanup_error is None and final_fds != baseline_fds:
        cleanup_error = RuntimeError(
            f"POSIX descriptor count changed across allocation cleanup: "
            f"{baseline_fds} -> {final_fds}"
        )
    _publish_stage_error("VMM cleanup", cleanup_error)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logical-bytes", type=int, default=64 * 1024)
    args = parser.parse_args()
    if args.logical_bytes < 8:
        raise ValueError("--logical-bytes must be at least 8")
    if torch.version.hip is None:
        raise RuntimeError("this probe requires a ROCm PyTorch build")
    if int(os.environ.get("WORLD_SIZE", "1")) != 8:
        raise RuntimeError("launch this probe with torchrun --nproc-per-node=8")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("gloo")
    try:
        _exercise_mapping(args.logical_bytes)
        if dist.get_rank() == 0:
            print("ROCm SharedEP VMM EP8 feasibility probe passed")
    finally:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
