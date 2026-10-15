"""Coordination for one host allocation read by every MLA attention-TP rank."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import ClassVar, Optional

import torch


@dataclass
class SharedCPUHostDedupContext:
    """Publish owner writes and preserve native HiCache per-layer load events.

    No GPU communicator is created. ``layerwise_overlap`` names the optional
    broadcast callback in MLAHostDedupContext; native H2D layer events remain
    active here without that callback.
    """

    process_group: torch.distributed.ProcessGroup
    producer_stream: Optional[object] = None
    last_write_finish_event: Optional[object] = None
    is_shared: ClassVar[bool] = True
    is_dummy_rank: ClassVar[bool] = False
    layerwise_overlap: ClassVar[bool] = False

    def __post_init__(self) -> None:
        if self.process_group is None or not torch.distributed.is_initialized():
            raise ValueError(
                "Shared MLA host restore requires an initialized TP group."
            )
        if torch.distributed.get_backend(self.process_group) != "gloo":
            raise ValueError("Shared MLA host restore requires a CPU/Gloo TP group.")
        if torch.distributed.get_world_size(self.process_group) <= 1:
            raise ValueError("Shared MLA host restore requires TP > 1.")

    def validate_host_rows(self, transfers, failure: Optional[str] = None) -> None:
        """Compare shared row order before accessing host memory on any rank.

        GPU destination rows may differ by rank. Host rows must identify the
        same ordered bytes, including each DSA side pool, on every reader.
        Catch local preparation errors so peers reach the same failure verdict.
        """
        digest = hashlib.blake2b(digest_size=8)
        try:
            for transfer in transfers:
                pool = transfer.host_pool
                if not getattr(pool.allocator, "is_shared", False):
                    raise ValueError(
                        "Shared MLA restore cannot include a private host pool."
                    )
                if pool.allocator.process_group is not self.process_group:
                    raise ValueError(
                        "Shared host allocation uses a different TP group."
                    )
                label = (
                    f"{type(pool).__name__}:{getattr(pool, 'pool_label', '')}".encode()
                )
                rows = transfer.host_indices.to(
                    device="cpu", dtype=torch.int64
                ).contiguous()
                digest.update(len(label).to_bytes(8, "little"))
                digest.update(label)
                digest.update(rows.numel().to_bytes(8, "little"))
                digest.update(rows.numpy().tobytes())
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
        value = int.from_bytes(digest.digest(), "little") & ((1 << 63) - 1)
        verdict = torch.tensor(
            [value, -value, -int(failure is not None)], dtype=torch.int64
        )
        torch.distributed.all_reduce(
            verdict, op=torch.distributed.ReduceOp.MIN, group=self.process_group
        )
        if verdict[2].item() < 0:
            self._raise_rank_errors(failure, "host row preparation")
        if verdict[0].item() != -verdict[1].item():
            raise RuntimeError("Shared CPU host row order diverged across TP ranks.")

    def synchronize_error(self, failure: Optional[str], *, operation: str) -> None:
        """Publish successful submission/completion only after all ranks agree."""
        verdict = torch.tensor([int(failure is not None)], dtype=torch.int32)
        torch.distributed.all_reduce(
            verdict, op=torch.distributed.ReduceOp.MAX, group=self.process_group
        )
        if verdict.item():
            self._raise_rank_errors(failure, operation)

    def _raise_rank_errors(self, failure: Optional[str], operation: str) -> None:
        errors = [None] * torch.distributed.get_world_size(self.process_group)
        torch.distributed.all_gather_object(errors, failure, group=self.process_group)
        details = "; ".join(
            f"rank {rank}: {error}" for rank, error in enumerate(errors) if error
        )
        raise RuntimeError(f"Shared MLA {operation} failed: {details}")

    def synchronize_readers(self) -> None:
        """All local copies must finish before reset may reuse shared rows."""
        torch.distributed.barrier(group=self.process_group)
