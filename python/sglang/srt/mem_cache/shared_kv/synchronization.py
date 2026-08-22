"""Mandatory device-side publication for owner-only shared writes."""

import torch

from sglang.kernels.ops.kvcache.shared_kv_publication import (
    compile_shared_kv_publication,
    shared_kv_publish,
    shared_kv_publish_status,
)
from sglang.srt.mem_cache.shared_kv.vmm import (
    RankMajorSharedTensor,
    _synchronize_vmm_stage,
    create_rank_major_shared_tensor,
)


class SharedWritePublisher:
    def __init__(self, attention_cp_group: object) -> None:
        self._rank = int(attention_cp_group.rank_in_group)
        self._world_size = int(attention_cp_group.world_size)
        if not 2 <= self._world_size <= 8:
            raise ValueError(
                "Shared KV publication attention CP size must be in [2, 8], "
                f"got {self._world_size}"
            )
        if not 0 <= self._rank < self._world_size:
            raise ValueError(
                f"Shared KV publication rank must be in [0, {self._world_size}), "
                f"got {self._rank}"
            )

        cpu_group = attention_cp_group.cpu_group
        flags = create_rank_major_shared_tensor(
            (2 * self._world_size,),
            dtype=torch.int32,
            cpu_group=cpu_group,
        )
        peer_ptrs = None
        epoch = None
        status_result = None
        initialization_error = None
        try:
            flags.local_view.zero_()
            peer_ptrs = torch.tensor(
                [
                    flags.global_view.data_ptr() + peer * flags.aligned_bytes_per_rank
                    for peer in range(self._world_size)
                ],
                dtype=torch.int64,
                device=flags.global_view.device,
            )
            epoch = torch.zeros(
                (1,), dtype=torch.int32, device=flags.global_view.device
            )
            status_result = torch.ones(
                (1,), dtype=torch.int32, device=flags.global_view.device
            )
            compile_shared_kv_publication(self._world_size)
            torch.cuda.synchronize()
        except BaseException as error:
            initialization_error = error
        try:
            _synchronize_vmm_stage(
                cpu_group,
                self._rank,
                "publisher initialization",
                initialization_error,
            )
        except BaseException:
            flags.close()
            raise

        assert peer_ptrs is not None and epoch is not None and status_result is not None
        self._flags: RankMajorSharedTensor | None = flags
        self._peer_ptrs: torch.Tensor | None = peer_ptrs
        self._epoch: torch.Tensor | None = epoch
        self._status_result: torch.Tensor | None = status_result

    @property
    def mapped_bytes_per_rank(self) -> int:
        if self._flags is None:
            return 0
        return self._flags.aligned_bytes_per_rank

    def publish(self) -> None:
        if self._flags is None or self._peer_ptrs is None or self._epoch is None:
            raise RuntimeError("Shared KV publisher is closed")
        shared_kv_publish(
            self._flags.global_view,
            self._peer_ptrs,
            self._epoch,
            self._rank,
            self._world_size,
        )

    def publish_status(self, local_success: bool) -> bool:
        if (
            self._flags is None
            or self._peer_ptrs is None
            or self._epoch is None
            or self._status_result is None
        ):
            raise RuntimeError("Shared KV publisher is closed")
        shared_kv_publish_status(
            self._flags.global_view,
            self._peer_ptrs,
            self._epoch,
            self._status_result,
            self._rank,
            self._world_size,
            local_success,
        )
        return bool(self._status_result.item())

    def close(self) -> None:
        flags = self._flags
        if flags is None:
            return
        self._flags = None
        self._peer_ptrs = None
        self._epoch = None
        self._status_result = None
        flags.close()
