import os
import unittest
from contextlib import nullcontext
from unittest.mock import MagicMock, call, patch

import torch
import torch.multiprocessing as mp

from sglang.srt.mem_cache.shared_kv import vmm
from sglang.srt.mem_cache.shared_kv.vmm import (
    RankMajorSharedSlab,
    _align_first_dim,
    _construct_rank_major_views,
    _release_partial_vmm_mapping,
    _release_vmm_handles_synchronized,
    _synchronize_vmm_stage,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-c", runner_config="4-gpu-b200")

PORT = 29821


def _destroy_distributed() -> None:
    from sglang.srt.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
    )

    destroy_model_parallel()
    destroy_distributed_environment()


def _run_neutral_rank_major_vmm(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    torch.cuda.set_device(rank)

    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.mem_cache.shared_kv.vmm import create_rank_major_shared_tensor
    from sglang.srt.runtime_context import get_parallel

    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        backend="nccl",
    )
    initialize_model_parallel(
        tensor_model_parallel_size=world_size,
        attention_context_model_parallel_size=world_size,
    )
    cpu_group = get_parallel().attn_cp_group.cpu_group

    for iteration in range(2):
        allocation = create_rank_major_shared_tensor(
            (64, 1, 8),
            dtype=torch.uint8,
            cpu_group=cpu_group,
            first_dim_multiple=64,
        )
        allocation.local_view.fill_(rank + iteration * world_size + 1)
        torch.cuda.synchronize()
        torch.distributed.barrier(group=cpu_group)

        for owner_rank in range(world_size):
            expected = owner_rank + iteration * world_size + 1
            global_start = owner_rank * allocation.local_rows
            relative_start = ((owner_rank - rank) % world_size) * allocation.local_rows
            assert torch.all(
                allocation.global_view.narrow(0, global_start, allocation.local_rows)
                == expected
            ).item()
            assert torch.all(
                allocation.rank_local_view.narrow(
                    0, relative_start, allocation.local_rows
                )
                == expected
            ).item()

        allocation.close()
        allocation.close()
        torch.distributed.barrier(group=cpu_group)

    _destroy_distributed()


def _run_neutral_preflight_failure(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    torch.cuda.set_device(rank)

    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.mem_cache.shared_kv import vmm as child_vmm
    from sglang.srt.runtime_context import get_parallel

    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        backend="nccl",
    )
    initialize_model_parallel(
        tensor_model_parallel_size=world_size,
        attention_context_model_parallel_size=world_size,
    )
    cpu_group = get_parallel().attn_cp_group.cpu_group
    driver_patch = (
        patch.object(
            child_vmm,
            "_get_cuda_driver",
            side_effect=RuntimeError("injected driver preflight failure"),
        )
        if rank == 1
        else nullcontext()
    )
    with driver_patch:
        try:
            child_vmm.create_rank_major_shared_tensor(
                (64, 8),
                dtype=torch.uint8,
                cpu_group=cpu_group,
            )
        except RuntimeError as error:
            assert "shared KV VMM preflight failed on rank 1" in str(error)
        else:
            raise AssertionError("expected symmetric preflight failure")
    _destroy_distributed()


class TestRankMajorVMMContracts(CustomTestCase):
    def test_preflight_failure_is_synchronized_before_first_allocation(self):
        local_error = RuntimeError("cuMemGetAllocationGranularity failed")

        with (
            patch.object(vmm, "_validate_same_host_group"),
            patch.object(vmm.dist, "get_rank", return_value=1),
            patch.object(vmm.dist, "get_world_size", return_value=2),
            patch.object(vmm, "_get_cuda_driver", side_effect=local_error),
            patch.object(
                vmm,
                "_synchronize_vmm_stage",
                side_effect=RuntimeError("symmetric preflight failure"),
            ) as synchronize,
            self.assertRaisesRegex(RuntimeError, "symmetric preflight failure"),
        ):
            vmm.create_rank_major_shared_tensor(
                (64, 8),
                dtype=torch.uint8,
                cpu_group="group",
            )

        synchronize.assert_called_once_with("group", 1, "preflight", local_error)

    def test_stage_failure_reports_local_rank_and_preserves_cause(self):
        local_error = RuntimeError("cuMemCreate: CUDA_ERROR_OUT_OF_MEMORY")

        with (
            patch.object(vmm.dist, "get_world_size", return_value=2),
            patch.object(
                vmm.dist,
                "all_gather_object",
                side_effect=lambda output, value, group: output.__setitem__(0, value),
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "shared KV VMM allocation failed on rank 0.*OUT_OF_MEMORY",
            ) as raised,
        ):
            _synchronize_vmm_stage("group", 0, "allocation", local_error)

        self.assertIs(raised.exception.__cause__, local_error)

    def test_stage_failure_reports_remote_rank(self):
        gathered_errors = [None, "cuMemMap(rank=0): CUDA_ERROR_INVALID_VALUE"]

        with (
            patch.object(vmm.dist, "get_world_size", return_value=2),
            patch.object(
                vmm.dist,
                "all_gather_object",
                side_effect=lambda output, _value, group: output.__setitem__(
                    slice(None), gathered_errors
                ),
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "shared KV VMM mapping failed on rank 1.*cuMemMap",
            ),
        ):
            _synchronize_vmm_stage("group", 0, "mapping", None)

    def test_partial_mapping_cleanup_releases_reverse_map_order_then_va(self):
        drv = MagicMock()
        mapped_addresses = [0x1000, 0x3000]

        _release_partial_vmm_mapping(
            drv,
            base_va=0x1000,
            total_bytes=0x4000,
            mapped_addresses=mapped_addresses,
            segment_bytes=0x1000,
        )

        self.assertEqual(
            drv.cuMemUnmap.call_args_list,
            [call(0x3000, 0x1000), call(0x1000, 0x1000)],
        )
        drv.cuMemAddressFree.assert_called_once_with(0x1000, 0x4000)
        self.assertEqual(mapped_addresses, [])

    def test_local_handle_release_failure_is_synchronized_and_retained(self):
        drv = MagicMock()
        retained_handles = [11, 22]
        release_error = RuntimeError("cuMemRelease: CUDA_ERROR_INVALID_HANDLE")

        with (
            patch.object(vmm, "check_drv", side_effect=release_error),
            patch.object(
                vmm,
                "_synchronize_vmm_stage",
                side_effect=RuntimeError("symmetric handle-release failure"),
            ) as synchronize,
            self.assertRaisesRegex(RuntimeError, "symmetric handle-release failure"),
        ):
            _release_vmm_handles_synchronized(
                drv,
                retained_handles=retained_handles,
                cpu_group="group",
                rank=1,
            )

        self.assertEqual(retained_handles, [11, 22])
        synchronize.assert_called_once_with("group", 1, "handle release", release_error)

    def test_remote_handle_release_failure_arrives_after_local_release(self):
        drv = MagicMock()
        retained_handles = [11, 22]

        with (
            patch.object(vmm, "check_drv"),
            patch.object(
                vmm,
                "_synchronize_vmm_stage",
                side_effect=RuntimeError("handle release failed on rank 0"),
            ) as synchronize,
            self.assertRaisesRegex(RuntimeError, "failed on rank 0"),
        ):
            _release_vmm_handles_synchronized(
                drv,
                retained_handles=retained_handles,
                cpu_group="group",
                rank=1,
            )

        self.assertEqual(retained_handles, [])
        synchronize.assert_called_once_with("group", 1, "handle release", None)

    def test_alignment_honors_driver_granularity_and_ownership_multiple(self):
        rows, aligned_bytes = _align_first_dim(
            (257, 3),
            dtype=torch.uint8,
            granularity=256,
            first_dim_multiple=128,
        )

        self.assertEqual(rows, 512)
        self.assertEqual(aligned_bytes, 1536)
        self.assertEqual(aligned_bytes % 256, 0)
        self.assertEqual(rows % 128, 0)

    def test_slab_close_drops_layer_aliases_before_allocation(self):
        allocation = MagicMock()
        slab = RankMajorSharedSlab(
            allocation=allocation,
            layer_rows=4,
            global_views=[torch.empty(0)],
            rank_local_views=[torch.empty(0)],
            local_views=[torch.empty(0)],
        )

        slab.close()
        slab.close()

        self.assertEqual(slab.global_views, [])
        self.assertEqual(slab.rank_local_views, [])
        self.assertEqual(slab.local_views, [])
        allocation.close.assert_called_once_with()

    def test_tensor_view_failure_is_synchronized_before_mapping_cleanup(self):
        view_error = RuntimeError("torch.from_dlpack rejected CUDA pointer")

        with (
            patch.object(vmm, "_tensor_from_cuda_ptr", side_effect=view_error),
            patch.object(
                vmm,
                "_synchronize_vmm_stage",
                side_effect=RuntimeError("symmetric tensor-view failure"),
            ) as synchronize,
            self.assertRaisesRegex(RuntimeError, "symmetric tensor-view failure"),
        ):
            _construct_rank_major_views(
                cpu_group="group",
                rank=1,
                base_va=0x1000,
                rank_local_base_va=0x3000,
                world_size=2,
                local_rows=64,
                row_shape=(1, 8),
                dtype=torch.uint8,
                device_id=1,
                refs=[],
            )

        synchronize.assert_called_once_with(
            "group", 1, "tensor view construction", view_error
        )

    def test_peer_reads_and_create_close_recreate(self):
        if torch.cuda.device_count() < 2:
            self.skipTest("neutral rank-major VMM test needs at least two GPUs")
        mp.spawn(
            _run_neutral_rank_major_vmm,
            args=(2, PORT),
            nprocs=2,
            join=True,
        )

    def test_two_rank_preflight_failure_is_symmetric(self):
        if torch.cuda.device_count() < 2:
            self.skipTest("neutral rank-major VMM test needs at least two GPUs")
        mp.spawn(
            _run_neutral_preflight_failure,
            args=(2, PORT + 1),
            nprocs=2,
            join=True,
        )


if __name__ == "__main__":
    unittest.main()
