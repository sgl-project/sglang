import unittest
from unittest.mock import MagicMock, Mock, call, patch

import torch

from sglang.srt.distributed.device_communicators import vmm_utils
from sglang.srt.layers.moe.shared_ep import vmm
from sglang.srt.layers.moe.shared_ep.state import SharedEpState
from sglang.srt.layers.moe.shared_ep.vmm import (
    SharedEpVmmAllocation,
    _construct_rank_major_views,
    _release_partial_vmm_mapping,
    _release_vmm_handles_synchronized,
    _synchronize_vmm_stage,
    _validate_same_host_group,
    allocate_rank_major_vmm,
    round_up_to_granularity,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-large")


class TestSharedEpStateLifecycle(unittest.TestCase):
    def test_close_invalidates_views_before_unmap(self):
        input_epoch = Mock()
        output_epoch = Mock()
        input_allocation = Mock()
        output_allocation = Mock()
        state = SharedEpState(
            layout=Mock(),
            input_allocation=input_allocation,
            output_allocation=output_allocation,
            input_epoch=input_epoch,
            output_epoch=output_epoch,
            global_input=object(),
            local_input=object(),
            global_output=object(),
            local_output=object(),
        )

        def assert_views_invalidated():
            self.assertIsNone(state.global_input)
            self.assertIsNone(state.local_input)
            self.assertIsNone(state.global_output)
            self.assertIsNone(state.local_output)

        output_allocation.close.side_effect = assert_views_invalidated
        input_allocation.close.side_effect = assert_views_invalidated

        state.close()
        state.close()

        input_epoch.close.assert_called_once_with()
        output_epoch.close.assert_called_once_with()
        input_allocation.close.assert_called_once_with()
        output_allocation.close.assert_called_once_with()


class TestSharedEpVmmHelpers(unittest.TestCase):
    def test_round_up_to_granularity(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            round_up_to_granularity(1, 0)
        self.assertEqual(round_up_to_granularity(1, 65536), 65536)
        self.assertEqual(round_up_to_granularity(65536, 65536), 65536)
        self.assertEqual(round_up_to_granularity(65537, 65536), 131072)

    def test_allocator_rejects_empty_rank_segment_before_cuda_setup(self):
        with self.assertRaisesRegex(ValueError, "logical_rank_bytes"):
            allocate_rank_major_vmm(
                cpu_group=object(),
                device=torch.device("cpu"),
                logical_rank_bytes=0,
            )

    def test_rank_major_offsets_use_mapped_stride(self):
        storage = torch.empty(0, dtype=torch.uint8)
        allocation = SharedEpVmmAllocation(
            local_storage=storage,
            global_storage=storage,
            rank=3,
            world_size=8,
            logical_rank_bytes=50000,
            mapped_rank_bytes=65536,
            granularity=65536,
        )

        self.assertEqual(allocation.rank_offset(0), 0)
        self.assertEqual(allocation.rank_offset(3), 3 * 65536)
        self.assertEqual(allocation.rank_offset(7), 7 * 65536)
        with self.assertRaisesRegex(IndexError, "rank 8"):
            allocation.rank_offset(8)

    def test_all_ranks_ok_uses_collective_backend_device(self):
        cases = (
            ("gloo", torch.device("cpu")),
            ("nccl", torch.device("cuda", torch.cuda.current_device())),
        )
        for backend, expected_device in cases:
            with self.subTest(backend=backend):
                seen = {}
                group = object()

                def fake_all_reduce(flag, *, op, group):
                    seen.update(device=flag.device, op=op, group=group)

                with (
                    patch.object(vmm_utils.dist, "get_backend", return_value=backend),
                    patch.object(
                        vmm_utils.dist,
                        "all_reduce",
                        side_effect=fake_all_reduce,
                    ),
                    torch.device("cuda:3"),
                ):
                    self.assertTrue(vmm_utils.all_ranks_ok(group, True))

                self.assertEqual(seen["device"], expected_device)
                self.assertEqual(seen["op"], torch.distributed.ReduceOp.BAND)
                self.assertIs(seen["group"], group)

    def test_posix_listener_failure_is_published_before_path_exchange(self):
        group = object()
        server = MagicMock()
        server.bind.side_effect = OSError("bind failed")
        published = []

        def fake_all_gather(output, value, *, group):
            published.append(value)
            output[:] = [value, None]

        with (
            patch("socket.socket", return_value=server),
            patch("tempfile.mkdtemp", return_value="/tmp/shared_ep_fd_test"),
            patch.object(vmm_utils.os, "unlink"),
            patch.object(vmm_utils.os, "rmdir"),
            patch.object(
                vmm_utils.dist,
                "all_gather_object",
                side_effect=fake_all_gather,
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "POSIX fd listener setup failed on rank 0.*bind failed",
            ),
        ):
            vmm_utils.exchange_posix_fds(
                group=group,
                rank=0,
                world_size=2,
                local_fds=[10],
                peer_base_counts=[1, 1],
            )

        self.assertEqual(published, ["OSError: bind failed"])

    def test_posix_send_failure_is_published_after_exchange_attempt(self):
        group = object()
        server = MagicMock()
        server.accept.side_effect = OSError("accept stopped")
        outgoing = MagicMock()
        outgoing.__enter__.return_value = outgoing
        outgoing.connect.side_effect = OSError("connect failed")
        published = []

        def fake_all_gather(output, value, *, group):
            published.append(value)
            if len(published) == 1:
                output[:] = [None, None]
            elif len(published) == 2:
                output[:] = [
                    "/tmp/shared_ep_fd_test/rank_0.sock",
                    "/tmp/shared_ep_fd_test/rank_1.sock",
                ]
            else:
                output[:] = [value, None]

        with (
            patch("socket.socket", side_effect=[server, outgoing]),
            patch("tempfile.mkdtemp", return_value="/tmp/shared_ep_fd_test"),
            patch.object(vmm_utils.os, "unlink"),
            patch.object(vmm_utils.os, "rmdir"),
            patch.object(
                vmm_utils.dist,
                "all_gather_object",
                side_effect=fake_all_gather,
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "POSIX fd exchange failed on rank 0.*connect failed",
            ),
        ):
            vmm_utils.exchange_posix_fds(
                group=group,
                rank=0,
                world_size=2,
                local_fds=[10],
                peer_base_counts=[1, 1],
            )

        self.assertEqual(
            published,
            [None, "/tmp/shared_ep_fd_test/rank_0.sock", "OSError: connect failed"],
        )

    def test_posix_descriptor_mismatch_is_published_before_mapping(self):
        group = object()
        server = MagicMock()
        incoming = MagicMock()
        incoming.__enter__.return_value = incoming
        server.accept.return_value = (incoming, None)
        outgoing = MagicMock()
        outgoing.__enter__.return_value = outgoing
        published = []

        def fake_all_gather(output, value, *, group):
            published.append(value)
            if len(published) == 1:
                output[:] = [None, None]
            elif len(published) == 2:
                output[:] = [
                    "/tmp/shared_ep_fd_test/rank_0.sock",
                    "/tmp/shared_ep_fd_test/rank_1.sock",
                ]
            else:
                output[:] = [value, None]

        with (
            patch("socket.socket", side_effect=[server, outgoing]),
            patch("tempfile.mkdtemp", return_value="/tmp/shared_ep_fd_test"),
            patch.object(vmm_utils, "_recv_fd", return_value=None),
            patch.object(vmm_utils, "_send_fd"),
            patch.object(vmm_utils.os, "unlink"),
            patch.object(vmm_utils.os, "rmdir"),
            patch.object(
                vmm_utils.dist,
                "all_gather_object",
                side_effect=fake_all_gather,
            ),
            self.assertRaisesRegex(
                RuntimeError,
                r"POSIX fd validation failed on rank 0.*missing=\[\(1, 0\)\]",
            ),
        ):
            vmm_utils.exchange_posix_fds(
                group=group,
                rank=0,
                world_size=2,
                local_fds=[10],
                peer_base_counts=[1, 1],
            )

        self.assertEqual(
            published,
            [
                None,
                "/tmp/shared_ep_fd_test/rank_0.sock",
                None,
                "missing=[(1, 0)], extra=[]",
            ],
        )

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
            allocate_rank_major_vmm(
                cpu_group="group",
                device=torch.device("cuda:1"),
                logical_rank_bytes=4096,
            )

        synchronize.assert_called_once_with(
            "group",
            1,
            "preflight",
            local_error,
        )

    def test_same_host_query_failure_is_synchronized(self):
        local_error = RuntimeError("hostname unavailable")

        with (
            patch.object(vmm.dist, "get_rank", return_value=1),
            patch.object(vmm.dist, "get_world_size", return_value=2),
            patch.object(vmm.os, "uname", side_effect=local_error),
            patch.object(
                vmm,
                "_synchronize_vmm_stage",
                side_effect=RuntimeError("symmetric host-query failure"),
            ) as synchronize,
            self.assertRaisesRegex(RuntimeError, "symmetric host-query failure"),
        ):
            _validate_same_host_group("group")

        synchronize.assert_called_once_with(
            "group",
            1,
            "host query",
            local_error,
        )

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
                "SharedEP VMM allocation failed on rank 0.*OUT_OF_MEMORY",
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
                "SharedEP VMM mapping failed on rank 1.*cuMemMap",
            ),
        ):
            _synchronize_vmm_stage("group", 0, "mapping", None)

    def test_partial_mapping_cleanup_is_reverse_order(self):
        driver = MagicMock()
        mapped_addresses = [0x1000, 0x3000]

        _release_partial_vmm_mapping(
            driver,
            base_va=0x1000,
            total_bytes=0x4000,
            mapped_addresses=mapped_addresses,
            segment_bytes=0x1000,
        )

        self.assertEqual(
            driver.cuMemUnmap.call_args_list,
            [call(0x3000, 0x1000), call(0x1000, 0x1000)],
        )
        driver.cuMemAddressFree.assert_called_once_with(0x1000, 0x4000)
        self.assertEqual(mapped_addresses, [])

    def test_local_handle_release_failure_is_synchronized_and_retained(self):
        driver = MagicMock()
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
                driver,
                retained_handles=retained_handles,
                cpu_group="group",
                rank=1,
            )

        self.assertEqual(retained_handles, [11, 22])
        synchronize.assert_called_once_with(
            "group",
            1,
            "handle release",
            release_error,
        )

    def test_remote_handle_release_failure_arrives_after_local_release(self):
        driver = MagicMock()
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
                driver,
                retained_handles=retained_handles,
                cpu_group="group",
                rank=1,
            )

        self.assertEqual(retained_handles, [])
        synchronize.assert_called_once_with(
            "group",
            1,
            "handle release",
            None,
        )

    def test_tensor_view_failure_is_synchronized(self):
        view_error = RuntimeError("torch.from_dlpack rejected CUDA pointer")

        with (
            patch.object(vmm, "_uint8_tensor_from_cuda_ptr", side_effect=view_error),
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
                total_bytes=0x4000,
                mapped_rank_bytes=0x2000,
                logical_rank_bytes=0x1000,
                device_id=1,
                refs=[],
            )

        synchronize.assert_called_once_with(
            "group",
            1,
            "tensor view construction",
            view_error,
        )


if __name__ == "__main__":
    unittest.main()
