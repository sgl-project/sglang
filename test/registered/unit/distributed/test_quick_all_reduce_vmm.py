import os
import tempfile
import threading
import types
import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.distributed import parallel_state
from sglang.srt.distributed.device_communicators import (
    quick_all_reduce,
    quick_all_reduce_vmm,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _BlockingStore:
    def __init__(self):
        self._condition = threading.Condition()
        self._values = {}

    def set(self, key, value):
        with self._condition:
            self._values[key] = value
            self._condition.notify_all()

    def get(self, key):
        with self._condition:
            ready = self._condition.wait_for(lambda: key in self._values, timeout=5)
            if not ready:
                raise TimeoutError(f"missing store key {key}")
            return self._values[key]

    def wait(self, keys, timeout):
        with self._condition:
            ready = self._condition.wait_for(
                lambda: all(key in self._values for key in keys),
                timeout=timeout.total_seconds(),
            )
            if not ready:
                missing = [key for key in keys if key not in self._values]
                raise TimeoutError(f"missing store keys {missing}")


class TestQuickAllReduceVMM(unittest.TestCase):
    def test_vmm_initialization_opens_native_peer_handles(self):
        local_fd = os.open(os.devnull, os.O_RDONLY)
        peer_fd = os.dup(local_fd)

        communicator = object.__new__(quick_all_reduce.QuickAllReduce)
        communicator.rank = 0
        communicator.world_size = 2
        communicator.group = object()
        communicator.device = torch.device("cuda:0")
        communicator._ptr = 0

        props = types.SimpleNamespace(gcnArchName="gfx950:sramecc+:xnack-")
        with (
            patch.object(
                quick_all_reduce.torch.cuda,
                "get_device_properties",
                return_value=props,
            ),
            patch.object(
                quick_all_reduce.dist.distributed_c10d,
                "_get_default_store",
                return_value="store",
            ),
            patch.object(
                quick_all_reduce.dist,
                "get_process_group_ranks",
                return_value=[3, 2],
            ),
            patch.object(
                quick_all_reduce,
                "_new_vmm_pool_name",
                return_value="sglang_quickreduce_2_3_gtest",
            ) as pool_name,
            patch.object(quick_all_reduce, "_raise_if_any_vmm_phase_failed"),
            patch.object(
                quick_all_reduce.ops,
                "init_custom_qr_vmm",
                return_value=(123, local_fd, 4096),
                create=True,
            ) as init_vmm,
            patch.object(
                quick_all_reduce_vmm,
                "exchange_vmm_fds",
                return_value=([-1, peer_fd], [4096, 4096]),
            ) as exchange,
            patch.object(
                quick_all_reduce.ops,
                "qr_open_vmm_handles",
                create=True,
            ) as open_peers,
            patch.object(
                quick_all_reduce.ops,
                "qr_destroy",
                create=True,
            ) as destroy,
        ):
            communicator._init_vmm(1024)

            self.assertEqual(communicator._ptr, 123)
            communicator.close()

        self.assertEqual(communicator._ptr, 0)
        self.assertTrue(communicator.disabled)
        init_vmm.assert_called_once_with(0, 2, 0, 1024, True)
        exchange.assert_called_once_with(
            0,
            2,
            "sglang_quickreduce_2_3_gtest",
            local_fd,
            4096,
            "store",
            "2_3",
        )
        pool_name.assert_called_once_with(
            communicator.group, "2_3", communicator.device
        )
        open_peers.assert_called_once_with(123, [-1, peer_fd], [4096, 4096])
        destroy.assert_called_once_with(123)
        with self.assertRaises(OSError):
            os.fstat(local_fd)
        with self.assertRaises(OSError):
            os.fstat(peer_fd)

    def test_fd_exchange_returns_rank_ordered_descriptors(self):
        store = _BlockingStore()
        files = [tempfile.TemporaryFile(), tempfile.TemporaryFile()]
        files[0].write(b"a")
        files[1].write(b"b")
        for file in files:
            file.flush()

        results = {}
        errors = []
        socket_dir_modes = []
        real_mkdtemp = quick_all_reduce_vmm.tempfile.mkdtemp

        def secure_mkdtemp(*args, **kwargs):
            path = real_mkdtemp(*args, **kwargs)
            socket_dir_modes.append(os.stat(path).st_mode & 0o777)
            return path

        def exchange(rank):
            try:
                results[rank] = quick_all_reduce_vmm.exchange_vmm_fds(
                    rank,
                    2,
                    "unit_test_pool",
                    files[rank].fileno(),
                    4096 + rank,
                    store,
                    "0_1",
                )
            except BaseException as exc:
                errors.append(exc)

        with patch.object(
            quick_all_reduce_vmm.tempfile,
            "mkdtemp",
            side_effect=secure_mkdtemp,
        ):
            threads = [
                threading.Thread(target=exchange, args=(rank,)) for rank in range(2)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(10)

        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        self.assertEqual(socket_dir_modes, [0o700, 0o700])
        self.assertEqual(results[0][1], [4096, 4097])
        self.assertEqual(results[1][1], [4096, 4097])
        self.assertEqual(os.pread(results[0][0][1], 1, 0), b"b")
        self.assertEqual(os.pread(results[1][0][0], 1, 0), b"a")
        self.assertFalse(os.get_inheritable(results[0][0][1]))
        self.assertFalse(os.get_inheritable(results[1][0][0]))

        for rank, (fds, _) in results.items():
            for peer_rank, fd in enumerate(fds):
                if peer_rank != rank:
                    os.close(fd)
        for file in files:
            file.close()

    def test_fd_exchange_closes_received_descriptors_on_metadata_error(self):
        store = _BlockingStore()
        store_prefix = "sgl_qr_vmm/0_1/unit_test_pool"
        store.set(f"{store_prefix}/socket/r1_from_r0", b"unused.sock")
        store.set(f"{store_prefix}/size/r1", b"not-an-integer")

        with tempfile.TemporaryFile() as local_file:
            received_fd = os.dup(local_file.fileno())
            with (
                patch.object(
                    quick_all_reduce_vmm,
                    "_recv_fd",
                    return_value=received_fd,
                ),
                patch.object(quick_all_reduce_vmm, "_send_fd"),
            ):
                with self.assertRaises(ValueError):
                    quick_all_reduce_vmm.exchange_vmm_fds(
                        0,
                        2,
                        "unit_test_pool",
                        local_file.fileno(),
                        4096,
                        store,
                        "0_1",
                    )

            with self.assertRaises(OSError):
                os.fstat(received_fd)

    def test_empty_all_reduce_still_participates_in_collective(self):
        coordinator = object.__new__(parallel_state.GroupCoordinator)
        coordinator.world_size = 2
        coordinator.local_size = 2
        coordinator.device_group = object()
        tensor = torch.empty(0)

        with (
            patch.object(parallel_state, "is_shm_available", return_value=False),
            patch.object(torch.distributed, "all_reduce") as all_reduce,
        ):
            self.assertIs(coordinator.all_reduce(tensor), tensor)

        all_reduce.assert_called_once_with(tensor, group=coordinator.device_group)

    def test_quick_reduce_group_honors_custom_allreduce_disable(self):
        with (
            patch.object(parallel_state, "_ENABLE_CUSTOM_ALL_REDUCE", False),
            patch.object(
                parallel_state,
                "_is_rocm_quick_reduce_requested",
                return_value=True,
            ),
        ):
            self.assertFalse(parallel_state._should_enable_rocm_quick_reduce_group())

        with (
            patch.object(parallel_state, "_ENABLE_CUSTOM_ALL_REDUCE", True),
            patch.object(
                parallel_state,
                "_is_rocm_quick_reduce_requested",
                return_value=True,
            ),
        ):
            self.assertTrue(parallel_state._should_enable_rocm_quick_reduce_group())

    def test_group_destroy_closes_quick_reduce(self):
        coordinator = object.__new__(parallel_state.GroupCoordinator)
        coordinator.device_group = None
        coordinator.cpu_group = None
        coordinator.device = torch.device("cpu")
        coordinator.device_module = torch
        coordinator.pynccl_comm = None
        coordinator.pymscclpp_comm = None
        coordinator.ca_comm = None
        coordinator.qr_comm = MagicMock()
        coordinator.mq_broadcaster = None

        quick_reduce = coordinator.qr_comm
        coordinator.destroy()

        quick_reduce.close.assert_called_once_with()
        self.assertIsNone(coordinator.qr_comm)

    def test_close_clears_native_pointer_before_cleanup_error(self):
        communicator = object.__new__(quick_all_reduce.QuickAllReduce)
        communicator._ptr = 123
        communicator.disabled = False

        with patch.object(
            quick_all_reduce.ops,
            "qr_destroy",
            side_effect=RuntimeError("cleanup failed"),
            create=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
                communicator.close()

        self.assertEqual(communicator._ptr, 0)
        # The finalizer must not retry the freed pointer or propagate.
        communicator.__del__()

    def test_quick_reduce_precedes_custom_allreduce(self):
        coordinator = object.__new__(parallel_state.GroupCoordinator)
        coordinator.world_size = 2
        coordinator.unique_name = "test_group"
        coordinator.hpu_communicator = None
        coordinator.xpu_communicator = None
        coordinator.npu_communicator = None
        coordinator.pynccl_comm = None
        coordinator.pymscclpp_comm = None
        coordinator.torch_symm_mem_comm = None
        coordinator.qr_comm = types.SimpleNamespace(
            disabled=False,
            should_quick_allreduce=MagicMock(return_value=True),
        )
        coordinator.ca_comm = types.SimpleNamespace(
            disabled=False,
            should_custom_ar=MagicMock(return_value=True),
        )
        tensor = types.SimpleNamespace(
            numel=lambda: 16,
            is_cpu=False,
            shape=(16,),
            dtype=torch.bfloat16,
        )

        with patch.object(
            parallel_state, "outplace_all_reduce", return_value="quick_reduce"
        ) as dispatch:
            result = coordinator.all_reduce(tensor)

        self.assertEqual(result, "quick_reduce")
        coordinator.qr_comm.should_quick_allreduce.assert_called_once_with(tensor)
        coordinator.ca_comm.should_custom_ar.assert_not_called()
        dispatch.assert_called_once_with(
            tensor,
            group_name="test_group",
            outplace_all_reduce_method="qr",
        )

    def test_auto_backend_prefers_vmm_on_gfx950(self):
        props = types.SimpleNamespace(gcnArchName="gfx950:sramecc+:xnack-")
        with (
            patch.dict(
                os.environ,
                {"ROCM_QUICK_REDUCE_IPC_BACKEND": "auto"},
            ),
            patch.object(
                quick_all_reduce.torch.cuda,
                "get_device_properties",
                return_value=props,
            ),
        ):
            backend = quick_all_reduce._select_quick_reduce_ipc_backend(
                torch.device("cuda:0")
            )

        self.assertEqual(backend, "vmm")

    def test_vmm_pool_name_uses_group_wide_generation(self):
        group = object()

        def broadcast_nonce(nonce, src, group):
            self.assertIs(group, group_under_test)
            self.assertEqual(src, 3)
            nonce.fill_(2)

        group_under_test = group
        with (
            patch.object(
                quick_all_reduce.dist,
                "get_process_group_ranks",
                return_value=[3, 2],
            ),
            patch.object(quick_all_reduce.dist, "get_rank", return_value=3),
            patch.object(
                quick_all_reduce.dist,
                "broadcast",
                side_effect=broadcast_nonce,
            ),
            patch.object(quick_all_reduce.os, "urandom", return_value=b"local123"),
        ):
            pool_name = quick_all_reduce._new_vmm_pool_name(
                group, "2_3", torch.device("cpu")
            )

        self.assertRegex(
            pool_name,
            r"^sglang_quickreduce_2_3_g[0-9a-f]{16}$",
        )

    def test_vmm_phase_failure_is_reported_consistently(self):
        def gather_status(statuses, local_status, group):
            self.assertIs(group, group_under_test)
            statuses[:] = [local_status, "RuntimeError: peer allocation failed"]

        group_under_test = object()
        with (
            patch.object(quick_all_reduce.dist, "get_world_size", return_value=2),
            patch.object(
                quick_all_reduce.dist,
                "all_gather_object",
                side_effect=gather_status,
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "rank 1: RuntimeError: peer allocation failed"
            ):
                quick_all_reduce._raise_if_any_vmm_phase_failed(
                    group_under_test, "local allocation", None
                )

    def test_explicit_backend_override_does_not_query_device(self):
        with (
            patch.dict(
                os.environ,
                {"ROCM_QUICK_REDUCE_IPC_BACKEND": "hipipc"},
            ),
            patch.object(
                quick_all_reduce.torch.cuda,
                "get_device_properties",
            ) as get_properties,
        ):
            backend = quick_all_reduce._select_quick_reduce_ipc_backend(
                torch.device("cuda:0")
            )

        self.assertEqual(backend, "hipipc")
        get_properties.assert_not_called()


if __name__ == "__main__":
    unittest.main()
