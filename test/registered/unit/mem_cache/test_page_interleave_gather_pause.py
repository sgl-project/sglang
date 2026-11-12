"""Only an active page-interleaved prefill pauses its multicast prefix gather.

CPU stand-ins check forward scoping, allocation, and grouped-launch ordering.
The multi-GPU page-interleave tests cover stream execution and gathered KV bytes.
"""

import threading
import unittest
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.distributed import parallel_state
from sglang.srt.distributed.device_communicators import gated_transfer
from sglang.srt.distributed.device_communicators.gated_transfer import Link
from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.environ import envs
from sglang.srt.mem_cache.page_interleave_pool import PageInterleaveKVPoolMixin
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Gate:
    def __init__(self):
        self.link, self.device = Link.NVLINK, 0
        self._lock = threading.Lock()
        self._pause_thread = self._pause_stream = None
        self.pending = True
        self.events = []

    def busy(self):
        return self.pending

    def pause(self, stream):
        self.events.append("pause")

    def resume(self, stream):
        self.events.append("resume")


class TestPageInterleaveGatherPause(CustomTestCase):
    def setUp(self):
        self.gate = _Gate()
        self.pool = PageInterleaveKVPoolMixin.__new__(PageInterleaveKVPoolMixin)
        self.pool._shard_extend_active = True
        self.pool._block_pages = 1
        self.pool._multicast_gather = SimpleNamespace(
            chain=SimpleNamespace(link=Link.NVLINK)
        )
        self.enterContext(envs.SGLANG_ENABLE_KV_SHARD_GATHER_PAUSE.override(True))
        self.enterContext(
            forward_context(
                ForwardContext(attn_backend=SimpleNamespace(token_to_kv_pool=self.pool))
            )
        )
        self.expect_pause = True
        self.grouped = None
        self.comm = SimpleNamespace(
            disabled=False,
            change_state=lambda **kwargs: nullcontext(),
            should_torch_symm_mem_allreduce=lambda tensor: True,
            should_mscclpp_allgather=lambda output, tensor: True,
            all_reduce=self._all_reduce,
            outplace_all_reduce=lambda tensor, out: self._all_reduce(tensor, out),
            all_gather=self._all_gather,
            group_start=self._group_start,
            group_end=self._group_end,
        )
        for owner, name, value in (
            (gated_transfer, "_gates", (self.gate,)),
            (torch.cuda, "current_device", lambda: 0),
            (torch.cuda, "current_stream", lambda device=None: "forward-stream"),
            (torch.cuda, "is_current_stream_capturing", lambda: False),
            (parallel_state, "is_hip", lambda: False),
        ):
            patcher = patch.object(owner, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _submitted(self, operation):
        self.assertEqual(self.gate._pause_thread is not None, self.expect_pause)
        self.gate.events.append(operation)

    def _all_reduce(self, tensor, out=None, **kwargs):
        self._submitted("all_reduce")
        if out is None:
            out = tensor
        torch.mul(tensor, 2, out=out)
        return out

    def _all_gather(self, output, tensor, sizes=None, **kwargs):
        self._submitted("all_gather")
        if self.grouped is not None:
            self.grouped.append((output, tensor, sizes))
            return
        self._fill_gather(output, tensor, sizes)

    @staticmethod
    def _fill_gather(output, tensor, sizes):
        if sizes is None:
            output.view(2, *tensor.shape).copy_(tensor.unsqueeze(0))
        else:
            output[: sizes[0]].copy_(tensor)
            output[sizes[0] :].fill_(7)

    def _group_start(self):
        self._submitted("group_start")
        self.grouped = []

    def _group_end(self):
        # NCCL submits the queued operations here, not in each all_gather.
        self._submitted("group_end")
        for output, tensor, sizes in self.grouped:
            self._fill_gather(output, tensor, sizes)
        self.grouped = None

    def _group(self, backend="pynccl", symmetric=False):
        return SimpleNamespace(
            world_size=2,
            rank_in_group=0,
            device_group=None,
            ca_comm=None,
            qr_comm=None,
            pcie_ipc_comm=None,
            pymscclpp_comm=self.comm if backend == "pymscclpp" else None,
            torch_symm_mem_comm=self.comm if backend == "torch_symm_mem" else None,
            get_available_pynccl_comm=lambda: (
                self.comm if backend == "pynccl" else None
            ),
            require_pynccl_comm=lambda operation: self.comm,
            is_symmetric_memory_enabled=lambda: symmetric,
            debug_check_symmetric_mempool=lambda *args: None,
            use_symmetric_memory=lambda *args, **kwargs: nullcontext(),
        )

    @contextmanager
    def _check_allocations(self, name):
        original = getattr(torch, name)
        allocations = []

        def allocate(*args, **kwargs):
            self.assertIsNone(self.gate._pause_thread, "allocation inside pause")
            allocations.append(name)
            return original(*args, **kwargs)

        with patch.object(torch, name, allocate):
            yield allocations

    def test_all_reduce_dispatch_and_output_allocation(self):
        for backend in ("pynccl", "torch"):
            with self.subTest(backend=backend):
                self.gate.events.clear()
                tensor = torch.arange(4, dtype=torch.float32)
                expected = tensor * 2
                with patch.object(torch.distributed, "all_reduce", self._all_reduce):
                    GroupCoordinator._all_reduce_in_place(self._group(backend), tensor)
                torch.testing.assert_close(tensor, expected)
                self.assertEqual(self.gate.events, ["pause", "all_reduce", "resume"])

        for backend in ("pynccl",):
            with self.subTest(outplace=backend):
                self.gate.events.clear()
                tensor = torch.arange(4, dtype=torch.float32)
                original = tensor.clone()
                with self._check_allocations("empty_like") as allocations:
                    output = GroupCoordinator._all_reduce_out_place(
                        self._group(backend), tensor, backend
                    )
                self.assertEqual(allocations, ["empty_like"])
                torch.testing.assert_close(tensor, original)
                torch.testing.assert_close(output, original * 2)
                self.assertEqual(self.gate.events, ["pause", "all_reduce", "resume"])

    def test_all_gather_dispatch(self):
        for backend, symmetric in (
            ("pynccl", False),
            ("pynccl", True),
            ("torch", False),
        ):
            with self.subTest(backend=backend, symmetric=symmetric):
                self.gate.events.clear()
                self.comm.disabled = symmetric
                tensor = torch.arange(6, dtype=torch.float32).view(2, 3)
                output = torch.empty(4, 3)
                with patch.object(
                    parallel_state, "all_gather_single", self._all_gather
                ):
                    GroupCoordinator._all_gather_into_tensor(
                        self._group(backend, symmetric), output, tensor
                    )
                torch.testing.assert_close(output, tensor.repeat(2, 1))
                self.assertEqual(self.gate.events, ["pause", "all_gather", "resume"])

    def test_all_gather_list_output(self):
        tensor = torch.arange(4, dtype=torch.float32)
        outputs = [torch.empty_like(tensor) for _ in range(2)]

        def gather(output_list, input_, **kwargs):
            self._submitted("all_gather")
            for output in output_list:
                output.copy_(input_)

        with patch.object(torch.distributed, "all_gather", gather):
            GroupCoordinator.all_gather(
                self._group(), tensor, output_tensor_list=outputs
            )
        for output in outputs:
            torch.testing.assert_close(output, tensor)
        self.assertEqual(self.gate.events, ["pause", "all_gather", "resume"])

    def test_all_gatherv_allocates_before_group_and_resumes_after_group_end(self):
        inputs = [torch.ones(2, 3), torch.full((2, 1), 4.0)]
        with self._check_allocations("empty") as allocations:
            outputs = GroupCoordinator.all_gatherv(self._group(), inputs, sizes=[2, 3])
        self.assertEqual(allocations, ["empty", "empty"])
        for output, tensor in zip(outputs, inputs):
            torch.testing.assert_close(output[:2], tensor)
            torch.testing.assert_close(output[2:], torch.full_like(output[2:], 7))
        self.assertEqual(
            self.gate.events,
            ["pause", "group_start", "all_gather", "all_gather", "group_end", "resume"],
        )

    def test_all_gatherv_reuses_supplied_output(self):
        tensor = torch.ones(2, 3)
        output = torch.empty(4, 3)
        with self._check_allocations("empty") as allocations:
            result = GroupCoordinator.all_gatherv(self._group(), tensor, output=output)
        self.assertEqual(allocations, [])
        self.assertIs(result[0], output)
        torch.testing.assert_close(output, tensor.repeat(2, 1))

    def test_collective_failure_releases_gate_and_closes_nccl_group(self):
        def fail(*args, **kwargs):
            self._submitted("failure")
            raise RuntimeError("collective failed")

        self.comm.all_reduce = fail
        with self.assertRaisesRegex(RuntimeError, "collective failed"):
            GroupCoordinator._all_reduce_in_place(self._group(), torch.ones(4))
        self.assertIsNone(self.gate._pause_thread)
        self.assertEqual(self.gate.events, ["pause", "failure", "resume"])

        self.gate.events.clear()
        self.comm.all_gather = fail
        with self.assertRaisesRegex(RuntimeError, "collective failed"):
            GroupCoordinator.all_gatherv(self._group(), torch.ones(2, 3))
        self.assertIsNone(self.gate._pause_thread)
        self.assertEqual(
            self.gate.events, ["pause", "group_start", "failure", "group_end", "resume"]
        )

    def test_no_pause_outside_the_active_page_interleaved_prefill(self):
        self.expect_pause = False
        cases = (
            ("disabled", envs.SGLANG_ENABLE_KV_SHARD_GATHER_PAUSE.override(False)),
            (
                "no_forward",
                patch("sglang.srt.model_executor.forward_context._current", None),
            ),
            (
                "ordinary_pool",
                forward_context(
                    ForwardContext(
                        attn_backend=SimpleNamespace(token_to_kv_pool=object())
                    )
                ),
            ),
            ("inactive", patch.object(self.pool, "_shard_extend_active", False)),
            ("empty_prefix", patch.object(self.pool, "_block_pages", 0)),
            ("nccl_prefix", patch.object(self.pool, "_multicast_gather", None)),
            ("ungated", patch.object(self.pool._multicast_gather.chain, "link", None)),
            (
                "compiling",
                patch.object(torch.compiler, "is_compiling", return_value=True),
            ),
        )
        for name, context in cases:
            with self.subTest(name=name), context:
                self.gate.events.clear()
                tensor, output = torch.ones(2), torch.empty(4)
                GroupCoordinator._all_reduce_in_place(self._group(), tensor)
                GroupCoordinator._all_gather_into_tensor(self._group(), output, tensor)
                torch.testing.assert_close(output, torch.full((4,), 2.0))
                self.assertEqual(self.gate.events, ["all_reduce", "all_gather"])

    def test_idle_and_capture_still_run_collectives_without_pausing(self):
        self.expect_pause = False
        for capturing in (False, True):
            with self.subTest(capturing=capturing):
                self.gate.events.clear()
                self.gate.pending = capturing
                tensor = torch.ones(2)
                output = torch.empty(4)
                with patch.object(
                    torch.cuda, "is_current_stream_capturing", return_value=capturing
                ):
                    GroupCoordinator._all_reduce_in_place(self._group(), tensor)
                    GroupCoordinator._all_gather_into_tensor(
                        self._group(), output, tensor
                    )
                torch.testing.assert_close(output, torch.full((4,), 2.0))
                self.assertEqual(self.gate.events, ["all_reduce", "all_gather"])


if __name__ == "__main__":
    unittest.main()
