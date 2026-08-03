"""Unit tests for the symmetric-memory DCP A2A workspace lifecycle.

Covers two sides of the same feature with shared CPU mocks:
  * the workspace allocator/reducer in ``sglang.srt.layers.dcp.comm``
    (slab allocation, geometry validation, LSE reduce dispatch)
  * the runner-level pre-init hook in ``BaseRunner`` that allocates the
    symmetric-memory workspace before CUDA graph capture

Both classes are CPU-only (no CUDA), so one CI registration covers them.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.dcp import comm  # noqa: E402
from sglang.srt.layers.dcp import estimate_symm_a2a_workspace_nbytes  # noqa: E402
from sglang.srt.model_executor.runner.base_runner import BaseRunner  # noqa: E402
from sglang.srt.runtime_context import (  # noqa: E402
    get_context,
    get_parallel,
    reset_context,
)


register_cpu_ci(est_time=5, suite="base-a-test-cpu")


# --- constants for the BaseRunner pre-init patches ---
_CUSTOM_AR_V2 = (
    "sglang.srt.distributed.device_communicators.custom_all_reduce_v2."
    "can_use_custom_all_reduce_v2"
)
_SAME_NODE = "sglang.srt.distributed.parallel_state.in_the_same_node_as"
_INIT_WORKSPACE = "sglang.srt.layers.dcp.init_symm_a2a_workspace"
_ESTIMATE_WORKSPACE = "sglang.srt.layers.dcp.estimate_symm_a2a_workspace_nbytes"


# --- shared fake tensors / handles ---
class _FakeTensor:
    def __init__(self, shape, dtype, device="cuda:0", ptr=1):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = torch.device(device)
        self._ptr = ptr
        self.zeroed = False
        self.value = 1
        self.new_empty_calls = 0

    def zero_(self):
        self.zeroed = True
        return self

    def data_ptr(self):
        return self._ptr

    def element_size(self):
        return {
            torch.uint8: 1,
            torch.float16: 2,
            torch.bfloat16: 2,
            torch.float32: 4,
            torch.int64: 8,
        }[self.dtype]

    def item(self):
        return self.value

    def view(self, *shape_or_dtype):
        if len(shape_or_dtype) == 1 and isinstance(shape_or_dtype[0], torch.dtype):
            dtype = shape_or_dtype[0]
            nbytes = self.numel() * self.element_size()
            itemsize = torch.empty((), dtype=dtype).element_size()
            if nbytes % itemsize != 0:
                raise ValueError("dtype view requires byte divisibility")
            return _FakeTensor(
                (nbytes // itemsize,),
                dtype,
                self.device,
                self._ptr,
            )
        shape = shape_or_dtype[0] if len(shape_or_dtype) == 1 else shape_or_dtype
        numel = 1
        for dim in shape:
            numel *= dim
        if numel != self.numel():
            raise ValueError("shape view must preserve numel")
        return _FakeTensor(shape, self.dtype, self.device, self._ptr)

    def numel(self):
        result = 1
        for dim in self.shape:
            result *= dim
        return result

    def new_empty(self, shape):
        self.new_empty_calls += 1
        return _FakeTensor(shape, self.dtype, self.device, ptr=self._ptr + 1000)

    def __getitem__(self, item):
        if isinstance(item, int):
            stride = 1
            for dim in self.shape[1:]:
                stride *= dim
            return _FakeTensor(
                self.shape[1:],
                self.dtype,
                self.device,
                self._ptr + item * stride * self.element_size(),
            )
        if isinstance(item, slice):
            start = item.start or 0
            stop = item.stop if item.stop is not None else self.shape[0]
            stride = 1
            for dim in self.shape[1:]:
                stride *= dim
            return _FakeTensor(
                (stop - start, *self.shape[1:]),
                self.dtype,
                self.device,
                self._ptr + start * stride * self.element_size(),
            )
        if isinstance(item, tuple):
            result = self
            for subitem in item:
                result = result[subitem]
            return result
        return self


class _FakeHandle:
    def __init__(self, views):
        self.views = views
        self.barrier_calls = 0

    def barrier(self):
        self.barrier_calls += 1

    def get_buffer(self, peer, shape, dtype, offset=0):
        return self.views[peer]



class _FakeModelConfig:
    def __init__(self, kv_lora_rank):
        self.kv_lora_rank = kv_lora_rank
        self.attn_tp_size = None

    def get_num_attention_heads(self, attn_tp_size):
        self.attn_tp_size = attn_tp_size
        return 16


class TestDirectSymmA2AWorkspace(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        self.group = SimpleNamespace(
            world_size=2,
            rank_in_group=1,
            device_group=SimpleNamespace(group_name="dcp-device"),
            cpu_group=SimpleNamespace(group_name="dcp-cpu"),
        )

    def _workspace(self, *, num_ubatches=2):
        workspace = object.__new__(comm.DirectSymmA2AWorkspace)
        workspace.cp_group = self.group
        workspace.device = torch.device("cuda:0")
        workspace.world_size = 2
        workspace.rank = 1
        workspace.max_num_tokens = 4
        workspace.heads_per_rank = 3
        workspace.head_dim = 8
        workspace.dtype = torch.float16
        workspace.num_ubatches = num_ubatches
        workspace.received_output = _FakeTensor(
            (num_ubatches, 2, 2, 4, 3, 8), torch.float16
        )
        workspace.received_lse = _FakeTensor((num_ubatches, 2, 2, 4, 3), torch.float32)
        workspace.received_signal = _FakeTensor((num_ubatches, 2, 2), torch.int64)
        workspace.peer_output_ptrs = _FakeTensor((num_ubatches, 2), torch.int64)
        workspace.peer_lse_ptrs = _FakeTensor((num_ubatches, 2), torch.int64)
        workspace.peer_signal_ptrs = _FakeTensor((num_ubatches, 2), torch.int64)
        workspace.epoch = _FakeTensor((num_ubatches,), torch.int64)
        workspace.combined_output = _FakeTensor(
            (num_ubatches, 4, 3, 8), torch.float16, ptr=10000
        )
        return workspace

    def test_workspace_estimator_includes_slab_and_local_tensors(self):
        self.assertEqual(
            comm.estimate_symm_a2a_workspace_nbytes(
                world_size=2,
                max_num_tokens=4,
                heads_per_rank=3,
                head_dim=8,
                dtype=torch.float16,
                num_ubatches=2,
            ),
            2480,
        )

    def test_constructor_validates_geometry(self):
        invalid = [
            ({"max_num_tokens": 0}, "max_num_tokens"),
            ({"heads_per_rank": 0}, "heads_per_rank"),
            ({"head_dim": 7}, "head_dim"),
            ({"dtype": torch.float32}, "dtype"),
            ({"num_ubatches": 0}, "num_ubatches"),
        ]
        defaults = dict(
            cp_group=self.group,
            device=torch.device("cuda:0"),
            max_num_tokens=4,
            heads_per_rank=3,
            head_dim=8,
            dtype=torch.float16,
            num_ubatches=1,
        )
        for overrides, message in invalid:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex((ValueError, TypeError), message):
                    comm.DirectSymmA2AWorkspace(**(defaults | overrides))

    def _bare_workspace(self):
        workspace = object.__new__(comm.DirectSymmA2AWorkspace)
        workspace.cp_group = self.group
        workspace.device = torch.device("cuda:0")
        workspace.world_size = 2
        workspace._allocations = []
        return workspace

    def test_single_slab_is_sliced_and_kept_alive(self):
        storage = _FakeTensor((4096,), torch.uint8, ptr=16)
        peer_views = [
            _FakeTensor((4096,), torch.uint8, ptr=1600),
            _FakeTensor((4096,), torch.uint8, ptr=3200),
        ]
        handle = _FakeHandle(peer_views)

        def fake_pointer_tensor(values, **kwargs):
            return _FakeTensor((len(values), len(values[0])), kwargs["dtype"])

        real_empty = comm.torch.empty

        def fake_empty(shape, **kwargs):
            if shape == ():
                return real_empty(shape, **kwargs)
            return _FakeTensor(shape, kwargs["dtype"], kwargs.get("device", "cuda:0"))

        with (
            patch.object(
                comm.DirectSymmA2AWorkspace,
                "_allocate_slab",
                return_value=(storage, handle, peer_views),
            ) as allocate,
            patch.object(
                comm.torch, "zeros", return_value=_FakeTensor((2,), torch.int64)
            ),
            patch.object(comm.torch, "tensor", side_effect=fake_pointer_tensor),
            patch.object(comm.torch, "empty", side_effect=fake_empty),
        ):
            workspace = comm.DirectSymmA2AWorkspace(
                self.group,
                torch.device("cuda:0"),
                max_num_tokens=4,
                heads_per_rank=3,
                head_dim=8,
                dtype=torch.float16,
                num_ubatches=2,
            )

        allocate.assert_called_once()
        self.assertEqual(allocate.call_args.args[0], 1984)
        self.assertEqual(workspace.received_output.data_ptr() % 16, 0)
        self.assertEqual(workspace.received_lse.data_ptr() % 16, 0)
        self.assertEqual(workspace.received_signal.data_ptr() % 16, 0)
        self.assertEqual(workspace._allocations, [(storage, handle, peer_views)])

    def test_plan_b_uses_one_cpu_group_allocation(self):
        storage = _FakeTensor((4096,), torch.uint8, ptr=16)
        peer_views = [
            _FakeTensor((4096,), torch.uint8, ptr=1600),
            _FakeTensor((4096,), torch.uint8, ptr=3200),
        ]
        handle = _FakeHandle(peer_views)
        with (
            patch(
                "sglang.srt.distributed.device_communicators.custom_all_reduce_v2._allocate_symmetric_memory",
                return_value=(storage, handle),
            ) as allocate,
        ):
            result = self._bare_workspace()._allocate_slab_plan_b(4096)

        self.assertEqual(result, (storage, handle, peer_views))
        allocate.assert_called_once_with(
            4096,
            device=torch.device("cuda:0"),
            group=self.group.cpu_group,
        )

    def test_null_peer_pointer_rejects_plan_b(self):
        storage = _FakeTensor((4096,), torch.uint8, ptr=16)
        views = [
            _FakeTensor((4096,), torch.uint8, ptr=0),
            _FakeTensor((4096,), torch.uint8, ptr=3200),
        ]
        handle = _FakeHandle(views)
        events = []
        workspace = self._bare_workspace()
        with (
            patch.object(
                workspace,
                "_allocate_slab_plan_b",
                return_value=(storage, handle, views),
            ),
            patch.object(storage, "zero_", side_effect=lambda: events.append("zero")),
            patch.object(
                comm.torch.cuda,
                "synchronize",
                side_effect=lambda *_: events.append("synchronize"),
            ),
            patch.object(
                comm.torch.distributed,
                "all_reduce",
                side_effect=lambda *_, **__: events.append("all_reduce"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "at least one DCP rank"):
                workspace._allocate_slab(4096)
        self.assertEqual(events, ["zero", "synchronize", "all_reduce"])
        self.assertEqual(handle.barrier_calls, 0)

    def test_get_buffer_error_is_reported_after_group_verdict(self):
        storage = _FakeTensor((4096,), torch.uint8, ptr=16)
        handle = MagicMock()
        handle.get_buffer.side_effect = [
            RuntimeError("peer mapping failed"),
            _FakeTensor((4096,), torch.uint8, ptr=3200),
        ]
        events = []
        workspace = self._bare_workspace()

        with (
            patch(
                "sglang.srt.distributed.device_communicators.custom_all_reduce_v2._allocate_symmetric_memory",
                return_value=(storage, handle),
            ),
            patch.object(storage, "zero_", side_effect=lambda: events.append("zero")),
            patch.object(
                comm.torch.cuda,
                "synchronize",
                side_effect=lambda *_: events.append("synchronize"),
            ),
            patch.object(
                comm.torch.distributed,
                "all_reduce",
                side_effect=lambda *_, **__: events.append("all_reduce"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "at least one DCP rank"):
                workspace._allocate_slab(4096)

        self.assertEqual(events, ["zero", "synchronize", "all_reduce"])
        self.assertEqual(handle.get_buffer.call_count, 2)

    def test_remote_peer_failure_is_raised_on_every_rank(self):
        storage = _FakeTensor((4096,), torch.uint8, ptr=16)
        views = [
            _FakeTensor((4096,), torch.uint8, ptr=1600),
            _FakeTensor((4096,), torch.uint8, ptr=3200),
        ]
        handle = _FakeHandle(views)
        events = []
        workspace = self._bare_workspace()

        def mark_global_failure(status, **_):
            events.append("all_reduce")
            status.fill_(0)

        with (
            patch.object(
                workspace,
                "_allocate_slab_plan_b",
                return_value=(storage, handle, views),
            ),
            patch.object(storage, "zero_", side_effect=lambda: events.append("zero")),
            patch.object(
                comm.torch.cuda,
                "synchronize",
                side_effect=lambda *_: events.append("synchronize"),
            ),
            patch.object(
                comm.torch.distributed,
                "all_reduce",
                side_effect=mark_global_failure,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "at least one DCP rank"):
                workspace._allocate_slab(4096)

        self.assertEqual(events, ["zero", "synchronize", "all_reduce"])

    def test_local_allocation_error_still_participates_in_group_verdict(self):
        events = []
        workspace = self._bare_workspace()

        with (
            patch.object(
                workspace,
                "_allocate_slab_plan_b",
                side_effect=RuntimeError("local allocation failed"),
            ),
            patch.object(
                comm.torch.distributed,
                "all_reduce",
                side_effect=lambda *_, **__: events.append("all_reduce"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "at least one DCP rank"):
                workspace._allocate_slab(4096)

        self.assertEqual(events, ["all_reduce"])

    def test_lse_reduce_selects_only_requested_ubatch(self):
        workspace = self._workspace(num_ubatches=2)
        attn_out = _FakeTensor((3, 6, 8), torch.float16)
        lse = _FakeTensor((3, 6), torch.float32)
        output = _FakeTensor((3, 3, 8), torch.float16)

        with patch(
            "sgl_kernel.direct_dcp_a2a_lse_reduce",
            return_value=output,
            create=True,
        ) as kernel:
            result = workspace.lse_reduce(
                attn_out,
                lse,
                is_lse_base_on_e=False,
                ubatch_id=1,
                output=output,
            )

        self.assertIs(result, output)
        kwargs = kernel.call_args.kwargs
        self.assertEqual(kwargs["peer_output_ptrs"].shape, (2,))
        self.assertEqual(kwargs["received_output"].shape, (2, 2, 4, 3, 8))
        self.assertEqual(kwargs["epoch"].shape, ())
        self.assertIs(kwargs["combined_output"], output)

    def test_lse_reduce_reuses_independent_ubatch_outputs(self):
        workspace = self._workspace(num_ubatches=2)
        attn_out = _FakeTensor((3, 6, 8), torch.float16)
        lse = _FakeTensor((3, 6), torch.float32)

        with patch(
            "sgl_kernel.direct_dcp_a2a_lse_reduce",
            side_effect=lambda **kwargs: kwargs["combined_output"],
            create=True,
        ):
            first = workspace.lse_reduce(attn_out, lse, ubatch_id=0)
            second = workspace.lse_reduce(attn_out, lse, ubatch_id=1)

        self.assertEqual(first.shape, (3, 3, 8))
        self.assertEqual(second.shape, (3, 3, 8))
        self.assertNotEqual(first.data_ptr(), second.data_ptr())
        self.assertEqual(attn_out.new_empty_calls, 0)

    def test_lse_reduce_rejects_invalid_inputs_and_slot(self):
        workspace = self._workspace(num_ubatches=2)
        valid_out = _FakeTensor((3, 6, 8), torch.float16)
        valid_lse = _FakeTensor((3, 6), torch.float32)
        cases = [
            (valid_out, valid_lse, 2, "ubatch_id"),
            (
                _FakeTensor((5, 6, 8), torch.float16),
                _FakeTensor((5, 6), torch.float32),
                0,
                "max_num_tokens",
            ),
            (
                _FakeTensor((3, 5, 8), torch.float16),
                _FakeTensor((3, 5), torch.float32),
                0,
                "heads",
            ),
            (_FakeTensor((3, 6, 7), torch.float16), valid_lse, 0, "head_dim"),
            (
                _FakeTensor((3, 6, 8), torch.float32),
                valid_lse,
                0,
                r"symm_a2a.*FP16 and BF16.*torch.float32.*--dcp-comm-backend a2a",
            ),
            (valid_out, _FakeTensor((3, 6), torch.float16), 0, "LSE.*float32"),
        ]
        for attn_out, lse, ubatch_id, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex((ValueError, TypeError), message):
                    workspace.lse_reduce(attn_out, lse, ubatch_id=ubatch_id)

    def test_init_is_idempotent_and_rejects_geometry_conflict(self):
        expected = self._workspace(num_ubatches=1)
        geometry = dict(
            device=torch.device("cuda:0"),
            max_num_tokens=4,
            heads_per_rank=3,
            head_dim=8,
            dtype=torch.float16,
            num_ubatches=1,
        )
        with patch.object(comm, "DirectSymmA2AWorkspace", return_value=expected) as cls:
            first = comm.init_symm_a2a_workspace(self.group, **geometry)
            second = comm.init_symm_a2a_workspace(self.group, **geometry)

        self.assertIs(first, expected)
        self.assertIs(second, expected)
        cls.assert_called_once()
        with self.assertRaisesRegex(RuntimeError, "geometry"):
            comm.init_symm_a2a_workspace(
                self.group, **(geometry | {"max_num_tokens": 8})
            )

    def test_init_rejects_different_process_groups(self):
        expected = self._workspace(num_ubatches=1)
        geometry = dict(
            device=torch.device("cuda:0"),
            max_num_tokens=4,
            heads_per_rank=3,
            head_dim=8,
            dtype=torch.float16,
            num_ubatches=1,
        )
        get_context().resources.buffers[comm._SYMM_A2A_WORKSPACE_KEY] = expected
        other_group = SimpleNamespace(
            world_size=2,
            rank_in_group=1,
            device_group=SimpleNamespace(group_name="other-device"),
            cpu_group=SimpleNamespace(group_name="other-cpu"),
        )
        with self.assertRaisesRegex(RuntimeError, "geometry"):
            comm.init_symm_a2a_workspace(other_group, **geometry)

    def test_symm_branch_requires_initialized_workspace(self):
        with self.assertRaisesRegex(AssertionError, "workspace not initialized"):
            comm.dcp_a2a_lse_reduce(
                _FakeTensor((3, 6, 8), torch.float16),
                _FakeTensor((3, 6), torch.float32),
                self.group,
                comm_backend="symm_a2a",
            )

    def test_symm_branch_forwards_ubatch_id(self):
        workspace = self._workspace(num_ubatches=2)
        get_context().resources.buffers[comm._SYMM_A2A_WORKSPACE_KEY] = workspace
        attn_out = _FakeTensor((3, 6, 8), torch.float16)
        lse = _FakeTensor((3, 6), torch.float32)
        expected = _FakeTensor((3, 3, 8), torch.float16)
        workspace.lse_reduce = MagicMock(return_value=expected)

        result = comm.dcp_a2a_lse_reduce(
            attn_out,
            lse,
            self.group,
            is_lse_base_on_e=False,
            comm_backend="symm_a2a",
            ubatch_id=1,
        )

        self.assertIs(result, expected)
        workspace.lse_reduce.assert_called_once_with(
            attn_out,
            lse,
            is_lse_base_on_e=False,
            ubatch_id=1,
        )

    def test_world_size_one_behavior_is_unchanged(self):
        group = SimpleNamespace(world_size=1)
        attn_out = _FakeTensor((3, 6, 8), torch.float16)
        result = comm.dcp_a2a_lse_reduce(
            attn_out,
            _FakeTensor((3, 6), torch.float32),
            group,
            comm_backend="symm_a2a",
            ubatch_id=99,
        )
        self.assertIs(result, attn_out)




class TestSymmA2APreinit(CustomTestCase):
    def setUp(self):
        super().setUp()
        reset_context()
        self.addCleanup(reset_context)

    def _run_preinit(
        self,
        *,
        dcp_size=2,
        backend="symm_a2a",
        can_use=True,
        same_node=(True, True, True, True),
        eager_max_bs=40,
        kv_lora_rank=512,
        capability_probe=None,
        same_node_probe=None,
    ):
        cp_group = SimpleNamespace(cpu_group=object(), world_size=4, rank_in_group=0)
        model_config = _FakeModelConfig(kv_lora_rank)
        init_workspace = MagicMock()
        capability_probe = capability_probe or MagicMock(return_value=can_use)
        same_node_probe = same_node_probe or MagicMock(return_value=list(same_node))

        with (
            get_context().override_server_args(
                dcp_size=dcp_size, dcp_comm_backend=backend
            ) as server_args,
            get_parallel().override(dcp_group=cp_group, attn_tp_size=2),
            patch(_CUSTOM_AR_V2, capability_probe),
            patch(_SAME_NODE, same_node_probe),
            patch(_INIT_WORKSPACE, init_workspace),
        ):
            model_runner = SimpleNamespace(
                server_args=server_args,
                device="cpu",
                dtype=torch.bfloat16,
                model_config=model_config,
                max_decode_logits_rows=lambda: 96,
            )
            runner = SimpleNamespace(
                model_runner=model_runner,
                _eager_max_bs=eager_max_bs,
                _eager_num_tokens_per_req=3,
            )
            BaseRunner._pre_initialize_symm_a2a_workspace(runner)

        return SimpleNamespace(
            cp_group=cp_group,
            model_runner=model_runner,
            model_config=model_config,
            init_workspace=init_workspace,
            capability_probe=capability_probe,
            same_node_probe=same_node_probe,
        )

    def test_gate_is_noop_unless_dcp_symm_a2a_is_configured(self):
        for dcp_size, backend in ((1, "symm_a2a"), (2, "a2a")):
            with self.subTest(dcp_size=dcp_size, backend=backend):
                result = self._run_preinit(dcp_size=dcp_size, backend=backend)
                result.init_workspace.assert_not_called()
                result.capability_probe.assert_not_called()

    def test_initializes_workspace_with_runner_geometry_and_max_decode_tokens(self):
        result = self._run_preinit()

        self.assertEqual(result.model_config.attn_tp_size, 2)
        result.init_workspace.assert_called_once_with(
            result.cp_group,
            device=torch.device("cpu"),
            max_num_tokens=120,
            heads_per_rank=16,
            head_dim=512,
            dtype=torch.bfloat16,
            num_ubatches=1,
        )

    def test_rejects_unsupported_or_multi_node_topology_before_allocation(self):
        for can_use, same_node in (
            (False, (True, True, True, True)),
            (True, (True, False, True, True)),
        ):
            with self.subTest(can_use=can_use, same_node=same_node):
                with self.assertRaisesRegex(RuntimeError, "a2a.*ag_rs|ag_rs.*a2a"):
                    self._run_preinit(can_use=can_use, same_node=same_node)

    def test_capability_failure_still_runs_same_node_collective(self):
        capability_probe = MagicMock(return_value=False)
        same_node_probe = MagicMock(return_value=[True] * 4)

        with self.assertRaises(RuntimeError):
            self._run_preinit(
                capability_probe=capability_probe,
                same_node_probe=same_node_probe,
            )

        capability_probe.assert_called_once()
        same_node_probe.assert_called_once()

    def test_missing_kv_lora_rank_fails_with_mla_specific_message(self):
        with self.assertRaisesRegex(RuntimeError, "MLA.*kv_lora_rank"):
            self._run_preinit(kv_lora_rank=None)

    def test_warns_when_workspace_estimate_exceeds_512_mib(self):
        with (
            patch(_ESTIMATE_WORKSPACE, return_value=512 * 1024**2 + 1) as estimate,
            self.assertLogs(
                "sglang.srt.model_executor.runner.base_runner", level="WARNING"
            ) as logs,
        ):
            result = self._run_preinit()

        estimate.assert_called_once_with(
            world_size=4,
            max_num_tokens=120,
            heads_per_rank=16,
            head_dim=512,
            dtype=torch.bfloat16,
            num_ubatches=1,
        )
        result.init_workspace.assert_called_once()
        self.assertEqual(len(logs.output), 1)
        self.assertIn("512 MiB", logs.output[0])

    def test_runner_uses_shared_workspace_estimator(self):
        with patch(
            _ESTIMATE_WORKSPACE, wraps=estimate_symm_a2a_workspace_nbytes
        ) as estimate:
            self._run_preinit()

        estimate.assert_called_once()



if __name__ == "__main__":
    unittest.main()
