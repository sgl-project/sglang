"""Unit tests for the FlashInfer PCIe-IPC all-reduce adapter.

These cover the parts that decide whether a reduction reaches the kernels at
all -- world-size gating, workspace sizing, and the shape guard -- because each
of them fails *silently* in production: an unsupported shape is not an error,
it is a fall back to NCCL, which looks exactly like the kernels being slow.
"""

import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.distributed.device_communicators import pcie_ipc_ar
from sglang.srt.distributed.device_communicators.pcie_ipc_ar import (
    PcieIpcCommunicator,
)
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

HIDDEN = 6144


def _make_comm(world_size=8, workspace_cls=None, cpu_group=None, tune=None):
    """Build a communicator with the collaborators stubbed out.

    The constructor is bypassed: it needs a live process group and a CUDA
    device, neither of which this test has, and neither of which the logic
    under test depends on.

    ``_ensure_workspace`` ends by calling ``_tune``, which imports FlashInfer
    before it reads ``_cpu_group``. Neither belongs to the sizing or shape
    logic these helpers serve, and the import fails outright on a CPU runner,
    so tuning is stubbed by default and covered on its own in ``TestTuning``.
    """
    comm = PcieIpcCommunicator.__new__(PcieIpcCommunicator)
    comm.disabled = False
    comm.max_numel = 0
    comm._workspace = None
    comm._bound_stream = None
    comm._group = MagicMock()
    comm._device = torch.device("cpu")
    comm._world_size = world_size
    comm._workspace_cls = workspace_cls or MagicMock()
    comm._build_failed = False
    comm._cpu_group = cpu_group
    comm._tune = tune if tune is not None else (lambda hidden: None)
    return comm


class TestWorldSizeGate(CustomTestCase):
    def test_unsupported_world_size_disables(self):
        """A world size the kernels have no IPC channels for must disable cleanly."""
        for world_size in (1, 3, 5, 6, 7, 16):
            with self.subTest(world_size=world_size):
                with patch.object(
                    pcie_ipc_ar.dist, "get_world_size", return_value=world_size
                ):
                    comm = PcieIpcCommunicator(group=MagicMock(), device=0)
                self.assertTrue(comm.disabled)

    def test_missing_flashinfer_disables_instead_of_raising(self):
        """A build without pcie_ipc_comm must degrade to NCCL, not crash the server."""
        with (
            patch.object(pcie_ipc_ar.dist, "get_world_size", return_value=8),
            patch("builtins.__import__", side_effect=ImportError("no pcie_ipc_comm")),
        ):
            comm = PcieIpcCommunicator(group=MagicMock(), device=0)
        self.assertTrue(comm.disabled)


class TestGroupEligibility(CustomTestCase):
    """Which groups get this backend, which is the rule that broke other models.

    ``"tp" in unique_name`` also matched attention_tp / moe_tp /
    pdmux_prefill_tp. Those groups carry no pynccl or custom all-reduce
    communicator, so a reduction dispatched to them hits an assertion instead
    of falling back, and Qwen3-8B TP4/DP2 and Qwen3-30B-A3B TP4/EP2 fail to
    start with the flag on.
    """

    def test_only_the_tensor_parallel_group_is_eligible(self):
        with envs.SGLANG_ENABLE_PCIE_IPC_ALLREDUCE.override(True):
            self.assertTrue(pcie_ipc_ar.eligible_group("tp", 4))
            for name in ("attention_tp", "moe_tp", "pdmux_prefill_tp", "world", "pp"):
                with self.subTest(group=name):
                    self.assertFalse(pcie_ipc_ar.eligible_group(name, 4))

    def test_symmetric_memory_wins_when_both_are_enabled(self):
        """Two independent opt-in flags; the pre-existing one keeps the path.

        Left undecided, eager execution returns through symmetric memory's NCCL
        branch before the dispatch reaches this backend while compiled
        execution selects IPC, so the same config behaves differently per mode.
        """
        with envs.SGLANG_ENABLE_PCIE_IPC_ALLREDUCE.override(True):
            with patch.object(pcie_ipc_ar, "_symm_mem_enabled", return_value=True):
                with self.assertLogs(pcie_ipc_ar.logger, level="WARNING") as logs:
                    pcie_ipc_ar._warn_symm_mem_wins.cache_clear()
                    self.assertFalse(pcie_ipc_ar.eligible_group("tp", 4))
                self.assertIn("--enable-symm-mem", "\n".join(logs.output))
            with patch.object(pcie_ipc_ar, "_symm_mem_enabled", return_value=False):
                self.assertTrue(pcie_ipc_ar.eligible_group("tp", 4))

    def test_symm_mem_probe_reads_the_real_flag(self):
        """Exercise the probe itself: a patched stand-in cannot catch a rename."""
        for flag in (True, False):
            with self.subTest(flag=flag):
                mod = MagicMock(is_symmetric_memory_enabled=lambda: flag)
                with patch.dict(
                    "sys.modules",
                    {
                        "sglang.srt.distributed.device_communicators."
                        "pynccl_allocator": mod
                    },
                ):
                    self.assertIs(pcie_ipc_ar._symm_mem_enabled(), flag)

    def test_the_symm_mem_probe_target_still_exists(self):
        """Pin the real symbol: the probe swallows exceptions by design.

        Embedded use has no runtime context, so the probe has to tolerate a
        failed read -- which means a renamed symbol would silently report "symm
        mem off" and let this backend take a path it should have left alone.
        """
        from sglang.srt.distributed.device_communicators import pynccl_allocator

        self.assertTrue(hasattr(pynccl_allocator, "is_symmetric_memory_enabled"))

    def test_single_rank_group_is_not_eligible(self):
        with envs.SGLANG_ENABLE_PCIE_IPC_ALLREDUCE.override(True):
            self.assertFalse(pcie_ipc_ar.eligible_group("tp", 1))

    def test_disabled_by_default(self):
        """The backend is opt-in; nothing attaches without the flag."""
        self.assertFalse(pcie_ipc_ar.eligible_group("tp", 4))

    def test_anonymous_group_is_not_eligible(self):
        with envs.SGLANG_ENABLE_PCIE_IPC_ALLREDUCE.override(True):
            self.assertFalse(pcie_ipc_ar.eligible_group(None, 4))
            self.assertFalse(pcie_ipc_ar.eligible_group("anonymous", 4))


class TestWorkspaceRelease(CustomTestCase):
    """The workspace must be released before the groups it was built on.

    FlashInfer collectives on that group while tearing the workspace down, so
    ordering is the whole point: releasing afterwards is not a leak that shows
    up in testing, it is a collective on a destroyed group.
    """

    def _coordinator(self, order):
        from sglang.srt.distributed.parallel_state import GroupCoordinator

        coordinator = GroupCoordinator.__new__(GroupCoordinator)
        coordinator.device_group = MagicMock(name="device_group")
        coordinator.cpu_group = MagicMock(name="cpu_group")
        coordinator.pynccl_comm = None
        coordinator.pymscclpp_comm = None
        coordinator.ca_comm = None
        coordinator.mq_broadcaster = None
        workspace = MagicMock()
        workspace.destroy.side_effect = lambda: order.append("workspace")
        coordinator.pcie_ipc_comm = workspace
        return coordinator

    def test_released_before_the_process_groups(self):
        order = []
        coordinator = self._coordinator(order)
        with patch.object(
            torch.distributed,
            "destroy_process_group",
            side_effect=lambda g: order.append("process_group"),
        ):
            coordinator.destroy()

        self.assertEqual(order[0], "workspace")
        self.assertEqual(order.count("process_group"), 2)
        self.assertIsNone(coordinator.pcie_ipc_comm)

    def test_destroy_without_the_backend_is_a_no_op(self):
        order = []
        coordinator = self._coordinator(order)
        coordinator.pcie_ipc_comm = None
        with patch.object(
            torch.distributed,
            "destroy_process_group",
            side_effect=lambda g: order.append("process_group"),
        ):
            coordinator.destroy()
        self.assertEqual(order, ["process_group", "process_group"])


class TestWorkspaceSizing(CustomTestCase):
    def test_sized_for_decode_not_prefill(self):
        """The default bound is the decode width, so prefill chunks stay on NCCL.

        Sizing for a prefill chunk was measured 66% worse on TTFT and bought
        nothing on TPOT, so this is the behaviour that must not regress.
        """
        comm = _make_comm()
        with patch.object(pcie_ipc_ar, "_decode_width", return_value=64):
            self.assertTrue(
                comm._ensure_workspace(torch.empty(1, HIDDEN, dtype=torch.bfloat16))
            )
        self.assertEqual(comm.max_numel, 64 * HIDDEN)

        prefill = torch.empty(16384, HIDDEN, dtype=torch.bfloat16)
        self.assertGreater(prefill.numel(), comm.max_numel)
        self.assertFalse(comm.should_pcie_ipc_ar(prefill))

    def test_falls_back_when_server_args_absent(self):
        """Embedded/unit use has no server args; the bound must still be finite."""
        comm = _make_comm()
        with patch.object(pcie_ipc_ar, "_decode_width", return_value=None):
            comm._ensure_workspace(torch.empty(1, HIDDEN, dtype=torch.bfloat16))
        self.assertEqual(comm.max_numel, pcie_ipc_ar._FALLBACK_DECODE_WIDTH * HIDDEN)

    def test_env_override_wins(self):
        comm = _make_comm()
        with envs.SGLANG_PCIE_IPC_MAX_NUMEL.override(123456):
            with patch.object(pcie_ipc_ar, "_decode_width", return_value=64):
                comm._ensure_workspace(torch.empty(1, HIDDEN, dtype=torch.bfloat16))
        self.assertEqual(comm.max_numel, 123456)

    def test_build_failure_disables_once(self):
        """A workspace that cannot be allocated must not be retried every layer."""
        cls = MagicMock(side_effect=RuntimeError("out of IPC handles"))
        comm = _make_comm(workspace_cls=cls)
        with patch.object(pcie_ipc_ar, "_decode_width", return_value=64):
            self.assertFalse(
                comm._ensure_workspace(torch.empty(1, HIDDEN, dtype=torch.bfloat16))
            )
            self.assertFalse(
                comm._ensure_workspace(torch.empty(1, HIDDEN, dtype=torch.bfloat16))
            )
        self.assertTrue(comm.disabled)
        self.assertEqual(cls.call_count, 1)


class TestShapeGuard(CustomTestCase):
    def _ready_comm(self):
        comm = _make_comm()
        with patch.object(pcie_ipc_ar, "_decode_width", return_value=64):
            comm._ensure_workspace(torch.empty(1, HIDDEN, dtype=torch.bfloat16))
        comm._workspace.supports.return_value = True
        return comm

    def test_delegates_to_flashinfer_supports(self):
        """Coverage is FlashInfer's decision; a rejected shape keeps the NCCL path."""
        comm = self._ready_comm()
        inp = torch.empty(4, HIDDEN, dtype=torch.bfloat16)
        self.assertTrue(comm.should_pcie_ipc_ar(inp))

        comm._workspace.supports.return_value = False
        self.assertFalse(comm.should_pcie_ipc_ar(inp))

    def test_rejects_noncontiguous_and_1d(self):
        comm = self._ready_comm()
        self.assertFalse(
            comm.should_pcie_ipc_ar(torch.empty(4, HIDDEN, dtype=torch.bfloat16).t())
        )
        self.assertFalse(
            comm.should_pcie_ipc_ar(torch.empty(HIDDEN, dtype=torch.bfloat16))
        )

    def test_rejects_dtypes_that_were_never_tuned(self):
        """FlashInfer keys tuning results by dtype, so fp16 would run untuned.

        This adapter only ever tunes bf16, so anything else must keep its NCCL
        path rather than run the seed policy while looking tuned.
        """
        comm = self._ready_comm()
        for dtype in (torch.float16, torch.float32):
            with self.subTest(dtype=dtype):
                self.assertFalse(
                    comm.should_pcie_ipc_ar(torch.empty(4, HIDDEN, dtype=dtype))
                )

    def test_disabled_communicator_never_claims_a_tensor(self):
        comm = self._ready_comm()
        comm.disabled = True
        self.assertFalse(
            comm.should_pcie_ipc_ar(torch.empty(4, HIDDEN, dtype=torch.bfloat16))
        )


class TestDecodeWidth(CustomTestCase):
    """The width must come from the server's own decode config.

    Reading it from ``ServerArgs`` raised ``AttributeError`` on every call once
    upstream moved the resolved config onto the exec bag, and a bare ``except``
    turned that into a permanent, silent fallback: whatever ``--cuda-graph-max-bs``
    the operator passed, the workspace was sized for a fixed 64 rows.
    """

    def _context(self, max_bs, algorithm=None, draft_tokens=None):
        cg = MagicMock()
        cg.decode.max_bs = max_bs
        exec_bag = MagicMock()
        exec_bag.graph.cuda_graph_config = cg
        spec = MagicMock()
        spec.speculative_algorithm = algorithm
        spec.speculative_num_draft_tokens = draft_tokens
        return {
            "sglang.srt.runtime_context": MagicMock(
                get_exec=lambda: exec_bag, get_spec=lambda: spec
            )
        }

    def test_width_follows_the_resolved_decode_max_bs(self):
        with patch.dict("sys.modules", self._context(max_bs=128)):
            self.assertEqual(pcie_ipc_ar._decode_width(), 128)

    def test_width_multiplies_by_the_speculative_draft_width(self):
        """max_bs counts requests; a verify pass carries several rows each."""
        with patch.dict(
            "sys.modules", self._context(max_bs=64, algorithm="EAGLE", draft_tokens=4)
        ):
            self.assertEqual(pcie_ipc_ar._decode_width(), 256)

    def test_no_speculation_means_one_row_per_request(self):
        with patch.dict(
            "sys.modules", self._context(max_bs=64, algorithm=None, draft_tokens=4)
        ):
            self.assertEqual(pcie_ipc_ar._decode_width(), 64)

    def test_absent_config_is_reported_not_swallowed(self):
        """The fallback is fine; being quiet about it is what hid this."""
        broken = MagicMock()
        broken.graph.cuda_graph_config = None
        with patch.dict(
            "sys.modules",
            {"sglang.srt.runtime_context": MagicMock(get_exec=lambda: broken)},
        ):
            self.assertIsNone(pcie_ipc_ar._decode_width())

    def test_the_attribute_path_it_reads_still_exists(self):
        """Mocks cannot catch a field that upstream renamed or moved.

        The defect this class covers was exactly that: the derivation kept
        reading a path that had stopped existing, and the tests around it were
        all mocked, so nothing noticed. Pin the real chain.
        """
        from sglang.srt import runtime_context
        from sglang.srt.model_executor.cuda_graph_config import (
            default_cuda_graph_config,
        )

        self.assertTrue(hasattr(runtime_context, "get_exec"))
        self.assertTrue(hasattr(runtime_context, "get_spec"))
        self.assertTrue(hasattr(default_cuda_graph_config().decode, "max_bs"))

    def test_fallback_width_is_announced(self):
        comm = _make_comm()
        with patch.object(pcie_ipc_ar, "_decode_width", return_value=None):
            with self.assertLogs(pcie_ipc_ar.logger, level="WARNING") as logs:
                comm._ensure_workspace(torch.empty(1, HIDDEN, dtype=torch.bfloat16))
        self.assertIn("fallback width", "\n".join(logs.output))
        self.assertEqual(comm.max_numel, pcie_ipc_ar._FALLBACK_DECODE_WIDTH * HIDDEN)


class TestTuning(CustomTestCase):
    """Every way tuning declines leaves the kernels on FlashInfer's seed policy.

    ``tune()`` returning is not evidence it measured anything: it declines
    shapes silently, and the three guards below return before calling it at
    all. That is indistinguishable from a successful tune unless asserted,
    which is how two earlier revisions of this adapter were misread.
    """

    def _comm(self, cpu_group=object(), tuned=("shape",)):
        comm = _make_comm(cpu_group=cpu_group)
        comm._workspace = MagicMock()
        comm._workspace.tune.return_value = list(tuned)
        comm.max_numel = 64 * HIDDEN
        return comm

    @staticmethod
    def _flashinfer(is_tuning_mode=False):
        """Stand in for FlashInfer, which a CPU runner does not have installed."""
        autotuner = MagicMock()
        autotuner.AutoTuner.get.return_value.is_tuning_mode = is_tuning_mode
        return {"flashinfer": MagicMock(), "flashinfer.autotuner": autotuner}

    def _run_tune(self, comm, is_tuning_mode=False, capturing=False):
        with (
            patch.dict("sys.modules", self._flashinfer(is_tuning_mode)),
            patch.object(
                torch.cuda, "is_current_stream_capturing", return_value=capturing
            ),
        ):
            PcieIpcCommunicator._tune(comm, HIDDEN)

    def test_autotune_disabled_reads_the_real_flag(self):
        """Exercise the helper itself, not a patched stand-in.

        Patching _autotune_disabled in the tests above cannot catch a wrong or
        renamed field inside it -- the same blind spot that let _decode_width
        read a path that had stopped existing.
        """
        for flag in (True, False):
            with self.subTest(flag=flag):
                bag = MagicMock()
                bag.kernel.disable_flashinfer_autotune = flag
                with patch.dict(
                    "sys.modules",
                    {"sglang.srt.runtime_context": MagicMock(get_exec=lambda: bag)},
                ):
                    self.assertIs(pcie_ipc_ar._autotune_disabled(), flag)

    def test_the_flag_still_exists_on_the_exec_bag(self):
        from sglang.srt.arg_groups.fields.exec_ import ExecKernel

        self.assertIn("disable_flashinfer_autotune", ExecKernel.__annotations__)

    def test_honours_disable_flashinfer_autotune(self):
        """--disable-flashinfer-autotune must stop this tuning path too.

        The flag gates should_run_flashinfer_autotune, which gates prepare().
        The lazy build on the first reduction is a second way in, and it used to
        tune and write a cache while the operator had asked for neither.
        """
        comm = self._comm()
        with patch.object(pcie_ipc_ar, "_autotune_disabled", return_value=True):
            with self.assertLogs(pcie_ipc_ar.logger, level="INFO") as logs:
                self._run_tune(comm)
        self.assertIn("--disable-flashinfer-autotune", "\n".join(logs.output))
        comm._workspace.tune.assert_not_called()

    def test_declines_inside_another_autotune_context(self):
        """FlashInfer will not profile a collective from a context it did not open."""
        comm = self._comm()
        with self.assertLogs(pcie_ipc_ar.logger, level="WARNING") as logs:
            self._run_tune(comm, is_tuning_mode=True)
        self.assertIn("another autotune context", "\n".join(logs.output))
        comm._workspace.tune.assert_not_called()

    def test_declines_without_a_host_group(self):
        """The autotuner rendezvouses on the host; no CPU group means no measurement."""
        comm = self._comm(cpu_group=None)
        with self.assertLogs(pcie_ipc_ar.logger, level="WARNING") as logs:
            self._run_tune(comm)
        self.assertIn("no host group", "\n".join(logs.output))
        comm._workspace.tune.assert_not_called()

    def test_declines_under_graph_capture(self):
        """Autotuning replays kernels, which cannot happen inside a capture."""
        comm = self._comm()
        with self.assertLogs(pcie_ipc_ar.logger, level="WARNING") as logs:
            self._run_tune(comm, capturing=True)
        self.assertIn("capture", "\n".join(logs.output).lower())
        comm._workspace.tune.assert_not_called()

    def test_warns_when_tune_covered_no_shapes(self):
        """tune() declining every shape must not read as a successful tune."""
        comm = self._comm(tuned=())
        with self.assertLogs(pcie_ipc_ar.logger, level="WARNING") as logs:
            self._run_tune(comm)
        comm._workspace.tune.assert_called_once()
        self.assertIn("covered no shapes", "\n".join(logs.output))

    def test_reports_the_shapes_it_measured(self):
        comm = self._comm(tuned=("a", "b", "c"))
        with self.assertLogs(pcie_ipc_ar.logger, level="INFO") as logs:
            self._run_tune(comm)
        comm._workspace.tune.assert_called_once()
        _, kwargs = comm._workspace.tune.call_args
        self.assertEqual(kwargs["dtype"], torch.bfloat16)
        self.assertIs(kwargs["tune_group"], comm._cpu_group)
        self.assertIn("autotuned 3 shape(s)", "\n".join(logs.output))

    def test_tune_failure_keeps_the_seed_policy(self):
        """A raising autotuner must not take the server down with it."""
        comm = self._comm()
        comm._workspace.tune.side_effect = RuntimeError("nvlink probe failed")
        with self.assertLogs(pcie_ipc_ar.logger, level="WARNING"):
            self._run_tune(comm)


if __name__ == "__main__":
    unittest.main()
