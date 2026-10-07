"""Unit tests for when TP all-gathers take FlashInfer's PCIe-IPC all-gather.

Each path here fails silently in production: a gather that should have moved
to PCIe-IPC and did not is a fall back to NCCL, which only shows up as slower
decode. The bit-exactness of the gather itself is covered on GPUs by
``test/registered/kernels/ops/communication/test_pcie_ipc_vocab_gather.py``.
"""

import contextlib
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import torch.distributed as dist

from sglang.srt.distributed.device_communicators import (
    triton_symm_mem_ag,
    vocab_gather,
)
from sglang.srt.distributed.device_communicators.pcie_ipc_ar import (
    PcieIpcCommunicator,
)
from sglang.srt.distributed.device_communicators.vocab_gather import (
    NcclVocabGather,
    PcieIpcVocabGather,
    make_vocab_gather,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

WIDTH = 64


def _group(world_size=4, pcie_ipc=True):
    group = MagicMock()
    group.world_size = world_size
    group.pcie_ipc_eligible = pcie_ipc
    group.pcie_ipc_comm = MagicMock(disabled=False) if pcie_ipc else None
    return group


class _FakeWorkspace:
    """Stands in for the FlashInfer workspace: rank r's slice is ``local * (r + 1)``
    and the output is rank-major, as FlashInfer returns it."""

    def __init__(self, group, max_numel, dtype):
        self.world_size = 4
        self.max_numel = max_numel
        self.dtype = dtype
        self.rebinds = 0
        self.configs = []
        self.destroyed = False

    def supports(self, x):
        return x.dtype == self.dtype and x.numel() <= self.max_numel

    def rebind_stream(self):
        self.rebinds += 1

    def destroy(self):
        self.destroyed = True

    def all_gather(self, x, *, config):
        self.configs.append(config)
        return torch.cat([x * (r + 1) for r in range(self.world_size)])


@contextlib.contextmanager
def _fake_flashinfer(
    workspace_cls=_FakeWorkspace, free_bytes=1 << 40, one_rank=True, missing=()
):
    """FlashInfer, free GPU memory and (with ``one_rank``) the cross-rank
    agreement, stubbed for a CPU process. ``free_bytes`` may be an exception
    for ``torch.cuda.mem_get_info`` to raise; ``missing`` names API symbols
    this FlashInfer lacks."""
    module = types.ModuleType("flashinfer.comm")
    module.PcieIpcAllGatherWorkspace = workspace_cls
    module.PcieIpcAllGatherLaunchConfig = lambda *a: ("config",) + a
    module.PcieIpcAllGatherVariant = MagicMock()
    for name in missing:
        delattr(module, name)
    if isinstance(free_bytes, Exception):
        mem_get_info = dict(side_effect=free_bytes)
    else:
        mem_get_info = dict(return_value=(free_bytes, 1 << 40))
    package = types.ModuleType("flashinfer")
    package.comm = module
    with contextlib.ExitStack() as stack:
        stack.enter_context(
            patch.dict(sys.modules, {"flashinfer": package, "flashinfer.comm": module})
        )
        stack.enter_context(patch.object(torch.cuda, "mem_get_info", **mem_get_info))
        if one_rank:
            stack.enter_context(
                patch.object(
                    vocab_gather,
                    "_count_ranks",
                    side_effect=lambda g, ok: int(ok) * g.world_size,
                )
            )
        yield


class TestMakeVocabGather(CustomTestCase):
    def _make(self, group, symm_rows=16, prefer_nvlink=True, prefer_pcie_ipc=True):
        with _fake_flashinfer():
            return make_vocab_gather(
                group,
                local_width=WIDTH,
                prefer_nvlink=prefer_nvlink,
                prefer_pcie_ipc=prefer_pcie_ipc,
                symm_rows=symm_rows,
            )

    def test_tp4_with_pcie_ipc_all_reduce_takes_pcie_ipc(self):
        group = _group()
        gather = self._make(group)
        self.assertIsInstance(gather, PcieIpcVocabGather)
        self.assertIsInstance(gather.fallback, NcclVocabGather)
        self.assertEqual(gather.workspace.max_numel, 16 * WIDTH)
        group.pcie_ipc_comm.adopt.assert_called_once_with(gather.workspace)

    def test_workspace_is_released_with_the_all_reduce_workspace(self):
        """Its destroy is collective on the group, so it has to go in
        GroupCoordinator.destroy, before the process groups."""
        order = []
        comm = PcieIpcCommunicator.__new__(PcieIpcCommunicator)
        comm._adopted = []
        comm._workspace = MagicMock()
        comm._workspace.destroy.side_effect = lambda: order.append("all_reduce")
        for name in ("vocab", "logits"):
            ws = MagicMock()
            ws.destroy.side_effect = lambda name=name: order.append(name)
            comm.adopt(ws)
        comm.destroy()
        self.assertEqual(order, ["vocab", "logits", "all_reduce"])
        self.assertEqual(comm._adopted, [])

    def test_without_nvlink_preference_still_takes_pcie_ipc(self):
        gather = self._make(_group(), prefer_nvlink=False)
        self.assertIsInstance(gather, PcieIpcVocabGather)

    def test_call_sites_that_did_not_opt_in_stay_on_nccl(self):
        gather = self._make(_group(), prefer_pcie_ipc=False)
        self.assertIsInstance(gather, NcclVocabGather)

    def test_stays_on_nccl_otherwise(self):
        cases = {
            "no PCIe-IPC all-reduce": (_group(pcie_ipc=False), 16),
            "TP2 (launch config not measured)": (_group(world_size=2), 16),
            "TP8 (launch config not measured)": (_group(world_size=8), 16),
            "no row capacity": (_group(), 0),
        }
        for name, (group, rows) in cases.items():
            with self.subTest(name):
                self.assertIsInstance(
                    self._make(group, symm_rows=rows), NcclVocabGather
                )

    def test_disabled_communicator_stays_on_nccl(self):
        group = _group()
        group.pcie_ipc_comm.disabled = True
        self.assertIsInstance(self._make(group), NcclVocabGather)

    def test_communicator_missing_on_this_rank_stays_on_nccl(self):
        """GroupCoordinator leaves pcie_ipc_comm None when its setup raised on
        this rank; the rank still answers the agreement, as not ready."""
        group = _group()
        group.pcie_ipc_comm = None
        with patch.object(vocab_gather, "PcieIpcVocabGather") as build:
            self.assertIsInstance(self._make(group), NcclVocabGather)
        build.assert_not_called()

    def test_local_probe_failures_stay_on_nccl(self):
        """Any failure of this rank's checks is its answer to the agreement;
        an exception escaping them would leave the other ranks waiting."""
        cases = {
            "no workspace": dict(missing=("PcieIpcAllGatherWorkspace",)),
            "no launch config": dict(missing=("PcieIpcAllGatherLaunchConfig",)),
            "no variant": dict(missing=("PcieIpcAllGatherVariant",)),
            "mem_get_info raises": dict(free_bytes=RuntimeError("CUDA error")),
        }
        for name, kwargs in cases.items():
            with self.subTest(name):
                group = _group()
                with _fake_flashinfer(**kwargs):
                    with self.assertLogs(vocab_gather.logger, level="WARNING"):
                        ready = vocab_gather._can_build_pcie_ipc_gather(group, 1)
                    self.assertFalse(ready)
                    gather = make_vocab_gather(
                        group, local_width=WIDTH, prefer_pcie_ipc=True, symm_rows=16
                    )
                self.assertIsInstance(gather, NcclVocabGather)
                group.pcie_ipc_comm.adopt.assert_not_called()

    def test_nvlink_multicast_wins_over_pcie_ipc(self):
        with patch.object(vocab_gather, "_nvlink_ca_comm", return_value=MagicMock()):
            with patch.object(vocab_gather, "NVLinkVocabGather") as nvlink:
                gather = self._make(_group())
        self.assertIs(gather, nvlink.return_value)

    def test_workspace_failure_falls_back_to_nccl(self):
        """FlashInfer raises the same error on every rank (its construction
        checks are joint); the server keeps NCCL."""

        def broken(**kwargs):
            raise RuntimeError("cudaIpcOpenMemHandle failed")

        with _fake_flashinfer(broken):
            with self.assertLogs(vocab_gather.logger, level="WARNING") as logs:
                gather = make_vocab_gather(
                    _group(), local_width=WIDTH, prefer_pcie_ipc=True, symm_rows=16
                )
        self.assertIsInstance(gather, NcclVocabGather)
        self.assertIn("cudaIpcOpenMemHandle", "\n".join(logs.output))

    def test_not_enough_free_memory_stays_on_nccl(self):
        group = _group()
        with _fake_flashinfer(free_bytes=1 << 20):
            gather = make_vocab_gather(
                group, local_width=WIDTH, prefer_pcie_ipc=True, symm_rows=4096
            )
        self.assertIsInstance(gather, NcclVocabGather)
        group.pcie_ipc_comm.adopt.assert_not_called()


def _agreement_rank(rank: int, init_file: str, failing: str) -> None:
    """One of four gloo ranks; rank 1 fails at ``failing``, the others succeed
    (with ``build_all``, every rank fails the build)."""
    dist.init_process_group(
        backend="gloo", init_method=Path(init_file).as_uri(), rank=rank, world_size=4
    )
    try:
        group = SimpleNamespace(
            world_size=4,
            cpu_group=dist.group.WORLD,
            device_group=None,
            pcie_ipc_eligible=True,
            pcie_ipc_comm=MagicMock(disabled=False),
        )
        if failing == "no_comm" and rank == 1:
            group.pcie_ipc_comm = None
        built = []

        def workspace(**kwargs):
            if failing == "build_all" or (failing == "build_one" and rank == 1):
                raise RuntimeError("cudaIpcOpenMemHandle failed")
            built.append(_FakeWorkspace(**kwargs))
            return built[-1]

        free_bytes = 1 << 40
        if rank == 1 and failing == "memory":
            free_bytes = 0
        if rank == 1 and failing == "probe":
            free_bytes = RuntimeError("CUDA error")
        with _fake_flashinfer(workspace, free_bytes=free_bytes, one_rank=False):
            try:
                gather = make_vocab_gather(
                    group, local_width=WIDTH, prefer_pcie_ipc=True, symm_rows=16
                )
            except RuntimeError as e:
                gather = e
        if failing == "build_one":
            # The three workspaces cannot be released without rank 1, so every
            # rank stops instead of serving with them.
            assert isinstance(gather, RuntimeError), (rank, gather)
            assert "built on 3 of 4 ranks" in str(gather), (rank, gather)
            assert len(built) == (0 if rank == 1 else 1), (rank, built)
            assert not any(ws.destroyed for ws in built), rank
        else:
            assert isinstance(gather, NcclVocabGather), (rank, type(gather))
        if group.pcie_ipc_comm is not None:
            group.pcie_ipc_comm.adopt.assert_not_called()
        if failing in ("memory", "no_comm", "probe"):
            # no rank may start the collective build
            assert not built, rank
    finally:
        dist.destroy_process_group()


class TestRanksAgree(CustomTestCase):
    """A rank that cannot build the workspace must take every rank to NCCL;
    one rank on PCIe-IPC and another on NCCL would hang the next gather."""

    def _spawn(self, failing):
        with TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(
                _agreement_rank,
                args=(str(Path(directory) / "gloo-init"), failing),
                nprocs=4,
                join=True,
            )

    def test_one_rank_without_memory_keeps_every_rank_off_the_build(self):
        self._spawn("memory")

    def test_one_rank_without_a_communicator_keeps_every_rank_off_the_build(self):
        self._spawn("no_comm")

    def test_one_rank_probe_exception_keeps_every_rank_off_the_build(self):
        self._spawn("probe")

    def test_build_failing_on_every_rank_takes_every_rank_to_nccl(self):
        self._spawn("build_all")

    def test_build_failing_on_one_rank_stops_every_rank(self):
        self._spawn("build_one")


class TestPcieIpcVocabGather(CustomTestCase):
    def setUp(self):
        self.fallback = MagicMock()
        with _fake_flashinfer():
            self.gather = PcieIpcVocabGather(
                group=_group(),
                local_width=WIDTH,
                dtype=torch.float32,
                max_rows=4,
                fallback=self.fallback,
            )
        streams = patch.object(torch.cuda, "current_stream", return_value="s0")
        streams.start()
        self.addCleanup(streams.stop)

    def test_side_by_side_layout_matches_all_gather_last_dim(self):
        for rows in (1, 3):
            with self.subTest(rows=rows):
                x = torch.randn(rows, WIDTH)
                expected = torch.cat([x * (r + 1) for r in range(4)], dim=-1)
                torch.testing.assert_close(self.gather(x), expected, rtol=0, atol=0)

    def test_stacked_layout_is_rank_major(self):
        x = torch.randn(3, WIDTH)
        expected = torch.cat([x * (r + 1) for r in range(4)], dim=0)
        torch.testing.assert_close(
            self.gather.gather_stacked(x), expected, rtol=0, atol=0
        )

    def test_one_fixed_config_for_every_shape(self):
        """Every rank must launch the same config; FlashInfer's per-shape
        default would fall back to its seed for untuned row counts."""
        for rows in (1, 2, 3, 4):
            self.gather(torch.randn(rows, WIDTH))
        configs = self.gather.workspace.configs
        self.assertEqual(len(configs), 4)
        self.assertTrue(all(c is self.gather.config for c in configs))

    def test_unsupported_slices_go_to_the_fallback(self):
        for x in (
            torch.randn(5, WIDTH),  # past max_rows
            torch.randn(2, WIDTH, dtype=torch.float16),  # other dtype
            # within the element bound but past max_rows
            torch.randn(8, WIDTH // 2),
        ):
            with self.subTest(shape=tuple(x.shape), dtype=x.dtype):
                self.assertIs(self.gather(x), self.fallback.return_value)
                self.assertIs(
                    self.gather.gather_stacked(x),
                    self.fallback.gather_stacked.return_value,
                )
        self.assertEqual(self.gather.workspace.configs, [])

    def test_strided_and_misaligned_slices_are_copied_not_sent_to_nccl(self):
        """Contiguity and alignment are per-rank properties; falling back on
        them could put one rank on NCCL while the others launch the kernel."""
        for x in (
            torch.randn(WIDTH, 2).t(),  # not contiguous
            torch.randn(2 * WIDTH + 1)[1:].view(2, WIDTH),  # not 16-byte aligned
        ):
            with self.subTest(contiguous=x.is_contiguous()):
                expected = torch.cat([x * (r + 1) for r in range(4)], dim=-1)
                torch.testing.assert_close(self.gather(x), expected, rtol=0, atol=0)
        self.assertEqual(len(self.gather.workspace.configs), 2)
        self.fallback.assert_not_called()

    def test_rebinds_only_when_the_stream_changes(self):
        """One workspace serves one ordered stream: rebind on capture -> replay
        transitions, not on every call."""
        x = torch.randn(1, WIDTH)
        with patch.object(
            torch.cuda, "current_stream", side_effect=["s0", "s0", "cap", "cap", "s0"]
        ):
            for _ in range(5):
                self.gather(x)
        self.assertEqual(self.gather.workspace.rebinds, 3)


_REAL = object()


class TestMultimemAllGathererPcieIpc(CustomTestCase):
    """Only an opted-in gatherer (the logits processor's) on a group with
    PCIe-IPC enabled moves to PCIe-IPC, and only where multimem is unavailable;
    everything else keeps its path."""

    def _call(
        self,
        x,
        *,
        pcie_ipc=True,
        comm=True,
        gather=None,
        capturing=False,
        multicast=False,
    ):
        """``gather`` is what make_pcie_ipc_gather returns; ``_REAL`` runs it."""
        gatherer = triton_symm_mem_ag.MultimemAllGatherer(
            128, enabled=False, pcie_ipc=pcie_ipc
        )
        gatherer._state = gatherer._UNINIT
        multimem = MagicMock()
        multimem.symm_mem_hdl.multicast_ptr = 1 if multicast else 0
        multimem.max_token_num = 128
        multimem.world_size = 4
        multimem.hidden_dim = 4 * x.shape[-1]
        nccl_out = torch.zeros(1)
        with (
            patch.object(
                triton_symm_mem_ag,
                "make_pcie_ipc_gather",
                **(
                    dict(wraps=vocab_gather.make_pcie_ipc_gather)
                    if gather is _REAL
                    else dict(return_value=gather)
                ),
            ) as make,
            patch.object(
                triton_symm_mem_ag, "create_state", return_value=multimem
            ) as create_state,
            patch.object(triton_symm_mem_ag, "all_gather_inner") as all_gather_inner,
            patch.object(
                torch.cuda, "is_current_stream_capturing", return_value=capturing
            ),
            patch.object(torch.cuda, "is_available", return_value=True),
            patch(
                "sglang.srt.distributed.parallel_state.get_tp_group",
                return_value=_group(pcie_ipc=comm),
            ),
            patch(
                "sglang.srt.distributed.tensor_model_parallel_all_gather",
                return_value=nccl_out,
            ),
            patch.object(triton_symm_mem_ag, "recommended_max_tokens", return_value=96),
        ):
            out = gatherer(x)
        return SimpleNamespace(
            out=out,
            nccl_out=nccl_out,
            multimem=multimem,
            multimem_out=all_gather_inner.return_value,
            make=make,
            create_state=create_state,
            gatherer=gatherer,
        )

    def test_opted_in_gatherer_without_multicast_uses_pcie_ipc(self):
        gather = MagicMock(spec=PcieIpcVocabGather)
        x = torch.randn(2, WIDTH, dtype=torch.bfloat16)
        r = self._call(x, gather=gather)
        self.assertIs(r.out, gather.return_value)
        self.assertIs(r.gatherer._state, gather)
        # multimem was tried first and found no multicast
        r.create_state.assert_called_once()
        # sized for decode (the smaller bound), not the caller's max_tokens
        self.assertEqual(r.make.call_args.kwargs["max_rows"], 96)
        self.assertEqual(r.make.call_args.kwargs["local_width"], WIDTH)

    def test_multimem_wins_over_pcie_ipc(self):
        """An NVLink host with SGLANG_ENABLE_PCIE_IPC_ALLREDUCE=1 keeps multimem,
        as make_vocab_gather keeps NVLink."""
        x = torch.randn(2, WIDTH, dtype=torch.bfloat16)
        r = self._call(x, gather=MagicMock(spec=PcieIpcVocabGather), multicast=True)
        self.assertIs(r.out, r.multimem_out)
        self.assertIs(r.gatherer._state, r.multimem)
        r.create_state.assert_called_once()
        r.make.assert_not_called()

    def test_not_built_under_capture(self):
        x = torch.randn(2, WIDTH, dtype=torch.bfloat16)
        r = self._call(x, gather=MagicMock(spec=PcieIpcVocabGather), capturing=True)
        self.assertIs(r.out, r.nccl_out)
        r.make.assert_not_called()
        r.create_state.assert_not_called()
        self.assertIs(r.gatherer._state, r.gatherer._UNINIT)

    def test_failed_build_stays_on_nccl(self):
        x = torch.randn(2, WIDTH, dtype=torch.bfloat16)
        r = self._call(x, gather=None)
        self.assertIs(r.out, r.nccl_out)
        self.assertIsNone(r.gatherer._state)
        r.make.assert_called_once()

    def test_other_users_keep_the_multimem_path(self):
        """e.g. Kimi-K2.5 EAGLE3's fc gather, which does not opt in."""
        x = torch.randn(2, WIDTH, dtype=torch.bfloat16)
        r = self._call(x, gather=MagicMock(spec=PcieIpcVocabGather), pcie_ipc=False)
        self.assertIs(r.out, r.nccl_out)
        r.make.assert_not_called()
        r.create_state.assert_called_once()

    def test_group_without_pcie_ipc_keeps_nccl_without_an_agreement(self):
        x = torch.randn(2, WIDTH, dtype=torch.bfloat16)
        with patch.object(vocab_gather, "_count_ranks") as count:
            r = self._call(x, gather=_REAL, comm=False)
        self.assertIs(r.out, r.nccl_out)
        self.assertIsNone(r.gatherer._state)
        count.assert_not_called()


if __name__ == "__main__":
    unittest.main()
