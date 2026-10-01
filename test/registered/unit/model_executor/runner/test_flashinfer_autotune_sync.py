"""FlashInfer autotune must reach the same tactics on every TP rank.

Without a cross-rank reduction each rank's ``argmin`` follows local timing noise
(measured: 20/20 tuned MoE shapes diverged across 4 ranks on gpt-oss-120b). The
reduction holds only if ranks also enter tuning with the same store, so these
cover that gate, the digest it decides on, and the lock that keeps another
process from changing a store mid-tuning.
"""

from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=57, suite="base-a-test-cpu")
register_cuda_ci(est_time=25, stage="base-b-kernel-unit", runner_config="1-gpu-large")

import multiprocessing
import os
import tempfile
import traceback
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist

from sglang.srt.model_executor.runner import base_runner
from sglang.srt.model_executor.runner import flashinfer_autotune as autotune
from sglang.srt.model_executor.runner.flashinfer_autotune import (
    _agree_on_autotune_store,
    _autotune_store_digest,
    _autotune_tactic_sync_group,
    _lock_autotune_store,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.test_utils import CustomTestCase, find_available_port

ENTRY = "v2/0123456789abcdef/entries/{}.json"


def _write_entry(root: Path, name: str, body: str) -> Path:
    path = root / ENTRY.format(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return path


def _gate_worker(rank, world_size, master_port, root, locked, writer):
    """Run the entry gate on one rank; report its decision and surviving entries."""
    try:
        os.environ.update(
            RANK=str(rank),
            WORLD_SIZE=str(world_size),
            MASTER_ADDR="localhost",
            MASTER_PORT=str(master_port),
        )
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        use_store = _agree_on_autotune_store(Path(root), locked, dist.group.WORLD)
        entries = sorted(p.name for p in Path(root).glob("**/entries/*.json"))
        writer.send(("ok", (use_store, entries)))
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        writer.send(("error", f"{e}"))
    finally:
        writer.close()
        if dist.is_initialized():
            dist.destroy_process_group()


class TestAutotuneTacticSyncGroup(CustomTestCase):
    def test_single_rank_has_nobody_to_agree_with(self):
        # A 1-rank group would add a collective per tactic for no agreement.
        tp_group = SimpleNamespace(world_size=1, cpu_group=object())
        self.assertIsNone(_autotune_tactic_sync_group(tp_group))


class TestAutotuneStoreDigest(CustomTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.rank0 = Path(self.tmp.name) / "rank0"
        self.rank1 = Path(self.tmp.name) / "rank1"
        for root in (self.rank0, self.rank1):
            root.mkdir()

    def test_same_entries_digest_alike_across_roots(self):
        # Ranks keep separate roots; only what they would hit may differ.
        for root in (self.rank0, self.rank1):
            _write_entry(root, "a", '{"tactic": 7}')
        self.assertEqual(
            _autotune_store_digest(self.rank0), _autotune_store_digest(self.rank1)
        )

    def test_empty_stores_digest_alike(self):
        self.assertEqual(
            _autotune_store_digest(self.rank0), _autotune_store_digest(self.rank1)
        )

    def test_entry_content_decides(self):
        _write_entry(self.rank0, "a", '{"tactic": 7}')
        _write_entry(self.rank1, "a", '{"tactic": 8}')
        self.assertNotEqual(
            _autotune_store_digest(self.rank0), _autotune_store_digest(self.rank1)
        )

    def test_a_missing_entry_decides(self):
        _write_entry(self.rank0, "a", '{"tactic": 7}')
        self.assertNotEqual(
            _autotune_store_digest(self.rank0), _autotune_store_digest(self.rank1)
        )

    def test_environment_namespace_decides(self):
        # Same entry under another environment hash: that rank reads nothing.
        _write_entry(self.rank0, "a", '{"tactic": 7}')
        other = self.rank1 / "v2/fedcba9876543210/entries/a.json"
        other.parent.mkdir(parents=True)
        other.write_text('{"tactic": 7}')
        self.assertNotEqual(
            _autotune_store_digest(self.rank0), _autotune_store_digest(self.rank1)
        )

    def test_files_outside_entries_do_not_count(self):
        _write_entry(self.rank0, "a", '{"tactic": 7}')
        _write_entry(self.rank1, "a", '{"tactic": 7}')
        (self.rank0 / ".lock").write_text("")
        (self.rank0 / "v2/0123456789abcdef/manifest.json").write_text("{}")
        self.assertEqual(
            _autotune_store_digest(self.rank0), _autotune_store_digest(self.rank1)
        )


class TestLockAutotuneStore(CustomTestCase):
    def test_a_held_store_is_refused_until_released(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rank"
            first = _lock_autotune_store(root)
            self.assertIsNotNone(first)
            self.addCleanup(first.close)
            self.assertIsNone(_lock_autotune_store(root))
            first.close()
            second = _lock_autotune_store(root)
            self.assertIsNotNone(second)
            second.close()

    def test_an_uncreatable_root_is_refused(self):
        with tempfile.NamedTemporaryFile() as not_a_dir:
            self.assertIsNone(_lock_autotune_store(Path(not_a_dir.name) / "rank"))


class TestAgreeOnAutotuneStore(CustomTestCase):
    """The gate itself, over a real gloo group and real stores."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def _run_gate(self, per_rank) -> list:
        """per_rank: (entries {name: body}, locked) for each rank."""
        world_size = len(per_rank)
        port = find_available_port(23456)
        ctx = multiprocessing.get_context("spawn")
        procs, readers = [], []
        for rank, (entries, locked) in enumerate(per_rank):
            root = self.dir / f"rank{rank}"
            root.mkdir()
            (root / ".lock").write_text("")
            for name, body in entries.items():
                _write_entry(root, name, body)
            reader, writer = ctx.Pipe(duplex=False)
            proc = ctx.Process(
                target=_gate_worker,
                args=(rank, world_size, port, str(root), locked, writer),
            )
            proc.start()
            writer.close()
            procs.append(proc)
            readers.append(reader)
        results = [r.recv() for r in readers]
        for proc in procs:
            proc.join(timeout=120)
        for status, value in results:
            self.assertEqual(status, "ok", msg=value)
        for rank in range(world_size):
            # The lock file outlives a wipe: it is what the process holds.
            self.assertTrue((self.dir / f"rank{rank}" / ".lock").exists())
        return [value for _, value in results]

    def test_matching_stores_are_kept(self):
        entries = {"a": '{"tactic": 7}'}
        self.assertEqual(
            self._run_gate([(entries, True), (entries, True)]),
            [(True, ["a.json"]), (True, ["a.json"])],
        )

    def test_diverged_stores_are_wiped_on_every_rank(self):
        # A rank that kept its store would skip profiles its peer still runs.
        self.assertEqual(
            self._run_gate([({"a": '{"tactic": 7}'}, True), ({}, True)]),
            [(True, []), (True, [])],
        )

    def test_an_unlocked_rank_keeps_every_rank_off_disk(self):
        # The unlocked rank's store may change mid-tuning, so no rank may trust
        # its own, and nothing is wiped on another process's behalf.
        entries = {"a": '{"tactic": 7}'}
        self.assertEqual(
            self._run_gate([(entries, True), (entries, False)]),
            [(False, ["a.json"]), (False, ["a.json"])],
        )


class TestModelPrefillAutotune(CustomTestCase):
    """Model kernel warmup must cover prefill without a speculative dummy batch."""

    def setUp(self):
        self.hook = Mock(return_value=1)
        self.mr = SimpleNamespace(
            model=SimpleNamespace(autotune_prefill_kernels=self.hook),
            is_generation=True,
            is_draft_worker=False,
            dtype=torch.bfloat16,
        )
        self.runner = SimpleNamespace(model_runner=self.mr)
        # No dummy-buffer or attention APIs: this path must not build a
        # TARGET_VERIFY batch or mutate request/KV state.
        for target, kwargs in (
            ("max_prefill_buffer_tokens", {"return_value": 65536}),
            (
                "flashinfer_autotune_context",
                {"side_effect": lambda *a, **k: nullcontext()},
            ),
        ):
            p = patch.object(autotune, target, **kwargs)
            setattr(self, target, p.start())
            self.addCleanup(p.stop)
        p = patch.object(
            autotune.envs.SGLANG_FLASHINFER_AUTOTUNE_EXTEND, "get", return_value=False
        )
        p.start()
        self.addCleanup(p.stop)

    def test_declining_model_never_enters_the_autotune_context(self):
        self.mr.model.wants_prefill_autotune = lambda: False
        autotune.maybe_flashinfer_autotune_extend(self.runner, decode_num_tokens=384)
        self.hook.assert_not_called()
        self.flashinfer_autotune_context.assert_not_called()

    def test_extend_pass_is_opt_in(self):
        # A draft worker keeps its own warmup; a model without the hook opts out.
        for draft, has_hook in ((True, True), (False, False)):
            with self.subTest(draft=draft, has_hook=has_hook):
                self.mr.is_draft_worker = draft
                if not has_hook:
                    del self.mr.model.autotune_prefill_kernels
                autotune.maybe_flashinfer_autotune_extend(
                    self.runner, decode_num_tokens=384
                )
                self.hook.assert_not_called()
                self.flashinfer_autotune_context.assert_not_called()


class TestWarmupAttachesStoreFirst(CustomTestCase):
    """PCIe-IPC tunes in its prepare(): the store must already be attached."""

    def _warmup(self, *, autotune_runs: bool, pcie_ipc: bool, disabled: bool = False):
        calls = []
        fake = SimpleNamespace(
            model_runner=SimpleNamespace(device="cuda", model=SimpleNamespace()),
            _pre_initialize_flashinfer_allreduce_workspace=lambda: None,
            _pre_initialize_fi_a2a_workspace=lambda: None,
            _pre_initialize_pcie_ipc_workspace=lambda: calls.append("pcie_ipc"),
            _autotune_buffers=lambda: (object(), 8),
            _flashinfer_autotune=lambda **_: calls.append("autotune"),
        )
        tp_group = SimpleNamespace(pcie_ipc_comm=object() if pcie_ipc else None)
        exec_ = SimpleNamespace(
            kernel=SimpleNamespace(disable_flashinfer_autotune=disabled)
        )
        with (
            patch.object(
                base_runner,
                "attach_flashinfer_autotune_store",
                side_effect=lambda mr: calls.append("attach"),
            ),
            patch.object(
                base_runner,
                "should_run_flashinfer_autotune",
                return_value=autotune_runs,
            ),
            patch.object(base_runner, "maybe_flashinfer_autotune_extend"),
            patch.object(base_runner, "get_exec", return_value=exec_),
            get_parallel().override(tp_group=tp_group),
        ):
            base_runner.BaseRunner.warmup(fake)
        return calls

    def test_store_attaches_before_pcie_ipc_tunes(self):
        for autotune_runs in (True, False):
            with self.subTest(autotune_runs=autotune_runs):
                calls = self._warmup(autotune_runs=autotune_runs, pcie_ipc=True)
                self.assertEqual(calls[:2], ["attach", "pcie_ipc"])

    def test_nothing_attaches_when_nothing_tunes(self):
        for pcie_ipc, disabled in ((False, False), (True, True)):
            with self.subTest(pcie_ipc=pcie_ipc, disabled=disabled):
                calls = self._warmup(
                    autotune_runs=False, pcie_ipc=pcie_ipc, disabled=disabled
                )
                self.assertNotIn("attach", calls)


@unittest.skipUnless(torch.cuda.is_available(), "FlashInfer requires CUDA")
class TestAutotuneStoreAttach(CustomTestCase):
    """Every tuning pass in a process tunes into the first attached store."""

    def setUp(self):
        from flashinfer.autotuner import AutoTuner

        self.tuner = AutoTuner.get()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.runner = SimpleNamespace(device="cuda", forward_stream=torch.cuda.Stream())
        for p in (
            patch.object(autotune, "_attached_store", None),
            patch.object(
                autotune, "get_flashinfer_autotune_skip_ops", return_value=set()
            ),
            get_parallel().override(tp_group=SimpleNamespace(world_size=1)),
        ):
            p.__enter__()
            self.addCleanup(p.__exit__, None, None, None)

    def test_target_and_draft_share_the_first_store(self):
        # Serving reads only the attached store's winners, so a draft pass that
        # re-attached elsewhere would hide every target winner from serving.
        target, draft = self.dir / "target", self.dir / "draft"
        with patch.object(
            autotune, "flashinfer_autotune_store_root", side_effect=[target, draft]
        ) as store_root:
            for _ in range(2):
                with autotune.flashinfer_autotune_context(
                    self.runner, run_lm_head=False
                ):
                    self.assertEqual(self.tuner._active_managed_store.root, target)
            self.assertEqual(store_root.call_count, 1)
        self.assertEqual(self.tuner._managed_cache.root, target)

    def test_a_store_held_elsewhere_tunes_in_memory(self):
        root = self.dir / "held"
        held = _lock_autotune_store(root)
        self.addCleanup(held.close)
        with patch.object(
            autotune, "flashinfer_autotune_store_root", return_value=root
        ):
            with autotune.flashinfer_autotune_context(self.runner, run_lm_head=False):
                self.assertIsNone(self.tuner._active_managed_store)
        self.assertIsNone(autotune._attached_store.root)


if __name__ == "__main__":
    unittest.main()
