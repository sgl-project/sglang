"""FlashInfer autotune must reach the same tactics on every TP rank.

Without a cross-rank reduction each rank's ``argmin`` follows local timing noise
(measured: 20/20 tuned MoE shapes diverged across 4 ranks on gpt-oss-120b). The
reduction holds only if ranks also enter tuning with the same cache, so these
cover that gate and the digest it decides on.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=52, suite="base-a-test-cpu")

import contextlib
import json
import multiprocessing
import os
import sys
import tempfile
import traceback
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch.distributed as dist

from sglang.srt.model_executor.runner.flashinfer_autotune import (
    _autotune_cache_digest,
    _autotune_tactic_sync_group,
    _drop_diverged_autotune_cache,
)
from sglang.test.test_utils import CustomTestCase, find_available_port

ENV = {"flashinfer_version": "0.6.17", "gpu": "NVIDIA GB300"}


def _gate_worker(rank, world_size, master_port, cache_path, writer):
    """Run the entry gate on one rank; report whether the cache survived."""
    try:
        os.environ.update(
            RANK=str(rank),
            WORLD_SIZE=str(world_size),
            MASTER_ADDR="localhost",
            MASTER_PORT=str(master_port),
        )
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        _drop_diverged_autotune_cache(Path(cache_path), dist.group.WORLD, ENV)
        writer.send(("ok", Path(cache_path).is_file()))
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


class TestAutotuneCacheDigest(CustomTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def _write(self, name: str, configs) -> Path:
        path = self.dir / name
        path.write_text(json.dumps(configs))
        return path

    def _digest(self, path: Path, env=ENV) -> str:
        return _autotune_cache_digest(path, env)

    def test_unusable_caches_read_as_empty(self):
        # Files yielding no loadable entries must digest alike, whichever way
        # they are unusable; a non-dict also has to not raise.
        self.assertEqual(self._digest(self.dir / "absent.json"), "")
        corrupt = self.dir / "corrupt.json"
        corrupt.write_text("{not json")
        self.assertEqual(self._digest(corrupt), "")
        self.assertEqual(self._digest(self._write("null.json", None)), "")
        self.assertEqual(self._digest(self._write("list.json", [])), "")

    def test_metadata_stamp_decides_whether_entries_load(self):
        # Equal tactics, different stamps: one rank loads them, the other
        # ignores the file.
        rank0 = self._write("rank0.json", {"_metadata": {"cublas": "12.8"}, "op": 7})
        rank1 = self._write("rank1.json", {"_metadata": {"cublas": "12.9"}, "op": 7})
        self.assertNotEqual(self._digest(rank0), self._digest(rank1))

    def test_environment_is_part_of_the_load_decision(self):
        # Same file, drifted environment on one rank: that rank loads nothing.
        cache = self._write("rank.json", {"_metadata": {"cublas": "12.8"}, "op": 7})
        self.assertNotEqual(
            self._digest(cache), self._digest(cache, {**ENV, "gpu": "NVIDIA B200"})
        )

    def test_key_order_does_not_matter(self):
        # Pins sort_keys: the same tactics must digest alike in any order.
        rank0 = self._write("rank0.json", {"a": 1, "b": 2})
        rank1 = self._write("rank1.json", {"b": 2, "a": 1})
        self.assertEqual(self._digest(rank0), self._digest(rank1))


class TestDropDivergedAutotuneCache(CustomTestCase):
    """The gate itself, over a real gloo group and real files."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def _run_gate(self, per_rank_configs) -> list:
        world_size = len(per_rank_configs)
        port = find_available_port(23456)
        ctx = multiprocessing.get_context("spawn")
        procs, readers = [], []
        for rank, configs in enumerate(per_rank_configs):
            path = self.dir / f"rank{rank}.json"
            path.write_text(json.dumps(configs))
            reader, writer = ctx.Pipe(duplex=False)
            proc = ctx.Process(
                target=_gate_worker,
                args=(rank, world_size, port, str(path), writer),
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
        return [value for _, value in results]

    def test_matching_caches_are_kept(self):
        entries = {"_metadata": {"cublas": "12.8"}, "op": 7}
        self.assertEqual(self._run_gate([entries, entries]), [True, True])

    def test_diverged_caches_are_dropped_on_every_rank(self):
        # A rank that kept its cache would skip profiles its peer still runs.
        meta = {"_metadata": {"cublas": "12.8"}}
        self.assertEqual(
            self._run_gate([{**meta, "op": 7}, {**meta, "op": 8}]), [False, False]
        )

    def test_caches_diverging_only_in_metadata_are_dropped(self):
        # Same desync, reached through the stamp instead of the tactics.
        self.assertEqual(
            self._run_gate(
                [
                    {"_metadata": {"cublas": "12.8"}, "op": 7},
                    {"_metadata": {"cublas": "12.9"}, "op": 7},
                ]
            ),
            [False, False],
        )


class _FakeAutoTuner:
    instance = None

    def __init__(self):
        self.file_configs = {}

    @classmethod
    def get(cls):
        if cls.instance is None:
            cls.instance = cls()
        return cls.instance

    def load_configs(self, path):
        entries = json.loads(Path(path).read_text())
        entries.pop("_metadata", None)
        self.file_configs.update(entries)

    def save_configs(self, path):
        Path(path).write_text(json.dumps(self.file_configs))


@contextlib.contextmanager
def _fake_autotune(tune_mode=True, cache=None, skip_ops=None):
    # flashinfer's autotune(cache=...) clears the loaded table on entry, saves on exit.
    tuner = _FakeAutoTuner.get()
    if cache is not None:
        tuner.file_configs.clear()
        if Path(cache).is_file():
            tuner.load_configs(cache)
    yield
    if cache is not None and tune_mode:
        tuner.save_configs(cache)


def _fake_flashinfer_autotuner():
    group = [None]
    return SimpleNamespace(
        AutoTuner=_FakeAutoTuner,
        autotune=_fake_autotune,
        _collect_metadata=lambda: {},
        get_autotune_process_group=lambda: group[0],
        set_autotune_process_group=lambda g: group.__setitem__(0, g),
    )


class TestAutotuneCachePhases(CustomTestCase):
    """Loaded target tactics survive draft warmup, unless cache reuse is disabled."""

    def test_target_and_draft_cache_reuse(self):
        from sglang.srt.model_executor.runner import flashinfer_autotune as warmup

        _FakeAutoTuner.instance = None
        tuner = _FakeAutoTuner.get()
        runner = SimpleNamespace(
            device="cpu",
            forward_stream=SimpleNamespace(wait_stream=lambda stream: None),
            tp_group=SimpleNamespace(world_size=1),
        )
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(current_stream=lambda: runner.forward_stream),
            get_device_module=lambda device: SimpleNamespace(
                stream=lambda stream: nullcontext()
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            target, draft = (
                Path(directory) / name for name in ("target.json", "draft.json")
            )
            target.write_text(json.dumps({"target_prefill": ["TestRunner", 7]}))
            draft.write_text(json.dumps({"draft_decode": ["TestRunner", 3]}))
            with (
                patch.dict(
                    sys.modules,
                    {
                        "flashinfer": SimpleNamespace(),
                        "flashinfer.autotuner": _fake_flashinfer_autotuner(),
                    },
                ),
                patch.object(warmup, "torch", fake_torch),
                patch.object(
                    warmup,
                    "flashinfer_autotune_cache_path",
                    side_effect=[target, draft, draft],
                ),
                patch.object(
                    warmup, "get_flashinfer_autotune_skip_ops", return_value=set()
                ),
                warmup.envs.SGLANG_FLASHINFER_AUTOTUNE_CACHE.override(True),
            ):
                with warmup.flashinfer_autotune_context(runner, run_lm_head=False):
                    self.assertEqual(
                        tuner.file_configs["target_prefill"], ["TestRunner", 7]
                    )
                with warmup.flashinfer_autotune_context(runner, run_lm_head=False):
                    self.assertEqual(
                        tuner.file_configs["target_prefill"], ["TestRunner", 7]
                    )
                    self.assertEqual(
                        tuner.file_configs["draft_decode"], ["TestRunner", 3]
                    )
                saved = json.loads(draft.read_text())
                self.assertEqual(saved["target_prefill"], ["TestRunner", 7])
                self.assertEqual(saved["draft_decode"], ["TestRunner", 3])
                with (
                    warmup.envs.SGLANG_FLASHINFER_AUTOTUNE_CACHE.override(False),
                    warmup.flashinfer_autotune_context(runner, run_lm_head=False),
                ):
                    self.assertNotIn("target_prefill", tuner.file_configs)
                    self.assertNotIn("draft_decode", tuner.file_configs)


if __name__ == "__main__":
    unittest.main()
