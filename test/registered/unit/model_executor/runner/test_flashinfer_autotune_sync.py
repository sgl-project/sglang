"""FlashInfer autotune must reach the same tactics on every TP rank.

Without a cross-rank reduction each rank's ``argmin`` follows local timing noise
(measured: 20/20 tuned MoE shapes diverged across 4 ranks on gpt-oss-120b). The
reduction holds only if ranks also enter tuning with the same cache, so these
cover that gate and the digest it decides on.
"""

from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=50, suite="base-a-test-cpu")
register_cuda_ci(est_time=25, stage="base-b-kernel-unit", runner_config="1-gpu-large")

import json
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

from sglang.srt.model_executor.runner import flashinfer_autotune as autotune
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


@unittest.skipUnless(torch.cuda.is_available(), "FlashInfer requires CUDA")
class TestAutotuneCachePhases(CustomTestCase):
    """Loaded target tactics survive draft warmup, unless cache reuse is off."""

    def test_target_and_draft_cache_reuse(self):
        from flashinfer.autotuner import AutoTuner, _collect_metadata

        tuner = AutoTuner.get()
        tuner.clear_cache()
        self.addCleanup(tuner.clear_cache)
        runner = SimpleNamespace(
            device="cuda",
            forward_stream=torch.cuda.Stream(),
            tp_group=SimpleNamespace(world_size=1),
        )
        with tempfile.TemporaryDirectory() as directory:
            target, draft = (
                Path(directory) / name for name in ("target.json", "draft.json")
            )
            for path, key, tactic in (
                (target, "target_prefill", 7),
                (draft, "draft_decode", 3),
            ):
                path.write_text(
                    json.dumps(
                        {"_metadata": _collect_metadata(), key: ["TestRunner", tactic]}
                    )
                )
            with (
                patch.object(
                    autotune,
                    "flashinfer_autotune_cache_path",
                    side_effect=[target, draft, draft],
                ),
                patch.object(
                    autotune, "get_flashinfer_autotune_skip_ops", return_value=set()
                ),
                autotune.envs.SGLANG_FLASHINFER_AUTOTUNE_CACHE.override(True),
            ):
                with autotune.flashinfer_autotune_context(runner, run_lm_head=False):
                    self.assertEqual(
                        tuner._file_configs["target_prefill"], ("TestRunner", 7)
                    )
                # No profiling: this models a restart that loads tactics from disk.
                self.assertFalse(tuner.profiling_cache)
                with autotune.flashinfer_autotune_context(runner, run_lm_head=False):
                    pass
                saved = json.loads(draft.read_text())
                self.assertEqual(saved["target_prefill"], ["TestRunner", 7])
                self.assertEqual(saved["draft_decode"], ["TestRunner", 3])
                with (
                    autotune.envs.SGLANG_FLASHINFER_AUTOTUNE_CACHE.override(False),
                    autotune.flashinfer_autotune_context(runner, run_lm_head=False),
                ):
                    self.assertNotIn("target_prefill", tuner._file_configs)
                    self.assertNotIn("draft_decode", tuner._file_configs)


if __name__ == "__main__":
    unittest.main()
