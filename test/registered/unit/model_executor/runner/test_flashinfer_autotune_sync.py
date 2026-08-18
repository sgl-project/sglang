"""FlashInfer autotune must reach the same tactics on every TP rank.

Without a cross-rank reduction each rank's ``argmin`` follows local timing noise
(observed: 20/20 tuned MoE shapes diverged across 4 ranks on gpt-oss-120b). The
reduction only holds if the ranks also enter tuning with the same cache, since a
cache hit skips a profile and desyncs it -- so the gate that enforces that, and
the digest it decides on, are what these tests cover.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=50, suite="base-a-test-cpu")

import json
import multiprocessing
import os
import tempfile
import traceback
import unittest
from pathlib import Path
from types import SimpleNamespace

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
        # The only case that tunes without a group; a 1-rank group would add a
        # collective per tactic for no agreement.
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
        # Every shape a cache file can take that yields no loadable entries has
        # to digest alike, or ranks holding different flavours of "nothing"
        # would drop a cache they both agree on.
        self.assertEqual(self._digest(self.dir / "absent.json"), "")
        corrupt = self.dir / "corrupt.json"
        corrupt.write_text("{not json")
        self.assertEqual(self._digest(corrupt), "")
        # Valid JSON, wrong type: FlashInfer's own reader guards this, and
        # `configs.pop` would raise on it.
        self.assertEqual(self._digest(self._write("null.json", None)), "")
        self.assertEqual(self._digest(self._write("list.json", [])), "")

    def test_metadata_stamp_decides_whether_entries_load(self):
        # load_configs ignores every entry when the stamp mismatches the
        # environment, so ranks with identical tactics but different stamps
        # would enter tuning with different caches.
        rank0 = self._write("rank0.json", {"_metadata": {"cublas": "12.8"}, "op": 7})
        rank1 = self._write("rank1.json", {"_metadata": {"cublas": "12.9"}, "op": 7})
        self.assertNotEqual(self._digest(rank0), self._digest(rank1))

    def test_environment_is_part_of_the_load_decision(self):
        # Same file, drifted environment on one rank (e.g. a driver upgrade
        # applied to one node): that rank loads nothing while its peer loads
        # everything.
        cache = self._write("rank.json", {"_metadata": {"cublas": "12.8"}, "op": 7})
        self.assertNotEqual(
            self._digest(cache), self._digest(cache, {**ENV, "gpu": "NVIDIA B200"})
        )

    def test_key_order_does_not_matter(self):
        # Pins sort_keys: two ranks that tuned the same tactics must agree
        # regardless of the order json.dump happened to write them in.
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
        # A rank that kept its cache would skip profiles its peer still runs,
        # and the per-tactic all-reduce would deadlock.
        meta = {"_metadata": {"cublas": "12.8"}}
        self.assertEqual(
            self._run_gate([{**meta, "op": 7}, {**meta, "op": 8}]), [False, False]
        )

    def test_caches_diverging_only_in_metadata_are_dropped(self):
        # Identical tactics, different stamps: one rank would load them and the
        # other would ignore the file, which is the same desync.
        self.assertEqual(
            self._run_gate(
                [
                    {"_metadata": {"cublas": "12.8"}, "op": 7},
                    {"_metadata": {"cublas": "12.9"}, "op": 7},
                ]
            ),
            [False, False],
        )


if __name__ == "__main__":
    unittest.main()
