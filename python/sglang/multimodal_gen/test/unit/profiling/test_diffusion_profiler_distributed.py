# SPDX-License-Identifier: Apache-2.0
"""Real two-rank Gloo stop/export coordination; NPU capture and parsing are mocked."""

import json
import multiprocessing
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist

from sglang.multimodal_gen.runtime.utils import profiler as profiler_module
from sglang.srt.observability import profile_analysis_npu


def stop_worker(rank, root, failure):
    dist.init_process_group(
        "gloo",
        init_method=(Path(root) / "rendezvous").as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=15),
    )
    try:
        profiler = profiler_module.SGLDiffusionProfiler.__new__(
            profiler_module.SGLDiffusionProfiler
        )
        profiler.rank = rank
        profiler.has_stopped = False
        profiler.profiler = SimpleNamespace(
            stop=Mock(), prof_if=SimpleNamespace(prof_path=f"rank{rank}_ascend_pt")
        )
        handler = profile_analysis_npu.NPUTraceHandler.__new__(
            profile_analysis_npu.NPUTraceHandler
        )
        handler.npu_auto_analysis_in_daemon = True
        handler.export = Mock(wraps=handler.export)
        profiler._npu_trace_handler = handler
        if rank == 1 and failure == "stop":
            profiler.profiler.stop.side_effect = RuntimeError("stop failed")
        profiler_module.SGLDiffusionProfiler._instance = profiler
        group = SimpleNamespace(world_size=2, cpu_group=dist.group.WORLD)
        result = {"error": None, "continued": False}
        with (
            patch.object(profiler_module.current_platform, "is_npu", return_value=True),
            patch.object(torch.cuda, "is_available", return_value=False),
            patch.object(
                torch, "npu", SimpleNamespace(synchronize=Mock()), create=True
            ),
            patch.object(
                profiler_module, "model_parallel_is_initialized", return_value=True
            ),
            patch.object(profiler_module, "get_replica_group", return_value=group),
            patch.object(
                profile_analysis_npu,
                "start_npu_profile_analysis",
                side_effect=OSError("parser launch failed")
                if rank == 1 and failure == "parser"
                else None,
            ),
        ):
            try:
                profiler.stop(dump_rank=0)
                # Represent the next collective in the remaining generation.
                dist.barrier()
                result["continued"] = True
            except RuntimeError as error:
                result["error"] = str(error)
            # Request cleanup must neither parse nor enter the collective again.
            profiler.stop(dump_rank=0)
        result["stop_calls"] = profiler.profiler.stop.call_count
        result["export_calls"] = profiler._npu_trace_handler.export.call_count
        result["instance_cleared"] = (
            profiler_module.SGLDiffusionProfiler.get_instance() is None
        )
        (Path(root) / f"rank{rank}.json").write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(dist.is_gloo_available(), "requires Gloo")
class TestDiffusionProfilerDistributed(unittest.TestCase):
    def run_workers(self, failure):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            workers = [
                multiprocessing.get_context("spawn").Process(
                    target=stop_worker, args=(rank, directory, failure)
                )
                for rank in range(2)
            ]
            try:
                for worker in workers:
                    worker.start()
                for worker in workers:
                    worker.join(40)
                    self.assertFalse(worker.is_alive(), "profiler collective hung")
                    self.assertEqual(worker.exitcode, 0)
                results = [
                    json.loads((Path(directory) / f"rank{rank}.json").read_text())
                    for rank in range(2)
                ]
            finally:
                for worker in workers:
                    if worker.is_alive():
                        worker.kill()
                    if worker.pid is not None:
                        worker.join(5)
        for result in results:
            self.assertEqual(result["stop_calls"], 1)
            self.assertTrue(result["instance_cleared"])
        return results

    def test_all_ranks_export_then_continue_generation(self):
        for result in self.run_workers(failure=None):
            self.assertIsNone(result["error"])
            self.assertTrue(result["continued"])
            self.assertEqual(result["export_calls"], 1)

    def test_non_driver_stop_failure_reaches_every_rank(self):
        results = self.run_workers("stop")
        self.assertEqual(results[0]["error"], results[1]["error"])
        for result in results:
            self.assertIn("rank 1: stop failed", result["error"])
            self.assertFalse(result["continued"])
            self.assertEqual(result["export_calls"], 0)

    def test_non_driver_parser_launch_failure_does_not_fail_generation(self):
        for result in self.run_workers("parser"):
            self.assertIsNone(result["error"])
            self.assertTrue(result["continued"])
            self.assertEqual(result["export_calls"], 1)


if __name__ == "__main__":
    unittest.main()
