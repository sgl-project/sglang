"""Exercise both profiler stop paths without a GPU or distributed group."""

import gzip
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_components.profiler_manager import (
    SchedulerProfilerManager,
)
from sglang.srt.observability import profile_utils
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class CapturedProfiler:
    def __init__(self, truncate=False):
        self.truncate = truncate

    def stop(self):
        pass

    def export_chrome_trace(self, path):
        with gzip.open(path, "wt") as stream:
            json.dump(
                {
                    "traceEvents": [
                        {
                            "cat": "kernel",
                            "ph": "X",
                            "name": "gpu_kernel",
                            "ts": 2,
                            "dur": 3,
                            "args": {"device": 0, "stream": 1},
                        }
                    ]
                },
                stream,
            )
        if self.truncate:
            path = Path(path)
            path.write_bytes(path.read_bytes()[:-3])


class TestExportHooks(unittest.TestCase):
    def test_legacy_and_stage_profiler_export_summary(self):
        ranks = SimpleNamespace(tp_rank=0, num_dp_ranks=1, pp_size=1, moe_ep_size=1)
        for legacy in (True, False):
            with self.subTest(legacy=legacy), TemporaryDirectory() as tmp:
                if legacy:
                    manager = SchedulerProfilerManager.__new__(SchedulerProfilerManager)
                    manager.profile_in_progress = True
                    manager.torch_profiler_output_dir = Path(tmp)
                    manager.profile_prefix = ""
                    manager.profile_id = "test"
                    manager.torch_profiler = CapturedProfiler()
                    manager.rpd_profiler = None
                    manager.profiler_activities = ["GPU"]
                    manager.dp_tp_cpu_group = None
                    manager.merge_profiles = False
                    module = "sglang.srt.managers.scheduler_components.profiler_manager"
                    stop = manager._stop_profile
                else:
                    manager = profile_utils._ProfilerTorch(
                        with_stack=False,
                        record_shapes=False,
                        activities=["GPU"],
                        output_dir=tmp,
                        output_prefix="",
                        output_suffix="",
                        profile_id="test",
                        cpu_group=None,
                        first_rank_in_node=True,
                    )
                    manager.torch_profiler = CapturedProfiler()
                    module = "sglang.srt.observability.profile_utils"
                    stop = manager.stop
                with (
                    envs.SGLANG_PROFILE_KERNEL_SUMMARY.override(True),
                    envs.SGLANG_PROFILE_STEP_KERNELS.override(True),
                    envs.SGLANG_PROFILE_V2.override(False),
                    patch(module + ".get_parallel", return_value=ranks),
                    patch(module + ".torch.distributed.barrier"),
                ):
                    stop()
                paths = list(Path(tmp).glob("*.kernel-summary.json"))
                self.assertEqual(len(paths), 1)
                result = json.loads(paths[0].read_text())
                self.assertEqual(result["devices"][0]["kernel_busy_union_ns"], 3000)
                step_files = list(Path(tmp).glob("*.steps.json"))
                self.assertEqual(len(step_files), 1)
                self.assertEqual(
                    json.loads(step_files[0].read_bytes())["metadata"][
                        "unmatched_kernels"
                    ],
                    1,
                )

    def test_stage_stop_reaches_barrier_after_bad_gzip(self):
        ranks = SimpleNamespace(tp_rank=0, num_dp_ranks=1, pp_size=1, moe_ep_size=1)
        with TemporaryDirectory() as tmp:
            manager = profile_utils._ProfilerTorch(
                with_stack=False,
                record_shapes=False,
                activities=["GPU"],
                output_dir=tmp,
                output_prefix="",
                output_suffix="",
                profile_id="bad",
                cpu_group=None,
                first_rank_in_node=True,
            )
            manager.torch_profiler = CapturedProfiler(truncate=True)
            with (
                envs.SGLANG_PROFILE_KERNEL_SUMMARY.override(True),
                envs.SGLANG_PROFILE_STEP_KERNELS.override(True),
                patch(
                    "sglang.srt.observability.profile_utils.get_parallel",
                    return_value=ranks,
                ),
                patch(
                    "sglang.srt.observability.profile_utils.torch.distributed.barrier"
                ) as barrier,
            ):
                manager.stop()
            barrier.assert_called_once_with(None)


if __name__ == "__main__":
    unittest.main()
