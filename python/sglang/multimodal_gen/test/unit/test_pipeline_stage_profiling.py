import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.utils.perf_logger import PerformanceLogger


class NamedNoOpStage(PipelineStage):
    def __init__(self):
        self.server_args = SimpleNamespace(
            comfyui_mode=True, enable_layerwise_nvtx_marker=False
        )

    def forward(self, batch: Req, server_args) -> Req:
        return batch


class TestPipelineStageProfiling(unittest.TestCase):
    def test_profiler_uses_profile_stage_name(self):
        stage = NamedNoOpStage()
        stage.set_profile_stage_name("profile_stage")
        batch = Req(perf_dump_path="/tmp/unused_perf.json")

        stage(batch, SimpleNamespace())

        self.assertIn("profile_stage", batch.metrics.stages)
        self.assertNotIn("NamedNoOpStage", batch.metrics.stages)

    def test_registered_stage_name_does_not_change_profile_name(self):
        stage = NamedNoOpStage()
        stage.set_registered_stage_name("prompt_encoding_stage_primary")
        batch = Req(perf_dump_path="/tmp/unused_perf.json")

        stage(batch, SimpleNamespace())

        self.assertIn("NamedNoOpStage", batch.metrics.stages)
        self.assertNotIn("prompt_encoding_stage_primary", batch.metrics.stages)


class TestGPUWorkerPerformanceReporting(unittest.TestCase):
    @staticmethod
    def _worker(rank: int) -> GPUWorker:
        worker = object.__new__(GPUWorker)
        worker.rank = rank
        worker.server_args = SimpleNamespace(model_path="test/model")
        return worker

    @patch.object(PerformanceLogger, "dump_benchmark_report")
    @patch.object(PerformanceLogger, "log_request_summary")
    def test_only_rank_zero_writes_tensor_parallel_report(
        self, log_request_summary: Mock, dump_benchmark_report: Mock
    ) -> None:
        metrics = object()
        request = SimpleNamespace(
            is_warmup=False,
            perf_dump_path="/tmp/shared-tp-report.json",
        )
        output_batch = SimpleNamespace(metrics=metrics)

        self._worker(rank=1)._report_request_performance(request, output_batch)

        log_request_summary.assert_not_called()
        dump_benchmark_report.assert_not_called()

        self._worker(rank=0)._report_request_performance(request, output_batch)

        log_request_summary.assert_called_once_with(metrics=metrics)
        dump_benchmark_report.assert_called_once_with(
            file_path=request.perf_dump_path,
            metrics=metrics,
            meta={"model": "test/model"},
            tag="server_perf_dump",
        )

    @patch.object(PerformanceLogger, "dump_benchmark_report")
    @patch.object(PerformanceLogger, "log_request_summary")
    def test_rank_zero_skips_warmup_report(
        self, log_request_summary: Mock, dump_benchmark_report: Mock
    ) -> None:
        request = SimpleNamespace(
            is_warmup=True,
            perf_dump_path="/tmp/warmup-report.json",
        )
        output_batch = SimpleNamespace(metrics=object())

        self._worker(rank=0)._report_request_performance(request, output_batch)

        log_request_summary.assert_not_called()
        dump_benchmark_report.assert_not_called()


if __name__ == "__main__":
    unittest.main()
