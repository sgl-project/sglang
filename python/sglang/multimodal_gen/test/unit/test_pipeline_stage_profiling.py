import unittest
from types import SimpleNamespace

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage


class NamedNoOpStage(PipelineStage):
    def __init__(self):
        super().__init__()
        self.server_args = SimpleNamespace(comfyui_mode=True)

    def forward(self, batch: Req, server_args) -> Req:
        return batch


class TestPipelineStageProfiling(unittest.TestCase):
    def test_profiler_uses_registered_stage_name(self):
        stage = NamedNoOpStage()
        stage.set_component_residency_manager(
            SimpleNamespace(state=SimpleNamespace(stage_name="registered_stage"))
        )
        batch = Req(perf_dump_path="/tmp/unused_perf.json")

        stage(batch, SimpleNamespace())

        self.assertIn("registered_stage", batch.metrics.stages)
        self.assertNotIn("NamedNoOpStage", batch.metrics.stages)


if __name__ == "__main__":
    unittest.main()
