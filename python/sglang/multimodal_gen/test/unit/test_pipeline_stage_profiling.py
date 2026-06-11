import unittest
from types import SimpleNamespace

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage


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


if __name__ == "__main__":
    unittest.main()
