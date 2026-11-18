import gzip
import json
import unittest
from types import SimpleNamespace

import pytest
import torch

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


@pytest.mark.parametrize("requested", [1, 3, -1])
def test_profiler_window_contains_requested_steps(tmp_path, requested):
    """The trace contains N post-warmup steps, or every step for -1."""
    from sglang.multimodal_gen.runtime.utils.profiler import SGLDiffusionProfiler

    profiler = SGLDiffusionProfiler(
        request_id="window",
        num_steps=requested,
        num_inference_steps=6,
        log_dir=str(tmp_path),
    )
    try:
        for step in range(6):
            with torch.profiler.record_function(f"denoise_step_{step}"):
                torch.ones(8).add_(1)
            profiler.step_denoising_step()
    finally:
        profiler.stop()
    path = next(tmp_path.glob("*.trace.json.gz"))
    with gzip.open(path, "rt") as file:
        events = json.load(file)["traceEvents"]
    actual = [
        e["name"]
        for e in events
        if e.get("ph") == "X" and e.get("name", "").startswith("denoise_step_")
    ]
    indices = range(6) if requested == -1 else range(1, requested + 1)
    assert actual == [f"denoise_step_{step}" for step in indices]


if __name__ == "__main__":
    unittest.main()
