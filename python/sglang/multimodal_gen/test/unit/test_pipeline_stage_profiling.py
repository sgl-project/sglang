import gzip
import json
import unittest
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

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


@pytest.mark.parametrize("rollback", [False, True], ids=["success", "rollback"])
def test_disk_weight_update_trace_identifies_worker_phases(tmp_path, rollback):
    """Distinguish per-module application from validation, cleanup and rollback."""
    from sglang.multimodal_gen.runtime.post_training.weights_updater import (
        WeightsUpdater,
    )
    from sglang.multimodal_gen.runtime.utils.profiler import SGLDiffusionProfiler

    modules = {
        name: torch.nn.Linear(2, 2, bias=False)
        for name in ("transformer", "text_encoder")
    }
    previous = tmp_path / "previous"
    checkpoint = tmp_path / "new"
    for name, module in modules.items():
        for root, weight in (
            (previous, module.weight.detach().clone()),
            (
                checkpoint,
                torch.ones(3, 3)
                if rollback and name == "text_encoder"
                else torch.ones(2, 2),
            ),
        ):
            folder = root / name
            folder.mkdir(parents=True)
            save_file({"weight": weight}, folder / "model.safetensors")
    original = modules["transformer"].weight.detach().clone()
    pipeline = SimpleNamespace(
        modules=modules, model_path=str(previous), get_module=modules.get
    )
    profiler = SGLDiffusionProfiler(
        request_id="disk-update",
        full_profile=True,
        activities=["CPU"],
        log_dir=str(tmp_path),
    )
    try:
        ok, message = WeightsUpdater(pipeline).update_weights_from_disk(str(checkpoint))
    finally:
        profiler.stop()
    assert ok == (not rollback), message
    torch.testing.assert_close(
        modules["transformer"].weight, original if rollback else torch.ones(2, 2)
    )

    with gzip.open(tmp_path / "disk-update-rank-0.trace.json.gz", "rt") as trace:
        events = json.load(trace)["traceEvents"]
    spans = [
        event
        for event in events
        if event.get("ph") == "X" and event.get("name", "").startswith("WEIGHT UPDATE ")
    ]
    cleanup_phases = {
        "WEIGHT UPDATE collect Python garbage",
        "WEIGHT UPDATE empty CUDA cache",
    }
    expected = {
        "WEIGHT UPDATE invalidate conditioning caches",
        "WEIGHT UPDATE resolve checkpoint",
        "WEIGHT UPDATE validate checkpoint",
        "WEIGHT UPDATE load transformer",
        "WEIGHT UPDATE load text_encoder",
        "WEIGHT UPDATE cleanup",
    }
    if rollback:
        expected.add("WEIGHT UPDATE rollback")
    else:
        cleanup_phases.update(
            {
                "WEIGHT UPDATE refresh weight-derived caches",
                "WEIGHT UPDATE reset TeaCache",
            }
        )

    totals = [span for span in spans if span["name"] == "WEIGHT UPDATE disk"]
    assert len(totals) == 1
    total = totals[0]
    child_spans = [span for span in spans if span is not total]
    assert sorted(span["name"] for span in child_spans) == sorted(
        expected | cleanup_phases
    )
    cleanup = next(
        span for span in child_spans if span["name"] == "WEIGHT UPDATE cleanup"
    )
    for span in child_spans:
        parent = cleanup if span["name"] in cleanup_phases else total
        assert parent["ts"] <= span["ts"]
        assert span["ts"] + span["dur"] <= parent["ts"] + parent["dur"]


if __name__ == "__main__":
    unittest.main()
