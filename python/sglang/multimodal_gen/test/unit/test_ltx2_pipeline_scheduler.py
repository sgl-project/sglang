from types import SimpleNamespace

from diffusers import FlowMatchEulerDiscreteScheduler as DiffusersFlowMatchScheduler

import sglang.multimodal_gen.runtime.pipelines.ltx_2_pipeline as ltx2_pipeline_module
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_ltx2_flow_match import (
    LTX2FlowMatchScheduler,
)
from sglang.multimodal_gen.runtime.pipelines.ltx_2_pipeline import _BaseLTX2Pipeline


class _PipelineForTest(_BaseLTX2Pipeline):
    def create_pipeline_stages(self, server_args):
        return None


def test_initialize_pipeline_rebuilds_scheduler_as_native_ltx2(monkeypatch):
    monkeypatch.setattr(
        ltx2_pipeline_module,
        "sync_ltx23_runtime_vae_markers",
        lambda *_args, **_kwargs: None,
    )

    original_scheduler = DiffusersFlowMatchScheduler(
        num_train_timesteps=321,
        shift=1.75,
        use_dynamic_shifting=True,
        base_shift=0.65,
        max_shift=1.8,
        time_shift_type="linear",
    )
    pipeline = _PipelineForTest.__new__(_PipelineForTest)
    pipeline.modules = {
        "scheduler": original_scheduler,
        "vae": SimpleNamespace(config=SimpleNamespace()),
    }
    server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(
            vae_config=SimpleNamespace(arch_config=SimpleNamespace())
        )
    )

    pipeline.initialize_pipeline(server_args)

    scheduler = pipeline.modules["scheduler"]
    assert isinstance(scheduler, LTX2FlowMatchScheduler)
    assert not isinstance(scheduler, DiffusersFlowMatchScheduler)
    assert scheduler.config.num_train_timesteps == 321
    assert scheduler.config.shift == 1.75
    assert scheduler.config.use_dynamic_shifting is True
    assert scheduler.config.base_shift == 0.65
    assert scheduler.config.max_shift == 1.8
    assert scheduler.config.time_shift_type == "linear"
