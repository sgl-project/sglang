import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.multimodal_gen.configs.pipeline_configs.flux import (
    Flux2KleinBasePipelineConfig,
)
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    base,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    timestep_preparation as module,
)


@pytest.mark.parametrize("warmup", [False, True])
@pytest.mark.parametrize("debug", [False, True])
def test_timestep_logging_preserves_scheduler_and_skips_unused_copy(warmup, debug):
    args = SimpleNamespace(pipeline_config=Flux2KleinBasePipelineConfig())
    scheduler = FlowMatchEulerDiscreteScheduler()
    with patch.object(base, "get_global_server_args", return_value=args):
        stage = module.TimestepPreparationStage(scheduler)
    batch = Req(sampling_params=SamplingParams(num_inference_steps=4))
    batch.is_warmup = warmup
    records = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record)
            self.format(record)

    test_logger = logging.Logger(
        "timestep-test", logging.DEBUG if debug else logging.INFO
    )
    test_logger.addHandler(Capture())
    with (
        patch.object(module, "logger", test_logger),
        patch.object(
            module, "get_local_torch_device", return_value=torch.device("cpu")
        ),
    ):
        if debug and not warmup:
            result = stage.forward(batch, args)
        else:
            with patch.object(
                torch.Tensor, "detach", side_effect=AssertionError("unused log copy")
            ):
                result = stage.forward(batch, args)
    assert result is batch
    assert batch.scheduler is scheduler
    assert batch.timesteps is scheduler.timesteps
    assert len(records) == int(debug and not warmup)
    if records:
        value = records[0].args[-1]
        assert value.device.type == "cpu"
        assert torch.equal(value, batch.timesteps)
        assert "TimestepPreparationStage" in records[0].getMessage()
