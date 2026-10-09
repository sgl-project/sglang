"""Regression test for GH-43189.

When the SGLD worker dies mid-sampler-run and `ensure_executor` rebuilds it,
the executor must not go on assuming the new worker's session cache holds the
conditioning the old worker had. This exercises the real executor bookkeeping
(`SGLDiffusionExecutor._mark_and_maybe_drop`, `SGLDiffusionGenerator.ensure_executor`),
the real `FluxAdapter`, the real session cache (`bind_comfyui_session`), and
the real `ComfyUILatentPreparationStage.verify_input`. Only worker liveness
and `load_model` are stubbed, since those require a live GPU process.
"""

from unittest import mock

import pytest
import torch

from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.core.generator import (
    SGLDiffusionGenerator,
)
from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.flux import (
    FluxExecutor,
)
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime import server_args as server_args_module
from sglang.multimodal_gen.runtime.pipelines_core import comfyui_mode
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.comfyui_latent_preparation import (
    ComfyUILatentPreparationStage,
)
from sglang.multimodal_gen.runtime.server_args import (
    ServerArgs,
    set_global_server_args,
)


@pytest.fixture(autouse=True)
def _global_server_args():
    previous = server_args_module._global_server_args
    set_global_server_args(ServerArgs(model_path="flux-test", comfyui_mode=True))
    try:
        yield
    finally:
        set_global_server_args(previous)


def _make_executor(generator) -> FluxExecutor:
    config = mock.Mock()
    config.unet_config = {"dtype": torch.float32}
    executor = FluxExecutor(
        generator=generator, model_path="flux-test", model=mock.Mock(), config=config
    )
    executor._sgld_reload = {"model_path": "flux-test"}
    return executor


def _pack_and_fill(executor: FluxExecutor, context: torch.Tensor, y: torch.Tensor) -> Req:
    # Same latents/conditioning ComfyUI would pass on consecutive denoise
    # steps of one sampler run: the cond key must match step to step so
    # _mark_and_maybe_drop can recognize "already sent".
    x = torch.randn(1, 16, 8, 8)
    packed = executor.adapter.pack(x, torch.tensor(500.0), context, y=y)
    executor._mark_and_maybe_drop(packed)

    req = Req(sampling_params=SamplingParams(prompt="a cat"))
    executor.adapter.fill_req(req, packed)
    req.height = packed.height
    req.width = packed.width
    req.generator = torch.Generator()
    extra = dict(req.extra or {})
    extra["comfyui_session_id"] = executor.comfyui_session_id()
    for key in ("comfyui_cond_key", "comfyui_cache_fp"):
        value = packed.extra_req.get(key)
        if value is not None:
            extra[key] = value
    req.extra = extra
    return req


def test_rebuild_resends_conditioning_the_new_worker_lacks():
    comfyui_mode._SESSIONS.clear()
    comfyui_mode._RUNS.clear()

    owner = SGLDiffusionGenerator()
    old_generator = mock.Mock()
    executor = _make_executor(old_generator)
    executor.begin_sampler_run()
    context = torch.randn(1, 5, 768)
    y = torch.randn(1, 768)

    # Step 1: old worker is alive, nothing to rebuild.
    with mock.patch.object(owner, "_owns_live", return_value=True):
        owner.ensure_executor(executor)

    req1 = _pack_and_fill(executor, context, y)
    stage = ComfyUILatentPreparationStage(scheduler=mock.Mock(), transformer=mock.Mock())
    result1 = stage.verify_input(req1, server_args=mock.Mock())
    assert result1.is_valid(), result1.get_failure_summary()
    assert len(req1.prompt_embeds) == 2

    # Step 2: the worker died and gets rebuilt. The new worker process has an
    # empty session cache -- simulated here since _SESSIONS/_RUNS are that
    # cache and really do start empty in a freshly launched worker process.
    new_generator = mock.Mock()
    comfyui_mode._SESSIONS.clear()
    comfyui_mode._RUNS.clear()
    with mock.patch.object(owner, "_owns_live", return_value=False), mock.patch.object(
        owner, "load_model", side_effect=lambda **_: setattr(owner, "generator", new_generator)
    ):
        owner.ensure_executor(executor)
    assert executor.generator is new_generator

    req2 = _pack_and_fill(executor, context, y)
    result2 = stage.verify_input(req2, server_args=mock.Mock())

    assert len(req2.prompt_embeds) == 2, (
        "conditioning not resent to the new worker after a mid-run rebuild; "
        f"got {len(req2.prompt_embeds)} embeds"
    )
    assert result2.is_valid(), result2.get_failure_summary()
