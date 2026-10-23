"""Shared denoising stages never block the host on the device.

Each test runs one request through a stage with a tiny DiT and the real
flow-match Euler scheduler. Model tests follow the same pattern with their own
stage and model.
"""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, PropertyMock, patch

import torch

from sglang.multimodal_gen.configs.task_type import ModelTaskType
from sglang.multimodal_gen.runtime.distributed.cfg_policy import CFGPolicy
from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import SDPABackend
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import denoising_dmd
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingContext,
    DenoisingStage,
)
from sglang.multimodal_gen.test.unit.host_sync_guard import SyncOnRead, no_cuda_sync


class _PipelineConfig:
    task_type = ModelTaskType.T2I

    def get_classifier_free_guidance_scale(self, batch, guidance_scale):
        return guidance_scale

    def slice_noise_pred(self, noise_pred, latents):
        return noise_pred

    def postprocess_cfg_noise(self, batch, noise_pred, noise_pred_cond):
        return noise_pred


class _TinyDiT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)

    def forward(self, hidden_states, timestep, encoder_hidden_states):
        hidden = self.proj(hidden_states.movedim(1, -1)).movedim(-1, 1)
        return hidden + encoder_hidden_states.mean() + timestep.view(-1, 1, 1, 1) / 1000


def _run_request(device, wrap_device_tensor=lambda t: t, guard=nullcontext()):
    """Denoise one two-branch CFG request through ``_run_denoising_loop``."""
    torch.manual_seed(0)
    transformer = _TinyDiT().to(device)
    scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0)
    scheduler.set_timesteps(4, device=device)
    scheduler.sigmas = wrap_device_tensor(scheduler.sigmas)
    server_args = SimpleNamespace(
        comfyui_mode=False,
        disable_autocast=True,
        enable_cfg_parallel=False,
        enable_breakable_cuda_graph=False,
        enable_layerwise_nvtx_marker=False,
        pipeline_config=_PipelineConfig(),
    )
    stage = DenoisingStage.__new__(DenoisingStage)
    stage.server_args = server_args
    stage.transformer = transformer
    stage.attn_backend = SDPABackend
    stage._extra_func_kwarg_names_cache = {}
    stage._manage_dit_use_site = Mock()
    stage.progress_bar = MagicMock()
    batch = SimpleNamespace(
        raw_latent_shape=(1, 4, 2, 2),
        image_latent=None,
        condition_image=None,
        do_classifier_free_guidance=True,
        guidance_scale=4.0,
        guidance_scale_2=None,
        cfg_normalization=0,
        guidance_rescale=0.0,
        is_cfg_negative=False,
        return_trajectory_latents=False,
        sampling_params=SimpleNamespace(cfg_gate_step=None),
        extra={},
        metrics=None,
        perf_dump_path=None,
        is_warmup=False,
    )
    prompt = torch.randn(1, 3, 4, device=device)
    ctx = DenoisingContext(
        scheduler=scheduler,
        extra_step_kwargs={},
        target_dtype=torch.float32,
        autocast_enabled=False,
        timesteps=wrap_device_tensor(scheduler.timesteps),
        num_inference_steps=4,
        num_warmup_steps=0,
        image_kwargs={},
        pos_cond_kwargs={"encoder_hidden_states": prompt},
        neg_cond_kwargs={"encoder_hidden_states": -prompt},
        latents=wrap_device_tensor(torch.randn(1, 4, 2, 2, device=device)),
        boundary_timestep=None,
        z=None,
        reserved_frames_mask=None,
        seq_len=None,
        guidance=None,
        is_warmup=False,
        cfg_policy=CFGPolicy().build(
            batch,
            {},
            {"encoder_hidden_states": prompt},
            {"encoder_hidden_states": -prompt},
        ),
    )
    with torch.no_grad(), guard:
        stage._run_denoising_loop(ctx, batch, server_args)
    return ctx.latents.as_subclass(torch.Tensor)


class _TinyVideoDiT(_TinyDiT):
    def forward(self, hidden_states, timestep, guidance=None):
        hidden = self.proj(hidden_states.movedim(1, -1)).movedim(-1, 1)
        return hidden + timestep.view(-1, 1, 1, 1, 1) / 1000


def _run_dmd_request(device_tensor=lambda t: t, nan_latents=False):
    """One DMD request; device timesteps go through ``device_tensor``."""
    torch.manual_seed(0)
    stage = denoising_dmd.DmdDenoisingStage.__new__(denoising_dmd.DmdDenoisingStage)
    stage.server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(should_use_guidance=False)
    )
    stage.scheduler = FlowMatchEulerDiscreteScheduler(shift=8.0)
    stage.transformer = _TinyVideoDiT()
    stage.transformer_2 = None
    stage.attn_backend = SDPABackend
    stage._extra_func_kwarg_names_cache = {}
    stage._manage_dit_use_site = Mock()
    stage.progress_bar = MagicMock()
    stage.log_info = Mock()
    final = {}
    stage._post_denoising_loop = lambda latents, **kwargs: final.update(latents=latents)
    latents = torch.randn(1, 2, 4, 2, 2)
    if nan_latents:
        latents[0, 1, 0, 0, 0] = float("nan")
    stage._prepare_denoising_loop = lambda batch, args: SimpleNamespace(
        target_dtype=torch.float32,
        autocast_enabled=False,
        num_warmup_steps=0,
        latents=latents,
        pos_cond_kwargs={},
    )
    batch = SimpleNamespace(
        image_embeds=[],
        image_latent=None,
        generator=[torch.Generator().manual_seed(1)],
        metrics=None,
        perf_dump_path=None,
        is_warmup=False,
        is_cfg_negative=False,
    )
    server_args = SimpleNamespace(
        disable_autocast=True,
        pipeline_config=SimpleNamespace(dmd_denoising_steps=[1000, 757, 522]),
    )
    tensor = torch.tensor

    def tensor_on_device(data, **kwargs):
        made = tensor(data, **kwargs)
        return device_tensor(made) if "device" in kwargs else made

    cpu = torch.device("cpu")
    with (
        torch.no_grad(),
        patch.object(denoising_dmd, "get_local_torch_device", return_value=cpu),
        patch.object(
            denoising_dmd.DmdDenoisingStage,
            "device",
            new_callable=PropertyMock,
            return_value=cpu,
        ),
        patch.object(torch, "tensor", tensor_on_device),
    ):
        stage.forward(batch, server_args)
    return final["latents"].as_subclass(torch.Tensor)


class TestDenoisingLoopHostSync(unittest.TestCase):
    def test_steps_never_read_device_schedule_on_host(self):
        expected = _run_request("cpu")
        actual = _run_request(
            "cpu", wrap_device_tensor=lambda t: t.as_subclass(SyncOnRead)
        )
        self.assertTrue(torch.equal(actual, expected))

    def test_dmd_steps_never_read_device_timesteps(self):
        expected = _run_dmd_request()
        actual = _run_dmd_request(device_tensor=lambda t: t.as_subclass(SyncOnRead))
        self.assertTrue(torch.equal(actual, expected))
        # the NaN guard still fires, read once after the loop
        with self.assertRaisesRegex(AssertionError, "contains nan"):
            _run_dmd_request(nan_latents=True)

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_cuda_loop_syncs_only_for_the_host_schedule(self):
        expected = _run_request("cuda")
        # the loop copies the schedule to the host once; steps must not sync
        guard = no_cuda_sync(allow=("stages/denoising.py:_run_denoising_loop",))
        actual = _run_request("cuda", guard=guard)
        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
