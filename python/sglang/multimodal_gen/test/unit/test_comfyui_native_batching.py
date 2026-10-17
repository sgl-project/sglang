# SPDX-License-Identifier: Apache-2.0
"""Opt-in native batching: fewer worker requests for a ComfyUI batch."""

import logging
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.core.generator import (
    _check_native_batching,
)
from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.base import (
    NATIVE_BATCHING_MODES,
    SGLDiffusionExecutor,
    WorkerOutOfMemoryError,
)
from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.flux import FluxAdapter
from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.qwen_image21 import (
    QwenImage21Adapter,
)
from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.zimage import (
    ZImageAdapter,
)
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingStage,
)


class _Executor(SGLDiffusionExecutor):
    """Real forward/pack/unpack; the worker round trip records the request."""

    def __init__(self, adapter, native_batching, max_rows=None):
        torch.nn.Module.__init__(self)
        self.adapter = adapter
        self.native_batching = native_batching
        # Requests with more rows than this "run out of GPU memory".
        self.max_rows = max_rows
        self.sent = []
        self._run_id = 0

    def _execute_packed(self, packed, x, timestep):
        rows = packed.latents.shape[0]
        if self.max_rows is not None and rows > self.max_rows:
            raise WorkerOutOfMemoryError(
                "SGLang Diffusion worker failed: CUDA out of memory"
            )
        self.sent.append(packed)
        # Fake noise_pred: each row is filled with its own context mean.
        rows = packed.prompt_embeds[0].reshape(packed.latents.shape[0], -1).mean(1)
        noise = rows.view(-1, 1, 1, 1, 1).expand_as(packed.latents).clone()
        return self.adapter.unpack(noise, packed, x)


def _zimage_batch(batch=2, seq=5):
    x = torch.zeros(batch, 16, 4, 6)
    timestep = torch.linspace(0.9, 0.3, batch)
    context = torch.stack([torch.full((seq, 8), float(i + 1)) for i in range(batch)])
    return x, timestep, context


def test_native_batch_is_one_request_with_per_row_timesteps() -> None:
    ex = _Executor(ZImageAdapter(), native_batching="full")
    x, timestep, context = _zimage_batch(batch=3)
    out = ex(x, timestep, context, transformer_options={"uuids": ["pos", "neg"]})

    assert len(ex.sent) == 1
    packed = ex.sent[0]
    assert packed.latents.shape == (3, 16, 1, 4, 6)
    assert torch.equal(packed.prompt_embeds[0], context)
    assert packed.prompt_seq_lens == [[5, 5, 5]]
    # One step for the loop; each row keeps its own timestep, in order.
    assert torch.equal(packed.timesteps, timestep[:1] * 1000.0)
    assert torch.equal(packed.extra_req["comfyui_row_timesteps"], timestep * 1000.0)
    assert packed.unpack_ctx["comfyui_cond_uuid"] == (("pos", "neg"), ("batch", None))
    assert out.shape == x.shape
    assert [float(out[i].mean()) for i in range(3)] == [1.0, 2.0, 3.0]


def test_default_still_sends_one_request_per_row() -> None:
    ex = _Executor(ZImageAdapter(), native_batching="off")
    x, timestep, context = _zimage_batch(batch=2)
    out = ex(x, timestep, context)
    assert len(ex.sent) == 2
    assert [p.latents.shape[0] for p in ex.sent] == [1, 1]
    assert all("comfyui_row_timesteps" not in p.extra_req for p in ex.sent)
    assert [float(out[i].mean()) for i in range(2)] == [1.0, 2.0]


def test_batch_of_one_is_unchanged_with_native_batching() -> None:
    ex = _Executor(ZImageAdapter(), native_batching="full")
    x, timestep, context = _zimage_batch(batch=1)
    ex(x, timestep, context)
    packed = ex.sent[0]
    assert packed.prompt_embeds[0].shape == (5, 8)
    assert packed.prompt_seq_lens == [[5]]
    assert "comfyui_row_timesteps" not in packed.extra_req


@pytest.mark.parametrize(
    "timestep, context",
    [
        (torch.tensor([0.5]), torch.zeros(2, 5, 8)),
        (torch.tensor(0.5), torch.zeros(2, 5, 8)),
        (torch.tensor([0.5, 0.5]), torch.zeros(1, 5, 8)),
    ],
)
def test_rows_that_do_not_match_the_batch_are_rejected(timestep, context) -> None:
    ex = _Executor(ZImageAdapter(), native_batching="full")
    with pytest.raises(ValueError, match="one (timestep|context) row per latent row"):
        ex(torch.zeros(2, 16, 4, 6), timestep, context)
    assert ex.sent == []


def _cfg_batch(images=2):
    """ComfyUI layout: a cond chunk then an uncond chunk, each with every image."""
    x, timestep, context = _zimage_batch(batch=2 * images)
    timestep = torch.cat([torch.linspace(0.9, 0.3, images)] * 2)
    options = {
        "cond_or_uncond": [0, 1],
        "uuids": ["pos", "neg"],
        "sigmas": timestep[:images],
    }
    return x, timestep, context, options


def test_cfg_only_sends_one_request_per_image() -> None:
    ex = _Executor(ZImageAdapter(), native_batching="cfg_only")
    x, timestep, context, options = _cfg_batch(images=2)
    out = ex(x, timestep, context, transformer_options=options)

    assert len(ex.sent) == 2
    for image, packed in enumerate(ex.sent):
        rows = [image, image + 2]  # this image's cond row and uncond row
        assert packed.latents.shape[0] == 2
        assert torch.equal(packed.prompt_embeds[0], context[rows])
        assert torch.equal(
            packed.extra_req["comfyui_row_timesteps"], timestep[rows] * 1000.0
        )
        assert packed.unpack_ctx["comfyui_cond_uuid"] == (
            ("pos", "neg"),
            ("batch", image),
        )
    # Rows come back in ComfyUI's order.
    assert [float(out[i].mean()) for i in range(4)] == [1.0, 2.0, 3.0, 4.0]


def test_cfg_only_with_one_image_is_one_request() -> None:
    ex = _Executor(ZImageAdapter(), native_batching="cfg_only")
    x, timestep, context, options = _cfg_batch(images=1)
    ex(x, timestep, context, transformer_options=options)
    assert len(ex.sent) == 1 and ex.sent[0].latents.shape[0] == 2


def test_cfg_only_needs_the_cond_chunks() -> None:
    ex = _Executor(ZImageAdapter(), native_batching="cfg_only")
    x, timestep, context = _zimage_batch(batch=4)
    with pytest.raises(ValueError, match="cond_or_uncond"):
        ex(x, timestep, context, transformer_options={"cond_or_uncond": [0, 1, 0]})
    assert ex.sent == []


def test_out_of_memory_steps_down_for_the_rest_of_the_run(caplog) -> None:
    ex = _Executor(ZImageAdapter(), native_batching="full", max_rows=2)
    x, timestep, context, options = _cfg_batch(images=2)
    ex.begin_sampler_run()
    with caplog.at_level(logging.WARNING):
        out = ex(x, timestep, context, transformer_options=options)
    # full (4 rows) ran out of memory; cfg_only (2 rows per request) fits.
    assert [p.latents.shape[0] for p in ex.sent] == [2, 2]
    assert "native_batching=full; using cfg_only" in caplog.text
    assert [float(out[i].mean()) for i in range(4)] == [1.0, 2.0, 3.0, 4.0]

    ex.sent.clear()
    ex(x, timestep, context, transformer_options=options)
    assert [p.latents.shape[0] for p in ex.sent] == [2, 2]  # no new full attempt

    ex.begin_sampler_run()  # a new run tries full again
    ex.max_rows = None
    ex.sent.clear()
    ex(x, timestep, context, transformer_options=options)
    assert [p.latents.shape[0] for p in ex.sent] == [4]


def test_out_of_memory_falls_back_to_per_row() -> None:
    ex = _Executor(ZImageAdapter(), native_batching="full", max_rows=1)
    x, timestep, context, options = _cfg_batch(images=2)
    ex(x, timestep, context, transformer_options=options)
    assert [p.latents.shape[0] for p in ex.sent] == [1, 1, 1, 1]


def test_out_of_memory_per_row_is_raised() -> None:
    ex = _Executor(ZImageAdapter(), native_batching="off", max_rows=0)
    x, timestep, context = _zimage_batch(batch=2)
    with pytest.raises(WorkerOutOfMemoryError):
        ex(x, timestep, context)


def test_worker_out_of_memory_error_is_recognised() -> None:
    ex = SGLDiffusionExecutor.__new__(SGLDiffusionExecutor)
    torch.nn.Module.__init__(ex)
    ex.adapter, ex.model_path = ZImageAdapter(), "/test-model"
    ex.session_id, ex._run_id, ex._sent_conds = "oom-test", 0, set()
    errors = iter(["CUDA out of memory. Tried to allocate 2.00 GiB", "shape mismatch"])
    ex.generator = SimpleNamespace(
        server_args=SimpleNamespace(attention_backend_config={}, enable_trace=False),
        _send_to_scheduler_and_wait_for_response=lambda reqs: SimpleNamespace(
            noise_pred=None, error=next(errors)
        ),
    )
    x, timestep, context = _zimage_batch(batch=1)
    packed = ex.adapter.pack(x, timestep, context)
    with (
        mock.patch.object(
            SamplingParams,
            "from_user_sampling_params_args",
            side_effect=lambda model_path, server_args, **kw: SamplingParams(**kw),
        ),
        mock.patch.object(torch, "Generator", side_effect=lambda device: object()),
    ):
        with pytest.raises(WorkerOutOfMemoryError, match="worker failed: CUDA out"):
            ex._execute_packed(packed, x, timestep)
        with pytest.raises(RuntimeError) as error:
            ex._execute_packed(packed, x, timestep)
    assert not isinstance(error.value, WorkerOutOfMemoryError)


def _flux_rows(ex, guidance):
    x = torch.zeros(2, 16, 8, 8)
    timestep = torch.tensor([0.5, 0.5])
    context = torch.stack([torch.full((7, 4096), 1.0), torch.full((7, 4096), 2.0)])
    y = torch.stack([torch.full((768,), 3.0), torch.full((768,), 4.0)])
    return ex(x, timestep, context, y=y, guidance=guidance)


class _FluxExecutor(_Executor):
    def _execute_packed(self, packed, x, timestep):
        self.sent.append(packed)
        # Fake noise_pred: each row is filled with its own T5 context mean.
        rows = packed.prompt_embeds[1].reshape(packed.latents.shape[0], -1).mean(1)
        noise = rows.view(-1, 1, 1).expand_as(packed.latents).clone()
        return self.adapter.unpack(noise, packed, x)


def test_flux_batch_sends_one_text_length_per_row() -> None:
    ex = _FluxExecutor(FluxAdapter(), native_batching="full")
    out = _flux_rows(ex, guidance=torch.tensor([3.5, 3.5]))
    assert len(ex.sent) == 1
    packed = ex.sent[0]
    assert packed.latents.shape == (2, 16, 64)
    assert packed.prompt_seq_lens == [[1, 1], [7, 7]]
    assert packed.pooled_embeds[0].shape == (2, 768)
    assert packed.guidance_scale == 3.5
    assert out.shape == (2, 16, 8, 8)
    assert [float(out[i].mean()) for i in range(2)] == [1.0, 2.0]


def test_flux_rows_with_different_guidance_are_not_batched() -> None:
    ex = _FluxExecutor(FluxAdapter(), native_batching="full")
    with pytest.raises(ValueError, match="different guidance"):
        _flux_rows(ex, guidance=torch.tensor([3.5, 1.0]))
    assert ex.sent == []
    # The default per-row path sends each row's own guidance.
    ex.native_batching = "off"
    _flux_rows(ex, guidance=torch.tensor([3.5, 1.0]))
    assert [p.guidance_scale for p in ex.sent] == [3.5, 1.0]


def test_log_suppression_accepts_a_timestep_per_row() -> None:
    assert SGLDiffusionExecutor.should_suppress_logs(torch.tensor([0.5, 0.2]))
    assert not SGLDiffusionExecutor.should_suppress_logs(torch.tensor([1.0, 0.2]))


def _expand(row_timesteps, latent_batch):
    batch = SimpleNamespace(
        raw_latent_shape=torch.Size([latent_batch, 16, 1, 4, 6]),
        comfyui_row_timesteps=row_timesteps,
    )
    t_device = torch.tensor(900.0, dtype=torch.float64)
    return DenoisingStage.expand_timestep_before_forward(
        None, batch, None, t_device, torch.bfloat16, None, None
    )


def test_worker_uses_one_timestep_per_row() -> None:
    timestep = _expand(torch.tensor([900.0, 600.0, 300.0]), 3)
    assert timestep.tolist() == [900.0, 600.0, 300.0]
    assert timestep.dtype == torch.float64


def test_worker_rejects_row_timesteps_of_the_wrong_length() -> None:
    with pytest.raises(ValueError, match="one timestep per latent row"):
        _expand(torch.tensor([900.0, 600.0]), 3)


def test_native_batching_is_refused_for_unverified_models() -> None:
    _check_native_batching("off", QwenImage21Adapter, {"num_gpus": 2})
    _check_native_batching("full", FluxAdapter, {})
    _check_native_batching("cfg_only", ZImageAdapter, {"num_gpus": 1})
    _check_native_batching("full", ZImageAdapter, {})
    with pytest.raises(ValueError, match="must be one of"):
        _check_native_batching(True, ZImageAdapter, {})
    with pytest.raises(
        ValueError, match=r"only verified for model types \['flux', 'lumina2'\]"
    ):
        _check_native_batching("cfg_only", QwenImage21Adapter, {})
    with pytest.raises(ValueError, match="one GPU"):
        _check_native_batching("full", ZImageAdapter(), {"num_gpus": 2})


def test_options_node_sets_native_batching_only_when_enabled() -> None:
    from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion import options

    node = options.SGLDOptions()
    spec = options.SGLDOptions.INPUT_TYPES()["optional"]["native_batching"]
    assert spec[0] == list(NATIVE_BATCHING_MODES)
    assert spec[1]["default"] == "off"
    assert "native_batching" not in node.create_options()[0]
    assert "native_batching" in options.PLUGIN_KEYS
    assert node.create_options(native_batching="cfg_only")[0]["native_batching"] == (
        "cfg_only"
    )
