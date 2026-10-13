# SPDX-License-Identifier: Apache-2.0
"""Opt-in native batching: one worker request for a whole ComfyUI batch."""

import sys
import types
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.core.generator import (
    _check_native_batching,
)
from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.base import (
    SGLDiffusionExecutor,
)
from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.flux import FluxAdapter
from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.zimage import (
    ZImageAdapter,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingStage,
)


class _Executor(SGLDiffusionExecutor):
    """Real forward/pack/unpack; the worker round trip records the request."""

    def __init__(self, adapter, native_batching):
        torch.nn.Module.__init__(self)
        self.adapter = adapter
        self.native_batching = native_batching
        self.sent = []

    def _execute_packed(self, packed, x, timestep):
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
    ex = _Executor(ZImageAdapter(), native_batching=True)
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
    assert packed.unpack_ctx["comfyui_cond_uuid"] == (("pos", "neg"), "batch")
    assert out.shape == x.shape
    assert [float(out[i].mean()) for i in range(3)] == [1.0, 2.0, 3.0]


def test_default_still_sends_one_request_per_row() -> None:
    ex = _Executor(ZImageAdapter(), native_batching=False)
    x, timestep, context = _zimage_batch(batch=2)
    out = ex(x, timestep, context)
    assert len(ex.sent) == 2
    assert [p.latents.shape[0] for p in ex.sent] == [1, 1]
    assert all("comfyui_row_timesteps" not in p.extra_req for p in ex.sent)
    assert [float(out[i].mean()) for i in range(2)] == [1.0, 2.0]


def test_batch_of_one_is_unchanged_with_native_batching() -> None:
    ex = _Executor(ZImageAdapter(), native_batching=True)
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
    ex = _Executor(ZImageAdapter(), native_batching=True)
    with pytest.raises(ValueError, match="one (timestep|context) row per latent row"):
        ex(torch.zeros(2, 16, 4, 6), timestep, context)
    assert ex.sent == []


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
    _check_native_batching(False, FluxAdapter, {})
    _check_native_batching(True, ZImageAdapter, {"num_gpus": 1})
    with pytest.raises(
        ValueError, match=r"only verified for model types \['lumina2'\]"
    ):
        _check_native_batching(True, FluxAdapter, {})
    with pytest.raises(ValueError, match="one GPU"):
        _check_native_batching(True, ZImageAdapter(), {"num_gpus": 2})


def test_options_node_sets_native_batching_only_when_enabled() -> None:
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.folder_names_and_paths = {}
    folder_paths.get_full_path = lambda folder, name: f"/{folder}/{name}"
    comfy_api = types.ModuleType("comfy_api")
    comfy_api_input = types.ModuleType("comfy_api.input")
    comfy_api_input.VideoInput = type("VideoInput", (), {})
    comfy_api.input = comfy_api_input
    stubs = {
        "folder_paths": folder_paths,
        "comfy_api": comfy_api,
        "comfy_api.input": comfy_api_input,
    }
    with mock.patch.dict(sys.modules, stubs):
        from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion import nodes
    node = nodes.SGLDOptions()
    spec = nodes.SGLDOptions.INPUT_TYPES()["optional"]["native_batching"]
    assert spec == ("BOOLEAN", {"default": False})
    assert "native_batching" not in node.create_options()[0]
    assert node.create_options(native_batching=True)[0]["native_batching"] is True
