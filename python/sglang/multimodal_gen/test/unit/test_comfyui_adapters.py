# SPDX-License-Identifier: Apache-2.0
"""Pack / unpack contract for ComfyUI model adapters."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.adapter import (
    get_adapter_class,
    registered_model_types,
)
from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.base import (
    SGLDiffusionExecutor,
)
from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.flux import FluxAdapter
from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.zimage import (
    ZImageAdapter,
)
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


def test_registered_comfyui_model_types() -> None:
    types = registered_model_types()
    assert "flux" in types
    assert "lumina2" in types
    assert get_adapter_class("lumina2") is ZImageAdapter
    assert get_adapter_class("flux") is FluxAdapter
    assert get_adapter_class("lumina2").pipeline_class_name == "ZImagePipeline"


def test_zimage_pack_sets_seq_lens_and_time_dim() -> None:
    adapter = ZImageAdapter()
    x = torch.ones(1, 16, 90, 160)
    timestep = torch.tensor([1.0])
    context = torch.ones(1, 19, 2560)
    packed = adapter.pack(x, timestep, context)
    assert packed.latents.shape == (1, 16, 1, 90, 160)
    assert packed.prompt_embeds[0].shape == (19, 2560)
    assert packed.prompt_seq_lens == [[19]]
    assert packed.height == 720
    assert packed.width == 1280
    assert torch.equal(packed.timesteps, timestep * 1000.0)

    pred = torch.ones(1, 16, 1, 90, 160)
    out = adapter.unpack(pred, packed, x)
    assert out.shape == x.shape


def test_flux_pack_and_unpack_roundtrip() -> None:
    adapter = FluxAdapter()
    x = torch.arange(1 * 16 * 8 * 8, dtype=torch.float32).reshape(1, 16, 8, 8)
    timestep = torch.tensor([0.5])
    context = torch.ones(1, 8, 4096)
    y = torch.ones(1, 768)
    packed = adapter.pack(x, timestep, context, y=y, guidance=torch.tensor([1.0]))
    assert packed.latents.ndim == 3
    assert packed.pooled_embeds[0] is y
    assert packed.guidance_scale == 1.0
    out = adapter.unpack(packed.latents, packed, x)
    assert out.shape == x.shape
    assert torch.equal(out, x)

    default = adapter.pack(x, timestep, context, y=y)
    assert default.guidance_scale == 3.5


def test_worker_error_is_raised_not_unpacked() -> None:
    """Unpacking a failed reply replaced the worker's message with a misleading
    adapter TypeError about noise_pred being None."""
    ex = SGLDiffusionExecutor.__new__(SGLDiffusionExecutor)
    torch.nn.Module.__init__(ex)
    ex.adapter, ex.model_path = FluxAdapter(), "/test-model"
    ex.session_id, ex._run_id, ex._sent_conds = "error-test", 0, set()
    ex.generator = SimpleNamespace(
        server_args=SimpleNamespace(attention_backend_config={}, enable_trace=False),
        _send_to_scheduler_and_wait_for_response=lambda reqs: SimpleNamespace(
            noise_pred=None, error="index_copy_(): shape mismatch"
        ),
    )
    x, t = torch.randn(1, 16, 8, 8), torch.full((1,), 0.5)
    packed = ex.adapter.pack(x, t, torch.randn(1, 7, 32), y=torch.randn(1, 768))
    with (
        patch.object(
            SamplingParams,
            "from_user_sampling_params_args",
            side_effect=lambda model_path, server_args, **kw: SamplingParams(**kw),
        ),
        patch.object(torch, "Generator", side_effect=lambda device: object()),
        pytest.raises(RuntimeError, match="worker failed: index_copy_"),
    ):
        ex._execute_packed(packed, x, t)


class _RecordingExecutor(SGLDiffusionExecutor):
    """The real forward(); records what would be sent to the worker."""

    def __init__(self, adapter):
        torch.nn.Module.__init__(self)
        self.adapter, self.sent = adapter, []

    def _execute_packed(self, packed, x, timestep):
        self.sent.append(packed)
        return x


def _flux_step(ex, **kwargs):
    x, t = torch.randn(1, 16, 8, 8), torch.full((1,), 0.5)
    ex(x, t, torch.randn(1, 7, 32), y=torch.randn(1, 768), **kwargs)


@pytest.mark.parametrize(
    "options",
    [
        {"patches_replace": {"dit": {("double_block", 3): object()}}},
        {"patches": {"attn1_patch": [object()]}},
        {"optimized_attention_override": object()},
    ],
)
def test_comfy_model_patches_are_rejected_not_dropped(options) -> None:
    """ComfyUI model patches (H3 Fun ControlNet block replace, attention backend
    override) never reach the worker; ignoring them gave bit-identical output."""
    ex = _RecordingExecutor(FluxAdapter())
    with pytest.raises(ValueError, match="cannot apply ComfyUI model patches"):
        _flux_step(ex, transformer_options=options)
    _flux_step(ex, transformer_options={"patches": {}, "patches_replace": {"dit": {}}})
    assert len(ex.sent) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"control": {"output": [torch.ones(1)]}},
        {"ref_latents": [torch.ones(1, 16, 8, 8)]},
    ],
)
def test_conditioning_the_worker_cannot_apply_is_rejected(kwargs) -> None:
    """ControlNet residuals and reference latents reach apply_model as kwargs; an
    adapter that does not forward them must fail instead of ignoring them."""
    ex = _RecordingExecutor(FluxAdapter())
    with pytest.raises(ValueError, match=next(iter(kwargs))):
        _flux_step(ex, **kwargs)
    assert ex.sent == []
