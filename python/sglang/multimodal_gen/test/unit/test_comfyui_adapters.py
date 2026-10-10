# SPDX-License-Identifier: Apache-2.0
"""Pack / unpack contract for ComfyUI model adapters."""

import pytest
import torch

from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.adapter import (
    ComfyUIModelAdapter,
    PackedForward,
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


class _RecordingExecutor(SGLDiffusionExecutor):
    """Real forward/pack/unpack; the worker round trip just records the request."""

    def __init__(self, adapter):
        torch.nn.Module.__init__(self)
        self.adapter = adapter
        self.sent = []

    def _execute_packed(self, packed, x, timestep, *, cond_uuid=None):
        self.sent.append((packed, timestep))
        # Fake noise_pred that identifies the row by its T5 context.
        noise = torch.full_like(packed.latents, float(packed.prompt_embeds[-1].mean()))
        return self.adapter.unpack(noise, packed, x)


def test_batched_forward_sends_one_request_per_row() -> None:
    # ComfyUI batches CFG cond/uncond into one call; the worker path is per-sample.
    ex = _RecordingExecutor(FluxAdapter())
    x = torch.zeros(2, 16, 8, 8)
    timestep = torch.tensor([0.5, 0.5])
    context = torch.stack([torch.full((8, 4096), 1.0), torch.full((8, 4096), 2.0)])
    y = torch.stack([torch.full((768,), 3.0), torch.full((768,), 4.0)])
    out = ex(x, timestep, context, y=y, guidance=torch.tensor([3.5, 3.5]))

    assert len(ex.sent) == 2
    for row, (packed, row_timestep) in enumerate(ex.sent):
        assert packed.latents.shape[0] == 1
        assert torch.equal(row_timestep, timestep[row : row + 1])
        assert torch.equal(packed.prompt_embeds[1], context[row : row + 1])
        assert torch.equal(packed.pooled_embeds[0], y[row : row + 1])
        assert packed.prompt_seq_lens == [[1], [8]]
    assert out.shape == x.shape
    assert torch.all(out[0] == 1.0) and torch.all(out[1] == 2.0)


class _KwargsAdapter(ComfyUIModelAdapter):
    def __init__(self):
        self.calls = []

    def pack(self, x, timestep, context, **kwargs):
        self.calls.append((x, timestep, context, kwargs))
        return PackedForward(
            latents=torch.zeros(1),
            timesteps=timestep,
            prompt_embeds=[torch.zeros(1)],
            height=1,
            width=1,
        )

    def unpack(self, noise_pred, packed, x):
        return x


def test_batched_forward_slices_only_batched_values() -> None:
    adapter = _KwargsAdapter()
    ex = _RecordingExecutor(adapter)
    shared_ref = torch.ones(1, 16, 4, 4)
    sample_sigmas = torch.linspace(1.0, 0.0, 5)
    patches = {"attn": [object()]}
    timestep = torch.tensor([0.5, 0.25])
    options = {
        "cond_or_uncond": [0, 1],
        "sigmas": timestep,
        "sample_sigmas": sample_sigmas,
        "patches": patches,
    }
    ex(
        torch.zeros(2, 16, 4, 4),
        timestep,
        torch.zeros(2, 3, 8),
        ref_latents=[
            torch.stack([torch.zeros(16, 4, 4), torch.ones(16, 4, 4)]),
            shared_ref,
        ],
        control={"input": [torch.stack([torch.zeros(4), torch.ones(4)]), None]},
        transformer_options=options,
    )
    assert len(adapter.calls) == 2
    for row, (_, _, _, kwargs) in enumerate(adapter.calls):
        per_row, shared = kwargs["ref_latents"]
        assert per_row.shape == (1, 16, 4, 4) and torch.all(per_row == row)
        assert shared is shared_ref
        control, missing = kwargs["control"]["input"]
        assert control.shape == (1, 4) and torch.all(control == row)
        assert missing is None
        row_options = kwargs["transformer_options"]
        assert row_options["cond_or_uncond"] == [row]
        assert torch.equal(row_options["sigmas"], timestep[row : row + 1])
        assert row_options["sample_sigmas"] is sample_sigmas
        assert row_options["patches"] is patches
    assert options["cond_or_uncond"] == [0, 1]


def test_batched_forward_slices_per_chunk_options() -> None:
    # batch_size=2 with CFG: two cond chunks of two rows each.
    adapter = _KwargsAdapter()
    ex = _RecordingExecutor(adapter)
    sigmas = torch.tensor([0.5, 0.25])
    ex(
        torch.zeros(4, 16, 4, 4),
        torch.cat([sigmas, sigmas]),
        torch.zeros(4, 3, 8),
        transformer_options={
            "cond_or_uncond": [0, 1],
            "uuids": ["pos", "neg"],
            "sigmas": sigmas,
        },
    )
    seen = [kwargs["transformer_options"] for *_, kwargs in adapter.calls]
    assert [o["cond_or_uncond"] for o in seen] == [[0], [0], [1], [1]]
    assert [o["uuids"] for o in seen] == [["pos"], ["pos"], ["neg"], ["neg"]]
    for row, options in enumerate(seen):
        assert torch.equal(options["sigmas"], sigmas[row % 2 : row % 2 + 1])


def test_batched_forward_rejects_ambiguous_leading_dim() -> None:
    ex = _RecordingExecutor(_KwargsAdapter())
    with pytest.raises(ValueError, match="cannot split a batch of 2"):
        ex(
            torch.zeros(2, 16, 4, 4),
            torch.tensor([0.5, 0.5]),
            torch.zeros(2, 3, 8),
            guidance=torch.zeros(3),
        )


def test_unbatched_forward_is_a_single_request() -> None:
    adapter = _KwargsAdapter()
    ex = _RecordingExecutor(adapter)
    x = torch.zeros(1, 16, 4, 4)
    ex(x, torch.tensor([0.5]), torch.zeros(1, 3, 8))
    assert len(adapter.calls) == 1 and adapter.calls[0][0] is x
    # MiniMax-H3 passes x as [video, audio]; that path is not split either.
    av = [torch.zeros(2, 16, 1, 4, 4), torch.zeros(2, 8, 4)]
    ex(av, torch.tensor([0.5, 0.5]), torch.zeros(2, 3, 8))
    assert len(adapter.calls) == 2 and adapter.calls[1][0] is av
