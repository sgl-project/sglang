# SPDX-License-Identifier: Apache-2.0
"""Pack / unpack contract for ComfyUI model adapters."""

import importlib
import logging
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.adapter import (
    ComfyUIModelAdapter,
    PackedForward,
    get_adapter_class,
    registered_model_types,
)
from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.base import (
    NATIVE_BATCHING_MODES,
    SGLDiffusionExecutor,
    WorkerOutOfMemoryError,
    _is_out_of_memory,
)
from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.flux import (
    FluxAdapter,
    FluxExecutor,
)
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


def test_flux_pack_zero_fills_missing_pooled() -> None:
    from sglang.multimodal_gen.runtime.layers.visual_embedding import (
        CombinedTimestepGuidanceTextProjEmbeddings,
    )

    # ComfyUI calls Flux with y=None when the cond has no pooled_output; its
    # native model zero-fills y, so the adapter must too.
    adapter = FluxAdapter()
    x = torch.zeros(2, 16, 8, 8)
    context = torch.ones(2, 8, 4096, dtype=torch.bfloat16)
    packed = adapter.pack(x, torch.tensor([0.5, 0.5]), context, y=None)
    y = packed.pooled_embeds[0]
    assert y.shape == (2, 768)
    assert y.dtype == context.dtype
    assert not y.any()
    assert packed.prompt_embeds[0] is y

    # The JIT timestep_embedding kernel only accepts CUDA tensors, so run the
    # embedding where the kernel can run.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embed = CombinedTimestepGuidanceTextProjEmbeddings(
        embedding_dim=32, pooled_projection_dim=768
    ).to(device)
    out = embed(
        torch.tensor([500.0, 500.0], device=device),
        torch.tensor([3.5, 3.5], device=device),
        y.float().to(device),
    )
    assert out.shape == (2, 32)


def test_flux_pack_rejects_kontext_reference_latents() -> None:
    """ComfyUI's Flux forward passes ref_latents from the ReferenceLatent node;
    FluxAdapter must reject it instead of silently running plain T2I."""
    adapter = FluxAdapter()
    x = torch.ones(1, 16, 8, 8)
    timestep = torch.tensor([0.5])
    context = torch.ones(1, 8, 4096)
    with pytest.raises(ValueError, match="Kontext"):
        adapter.pack(
            x,
            timestep,
            context,
            y=torch.ones(1, 768),
            ref_latents=[torch.ones(1, 16, 8, 8)],
        )


def test_flux_pack_and_unpack_roundtrip_odd_latent_size() -> None:
    """Regression: a width/height not divisible by patch_size=2 (e.g. a
    1032px-wide ComfyUI latent, 1032 // 8 = 129) raised a view() RuntimeError
    in _pack_latents before it was padded like QwenImageAdapter."""
    adapter = FluxAdapter()
    x = torch.arange(1 * 16 * 8 * 9, dtype=torch.float32).reshape(1, 16, 8, 9)
    timestep = torch.tensor([0.5])
    context = torch.ones(1, 8, 4096)
    y = torch.ones(1, 768)
    packed = adapter.pack(x, timestep, context, y=y, guidance=torch.tensor([1.0]))
    out = adapter.unpack(packed.latents, packed, x)
    assert out.shape == x.shape
    assert torch.equal(out, x)


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


def test_conditioning_the_worker_cannot_apply_is_rejected() -> None:
    """Reference latents reach apply_model as a kwarg; an adapter that does not
    forward them must fail instead of ignoring them."""
    ex = _RecordingExecutor(FluxAdapter())
    with pytest.raises(ValueError, match="ref_latents"):
        _flux_step(ex, ref_latents=[torch.ones(1, 16, 8, 8)])
    assert ex.sent == []


def test_comfyui_weight_access_gets_a_clear_error() -> None:
    """ComfyUI's BaseModel holds the executor as diffusion_model; LoraLoader's key
    map calls state_dict() on it, which must not recurse through the executor."""
    base = torch.nn.Module()
    config = SimpleNamespace(unet_config={"dtype": torch.bfloat16})
    base.diffusion_model = FluxExecutor(None, "flux.safetensors", base, config)
    base.to("cpu")
    with pytest.raises(RuntimeError, match="SGLDLoraLoader"):
        base.state_dict()


def _qwen_image_edit_adapter():
    """Import the Qwen adapters with the one ComfyUI helper they use stubbed."""
    common_dit = types.ModuleType("comfy.ldm.common_dit")
    common_dit.pad_to_patch_size = lambda x, patch_size: x
    comfy, comfy_ldm = types.ModuleType("comfy"), types.ModuleType("comfy.ldm")
    comfy.ldm, comfy_ldm.common_dit = comfy_ldm, common_dit
    stubs = {"comfy": comfy, "comfy.ldm": comfy_ldm, "comfy.ldm.common_dit": common_dit}
    with patch.dict(sys.modules, stubs):
        module = importlib.import_module(
            "sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.qwen_image"
        )
    return module.QwenImageEditAdapter()


def test_qwen_image_edit_pack_keeps_all_ref_latents() -> None:
    """Qwen-Image-Edit packed only the first of several reference images."""
    adapter = _qwen_image_edit_adapter()
    x = torch.ones(1, 16, 1, 90, 160)
    timestep = torch.tensor([1.0])
    context = torch.ones(1, 19, 2560)
    ref_a = torch.ones(1, 16, 1, 64, 64)
    ref_b = torch.ones(1, 16, 1, 32, 96)
    ref_c = torch.ones(1, 16, 1, 48, 48)

    packed = adapter.pack(x, timestep, context, ref_latents=[ref_a, ref_b, ref_c])

    sizes = packed.extra_req["vae_image_sizes"]
    assert sizes == [(64, 64), (96, 32), (48, 48)]

    expected_tokens = sum(
        (h // 2) * (w // 2) for h, w in [(64, 64), (32, 96), (48, 48)]
    )
    assert packed.extra_req["image_latent"].shape[1] == expected_tokens


def test_qwen_image_edit_pack_rejects_unsupported_kwargs() -> None:
    """attention_mask, additional_t_cond and ref_latents_method must be
    rejected instead of silently dropped."""
    adapter = _qwen_image_edit_adapter()
    x = torch.ones(1, 16, 1, 90, 160)
    timestep = torch.tensor([1.0])
    context = torch.ones(1, 19, 2560)

    with pytest.raises(ValueError, match="attention_mask"):
        adapter.pack(x, timestep, context, attention_mask=torch.ones(1, 19))
    with pytest.raises(ValueError, match="additional_t_cond"):
        adapter.pack(x, timestep, context, additional_t_cond=torch.zeros(1))
    with pytest.raises(ValueError, match="ref_latents_method"):
        adapter.pack(x, timestep, context, ref_latents_method="negative_index")

    # The default method and absent kwargs must not be rejected.
    adapter.pack(x, timestep, context, ref_latents_method="index")
    adapter.pack(x, timestep, context)


def test_zimage_pack_rejects_omni_reference_conditioning() -> None:
    """siglip_feats/ref_contexts must be rejected; num_tokens/attention_mask
    (sent on every real call) must keep working."""
    adapter = ZImageAdapter()
    x = torch.ones(1, 16, 90, 160)
    timestep = torch.tensor([1.0])
    context = torch.ones(1, 19, 2560)

    with pytest.raises(ValueError, match="siglip_feats"):
        adapter.pack(x, timestep, context, siglip_feats=[torch.ones(1, 4, 8)])
    with pytest.raises(ValueError, match="ref_contexts"):
        adapter.pack(x, timestep, context, ref_contexts=[torch.ones(1, 4)])

    # Sent on every real ComfyUI call; must not be rejected.
    adapter.pack(x, timestep, context, num_tokens=19, attention_mask=torch.ones(1, 19))
    adapter.pack(x, timestep, context, siglip_feats=[], ref_contexts=None)
    adapter.pack(x, timestep, context)


class _RowRecordingExecutor(SGLDiffusionExecutor):
    """Real forward/pack/unpack; the worker round trip just records the request."""

    def __init__(self, adapter):
        torch.nn.Module.__init__(self)
        self.adapter = adapter
        self.sent = []

    def _execute_packed(self, packed, x, timestep):
        self.sent.append((packed, timestep))
        # Fake noise_pred that identifies the row by its T5 context.
        noise = torch.full_like(packed.latents, float(packed.prompt_embeds[-1].mean()))
        return self.adapter.unpack(noise, packed, x)


def test_batched_forward_sends_one_request_per_row() -> None:
    # ComfyUI batches CFG cond/uncond into one call; the worker path is per-sample.
    ex = _RowRecordingExecutor(FluxAdapter())
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
    applied_conditioning = ("control", "ref_latents")

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
    ex = _RowRecordingExecutor(adapter)
    shared_ref = torch.ones(1, 16, 4, 4)
    sample_sigmas = torch.linspace(1.0, 0.0, 5)
    patches = {"attn": []}
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
    ex = _RowRecordingExecutor(adapter)
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
    ex = _RowRecordingExecutor(_KwargsAdapter())
    with pytest.raises(ValueError, match="cannot split a batch of 2"):
        ex(
            torch.zeros(2, 16, 4, 4),
            torch.tensor([0.5, 0.5]),
            torch.zeros(2, 3, 8),
            guidance=torch.zeros(3),
        )


def test_unbatched_forward_is_a_single_request() -> None:
    adapter = _KwargsAdapter()
    ex = _RowRecordingExecutor(adapter)
    x = torch.zeros(1, 16, 4, 4)
    ex(x, torch.tensor([0.5]), torch.zeros(1, 3, 8))
    assert len(adapter.calls) == 1 and adapter.calls[0][0] is x
    # MiniMax-H3 passes x as [video, audio]; that path is not split either.
    av = [torch.zeros(2, 16, 1, 4, 4), torch.zeros(2, 8, 4)]
    ex(av, torch.tensor([0.5, 0.5]), torch.zeros(2, 3, 8))
    assert len(adapter.calls) == 2 and adapter.calls[1][0] is av


def _pack_flux(**kwargs):
    x = torch.zeros(2, 16, 8, 8)
    return FluxAdapter().pack(
        x, torch.tensor([0.5]), torch.ones(2, 8, 4096), y=torch.ones(2, 768), **kwargs
    )


def test_flux_pack_forwards_control_residuals_with_gaps() -> None:
    seq = (8 // 2) * (8 // 2)
    double = [torch.ones(2, seq, 3072), None, torch.ones(2, seq, 3072)]
    single = [torch.ones(2, seq, 3072)]
    packed = _pack_flux(control={"input": double, "output": single})
    assert packed.control["input"][1] is None
    assert len(packed.control["input"]) == 3
    assert len(packed.control["output"]) == 1
    assert _pack_flux().control is None
    assert _pack_flux(control={"input": [None]}).control is None


def test_flux_pack_rejects_misshaped_control() -> None:
    with pytest.raises(ValueError, match=r"control input\[0\]"):
        _pack_flux(control={"input": [torch.ones(2, 3, 3072)]})


def test_control_is_rejected_for_adapters_that_do_not_apply_it() -> None:
    class _Executor(SGLDiffusionExecutor):
        adapter_cls = ZImageAdapter

        def __init__(self):
            torch.nn.Module.__init__(self)
            self.adapter = ZImageAdapter()

    executor = _Executor()
    control = {"input": [torch.ones(1, 4, 8)]}
    with pytest.raises(ValueError, match="does not apply control"):
        executor(
            torch.ones(1, 16, 8, 8),
            torch.tensor([1.0]),
            torch.ones(1, 8, 16),
            control=control,
        )
    assert FluxAdapter.applied_conditioning == ("control",)
    assert ZImageAdapter.applied_conditioning == ()


# --- opt-in native batching ---------------------------------------------------


class _BatchingExecutor(SGLDiffusionExecutor):
    """Real forward/pack/unpack; the worker round trip records the request.

    Each returned row is filled with its own text-embedding mean, so the
    output order shows which context each row was computed from.
    """

    def __init__(self, adapter, native_batching, max_rows=None, embed=0):
        torch.nn.Module.__init__(self)
        self.adapter, self.native_batching = adapter, native_batching
        self.max_rows, self.embed = max_rows, embed  # more rows "run out of memory"
        self.sent, self._run_id = [], 0

    def _execute_packed(self, packed, x, timestep):
        rows = packed.latents.shape[0]
        if self.max_rows is not None and rows > self.max_rows:
            raise WorkerOutOfMemoryError("worker failed: CUDA out of memory")
        self.sent.append(packed)
        means = packed.prompt_embeds[self.embed].reshape(rows, -1).mean(1)
        shape = (-1,) + (1,) * (packed.latents.ndim - 1)
        noise = means.view(shape).expand_as(packed.latents).clone()
        return self.adapter.unpack(noise, packed, x)


def _zimage_batch(batch=2):
    x = torch.zeros(batch, 16, 4, 6)
    timestep = torch.linspace(0.9, 0.3, batch)
    context = torch.stack([torch.full((5, 8), float(i + 1)) for i in range(batch)])
    return x, timestep, context


def _cfg_batch(images=2):
    """ComfyUI layout: a cond chunk then an uncond chunk, each with every image."""
    x, _, context = _zimage_batch(batch=2 * images)
    timestep = torch.cat([torch.linspace(0.9, 0.3, images)] * 2)
    options = {"cond_or_uncond": [0, 1], "uuids": ["pos", "neg"]}
    return x, timestep, context, {**options, "sigmas": timestep[:images]}


def _row_means(out):
    return [float(row.mean()) for row in out]


def test_native_batch_is_one_request_with_per_row_timesteps() -> None:
    ex = _BatchingExecutor(ZImageAdapter(), native_batching="full")
    x, timestep, context = _zimage_batch(batch=3)
    out = ex(x, timestep, context, transformer_options={"uuids": ["pos", "neg"]})
    (packed,) = ex.sent
    assert packed.latents.shape == (3, 16, 1, 4, 6)
    assert packed.prompt_seq_lens == [[5, 5, 5]]
    # One step for the worker's loop; each row keeps its own timestep.
    assert torch.equal(packed.timesteps, timestep[:1] * 1000.0)
    assert torch.equal(packed.extra_req["comfyui_row_timesteps"], timestep * 1000.0)
    assert packed.unpack_ctx["comfyui_cond_uuid"] == (("pos", "neg"), ("batch", None))
    assert _row_means(out) == [1.0, 2.0, 3.0]
    assert SGLDiffusionExecutor.should_suppress_logs(torch.tensor([0.5, 0.2]))
    assert not SGLDiffusionExecutor.should_suppress_logs(torch.tensor([1.0, 0.2]))


@pytest.mark.parametrize("batch, mode", [(2, "off"), (1, "full")])
def test_off_and_batch_of_one_send_single_rows(batch, mode) -> None:
    ex = _BatchingExecutor(ZImageAdapter(), native_batching=mode)
    out = ex(*_zimage_batch(batch=batch))
    assert [p.prompt_seq_lens for p in ex.sent] == [[[5]]] * batch
    assert all("comfyui_row_timesteps" not in p.extra_req for p in ex.sent)
    assert _row_means(out) == [1.0, 2.0][:batch]


@pytest.mark.parametrize(
    "timestep, context",
    [
        (torch.tensor(0.5), torch.zeros(2, 5, 8)),
        (torch.tensor([0.5, 0.5]), torch.zeros(1, 5, 8)),
    ],
)
def test_rows_that_do_not_match_the_batch_are_rejected(timestep, context) -> None:
    ex = _BatchingExecutor(ZImageAdapter(), native_batching="full")
    with pytest.raises(ValueError, match="one (timestep|context) row per latent row"):
        ex(torch.zeros(2, 16, 4, 6), timestep, context)
    assert ex.sent == []


def test_cfg_only_sends_one_request_per_image() -> None:
    ex = _BatchingExecutor(ZImageAdapter(), native_batching="cfg_only")
    x, timestep, context, options = _cfg_batch(images=2)
    out = ex(x, timestep, context, transformer_options=options)
    for image, packed in enumerate(ex.sent):
        rows = [image, image + 2]  # this image's cond row and uncond row
        assert torch.equal(packed.prompt_embeds[0], context[rows])
        assert torch.equal(
            packed.extra_req["comfyui_row_timesteps"], timestep[rows] * 1000.0
        )
        assert packed.unpack_ctx["comfyui_cond_uuid"][1] == ("batch", image)
    assert len(ex.sent) == 2
    assert _row_means(out) == [1.0, 2.0, 3.0, 4.0]  # ComfyUI's row order

    # Without cond chunks to group by, cfg_only sends one request per row.
    ex.sent.clear()
    ex(*_zimage_batch(batch=4), transformer_options={"cond_or_uncond": [0, 1, 0]})
    assert [p.latents.shape[0] for p in ex.sent] == [1, 1, 1, 1]


def test_out_of_memory_steps_down_for_the_rest_of_the_run(caplog) -> None:
    ex = _BatchingExecutor(ZImageAdapter(), native_batching="full", max_rows=2)
    x, timestep, context, options = _cfg_batch(images=2)
    ex.begin_sampler_run()
    with caplog.at_level(logging.WARNING):
        out = ex(x, timestep, context, transformer_options=options)
    assert [p.latents.shape[0] for p in ex.sent] == [2, 2]  # full ran out of memory
    assert "native_batching=full; using cfg_only" in caplog.text
    assert _row_means(out) == [1.0, 2.0, 3.0, 4.0]

    ex.sent.clear()
    ex(x, timestep, context, transformer_options=options)
    assert [p.latents.shape[0] for p in ex.sent] == [2, 2]  # no new full attempt

    ex.begin_sampler_run()  # a new run tries full again
    ex.max_rows = None
    ex.sent.clear()
    ex(x, timestep, context, transformer_options=options)
    assert [p.latents.shape[0] for p in ex.sent] == [4]


@pytest.mark.parametrize("images, options", [(2, True), (1, True), (2, False)])
def test_out_of_memory_falls_back_to_per_row(images, options) -> None:
    """cfg_only is skipped when it would resend the same rows or cannot group."""
    ex = _BatchingExecutor(ZImageAdapter(), native_batching="full", max_rows=1)
    x, timestep, context, cfg_options = _cfg_batch(images=images)
    ex(x, timestep, context, transformer_options=cfg_options if options else {})
    assert [p.latents.shape[0] for p in ex.sent] == [1] * (2 * images)

    ex = _BatchingExecutor(ZImageAdapter(), native_batching="off", max_rows=0)
    with pytest.raises(WorkerOutOfMemoryError):
        ex(*_zimage_batch(batch=2))


def test_worker_out_of_memory_is_recognised() -> None:
    assert _is_out_of_memory("CUDA out of memory. Tried to allocate 2.00 GiB")
    assert _is_out_of_memory(torch.cuda.OutOfMemoryError("x"))
    assert not _is_out_of_memory("shape mismatch")


def test_flux_batch_sends_one_text_length_per_row_and_one_guidance() -> None:
    ex = _BatchingExecutor(FluxAdapter(), native_batching="full", embed=1)
    x, timestep = torch.zeros(2, 16, 8, 8), torch.tensor([0.5, 0.5])
    context = torch.stack([torch.full((7, 4096), 1.0), torch.full((7, 4096), 2.0)])
    y = torch.ones(2, 768)
    out = ex(x, timestep, context, y=y, guidance=torch.tensor([3.5, 3.5]))
    (packed,) = ex.sent
    assert packed.prompt_seq_lens == [[1, 1], [7, 7]]
    assert packed.guidance_scale == 3.5
    assert _row_means(out) == [1.0, 2.0]

    # The worker takes one guidance per request; per-row sends each its own.
    with pytest.raises(ValueError, match="different guidance"):
        ex(x, timestep, context, y=y, guidance=torch.tensor([3.5, 1.0]))
    ex.native_batching = "off"
    ex.sent.clear()
    ex(x, timestep, context, y=y, guidance=torch.tensor([3.5, 1.0]))
    assert [p.guidance_scale for p in ex.sent] == [3.5, 1.0]


def test_worker_uses_one_timestep_per_row() -> None:
    from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
        DenoisingStage,
    )

    def expand(row_timesteps):
        batch = SimpleNamespace(
            raw_latent_shape=torch.Size([3, 16, 1, 4, 6]),
            comfyui_row_timesteps=row_timesteps,
        )
        t_device = torch.tensor(900.0, dtype=torch.float64)
        return DenoisingStage.expand_timestep_before_forward(
            None, batch, None, t_device, torch.bfloat16, None, None
        )

    timestep = expand(torch.tensor([900.0, 600.0, 300.0]))
    assert timestep.tolist() == [900.0, 600.0, 300.0]
    assert timestep.dtype == torch.float64
    with pytest.raises(ValueError, match="one timestep per latent row"):
        expand(torch.tensor([900.0, 600.0]))


def test_native_batching_is_opt_in_and_refused_for_unverified_setups() -> None:
    from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion import options
    from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.core.generator import (
        _check_native_batching,
    )
    from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.qwen_image21 import (
        QwenImage21Adapter,
    )

    node = options.SGLDOptions()
    spec = options.SGLDOptions.INPUT_TYPES()["optional"]["native_batching"]
    assert spec[0] == list(NATIVE_BATCHING_MODES) and spec[1]["default"] == "off"
    assert "native_batching" not in node.create_options()[0]
    assert node.create_options(native_batching="full")[0]["native_batching"] == "full"

    _check_native_batching("off", QwenImage21Adapter, {"num_gpus": 2})
    _check_native_batching("cfg_only", ZImageAdapter, {"num_gpus": 1})
    _check_native_batching("full", FluxAdapter, {})
    with pytest.raises(ValueError, match="must be one of"):
        _check_native_batching(True, ZImageAdapter, {})
    with pytest.raises(ValueError, match=r"\['flux', 'lumina2'\]"):
        _check_native_batching("full", QwenImage21Adapter, {})
    with pytest.raises(ValueError, match="one GPU"):
        _check_native_batching("full", ZImageAdapter, {"num_gpus": 2})
