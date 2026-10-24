# SPDX-License-Identifier: Apache-2.0
"""Pack / unpack contract for ComfyUI model adapters."""

import copy
import importlib
import pickle
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
    SGLDiffusionExecutor,
)
from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.cache_options import (
    CACHE_OPTIONS_KEY,
    build_cache_options,
    expected_model_calls,
    guider_runs_cfg,
    read_cache_options,
    request_cache_dit_for_run,
    with_cache_options,
)
from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.flux import (
    FluxAdapter,
    FluxExecutor,
)
from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.minimax_h3 import (
    MiniMaxH3Executor,
    worker_transformer_options,
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


class _FakePatcher:
    def __init__(self, model_options, executor):
        self.model_options = model_options
        self.model = SimpleNamespace(diffusion_model=executor)

    def clone(self):
        return _FakePatcher(
            copy.deepcopy(self.model_options), self.model.diffusion_model
        )


def _bare_executor(cls, enable_cache_dit=None):
    executor = cls.__new__(cls)
    torch.nn.Module.__init__(executor)
    executor.adapter = cls.adapter_cls()
    executor.enable_cache_dit = enable_cache_dit
    return executor


def test_cache_options_node_returns_clone_and_keeps_input() -> None:
    patcher = _FakePatcher(
        {"transformer_options": {"sample_sigmas": [1.0, 0.0]}},
        _bare_executor(MiniMaxH3Executor),
    )
    options = build_cache_options(True, {"max_warmup_steps": 2})
    out = with_cache_options(patcher, options)
    assert out is not patcher
    assert patcher.model_options["transformer_options"] == {"sample_sigmas": [1.0, 0.0]}
    stored = out.model_options["transformer_options"]
    assert stored["sample_sigmas"] == [1.0, 0.0]
    assert read_cache_options(stored) == options
    pickle.dumps(options)
    # Switching a tuned node to off must not raise; default keeps the knobs.
    knobs = {"max_warmup_steps": 2}
    assert build_cache_options(False, knobs)["cache_dit_params"] is None
    assert build_cache_options(None, knobs)["cache_dit_params"] == knobs


@pytest.mark.parametrize(
    "enable, params, error",
    [
        (True, {"not_a_knob": 1}, "Unknown cache_dit_params"),
        # cache-dit would only assert Fn >= 1 mid-sampling
        (None, {"Fn_compute_blocks": 0}, "Fn_compute_blocks must be >= 1"),
        (True, {"secondary": {"Fn_compute_blocks": 0}}, "Fn_compute_blocks"),
    ],
)
def test_cache_options_reject_bad_requests(enable, params, error) -> None:
    with pytest.raises(ValueError, match=error):
        build_cache_options(enable, params)


@pytest.mark.parametrize(
    "sampler, sigmas, calls",
    [
        ("sample_euler", [3.0, 2.0, 1.0, 0.0], 3),
        ("sample_dpmpp_2m", [3.0, 2.0, 1.0, 0.05], 3),
        # two calls per step, one on a final step that lands on sigma 0
        ("sample_heun", [3.0, 2.0, 1.0, 0.0], 5),
        ("sample_dpmpp_sde", [3.0, 2.0, 1.0, 0.05], 6),
        ("sample_heunpp2", [5.0, 4.0, 3.0, 2.0, 1.0, 0.0], 12),
        ("sample_heunpp2", [1.0, 0.0], 1),
        # unknown samplers get the 3N upper bound
        ("sample_custom", [3.0, 2.0, 1.0, 0.0], 9),
        (None, [3.0, 2.0, 1.0, 0.0], 9),
        ("sample_euler", [1.0], 0),
    ],
)
def test_expected_model_calls(sampler, sigmas, calls) -> None:
    assert expected_model_calls(sampler, sigmas) == calls


def test_expected_model_calls_accepts_the_sampler_function() -> None:
    def sample_heun():
        pass

    assert expected_model_calls(sample_heun, [3.0, 2.0, 1.0, 0.0]) == 5


def test_cache_dit_run_sends_sampler_steps_and_cfg_mode() -> None:
    """The worker counts DiT calls as Cache-DiT steps, so it needs the run's
    step count and whether cond and uncond calls alternate within a step."""
    sigmas = torch.linspace(1, 0, 26)
    packed = SimpleNamespace(extra_req={})
    request_cache_dit_for_run(
        packed, {"sample_sigmas": sigmas, "cond_or_uncond": [0]}, False, 49
    )
    assert packed.extra_req["comfyui_num_steps"] == 49
    # without the sampler's call count, assume the upper bound
    request_cache_dit_for_run(
        packed, {"sample_sigmas": sigmas, "cond_or_uncond": [0]}, False
    )
    assert packed.extra_req["comfyui_num_steps"] == 75
    assert packed.extra_req["comfyui_separate_cfg"] is False
    request_cache_dit_for_run(
        packed, {"sample_sigmas": sigmas, "cond_or_uncond": [1]}, True
    )
    assert packed.extra_req["comfyui_separate_cfg"] is True
    with pytest.raises(ValueError, match="does not run CFG"):
        request_cache_dit_for_run(
            packed, {"sample_sigmas": [1.0, 0.0], "cond_or_uncond": [1]}, False
        )
    with pytest.raises(ValueError, match="sample_sigmas"):
        request_cache_dit_for_run(packed, {"cond_or_uncond": [0]}, False)


def test_only_a_cfg_guider_with_a_negative_runs_both_branches() -> None:
    """ComfyUI skips the uncond call at cfg 1, and BasicGuider has no cfg."""
    conds = {"positive": [], "negative": []}
    assert guider_runs_cfg(SimpleNamespace(cfg=4.0, conds=conds))
    assert not guider_runs_cfg(SimpleNamespace(cfg=1.0, conds=conds))
    assert not guider_runs_cfg(SimpleNamespace(cfg=4.0, conds={"positive": []}))
    assert not guider_runs_cfg(SimpleNamespace(conds={"positive": []}))


def _h3_kwargs(executor, cache_options):
    packed = SimpleNamespace(
        guidance_scale=1.0, height=8, width=8, cache_options=cache_options
    )
    return executor._sampling_params_kwargs(packed, torch.tensor([0.5]))


@pytest.mark.parametrize("cls", [MiniMaxH3Executor, FluxExecutor])
def test_per_run_cache_dit_overrides_loader_flag(cls) -> None:
    loader_on = _bare_executor(cls, enable_cache_dit=True)
    run_off = build_cache_options(False, None)
    assert _h3_kwargs(loader_on, run_off)["enable_cache_dit"] is False
    # Unset per-run value falls back to the loader option (backward compatible).
    assert _h3_kwargs(loader_on, build_cache_options(None, None))["enable_cache_dit"]
    assert "enable_cache_dit" not in _h3_kwargs(_bare_executor(cls), {})
    params = {"max_warmup_steps": 3}
    kwargs = _h3_kwargs(loader_on, build_cache_options(True, params))
    assert kwargs["cache_dit_params"] == params
    # left to the server default, the node's knobs still go along
    kwargs = _h3_kwargs(_bare_executor(cls), build_cache_options(None, params))
    assert "enable_cache_dit" not in kwargs
    assert kwargs["cache_dit_params"] == params
    assert "cache_dit_params" not in _h3_kwargs(
        loader_on, {"enable_cache_dit": False, "cache_dit_params": params}
    )


def test_cache_options_not_sent_in_h3_worker_transformer_options() -> None:
    opts = {CACHE_OPTIONS_KEY: build_cache_options(True, None), "sample_sigmas": [1.0]}
    assert worker_transformer_options(opts) == {"sample_sigmas": [1.0]}


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
