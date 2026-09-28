from contextlib import nullcontext
from itertools import pairwise
from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.configs.pipeline_configs.sana_video2 import (
    SanaVideo2PipelineConfig,
)
from sglang.multimodal_gen.configs.sample.sana_video2 import SanaVideo2SamplingParams
from sglang.multimodal_gen.registry import get_model_info
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages import (
    sana_video2,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_video2 import (
    sample_flow_dpm,
    sample_ltx_euler,
)


def test_native_registry_precedes_sana_video(monkeypatch):
    monkeypatch.setattr(
        "sglang.multimodal_gen.registry.maybe_download_model_index",
        lambda _: None,
    )
    get_model_info.cache_clear()
    info = get_model_info("Efficient-Large-Model/SANA-Video_2.0_5B_720p")
    assert info.pipeline_config_cls is SanaVideo2PipelineConfig
    assert info.sampling_param_cls is SanaVideo2SamplingParams
    get_model_info.cache_clear()


def test_latent_layout_and_frame_alignment():
    config = SanaVideo2PipelineConfig()
    assert config.adjust_num_frames(193) == 193
    assert config.adjust_num_frames(192) == 185
    assert config.prepare_latent_shape(
        SimpleNamespace(height=736, width=1280), 2, 25
    ) == (2, 128, 25, 23, 40)
    assert config.get_latent_dtype(torch.bfloat16) == torch.float32
    defaults = SanaVideo2SamplingParams()
    assert (defaults.height, defaults.width, defaults.num_frames, defaults.fps) == (
        736,
        1280,
        193,
        24,
    )
    assert defaults.num_inference_steps == 50


def test_bcg_requires_resident_transformer_only_when_enabled():
    config = SanaVideo2PipelineConfig()
    for enabled in (False, True):
        calls = []
        args = SimpleNamespace(
            enable_breakable_cuda_graph=enabled,
            require_component_resident=lambda *args, **kwargs: calls.append(
                (args, kwargs)
            ),
        )
        config.validate_server_args(args)
        assert calls == (
            [
                (
                    ("transformer",),
                    {"feature_name": "SANA-Video 2.0 breakable CUDA graphs"},
                )
            ]
            if enabled
            else []
        )


def test_dpm_constant_data_prediction_reaches_clean_endpoint():
    initial = torch.tensor([[[[[0.4, -0.7]]]]], dtype=torch.float32)
    clean = torch.tensor([[[[[0.2, 0.9]]]]], dtype=torch.float32)
    times = []

    def predict_noise(x, time):
        times.append(time.item())
        return (x - (1 - time) * clean) / time

    actual = sample_flow_dpm(predict_noise, initial, steps=5, shift=12.0)
    torch.testing.assert_close(actual, clean, atol=1e-5, rtol=1e-5)
    assert len(times) == 5
    assert all(a > b for a, b in pairwise(times))
    assert actual.dtype == torch.float32


def test_ti2v_euler_preserves_condition_and_uses_frame_timestep():
    initial = torch.zeros(1, 2, 3, 1, 1)
    initial[:, :, 0] = 7
    seen = []

    def predict_flow(x, time):
        seen.append(time.clone())
        return torch.ones_like(x)

    actual = sample_ltx_euler(predict_flow, initial.clone(), steps=4, shift=12.0)
    torch.testing.assert_close(actual[:, :, 0], initial[:, :, 0], atol=0, rtol=0)
    torch.testing.assert_close(actual[:, :, 1:], -torch.ones_like(actual[:, :, 1:]))
    assert len(seen) == 4
    assert all(time.shape == (1, 1, 3, 1, 1) for time in seen)
    assert all(torch.count_nonzero(time[:, :, 0]) == 0 for time in seen)
    assert all(torch.all(time[:, :, 1:] > 0) for time in seen)


@pytest.mark.parametrize("conditioned", [False, True])
@pytest.mark.parametrize("flow_shift, expected", [(None, 12.0), (3.0, 3.0)])
def test_denoising_uses_request_flow_shift(
    monkeypatch, conditioned, flow_shift, expected
):
    shifts = []

    def sample(predict, latents, steps, shift, callback):
        shifts.append(shift)
        return latents

    monkeypatch.setattr(
        sana_video2, "sample_ltx_euler" if conditioned else "sample_flow_dpm", sample
    )
    stage = object.__new__(sana_video2.SanaVideo2DenoisingStage)
    stage.transformer = None
    stage.begin_declared_component_use = lambda **_: None
    stage.progress_bar = lambda **_: nullcontext()
    batch = SimpleNamespace(
        do_classifier_free_guidance=False,
        prompt_embeds=[torch.zeros(1, 300, 4)],
        prompt_attention_mask=[torch.ones(1, 300)],
        condition_image=object() if conditioned else None,
        latents=torch.zeros(1, 2, 3, 1, 1),
        num_inference_steps=2,
        is_warmup=False,
        flow_shift=flow_shift,
    )
    stage.forward(
        batch,
        SimpleNamespace(
            pipeline_config=SanaVideo2PipelineConfig(),
            enable_breakable_cuda_graph=False,
        ),
    )
    assert shifts == [expected]


def test_denoising_stage_role():
    stage = object.__new__(sana_video2.SanaVideo2DenoisingStage)
    assert stage.role_affinity is RoleType.DENOISER


def _denoising_stage(transformer):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_video2 import (
        SanaVideo2DenoisingStage,
    )

    stage = object.__new__(SanaVideo2DenoisingStage)
    stage.transformer = transformer
    stage._bcg_runner = None
    stage.begin_declared_component_use = lambda **_: None
    stage.progress_bar = lambda **_: nullcontext(SimpleNamespace(update=lambda: None))
    return stage


def _denoising_batch(*, condition_image=None, is_warmup=False, cfg=True):
    return SimpleNamespace(
        condition_image=condition_image,
        do_classifier_free_guidance=cfg,
        prompt_embeds=[torch.ones(1, 300, 4)],
        prompt_attention_mask=[torch.ones(1, 300, dtype=torch.long)],
        negative_prompt_embeds=[torch.zeros(1, 300, 4)],
        negative_attention_mask=[torch.ones(1, 300, dtype=torch.long)],
        latents=torch.ones(1, 2, 3, 1, 1),
        guidance_scale=8.0,
        num_inference_steps=2,
        is_warmup=is_warmup,
        flow_shift=None,
    )


def test_denoising_bcg_captures_t2v_and_ti2v_at_warmup_only(monkeypatch):
    from sglang.multimodal_gen.runtime.breakable_cuda_graph import runner as bcg_module

    instances = []

    class Transformer(torch.nn.Module):
        def forward(self, **kwargs):
            return torch.zeros_like(kwargs["hidden_states"])

    class Runner:
        def __init__(self, transformer, device):
            self.transformer = transformer
            self.captures = []
            self.calls = []
            instances.append(self)

        def capture(self, **kwargs):
            self.captures.append(kwargs)
            return True

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            return self.transformer(**kwargs)

    monkeypatch.setattr(bcg_module, "DiffusionBreakableCudaGraphRunner", Runner)
    stage = _denoising_stage(Transformer())
    args = SimpleNamespace(
        enable_breakable_cuda_graph=True,
        pipeline_config=SimpleNamespace(flow_shift=12.0),
    )

    stage.forward(_denoising_batch(is_warmup=True), args)
    assert len(instances) == 1
    runner = instances[0]
    assert {tuple(call["timestep"].shape) for call in runner.captures} == {
        (2,),
        (2, 1, 3, 1, 1),
    }
    assert all(
        call["encoder_hidden_states"].shape == (2, 300, 4) for call in runner.captures
    )
    assert all(call["timestep"].dtype == torch.float32 for call in runner.captures)
    framewise = next(
        call["timestep"] for call in runner.captures if call["timestep"].ndim == 5
    )
    assert torch.count_nonzero(framewise[:, :, 0]) == 0
    assert torch.all(framewise[:, :, 1:] > 0)
    captures = len(runner.captures)
    stage.forward(_denoising_batch(condition_image=object()), args)
    assert len(instances) == 1
    assert len(runner.captures) == captures
    assert any(call["timestep"].shape == (2, 1, 3, 1, 1) for call in runner.calls)


def test_denoising_without_bcg_calls_transformer_directly():
    class Transformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, **kwargs):
            self.calls += 1
            return torch.zeros_like(kwargs["hidden_states"])

    stage = _denoising_stage(Transformer())
    stage.forward(
        _denoising_batch(cfg=False),
        SimpleNamespace(
            enable_breakable_cuda_graph=False,
            pipeline_config=SimpleNamespace(flow_shift=12.0),
        ),
    )
    assert stage.transformer.calls == 2


def test_vae_statistics_are_inverted_for_decode():
    config = SanaVideo2PipelineConfig()
    config.vae_config.arch_config.scaling_factor = 2.0
    vae = SimpleNamespace(
        latents_mean=torch.tensor([1.0, -1.0]), latents_std=torch.tensor([2.0, 4.0])
    )
    scale, shift = config.get_decode_scale_and_shift(
        torch.device("cpu"), torch.float32, vae
    )
    normalized = torch.tensor([2.0, 3.0]).reshape(1, 2, 1, 1, 1)
    decoded = normalized / scale + shift
    torch.testing.assert_close(decoded.flatten(), torch.tensor([3.0, 5.0]))


def test_prompt_window_preserves_case_and_only_positive_motion_suffix():
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_video2 import (
        SanaVideo2TextEncodingStage,
    )

    stage = object.__new__(SanaVideo2TextEncodingStage)
    stage.instruction = "instruction: "
    stage.tokenizers = [SimpleNamespace(encode=lambda _: [1, 2, 3, 4, 5])]
    calls = []

    def encode(text, server_args, **kwargs):
        calls.append((text, kwargs["max_length"]))
        length = kwargs["max_length"]
        values = torch.arange(length).reshape(1, length, 1)
        mask = torch.ones(1, length, dtype=torch.long)
        return [values], [mask], [], [mask.bool()], [[length]]

    stage.encode_text = encode
    batch = SimpleNamespace(
        prompt="  A RED fox  ",
        negative_prompt="Bad motion",
        max_sequence_length=4,
        extra={"motion_score": 10},
        do_classifier_free_guidance=True,
        prompt_embeds=[],
        pooled_embeds=[],
        prompt_attention_mask=None,
        negative_prompt_embeds=[],
        negative_pooled_embeds=[],
        negative_attention_mask=None,
    )
    stage.forward(batch, SimpleNamespace())
    assert calls == [
        (["instruction: A RED fox motion score: 10."], 7),
        ("Bad motion", 4),
    ]
    assert batch.prompt_embeds[0].flatten().tolist() == [0, 4, 5, 6]
    assert batch.negative_prompt_embeds[0].flatten().tolist() == [0, 1, 2, 3]
    assert batch.prompt_attention_mask[0].shape == (1, 4)


def test_native_model_index_and_component_directory_contract(tmp_path):
    import yaml

    from sglang.multimodal_gen.runtime.pipelines.sana_video2 import SanaVideo2Pipeline

    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"text_encoder": {"chi_prompt": ["first", "last"]}})
    )
    vae_root = tmp_path / "ltx"
    (vae_root / "vae").mkdir(parents=True)
    (vae_root / "vae/config.json").write_text("{}")
    pipeline = object.__new__(SanaVideo2Pipeline)
    pipeline.model_path = str(tmp_path)
    pipeline.server_args = SimpleNamespace(
        revision=None,
        pipeline_config=SanaVideo2PipelineConfig(),
        component_paths={"vae": str(vae_root), "text_encoder": str(tmp_path)},
    )
    index = pipeline._load_config()
    assert index["_class_name"] == "SanaVideo2Pipeline"
    assert "_diffusers_version" in index
    assert pipeline.server_args.pipeline_config.prompt_instruction == "first\nlast"
    assert pipeline._resolve_component_path(pipeline.server_args, "vae", "vae") == str(
        vae_root / "vae"
    )
    assert pipeline._resolve_component_path(
        pipeline.server_args, "tokenizer", "tokenizer"
    ) == str(tmp_path)


def test_released_pth_layout_loads_strictly_and_materializes_rope(tmp_path):
    import yaml

    from sglang.multimodal_gen.configs.models.dits.sana_video2 import SanaVideo2Config
    from sglang.multimodal_gen.runtime.models.dits.sana_video2 import (
        SanaVideo2Transformer3DModel,
    )
    from sglang.multimodal_gen.runtime.pipelines.sana_video2 import (
        SanaVideo2TransformerLoader,
    )

    arch = {
        "hidden_size": 32,
        "depth": 2,
        "num_heads": 4,
        "linear_head_dim": 8,
        "softmax_head_dim": 8,
        "softmax_ratio": 0.5,
        "in_channels": 4,
        "caption_channels": 12,
        "model_max_length": 5,
        "input_size": 2,
    }
    config = SanaVideo2Config()
    config.update_model_arch(arch)
    original = SanaVideo2Transformer3DModel(config)
    (tmp_path / "checkpoints").mkdir()
    torch.save(
        {"state_dict": original.state_dict()},
        tmp_path / "checkpoints/SANA_Video_2.0_5B_720p.pth",
    )
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"model": arch, "text_encoder": {}})
    )
    pipeline_config = SanaVideo2PipelineConfig()
    pipeline_config.dit_precision = "fp32"
    args = SimpleNamespace(
        pipeline_config=pipeline_config, component_precisions={}, model_paths={}
    )
    loaded = SanaVideo2TransformerLoader().load_customized(
        str(tmp_path), args, "transformer"
    )
    assert not any(value.is_meta for value in loaded.buffers())
    for name, value in original.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value, rtol=0, atol=0)


def test_local_native_root_does_not_require_diffusers_model_index(tmp_path):
    root = tmp_path / "sana-video2-5b"
    root.mkdir()
    (root / "config.yaml").write_text("model:\n  model: SanaVideo2_5B\n")
    get_model_info.cache_clear()
    info = get_model_info(str(root))
    assert info.pipeline_config_cls is SanaVideo2PipelineConfig
    assert info.sampling_param_cls is SanaVideo2SamplingParams
    get_model_info.cache_clear()


def test_diffusers_local_root_still_reads_model_index(tmp_path):
    from sglang.multimodal_gen.configs.pipeline_configs.sana_video import (
        SanaVideoPipelineConfig,
    )

    root = tmp_path / "sana-video-original"
    root.mkdir()
    (root / "transformer").mkdir()
    (root / "model_index.json").write_text(
        '{"_class_name": "SanaVideoPipeline", "_diffusers_version": "0.37.0", '
        '"transformer": ["diffusers", "SanaVideoTransformer3DModel"]}'
    )
    get_model_info.cache_clear()
    info = get_model_info(str(root))
    assert info.pipeline_config_cls is SanaVideoPipelineConfig
    get_model_info.cache_clear()


def test_two_outputs_expand_with_independent_generators_before_latent_preparation():
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
    from sglang.multimodal_gen.runtime.pipelines_core.stages.input_validation import (
        InputValidationStage,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_video2 import (
        SanaVideo2LatentPreparationStage,
    )

    sampling = SanaVideo2SamplingParams(
        request_id="two-videos",
        prompt="A fox",
        height=32,
        width=32,
        num_frames=9,
        num_outputs_per_prompt=2,
        seed=[11, 12],
        generator_device="cpu",
        output_path="/tmp",
        output_file_name="video.mp4",
    )
    request = Req(sampling_params=sampling)
    args = SimpleNamespace(pipeline_config=SanaVideo2PipelineConfig())
    requests = list(InputValidationStage().iter_sequential_requests(request, args))
    assert len(requests) == 2
    assert [item.num_outputs_per_prompt for item in requests] == [1, 1]
    assert [item.seed for item in requests] == [11, 12]
    stage = SanaVideo2LatentPreparationStage(vae=None)
    latents = []
    for item in requests:
        item.prompt_embeds = [torch.zeros(1, 5, 12)]
        stage.forward(item, args)
        assert item.latents.shape == (1, 128, 2, 1, 1)
        expected = torch.randn(
            item.latents.shape, generator=torch.Generator().manual_seed(item.seed)
        )
        torch.testing.assert_close(item.latents, expected)
        latents.append(item.latents)
    assert not torch.equal(latents[0], latents[1])
