# SPDX-License-Identifier: Apache-2.0
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from diffusers.image_processor import VaeImageProcessor
from PIL import Image
from transformers import BatchFeature

from sglang.multimodal_gen.configs.models.dits.qwenimage21 import (
    QwenImage21ArchConfig,
    QwenImage21DitConfig,
)
from sglang.multimodal_gen.configs.models.vaes.qwenimage21 import (
    QwenImage21VAEArchConfig,
    QwenImage21VAEConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.qwen_image21 import (
    QwenImage21PipelineConfig,
)
from sglang.multimodal_gen.configs.sample.qwenimage21 import QwenImage21SamplingParams
from sglang.multimodal_gen.registry import _get_config_info
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ResidencyState,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_residency_strategies import (
    ComponentOffloadStrategy,
)
from sglang.multimodal_gen.runtime.managers.scheduler import Scheduler
from sglang.multimodal_gen.runtime.models.dits.qwen_image21 import build_layout
from sglang.multimodal_gen.runtime.models.encoders.qwen3vl_vision import (
    Qwen3VLVisionRotaryEmbedding,
)
from sglang.multimodal_gen.runtime.models.vaes.autoencoder_kl_qwenimage21 import (
    AutoencoderKLQwenImage21,
    QwenImage21RMS_norm,
    QwenImage21Upsample,
    _patchify,
    _unpatchify,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.input_validation import (
    InputValidationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.qwen_image21 import (
    QwenImage21EncodingStage,
    QwenImage21InputValidationStage,
    collapse_image_slots,
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["contiguous", "channels_last", "transposed"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_nearest_upsample_preserves_every_finite_low_precision_value(
    dtype, layout, device
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    values = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(dtype)
    values = values[torch.isfinite(values)].reshape(1, 2, -1, 128).to(device)
    if layout == "channels_last":
        values = values.contiguous(memory_format=torch.channels_last)
    elif layout == "transposed":
        values = values.transpose(2, 3)
    upsample = QwenImage21Upsample(scale_factor=2, mode="nearest-exact")
    expected = torch.nn.functional.interpolate(
        values.float(), scale_factor=2, mode="nearest-exact"
    ).to(dtype)
    actual = upsample(values)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("prompt", ["edit", ""])
@pytest.mark.parametrize("image_count", [0, 1, 2])
def test_prompt_conditioning_uses_training_template_and_pre_norm(prompt, image_count):
    hidden = torch.arange(24).reshape(1, 6, 4).float()
    inputs = BatchFeature(
        data={
            "input_ids": torch.tensor([[1, 2, 99, 99, 3, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 0]]),
        }
    )
    processor = Mock(return_value=inputs)
    processor.tokenizer.convert_tokens_to_ids.return_value = 99
    processor.apply_chat_template.return_value = [[1]]
    encoder = Mock(return_value=SimpleNamespace(hidden_states=(hidden,)))
    stage = QwenImage21EncodingStage(encoder, processor, None, None)
    stage.use_declared_component = Mock(return_value=nullcontext(encoder))
    images = [Image.new("RGBA", (2, 1), (12, 34, 56, 0)) for _ in range(image_count)]
    for image in images:
        image.putpixel((1, 0), (12, 34, 56, 255))
    actual, slots = stage.encode_prompt(prompt, images, "cpu")
    torch.testing.assert_close(actual, hidden[0, [1, 2, 4]])
    assert slots.tolist() == [False, True, False]
    encoder.model.language_model.norm.assert_not_called()
    assert encoder.model.visual.fp32_position_interpolation is False
    assert encoder.model.visual.rotary_pos_emb.recompute_on_device_change is True
    kwargs = processor.call_args.kwargs
    prefix = " ".join(
        f"<image{i + 1}><|vision_start|><|image_pad|><|vision_end|>"
        for i in range(image_count)
    )
    assert kwargs["text"] == [
        "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
        f"<|im_start|>user\n{prefix}{prompt or ' '}<|im_end|>\n<|im_start|>assistant\n"
    ]
    assert kwargs["padding_side"] == "left"
    for image in kwargs.get("images", []):
        assert image.mode == "RGB"
        assert image.getpixel((0, 0)) == (255, 255, 255)
        assert image.getpixel((1, 0)) == (12, 34, 56)
    for image in images:
        assert image.mode == "RGBA"
        assert image.getpixel((0, 0)) == (12, 34, 56, 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_encoder_component_offload_preserves_loaded_dtypes(monkeypatch):
    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.managers.memory_managers."
        "component_residency_strategies.get_local_torch_device",
        lambda: torch.device("cuda", torch.cuda.current_device()),
    )
    encoder = torch.nn.Module()
    encoder.model = torch.nn.Module()
    encoder.model.visual = torch.nn.Module()
    encoder.model.visual.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(36)
    with torch.device("cuda"):
        expected_rope = Qwen3VLVisionRotaryEmbedding(36)(64)
    encoder.register_parameter(
        "embedding", torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
    )
    encoder.register_parameter(
        "weight",
        torch.nn.Parameter(
            torch.tensor([0.25, -0.5]).to(torch.float8_e4m3fn), requires_grad=False
        ),
    )
    frequencies = torch.tensor([1.0 / 3, 1.0 / 7])
    encoder.register_buffer("inv_freq", frequencies.clone())
    processor = Mock()
    processor.apply_chat_template.return_value = [[1]]
    stage = QwenImage21EncodingStage(encoder, processor, None, None)
    use = stage.component_uses(None, "conditioning")[0]
    strategy = ComponentOffloadStrategy()
    state = ResidencyState(batch_is_warmup=False)
    weight_bytes = encoder.weight.view(torch.uint8).clone()

    for _ in range(2):
        strategy.prefetch_for_use(encoder, use, state)
        strategy.wait_for_use(encoder, use, state)
        assert encoder.embedding.device.type == "cuda"
        assert encoder.embedding.dtype == torch.bfloat16
        assert encoder.weight.dtype == torch.float8_e4m3fn
        assert encoder.inv_freq.dtype == torch.float32
        torch.testing.assert_close(encoder.inv_freq.cpu(), frequencies, atol=0, rtol=0)
        torch.testing.assert_close(
            encoder.model.visual.rotary_pos_emb(64), expected_rope, atol=0, rtol=0
        )
        assert torch.equal(encoder.weight.view(torch.uint8).cpu(), weight_bytes)
        strategy.finish_use(encoder, use, state)
        torch.cuda.synchronize()
        assert encoder.embedding.device.type == "cpu"


def test_condition_slots_expand_to_actual_latent_grid():
    hidden = torch.randn(22, 8)
    ids = torch.tensor([1, 2] + [99] * 16 + [3, 4, 5, 6])
    collapsed, slots = collapse_image_slots(hidden, ids, 99)
    assert collapsed.shape == (7, 8)
    layout = build_layout(slots.tolist(), [(1, 4, 8), (1, 2, 2)], (4, 6, 6), "cpu")
    assert len(layout["image_indices"]) == 32
    assert len(layout["prefix_rope"]) == 38
    assert layout["segments"] == ((0, 2, False), (2, 34, True), (34, 38, False))
    torch.testing.assert_close(collapsed[slots][0], hidden[2])


class _RecordingBlock(torch.nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self._layer_id = layer_id
        self.seen = None

    def forward(self, hidden_states, *args):
        caches = [cache[self._layer_id] for cache in args[-1]]
        self.seen = caches[0]
        return hidden_states


class _UnifiedBlocks(torch.nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList(blocks)

    def forward(self, hidden_states, *args):
        x = hidden_states
        for block in self.transformer_blocks:
            x = block(x, *args)
        return x


def _run_blocks(blocks, prefix_caches):
    x = torch.zeros(1)
    for block in blocks:
        x = block(x, prefix_caches)
    return x


def test_cache_dit_wrapper_keeps_per_layer_prefix_kv():
    inner = [_RecordingBlock(0), _RecordingBlock(1), _RecordingBlock(2)]
    wrapped = torch.nn.ModuleList([_UnifiedBlocks(inner)])
    prefix_caches = [[{"layer": 0}, {"layer": 1}, {"layer": 2}]]

    _run_blocks(wrapped, prefix_caches)
    assert [block.seen for block in inner] == prefix_caches[0]


def test_plain_blocks_still_get_per_layer_prefix_kv():
    blocks = torch.nn.ModuleList([_RecordingBlock(0), _RecordingBlock(1)])
    prefix_caches = [[{"layer": 0}, {"layer": 1}]]

    _run_blocks(blocks, prefix_caches)
    assert [block.seen for block in blocks] == prefix_caches[0]


class _FirstSlotBlock(torch.nn.Module):
    """Old loop body: take caches[0] and broadcast it to every layer."""

    def __init__(self):
        super().__init__()
        self.seen = None

    def forward(self, hidden_states, *args):
        self.seen = args[-1][0]
        return hidden_states


def test_first_slot_only_caches_are_shared_across_layers():
    inner = [_FirstSlotBlock(), _FirstSlotBlock()]
    unified = _UnifiedBlocks(inner)
    prefix_caches = [[{"layer": 0}, {"layer": 1}]]

    unified(torch.zeros(1), [cache[0] for cache in prefix_caches])
    assert [block.seen for block in inner] == [prefix_caches[0][0], prefix_caches[0][0]]


def test_adjacent_image_slots_stay_distinct():
    layout = build_layout(
        [False, True, True, False], [(1, 2, 2), (1, 4, 2), (1, 2, 2)], (4, 6, 6), "cpu"
    )
    assert layout["segments"] == (
        (0, 1, False),
        (1, 5, True),
        (5, 13, True),
        (13, 14, False),
    )
    with pytest.raises(ValueError, match="slots"):
        build_layout([False], [(1, 2, 2), (1, 2, 2)], (4, 6, 6), "cpu")


def test_latent_pack_decode_contract():
    config = QwenImage21PipelineConfig()
    batch = SimpleNamespace(
        height=64, width=96, extra={"qwen21_positive": {}, "qwen21_negative": {}}
    )
    shape = config.prepare_latent_shape(batch, 2, 1)
    x = torch.arange(torch.tensor(shape).prod()).reshape(shape)
    packed = config.maybe_pack_latents(x, 2, batch)
    assert packed.shape == (2, 24, 64)
    decoded = config.post_denoising_loop(packed, batch)
    assert not batch.extra
    torch.testing.assert_close(decoded[:, :, 0], x[:, 0])
    scale, shift = config.get_decode_scale_and_shift("cpu", torch.float32, None)
    torch.testing.assert_close(
        (decoded.float() - shift) * scale / scale + shift, decoded.float()
    )


@pytest.mark.parametrize("outputs", [1, 2])
def test_dynamic_batching_preserves_output_order_and_seeds(outputs):
    scheduler = object.__new__(Scheduler)
    config = QwenImage21PipelineConfig()
    scheduler.server_args = SimpleNamespace(pipeline_config=config)
    scheduler._batch_admission = SimpleNamespace(enabled=True)
    assert scheduler._dynamic_batching_enabled()
    requests = []
    for i, prompt in enumerate(["short", "a longer prompt"]):
        params = QwenImage21SamplingParams(
            prompt=prompt, seed=7 + i * 10, num_outputs_per_prompt=outputs
        )
        requests.append(Req(request_id=f"request-{i}", sampling_params=params))
    merged = scheduler._try_merge_generation_reqs(requests)
    assert merged.prompt == ["short", "a longer prompt"]
    assert merged.extra["dynamic_batch_seeds"] == [7, 17]
    result = torch.arange(outputs * 2).reshape(-1, 1)
    split = scheduler._split_batched_output(OutputBatch(output=result), requests)
    assert len(split) == 2
    torch.testing.assert_close(split[0].output, result[:outputs])
    torch.testing.assert_close(split[1].output, result[outputs:])
    requests[1].image_path = "reference.png"
    assert scheduler._try_merge_generation_reqs(requests) is None


@pytest.mark.parametrize("channels", [3, 4])
@pytest.mark.parametrize("tiling", [False, True])
def test_native_vae_roundtrip_shapes_and_checkpoint_names(channels, tiling):
    ac = QwenImage21VAEArchConfig(
        base_dim=4,
        decoder_base_dim=4,
        z_dim=4,
        dim_mult=(1, 2, 4, 4, 4),
        num_res_blocks=1,
        temperal_downsample=(False, False, False, False),
        in_channels=channels,
        out_channels=channels,
    )
    model = AutoencoderKLQwenImage21(QwenImage21VAEConfig(arch_config=ac)).eval()
    assert not model.use_tiling
    assert ac.scale_factor_spatial == ac.spatial_compression_ratio == 16
    model.use_tiling = tiling
    model.use_parallel_tiling = False
    model.tile_sample_min_height = model.tile_sample_min_width = 32
    model.tile_sample_stride_height = model.tile_sample_stride_width = 16
    with torch.no_grad():
        latent = model.encode(torch.randn(1, channels, 1, 32, 64)).mode()
        assert latent.shape == (1, 4, 1, 2, 4)
        output = model.decode(latent)
        assert output.shape == (1, channels, 1, 32, 64)
    assert model.state_dict()["encoder.conv_in.weight"].ndim == 4
    x = torch.randn(2, 3, 1, 8, 12)
    torch.testing.assert_close(_unpatchify(_patchify(x, 2), 2), x)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_vae_rms_norm_normalizes_in_float32(dtype):
    norm = QwenImage21RMS_norm(8, images=False).to(dtype)
    x = torch.linspace(-60000, 60000, 256).reshape(1, 8, 1, 4, 8).to(dtype)
    expected = (
        torch.nn.functional.normalize(x.float(), dim=1).to(dtype)
        * norm.scale
        * norm.gamma
    )
    torch.testing.assert_close(norm(x), expected, atol=0, rtol=0)


@pytest.mark.parametrize("tiling", [False, True])
def test_condition_pixels_match_reference_preprocessing(monkeypatch, tiling):
    module = "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.qwen_image21"
    monkeypatch.setattr(f"{module}.get_local_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(f"{module}.set_forward_context", lambda **kwargs: nullcontext())
    image = Image.frombytes("RGBA", (32, 32), bytes(range(256)) * 16)
    vae = Mock()
    vae.encode.return_value.mode.return_value = torch.zeros(1, 64, 1, 2, 2)
    processor = Mock()
    processor.apply_chat_template.return_value = [[1]]
    stage = QwenImage21EncodingStage(Mock(), processor, vae, SimpleNamespace(config={}))
    stage.use_declared_component = Mock(return_value=nullcontext(vae))
    stage.encode_prompt = Mock(
        return_value=(torch.zeros(3, 8), torch.tensor([False, True, False]))
    )
    batch = SimpleNamespace(
        height=32,
        width=32,
        condition_image=image,
        prompt="edit",
        negative_prompt=None,
        num_outputs_per_prompt=1,
        do_classifier_free_guidance=False,
        extra={},
    )
    stage.forward(
        batch,
        SimpleNamespace(pipeline_config=QwenImage21PipelineConfig(vae_tiling=tiling)),
    )
    assert vae.use_tiling is tiling
    expected = VaeImageProcessor(vae_scale_factor=16).preprocess(image).unsqueeze(2)
    actual = vae.encode.call_args.args[0]
    torch.testing.assert_close(actual, expected.bfloat16(), atol=0, rtol=0)
    assert actual.stride() == expected.stride()


def test_condition_image_loading_preserves_alpha(tmp_path):
    path = tmp_path / "condition.png"
    Image.new("RGBA", (32, 32), (12, 34, 56, 78)).save(path)
    image = QwenImage21InputValidationStage().load_condition_image(str(path))
    assert image.mode == "RGBA"
    assert image.getpixel((0, 0)) == (12, 34, 56, 78)
    assert InputValidationStage().load_condition_image(str(path)).mode == "RGB"


def test_architecture_derived_dimensions():
    config = QwenImage21DitConfig(
        arch_config=QwenImage21ArchConfig(num_attention_heads=2, attention_head_dim=16)
    )
    assert config.hidden_size == 32


def test_registry_routes_local_checkpoint_and_preserves_legacy():
    assert (
        _get_config_info("Qwen/Qwen-Image-2.1").pipeline_config_cls
        is QwenImage21PipelineConfig
    )
    assert (
        _get_config_info(
            "/models/private", model_id="Qwen-Image-2.1"
        ).pipeline_config_cls
        is QwenImage21PipelineConfig
    )
    assert (
        _get_config_info("Qwen/Qwen-Image").pipeline_config_cls
        is not QwenImage21PipelineConfig
    )
