"""CPU regression tests for LFM3 audio request formatting and feature geometry."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from sglang.srt.configs.lfm3_audio import Lfm3AudioConfig
from sglang.srt.configs.linear_attn_model_registry import get_linear_attn_spec
from sglang.srt.models.lfm3_audio import Lfm3AudioDownsampler
from sglang.srt.multimodal.lfm3_audio import (
    AUDIO_TOKEN,
    Lfm3AudioProcessor,
)
from sglang.srt.multimodal.processors.lfm3_audio import Lfm3AudioMultimodalProcessor
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


@pytest.fixture
def tokenizer():
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({"[UNK]": 0, AUDIO_TOKEN: 1}, unk_token="[UNK]")
        ),
        unk_token="[UNK]",
        additional_special_tokens=[AUDIO_TOKEN],
    )
    tokenizer.chat_template = (
        "{% for m in messages %}{{ m.role }}:"
        "{% if m.content is string %}{{ m.content }}{% else %}"
        "{% for p in m.content %}"
        "{% if p.type in ['audio', 'audio_url', 'input_audio'] %}<|audio|>"
        "{% else %}{{ p.text }}{% endif %}{% endfor %}{% endif %}"
        "{% if m.tool_calls is defined %}{{ m.tool_calls | tojson }}{% endif %}"
        "{% endfor %}{% if add_generation_prompt %}assistant:{% endif %}"
    )
    return tokenizer


@pytest.mark.parametrize("cache_mb", [0, 256])
def test_processor_initializes_with_preprocess_cache(
    tokenizer, cache_mb, lfm3_audio_reference
):
    from sglang.srt.runtime_context import publish, reset_context
    from sglang.srt.server_args import ServerArgs

    args = ServerArgs(
        model_path="dummy", mm_preprocess_cache_size_mb=cache_mb, mm_process_config={}
    )
    publish(args, role="tokenizer")
    processor = None
    try:
        processor = Lfm3AudioMultimodalProcessor(
            Lfm3AudioConfig.from_hf_config(lfm3_audio_reference[0]),
            args,
            Lfm3AudioProcessor(tokenizer, deepcopy(lfm3_audio_reference[1])),
            transport_mode="default",
            skip_mm_pool=True,
        )
        assert processor.mm_preprocess_cache.enabled == bool(cache_mb)
        assert bool(processor.processor_fingerprint) == bool(cache_mb)
    finally:
        if processor is not None:
            processor.shutdown()
        reset_context()


def test_checkpoint_audio_template_is_unchanged(tokenizer, lfm3_audio_reference):
    original_template = tokenizer.chat_template
    text = [{"role": "user", "content": "hello"}]
    original = tokenizer.apply_chat_template(text, tokenize=False)
    processor = Lfm3AudioProcessor(tokenizer, deepcopy(lfm3_audio_reference[1]))
    assert tokenizer.chat_template == original_template
    assert processor.chat_template == original_template
    assert tokenizer.apply_chat_template(text, tokenize=False) == original
    messages = [
        {
            "role": "user",
            "tool_calls": [{"id": "call_1"}],
            "content": [
                {"type": "text", "text": "before"},
                {"type": "audio"},
                {"type": "text", "text": "between"},
                {"type": "input_audio", "input_audio": {"data": "ignored"}},
            ],
        }
    ]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    assert f"before{AUDIO_TOKEN}between{AUDIO_TOKEN}" in rendered
    assert '"call_1"' in rendered
    assert rendered.endswith("assistant:")
    assert messages[0]["content"][1] == {"type": "audio"}
    # Neither re-wrapping nor tokenizer replacement may rewrite the template.
    Lfm3AudioProcessor(processor.tokenizer, processor.feature_extractor)
    assert (
        tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        == rendered
    )
    replacement = deepcopy(tokenizer)
    replacement.chat_template = original_template
    processor.tokenizer = replacement
    assert replacement.chat_template == original_template
    assert processor.chat_template == original_template
    assert (
        replacement.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        == rendered
    )


@pytest.mark.parametrize(
    "samples,frames", [(1, 1), (400, 1), (401, 2), (1520, 8), (1521, 9), (16000, 99)]
)
def test_mel_padding_and_encoder_geometry(samples, frames, lfm3_audio_reference):
    mel = deepcopy(lfm3_audio_reference[1])(
        torch.zeros(samples), sampling_rate=16000, return_tensors="pt"
    ).input_features[0]
    assert mel.shape == (128, frames)
    assert torch.isfinite(mel).all()
    # Silence has the defined log floor after affine normalization.
    assert torch.equal(mel, torch.full_like(mel, -2.0))
    downsampler = Lfm3AudioDownsampler(128, 4, 8)
    with torch.no_grad():
        output = downsampler(mel.unsqueeze(0))
    assert output.shape == (1, 8, (frames + 7) // 8)


@pytest.mark.parametrize(
    "waveform", [torch.empty(0), torch.tensor([float("nan")]), torch.zeros(2, 400)]
)
def test_invalid_waveform(waveform, lfm3_audio_reference):
    processor = object.__new__(Lfm3AudioMultimodalProcessor)
    processor._tokenizer = SimpleNamespace(encode=lambda *args, **kwargs: [])
    processor.feature_extractor = deepcopy(lfm3_audio_reference[1])
    with pytest.raises(ValueError, match="finite, nonempty mono"):
        Lfm3AudioMultimodalProcessor._expand_audio(processor, AUDIO_TOKEN, [waveform])


def test_multiple_audio_offsets_and_mismatch(tokenizer, lfm3_audio_reference):
    processor = object.__new__(Lfm3AudioMultimodalProcessor)
    processor._tokenizer = tokenizer
    processor.audio_token_id = 1
    processor.feature_extractor = deepcopy(lfm3_audio_reference[1])
    result = Lfm3AudioMultimodalProcessor._expand_audio(
        processor,
        AUDIO_TOKEN + "between" + AUDIO_TOKEN,
        [torch.zeros(400), torch.ones(1521)],
    )
    assert [item.offsets for item in result.mm_items] == [[(0, 0)], [(2, 3)]]
    assert result.input_ids == [1, 0, 1, 1]
    assert [item.feature.shape[-1] for item in result.mm_items] == [1, 9]
    with pytest.raises(ValueError, match="placeholder count"):
        Lfm3AudioMultimodalProcessor._expand_audio(processor, AUDIO_TOKEN, [])


@pytest.mark.parametrize(
    "frame_lengths", [(25, 25, 25), (1, 7, 8, 9, 15, 16, 17, 65, 9)]
)
def test_audio_batches_variable_lengths_and_restores_item_order(frame_lengths):
    from transformers import Lfm2Config

    from sglang.srt.models.lfm3_audio import (
        Lfm3AudioEncoder,
        Lfm3AudioForConditionalGeneration,
        Lfm3AudioProjector,
    )

    torch.manual_seed(0)
    config = SimpleNamespace(
        num_mel_bins=8,
        conv_channels=4,
        hidden_size=16,
        sliding_window=4,
        text_config=Lfm2Config(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            vocab_size=2,
            layer_types=["conv", "full_attention"],
        ),
    )
    model = SimpleNamespace(
        encoder=Lfm3AudioEncoder(config).eval(),
        mlp=Lfm3AudioProjector(16, 16).eval(),
    )
    # Interleaved lengths and distinct values detect accidental reordering.
    items = [SimpleNamespace(feature=torch.randn(8, n)) for n in frame_lengths]
    with torch.no_grad():
        expected = torch.cat(
            [
                model.mlp(model.encoder(item.feature.unsqueeze(0))).squeeze(0)
                for item in items
            ]
        )
        batches = []
        hook = model.encoder.lfm.register_forward_pre_hook(
            lambda module, args: batches.append(tuple(args[0].shape))
        )
        try:
            actual = Lfm3AudioForConditionalGeneration.get_audio_feature(model, items)
        finally:
            hook.remove()
    assert batches == [(len(items), (max(frame_lengths) + 7) // 8, 16)]
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    items[0].feature = items[0].feature.unsqueeze(0)
    with torch.no_grad():
        actual = Lfm3AudioForConditionalGeneration.get_audio_feature(model, items)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_config_roundtrip_retains_hybrid_registration(lfm3_audio_reference):
    reference = type(lfm3_audio_reference[0])(
        architectures=["Lfm3AudioForConditionalGeneration"],
        lfm={
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "layer_types": ["conv", "full_attention"],
            "conv_L_cache": 7,
        },
    )
    config = Lfm3AudioConfig.from_hf_config(reference)
    restored = Lfm3AudioConfig.from_dict(config.to_dict())
    assert (
        restored.audio_encoder_config.hidden_size
        == config.audio_encoder_config.hidden_size
    )
    assert restored.architectures == ["Lfm3AudioForConditionalGeneration"]
    assert restored.get_text_config().hidden_size == 32
    assert restored.linear_layer_ids == [0]
    assert restored.full_attention_layer_ids == [1]
    assert get_linear_attn_spec(restored).arch_names == [
        "Lfm3AudioForConditionalGeneration"
    ]


@pytest.mark.parametrize(
    "language,system_prompt", [(None, "Perform ASR."), ("en", "Perform ASR in en.")]
)
def test_transcription_adapter_does_not_use_whisper_detection(
    tokenizer, language, system_prompt
):
    from sglang.srt.entrypoints.openai.transcription_adapters import resolve_adapter
    from sglang.srt.entrypoints.openai.transcription_adapters.lfm3_audio import (
        Lfm3AudioAdapter,
    )
    from sglang.srt.sampling.sampling_params import SamplingParams

    adapter = resolve_adapter(["Lfm3AudioForConditionalGeneration"])
    assert isinstance(adapter, Lfm3AudioAdapter)
    assert not adapter.supports_language_detection
    from sglang.srt.entrypoints.openai.protocol import TranscriptionRequest
    from sglang.srt.entrypoints.openai.serving_transcription import (
        OpenAIServingTranscription,
    )

    serving = SimpleNamespace(
        _adapter=adapter,
        tokenizer_manager=SimpleNamespace(tokenizer=tokenizer),
        extract_routing_key=lambda request: None,
    )
    request = TranscriptionRequest(audio_data=b"audio", language=language)
    internal, _ = OpenAIServingTranscription._convert_to_internal_request(
        serving, request
    )
    assert internal.text == f"system:{system_prompt}user:{AUDIO_TOKEN}assistant:"
    chunk = OpenAIServingTranscription._build_chunk_request(
        serving, internal, b"chunk", False
    )
    assert chunk.text == internal.text
    params = adapter.build_sampling_params(
        SimpleNamespace(audio_duration_s=2.0, temperature=0)
    )
    assert SamplingParams(**params).max_new_tokens >= 24


@pytest.mark.parametrize("granularities", [["word"], ["segment"], ["word", "segment"]])
def test_transcription_rejects_timestamps(tokenizer, granularities):
    from sglang.srt.entrypoints.openai.protocol import TranscriptionRequest
    from sglang.srt.entrypoints.openai.transcription_adapters.lfm3_audio import (
        Lfm3AudioAdapter,
    )

    request = TranscriptionRequest(
        audio_data=b"audio",
        response_format="verbose_json",
        timestamp_granularities=granularities,
    )
    with pytest.raises(ValueError, match="LFM3-Audio does not support timestamp"):
        Lfm3AudioAdapter().build_prompt(request, tokenizer)

    request.timestamp_granularities = []
    assert Lfm3AudioAdapter().build_prompt(request, tokenizer)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))


@pytest.mark.parametrize("layout", ["audio", "omni"])
def test_weight_loader_accepts_both_layouts_and_rejects_unknown_weights(layout):
    from sglang.srt.models.lfm3_audio import Lfm3AudioForConditionalGeneration

    names = [
        ("encoder.downsample.weight", "model.audio_encoder.downsample.weight"),
        ("encoder.lfm.lfm.weight", "model.audio_encoder.lfm.weight"),
        ("mlp.model.0.weight", "model.audio_projector.layer_norm.weight"),
        ("mlp.model.1.weight", "model.audio_projector.linear_1.weight"),
        ("mlp.model.3.weight", "model.audio_projector.linear_2.weight"),
        ("lfm.model.weight", "model.language_model.weight"),
        ("lfm.lm_head.weight", "lm_head.weight"),
        ("depthformer.weight", "model.depth_decoder.weight"),
        ("codebook_embedding.weight", "model.codebook_embedding.weight"),
    ]
    params = {name: torch.nn.Parameter(torch.zeros(1)) for name, _ in names[:5]}
    language = {}

    def load_language(weights):
        language.update(weights)
        return set(language)

    model = SimpleNamespace(
        named_parameters=lambda: iter(params.items()),
        language_model=SimpleNamespace(
            load_weights=load_language,
            named_parameters=lambda: iter(
                [("model.weight", None), ("lm_head.weight", None)]
            ),
        ),
    )
    weights = [
        (pair[layout == "omni"], torch.tensor([float(i + 1)]))
        for i, pair in enumerate(names)
    ]
    loaded = Lfm3AudioForConditionalGeneration.load_weights(model, iter(weights))
    assert loaded == set(params) | {
        "language_model.model.weight",
        "language_model.lm_head.weight",
    }
    for i, parameter in enumerate(params.values(), 1):
        assert parameter.item() == i
    assert language["model.weight"].item() == 6
    assert language["lm_head.weight"].item() == 7
    with pytest.raises(ValueError, match="Unexpected LFM3 audio weight"):
        Lfm3AudioForConditionalGeneration.load_weights(
            model, iter([("unknown.weight", torch.ones(1))])
        )
    with pytest.raises(ValueError, match="Missing LFM3 audio weights"):
        Lfm3AudioForConditionalGeneration.load_weights(model, iter(weights[1:]))
