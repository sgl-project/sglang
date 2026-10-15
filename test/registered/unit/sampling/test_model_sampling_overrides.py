"""Exercise model sampling aliases through the tokenizer request boundary."""

from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from transformers import GenerationConfig

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.configs.model_config import ModelConfig  # noqa: E402
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest  # noqa: E402
from sglang.srt.managers.io_struct import GenerateReqInput  # noqa: E402
from sglang.srt.managers.tokenizer_manager import TokenizerManager  # noqa: E402
from sglang.srt.sampling.sampling_params import TOP_K_ALL, SamplingParams  # noqa: E402

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

MUSE_ARCHITECTURES = ["MuseSparkForConditionalGeneration", "MuseSpark1xForCausalLM"]


def tokenize(architectures, sampling_params, preferred=None):
    manager = TokenizerManager.__new__(TokenizerManager)
    manager.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=architectures), vocab_size=202048
    )
    manager.preferred_sampling_params = preferred
    manager.sampling_params_class = SamplingParams
    manager.tokenizer = None
    manager.rid_to_state = {"test": SimpleNamespace(time_stats=Mock())}
    request = GenerateReqInput(
        rid="test", input_ids=[1], sampling_params=sampling_params, bootstrap_room=1
    )
    return manager._create_tokenized_object(request, "", [1]).sampling_params


@pytest.mark.parametrize("architecture", MUSE_ARCHITECTURES)
def test_generation_config_zero_disables_top_k(architecture):
    with TemporaryDirectory() as folder:
        GenerationConfig(
            do_sample=True, temperature=1.0, top_p=1.0, top_k=0
        ).save_pretrained(folder)
        model = ModelConfig.__new__(ModelConfig)
        model.sampling_defaults = "model"
        model.hf_generation_config = GenerationConfig.from_pretrained(folder)
        defaults = model.get_default_sampling_params()
        request = ChatCompletionRequest(
            model="muse", messages=[{"role": "user", "content": "hello"}]
        )
        params = request.to_sampling_params(stop=[], model_generation_config=defaults)
        result = tokenize([architecture], params)
        assert result.top_k == TOP_K_ALL
        assert result.temperature == result.top_p == 1.0
        assert defaults["top_k"] == params["top_k"] == 0
        assert model.hf_generation_config.top_k == 0


@pytest.mark.parametrize("architecture", MUSE_ARCHITECTURES)
@pytest.mark.parametrize("top_k, expected", [(0, TOP_K_ALL), (7, 7), (-1, TOP_K_ALL)])
def test_explicit_chat_sampling_wins_over_model_defaults(architecture, top_k, expected):
    request = ChatCompletionRequest(
        model="muse", messages=[{"role": "user", "content": "hello"}], top_k=top_k
    )
    params = request.to_sampling_params(stop=[], model_generation_config={"top_k": 64})
    result = tokenize([architecture], params, preferred={"top_k": 32})
    assert result.top_k == expected
    assert params["top_k"] == top_k


def test_native_zero_preserves_other_sampling_and_input():
    params = {"top_k": 0, "temperature": 0.7, "top_p": 0.9}
    result = tokenize([MUSE_ARCHITECTURES[0]], params)
    assert result.top_k == TOP_K_ALL
    assert result.temperature == 0.7
    assert result.top_p == 0.9
    assert params == {"top_k": 0, "temperature": 0.7, "top_p": 0.9}


def test_zero_temperature_remains_greedy():
    result = tokenize([MUSE_ARCHITECTURES[0]], {"top_k": 0, "temperature": 0})
    assert result.top_k == 1


@pytest.mark.parametrize(
    "architectures",
    [
        ["LlamaForCausalLM"],
        ["MuseGlimmerForConditionalGeneration"],
        ["DFlash2DraftModel"],
        [],
        None,
    ],
)
def test_other_models_still_reject_zero(architectures):
    with pytest.raises(ValueError, match="top_k must be -1"):
        tokenize(architectures, {"top_k": 0})


def test_muse_still_rejects_other_invalid_top_k():
    with pytest.raises(ValueError, match="top_k must be -1"):
        tokenize([MUSE_ARCHITECTURES[0]], {"top_k": -2})
