import json
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoConfig

from examples.voicechat.convert_duplex_stage import _configure_duplex_config
from sglang.srt.configs.eartts import EarTTSConfig
from sglang.srt.layers.layernorm import Gemma3RMSNorm
from sglang.srt.models.eartts import EarTTSForCausalLM, MaskGITSampler
from sglang.srt.models.nemotron_duplex_h import NemotronDuplexHForCausalLM
from sglang.srt.models.registry import ModelRegistry


def test_voicechat_model_architectures_registered():
    supported = ModelRegistry.get_supported_archs()
    assert "NemotronDuplexHForCausalLM" in supported
    assert "EarTTSForCausalLM" in supported


def test_eartts_auto_config_roundtrip(tmp_path):
    config = EarTTSConfig(
        architectures=["EarTTSForCausalLM"],
        hidden_size=16,
        num_hidden_layers=2,
    )
    (tmp_path / "config.json").write_text(json.dumps(config.to_dict()))

    loaded = AutoConfig.from_pretrained(tmp_path)

    assert isinstance(loaded, EarTTSConfig)
    assert loaded.architectures == ["EarTTSForCausalLM"]
    assert loaded.rope_parameters["full_attention"]["rope_theta"] == 1_000_000


def test_maskgit_outputs_one_valid_code_per_quantizer():
    torch.manual_seed(1)
    config = EarTTSConfig(
        hidden_size=8,
        intermediate_size=16,
        num_quantizers=3,
        codebook_size=4,
        latent_size=4,
        num_iter=2,
        exponent=3.0,
        mog_low_rank=2,
        mog_num_layers=1,
        mog_num_predictions=5,
        top_p_or_k=1.0,
    )
    sampler = MaskGITSampler(config)

    codes = sampler(torch.randn(2, config.hidden_size))

    assert codes.shape == (2, config.num_quantizers)
    assert codes.dtype == torch.long
    assert torch.all(codes >= 0)
    assert torch.all(codes < config.codebook_size)


class _FakeEarTTSBackbone:
    def load_weights(self, weights):
        [(name, _)] = weights
        return {name}


def _fake_eartts_model():
    model = SimpleNamespace(backbone=_FakeEarTTSBackbone())
    params = [
        ("backbone.model.embed_tokens.weight", torch.nn.Parameter(torch.zeros(1))),
        ("backbone.model.layers.0.weight", torch.nn.Parameter(torch.zeros(1))),
        ("total_emb.bos_emb", torch.nn.Parameter(torch.zeros(1))),
    ]
    buffers = [("sil_tokens", torch.zeros(1, dtype=torch.int32))]
    model.named_parameters = lambda: iter(params)
    model.named_buffers = lambda: iter(buffers)
    return model


def test_eartts_weight_loader_requires_complete_checkpoint():
    model = _fake_eartts_model()

    with pytest.raises(RuntimeError, match="total_emb.bos_emb"):
        EarTTSForCausalLM.load_weights(
            model,
            [
                ("model.backbone.layers.0.weight", torch.ones(1)),
                ("model.sil_tokens", torch.ones(1, dtype=torch.int32)),
            ],
        )


def test_eartts_weight_loader_accepts_complete_checkpoint():
    model = _fake_eartts_model()

    loaded = EarTTSForCausalLM.load_weights(
        model,
        [
            ("model.backbone.layers.0.weight", torch.ones(1)),
            ("model.total_emb.bos_emb", torch.ones(1)),
            ("model.sil_tokens", torch.ones(1, dtype=torch.int32)),
        ],
    )

    assert loaded == {
        "backbone.model.layers.0.weight",
        "total_emb.bos_emb",
        "sil_tokens",
    }


def test_gemma3_rmsnorm_float32_cuda_path_falls_back_to_native():
    norm = Gemma3RMSNorm(8)
    x = torch.randn(3, 8, dtype=torch.float32)
    residual = torch.randn_like(x)

    expected = norm.forward_native(x.clone(), residual.clone())
    actual = norm.forward_cuda(x.clone(), residual.clone())

    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


def test_duplex_unified_checkpoint_name_mapping():
    mapper = NemotronDuplexHForCausalLM._map_voicechat_weight_name
    assert mapper("stt_model.llm.backbone.layers.0.weight") == ("model.layers.0.weight")
    assert mapper("stt_model.llm.layers.0.weight") == "model.layers.0.weight"
    assert mapper("stt_model.function_head.weight") == "function_head.weight"


def test_duplex_conversion_pins_function_channel_and_fp32_mamba_state():
    config = _configure_duplex_config(
        SimpleNamespace(),
        {
            "duplex_text_channel_weight": 1,
            "duplex_user_channel_weight": 1,
            "duplex_function_channel_weight": 2,
        },
    )

    assert config.architectures == ["NemotronDuplexHForCausalLM"]
    assert not config.predict_user_text
    assert config.use_function_head
    assert config.mamba_ssm_dtype == "float32"
    assert config.duplex_text_channel_weight == 1.0
    assert config.duplex_user_channel_weight == 1.0
    assert config.duplex_function_channel_weight == 2.0
    assert config.fuse_method == "add"
