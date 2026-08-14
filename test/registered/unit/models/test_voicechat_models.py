import json

import torch
from transformers import AutoConfig

from sglang.srt.configs.eartts import EarTTSConfig
from sglang.srt.layers.layernorm import Gemma3RMSNorm
from sglang.srt.models.eartts import MaskGITSampler
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
