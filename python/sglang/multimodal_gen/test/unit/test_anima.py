# SPDX-License-Identifier: Apache-2.0
import json
import sys
from types import SimpleNamespace

import pytest
import torch
from diffusers.models.transformers.transformer_cosmos import CosmosRotaryPosEmbed
from transformers import BatchEncoding

from sglang.multimodal_gen.configs.models.dits.anima import AnimaArchConfig
from sglang.multimodal_gen.configs.pipeline_configs.anima import AnimaPipelineConfig
from sglang.multimodal_gen.configs.sample.anima import AnimaSamplingParams
from sglang.multimodal_gen.registry import get_model_info
from sglang.multimodal_gen.runtime.models.dits.anima import AnimaRotaryEmbedding
from sglang.multimodal_gen.runtime.models.encoders.qwen3 import Qwen3Attention
from sglang.multimodal_gen.runtime.models.registry import ModelRegistry
from sglang.multimodal_gen.runtime.pipelines.anima_pipeline import AnimaPipeline
from sglang.multimodal_gen.runtime.utils import hf_diffusers_utils


def test_registry_recognizes_official_and_local_anima(monkeypatch):
    monkeypatch.setattr(
        "sglang.multimodal_gen.registry.maybe_download_model_index",
        lambda _: {"_class_name": "AnimaModularPipeline"},
    )
    get_model_info.cache_clear()
    for model in ("circlestone-labs/Anima-Base-v1.0-Diffusers", "/models/custom-anima"):
        info = get_model_info(model)
        assert info.pipeline_cls is AnimaPipeline
        assert info.pipeline_config_cls is AnimaPipelineConfig
        assert info.sampling_param_cls is AnimaSamplingParams
    get_model_info.cache_clear()
    assert (
        ModelRegistry.resolve_model_cls("Qwen3Model")[0].__name__ == "Qwen3ForCausalLM"
    )
    assert (
        ModelRegistry.resolve_model_cls("AnimaTextConditioner")[0].__name__
        == "AnimaTextConditioner"
    )


def test_empty_prompt_keeps_a_masked_qwen_token():
    def tokenizer(prompts, **kwargs):
        assert kwargs["padding"] == "longest"
        return BatchEncoding(
            {
                "input_ids": torch.empty(len(prompts), 0, dtype=torch.long),
                "attention_mask": torch.empty(len(prompts), 0, dtype=torch.long),
            }
        )

    inputs = AnimaPipelineConfig().tokenize_prompt([""], tokenizer, {})
    assert inputs.input_ids.shape == (1, 1)
    assert not inputs.attention_mask.any()


def test_qwen_all_masked_row_never_calls_attention_with_empty_kv():
    def attention(*args):
        raise AssertionError("all-masked rows have no valid KV")

    module = SimpleNamespace(attn=attention)
    q = torch.randn(1, 1, 2, 8)
    out = Qwen3Attention._masked_causal_attention(module, q, q, q, (0,))
    torch.testing.assert_close(out, torch.zeros_like(q))


@pytest.mark.parametrize("height,width", [(4, 6), (16, 16), (10, 14)])
def test_rope_matches_cosmos(height, width):
    arch = AnimaArchConfig()
    x = torch.zeros(1, 16, 1, height, width)
    actual = AnimaRotaryEmbedding(arch)(x)
    reference = CosmosRotaryPosEmbed(
        arch.attention_head_dim,
        max_size=arch.max_size,
        patch_size=arch.patch_size,
        rope_scale=arch.rope_scale,
    )(x)
    for a, b in zip(actual, reference):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_latents_and_conditioning_preserve_sample_batch():
    config = AnimaPipelineConfig()
    config.vae_config.post_init()
    batch = SimpleNamespace(
        height=512,
        width=768,
        num_outputs_per_prompt=2,
        prompt_embeds=[torch.randn(2, 512, 8)],
        negative_prompt_embeds=[torch.randn(2, 512, 8)],
        do_classifier_free_guidance=True,
    )
    original = batch.prompt_embeds[0].clone()
    assert config.prepare_latent_shape(batch, 4, 1) == (4, 16, 1, 64, 96)
    assert config.get_latent_dtype(torch.bfloat16) == torch.float32
    config.expand_conditioning_to_sample_batch(batch)
    torch.testing.assert_close(batch.prompt_embeds[0], original.repeat_interleave(2, 0))
    assert batch.negative_prompt_embeds[0].shape == (4, 512, 8)
    latents = torch.randn(4, 16, 1, 64, 96)
    value, sharded = config.shard_latents_for_sp(batch, latents)
    assert value is latents and not sharded


def test_modular_index_uses_existing_component_loaders(tmp_path, monkeypatch):
    index = {
        "_class_name": "AnimaModularPipeline",
        "_blocks_class_name": "AnimaAutoBlocks",
        "_diffusers_version": "0.39.0.dev0",
        "transformer": [
            "diffusers",
            "CosmosTransformer3DModel",
            {"subfolder": "transformer"},
        ],
        "t5_tokenizer": ["transformers", "T5Tokenizer", {"subfolder": "t5_tokenizer"}],
    }
    path = tmp_path / "modular_model_index.json"
    path.write_text(json.dumps(index))
    (tmp_path / "transformer").mkdir()
    (tmp_path / "transformer" / "diffusion_pytorch_model.safetensors").touch()
    (tmp_path / "t5_tokenizer").mkdir()
    config = hf_diffusers_utils.verify_model_config_and_directory(str(tmp_path))
    assert config["transformer"] == index["transformer"][:2]
    assert "_blocks_class_name" not in config
    assert hf_diffusers_utils._verify_diffusers_model_complete(str(tmp_path))
    calls = []

    def download(repo_id, filename):
        calls.append(filename)
        if filename == "model_index.json":
            raise hf_diffusers_utils.EntryNotFoundError("not found")
        return str(path)

    monkeypatch.setattr(hf_diffusers_utils, "hf_hub_download", download)
    remote = hf_diffusers_utils.maybe_download_model_index("test/anima")
    assert remote["transformer"] == config["transformer"]
    assert calls == ["model_index.json", "modular_model_index.json"]
    (tmp_path / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "Legacy",
                "_diffusers_version": "0.37.0",
                "transformer": index["transformer"][:2],
            }
        )
    )
    assert (
        hf_diffusers_utils.verify_model_config_and_directory(str(tmp_path))[
            "_class_name"
        ]
        == "Legacy"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
