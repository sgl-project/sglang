# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from sglang.multimodal_gen.configs.models.encoders.qwen3_5 import Qwen3_5EncoderConfig
from sglang.multimodal_gen.runtime.loader.utils import set_default_torch_dtype
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.encoders.qwen3_5 import Qwen3_5Model
from sglang.multimodal_gen.runtime.server_args import get_global_server_args


@pytest.fixture(autouse=True)
def restore_attention_backend(monkeypatch):
    monkeypatch.setattr(get_global_server_args(), "attention_backend", "torch_sdpa")


@pytest.fixture
def small_text_encoder():
    from transformers import Qwen3_5Config, Qwen3_5Model

    config = Qwen3_5Config(
        text_config=dict(
            vocab_size=64,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=32,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
            max_position_embeddings=512,
            layer_types=["linear_attention", "full_attention"],
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.25,
                "mrope_section": [1, 1, 2],
                "mrope_interleaved": True,
            },
        ),
        vision_config=dict(
            depth=1,
            hidden_size=32,
            intermediate_size=64,
            num_heads=2,
            patch_size=2,
            temporal_patch_size=1,
            out_hidden_size=64,
            num_position_embeddings=16,
        ),
    )
    config._attn_implementation = "sdpa"
    return Qwen3_5Model(config).eval()


def native_encoder(reference):
    config = Qwen3_5EncoderConfig()
    config.update_model_arch(reference.config.to_dict())
    config.post_diffusers_config_update()
    with set_default_torch_dtype(reference.dtype), torch.device(reference.device):
        model = Qwen3_5Model(config).eval()
    loaded = model.load_weights(reference.state_dict().items())
    model.to(reference.device)
    assert loaded == set(dict(model.named_parameters()))
    return model


def cuda_reference(reference):
    reference = reference.to("cuda")
    # Match checkpoint loading: BF16 weights with FP32 derived RoPE buffers.
    for parameter in reference.parameters():
        parameter.data = parameter.data.to(torch.bfloat16)
    return reference


def test_native_encoder_registry_and_checkpoint(small_text_encoder):
    from sglang.multimodal_gen.runtime.models.registry import ModelRegistry

    cls, _ = ModelRegistry.resolve_model_cls(["Qwen3_5Model"])
    assert cls is Qwen3_5Model
    model = native_encoder(small_text_encoder)
    assert model.config.hidden_size == small_text_encoder.config.text_config.hidden_size
    assert not hasattr(model, "visual")
    assert not model.should_materialize_checkpoint_weight(
        "visual.patch_embed.proj.weight"
    )
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(
            parameter, small_text_encoder.state_dict()[name], rtol=0, atol=0
        )
    with pytest.raises(KeyError):
        model.load_weights([("language_model.invalid.weight", torch.zeros(1))])


@pytest.mark.parametrize("backend", [None, "torch_sdpa", "fa"])
def test_text_attention_default_and_explicit_selection(small_text_encoder, backend):
    from sglang.multimodal_gen.runtime.platforms import (
        AttentionBackendEnum,
        current_platform,
    )

    if backend == "fa" and not torch.cuda.is_available():
        pytest.skip("FA selection requires CUDA")
    if backend == "fa" and current_platform.is_rocm():
        pytest.skip("the platform currently resolves FA to SDPA on ROCm")
    get_global_server_args().attention_backend = backend
    config = Qwen3_5EncoderConfig()
    config.update_model_arch(small_text_encoder.config.to_dict())
    config.post_diffusers_config_update()
    with set_default_torch_dtype(torch.bfloat16):
        model = Qwen3_5Model(config)
    expected = (
        AttentionBackendEnum.FA if backend == "fa" else AttentionBackendEnum.TORCH_SDPA
    )
    assert model.language_model.layers[1].self_attn.attn.backend is expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("length", [1, 17, 65, 129])
@pytest.mark.parametrize("padding", [None, "left", "right"])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_native_hybrid_encoder_matches_hf(
    small_text_encoder, length, padding, batch_size
):
    torch.manual_seed(0)
    reference = cuda_reference(small_text_encoder)
    model = native_encoder(reference)
    ids = torch.randint(1, 40, (batch_size, length), device="cuda")
    mask = torch.ones_like(ids)
    if padding == "left":
        mask[-1, : length // 3] = 0
    elif padding == "right" and length > 1:
        mask[-1, -max(length // 3, 1) :] = 0
    positions = torch.arange(length, device="cuda")[None].expand(batch_size, -1) + 3
    with torch.inference_mode(), set_forward_context(0, None):
        # Use math attention as the reference, including masked partial tiles.
        with sdpa_kernel(SDPBackend.MATH):
            expected = reference(
                ids, attention_mask=mask, position_ids=positions, use_cache=False
            ).last_hidden_state
        result = model(
            ids, attention_mask=mask, position_ids=positions, output_hidden_states=True
        )
        assert (
            len(result.hidden_states)
            == reference.config.text_config.num_hidden_layers + 1
        )
        valid = mask.bool()
        torch.testing.assert_close(
            result.last_hidden_state[valid], expected[valid], atol=0.04, rtol=0.01
        )
        # Every call starts a fresh recurrent state, including after a different length.
        model(ids[:, :1], use_cache=False)
        again = model(
            inputs_embeds=model.language_model.embed_tokens(ids),
            attention_mask=mask,
            position_ids=positions,
            return_dict=False,
        )[0]
        torch.testing.assert_close(again, result.last_hidden_state, atol=0, rtol=0)


def test_encoder_rejects_non_text_prefill(small_text_encoder):
    model = native_encoder(small_text_encoder)
    ids = torch.ones(1, 4, dtype=torch.long)
    with pytest.raises(ValueError, match="cache"):
        model(ids, use_cache=True)
    with pytest.raises(ValueError, match="text inputs only"):
        model(ids, pixel_values=torch.zeros(1))
    with pytest.raises(ValueError, match="exactly one"):
        model(ids, inputs_embeds=torch.zeros(1, 4, 64))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("length", [1, 13, 63, 64, 65, 96, 127, 128, 129, 2524, 4096])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("qk_norm", [False, True])
def test_gdn_prefill_matches_hf_production_exactly(length, dtype, qk_norm):
    from transformers.models.qwen3_5 import modeling_qwen3_5

    from sglang.multimodal_gen.runtime.models.encoders.qwen3_5 import (
        _chunk_gated_delta_rule,
    )

    torch.manual_seed(0)
    shape = (2, length, 4, 32)
    q, k, v = [torch.randn(shape, device="cuda", dtype=dtype) for _ in range(3)]
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    g = -torch.rand(shape[:-1], device="cuda")
    beta = torch.rand(shape[:-1], device="cuda", dtype=dtype)
    with torch.inference_mode():
        expected, _ = modeling_qwen3_5.torch_chunk_gated_delta_rule(
            q, k, v, g, beta, use_qk_l2norm_in_kernel=qk_norm
        )
        actual = _chunk_gated_delta_rule(q, k, v, g, beta, use_qk_l2norm=qk_norm)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("resident_layers", [0, 1])
@pytest.mark.parametrize("host_embedding", [False, True])
def test_native_layerwise_and_weight_updates(
    small_text_encoder, monkeypatch, resident_layers, host_embedding
):
    from sglang.multimodal_gen.runtime.managers.memory_managers import layerwise_offload
    from sglang.multimodal_gen.runtime.post_training.weights_updater import (
        _load_weights_into_module,
    )

    torch.manual_seed(0)
    encoder = native_encoder(cuda_reference(small_text_encoder))
    monkeypatch.setattr(
        layerwise_offload,
        "HOST_RESIDENT_TABLE_MIN_BYTES",
        0 if host_embedding else 2**30,
    )
    tokens = [
        torch.randint(1, 40, (1, length), device="cuda") for length in (17, 5, 33, 7)
    ]
    with torch.inference_mode(), set_forward_context(0, None):
        expected = [encoder(ids).last_hidden_state.cpu() for ids in tokens]
    name = "language_model.layers.0.mlp.down_proj.weight"
    changed_weight = small_text_encoder.state_dict()[name].cpu().clone() + 0.125
    encoder.configure_layerwise_offload(
        SimpleNamespace(
            performance_mode="manual",
            pin_cpu_memory=True,
            layerwise_tuning_for=lambda *args, **kwargs: (
                1,
                resident_layers,
                "strided",
                "forward",
            ),
        ),
        component_name="text_encoder",
    )
    [manager] = encoder.layerwise_offload_managers
    assert manager.enabled and manager.num_layers == 2
    assert (
        encoder.language_model.embed_tokens.weight.device.type == "cpu"
    ) == host_embedding
    torch.cuda.synchronize()
    try:
        for start in (0, 2):
            with (
                torch.cuda.stream(torch.cuda.Stream()),
                torch.inference_mode(),
                set_forward_context(0, None),
            ):
                encoder.prepare_for_next_req()
                outputs = [
                    encoder(ids).last_hidden_state for ids in tokens[start : start + 2]
                ]
                manager.release_all()
            torch.cuda.synchronize()
            for output, reference in zip(outputs, expected[start : start + 2]):
                torch.testing.assert_close(output.cpu(), reference, rtol=0, atol=0)
        _load_weights_into_module(encoder, [(name, changed_weight)])
        with torch.inference_mode(), set_forward_context(0, None):
            encoder.prepare_for_next_req()
            updated = encoder(tokens[0]).last_hidden_state.cpu()
            manager.release_all()
            encoder.disable_offload()
            resident = encoder(tokens[0]).last_hidden_state.cpu()
        torch.testing.assert_close(updated, resident, rtol=0, atol=0)
        assert not torch.equal(updated, expected[0])
    finally:
        encoder.disable_offload()
        manager.release_host_stores()
