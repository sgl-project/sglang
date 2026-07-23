# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.configs.pipeline_configs.minwm import (
    MINWM_ACTION_WEIGHTS_CONDITION,
    MINWM_TOTAL_CHUNKS_CONDITION,
    MinWMCausalDMDConfig,
    minwm_t5_postprocess_text,
)
from sglang.multimodal_gen.runtime.models.dits.minwm_action import (
    PrimitiveTokenResidualActionEncoder,
    action_labels_to_primitive_bits,
    key_state_to_action_label,
    validate_action_labels,
    validate_action_weights,
)
from sglang.multimodal_gen.runtime.models.dits.minwm import (
    MinWMPatchEmbed,
    MinWMRMSNorm,
    _frame_gate,
    _frame_modulation,
    _minwm_adaln_op,
    _minwm_adaln_modulation,
    _minwm_layer_norm,
    _minwm_packed_attention_backend,
    _minwm_qk_norm_rope_op,
    apply_minwm_rotary_embedding,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.adapters.minwm_realtime_adapter import (
    MinWMRealtimeAdapter,
    MinWMRealtimeState,
)
from sglang.multimodal_gen.runtime.pipelines.minwm_causal_dmd_pipeline import (
    MinWMCausalDMDPipeline,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minwm.minwm_causal_denoising import (
    MinWMCausalDMDDenoisingStage,
    MinWMChunkLatentPreparationStage,
)
from sglang.multimodal_gen.runtime.realtime.session import RealtimeSession
from sglang.multimodal_gen.tools.convert_minwm_checkpoint import DEFAULT_SOURCE_URI


@pytest.mark.parametrize(
    ("keys", "expected"),
    [
        ([], 0),
        (["w"], 9),
        (["l"], 1),
        (["w", "l"], 10),
        (["w", "a"], 45),
        (["i", "j"], 7),
        (["up", "right"], 5),
    ],
)
def test_minwm_action_key_ontology(keys, expected):
    assert key_state_to_action_label(keys) == expected


def test_minwm_action_label_bits_match_wasd_ijkl_order():
    labels = torch.tensor([[0, 9, 1, 10, 45, 7]])
    bits = action_labels_to_primitive_bits(labels).to(torch.int64)
    assert bits.tolist() == [
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
        ]
    ]


def test_minwm_action_validation_is_exact_integer_labels():
    assert validate_action_labels([0, 80], expected_frames=2) == [0, 80]
    for invalid in ([True], [81], [-1], [1.0], torch.tensor([1])):
        with pytest.raises(ValueError):
            validate_action_labels(invalid)


def test_minwm_action_weight_validation_preserves_fractional_amplitude():
    row = [0.8, 0, 0, 0, 0, 0, 0, 0]
    assert validate_action_weights([row], expected_frames=1) == [
        [0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ]
    for invalid in ([[1.1] + [0] * 7], [[float("nan")] + [0] * 7], [[1] * 7]):
        with pytest.raises(ValueError):
            validate_action_weights(invalid)


def test_minwm_binary_weight_windows_are_label_path_degenerate_case():
    torch.manual_seed(17)
    encoder = PrimitiveTokenResidualActionEncoder(
        dim=24, embed_dim=8, hidden_dim=16, kernel_size=3
    )
    labels = torch.tensor([[0, 9, 10, 1]])
    windows = action_labels_to_primitive_bits(labels).unsqueeze(2).expand(-1, -1, 4, -1)
    # The two forms are mathematically equivalent. Their different GEMM batch
    # shapes can select slightly different CPU kernels, so this is a numerical
    # semantic check; end-to-end parity compares the same weighted path.
    torch.testing.assert_close(
        encoder.frame_states(windows),
        encoder.frame_states(labels),
        rtol=3e-5,
        atol=3e-7,
    )


def test_minwm_action_history_chunk_matches_full_sequence():
    torch.manual_seed(7)
    encoder = PrimitiveTokenResidualActionEncoder(
        dim=24, embed_dim=8, hidden_dim=16, kernel_size=3
    )
    labels = torch.tensor([[0, 9, 10, 1, 45, 7, 36, 4, 80]])
    full = encoder.frame_states(labels)
    chunk_window = encoder.frame_states(labels[:, -8:])
    # Convolution kernels can accumulate the full sequence and the sliced
    # history window in a different order even on CPU. The causal receptive
    # fields are mathematically identical; observed PyTorch drift is < 6e-8.
    torch.testing.assert_close(chunk_window[:, -4:], full[:, -4:], rtol=1e-6, atol=1e-7)
    token_residual = encoder.token_residual(
        labels[:, -8:],
        num_current_frames=4,
        tokens_per_frame=6,
        dtype=torch.float32,
    )
    assert token_residual.is_contiguous()
    assert token_residual.stride(-1) == 1


def test_minwm_bounded_session_presamples_reference_and_full_horizon(monkeypatch):
    import sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minwm.minwm_causal_denoising as stage_module

    monkeypatch.setattr(
        stage_module, "get_local_torch_device", lambda: torch.device("cpu")
    )
    transformer = SimpleNamespace(
        config=SimpleNamespace(
            arch_config=SimpleNamespace(num_frames_per_block=4, out_channels=2)
        )
    )
    stage = MinWMChunkLatentPreparationStage(transformer)
    session = RealtimeSession()
    generator = torch.Generator().manual_seed(123)
    condition = torch.zeros(1, 2, 1, 1, 3)

    expected_generator = torch.Generator().manual_seed(123)
    expected = torch.randn(
        (1, 9, 2, 1, 3), generator=expected_generator, dtype=torch.float32
    )[:, 1:]

    def make_batch(block_idx):
        return SimpleNamespace(
            latents=None,
            image_latent=condition,
            realtime_chunk_size=4,
            generator=generator,
            session=session,
            block_idx=block_idx,
            condition_inputs={MINWM_TOTAL_CHUNKS_CONDITION: 2},
            raw_latent_shape=None,
        )

    first = stage.forward(make_batch(0), SimpleNamespace())
    second = stage.forward(make_batch(1), SimpleNamespace())
    torch.testing.assert_close(
        first.latents,
        expected[:, :4].permute(0, 2, 1, 3, 4),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        second.latents,
        expected[:, 4:].permute(0, 2, 1, 3, 4),
        rtol=0,
        atol=0,
    )
    assert torch.equal(generator.get_state(), expected_generator.get_state())


def test_minwm_default_kv_horizon_retains_complete_bounded_session():
    stage = MinWMCausalDMDDenoisingStage.__new__(MinWMCausalDMDDenoisingStage)
    stage.sliding_window_num_frames = 128
    stage.num_frames_per_block = 4
    batch = SimpleNamespace(
        realtime_causal_sink_size=None,
        realtime_causal_kv_cache_num_frames=None,
        condition_inputs={MINWM_TOTAL_CHUNKS_CONDITION: 8},
    )
    pipeline_config = SimpleNamespace(
        realtime_causal_sink_size=None,
        realtime_causal_kv_cache_num_frames=None,
    )
    stage._apply_causal_cache_overrides(
        batch, SimpleNamespace(pipeline_config=pipeline_config)
    )
    assert stage.sliding_window_num_frames == 33
    assert stage._causal_kv_cache_kwargs(None) == {"allow_growth": True}

    batch.realtime_causal_kv_cache_num_frames = 45
    stage._apply_causal_cache_overrides(
        batch, SimpleNamespace(pipeline_config=pipeline_config)
    )
    assert stage.sliding_window_num_frames == 45
    assert stage._causal_kv_cache_kwargs(None) == {"allow_growth": False}


def test_minwm_unbounded_kv_policy_reaches_cache_allocation():
    stage = MinWMCausalDMDDenoisingStage.__new__(MinWMCausalDMDDenoisingStage)
    stage.num_transformer_blocks = 2
    stage.num_token_per_frame = 3
    stage.sliding_window_num_frames = 5
    stage.local_attn_size = -1
    stage.sink_size = 0
    stage.transformer = SimpleNamespace(
        num_attention_heads=2,
        attention_head_dim=4,
    )

    stage._initialize_kv_cache(
        batch_size=1,
        dtype=torch.float32,
        device=torch.device("cpu"),
        allow_growth=True,
    )

    assert len(stage.causal_kv_cache) == 2
    assert all(cache.allow_growth for cache in stage.causal_kv_cache)
    assert stage.causal_kv_cache[0].k.shape == (1, 15, 2, 4)


def test_minwm_fractional_action_windows_reach_denoiser_unchanged():
    stage = MinWMCausalDMDDenoisingStage.__new__(MinWMCausalDMDDenoisingStage)
    stage.transformer = SimpleNamespace(
        config=SimpleNamespace(arch_config=SimpleNamespace(action_history_frames=4))
    )
    window = [[0.8, 0, 0, 0, 0, 0, 0, 0]] * 4
    batch = SimpleNamespace(
        session=RealtimeSession(),
        block_idx=0,
        latents=torch.zeros(1, 2, 4, 1, 1),
        condition_inputs={MINWM_ACTION_WEIGHTS_CONDITION: [window] * 4},
    )
    server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(
            vae_config=SimpleNamespace(
                arch_config=SimpleNamespace(scale_factor_temporal=4)
            )
        )
    )
    action = stage._prepare_causal_dmd_pos_cond_kwargs(
        batch, server_args, torch.bfloat16
    )["action"]
    assert action.shape == (1, 5, 4, 8)
    assert torch.count_nonzero(action[:, :1]).item() == 0
    torch.testing.assert_close(
        action[:, 1:, :, 0], torch.full((1, 4, 4), 0.8), rtol=0, atol=0
    )


def test_minwm_realtime_adapter_groups_pixel_weights_by_vae_factor():
    state = MinWMRealtimeState()
    row = [0.8, 0, 0, 0, 0, 0, 0, 0]
    state.receive_action_weights([row] * 16)
    session = SimpleNamespace(
        adapter_state=state,
        request=SimpleNamespace(prompt="test", max_chunks=8),
    )
    server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(
            vae_config=SimpleNamespace(
                arch_config=SimpleNamespace(scale_factor_temporal=4)
            )
        )
    )
    inputs = MinWMRealtimeAdapter().sample_chunk_inputs(
        session,
        server_args,
        SimpleNamespace(index=0),
        chunk_size=4,
    )
    windows = inputs.condition_inputs[MINWM_ACTION_WEIGHTS_CONDITION]
    assert len(windows) == 4
    assert all(len(window) == 4 for window in windows)
    assert windows == [[row] * 4] * 4
    assert inputs.condition_inputs[MINWM_TOTAL_CHUNKS_CONDITION] == 8


def test_minwm_text_context_keeps_zero_padded_512_contract():
    hidden = torch.randn(1, 1024, 8)
    attention_mask = torch.zeros(1, 1024, dtype=torch.long)
    attention_mask[:, :17] = 1
    output = minwm_t5_postprocess_text(
        SimpleNamespace(last_hidden_state=hidden, attention_mask=attention_mask),
        None,
    )
    assert output.prompt_seq_lens == [512]
    assert output.prompt_embeds_mask.sum().item() == 512
    torch.testing.assert_close(
        output.prompt_embeds[:, :17], hidden[:, :17], rtol=0, atol=0
    )
    assert torch.count_nonzero(output.prompt_embeds[:, 17:]).item() == 0


def test_minwm_native_hf_text_output_uses_tokenizer_attention_mask():
    hidden = torch.randn(1, 1024, 8)
    attention_mask = torch.zeros(1, 1024, dtype=torch.long)
    attention_mask[:, :23] = 1
    output = minwm_t5_postprocess_text(
        SimpleNamespace(last_hidden_state=hidden),
        {"attention_mask": attention_mask},
    )
    assert output.prompt_seq_lens == [512]
    assert output.prompt_embeds_mask.sum().item() == 512
    assert torch.count_nonzero(output.prompt_embeds[:, 23:]).item() == 0


def test_minwm_requires_baseline_native_text_and_vae_components():
    config = MinWMCausalDMDConfig()
    assert config.native_component_names == ("text_encoder", "vae")
    assert config.enable_autocast is False


def test_minwm_converter_defaults_to_requested_0721_checkpoint():
    assert (
        "wan22-5B-stage3-dmd-8-0721-6a531f0e067/global_step_003200"
        in DEFAULT_SOURCE_URI
    )


def test_minwm_rejects_unimplemented_sequence_parallelism():
    MinWMCausalDMDPipeline._validate_sequence_parallelism_args(
        SimpleNamespace(sp_degree=1, ulysses_degree=1, ring_degree=1)
    )
    with pytest.raises(ValueError, match="does not support sequence parallelism"):
        MinWMCausalDMDPipeline._validate_sequence_parallelism_args(
            SimpleNamespace(sp_degree=2, ulysses_degree=2, ring_degree=1)
        )


@pytest.mark.parametrize(
    ("capability", "available", "expected"),
    [
        (10, {"flash_attn.cute", "flash_attn"}, "fa4"),
        (9, {"flash_attn_interface", "flash_attn"}, "fa3"),
        (9, {"flash_attn"}, "fa2"),
    ],
)
def test_minwm_attention_backend_matches_source_device_fallback(
    monkeypatch, capability, available, expected
):
    import sglang.multimodal_gen.runtime.models.dits.minwm as minwm_module

    monkeypatch.setattr(
        minwm_module.torch.cuda,
        "get_device_capability",
        lambda _device: (capability, 0),
    )
    monkeypatch.setattr(
        minwm_module.importlib.util,
        "find_spec",
        lambda name: object() if name in available else None,
    )
    assert _minwm_packed_attention_backend(torch.device("cuda")) == expected


def test_minwm_allows_benchmark_component_ablation(monkeypatch):
    monkeypatch.setenv("MINWM_NATIVE_COMPONENTS", "")
    config = MinWMCausalDMDConfig()
    assert config.native_component_names == ()


def test_minwm_rms_norm_rounds_before_weight_multiply():
    layer = MinWMRMSNorm(8, eps=1e-6).to(torch.bfloat16)
    hidden = torch.tensor(
        [[[0.1, -0.2, 0.3, -0.4, 1.5, -2.0, 3.25, -4.5]]],
        dtype=torch.bfloat16,
    )
    layer.weight.data.copy_(
        torch.tensor([0.7, 0.9, 1.1, 1.3, 0.8, 1.2, 1.4, 0.6], dtype=torch.bfloat16)
    )
    normalized = hidden.float() * torch.rsqrt(
        hidden.float().pow(2).mean(dim=-1, keepdim=True) + layer.eps
    )
    expected = normalized.to(hidden.dtype) * layer.weight
    multiply_before_rounding = (normalized * layer.weight).to(hidden.dtype)
    torch.testing.assert_close(layer(hidden), expected, rtol=0, atol=0)
    assert not torch.equal(expected, multiply_before_rounding)


def test_minwm_patch_embed_uses_native_conv3d_path():
    torch.manual_seed(5)
    layer = MinWMPatchEmbed(
        patch_size=(1, 2, 2), in_chans=3, embed_dim=7, flatten=False
    )
    hidden = torch.randn(1, 3, 2, 6, 8)
    expected = layer.proj(hidden)
    torch.testing.assert_close(layer(hidden), expected, rtol=0, atol=0)


def test_minwm_adaln_uses_bf16_modulation_sum_and_fp32_layer_norm():
    hidden = torch.tensor(
        [
            [
                [0.25, -0.5, 1.0, -2.0],
                [0.5, -1.0, 2.0, -4.0],
                [0.75, -1.5, 3.0, -6.0],
                [1.0, -2.0, 4.0, -8.0],
            ]
        ],
        dtype=torch.bfloat16,
    )
    model_value = torch.tensor([[0.101, -0.202, 0.303, -0.404]], dtype=torch.float32)
    timestep_value = torch.tensor(
        [[[0.011, -0.022, 0.033, -0.044], [0.055, -0.066, 0.077, -0.088]]],
        dtype=torch.bfloat16,
    )
    actual_modulation = _frame_modulation(
        hidden, model_value, timestep_value, num_frames=2
    )
    expected_frame_values = model_value.to(torch.bfloat16) + timestep_value
    expected_modulation = (
        expected_frame_values.unsqueeze(2).expand(-1, -1, 2, -1).flatten(1, 2)
    )
    torch.testing.assert_close(actual_modulation, expected_modulation, rtol=0, atol=0)

    actual_gate = _frame_gate(hidden, model_value, timestep_value, num_frames=2)
    expected_gate = (
        (model_value.to(torch.bfloat16).float() + timestep_value.float())
        .unsqueeze(2)
        .expand(-1, -1, 2, -1)
        .flatten(1, 2)
    )
    torch.testing.assert_close(actual_gate, expected_gate, rtol=0, atol=0)

    expected_norm = torch.nn.functional.layer_norm(
        hidden.float(), (hidden.shape[-1],), eps=1e-6
    ).to(torch.bfloat16)
    torch.testing.assert_close(
        _minwm_layer_norm(hidden, eps=1e-6), expected_norm, rtol=0, atol=0
    )

    expected_adaln = (
        torch.nn.functional.layer_norm(hidden.float(), (hidden.shape[-1],), eps=1e-6)
        * (1 + actual_modulation.float())
        + actual_modulation.float()
    ).to(torch.bfloat16)
    torch.testing.assert_close(
        _minwm_adaln_modulation(
            hidden,
            actual_modulation,
            actual_modulation,
            eps=1e-6,
        ),
        expected_adaln,
        rtol=0,
        atol=0,
    )


def test_minwm_fused_segments_match_main_eager_formulas():
    torch.manual_seed(11)
    hidden = torch.randn(1, 6, 8, dtype=torch.bfloat16)
    residual = torch.randn_like(hidden)
    model = torch.randn(1, 8, dtype=torch.bfloat16)
    timestep = torch.randn_like(hidden)
    expected_residual = (
        hidden.float() + residual.float() * (model.float() + timestep.float())
    ).to(hidden.dtype)
    actual_residual = _minwm_adaln_op(
        hidden,
        y=residual,
        m_gate=model,
        e_gate=timestep,
    )
    torch.testing.assert_close(actual_residual, expected_residual, rtol=0, atol=0)

    query = torch.randn(1, 6, 8, dtype=torch.bfloat16)
    key = torch.randn_like(query)
    query_weight = torch.randn(8, dtype=torch.bfloat16)
    key_weight = torch.randn(8, dtype=torch.bfloat16)
    angles = torch.randn(6, 2, dtype=torch.float32)
    rope = torch.stack((angles.cos(), angles.sin()), dim=-1)
    actual_query, actual_key = _minwm_qk_norm_rope_op(
        query, key, query_weight, key_weight, 1e-6, rope, 2
    )

    def expected(value, weight):
        value_float = value.float()
        value = (
            value_float
            * torch.rsqrt(value_float.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        ).to(value.dtype) * weight
        value = value.reshape(1, 6, 2, 4)
        real, imaginary = value[..., 0::2].float(), value[..., 1::2].float()
        cos = rope[..., 0].view(1, 6, 1, 2)
        sin = rope[..., 1].view(1, 6, 1, 2)
        return (
            torch.stack(
                (real * cos - imaginary * sin, real * sin + imaginary * cos),
                dim=-1,
            )
            .flatten(-2)
            .to(value.dtype)
        )

    torch.testing.assert_close(
        actual_query, expected(query, query_weight), rtol=0, atol=0
    )
    torch.testing.assert_close(actual_key, expected(key, key_weight), rtol=0, atol=0)


def test_minwm_rotary_embedding_matches_main_explicit_formula():
    torch.manual_seed(9)
    hidden = torch.randn(1, 5, 3, 8, dtype=torch.bfloat16)
    angles = torch.randn(5, 4, dtype=torch.float64)
    cos = angles.cos().float()
    sin = angles.sin().float()
    real = hidden[..., 0::2].float()
    imaginary = hidden[..., 1::2].float()
    expected = torch.stack(
        (
            real * cos.view(1, 5, 1, 4) - imaginary * sin.view(1, 5, 1, 4),
            real * sin.view(1, 5, 1, 4) + imaginary * cos.view(1, 5, 1, 4),
        ),
        dim=-1,
    ).flatten(-2)
    expected = expected.to(hidden.dtype)
    torch.testing.assert_close(
        apply_minwm_rotary_embedding(hidden, cos, sin), expected, rtol=0, atol=0
    )


def test_minwm_vae_pixel_and_latent_arithmetic_matches_main():
    config = MinWMCausalDMDConfig()
    assert config.preprocess_vae_encode_before_dtype_cast is True
    pixels = torch.arange(256, dtype=torch.uint8).reshape(1, 1, 1, 16, 16)
    generic_normalized = pixels.float().div(255.0).mul(2.0).sub(1.0)
    actual_pixels = config.preprocess_vae_encode(generic_normalized, None)
    expected_pixels = pixels.to(torch.bfloat16).div(127.5).sub(1.0)
    torch.testing.assert_close(actual_pixels, expected_pixels, rtol=0, atol=0)

    vae = SimpleNamespace(
        config=SimpleNamespace(latents_mean=[0.125], latents_std=[0.75])
    )
    posterior = torch.tensor([[[[[-1.5, 0.5]]]]], dtype=torch.bfloat16)
    normalized = config.normalize_vae_encode(posterior, vae)
    expected_normalized = (
        posterior - torch.tensor(0.125, dtype=torch.bfloat16)
    ) / torch.tensor(0.75, dtype=torch.bfloat16)
    expected_normalized = (
        expected_normalized.float().to(torch.float16).to(torch.bfloat16)
    )
    torch.testing.assert_close(normalized, expected_normalized, rtol=0, atol=0)
    decoded = config.preprocess_decoding(normalized, vae=vae)
    expected_decoded = normalized * torch.tensor(
        0.75, dtype=torch.bfloat16
    ) + torch.tensor(0.125, dtype=torch.bfloat16)
    torch.testing.assert_close(decoded, expected_decoded, rtol=0, atol=0)


def test_minwm_reference_latent_does_not_get_generic_i2v_mask_channels():
    config = MinWMCausalDMDConfig()
    latent = torch.randn(1, 48, 1, 30, 52)
    assert config.postprocess_image_latent(latent, None) is latent
    with pytest.raises(ValueError, match="48 channels"):
        config.postprocess_image_latent(torch.randn(1, 52, 1, 30, 52), None)
