# SPDX-License-Identifier: Apache-2.0
"""BF16 forward, attention dispatch, CUDA graph and cache lifetime regressions."""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.configs.models.dits.kandinsky6 import (
    Kandinsky6VideoAudioConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.kandinsky6 import (
    Kandinsky6TI2VAPipelineConfig,
)
from sglang.multimodal_gen.runtime.cache.cache_dit_integration import (
    CacheDitConfig,
    disable_cache_on_transformer,
    enable_cache_on_transformer,
    refresh_context_on_transformer,
)
from sglang.multimodal_gen.runtime.distributed.cfg_policy import CFGBranch, CFGPolicy
from sglang.multimodal_gen.runtime.layers.attention.selector import (
    global_force_attn_backend_context_manager,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits.kandinsky6 import (
    Kandinsky6Attention,
    Kandinsky6Modulation,
    Kandinsky6QKNorm,
    Kandinsky6TimeEmbeddings,
    Kandinsky6Transformer3DModel,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6.denoising import (
    Kandinsky6DenoisingStage,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils import precision

pytestmark = pytest.mark.usefixtures("single_process_model_parallel")


def _randomize(module: torch.nn.Module, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for param in module.parameters():
            param.copy_(torch.randn(param.shape, generator=generator) * 0.05)


@pytest.mark.parametrize(
    "modulation", [False, True], ids=["time-embedding", "modulation"]
)
def test_bf16_projection_accepts_fp32_conditioning(modulation):
    module = (
        Kandinsky6Modulation(time_dim=8, model_dim=16, num_params=9)
        if modulation
        else Kandinsky6TimeEmbeddings(model_dim=16, time_dim=8)
    )
    _randomize(module, seed=0)
    inputs = torch.randn(2, 8) if modulation else torch.tensor([0.0, 500.0, 999.0])
    out = module.to(torch.bfloat16)(inputs)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out.float()).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA attention")
@pytest.mark.parametrize("backend", ["FA", "TORCH_SDPA", "SAGE_ATTN_3"])
@pytest.mark.parametrize("lengths", [(7, None), (129, None), (193, 65), (17, 129)])
@torch.no_grad()
def test_attention_backend_dispatch_and_repeated_forward(monkeypatch, backend, lengths):
    if backend == "FA" and torch.cuda.get_device_capability()[0] == 12:
        pytest.skip("the platform currently resolves FA to SDPA on SM12.x")
    if backend == "SAGE_ATTN_3":
        if importlib.util.find_spec("sageattn3") is None:
            pytest.skip("requires the optional SageAttention3 extension")
        if torch.cuda.get_device_capability() not in {(12, 0), (12, 1)}:
            pytest.skip("upstream SageAttention3 requires SM120 or SM121")
    monkeypatch.setattr(
        precision._mixed_precision_state,
        "state",
        precision.MixedPrecisionState(param_dtype=torch.bfloat16),
    )
    query_len, key_len = lengths
    selected = AttentionBackendEnum[backend]
    with global_force_attn_backend_context_manager(selected):
        layer = Kandinsky6Attention(
            256,
            128,
            Kandinsky6Transformer3DModel._supported_attention_backends,
            kv_dim=128 if key_len is not None else None,
            is_cross_attention=key_len is not None,
        ).eval()
    assert layer.attention.backend == selected, "must not silently fall back"
    _randomize(layer, seed=419)
    layer.query_norm.weight.fill_(1)
    layer.key_norm.weight.fill_(1)
    layer.cuda().bfloat16()
    torch.manual_seed(420)
    query = torch.randn(1, query_len, 256, device="cuda", dtype=torch.bfloat16)
    key = (
        torch.randn(1, key_len, 128, device="cuda", dtype=torch.bfloat16)
        if key_len is not None
        else None
    )
    original_query = query.clone()
    original_key = key.clone() if key is not None else None
    implementation = layer.attention.attn_impl
    forward = implementation.forward
    calls = 0

    def counted_forward(*args, **kwargs):
        nonlocal calls
        calls += 1
        return forward(*args, **kwargs)

    monkeypatch.setattr(implementation, "forward", counted_forward)
    with set_forward_context(current_timestep=0, attn_metadata=None):
        first = layer(query, key)
        repeated = layer(query, key)
    assert calls == 2, "must execute the requested backend, not a masked bypass"
    assert first.shape == query.shape and first.dtype == query.dtype
    assert torch.isfinite(first).all()
    torch.testing.assert_close(repeated, first, rtol=0, atol=0)
    torch.testing.assert_close(query, original_query, rtol=0, atol=0)
    if key is not None:
        torch.testing.assert_close(key, original_key, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("width", [16, 64, 128, 256])
@pytest.mark.parametrize("eps", [None, 1e-5])
@torch.no_grad()
def test_qk_norm_exact_output_and_parameter_storage(dtype, width, eps):
    torch.manual_seed(943)
    norm = Kandinsky6QKNorm(width, eps=eps, device="cuda", dtype=dtype)
    norm.weight.normal_()
    reference = torch.nn.RMSNorm(width, eps=eps, device="cuda", dtype=dtype)
    reference.load_state_dict(norm.state_dict())
    parameter = norm.weight
    pointer = parameter.data_ptr()
    for scale in (0.0, 0.001, 1.0, 100.0):
        x = (torch.randn(4096, width * 2, device="cuda", dtype=dtype) * scale)[:, ::2]
        expected = reference(x.float()).to(dtype)
        actual = norm(x.float()).to(dtype)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert norm.weight is parameter
        assert norm.weight.data_ptr() == pointer
        assert norm.weight.dtype == dtype

    norm.cpu().cuda()
    torch.testing.assert_close(norm(x.float()).to(dtype), expected, rtol=0, atol=0)
    torch.testing.assert_close(norm.weight, reference.weight, rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_qk_norm_autograd_preserves_native_math(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.manual_seed(944)
    norm = Kandinsky6QKNorm(128, device=device, dtype=torch.bfloat16)
    reference = torch.nn.RMSNorm(128, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        norm.weight.normal_()
    reference.load_state_dict(norm.state_dict())
    x = torch.randn(7, 128, device=device, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    expected = reference(ref_x)
    actual = norm(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(norm.weight.grad, reference.weight.grad, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, ref_x.grad, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA graphs")
@torch.no_grad()
def test_qk_norm_graph_replay_observes_weight_updates():
    norm = Kandinsky6QKNorm(128, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(17, 128, device="cuda", dtype=torch.bfloat16).float()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            norm(x)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = norm(x).to(torch.bfloat16)
    for _ in range(2):
        norm.weight.add_(0.125)
        graph.replay()
        expected = torch.nn.functional.rms_norm(
            x, norm.normalized_shape, norm.weight, norm.eps
        ).to(torch.bfloat16)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)


def _tiny_multimodal_arch_kwargs() -> dict:
    """Small but structurally real T2VA config: is_multimodal=True (every released
    Kandinsky6 checkpoint), audio tower dims intentionally different from the video
    tower's to exercise both ``Kandinsky6TimeEmbeddings``/``Kandinsky6Modulation``
    instances the fused block owns."""
    return dict(
        model_dim=32,
        time_dim=16,
        ff_dim=64,
        axes_dims=(8, 4, 4),
        in_visual_dim=4,
        out_visual_dim=4,
        in_text_dim=8,
        in_text_dim2=8,
        in_audio_dim=4,
        num_text_blocks=1,
        num_visual_blocks=1,
        patch_size=(1, 1, 1),
        visual_cond=False,
        visual_token_type_num_embeddings=0,
        is_multimodal=True,
        model_dim_a=16,
        time_dim_a=16,
        ff_dim_a=32,
        axes_dims_a=(4, 2, 2),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA graphs")
@pytest.mark.parametrize("use_cfg", [False, True])
@torch.no_grad()
def test_bcg_joint_cfg_output_lifetime_and_text_shape_fallback(monkeypatch, use_cfg):
    args = get_global_server_args()
    args.attention_backend = "fa"
    args.pipeline_config = Kandinsky6TI2VAPipelineConfig()
    config = Kandinsky6VideoAudioConfig()
    config.update_model_arch(_tiny_multimodal_arch_kwargs())
    with torch.device("meta"):
        dit = Kandinsky6Transformer3DModel(config, {}).eval()
    torch.manual_seed(7)
    dit.load_state_dict(
        {
            name: (torch.randn(param.shape, device="cuda") * 0.05).to(torch.bfloat16)
            for name, param in dit.named_parameters()
        },
        assign=True,
    )
    dit.post_load_weights()
    stage = Kandinsky6DenoisingStage(dit, scheduler=None)
    batch = SimpleNamespace(
        is_warmup=False,
        do_classifier_free_guidance=use_cfg,
        guidance_scale=5.0,
        cfg_normalization=0.0,
        guidance_rescale=0.0,
    )
    inputs = dict(
        transformer=dit,
        batch=batch,
        server_args=args,
        step_index=0,
        video_input=torch.randn(1, 2, 2, 2, 4, device="cuda", dtype=torch.bfloat16),
        audio_input=torch.randn(1, 5, 4, device="cuda", dtype=torch.bfloat16),
        t_expand=torch.tensor([500.0], device="cuda"),
        visual_rope_pos=[torch.arange(2, device="cuda") for _ in range(3)],
        scale_factor=(1.0, 1.0, 1.0),
        visual_token_type_ids=None,
    )

    def policy(text_len):
        return CFGPolicy(
            branches=[
                CFGBranch(
                    name,
                    conditional,
                    {
                        "encoder_hidden_states": torch.randn(
                            1, text_len, 8, device="cuda", dtype=torch.bfloat16
                        ),
                        "pooled_projections": torch.randn(
                            1, 8, device="cuda", dtype=torch.bfloat16
                        ),
                        "text_rope_pos": torch.arange(text_len, device="cuda"),
                    },
                )
                for name, conditional in (
                    ("conditional", True),
                    ("unconditional", False),
                )
                if conditional or use_cfg
            ]
        )

    policies = [policy(3), policy(3), policy(5)]
    references = [
        stage._predict_joint_velocity(cfg_policy=p, **inputs) for p in policies
    ]
    args.enable_breakable_cuda_graph = True
    batch.is_warmup = True
    stage._predict_joint_velocity(cfg_policy=policies[0], **inputs)
    batch.is_warmup = False
    runner = stage._bcg_runners[id(dit)]
    replay = runner.replay
    replay_calls = 0

    def record_replay(entry, kwargs):
        nonlocal replay_calls
        replay_calls += 1
        return replay(entry, kwargs)

    monkeypatch.setattr(runner, "replay", record_replay)
    try:
        assert len(runner.entries) == 1, "eager fallback is not a capture test"
        assert all(entry.num_segments > 0 for entry in runner.entries.values())
        outputs = [
            stage._predict_joint_velocity(cfg_policy=p, **inputs) for p in policies
        ]
        assert replay_calls == (4 if use_cfg else 2)
        # validate after every replay: neither CFG nor subsequent requests may
        # overwrite a previously returned video/audio tensor
        for actual, expected in zip(outputs, references, strict=True):
            for tensor, reference in zip(actual, expected, strict=True):
                torch.testing.assert_close(tensor, reference, rtol=0, atol=0)
        assert len(runner.entries) == 1, "serving must not capture unseen text lengths"
    finally:
        runner.reset()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("separate_cfg", [False, True])
@torch.no_grad()
def test_bf16_joint_forward_and_cache_dit_lifetime(separate_cfg):
    config = Kandinsky6VideoAudioConfig()
    arch = _tiny_multimodal_arch_kwargs() | {"num_visual_blocks": 4}
    config.update_model_arch(arch)
    dit = Kandinsky6Transformer3DModel(config, {}).eval()
    _randomize(dit, seed=11)
    dit = dit.to(device="cuda", dtype=torch.bfloat16)
    torch.manual_seed(13)
    inputs = dict(
        hidden_states=torch.randn(1, 2, 2, 2, 4, device="cuda", dtype=torch.bfloat16),
        hidden_states_audio=torch.randn(1, 5, 4, device="cuda", dtype=torch.bfloat16),
        encoder_hidden_states=torch.randn(1, 3, 8, device="cuda", dtype=torch.bfloat16),
        pooled_projections=torch.randn(1, 8, device="cuda", dtype=torch.bfloat16),
        visual_rope_pos=[torch.arange(2, device="cuda") for _ in range(3)],
        text_rope_pos=torch.arange(3, device="cuda"),
    )

    def run_steps():
        outputs = []
        for step in range(3):
            for branch in range(2 if separate_cfg else 1):
                outputs.append(
                    dit(
                        **(
                            inputs
                            | {
                                "pooled_projections": inputs["pooled_projections"]
                                + branch
                            }
                        ),
                        timestep=torch.tensor([500.0 - step * 100], device="cuda"),
                    )
                )
        return outputs

    def assert_exact(actual, expected):
        for pair, reference in zip(actual, expected, strict=True):
            for tensor, target in zip(pair, reference, strict=True):
                torch.testing.assert_close(tensor, target, rtol=0, atol=0)

    reference = run_steps()
    for video, audio in reference:
        assert video.shape == inputs["hidden_states"].shape
        assert audio.shape == inputs["hidden_states_audio"].shape
        for tensor in (video, audio):
            assert tensor.dtype == torch.bfloat16
            assert torch.isfinite(tensor).all()
    enable_cache_on_transformer(
        dit,
        CacheDitConfig(
            enabled=True,
            num_inference_steps=3,
            Fn_compute_blocks=1,
            Bn_compute_blocks=1,
            max_warmup_steps=1,
            steps_computation_mask=[1, 1, 1],
            steps_computation_policy="static",
        ),
        has_separate_cfg=separate_cfg,
    )
    try:
        assert_exact(run_steps(), reference)
        cached_requests = []
        for _ in range(2):
            refresh_context_on_transformer(
                dit,
                3,
                steps_computation_mask=[1, 0, 1],
                steps_computation_policy="static",
            )
            outputs = run_steps()
            for video, audio in outputs:
                assert torch.isfinite(video).all()
                assert torch.isfinite(audio).all()
            assert dit._context_manager.get_cached_steps() == [1]
            if separate_cfg:
                assert dit._context_manager.get_cfg_cached_steps() == [1]
            cached_requests.append(outputs)
        assert_exact(*cached_requests)
    finally:
        disable_cache_on_transformer(dit)
    assert_exact(run_steps(), reference)
