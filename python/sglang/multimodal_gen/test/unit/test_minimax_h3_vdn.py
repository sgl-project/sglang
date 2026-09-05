# SPDX-License-Identifier: Apache-2.0
"""VDN-H3 (hybrid window-softmax + Video Delta linear attention MiniMax-H3):
registration, admission, and the linear branch's arithmetic against
step-by-step references (no weights, CPU + small CUDA shapes)."""

from __future__ import annotations

import math
import re
import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
    MiniMaxH3DiTArchConfig,
    MiniMaxH3DiTConfig,
    VDNHybridAttentionArchConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import (
    MiniMaxH3PipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3_vdn import (
    VDNH3PipelineConfig,
)
from sglang.multimodal_gen.configs.sample.minimax_h3_vdn import VDNH3SamplingParams
from sglang.multimodal_gen.registry import (
    get_model_info,
    get_non_diffusers_pipeline_name,
)
from sglang.multimodal_gen.runtime.models.dits.minimax_h3_vdn import (
    TEXT_STATE_SCALE,
    MiniMaxH3VDNLinearBranch,
    VDNH3Layout,
    delta_factor_apply,
    frame_statistics,
    gather_linear_state,
    run_scans,
    vdn_h3_layout_from_packed,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

VDN_MODEL_ID = "OpenVDN/vdn-minimax-h3"
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


# --------------------------------------------------------------------------
# registration and admission
# --------------------------------------------------------------------------


def test_registry_resolves_vdn_h3_configs() -> None:
    info = get_model_info(VDN_MODEL_ID)
    assert info.sampling_param_cls is VDNH3SamplingParams
    assert info.pipeline_config_cls is VDNH3PipelineConfig
    assert get_non_diffusers_pipeline_name(VDN_MODEL_ID) == "VDNH3Pipeline"
    assert get_non_diffusers_pipeline_name("/models/OpenVDN/vdn-minimax-h3") == (
        "VDNH3Pipeline"
    )
    # the base H3 detector must not swallow the VDN id (it contains "minimax-h3")
    base = get_model_info("MiniMaxAI/MiniMax-H3")
    assert base.pipeline_config_cls is MiniMaxH3PipelineConfig


def test_vdn_h3_sampling_defaults_and_rejections() -> None:
    params = VDNH3SamplingParams(prompt="p")
    assert params.num_inference_steps == 9  # 8 NFE
    with pytest.raises(ValueError, match="exactly nine sigma grid points"):
        VDNH3SamplingParams(prompt="p", num_inference_steps=8)
    with pytest.raises(ValueError, match="t2va only"):
        VDNH3SamplingParams(
            prompt="p",
            task="fl2va",
            conditions=[{"type": "image", "uri": "x.png", "role": "first_frame"}],
            target={"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5.0},
        )


def _server_args(**overrides) -> SimpleNamespace:
    args = dict(
        model_variant=None,
        attention_backend=None,
        component_attention_backends={},
        attention_backend_config=None,
        ring_degree=1,
        enable_torch_compile=False,
        enable_breakable_cuda_graph=False,
        quantization=None,
    )
    args.update(overrides)
    ns = SimpleNamespace(**args)
    ns.resolve_component_attention_backend = lambda name: (
        (
            AttentionBackendEnum[str(ns.component_attention_backends[name]).upper()]
            if name in ns.component_attention_backends
            else None
        ),
        None,
    )
    return ns


def test_vdn_h3_pipeline_config_rejections() -> None:
    config = VDNH3PipelineConfig()
    with pytest.raises(ValueError, match="--model-variant does not apply"):
        config.validate_server_args(_server_args(model_variant="ref2va"))
    with pytest.raises(
        ValueError, match="requires --attention-backend hybrid_window_attn_h3"
    ):
        config.validate_server_args(_server_args(attention_backend="fa"))
    with pytest.raises(ValueError, match="ring-degree"):
        config.validate_server_args(_server_args(ring_degree=2))
    with pytest.raises(ValueError, match="torch.compile"):
        config.validate_server_args(_server_args(enable_torch_compile=True))
    with pytest.raises(ValueError, match="breakable CUDA graph"):
        config.validate_server_args(_server_args(enable_breakable_cuda_graph=True))
    with pytest.raises(ValueError, match="no.*audited high-quality deployment"):
        config.validate_quality_deployment(server_args=None)
    args = _server_args()
    config.validate_server_args(args)
    assert args.attention_backend == "hybrid_window_attn_h3"


def test_vdn_h3_quantization_defaults_to_mxfp8_on_blackwell(monkeypatch) -> None:
    """Online MXFP8 is the default on SM100+ (SM120 included) and what `fp8`
    maps to there; `bf16` opts out; before SM100 the block-scaled GEMM does
    not exist, so an unset flag stays bf16 and `fp8` stays the per-channel
    path."""
    from sglang.multimodal_gen.configs.pipeline_configs import minimax_h3_vdn as module

    def resolved(
        quantization: str | None, blackwell: bool, sm120: bool = False
    ) -> str | None:
        monkeypatch.setattr(module.current_platform, "is_blackwell", lambda: blackwell)
        monkeypatch.setattr(module.current_platform, "is_sm120", lambda: sm120)
        args = _server_args(quantization=quantization)
        VDNH3PipelineConfig().validate_server_args(args)
        return args.quantization

    assert resolved(None, True) == "mxfp8"
    assert resolved("fp8", True) == "mxfp8"
    assert resolved("bf16", True) is None
    assert resolved(None, False) is None
    assert resolved("fp8", False) == "fp8"
    assert resolved(None, False, sm120=True) == "mxfp8"
    assert resolved("fp8", False, sm120=True) == "mxfp8"
    assert resolved("bf16", False, sm120=True) is None


def test_hybrid_arch_config_from_transform_config_and_mapping() -> None:
    transform = {
        "anchor_frames": "both",
        "enable_softmax_gate": True,
        "linear_attention": {
            "a_fp32": True,
            "bridge": "alpha",
            "delta_rule": "vdn_solve",
            "enable_text_state": True,
            "linear_head_dim": 128,
            "short_conv": {"targets": ["k", "v"]},
        },
        "softmax_attention": {"chunk": 5, "radius": 1},
    }
    dit = MiniMaxH3DiTConfig()
    dit.update_model_arch({"hybrid_attention": transform, "num_layers": 2})
    hybrid = dit.arch_config.hybrid_attention
    assert isinstance(hybrid, VDNHybridAttentionArchConfig)
    assert hybrid.short_conv == ("k", "v") and hybrid.chunk == 5
    assert MiniMaxH3DiTConfig().arch_config.hybrid_attention is None
    with pytest.raises(ValueError, match="delta_rule"):
        VDNHybridAttentionArchConfig(delta_rule="bogus")

    mapping = MiniMaxH3DiTArchConfig().param_names_mapping
    for source, expected in (
        (
            "transformer_blocks.7.attn.linear_attention.alpha.A_log",
            "blocks.7.attn.hybrid.linear_attention.alpha.A_log",
        ),
        (
            "transformer_blocks.7.attn.softmax_gate.up.bias",
            "blocks.7.attn.hybrid.softmax_gate.up.bias",
        ),
        (
            "transformer_blocks.7.attn.to_out_linear.weight",
            "blocks.7.attn.hybrid.to_out_linear.weight",
        ),
    ):
        targets = [
            re.sub(pattern, target if isinstance(target, str) else target[0], source)
            for pattern, target in mapping.items()
            if re.match(pattern, source)
        ]
        assert targets == [expected], (source, targets)


def test_layout_from_packed_t2va() -> None:
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_sequence import (
        minimax_h3_packed_sequence,
    )

    packed = minimax_h3_packed_sequence(
        text_len=70,
        latent_t=12,
        latent_h=12,
        latent_w=16,
        audio_t=50,
        include_keyframe_cond=False,
    )
    layout = vdn_h3_layout_from_packed(packed, latent_t=12, latent_h=12, latent_w=16)
    assert layout.text_len == 70
    assert layout.video_start == 70 + 100
    assert layout.tokens_per_frame == 48 and layout.frame_size == (6, 8)
    assert layout.used == 70 + 100 + 12 * 48
    assert layout.seq_len == int(packed["seq_len"]) and layout.seq_len % 64 == 0
    assert layout.global_ranges == [(0, 170)]


# --------------------------------------------------------------------------
# the linear branch arithmetic
# --------------------------------------------------------------------------

F_, H_, S_, D_ = 6, 2, 16, 32


def _random_stats(device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    k = torch.nn.functional.normalize(
        torch.randn(F_, H_, S_, D_, generator=g), dim=-1
    ).to(device)
    v = torch.randn(F_, H_, S_, D_, generator=g).to(device)
    beta = torch.rand(F_, H_, S_, generator=g).to(device)
    alpha = torch.rand(F_, H_, D_, generator=g).to(device) * 0.5 + 0.5
    return k, v, beta, alpha


def test_frame_statistics_and_delta_rule_match_dense_algebra() -> None:
    k, v, beta, alpha = _random_stats("cpu")
    A, B = frame_statistics(k, v, beta, a_fp32=True)
    A_ref = torch.einsum("fhsk,fhs,fhsl->fhkl", k, beta, k)
    B_ref = torch.einsum("fhsv,fhs,fhsk->fhvk", v, beta, k)
    assert torch.allclose(A, A_ref, atol=1e-4) and torch.allclose(B, B_ref, atol=1e-4)
    transition, injection = delta_factor_apply(
        "vdn_solve", alpha, A, B, tokens_per_frame=S_
    )
    inv = torch.linalg.inv(torch.eye(D_) + A)
    assert torch.allclose(transition, alpha.unsqueeze(-1) * inv, atol=1e-4)
    assert torch.allclose(injection, B @ inv, atol=1e-4)


def test_scans_match_step_reference_and_text_seed() -> None:
    k, v, beta, alpha = _random_stats("cpu", seed=1)
    A, B = frame_statistics(k, v, beta, a_fp32=True)
    transition, injection = delta_factor_apply(
        "vdn_solve", alpha, A, B, tokens_per_frame=S_
    )
    text_state = torch.randn(H_, D_, D_)
    prefix, suffix = run_scans(transition, injection, text_state)
    state = text_state.clone()
    for f in range(F_):
        state = state @ transition[f] + injection[f]
        assert torch.allclose(prefix[f], state, atol=1e-4)
    state = text_state.clone()
    for f in range(F_ - 1, -1, -1):
        state = state @ transition[f] + injection[f]
        assert torch.allclose(suffix[f], state, atol=1e-4)


@pytest.mark.parametrize("world", [1, 2, 5, 10])
def test_frame_partial_sums_match_index_add(world: int) -> None:
    """The Ulysses frame-mean partial sums (reshape-sum over whole frames plus
    two edge rows sums, deterministic) equal the index_add formulation."""
    from sglang.multimodal_gen.runtime.models.dits.minimax_h3_vdn_attention import (
        _vdn_frame_partial_sums,
    )

    frames, tpf, hidden = 12, 50, 64
    video_start = 20
    video_end = video_start + frames * tpf
    seq = video_end + 30  # 650 rows, divisible by every ``world`` above
    g = torch.Generator(device="cpu").manual_seed(0)
    x = torch.randn(seq, hidden, generator=g).to(torch.bfloat16)
    local = seq // world
    total = torch.zeros(frames, hidden)
    for rank in range(world):
        total += _vdn_frame_partial_sums(
            x[rank * local : (rank + 1) * local],
            row_start=rank * local,
            video_start=video_start,
            video_end=video_end,
            num_frames=frames,
            tokens_per_frame=tpf,
        )
    ref = x[video_start:video_end].float().view(frames, tpf, hidden).sum(1)
    torch.testing.assert_close(total, ref, rtol=1e-5, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("anchor_frames", ["both", "none"])
@pytest.mark.parametrize("reference", ["frame_chain_scans", "eager_kernels"])
def test_branch_forward_matches_reference(
    anchor_frames: str, reference: str, monkeypatch
) -> None:
    """The shipped branch (boundary scans, fused Triton kernels) against the
    same module with the plain frame-chain scans, and against the eager
    kernel chain; both across the anchor-frame shift of the chunk grid."""
    from sglang.multimodal_gen.runtime.models.dits import minimax_h3_vdn as module

    device = torch.device("cuda")
    hidden, heads, head_dim = 64, 4, 32
    num_frames, fh, fw = 13, 4, 6
    tpf = fh * fw
    hybrid = VDNHybridAttentionArchConfig(
        chunk=3, radius=1, anchor_frames=anchor_frames, linear_head_dim=head_dim
    )
    branch = _branch(hybrid, heads, hidden, head_dim, seed=0).to(device)
    layout = VDNH3Layout(
        seq_len=64 * 6,
        used=10 + num_frames * tpf,
        text_len=10,
        video_start=10,
        num_frames=num_frames,
        tokens_per_frame=tpf,
        frame_height=fh,
        frame_width=fw,
    )
    g = torch.Generator(device="cpu").manual_seed(1)
    V = num_frames * tpf
    q, k, v = (
        torch.randn(V, heads, head_dim, generator=g).to(device, torch.bfloat16)
        for _ in range(3)
    )
    tk, tv = (
        torch.randn(10, heads, head_dim, generator=g).to(device, torch.bfloat16)
        for _ in range(2)
    )
    x = torch.randn(V, hidden, generator=g).to(device, torch.bfloat16)
    tx = torch.randn(10, hidden, generator=g).to(device, torch.bfloat16)
    kwargs = dict(
        q_raw=q,
        k_raw=k,
        v_raw=v,
        beta=branch.beta(x),
        gate=branch.gate(x),
        frame_mean=x.view(num_frames, tpf, hidden).mean(1, dtype=torch.float32),
        layout=layout,
        text_k_raw=tk,
        text_v_raw=tv,
        text_beta=branch.beta(tx),
    )
    fast = branch(**kwargs)
    if reference == "frame_chain_scans":
        monkeypatch.setattr(
            module,
            "run_boundary_scans",
            lambda t, i, ts, *, chunk, frame_offset=0: run_scans(t, i, ts),
        )
    else:
        branch.fused_kernels = False
    expected = branch(**kwargs)
    assert expected.abs().sum() > 0
    # fused kernels skip the eager chain's bf16 roundings; the scans re-associate fp32
    tolerance = {"frame_chain_scans": 5e-3, "eager_kernels": 2e-2}[reference]
    rel = (fast.float() - expected.float()).norm() / expected.float().norm()
    assert rel < tolerance, rel


def test_gather_is_the_exact_window_complement() -> None:
    """With alpha = 1 and one-hot frame indicators, the gathered state must
    be exactly the indicator of the frames outside the window."""
    num_frames = 9
    hybrid = VDNHybridAttentionArchConfig(chunk=3, radius=1, anchor_frames="none")
    bounds = hybrid.window_bounds(num_frames)
    # injection[f] = one-hot(f) laid along dv; transition = identity
    eye = torch.eye(num_frames)
    injection = (
        eye.view(num_frames, 1, num_frames, 1)
        .expand(num_frames, 1, num_frames, 1)
        .clone()
    )
    transition = torch.eye(1).view(1, 1, 1, 1).expand(num_frames, 1, 1, 1).clone()
    prefix, suffix = run_scans(transition, injection, None)
    alpha = torch.ones(num_frames, 1, 1)
    gathered = gather_linear_state(
        prefix,
        suffix,
        alpha,
        bounds,
        bridge="alpha",
        text_state=None,
        out_dtype=torch.float32,
    )
    for t in range(num_frames):
        lo, hi = max(bounds[t][0], 0), min(bounds[t][1], num_frames - 1)
        expected = torch.tensor(
            [1.0 if (f < lo or f > hi) else 0.0 for f in range(num_frames)]
        )
        assert torch.equal(gathered[t, 0, :, 0], expected), (t, gathered[t, 0, :, 0])


def test_gather_text_state_decays_over_skipped_frames() -> None:
    """A clip-end frame reads the text state decayed by prod alpha over
    exactly the frames between the boundary and t (VDN's bridge indices)."""
    num_frames = 4
    bounds = [
        (t, t) for t in range(num_frames)
    ]  # radius 0: complement = everything else
    prefix = torch.zeros(num_frames, 1, 1, 1)
    suffix = torch.zeros(num_frames, 1, 1, 1)
    alpha = torch.tensor([0.5, 0.25, 0.5, 0.5]).view(num_frames, 1, 1)
    text_state = torch.ones(1, 1, 1)
    out = gather_linear_state(
        prefix,
        suffix,
        alpha,
        bounds,
        bridge="alpha",
        text_state=text_state,
        out_dtype=torch.float32,
    ).view(num_frames)
    # frame 0 reads the text state through alpha[0]; frame 3 through alpha[3]
    assert math.isclose(out[0].item(), 0.5, rel_tol=1e-6)
    assert math.isclose(out[3].item(), 0.5, rel_tol=1e-6)
    assert out[1].item() == 0.0, "both neighbours in range and zero"


def test_temporal_shift_features_match_conv1d() -> None:
    from sglang.multimodal_gen.runtime.models.dits.minimax_h3_vdn import _temporal_shift

    x = torch.randn(7, 5, 6)  # [F, S, C]
    w = torch.randn(6, 5)
    got = _temporal_shift(x, w)
    ref = (
        torch.nn.functional.conv1d(
            x.permute(1, 2, 0).reshape(5, 6, 7), w.view(6, 1, 5), padding=2, groups=6
        )
        .reshape(5, 6, 7)
        .permute(2, 0, 1)
    )
    assert torch.allclose(got, ref, atol=1e-5)


def _branch(
    hybrid: VDNHybridAttentionArchConfig,
    heads: int,
    hidden: int,
    head_dim: int,
    seed: int,
):
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        maybe_init_distributed_environment_and_model_parallel,
        model_parallel_is_initialized,
    )
    from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
        ensure_distributed_env_defaults,
    )

    if not model_parallel_is_initialized():
        ensure_distributed_env_defaults()
        maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)
    arch = MiniMaxH3DiTArchConfig(
        num_attention_heads=heads, attention_head_dim=head_dim, hidden_size=hidden
    )
    branch = MiniMaxH3VDNLinearBranch(arch, hybrid, local_heads=heads)
    g = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for name, p in branch.named_parameters():
            if name.endswith("A_log"):
                p.copy_(torch.log(torch.empty_like(p).uniform_(1, 16, generator=g)))
            elif name.endswith("dt_bias"):
                p.copy_(torch.randn(p.shape, generator=g) - 3)
            elif name.endswith("norm.weight"):
                p.copy_(torch.ones_like(p))
            else:
                p.copy_(
                    torch.randn(p.shape, generator=g, dtype=torch.float32).to(p.dtype)
                    * 0.1
                )
    return branch


@requires_cuda
def test_branch_head_slice_equals_full_run() -> None:
    """The Ulysses contract: the branch is per-head independent given
    (beta, gate, alpha), so a head-sliced run equals the full run's slice."""
    device = torch.device("cuda")
    hidden, heads, head_dim = 64, 4, 32
    num_frames, fh, fw = 7, 4, 6
    tpf = fh * fw
    hybrid = VDNHybridAttentionArchConfig(
        chunk=2, radius=1, anchor_frames="both", linear_head_dim=head_dim
    )
    branch = _branch(hybrid, heads, hidden, head_dim, seed=0).to(device)
    layout = VDNH3Layout(
        seq_len=64 * 4,
        used=10 + num_frames * tpf,
        text_len=10,
        video_start=10,
        num_frames=num_frames,
        tokens_per_frame=tpf,
        frame_height=fh,
        frame_width=fw,
    )
    g = torch.Generator(device="cpu").manual_seed(1)
    V = num_frames * tpf
    q, k, v = (
        torch.randn(V, heads, head_dim, generator=g).to(device, torch.bfloat16)
        for _ in range(3)
    )
    tk, tv = (
        torch.randn(10, heads, head_dim, generator=g).to(device, torch.bfloat16)
        for _ in range(2)
    )
    x = torch.randn(V, hidden, generator=g).to(device, torch.bfloat16)
    tx = torch.randn(10, hidden, generator=g).to(device, torch.bfloat16)
    beta, gate = branch.beta(x), branch.gate(x)
    tbeta = branch.beta(tx)
    frame_mean = x.view(num_frames, tpf, hidden).mean(1, dtype=torch.float32)
    full = branch(
        q_raw=q,
        k_raw=k,
        v_raw=v,
        beta=beta,
        gate=gate,
        frame_mean=frame_mean,
        layout=layout,
        text_k_raw=tk,
        text_v_raw=tv,
        text_beta=tbeta,
    ).view(V, heads, head_dim)
    # anchors read zero
    assert torch.all(full[:tpf] == 0) and torch.all(full[-tpf:] == 0)
    assert full[tpf:-tpf].abs().sum() > 0

    # the same module on a head range of the full sequence
    hs = slice(1, 3)
    part = branch(
        q_raw=q[:, hs],
        k_raw=k[:, hs],
        v_raw=v[:, hs],
        beta=beta[:, hs],
        gate=gate[:, hs],
        frame_mean=frame_mean,
        layout=layout,
        text_k_raw=tk[:, hs],
        text_v_raw=tv[:, hs],
        text_beta=tbeta[:, hs],
        heads=hs,
    ).view(V, 2, head_dim)
    diff = (part.float() - full[:, hs].float()).abs().max().item()
    assert diff < 2e-2, f"head slice vs full run max diff {diff}"


@requires_cuda
def test_branch_matches_eager_reference_algorithm() -> None:
    """The module against a from-scratch spelling of VDN's _readout (no
    skip_ends), including the text state seed."""
    device = torch.device("cuda")
    hidden, heads, head_dim = 48, 2, 32
    num_frames, fh, fw = 5, 3, 4
    tpf = fh * fw
    hybrid = VDNHybridAttentionArchConfig(
        chunk=0, radius=1, anchor_frames="none", linear_head_dim=head_dim, short_conv=()
    )
    branch = _branch(hybrid, heads, hidden, head_dim, seed=2).to(device)
    layout = VDNH3Layout(
        seq_len=256,
        used=8 + num_frames * tpf,
        text_len=8,
        video_start=8,
        num_frames=num_frames,
        tokens_per_frame=tpf,
        frame_height=fh,
        frame_width=fw,
    )
    g = torch.Generator(device="cpu").manual_seed(3)
    V = num_frames * tpf
    q, k, v = (
        torch.randn(V, heads, head_dim, generator=g).to(device, torch.bfloat16)
        for _ in range(3)
    )
    tk, tv = (
        torch.randn(8, heads, head_dim, generator=g).to(device, torch.bfloat16)
        for _ in range(2)
    )
    x = torch.randn(V, hidden, generator=g).to(device, torch.bfloat16)
    tx = torch.randn(8, hidden, generator=g).to(device, torch.bfloat16)
    beta, gate, tbeta = branch.beta(x), branch.gate(x), branch.beta(tx)
    frame_mean = x.view(num_frames, tpf, hidden).mean(1, dtype=torch.float32)
    got = branch(
        q_raw=q,
        k_raw=k,
        v_raw=v,
        beta=beta,
        gate=gate,
        frame_mean=frame_mean,
        layout=layout,
        text_k_raw=tk,
        text_v_raw=tv,
        text_beta=tbeta,
    )

    # reference in fp32
    def feat(t, l2):
        y = torch.nn.functional.silu(t.float())
        return torch.nn.functional.normalize(y, dim=-1, eps=1e-6) if l2 else y

    qf, kf, vf = feat(q, True), feat(k, True), feat(v, False)
    bounds = hybrid.window_bounds(num_frames)
    kb = kf.view(num_frames, tpf, heads, head_dim).permute(0, 2, 1, 3)
    vb = vf.view(num_frames, tpf, heads, head_dim).permute(0, 2, 1, 3)
    bb = beta.float().view(num_frames, tpf, heads).permute(0, 2, 1)
    A = torch.einsum("fhsk,fhs,fhsl->fhkl", kb, bb, kb)
    B = torch.einsum("fhsv,fhs,fhsk->fhvk", vb, bb, kb)
    inv = torch.linalg.inv(torch.eye(head_dim, device=device) + A)
    alpha = branch.alpha(frame_mean)
    trans = alpha.unsqueeze(-1) * inv
    inj = B @ inv
    # text state
    tkf, tvf = feat(tk, True), feat(tv, False)
    tkb = tkf.view(1, 8, heads, head_dim).permute(0, 2, 1, 3)
    tvb = tvf.view(1, 8, heads, head_dim).permute(0, 2, 1, 3)
    tbb = tbeta.float().view(1, 8, heads).permute(0, 2, 1)
    tA = torch.einsum("fhsk,fhs,fhsl->fhkl", tkb, tbb, tkb)[0]
    tB = torch.einsum("fhsv,fhs,fhsk->fhvk", tvb, tbb, tkb)[0]
    text_state = TEXT_STATE_SCALE * (
        tB @ torch.linalg.inv(torch.eye(head_dim, device=device) + tA)
    )
    prefix, suffix = [], [None] * num_frames
    s = text_state.clone()
    for f in range(num_frames):
        s = s @ trans[f] + inj[f]
        prefix.append(s)
    s = text_state.clone()
    for f in range(num_frames - 1, -1, -1):
        s = s @ trans[f] + inj[f]
        suffix[f] = s
    outs = []
    for t in range(num_frames):
        lo, hi = bounds[t]
        left = prefix[lo - 1] if lo - 1 >= 0 else text_state
        right = suffix[hi + 1] if hi + 1 < num_frames else text_state
        a_before = torch.prod(alpha[max(lo, 0) : t + 1], dim=0)
        a_after = torch.prod(alpha[t : min(hi, num_frames - 1) + 1], dim=0)
        state = left * a_before.unsqueeze(1) + right * a_after.unsqueeze(1)
        qt = qf.view(num_frames, tpf, heads, head_dim)[t]  # [S, H, d]
        ro = torch.einsum("shk,hvk->shv", qt, state)
        ms = ro.pow(2).mean(-1, keepdim=True)
        ro = ro * torch.rsqrt(ms + branch.norm.eps) * branch.norm.weight.float()
        outs.append(ro)
    ref = (torch.cat(outs) * gate.float()).reshape(V, heads * head_dim)
    diff = (got.float() - ref).abs().max().item()
    scale = ref.abs().max().item()
    assert diff < 3e-2 * max(scale, 1.0), (
        f"branch vs reference max diff {diff} (scale {scale})"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
