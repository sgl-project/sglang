# SPDX-License-Identifier: Apache-2.0
"""VDN-H3 (hybrid window-softmax + Video Delta linear attention MiniMax-H3):
registration, admission, and the linear branch's arithmetic against
step-by-step references (no weights, CPU + small CUDA shapes)."""

from __future__ import annotations

import json
import math
import os
import re
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
    VDNH3PipelineConfig,
)
from sglang.multimodal_gen.configs.sample.minimax_h3 import VDNH3SamplingParams
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
    linear_features,
    run_boundary_scans,
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
    assert params.guidance_scale == 1.0
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


@requires_cuda
def test_vdn_h3_pipeline_config_forces_hybrid_backend_when_unset() -> None:
    """The runtime selector reads server_args.attention_backend; an unset
    backend must become the hybrid one, not the platform default."""
    config = VDNH3PipelineConfig()
    args = _server_args()
    config.validate_server_args(args)
    assert args.attention_backend == "hybrid_window_attn_h3"
    explicit = _server_args(attention_backend="hybrid_window_attn_h3")
    config.validate_server_args(explicit)
    assert explicit.attention_backend == "hybrid_window_attn_h3"


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
            "blocks.7.attn.linear_attention.alpha.A_log",
        ),
        (
            "transformer_blocks.7.attn.softmax_gate.up.bias",
            "blocks.7.attn.softmax_gate.up.bias",
        ),
        (
            "transformer_blocks.7.attn.to_out_linear.weight",
            "blocks.7.attn.to_out_linear.weight",
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
    # (I + A)^-1 is a contraction: eigenvalues in (0, 1]
    eig = torch.linalg.eigvalsh(inv)
    assert eig.max() <= 1.0 + 1e-5 and eig.min() > 0


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels need CUDA")
@pytest.mark.parametrize("with_conv", [False, True])
def test_linear_features_frame_major_layout(with_conv: bool) -> None:
    """The fused feature kernels can write [F, H, S, d] directly (the readout's
    bmm layout); it must equal the token-major result permuted."""
    from sglang.multimodal_gen.runtime.models.dits.minimax_h3_vdn import (
        VDNShortConv,
        linear_features,
    )

    device = torch.device("cuda")
    frames, fh, fw, heads, head_dim = 5, 4, 6, 3, 32
    per_frame = fh * fw
    g = torch.Generator(device="cpu").manual_seed(0)
    tokens = torch.randn(frames * per_frame, heads, head_dim, generator=g).to(
        device, torch.bfloat16
    )
    conv = None
    if with_conv:
        conv = VDNShortConv(heads * head_dim, ("q",)).to(device)
        with torch.no_grad():
            for p in conv.parameters():
                p.copy_(torch.randn(p.shape, generator=g).to(p.dtype) * 0.2)
    kwargs = dict(proj="q", conv=conv, num_frames=frames, frame_size=(fh, fw))
    token_major = linear_features(tokens, **kwargs)
    frame_major = linear_features(tokens, frame_major=True, **kwargs)
    assert frame_major.shape == (frames, heads, per_frame, head_dim)
    assert frame_major.is_contiguous()
    expected = token_major.view(frames, per_frame, heads, head_dim).permute(0, 2, 1, 3)
    assert torch.equal(frame_major, expected)


@pytest.mark.parametrize("world", [1, 2, 5, 10])
def test_frame_partial_sums_match_index_add(world: int) -> None:
    """The Ulysses frame-mean partial sums (reshape-sum over whole frames plus
    two edge rows sums, deterministic) equal the index_add formulation."""
    from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
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


@pytest.mark.parametrize("chunk", [1, 2, 3, 5])
def test_boundary_scans_match_frame_chain_at_chunk_bounds(chunk: int) -> None:
    """The chunked gather reads prefix at chunk ends and suffix at chunk
    starts only; the batched per-chunk composition must reproduce the frame
    chain there (fp32, re-associated products)."""
    k, v, beta, alpha = _random_stats("cpu", seed=2)
    A, B = frame_statistics(k, v, beta, a_fp32=True)
    transition, injection = delta_factor_apply(
        "vdn_solve", alpha, A, B, tokens_per_frame=S_
    )
    text_state = torch.randn(H_, D_, D_)
    prefix, suffix = run_scans(transition, injection, text_state)
    b_prefix, b_suffix = run_boundary_scans(
        transition, injection, text_state, chunk=chunk
    )
    num_chunks = -(-F_ // chunk)
    ends = [min((c + 1) * chunk - 1, F_ - 1) for c in range(num_chunks)]
    starts = [c * chunk for c in range(num_chunks)]
    torch.testing.assert_close(b_prefix[ends], prefix[ends], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(b_suffix[starts], suffix[starts], rtol=1e-4, atol=1e-4)
    # everything the gather never reads is zero, not garbage
    unread = sorted(set(range(F_)) - set(ends))
    assert torch.all(b_prefix[unread] == 0)
    unread = sorted(set(range(F_)) - set(starts))
    assert torch.all(b_suffix[unread] == 0)


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
    # frame 0: before-side reads text over [0..0] -> 0.5; after-side reads
    # suffix[1] (zero) bridged (no text substitution since 1 < F)
    assert math.isclose(out[0].item(), 0.5, rel_tol=1e-6)
    # frame 3 (last): after-side reads text over [3..3] -> 0.5; before-side
    # reads prefix[2] = 0
    assert math.isclose(out[3].item(), 0.5, rel_tol=1e-6)
    # frame 1: both neighbours in range and zero
    assert out[1].item() == 0.0


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
    feat = linear_features(
        torch.randn(10, 2, 8), proj="q", conv=None, num_frames=None, frame_size=None
    )
    assert torch.allclose(feat.norm(dim=-1), torch.ones(10, 2), atol=1e-4)


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

    # the Ulysses contract: the same module on a head range of the full
    # sequence (sliced q/k/v/beta/gate, per-head params sliced inside)
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


@requires_cuda
def test_fused_branch_kernels_match_eager_chain() -> None:
    """Each fused Triton stage against the eager spelling it replaces (one
    rounding instead of one per op: ~1 bf16 ulp), and the whole branch with
    fused kernels vs the eager chain."""
    from sglang.kernels.ops.diffusion import (
        vdn_frame_stats_prep,
        vdn_linear_epilogue,
        vdn_silu_l2norm,
        vdn_temporal_conv_act,
    )
    from sglang.multimodal_gen.runtime.models.dits import minimax_h3_vdn as vdn

    device = torch.device("cuda")
    g = torch.Generator(device="cpu").manual_seed(4)
    F_, S_, H_, D_ = 6, 24, 3, 32
    x = torch.randn(F_, S_, H_ * D_, generator=g).to(device, torch.bfloat16)
    w = (torch.randn(H_ * D_, 5, generator=g) * 0.4).to(device, torch.bfloat16)
    got = vdn_temporal_conv_act(x, w, H_, D_, True)
    ref = vdn._activate(vdn._temporal_shift(x, w).reshape(-1, H_, D_), True)
    assert (got.float() - ref.float()).abs().max().item() < 2e-2

    tokens = torch.randn(F_ * S_, 3 * H_ * D_, generator=g).to(device, torch.bfloat16)
    strided = tokens[:, : H_ * D_].view(F_ * S_, H_, D_)  # a q view of a fused qkv
    got = vdn_silu_l2norm(strided, True)
    ref = vdn._activate(strided, True)
    assert got.is_contiguous() and (got.float() - ref.float()).abs().max().item() < 2e-2
    got_v = vdn_silu_l2norm(strided, False)
    assert torch.allclose(
        got_v.float(), torch.nn.functional.silu(strided.float()), atol=2e-2
    )

    key = torch.randn(F_ * S_, H_, D_, generator=g).to(device, torch.bfloat16)
    value = torch.randn(F_ * S_, H_, D_, generator=g).to(device, torch.bfloat16)
    beta = torch.rand(F_ * S_, H_, generator=g).to(device, torch.bfloat16)
    k16, k32, kb32, vb = vdn_frame_stats_prep(key, value, beta, F_, S_)
    kf = key.view(F_, S_, H_, D_).permute(0, 2, 1, 3)
    vf = value.view(F_, S_, H_, D_).permute(0, 2, 1, 3)
    bf = beta.view(F_, S_, H_).permute(0, 2, 1)
    assert torch.equal(k16, kf.contiguous())
    assert torch.equal(k32, kf.float().contiguous())
    assert torch.equal(kb32, (kf.float() * bf.unsqueeze(-1).float()).contiguous())
    assert torch.equal(vb, (vf * bf.unsqueeze(-1).to(vf.dtype)).contiguous())

    readout = torch.randn(F_, H_, S_, D_, generator=g).to(device, torch.bfloat16)
    weight = (1 + 0.1 * torch.randn(D_, generator=g)).to(device, torch.bfloat16)
    gate = torch.rand(F_ * S_, H_, D_, generator=g).to(device, torch.bfloat16)
    got = vdn_linear_epilogue(readout, weight, gate, 1e-6)
    ref = vdn.linear_epilogue(readout, weight, gate, 1e-6)
    # one rounding vs one per op: within one bf16 ulp of the reference scale
    assert (got.float() - ref.float()).abs().max().item() <= 2e-2 * max(
        1.0, ref.float().abs().max().item()
    )

    # whole branch: fused vs eager
    hidden, heads, head_dim = 64, 4, 32
    num_frames, fh, fw = 7, 4, 6
    tpf = fh * fw
    hybrid = VDNHybridAttentionArchConfig(
        chunk=2, radius=1, anchor_frames="both", linear_head_dim=head_dim
    )
    branch = _branch(hybrid, heads, hidden, head_dim, seed=0).to(device)
    layout = VDNH3Layout(
        seq_len=256,
        used=10 + num_frames * tpf,
        text_len=10,
        video_start=10,
        num_frames=num_frames,
        tokens_per_frame=tpf,
        frame_height=fh,
        frame_width=fw,
    )
    V = num_frames * tpf
    q, k, v = (
        torch.randn(V, heads, head_dim, generator=g).to(device, torch.bfloat16)
        for _ in range(3)
    )
    tk, tv = (
        torch.randn(10, heads, head_dim, generator=g).to(device, torch.bfloat16)
        for _ in range(2)
    )
    xx = torch.randn(V, hidden, generator=g).to(device, torch.bfloat16)
    tx = torch.randn(10, hidden, generator=g).to(device, torch.bfloat16)
    args = dict(
        q_raw=q,
        k_raw=k,
        v_raw=v,
        beta=branch.beta(xx),
        gate=branch.gate(xx),
        frame_mean=xx.view(num_frames, tpf, hidden).mean(1, dtype=torch.float32),
        layout=layout,
        text_k_raw=tk,
        text_v_raw=tv,
        text_beta=branch.beta(tx),
    )
    vdn.set_fused_kernels_enabled(True)
    fused = branch(**args)
    vdn.set_fused_kernels_enabled(False)
    try:
        eager = branch(**args)
    finally:
        vdn.set_fused_kernels_enabled(True)
    diff = (fused.float() - eager.float()).abs().max().item()
    assert diff < 3e-2 * max(eager.abs().max().item(), 1.0), diff


@requires_cuda
def test_fused_gather_matches_eager_gather() -> None:
    from sglang.multimodal_gen.runtime.models.dits import minimax_h3_vdn as vdn

    g = torch.Generator(device="cpu").manual_seed(5)
    F_, H_, D_ = 9, 2, 32
    hybrid = VDNHybridAttentionArchConfig(chunk=3, radius=1, anchor_frames="none")
    bounds = hybrid.window_bounds(F_)
    prefix = torch.randn(F_, H_, D_, D_, generator=g).cuda()
    suffix = torch.randn(F_, H_, D_, D_, generator=g).cuda()
    alpha = (torch.rand(F_, H_, D_, generator=g) * 0.5 + 0.5).cuda()
    text = torch.randn(H_, D_, D_, generator=g).cuda()
    for ts in (None, text):
        for bridge in ("alpha", "none"):
            vdn.set_fused_kernels_enabled(False)
            try:
                ref = gather_linear_state(
                    prefix,
                    suffix,
                    alpha,
                    bounds,
                    bridge=bridge,
                    text_state=ts,
                    out_dtype=torch.float32,
                )
            finally:
                vdn.set_fused_kernels_enabled(True)
            got = gather_linear_state(
                prefix,
                suffix,
                alpha,
                bounds,
                bridge=bridge,
                text_state=ts,
                out_dtype=torch.float32,
            )
            assert torch.allclose(got, ref, atol=1e-5, rtol=1e-5), (bridge, ts is None)


@requires_cuda
def test_out_of_place_qknorm_rope_matches_inplace_and_keeps_inputs() -> None:
    from sglang.kernels.ops.diffusion import (
        can_use_fused_inplace_qknorm_rope,
        fused_inplace_qknorm_rope,
        fused_qknorm_rope_out_of_place,
    )

    T, H, D, R = 512, 4, 128, 96
    if not can_use_fused_inplace_qknorm_rope(
        D, R, True, torch.bfloat16, torch.bfloat16, True
    ):
        pytest.skip("fused qknorm+rope JIT kernel unavailable")
    g = torch.Generator(device="cpu").manual_seed(0)
    qkv = torch.randn(T, 3 * H * D, generator=g).to("cuda", torch.bfloat16)
    q = qkv[:, : H * D].view(T, H, D)  # strided views of the fused projection
    k = qkv[:, H * D : 2 * H * D].view(T, H, D)
    qw = (torch.rand(D, generator=g) + 0.5).to("cuda", torch.bfloat16)
    kw = (torch.rand(D, generator=g) + 0.5).to("cuda", torch.bfloat16)
    freqs = torch.randn(T, R // 2, generator=g).to("cuda")
    cache = torch.cat((freqs.cos(), freqs.sin()), -1).to(torch.bfloat16).contiguous()
    pos = torch.arange(T, device="cuda")
    kwargs = dict(
        is_neox=True, eps=1e-5, head_dim=D, rope_dim=R, round_norm_before_rope=True
    )
    q_ref, k_ref = q.clone(), k.clone()
    fused_inplace_qknorm_rope(q_ref, k_ref, qw, kw, cache, pos, **kwargs)
    q_out = torch.empty(T, H, D, device="cuda", dtype=torch.bfloat16)
    k_out = torch.empty_like(q_out)
    q_before = qkv.clone()
    fused_qknorm_rope_out_of_place(q, k, q_out, k_out, qw, kw, cache, pos, **kwargs)
    assert torch.equal(qkv, q_before)
    assert torch.equal(q_out, q_ref) and torch.equal(k_out, k_ref)


# --------------------------------------------------------------------------
# the overlay materializer's prefuse (fetched like the runtime fetches it)
# --------------------------------------------------------------------------


def _load_materializer():
    import importlib.util

    from sglang.multimodal_gen import envs
    from sglang.multimodal_gen.runtime.utils.model_overlay import (
        BUILTIN_MODEL_OVERLAY_REGISTRY,
    )

    spec = BUILTIN_MODEL_OVERLAY_REGISTRY[VDN_MODEL_ID]
    local = envs.SGLANG_DIFFUSION_TEST_VDN_H3_OVERLAY_DIR
    if local:
        path = os.path.join(local, "_overlay", "materialize.py")
    else:
        from huggingface_hub import hf_hub_download

        try:
            path = hf_hub_download(
                spec["overlay_repo_id"],
                "_overlay/materialize.py",
                revision=spec.get("overlay_revision"),
            )
        except Exception as exc:  # network / repo not yet pushed
            pytest.skip(f"overlay materializer unavailable: {exc}")
    module_spec = importlib.util.spec_from_file_location("_vdn_materializer", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def test_prefuse_folds_both_adapters_in_diffusers_layout(tmp_path) -> None:
    from safetensors.torch import save_file

    mat = _load_materializer()
    src = tmp_path / "src"
    base_t = src / "h3-base" / "transformer"
    base_t.mkdir(parents=True)
    ckpt = src / mat.VDN_CHECKPOINT
    torch.manual_seed(0)
    # a tiny "transformer": one attention projection (both adapters), one
    # adaln (turbo only, rank 2 pattern), one untouched norm
    weights = {
        "transformer_blocks.0.attn.to_q.weight": torch.randn(8, 6).to(torch.bfloat16),
        "transformer_blocks.0.adaln_proj.linear.weight": torch.randn(10, 4).to(
            torch.bfloat16
        ),
        "transformer_blocks.0.norm1.weight": torch.ones(6),
        "proj_in.weight": torch.randn(6, 3),
    }
    save_file(
        weights, str(base_t / "diffusion_pytorch_model-00001-of-00001.safetensors")
    )
    (base_t / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 1},
                "weight_map": {
                    k: "diffusion_pytorch_model-00001-of-00001.safetensors"
                    for k in weights
                },
            }
        )
    )
    (base_t / "config.json").write_text(
        json.dumps({"rope_freq_dim": 16, "_class_name": "X"})
    )
    adapters = {
        "default": (
            {"rank": 4, "alpha": 4, "targets": []},
            {
                "transformer_blocks.0.attn.orig.to_q.lora_A.default.weight": torch.randn(
                    4, 6
                ).to(torch.bfloat16),
                "transformer_blocks.0.attn.orig.to_q.lora_B.default.weight": torch.randn(
                    8, 4
                ).to(torch.bfloat16),
            },
        ),
        "turbo": (
            {
                "rank": 4,
                "alpha": 4,
                "rank_pattern": {"transformer_blocks.0.adaln_proj.linear": 2},
                "alpha_pattern": {"transformer_blocks.0.adaln_proj.linear": 2},
            },
            {
                "transformer_blocks.0.attn.orig.to_q.lora_A.turbo.weight": torch.randn(
                    4, 6
                ).to(torch.bfloat16),
                "transformer_blocks.0.attn.orig.to_q.lora_B.turbo.weight": torch.randn(
                    8, 4
                ).to(torch.bfloat16),
                "transformer_blocks.0.adaln_proj.linear.lora_A.turbo.weight": torch.randn(
                    2, 4
                ).to(torch.bfloat16),
                "transformer_blocks.0.adaln_proj.linear.lora_B.turbo.weight": torch.randn(
                    10, 2
                ).to(torch.bfloat16),
            },
        ),
    }
    for name, (cfg, tensors) in adapters.items():
        d = ckpt / "adapters" / name
        d.mkdir(parents=True)
        (d / "adapter_config.json").write_text(json.dumps({"config": cfg}))
        save_file(tensors, str(d / "adapter_model.safetensors"))
    branch = ckpt / "linear_branch"
    branch.mkdir()
    branch_tensors = {
        f"transformer_blocks.0.attn.linear_attention.p{i}": torch.zeros(1)
        for i in range(mat.EXPECTED_LINEAR_BRANCH_KEYS)
    }
    save_file(branch_tensors, str(branch / "model.safetensors"))
    (branch / "config.json").write_text(
        json.dumps(
            {
                "type": "hybrid_attention",
                "version": 2,
                "config": {"softmax_attention": {"chunk": 5, "radius": 1}},
            }
        )
    )
    (ckpt / "metadata.json").write_text(json.dumps({"stage": "dmd"}))

    out = tmp_path / "out"
    out.mkdir()
    mat.EXPECTED_PAIRS = {"default": 1, "turbo": 2}
    record = mat._prefuse_transformer(source_dir=str(src), output_dir=str(out))
    assert record["merge"]["pairs_merged"] == 3

    from safetensors import safe_open

    with safe_open(
        str(out / "transformer" / "diffusion_pytorch_model-00001-of-00001.safetensors"),
        "pt",
    ) as f:
        got_q = f.get_tensor("transformer_blocks.0.attn.to_q.weight")
        got_ada = f.get_tensor("transformer_blocks.0.adaln_proj.linear.weight")
        got_norm = f.get_tensor("transformer_blocks.0.norm1.weight")
    exp_q = weights["transformer_blocks.0.attn.to_q.weight"].float()
    for name in ("default", "turbo"):
        t = adapters[name][1]
        exp_q = (
            exp_q
            + t[f"transformer_blocks.0.attn.orig.to_q.lora_B.{name}.weight"].float()
            @ t[f"transformer_blocks.0.attn.orig.to_q.lora_A.{name}.weight"].float()
        )
    assert torch.equal(got_q, exp_q.to(torch.bfloat16))
    t = adapters["turbo"][1]
    exp_ada = weights["transformer_blocks.0.adaln_proj.linear.weight"].float() + (
        t["transformer_blocks.0.adaln_proj.linear.lora_B.turbo.weight"].float()
        @ t["transformer_blocks.0.adaln_proj.linear.lora_A.turbo.weight"].float()
    )
    assert torch.equal(got_ada, exp_ada.to(torch.bfloat16))
    assert torch.equal(got_norm, weights["transformer_blocks.0.norm1.weight"])
    index = json.loads(
        (
            out / "transformer" / "diffusion_pytorch_model.safetensors.index.json"
        ).read_text()
    )
    assert (
        index["weight_map"]["transformer_blocks.0.attn.linear_attention.p0"]
        == mat.LINEAR_BRANCH_FILE
    )
    config = json.loads((out / "transformer" / "config.json").read_text())
    assert config["hybrid_attention"]["softmax_attention"] == {"chunk": 5, "radius": 1}
    assert (out / "transformer" / mat.LINEAR_BRANCH_FILE).exists()
