from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401

from sglang.kernels.ops.attention.fla import chunk_delta_h, chunk_intra, kda
from sglang.kernels.ops.kimi_k3 import kimi_k3_tiny_gemm, mla_output_gate
from sglang.kernels.ops.speculative.dspark import dspark_verify_window
from sglang.srt.hardware_backend.npu.moe import topk as npu_topk
from sglang.srt.hardware_backend.npu.moe.activation import NPUSituDeepEPKernel
from sglang.srt.layers import attn_residual
from sglang.srt.layers.moe.moe_runner import ascend as ascend_moe_runner
from sglang.srt.models import kimi_k3
from sglang.srt.models.kimi_k3 import KimiK3MLP, KimiK3MoE, _add3
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=5, suite="stage-a-unit-test-npu")


def test_npu_grouped_topk_honors_renormalize(monkeypatch):
    captured = {}

    def fake_grouped_topk(*args, **kwargs):
        captured.update(kwargs)
        return (
            torch.tensor([[0.5, 0.5]], dtype=torch.float32),
            torch.tensor([[1, 2]], dtype=torch.int32),
            None,
        )

    monkeypatch.setattr(
        torch.ops.npu, "npu_moe_gating_top_k", fake_grouped_topk
    )
    monkeypatch.setattr(
        npu_topk,
        "get_global_expert_distribution_recorder",
        lambda: SimpleNamespace(on_select_experts=lambda **kwargs: None),
    )
    monkeypatch.setattr(
        npu_topk, "capture_routed_experts_if_allowed", lambda *args: None
    )
    config = SimpleNamespace(
        use_grouped_topk=True,
        renormalize=True,
        correction_bias=torch.zeros(4),
        scoring_func="sigmoid",
        top_k=2,
        topk_group=1,
        num_expert_group=1,
        apply_routed_scaling_factor_on_output=False,
        routed_scaling_factor=1.0,
    )

    npu_topk.fused_topk_npu(
        hidden_states=torch.zeros(1, 4),
        router_logits=torch.zeros(1, 4),
        topk_config=config,
    )

    assert captured["renorm"] == 1
    assert captured["norm_type"] == 1


def test_kda_small_grid_fusions_remain_cuda_only():
    assert torch.cuda.is_available() is False
    assert kda._use_small_grid_fusions(1) is False


def test_kda_inter_solve_skips_autotune_on_non_cuda_backends():
    configs = chunk_intra._inter_solve_configs()

    assert len(configs) == 1
    assert configs[0].kwargs == {"BK": 32, "BV": 64}


def test_kda_recompute_uses_original_fixed_npu_config_without_cuda():
    assert torch.cuda.is_available() is False
    configs = kda._recompute_w_u_configs()

    assert len(configs) == 1
    assert configs[0].kwargs == {"BK": 64, "BV": 64}
    assert kda._get_k3_recompute_w_u_config(
        torch.empty(0), None, None, 128, 128, 64
    ) == {"BK": 64, "BV": 64}


def test_kda_output_uses_fixed_npu_config_without_cuda():
    assert torch.cuda.is_available() is False
    configs = kda._chunk_gla_fwd_o_configs()

    assert len(configs) == 1
    assert configs[0].kwargs == {"BK": 64, "BV": 64}
    assert kda.chunk_gla_fwd_kernel_o.fn is kda._chunk_gla_fwd_kernel_o


def test_kda_npu_chunk_state_uses_kv_layout_and_updates_envelope_state():
    torch.manual_seed(1)
    B, T, H, K, V = 1, 64, 1, 128, 128
    k_cpu = (torch.randn(B, T, H, K) * 0.05).to(torch.bfloat16)
    u_cpu = (torch.randn(B, T, H, V) * 0.05).to(torch.bfloat16)
    k = k_cpu.npu()
    u = u_cpu.npu()
    w = torch.zeros_like(k)
    gk = torch.zeros(B, T, H, K, dtype=torch.float32, device="npu")

    # Select a layer from an envelope so stride(0) is larger than H*V*K,
    # matching the unified KDA state pool used by Kimi K3.
    envelope = torch.full(
        (3, 2, H, V, K), -7.0, dtype=torch.float32, device="npu"
    )
    initial_state = envelope[:, 1]
    initial_state[2].zero_()
    state_indices = torch.tensor([2], dtype=torch.long, device="npu")

    h, v_new = chunk_delta_h.chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        gk=gk,
        initial_state=initial_state,
        initial_state_indices=state_indices,
        use_exp2=True,
    )

    expected_state = torch.einsum(
        "bthv,bthk->bhvk", u_cpu.float(), k_cpu.float()
    )
    assert h.shape == (B, 1, H, K, V)
    torch.testing.assert_close(h.cpu(), torch.zeros_like(h.cpu()))
    torch.testing.assert_close(v_new.cpu(), u_cpu)
    torch.testing.assert_close(
        initial_state[2].cpu(), expected_state[0], rtol=0.05, atol=0.01
    )
    torch.testing.assert_close(
        envelope[:2].cpu(), torch.full_like(envelope[:2].cpu(), -7.0)
    )
    torch.testing.assert_close(
        envelope[2, 0].cpu(), torch.full_like(envelope[2, 0].cpu(), -7.0)
    )


def test_kda_npu_output_consumes_kv_chunk_state_without_transpose():
    torch.manual_seed(2)
    B, T, H, K, V = 1, 64, 1, 128, 128
    scale = 0.5
    q_cpu = (torch.randn(B, T, H, K) * 0.05).to(torch.bfloat16)
    h_cpu = (torch.randn(B, 1, H, K, V) * 0.05).to(torch.bfloat16)
    q = q_cpu.npu()
    h = h_cpu.npu()
    v = torch.zeros(B, T, H, V, dtype=torch.bfloat16, device="npu")
    g = torch.zeros(B, T, H, K, dtype=torch.float32, device="npu")
    A = torch.zeros(B, T, H, 64, dtype=torch.float32, device="npu")

    out = kda.chunk_gla_fwd_o_gk(q=q, v=v, g=g, A=A, h=h, o=v, scale=scale)

    expected = torch.einsum(
        "bthk,bnhkv->bthv", (q_cpu * scale).float(), h_cpu.float()
    ).to(torch.bfloat16)
    torch.testing.assert_close(out.cpu(), expected, rtol=0.05, atol=0.01)


def test_kda_npu_output_varlen_matches_full_model_warmup_shape():
    B, T, H, K, V = 1, 768, 6, 128, 128
    q = torch.zeros(B, T, H, K, dtype=torch.bfloat16, device="npu")
    h = torch.zeros(B, T // 64, H, K, V, dtype=torch.bfloat16, device="npu")
    v = torch.zeros(B, T, H, V, dtype=torch.bfloat16, device="npu")
    g = torch.zeros(B, T, H, K, dtype=torch.float32, device="npu")
    A = torch.zeros(B, T, H, 64, dtype=torch.float32, device="npu")
    cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device="npu")

    out = kda.chunk_gla_fwd_o_gk(
        q=q,
        v=v,
        g=g,
        A=A,
        h=h,
        o=v,
        scale=0.5,
        cu_seqlens=cu_seqlens,
    )

    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), torch.zeros_like(out.cpu()))


def test_kda_npu_chunk_state_varlen_matches_full_model_warmup_shape():
    B, T, H, K, V = 1, 768, 6, 128, 128
    k = torch.zeros(B, T, H, K, dtype=torch.bfloat16, device="npu")
    u = torch.zeros(B, T, H, V, dtype=torch.bfloat16, device="npu")
    w = torch.zeros_like(k)
    gk = torch.zeros(B, T, H, K, dtype=torch.float32, device="npu")
    envelope = torch.zeros(2, 2, H, V, K, dtype=torch.float32, device="npu")
    initial_state = envelope[:, 1]
    state_indices = torch.tensor([1], dtype=torch.long, device="npu")
    cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device="npu")

    h, v_new = chunk_delta_h.chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        gk=gk,
        initial_state=initial_state,
        initial_state_indices=state_indices,
        cu_seqlens=cu_seqlens,
        use_exp2=True,
    )

    torch.npu.synchronize()
    assert h.shape == (B, T // 64, H, K, V)
    torch.testing.assert_close(h.cpu(), torch.zeros_like(h.cpu()))
    torch.testing.assert_close(v_new.cpu(), torch.zeros_like(v_new.cpu()))
    torch.testing.assert_close(envelope.cpu(), torch.zeros_like(envelope.cpu()))


def test_kda_npu_recompute_varlen_matches_full_model_warmup_shape():
    B, T, H, K, V = 1, 768, 6, 128, 128
    k = torch.zeros(B, T, H, K, dtype=torch.bfloat16, device="npu")
    v = torch.zeros(B, T, H, V, dtype=torch.bfloat16, device="npu")
    beta = torch.zeros(B, T, H, dtype=torch.bfloat16, device="npu")
    A = torch.zeros(B, T, H, 64, dtype=torch.bfloat16, device="npu")
    gk = torch.zeros(B, T, H, K, dtype=torch.float32, device="npu")
    cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device="npu")

    w, u, kg = kda.recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        gk=gk,
        cu_seqlens=cu_seqlens,
    )

    torch.npu.synchronize()
    torch.testing.assert_close(w.cpu(), torch.zeros_like(w.cpu()))
    torch.testing.assert_close(u.cpu(), torch.zeros_like(u.cpu()))
    torch.testing.assert_close(kg.cpu(), torch.zeros_like(kg.cpu()))


def test_kda_npu_full_prefill_varlen_matches_full_model_warmup_shape():
    B, T, H, K, V = 1, 768, 6, 128, 128
    q = torch.zeros(B, T, H, K, dtype=torch.bfloat16, device="npu")
    k = torch.zeros_like(q)
    v = torch.zeros(B, T, H, V, dtype=torch.bfloat16, device="npu")
    g = torch.zeros_like(q)
    beta = torch.zeros(B, T, H, dtype=torch.bfloat16, device="npu")
    A_log = torch.zeros(H, dtype=torch.float32, device="npu")
    dt_bias = torch.zeros(H, K, dtype=torch.float32, device="npu")
    envelope = torch.zeros(2, 2, H, V, K, dtype=torch.float32, device="npu")
    initial_state = envelope[:, 1]
    state_indices = torch.tensor([1], dtype=torch.long, device="npu")
    cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device="npu")

    out = kda.chunk_kda_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=K**-0.5,
        initial_state=initial_state,
        initial_state_indices=state_indices,
        cu_seqlens=cu_seqlens,
        A_log=A_log,
        dt_bias=dt_bias,
    )

    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), torch.zeros_like(out.cpu()))
    torch.testing.assert_close(envelope.cpu(), torch.zeros_like(envelope.cpu()))


def test_kda_npu_full_prefill_matches_sequential_recurrence():
    """Nonzero prefill must preserve the public VxK state contract."""
    torch.manual_seed(3)
    B, T, H, K, V = 1, 64, 1, 128, 128
    scale = K**-0.5
    q_cpu = (torch.randn(B, T, H, K) * 0.05).to(torch.bfloat16)
    k_cpu = (torch.randn_like(q_cpu) * 0.05).to(torch.bfloat16)
    v_cpu = (torch.randn(B, T, H, V) * 0.05).to(torch.bfloat16)
    raw_gate_cpu = (torch.randn(B, T, H, K) * 0.2).to(torch.bfloat16)
    beta_cpu = torch.sigmoid(torch.randn(B, T, H) * 0.2).to(torch.bfloat16)
    a_log_cpu = torch.randn(H, dtype=torch.float32) * 0.05
    dt_bias_cpu = torch.randn(H, K, dtype=torch.float32) * 0.05
    initial_cpu = torch.randn(H, V, K, dtype=torch.float32) * 0.01

    q = q_cpu.npu()
    k = k_cpu.npu()
    v = v_cpu.npu()
    raw_gate = raw_gate_cpu.npu()
    beta = beta_cpu.npu()
    a_log = a_log_cpu.npu()
    dt_bias = dt_bias_cpu.npu()
    envelope = torch.full(
        (3, 2, H, V, K), -7.0, dtype=torch.float32, device="npu"
    )
    initial_state = envelope[:, 1]
    initial_state[2].copy_(initial_cpu)
    state_indices = torch.tensor([2], dtype=torch.long, device="npu")

    out = kda.chunk_kda(
        q=q,
        k=k,
        v=v,
        g=raw_gate,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        initial_state_indices=state_indices,
        use_qk_l2norm_in_kernel=True,
        A_log=a_log,
        dt_bias=dt_bias,
        lower_bound=-5.0,
    )
    torch.npu.synchronize()

    q_ref = q_cpu.float()
    q_ref /= torch.sqrt(torch.sum(q_ref * q_ref, dim=-1, keepdim=True)) + 1e-6
    k_ref = k_cpu.float()
    k_ref /= torch.sqrt(torch.sum(k_ref * k_ref, dim=-1, keepdim=True)) + 1e-6
    log_gate = -5.0 * torch.sigmoid(
        torch.exp(a_log_cpu).view(1, 1, H, 1)
        * (raw_gate_cpu.float() + dt_bias_cpu.view(1, 1, H, K))
    )
    state_ref = initial_cpu.clone()
    out_ref = torch.empty(B, T, H, V, dtype=torch.float32)
    for token in range(T):
        state_ref *= torch.exp(log_gate[0, token]).unsqueeze(-2)
        predicted = torch.einsum(
            "hvk,hk->hv", state_ref, k_ref[0, token]
        )
        delta = (v_cpu[0, token].float() - predicted) * beta_cpu[
            0, token
        ].float().unsqueeze(-1)
        state_ref += torch.einsum("hv,hk->hvk", delta, k_ref[0, token])
        out_ref[0, token] = torch.einsum(
            "hvk,hk->hv", state_ref, q_ref[0, token] * scale
        )

    torch.testing.assert_close(
        out.cpu().float(), out_ref, rtol=0.08, atol=0.025
    )
    torch.testing.assert_close(
        initial_state[2].cpu(), state_ref, rtol=0.08, atol=0.025
    )
    torch.testing.assert_close(
        envelope[:2].cpu(), torch.full_like(envelope[:2].cpu(), -7.0)
    )
    torch.testing.assert_close(
        envelope[2, 0].cpu(), torch.full_like(envelope[2, 0].cpu(), -7.0)
    )


def test_kda_gpu_autotune_search_spaces_remain_unchanged(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert len(chunk_intra._inter_solve_configs()) == 6
    assert len(kda._recompute_w_u_configs()) == 36
    assert len(kda._chunk_gla_fwd_o_configs()) == 9


def test_dspark_build_out_tokens_uses_0728_torch_path_on_npu(monkeypatch):
    assert torch.cuda.is_available() is False

    def fail_if_triton_is_used(cls, *args, **kwargs):
        raise AssertionError("Ascend must not compile the CUDA BuildOutTokens kernel")

    monkeypatch.setattr(
        dspark_verify_window.BuildOutTokens,
        "triton",
        classmethod(fail_if_triton_is_used),
    )
    out = dspark_verify_window.BuildOutTokens.execute(
        draft_tokens=torch.tensor([[10, 11, 12], [20, 21, 22]], device="npu"),
        correct_len=torch.tensor([1, 3], device="npu"),
        bonus=torch.tensor([99, 88], device="npu"),
        verify_num_draft_tokens=4,
        gamma=3,
    )

    torch.npu.synchronize()
    torch.testing.assert_close(
        out.cpu(), torch.tensor([[10, 99, 12, 0], [20, 21, 22, 88]])
    )


def test_attn_residual_uses_portable_fallback_without_cuda_capability():
    attn_residual._FAST_SUPPORTED = None
    assert attn_residual._use_fast(7168) is False


def test_attn_residual_empty_tokens_skip_triton_launch(monkeypatch):
    prefix = torch.empty((0, 7168), dtype=torch.bfloat16)
    bank = torch.empty((0, 8, 7168), dtype=torch.bfloat16)

    def fail_if_launched(*args, **kwargs):
        raise AssertionError("empty token batches must not launch Triton kernels")

    monkeypatch.setattr(attn_residual, "get_cw", fail_if_launched)
    out = attn_residual._mix_fused(prefix, bank, 1, None, None)

    assert out is prefix


def test_attn_residual_npu_fused_mix_matches_torch_definition():
    torch.manual_seed(4)
    tokens, hidden_size, num_valid_blocks = 2, 7168, 3
    prefix = (torch.randn(tokens, hidden_size) * 0.05).to(
        device="npu", dtype=torch.bfloat16
    )
    bank = (torch.randn(tokens, 5, hidden_size) * 0.05).to(
        device="npu", dtype=torch.bfloat16
    )
    proj = SimpleNamespace(
        weight=(torch.randn(1, hidden_size, device="npu") * 0.01).to(
            torch.bfloat16
        )
    )
    norm = SimpleNamespace(
        weight=(torch.randn(hidden_size, device="npu") * 0.01).to(
            torch.bfloat16
        ),
        variance_epsilon=1e-6,
    )

    actual = attn_residual._mix_fused(
        prefix, bank, num_valid_blocks, proj, norm
    )
    torch.npu.synchronize()

    rows = torch.cat(
        [bank[:, :num_valid_blocks].float(), prefix.float().unsqueeze(1)], dim=1
    )
    cw = norm.weight.float() * proj.weight.squeeze(0).float()
    scores = torch.sum(rows * cw, dim=-1) * torch.rsqrt(
        torch.mean(rows.square(), dim=-1) + norm.variance_epsilon
    )
    expected = torch.sum(
        torch.softmax(scores, dim=-1).unsqueeze(-1) * rows, dim=1
    )
    torch.testing.assert_close(
        actual.float(), expected, rtol=0.03, atol=0.01
    )


def test_kimi_k3_deepep_situ_quant_matches_definition():
    torch.manual_seed(5)
    rows, intermediate = 4, 3072
    x = (torch.randn(rows, 2 * intermediate) * 0.2).to(
        device="npu", dtype=torch.bfloat16
    )
    group_list = torch.tensor([1, 2, 1], dtype=torch.int64, device="npu")
    kernel = NPUSituDeepEPKernel(
        need_quant=True, beta=4.0, linear_beta=25.0
    )

    quantized, scale = kernel._apply_activation(x, group_list, 1)
    torch.npu.synchronize()

    gate, up = x.float().chunk(2, dim=-1)
    expected = (
        4.0
        * torch.tanh(gate / 4.0)
        * torch.sigmoid(gate)
        * (25.0 * torch.tanh(up / 25.0))
    )
    reconstructed = quantized.float() * scale.unsqueeze(-1)
    max_quant_error = float(scale.max().cpu()) * 1.1 + 1e-5
    torch.testing.assert_close(
        reconstructed, expected, rtol=0.03, atol=max_quant_error
    )


def test_kimi_k3_deepep_runner_selects_situ(monkeypatch):
    backend = SimpleNamespace(is_deepep=lambda: True)
    monkeypatch.setattr(
        ascend_moe_runner, "get_moe_a2a_backend", lambda: backend
    )
    config = ascend_moe_runner.MoeRunnerConfig(
        activation="situ",
        gemm1_alpha=4.0,
        gemm1_clamp_limit=25.0,
        layer=SimpleNamespace(
            w2_kernel=ascend_moe_runner.NPUW4A8Int8MoEMethod()
        ),
    )

    runner = ascend_moe_runner.AscendRunnerCore(config)

    assert isinstance(runner.activation, NPUSituDeepEPKernel)
    assert runner.activation.need_quant is True
    assert runner.activation.beta == 4.0
    assert runner.activation.linear_beta == 25.0


def test_kimi_k3_moe_empty_tokens_still_enter_ep_experts(monkeypatch):
    moe = KimiK3MoE.__new__(KimiK3MoE)
    torch.nn.Module.__init__(moe)
    moe.shared_experts = None
    moe._sbo_shared_overlap = False
    moe.use_latent_moe = True
    moe.moe_hidden_size = 8
    moe._use_mega_moe = False
    moe.gate = lambda x: x.new_empty((0, 4))
    moe.topk = lambda x, logits: object()

    def fail_if_projected(*args, **kwargs):
        raise AssertionError("empty token batches must not launch projections")

    moe.routed_expert_down_proj = fail_if_projected
    moe.routed_expert_up_proj = fail_if_projected
    expert_calls = []

    def enter_ep_experts(x, topk_output):
        expert_calls.append(x.shape)
        return x

    moe.experts = enter_ep_experts
    monkeypatch.setattr(moe, "_ep_front", lambda x: None)
    monkeypatch.setattr(moe, "_ep_front_overlap", lambda x: None)

    hidden_states = torch.empty((0, 16), dtype=torch.bfloat16)
    out = moe._forward_unfused(hidden_states, prefix_sum=None)

    assert expert_calls == [torch.Size([0, 8])]
    assert out.shape == hidden_states.shape


def test_kimi_k3_dense_mlp_can_keep_0728_attention_tp_layout(monkeypatch):
    created = {}

    class FakeColumn(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            created["column"] = kwargs

        def forward(self, x):
            return torch.cat((x, x), dim=-1), None

    class FakeRow(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            created["row"] = kwargs

        def forward(self, x):
            return x, None

    class FakeAct(torch.nn.Module):
        def forward(self, x):
            return x[..., : x.shape[-1] // 2]

    monkeypatch.setattr(kimi_k3, "_k3_dense_mlp_attn_tp", True)
    monkeypatch.setattr(kimi_k3, "is_dp_attention_enabled", lambda: True)
    monkeypatch.setattr(
        kimi_k3,
        "get_parallel",
        lambda: SimpleNamespace(attn_tp_rank=3, attn_tp_size=16),
    )
    monkeypatch.setattr(kimi_k3, "MergedColumnParallelLinear", FakeColumn)
    monkeypatch.setattr(kimi_k3, "RowParallelLinear", FakeRow)
    monkeypatch.setattr(kimi_k3, "SiluAndMul", FakeAct)

    mlp = KimiK3MLP(
        hidden_size=8,
        intermediate_size=8,
        hidden_act="silu",
    )
    out = mlp(torch.randn(2, 8), forward_batch=SimpleNamespace())

    assert mlp._dense_attn_tp is True
    assert created["column"]["tp_rank"] == 3
    assert created["column"]["tp_size"] == 16
    assert created["row"]["tp_rank"] == 3
    assert created["row"]["tp_size"] == 16
    assert created["row"]["use_dp_attention_reduce"] is True
    assert out.shape == (2, 8)


def test_kimi_k3_add3_uses_portable_fallback_for_non_cuda_tensors():
    a = torch.randn(2, 16, dtype=torch.bfloat16)
    b = torch.randn_like(a)
    c = torch.randn_like(a)

    torch.testing.assert_close(_add3(a, b, c), (a + b) + c)


def test_kimi_k3_add3_uses_portable_fallback_for_npu_tensors():
    from sglang.kernels.ops.elementwise import add3

    a = torch.randn(2, 16, device="npu", dtype=torch.bfloat16)
    b = torch.randn_like(a)
    c = torch.randn_like(a)

    assert a.device.type == "npu"
    assert not add3.covered(a, b, c)
    torch.testing.assert_close(_add3(a, b, c), (a + b) + c)


def test_kimi_k3_output_gate_cuda_jit_is_not_covered_on_npu():
    x = torch.randn(2, 16, device="npu", dtype=torch.bfloat16)
    gate = torch.randn_like(x)

    assert not mla_output_gate.covered(x, gate)


def test_kimi_k3_tiny_gemm_uses_torch_fallback_on_npu():
    x = torch.randn(2, 7168, device="npu", dtype=torch.bfloat16)
    weight = torch.randn(144, 7168, device="npu", dtype=torch.bfloat16)

    torch.testing.assert_close(
        kimi_k3_tiny_gemm(x, weight),
        torch.nn.functional.linear(x, weight),
    )
