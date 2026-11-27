"""M3.1 NVFP4 experts on flashinfer_trtllm_routed, MegaMoE prefill, and flashinfer_cutedsl."""

import socket
import sys
from types import SimpleNamespace
from typing import Optional
from unittest import mock

import pytest
import torch

# Import quantization before the MoE runners; the reverse is a circular import.
# isort: off
from sglang.srt.layers.quantization import minimax_m3_nvfp4
from sglang.srt.layers.moe import flashinfer_cutedsl_moe
from sglang.srt.layers.moe.moe_runner import flashinfer_cutedsl, flashinfer_trtllm

# isort: on
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.deepep import (
    DeepEPLLDispatchOutput,
    _quantize_per_token_nvfp4,
)
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
from sglang.srt.layers.quantization.utils import swizzle_blockscale
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

E, H, I, TOPK, T = 8, 1024, 512, 4, 96
ALPHA, LIMIT = 1.702, 7.0
_E2M1 = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]

requires_sm100 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs SM100",
)

requires_fused_masked_gemm = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not flashinfer_cutedsl_moe.supports_masked_gemm_fused_swiglu(
        torch.device("cuda")
    ),
    reason="needs SM100/SM103 and FlashInfer's fused masked GEMM epilogue",
)


def _dequant(q, sf, gs, shape):
    lut = torch.tensor(_E2M1, device=q.device)
    vals = torch.stack([lut[(q & 0xF).long()], lut[(q >> 4).long()]], -1)
    return vals.reshape(shape) * sf.float().repeat_interleave(16, -1) / gs


def _quantize(w):
    """Compressed-tensors NVFP4: packed e2m1, e4m3 block scales, encode global."""
    from flashinfer import fp4_quantize

    gs = (448 * 6 / w.abs().amax()).float().reshape(1)
    q, sf = fp4_quantize(w, gs, 16, False, False)
    sf = sf.view(torch.float8_e4m3fn).reshape(w.shape[0], -1)
    return q.view(torch.uint8), sf, gs, _dequant(q.view(torch.uint8), sf, gs, w.shape)


def _fake_quant_per_token(a):
    """What the kernel's dynamic per-token NVFP4 activation quant keeps of `a`."""
    return torch.cat([_quantize(row[None].bfloat16())[3] for row in a])


def _fp4_static(h, gs: float):
    from flashinfer import fp4_quantize

    q, sf = fp4_quantize(
        h.bfloat16(), torch.tensor([gs], device="cuda"), 16, False, False
    )
    return q.view(torch.uint8), sf.view(torch.float8_e4m3fn).reshape(h.shape[0], -1)


def _swigluoai(x):
    gate, up = x.chunk(2, -1)
    gate, up = gate.clamp(max=LIMIT), up.clamp(-LIMIT, LIMIT)
    return gate * torch.sigmoid(ALPHA * gate) * (up + 1)


def _build_layer(
    gate_up_scale_ratio: float = 1.0,
    w2_input_scale: float = 1.0,
):
    layer = torch.nn.Module()
    w13 = torch.randn(E, 2 * I, H, device="cuda", dtype=torch.bfloat16) / 4
    w2 = torch.randn(E, H, I, device="cuda", dtype=torch.bfloat16) / 8
    if gate_up_scale_ratio == 1.0:
        q13, s13, g13, d13 = zip(*(_quantize(w) for w in w13))
        w13_packed = torch.stack(q13)
        w13_scale = torch.stack(s13)
        w13_global_scale = torch.cat(g13)[:, None].repeat(1, 2)
        d13 = torch.stack(d13)
    else:
        # Distinct gate and up global scales catch a dropped alpha_up.
        w13[:, I:] *= gate_up_scale_ratio
        gate = [_quantize(w[:I]) for w in w13]
        up = [_quantize(w[I:]) for w in w13]
        w13_packed = torch.stack([torch.cat((g[0], u[0])) for g, u in zip(gate, up)])
        w13_scale = torch.stack([torch.cat((g[1], u[1])) for g, u in zip(gate, up)])
        w13_global_scale = torch.stack(
            [torch.cat((g[2], u[2])) for g, u in zip(gate, up)]
        )
        d13 = torch.stack([torch.cat((g[3], u[3])) for g, u in zip(gate, up)])
    q2, s2, g2, d2 = zip(*(_quantize(w) for w in w2))
    params = dict(
        w13_weight_packed=w13_packed,
        w13_weight_scale=w13_scale,
        w13_weight_global_scale=w13_global_scale,
        w13_input_global_scale=torch.full((E, 2), 1.0, device="cuda"),
        w2_weight_packed=torch.stack(q2),
        w2_weight_scale=torch.stack(s2),
        w2_weight_global_scale=torch.cat(g2),
        w2_input_global_scale=torch.full((E,), w2_input_scale, device="cuda"),
    )
    for name, t in params.items():
        layer.register_parameter(name, torch.nn.Parameter(t, requires_grad=False))
    layer.num_experts = layer.num_local_experts = E
    layer.moe_ep_rank = 0
    layer.intermediate_size_per_partition = I
    layer.routing_method_type = None
    layer._minimax_activation = "swigluoai"
    layer._mega_moe_trtllm_layout = False
    layer.dispatcher = mock.Mock()
    layer.moe_runner_config = MoeRunnerConfig(
        num_experts=E,
        num_local_experts=E,
        hidden_size=H,
        intermediate_size_per_partition=I,
        top_k=TOPK,
        gemm1_alpha=ALPHA,
        gemm1_beta=1.0,
        gemm1_clamp_limit=LIMIT,
        inplace=False,
    )
    return layer, d13, torch.stack(d2)


def _reference(
    x,
    ids,
    weights,
    d13,
    d2,
    w2_input_scale: Optional[float] = None,
):
    out = torch.zeros(x.shape, device="cuda", dtype=torch.float32)
    xq = _fake_quant_per_token(x)
    for k in range(TOPK):
        h = _swigluoai(torch.einsum("th,tnh->tn", xq, d13[ids[:, k]]))
        if w2_input_scale is None:
            h = _fake_quant_per_token(h)
        else:
            q, sf = _fp4_static(h, w2_input_scale)
            h = _dequant(q, sf, w2_input_scale, h.shape)
        out += weights[:, k : k + 1] * torch.einsum("ti,thi->th", h, d2[ids[:, k]])
    return out


def _scheme(runner, a2a):
    with (
        mock.patch.object(
            minimax_m3_nvfp4, "get_moe_runner_backend", return_value=runner
        ),
        mock.patch.object(minimax_m3_nvfp4, "get_moe_a2a_backend", return_value=a2a),
    ):
        return minimax_m3_nvfp4.MiniMaxM3Nvfp4MoE()


@requires_sm100
def test_trtllm_routed_matches_dequantized_reference():
    torch.manual_seed(0)
    backend = MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED
    with mock.patch.object(
        minimax_m3_nvfp4, "get_moe_runner_backend", return_value=backend
    ):
        scheme = minimax_m3_nvfp4.MiniMaxM3Nvfp4MoE()
    assert scheme.use_flashinfer_trtllm and not scheme.load_up_proj_weight_first
    layer, d13, d2 = _build_layer()
    scheme._process_trtllm_weights(layer)

    # Large enough activations that the SwiGLU clamp fires.
    x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
    ids = torch.stack([torch.randperm(E, device="cuda")[:TOPK] for _ in range(T)])
    weights = torch.rand(T, TOPK, device="cuda").softmax(-1)
    ref = _reference(x, ids, weights, d13, d2)
    pre_act = torch.einsum("th,tnh->tn", x.float(), d13[ids[:, 0]])
    assert (pre_act.abs() > LIMIT).float().mean() > 0.2

    captured = {}
    scheme.runner = SimpleNamespace(
        run=lambda d, q: captured.setdefault("q", q) and None
    )
    scheme.apply_weights(layer, None)
    dispatch = StandardDispatchOutput(
        hidden_states=x,
        hidden_states_scale=None,
        topk_output=StandardTopKOutput(
            topk_weights=weights, topk_ids=ids.int(), router_logits=None
        ),
    )
    with (
        mock.patch.object(flashinfer_trtllm, "is_allocation_symmetric", lambda: False),
        mock.patch(
            "sglang.srt.runtime_context.get_forward",
            lambda: SimpleNamespace(moe_output_buffer=torch.empty_like(x)),
        ),
    ):
        out = flashinfer_trtllm.fused_experts_none_to_flashinfer_trtllm_fp4(
            dispatch, captured["q"], layer.moe_runner_config, use_routed_topk=True
        ).hidden_states.float()

    # ~0.09 is the activation-rounding floor; a dropped clamp or beta costs >= 0.2.
    rel = ((out - ref).norm() / ref.norm()).item()
    fit = ((out * ref).sum() / (ref * ref).sum()).item()
    assert rel < 0.13 and abs(fit - 1) < 0.02, (rel, fit)


@pytest.fixture
def ep_group():
    import torch.distributed as dist

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    dist.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1
    )
    try:
        with get_parallel().override(
            moe_ep_group=SimpleNamespace(device_group=dist.group.WORLD)
        ):
            yield
    finally:
        dist.destroy_process_group()


@requires_sm100
def test_mega_moe_on_trtllm_weights_matches_canonical_mega_moe(ep_group):
    """MegaMoE prefill over TRT-LLM decode weights matches canonical MegaMoE."""
    from sglang.srt.layers.moe.mega_moe import run_mega_routed_experts

    canonical = _scheme(MoeRunnerBackend.DEEP_GEMM, MoeA2ABackend.MEGAMOE)
    hybrid = _scheme(MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED, MoeA2ABackend.MEGAMOE)
    assert not canonical.use_flashinfer_trtllm and hybrid.use_mega_moe_prefill
    torch.manual_seed(0)
    # Non-unit w2_input_scale so a dropped or double-applied fc2 input scale shows up.
    layer_canonical, _, _ = _build_layer(w2_input_scale=2.0)
    torch.manual_seed(0)
    layer_hybrid, _, _ = _build_layer(w2_input_scale=2.0)
    hybrid.create_moe_runner(layer_hybrid, layer_hybrid.moe_runner_config)
    canonical.process_weights_after_loading(layer_canonical)
    hybrid.process_weights_after_loading(layer_hybrid)
    assert layer_hybrid._mega_moe_trtllm_layout
    assert (
        layer_hybrid.mega_l1_weights[0].data_ptr() == layer_hybrid.w13_weight.data_ptr()
    )

    x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
    ids = torch.stack([torch.randperm(E, device="cuda")[:TOPK] for _ in range(T)])
    weights = torch.rand(T, TOPK, device="cuda").softmax(-1)

    def run(layer):
        return run_mega_routed_experts(
            layer,
            x,
            ids,
            weights,
            hidden_size=H,
            intermediate_size=I,
            top_k=TOPK,
            num_tokens=T,
            activation_clamp=LIMIT,
            activation="swigluoai",
        ).clone()

    y_canonical, y_hybrid = run(layer_canonical), run(layer_hybrid)
    assert y_canonical.abs().sum() > 0
    assert torch.equal(y_hybrid, y_canonical), (
        (y_hybrid.float() - y_canonical.float()).abs().max().item()
    )


def _ll_dispatch(x, ids, weights, capacity):
    """Single-rank stand-in for DeepEP low_latency_dispatch_fp4(swizzle_scales=True)."""
    q, sf, pts = _quantize_per_token_nvfp4(x)
    recv_x = torch.zeros(E, capacity, H // 2, dtype=torch.uint8, device="cuda")
    recv_sf = torch.zeros(E, capacity, H // 16, dtype=torch.uint8, device="cuda")
    # Garbage in unused rows must not leak into valid outputs.
    recv_pts = torch.full((E, capacity), float("nan"), device="cuda")
    counts = torch.zeros(E, dtype=torch.int32, device="cuda")
    slots = []
    for t in range(x.shape[0]):
        for k in range(TOPK):
            e = int(ids[t, k])
            r = int(counts[e])
            recv_x[e, r], recv_sf[e, r], recv_pts[e, r] = q[t], sf[t], pts[t]
            counts[e] += 1
            slots.append((t, k, e, r))
    rm, rk = capacity // 128, H // 64
    swizzled = (
        swizzle_blockscale(recv_sf.view(torch.float8_e4m3fn))
        .view(E, rm, rk, 32, 4, 4)
        .permute(3, 4, 1, 5, 2, 0)
    )
    dispatch = DeepEPLLDispatchOutput(
        hidden_states=recv_x,
        hidden_states_scale=swizzled,
        topk_ids=ids,
        topk_weights=weights,
        masked_m=counts,
        expected_m=capacity,
        per_token_scale=recv_pts,
    )
    return dispatch, slots


@requires_fused_masked_gemm
def test_cutedsl_masked_matches_dequantized_reference():
    torch.manual_seed(0)
    scheme = _scheme(MoeRunnerBackend.FLASHINFER_CUTEDSL, MoeA2ABackend.DEEPEP)
    assert scheme.use_flashinfer_cutedsl and not scheme.load_up_proj_weight_first
    layer, d13, d2 = _build_layer(gate_up_scale_ratio=3.0, w2_input_scale=2.0)
    scheme.process_weights_after_loading(layer)
    layer.dispatcher.set_quant_config.assert_called_once_with(
        {"dispatcher_output_dtype": "nvfp4", "nvfp4_per_token_activation": True}
    )

    x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
    # Uneven routing: expert E - 1 receives no tokens.
    ids = torch.stack(
        [torch.randperm(E - 1, device="cuda")[:TOPK] for _ in range(T)]
    ).int()
    weights = torch.rand(T, TOPK, device="cuda").softmax(-1)
    ref = _reference(x, ids, weights, d13, d2, w2_input_scale=2.0)
    pre_act = torch.einsum("th,tnh->tn", x.float(), d13[ids[:, 0]])
    assert (pre_act.abs() > LIMIT).float().mean() > 0.2

    captured = {}
    scheme.runner = SimpleNamespace(run=lambda d, q: captured.setdefault("q", q))
    scheme.apply_weights(layer, None)
    dispatch, slots = _ll_dispatch(x, ids, weights, capacity=256)
    expert_out = flashinfer_cutedsl.fused_experts_deepep_to_flashinfer_cutedsl_fp4(
        dispatch, captured["q"], layer.moe_runner_config
    ).hidden_states
    assert expert_out.shape == (E, 256, H)

    out = torch.zeros(T, H, device="cuda", dtype=torch.float32)
    for t, k, e, r in slots:
        out[t] += weights[t, k] * expert_out[e, r].float()
    rel = ((out - ref).norm() / ref.norm()).item()
    fit = ((out * ref).sum() / (ref * ref).sum()).item()
    assert rel < 0.13 and abs(fit - 1) < 0.02, (rel, fit)


def test_scheme_rejects_cutedsl_without_deepep():
    with (
        mock.patch.object(
            minimax_m3_nvfp4,
            "get_platform",
            return_value=SimpleNamespace(
                is_blackwell=True, is_hip=False, is_gfx95=False
            ),
        ),
        pytest.raises(ValueError, match="deepep"),
    ):
        _scheme(MoeRunnerBackend.FLASHINFER_CUTEDSL, MoeA2ABackend.NONE)


def test_process_weights_rejects_missing_fused_epilogue():
    with mock.patch.object(
        minimax_m3_nvfp4,
        "get_platform",
        return_value=SimpleNamespace(is_blackwell=True, is_hip=False, is_gfx95=False),
    ):
        scheme = _scheme(MoeRunnerBackend.FLASHINFER_CUTEDSL, MoeA2ABackend.DEEPEP)
    layer = torch.nn.Module()
    layer.register_parameter(
        "w13_weight_packed",
        torch.nn.Parameter(
            torch.zeros((2, 128, 32), dtype=torch.uint8), requires_grad=False
        ),
    )
    with (
        mock.patch.object(
            flashinfer_cutedsl_moe,
            "supports_masked_gemm_fused_swiglu",
            return_value=False,
        ),
        pytest.raises(RuntimeError, match="grouped_gemm_nt_masked"),
    ):
        scheme.process_weights_after_loading(layer)


def test_cutedsl_masked_rejects_gemm1_clamp_without_per_token_scale():
    dispatch = DeepEPLLDispatchOutput(
        hidden_states=torch.empty((1, 1, 1), dtype=torch.uint8),
        hidden_states_scale=None,
        topk_ids=torch.empty((1, 1), dtype=torch.int64),
        topk_weights=torch.empty((1, 1), dtype=torch.float32),
        masked_m=torch.empty((1,), dtype=torch.int32),
        expected_m=1,
        per_token_scale=None,
    )
    cfg = MoeRunnerConfig(gemm1_clamp_limit=LIMIT, gemm1_alpha=ALPHA, gemm1_beta=1.0)
    with pytest.raises(AssertionError, match="gemm1_clamp_limit"):
        flashinfer_cutedsl.fused_experts_deepep_to_flashinfer_cutedsl_fp4(
            dispatch, SimpleNamespace(), cfg
        )


def test_deepep_ll_pre_permute_accepts_default_per_token_scale():
    """Regression: 7-field DeepEPLLDispatchOutput must not break other LL consumers."""
    from sglang.srt.layers.moe.moe_runner.ascend import pre_permute_deepep_ll_to_ascend
    from sglang.srt.layers.moe.moe_runner.deep_gemm import (
        pre_permute_deepep_ll_to_deep_gemm,
    )
    from sglang.srt.layers.quantization.w4afp8 import W4AFp8Config, W4AFp8MoEMethod

    dispatch = DeepEPLLDispatchOutput(
        hidden_states=torch.zeros((2, 4, 8), dtype=torch.float8_e4m3fn),
        hidden_states_scale=torch.ones((2, 4, 1), dtype=torch.float32),
        topk_ids=torch.zeros((3, 2), dtype=torch.int64),
        topk_weights=torch.ones((3, 2), dtype=torch.float32),
        masked_m=torch.tensor([2, 1], dtype=torch.int32),
        expected_m=2,
    )
    dg_state = {}
    dg_in = pre_permute_deepep_ll_to_deep_gemm(
        dispatch, SimpleNamespace(), MoeRunnerConfig(), dg_state
    )
    assert dg_in.expected_m == 2 and dg_in.masked_m is dispatch.masked_m
    assert dg_state["topk_ids"] is dispatch.topk_ids

    asc_state = {}
    asc_in = pre_permute_deepep_ll_to_ascend(
        dispatch, SimpleNamespace(), MoeRunnerConfig(), asc_state
    )
    assert asc_in.expert_tokens.dtype == torch.int64
    assert asc_state["topk_weights"] is dispatch.topk_weights

    w4_method = W4AFp8MoEMethod(W4AFp8Config())
    with pytest.raises(RuntimeError, match="requires FP8 dispatcher output"):
        w4_method.apply_deepep_ll(
            SimpleNamespace(), dispatch._replace(hidden_states_scale=None)
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
