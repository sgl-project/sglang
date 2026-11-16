"""M3.1 NVFP4 experts on flashinfer_trtllm_routed, alone and under MegaMoE prefill."""

import socket
import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

# Import quantization before flashinfer_trtllm; the reverse is a circular import.
# isort: off
from sglang.srt.layers.quantization import minimax_m3_nvfp4
from sglang.srt.layers.moe.moe_runner import flashinfer_trtllm

# isort: on
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

E, H, I, TOPK, T = 8, 1024, 512, 4, 96
ALPHA, LIMIT = 1.702, 7.0
_E2M1 = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]


def _quantize(w):
    """Compressed-tensors NVFP4: packed e2m1, e4m3 block scales, encode global."""
    from flashinfer import fp4_quantize

    gs = (448 * 6 / w.abs().amax()).float().reshape(1)
    q, sf = fp4_quantize(w, gs, 16, False, False)
    sf = sf.view(torch.float8_e4m3fn).reshape(w.shape[0], -1)
    lut = torch.tensor(_E2M1, device=w.device)
    vals = torch.stack([lut[(q & 0xF).long()], lut[(q >> 4).long()]], -1)
    deq = vals.reshape(w.shape) * sf.float().repeat_interleave(16, -1) / gs
    return q.view(torch.uint8), sf, gs, deq


def _fake_quant_per_token(a):
    """What the kernel's dynamic per-token NVFP4 activation quant keeps of `a`."""
    return torch.cat([_quantize(row[None].bfloat16())[3] for row in a])


def _swigluoai(x):
    gate, up = x.chunk(2, -1)
    gate, up = gate.clamp(max=LIMIT), up.clamp(-LIMIT, LIMIT)
    return gate * torch.sigmoid(ALPHA * gate) * (up + 1)


def _build_layer():
    layer = torch.nn.Module()
    w13 = torch.randn(E, 2 * I, H, device="cuda", dtype=torch.bfloat16) / 4
    w2 = torch.randn(E, H, I, device="cuda", dtype=torch.bfloat16) / 8
    q13, s13, g13, d13 = zip(*(_quantize(w) for w in w13))
    q2, s2, g2, d2 = zip(*(_quantize(w) for w in w2))
    params = dict(
        w13_weight_packed=torch.stack(q13),
        w13_weight_scale=torch.stack(s13),
        w13_weight_global_scale=torch.cat(g13)[:, None].repeat(1, 2),
        w13_input_global_scale=torch.full((E, 2), 1.0, device="cuda"),
        w2_weight_packed=torch.stack(q2),
        w2_weight_scale=torch.stack(s2),
        w2_weight_global_scale=torch.cat(g2),
        w2_input_global_scale=torch.full((E,), 1.0, device="cuda"),
    )
    for name, t in params.items():
        layer.register_parameter(name, torch.nn.Parameter(t, requires_grad=False))
    layer.num_experts = layer.num_local_experts = E
    layer.moe_ep_rank = 0
    layer.intermediate_size_per_partition = I
    layer.routing_method_type = None
    layer._minimax_activation = "swigluoai"
    layer._mega_moe_trtllm_layout = False
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
    return layer, torch.stack(d13), torch.stack(d2)


def _reference(x, ids, weights, d13, d2):
    out = torch.zeros(x.shape, device="cuda", dtype=torch.float32)
    xq = _fake_quant_per_token(x)
    for k in range(TOPK):
        h = _swigluoai(torch.einsum("th,tnh->tn", xq, d13[ids[:, k]]))
        h = _fake_quant_per_token(h)
        out += weights[:, k : k + 1] * torch.einsum("ti,thi->th", h, d2[ids[:, k]])
    return out


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="TRT-LLM FP4 MoE needs SM100",
)
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


def _scheme(runner, a2a):
    with (
        mock.patch.object(
            minimax_m3_nvfp4, "get_moe_runner_backend", return_value=runner
        ),
        mock.patch.object(minimax_m3_nvfp4, "get_moe_a2a_backend", return_value=a2a),
    ):
        return minimax_m3_nvfp4.MiniMaxM3Nvfp4MoE()


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


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs SM100",
)
def test_mega_moe_on_trtllm_weights_matches_canonical_mega_moe(ep_group):
    """MegaMoE prefill over TRT-LLM decode weights matches canonical MegaMoE."""
    from sglang.srt.layers.moe.mega_moe import run_mega_routed_experts

    canonical = _scheme(MoeRunnerBackend.DEEP_GEMM, MoeA2ABackend.MEGAMOE)
    hybrid = _scheme(MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED, MoeA2ABackend.MEGAMOE)
    assert not canonical.use_flashinfer_trtllm and hybrid.use_mega_moe_prefill
    torch.manual_seed(0)
    layer_canonical, _, _ = _build_layer()
    torch.manual_seed(0)
    layer_hybrid, _, _ = _build_layer()
    # Non-unit so a dropped or double-applied fc2 input scale shows up.
    layer_canonical.w2_input_global_scale.data.fill_(2.0)
    layer_hybrid.w2_input_global_scale.data.fill_(2.0)
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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
