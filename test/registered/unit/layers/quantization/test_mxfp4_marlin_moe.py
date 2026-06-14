"""Unit test for MXFP4 Marlin MoE with expert parallelism (EP) support.

Verifies that expert_map and global_num_experts are correctly used by the
MXFP4 Marlin kernel to handle -1 expert IDs when EP is enabled. Before the fix
in Mxfp4MarlinMoEMethod.apply(), the kernel would compute a negative weight
offset from -1 expert IDs, causing CUDA illegal memory access.

Run on H100/H200:

    python -m pytest test/registered/unit/layers/quantization/test_mxfp4_marlin_moe.py -v
"""

from __future__ import annotations

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from sglang.srt.utils import is_sm90_supported, is_sm100_supported

if not is_sm90_supported() or is_sm100_supported():
    pytest.skip(
        "SM90/Hopper required; SM100 promotion not supported",
        allow_module_level=True,
    )

from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import fused_marlin_moe
from sglang.srt.layers.quantization.marlin_utils_fp4 import (
    prepare_moe_mxfp4_layer_for_marlin,
)

FP4_BLOCK_K = 32  # MXFP4 group size for Marlin


class _MockLayer:
    """Minimal layer carrying MXFP4 weights for Marlin prep."""

    pass


def _make_mxfp4_layer(num_experts, hidden, inter, seed=0):
    """Create a mock layer with MXFP4-format weights and prepare for Marlin."""
    g = torch.Generator(device="cuda").manual_seed(seed)

    w13 = torch.randint(
        -128,
        128,
        (num_experts, 2 * inter, hidden // 2),
        dtype=torch.int8,
        device="cuda",
        generator=g,
    )
    w2 = torch.randint(
        -128,
        128,
        (num_experts, hidden, inter // 2),
        dtype=torch.int8,
        device="cuda",
        generator=g,
    )
    # Raw E8M0 exponent bytes: 120-135  =>  2^{-7} ... 2^{8} (sane range).
    w13_s = torch.randint(
        120,
        135,
        (num_experts, 2 * inter, hidden // FP4_BLOCK_K),
        dtype=torch.uint8,
        device="cuda",
        generator=g,
    )
    w2_s = torch.randint(
        120,
        135,
        (num_experts, hidden, inter // FP4_BLOCK_K),
        dtype=torch.uint8,
        device="cuda",
        generator=g,
    )

    layer = _MockLayer()
    layer.w13_weight = torch.nn.Parameter(w13.clone(), requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(w2.clone(), requires_grad=False)
    layer.w13_weight_scale_inv = torch.nn.Parameter(w13_s.clone(), requires_grad=False)
    layer.w2_weight_scale_inv = torch.nn.Parameter(w2_s.clone(), requires_grad=False)
    layer.orig_dtype = torch.bfloat16

    prepare_moe_mxfp4_layer_for_marlin(layer)
    return layer


def get_slice(layer, count):
    """Return a slice of the first `count` experts' weights from a marlin-prep'd layer."""
    clone = _MockLayer()
    for attr in (
        "w13_weight",
        "w2_weight",
        "w13_weight_scale",
        "w2_weight_scale",
    ):
        setattr(clone, attr, getattr(layer, attr)[:count].clone())
    return clone


# =============================================================================
# fused_marlin_moe kernel tests
# =============================================================================


@pytest.mark.parametrize(
    "num_experts,hidden,inter,top_k",
    [
        (4, 256, 256, 2),
        (8, 512, 256, 2),
    ],
)
def test_mxfp4_marlin_identity_expert_map(num_experts, hidden, inter, top_k):
    """Identity expert_map (all experts local) produces same output as no EP."""
    m = 16
    layer = _make_mxfp4_layer(num_experts, hidden, inter)

    torch.manual_seed(42)
    x = torch.randn(m, hidden, dtype=torch.bfloat16, device="cuda") * 0.1

    gating_output = torch.randn(m, num_experts, dtype=torch.float32, device="cuda")
    topk_weights, topk_ids = torch.topk(
        torch.softmax(gating_output, dim=-1), top_k
    )
    topk_weights = topk_weights.to(torch.float32)
    topk_ids = topk_ids.to(torch.int32)

    identity_map = torch.arange(num_experts, dtype=torch.int32, device="cuda")

    kwargs = dict(
        hidden_states=x.clone(),
        w1=layer.w13_weight,
        w2=layer.w2_weight,
        w1_scale=layer.w13_weight_scale,
        w2_scale=layer.w2_weight_scale,
        gating_output=gating_output,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        num_bits=4,
        is_k_full=True,
    )

    out_no_ep = fused_marlin_moe(**kwargs)

    out_ep = fused_marlin_moe(
        **kwargs,
        global_num_experts=num_experts,
        expert_map=identity_map,
    )

    torch.testing.assert_close(
        out_no_ep,
        out_ep,
        atol=1e-2,
        rtol=1e-2,
        msg=lambda s: (
            f"EP identity-map output differs from non-EP path; "
            f"max diff = {(out_no_ep.float() - out_ep.float()).abs().max().item():.4g}"
        ),
    )


@pytest.mark.parametrize(
    "num_experts,hidden,inter,top_k",
    [
        (8, 256, 256, 2),
    ],
)
def test_mxfp4_marlin_partial_map_no_remote_tokens(
    num_experts, hidden, inter, top_k
):
    """Partial expert_map: all tokens routed to local experts => EP == non-EP."""
    m = 16
    ep_size = 2
    local_e = num_experts // ep_size

    # Build a full-expert layer, then slice to local experts.
    full_layer = _make_mxfp4_layer(num_experts, hidden, inter)
    layer = get_slice(full_layer, local_e)

    # Map: first *local_e* experts → themselves; rest → -1 (no token goes there).
    e_map = torch.full((num_experts,), -1, dtype=torch.int32, device="cuda")
    e_map[:local_e] = torch.arange(local_e, dtype=torch.int32, device="cuda")

    torch.manual_seed(42)
    x = torch.randn(m, hidden, dtype=torch.bfloat16, device="cuda") * 0.1

    # Bias routing so every token only lands on local experts (0..3).
    gating_output = torch.randn(m, num_experts, dtype=torch.float32, device="cuda")
    gating_output[:, :local_e] += 20.0
    topk_weights, topk_ids = torch.topk(
        torch.softmax(gating_output, dim=-1), top_k
    )
    topk_weights = topk_weights.to(torch.float32)
    topk_ids = topk_ids.to(torch.int32)

    # Remap (no -1 produced because all tokens map to local experts).
    remapped_ids = e_map[topk_ids]
    assert (remapped_ids != -1).all(), "Expected no -1 after remapping"

    kwargs = dict(
        hidden_states=x.clone(),
        w1=layer.w13_weight,
        w2=layer.w2_weight,
        w1_scale=layer.w13_weight_scale,
        w2_scale=layer.w2_weight_scale,
        gating_output=gating_output,
        topk_weights=topk_weights,
        num_bits=4,
        is_k_full=True,
    )

    # Non-EP reference: local weights with local-only topk_ids.
    out_ref = fused_marlin_moe(topk_ids=remapped_ids, **kwargs)

    # EP path: same local weights, partial expert_map, global num_experts.
    out_ep = fused_marlin_moe(
        **kwargs,
        topk_ids=remapped_ids,
        global_num_experts=num_experts,
        expert_map=e_map,
    )

    torch.testing.assert_close(
        out_ref,
        out_ep,
        atol=1e-2,
        rtol=1e-2,
        msg=lambda s: (
            f"Partial EP output differs from non-EP reference; "
            f"max diff = {(out_ref.float() - out_ep.float()).abs().max().item():.4g}"
        ),
    )


@pytest.mark.parametrize(
    "num_experts,hidden,inter,top_k",
    [
        (8, 256, 256, 2),
    ],
)
def test_mxfp4_marlin_negative_expert_ids(num_experts, hidden, inter, top_k):
    """Kernel handles -1 expert IDs without crash and produces finite output."""
    m = 32
    ep_size = 2
    local_e = num_experts // ep_size

    layer = _make_mxfp4_layer(local_e, hidden, inter)

    e_map = torch.full((num_experts,), -1, dtype=torch.int32, device="cuda")
    e_map[:local_e] = torch.arange(local_e, dtype=torch.int32, device="cuda")

    torch.manual_seed(42)
    x = torch.randn(m, hidden, dtype=torch.bfloat16, device="cuda") * 0.1

    # Uniform routing: roughly half the tokens land on remote experts (4..7).
    gating_output = torch.randn(m, num_experts, dtype=torch.float32, device="cuda")
    topk_weights, topk_ids = torch.topk(
        torch.softmax(gating_output, dim=-1), top_k
    )
    topk_weights = topk_weights.to(torch.float32)
    topk_ids = topk_ids.to(torch.int32)

    remapped_ids = e_map[topk_ids]

    # Sanity: at least some -1 entries exist.
    num_neg = (remapped_ids == -1).sum().item()
    assert num_neg > 0, (
        f"Expected some -1 expert IDs after remapping, got none "
        f"(num_experts={num_experts}, local_e={local_e})"
    )

    out = fused_marlin_moe(
        hidden_states=x,
        w1=layer.w13_weight,
        w2=layer.w2_weight,
        w1_scale=layer.w13_weight_scale,
        w2_scale=layer.w2_weight_scale,
        gating_output=gating_output,
        topk_weights=topk_weights,
        topk_ids=remapped_ids,
        global_num_experts=num_experts,
        expert_map=e_map,
        num_bits=4,
        is_k_full=True,
    )

    assert not torch.isnan(out).any(), "Output contains NaN"
    assert not torch.isinf(out).any(), "Output contains Inf"
    # Local experts should produce non-zero contributions.
    assert out.abs().max().item() > 0, "Output is all zeros"


# =============================================================================
# Mxfp4MarlinMoEMethod.apply() extraction test
# =============================================================================


class _MockDispatcher:
    def __init__(self, local_expert_mapping=None):
        self.local_expert_mapping = local_expert_mapping


class _MockDispatchOutput:
    def __init__(self, hidden_states, topk_weights, topk_ids):
        self.hidden_states = hidden_states
        from sglang.srt.layers.moe.topk import StandardTopKOutput

        router_logits = torch.zeros(
            topk_ids.shape[0],
            int(topk_ids.max().item()) + 1 if topk_ids.numel() else 1,
            dtype=torch.float32,
            device=topk_ids.device,
        )
        self.topk_output = StandardTopKOutput(
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            router_logits=router_logits,
        )


class _CapturingRunner:
    """Mock MoeRunner that records the quant_info it receives."""

    class config:
        num_experts = 8

    def __init__(self):
        self.last_quant_info = None

    def run(self, dispatch_output, quant_info=None):
        self.last_quant_info = quant_info
        return type("_MockRunnerOutput", (), {
            "hidden_states": dispatch_output.hidden_states,
        })()


def test_apply_extracts_expert_map_from_dispatcher():
    """Mxfp4MarlinMoEMethod.apply() passes expert_map from dispatcher to runner."""
    from sglang.srt.layers.quantization.mxfp4_marlin_moe import Mxfp4MarlinMoEMethod

    method = Mxfp4MarlinMoEMethod.__new__(Mxfp4MarlinMoEMethod)
    runner = _CapturingRunner()
    method.runner = runner

    expert_map = torch.tensor(
        [0, 1, 2, 3, -1, -1, -1, -1], dtype=torch.int32, device="cuda"
    )

    layer = _MockLayer()
    layer._dsv4_mxfp4_backend = "marlin"
    layer.dispatcher = _MockDispatcher(local_expert_mapping=expert_map)

    # Dummy weights aren't used because runner is mocked, but the apply()
    # method still reads them to build MarlinMoeQuantInfo.
    hidden = 256
    inter = 256
    num_experts = 4
    layer.w13_weight = torch.nn.Parameter(
        torch.zeros(num_experts, 2 * inter, hidden // 2, dtype=torch.int8, device="cuda"),
        requires_grad=False,
    )
    layer.w2_weight = torch.nn.Parameter(
        torch.zeros(num_experts, hidden, inter // 2, dtype=torch.int8, device="cuda"),
        requires_grad=False,
    )
    layer.w13_weight_scale = torch.nn.Parameter(
        torch.ones(1, dtype=torch.float8_e8m0fnu, device="cuda"),
        requires_grad=False,
    )
    layer.w2_weight_scale = torch.nn.Parameter(
        torch.ones(1, dtype=torch.float8_e8m0fnu, device="cuda"),
        requires_grad=False,
    )

    x = torch.randn(4, hidden, dtype=torch.bfloat16, device="cuda") * 0.1
    topk_weights = torch.rand(4, 2, dtype=torch.float32, device="cuda")
    topk_ids = torch.randint(0, num_experts, (4, 2), dtype=torch.int32, device="cuda")
    dispatch_output = _MockDispatchOutput(x, topk_weights, topk_ids)

    method.apply(layer, dispatch_output)

    assert runner.last_quant_info is not None, "Runner.run() was not called"
    assert runner.last_quant_info.expert_map is expert_map, (
        f"expert_map not passed through; got {runner.last_quant_info.expert_map}"
    )
    assert runner.last_quant_info.global_num_experts == 8, (
        f"global_num_experts mismatch; got {runner.last_quant_info.global_num_experts}"
    )


def test_apply_expert_map_none_when_no_dispatcher():
    """When layer has no dispatcher, expert_map is None."""
    from sglang.srt.layers.quantization.mxfp4_marlin_moe import Mxfp4MarlinMoEMethod

    method = Mxfp4MarlinMoEMethod.__new__(Mxfp4MarlinMoEMethod)
    runner = _CapturingRunner()
    method.runner = runner

    layer = _MockLayer()
    layer._dsv4_mxfp4_backend = "marlin"
    hidden = 256
    inter = 256
    num_experts = 4
    layer.w13_weight = torch.nn.Parameter(
        torch.zeros(num_experts, 2 * inter, hidden // 2, dtype=torch.int8, device="cuda"),
        requires_grad=False,
    )
    layer.w2_weight = torch.nn.Parameter(
        torch.zeros(num_experts, hidden, inter // 2, dtype=torch.int8, device="cuda"),
        requires_grad=False,
    )
    layer.w13_weight_scale = torch.nn.Parameter(
        torch.ones(1, dtype=torch.float8_e8m0fnu, device="cuda"),
        requires_grad=False,
    )
    layer.w2_weight_scale = torch.nn.Parameter(
        torch.ones(1, dtype=torch.float8_e8m0fnu, device="cuda"),
        requires_grad=False,
    )

    x = torch.randn(4, hidden, dtype=torch.bfloat16, device="cuda") * 0.1
    topk_weights = torch.rand(4, 2, dtype=torch.float32, device="cuda")
    topk_ids = torch.randint(0, num_experts, (4, 2), dtype=torch.int32, device="cuda")
    dispatch_output = _MockDispatchOutput(x, topk_weights, topk_ids)

    method.apply(layer, dispatch_output)

    assert runner.last_quant_info.expert_map is None, (
        f"Expected expert_map=None without dispatcher, got {runner.last_quant_info.expert_map}"
    )
    assert runner.last_quant_info.global_num_experts == -1


def test_apply_expert_map_none_when_dispatcher_has_no_mapping():
    """When dispatcher exists but local_expert_mapping is None, expert_map is None."""
    from sglang.srt.layers.quantization.mxfp4_marlin_moe import Mxfp4MarlinMoEMethod

    method = Mxfp4MarlinMoEMethod.__new__(Mxfp4MarlinMoEMethod)
    runner = _CapturingRunner()
    method.runner = runner

    layer = _MockLayer()
    layer._dsv4_mxfp4_backend = "marlin"
    layer.dispatcher = _MockDispatcher(local_expert_mapping=None)

    hidden = 256
    inter = 256
    num_experts = 4
    layer.w13_weight = torch.nn.Parameter(
        torch.zeros(num_experts, 2 * inter, hidden // 2, dtype=torch.int8, device="cuda"),
        requires_grad=False,
    )
    layer.w2_weight = torch.nn.Parameter(
        torch.zeros(num_experts, hidden, inter // 2, dtype=torch.int8, device="cuda"),
        requires_grad=False,
    )
    layer.w13_weight_scale = torch.nn.Parameter(
        torch.ones(1, dtype=torch.float8_e8m0fnu, device="cuda"),
        requires_grad=False,
    )
    layer.w2_weight_scale = torch.nn.Parameter(
        torch.ones(1, dtype=torch.float8_e8m0fnu, device="cuda"),
        requires_grad=False,
    )

    x = torch.randn(4, hidden, dtype=torch.bfloat16, device="cuda") * 0.1
    topk_weights = torch.rand(4, 2, dtype=torch.float32, device="cuda")
    topk_ids = torch.randint(0, num_experts, (4, 2), dtype=torch.int32, device="cuda")
    dispatch_output = _MockDispatchOutput(x, topk_weights, topk_ids)

    method.apply(layer, dispatch_output)

    assert runner.last_quant_info.expert_map is None, (
        f"Expected expert_map=None when mapping is None, got {runner.last_quant_info.expert_map}"
    )
    assert runner.last_quant_info.global_num_experts == -1


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
