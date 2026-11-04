"""Bitwise parity of the fused MoE finalize + shared-expert gate against TRT-LLM's
finalize followed by the gate kernel, below and above the small-grid finalize range.

Outputs are NaN-poisoned before each call, the test records which TRT-LLM finalize
variant the reference used, and a seeded positive control (one perturbed
shared-expert element) must change the fused output.
"""

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

flashinfer = pytest.importorskip("flashinfer")

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
        reason="needs an SM100-class GPU (TRT-LLM-gen FP4 MoE)",
    ),
]

# Qwen3.8-Flash-Next routed MoE at TP1.
HIDDEN, INTERMEDIATE, EXPERTS, TOP_K = 2560, 640, 512, 10
# TRT-LLM runs its small-grid finalize while 10 * tokens < 1184.
SMALL_GRID_MAX_TOKENS = 118


@pytest.fixture(scope="module")
def experts():
    from flashinfer.tllm_enums import RoutingMethodType  # noqa: F401

    gen = torch.Generator(device="cuda").manual_seed(0)
    dev = torch.device("cuda")

    def fp4_bytes(*shape):
        return torch.randint(0, 256, shape, generator=gen, device=dev, dtype=torch.uint8)

    def block_scales(*shape):
        exponent = -6.0 + 12.0 * torch.rand(shape, generator=gen, device=dev)
        return torch.exp2(exponent).to(torch.float8_e4m3fn)

    gate_alpha = 0.01 * (0.5 + torch.rand(EXPERTS, generator=gen, device=dev))
    return dict(
        gemm1_weights=fp4_bytes(EXPERTS, 2 * INTERMEDIATE, HIDDEN // 2),
        gemm1_weights_scale=block_scales(EXPERTS, 2 * INTERMEDIATE, HIDDEN // 16),
        gemm2_weights=fp4_bytes(EXPERTS, HIDDEN, INTERMEDIATE // 2),
        gemm2_weights_scale=block_scales(EXPERTS, HIDDEN, INTERMEDIATE // 16),
        output1_scale_scalar=(gate_alpha * 270.0).float(),
        output1_scale_gate_scalar=gate_alpha.float(),
        output2_scale_scalar=(
            3.7e-3 * (0.5 + torch.rand(EXPERTS, generator=gen, device=dev))
        ).float(),
        shared_gate=(0.02 * torch.randn(HIDDEN, generator=gen, device=dev)).to(
            torch.bfloat16
        ),
    )


def _moe_kwargs(experts, tokens, seed, *, do_finalize):
    from flashinfer.tllm_enums import ActivationType, RoutingMethodType

    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(tokens, HIDDEN, generator=gen, device="cuda")
    hidden = x.to(torch.bfloat16)
    global_scale = torch.tensor(
        [448.0 * 6.0 / float(x.abs().max())], dtype=torch.float32, device="cuda"
    )
    x_fp4, x_sf = flashinfer.fp4_quantize(hidden, global_scale, 16, False, False)
    kwargs = dict(
        routing_logits=torch.randn(tokens, EXPERTS, generator=gen, device="cuda").to(
            torch.bfloat16
        ),
        routing_bias=None,
        hidden_states=x_fp4.reshape(tokens, HIDDEN // 2),
        hidden_states_scale=x_sf.view(torch.float8_e4m3fn).reshape(
            tokens, HIDDEN // 16
        ),
        gemm1_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_bias=None,
        per_token_scale=None,
        num_experts=EXPERTS,
        top_k=TOP_K,
        n_group=None,
        topk_group=None,
        intermediate_size=INTERMEDIATE,
        local_expert_offset=0,
        local_num_experts=EXPERTS,
        routed_scaling_factor=None,
        routing_method_type=RoutingMethodType.RenormalizeNaive,
        do_finalize=do_finalize,
        activation_type=ActivationType.Swiglu,
        tune_max_num_tokens=1 << (tokens - 1).bit_length(),
        enable_pdl=False,
        output=torch.full(
            (tokens, HIDDEN), float("nan"), dtype=torch.bfloat16, device="cuda"
        ),
        **{k: v for k, v in experts.items() if k != "shared_gate"},
    )
    shared = (0.05 * torch.randn(tokens, HIDDEN, generator=gen, device="cuda")).to(
        torch.bfloat16
    )
    return hidden, shared, kwargs


@pytest.mark.parametrize("tokens", [1, 16, 64, 65, 118, 119, 576, 1024])
def test_fused_finalize_gate_is_bitwise(experts, tokens, record_property):
    from sglang.kernels.ops.elementwise.elementwise import (
        fused_gate_sigmoid_mul_add,
        fused_moe_finalize_gate_sigmoid_mul_add,
    )

    gate = experts["shared_gate"]
    hidden, shared, finalized_kwargs = _moe_kwargs(experts, tokens, seed=tokens, do_finalize=True)
    _, _, deferred_kwargs = _moe_kwargs(experts, tokens, seed=tokens, do_finalize=False)

    reference = flashinfer.trtllm_fp4_block_scale_moe(**finalized_kwargs)[0]
    fused_gate_sigmoid_mul_add(hidden, gate, shared, reference)
    gemm2_out, expert_weights, permuted_idx = flashinfer.trtllm_fp4_block_scale_moe(
        **deferred_kwargs
    )[:3]
    fused = fused_moe_finalize_gate_sigmoid_mul_add(
        hidden, gate, shared, gemm2_out, permuted_idx, expert_weights
    )
    torch.cuda.synchronize()

    variant = "small-grid" if tokens <= SMALL_GRID_MAX_TOKENS else "vectorized"
    record_property("trtllm_finalize", variant)
    assert not torch.isnan(reference).any(), "TRT-LLM finalize left rows unwritten"
    assert torch.equal(fused.view(torch.int16), reference.view(torch.int16))

    perturbed = shared.clone()
    row = int(torch.randint(tokens, (1,), generator=torch.Generator().manual_seed(tokens)))
    # Large next to the routed term, so BF16 rounding cannot absorb the change.
    perturbed[row] = (reference[row].float().abs() * 4 + 1).to(torch.bfloat16)
    control = fused_moe_finalize_gate_sigmoid_mul_add(
        hidden, gate, perturbed, gemm2_out, permuted_idx, expert_weights
    )
    assert not torch.equal(control.view(torch.int16), fused.view(torch.int16))


def _poison_next_like(tensor):
    # The kernel allocates its output with torch.empty_like; leave NaN in the block
    # it will reuse so an unwritten element cannot pass.
    poison = torch.full_like(tensor, float("nan"))
    del poison


@pytest.mark.parametrize("chunk", [128, 256, 512, 1024])
@pytest.mark.parametrize("tokens", [1, 2, 7, 16, 33, 64])
def test_column_chunk_grid_is_bitwise(tokens, chunk, monkeypatch, record_property):
    """The (token, column-chunk) grid, with its earlier PDL trigger, writes the same
    bits as one program per token."""
    from sglang.kernels.ops.elementwise import elementwise

    gen = torch.Generator(device="cuda").manual_seed(tokens * 7 + chunk)
    permuted_rows = tokens * TOP_K
    hidden = torch.randn(tokens, HIDDEN, generator=gen, device="cuda").to(torch.bfloat16)
    gate = (0.02 * torch.randn(HIDDEN, generator=gen, device="cuda")).to(torch.bfloat16)
    shared = (0.05 * torch.randn(tokens, HIDDEN, generator=gen, device="cuda")).to(
        torch.bfloat16
    )
    gemm2_out = torch.randn(permuted_rows, HIDDEN, generator=gen, device="cuda").to(
        torch.bfloat16
    )
    permuted = torch.randperm(permuted_rows, generator=gen, device="cuda").to(torch.int32)
    permuted[::7] = -1  # slots routed to another rank
    weights = torch.rand(tokens, TOP_K, generator=gen, device="cuda")
    args = (hidden, gate, shared, gemm2_out, permuted, weights)

    monkeypatch.setattr(elementwise, "_FINALIZE_GATE_COLUMN_CHUNK", 0)
    _poison_next_like(shared)
    row = elementwise.fused_moe_finalize_gate_sigmoid_mul_add(*args)
    monkeypatch.setattr(elementwise, "_FINALIZE_GATE_COLUMN_CHUNK", chunk)
    _poison_next_like(shared)
    chunked = elementwise.fused_moe_finalize_gate_sigmoid_mul_add(*args)
    torch.cuda.synchronize()

    record_property("grid", (tokens, -(-HIDDEN // chunk)))
    assert not torch.isnan(chunked).any(), "a column chunk was never written"
    assert torch.equal(chunked.view(torch.int16), row.view(torch.int16))

    perturbed = gemm2_out.clone()
    first = int(permuted[permuted >= 0][0])
    perturbed[first] = perturbed[first] * 4 + 1
    control = elementwise.fused_moe_finalize_gate_sigmoid_mul_add(
        hidden, gate, shared, perturbed, permuted, weights
    )
    assert not torch.equal(control.view(torch.int16), chunked.view(torch.int16))
