import pytest
import torch

from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate
from sglang.kernels.ops.moe.moe_route_radix import route_radix
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

NUM_EXPERTS = 896
TOPK = 16


def _triton_reference(
    scores: torch.Tensor,
    bias: torch.Tensor,
    renormalize: bool,
    routed_scaling_factor: float,
    apply_scale: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    # FP32 scores are outside route_radix.covered(), so this exercises the
    # Triton implementation even though radix is the canonical BF16 path.
    return moe_fused_gate(
        scores.float(),
        bias,
        topk=TOPK,
        scoring_func="sigmoid",
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=apply_scale,
    )


def _make_case(case: str, num_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.randn(num_tokens, NUM_EXPERTS, dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(NUM_EXPERTS, dtype=torch.float32, device="cuda")
    if case == "random":
        pass
    elif case == "all_equal":
        # Full 896-way key tie: min-id tie-break must pick experts 0..15.
        scores.fill_(0.5)
        bias.zero_()
    elif case == "few_values":
        # bf16-quantized duplicates: many key ties at the top-16 boundary.
        scores = (
            torch.randint(0, 8, (num_tokens, NUM_EXPERTS), device="cuda").to(
                torch.bfloat16
            )
            * 0.125
        )
        bias.zero_()
    elif case == "nan_mixed":
        scores[:, ::3] = float("nan")
    elif case == "mostly_nan":
        # Fewer than topk non-NaN entries: NaN-floored experts get selected
        # and their raw-sigmoid weights are NaN (Triton semantics).
        scores[:, : NUM_EXPERTS - 10] = float("nan")
    elif case == "huge_negative_bias":
        # biased values below the -1e30 NaN floor.
        bias = bias * 1e31 - 1e31
    else:
        raise AssertionError(case)
    return scores, bias


@pytest.mark.parametrize("num_tokens", [1, 3, 8, 512])
@pytest.mark.parametrize(
    "case",
    [
        "random",
        "all_equal",
        "few_values",
        "nan_mixed",
        "mostly_nan",
        "huge_negative_bias",
    ],
)
@pytest.mark.parametrize(
    "renormalize,apply_scale", [(True, True), (False, False), (True, False)]
)
def test_route_radix_vs_triton(num_tokens, case, renormalize, apply_scale):
    torch.manual_seed(num_tokens)
    scores, bias = _make_case(case, num_tokens)
    args = (scores, bias, TOPK, renormalize, 2.5, apply_scale)

    ref_w, ref_i = _triton_reference(scores, bias, renormalize, 2.5, apply_scale)
    ref_order = ref_i.argsort(dim=-1)
    for sorted_output in (True, False):
        w, i = route_radix(*args, sorted=sorted_output)
        order = i.argsort(dim=-1)
        assert torch.equal(
            ref_i.to(torch.int32).gather(1, ref_order), i.gather(1, order)
        ), f"winner set diverges from Triton: {case}, sorted={sorted_output}"
        torch.testing.assert_close(
            ref_w.gather(1, ref_order),
            w.gather(1, order),
            rtol=1e-6,
            atol=0.0,
            equal_nan=True,
        )


def test_route_radix_unsorted_id_order():
    # sorted=False documents compaction (expert-id ascending) output order.
    torch.manual_seed(0)
    scores = torch.randn(4, NUM_EXPERTS, dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(NUM_EXPERTS, dtype=torch.float32, device="cuda")
    _, ids = route_radix(scores, bias, TOPK, True, 2.5, True, sorted=False)
    assert torch.equal(ids, ids.sort(dim=-1).values)


def test_route_radix_automatic_dispatch_and_fallback():
    # Covered BF16 inputs route to radix-unsorted automatically. Unsupported FP32
    # inputs retain the Triton fallback.
    torch.manual_seed(0)
    scores = torch.randn(2, NUM_EXPERTS, dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(NUM_EXPERTS, dtype=torch.float32, device="cuda")
    common = dict(
        topk=TOPK,
        scoring_func="sigmoid",
        renormalize=True,
        routed_scaling_factor=2.5,
        apply_routed_scaling_factor_on_output=True,
    )
    ref_w, ref_i = moe_fused_gate(scores.float(), bias, **common)
    w, i = moe_fused_gate(scores, bias, **common)
    assert torch.equal(
        i, i.sort(dim=-1).values
    ), "dispatch did not reach radix-unsorted"
    ref_order = ref_i.argsort(dim=-1)
    assert torch.equal(ref_i.to(torch.int32).gather(1, ref_order), i)
    torch.testing.assert_close(ref_w.gather(1, ref_order), w, rtol=1e-6, atol=0.0)

    fw, fi = moe_fused_gate(scores.float(), bias, **common)
    assert torch.equal(ref_i.to(fi.dtype), fi)
    torch.testing.assert_close(ref_w, fw, rtol=0.0, atol=0.0)


def test_route_radix_all_equal_min_id():
    scores = torch.full((2, NUM_EXPERTS), 0.5, dtype=torch.bfloat16, device="cuda")
    bias = torch.zeros(NUM_EXPERTS, dtype=torch.float32, device="cuda")
    for sorted_flag in (True, False):
        _, ids = route_radix(scores, bias, TOPK, True, 2.5, True, sorted=sorted_flag)
        expected = torch.arange(TOPK, dtype=torch.int32, device="cuda")
        assert torch.equal(ids, expected.expand(2, -1))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
