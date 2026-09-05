"""The fused act+quant emits the same bf16 rows as the unfused act, and the
same fp8 rows + scales the launcher's separate per-token-group quant would
have produced from those bf16 rows."""

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-large")

from sglang.srt.lora.moe.kernels.activation_delta import act_delta_contiguous


def _skip_unless_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")


@pytest.mark.parametrize("inter", [128, 256, 384, 1152])
@pytest.mark.parametrize("with_delta", [True, False])
def test_fused_act_quant_matches_unfused(inter, with_delta):
    _skip_unless_cuda()
    torch.manual_seed(0)
    device = torch.device("cuda")
    num_tokens, top_k, num_experts, group = 7, 3, 8, 128
    num_pairs = num_tokens * top_k

    gateup = torch.randn(num_pairs, 2 * inter, dtype=torch.bfloat16, device=device)
    delta = (
        0.1
        * torch.randn(num_tokens, top_k, 2 * inter, dtype=torch.bfloat16, device=device)
        if with_delta
        else None
    )
    topk_ids = torch.randint(
        0, num_experts, (num_tokens, top_k), dtype=torch.int32, device=device
    )
    topk_ids[0, 0] = -1  # one invalid pair keeps the masking honest
    pair_to_row = torch.arange(num_pairs, dtype=torch.int32, device=device)

    def run(act_quant):
        act = torch.zeros(num_pairs, inter, dtype=torch.bfloat16, device=device)
        pairs = torch.zeros(
            num_tokens, top_k, inter, dtype=torch.bfloat16, device=device
        )
        act_delta_contiguous(
            gateup,
            delta,
            act,
            pairs,
            pair_to_row,
            topk_ids,
            num_experts,
            act_quant=act_quant,
        )
        return act, pairs

    ref_act, ref_pairs = run(None)
    act_q = torch.zeros(num_pairs, inter, dtype=torch.float8_e4m3fn, device=device)
    act_s = torch.zeros(num_pairs, inter // group, dtype=torch.float32, device=device)
    fused_act, fused_pairs = run((act_q, act_s, group))

    # The bf16 outputs are byte-identical: same kernel, extra stores only.
    assert torch.equal(fused_act, ref_act)
    assert torch.equal(fused_pairs, ref_pairs)

    # The fp8 side matches the launcher's separate quant of the bf16 rows.
    from sglang.kernels.ops.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )

    ref_q, ref_s = sglang_per_token_group_quant_fp8(ref_act, group)
    valid_rows = (topk_ids.flatten() >= 0).nonzero(as_tuple=True)[0]
    # Same arithmetic as the reference quantizer (amax / 448, reciprocal
    # multiply, clamp), so codes and scales are bitwise equal.
    assert torch.equal(act_s[valid_rows], ref_s[valid_rows])
    assert torch.equal(
        act_q[valid_rows].view(torch.uint8), ref_q[valid_rows].view(torch.uint8)
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
