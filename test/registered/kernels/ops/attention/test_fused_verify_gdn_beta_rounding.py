"""GDN target verify rounds beta = sigmoid(b) to the input dtype.

Prefill (``fused_gdn_gating``) and decode (``fused_recurrent``) store beta in
the activation dtype; the fused verify kernel kept it in fp32, so a verified
token updated the recurrent state slightly differently from the decode step it
stands for. The kernel must match the reference that rounds beta much more
closely than the beta rounding itself moves the result.
"""

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from sglang.kernels.ops.attention.fla.fused_recurrent import (  # noqa: E402
    fused_recurrent_gated_delta_rule_update,
)
from sglang.kernels.ops.attention.linear.fused_gdn_gating import (  # noqa: E402
    fused_gdn_gating,
)
from sglang.kernels.ops.attention.linear.fused_sigmoid_gating_recurrent import (  # noqa: E402
    fused_sigmoid_gating_delta_rule_update,
)


@pytest.mark.parametrize("N,T", [(1, 4), (4, 8)])
def test_verify_beta_matches_decode_rounding(N, T):
    H, HV, K, V = 4, 8, 128, 128
    torch.manual_seed(7)
    dev, bf16 = "cuda", torch.bfloat16
    A_log = torch.randn(HV, device=dev)
    dt_bias = torch.randn(HV, dtype=bf16, device=dev)
    a = torch.randn(1, N * T, HV, dtype=bf16, device=dev)
    b = torch.randn(1, N * T, HV, dtype=bf16, device=dev)
    q = torch.randn(1, N * T, H, K, dtype=bf16, device=dev)
    k = torch.randn(1, N * T, H, K, dtype=bf16, device=dev)
    v = torch.randn(1, N * T, HV, V, dtype=bf16, device=dev) * 4
    state = torch.randn(N, HV, K, V, device=dev)
    idx = torch.arange(N, dtype=torch.int32, device=dev)
    cu = torch.arange(0, N * T + 1, T, dtype=torch.int32, device=dev)

    g, beta_rounded = fused_gdn_gating(A_log, a[0], b[0], dt_bias)
    beta_fp32 = b[0].float().sigmoid().view(beta_rounded.shape)
    assert not torch.equal(beta_rounded, beta_fp32)

    def reference(beta):
        return fused_recurrent_gated_delta_rule_update(
            q=q,
            k=k,
            v=v,
            g=g.view(a.shape),
            beta=beta.view(b.shape),
            initial_state_source=state.clone(),
            initial_state_indices=idx,
            cu_seqlens=cu,
            use_qk_l2norm_in_kernel=True,
            disable_state_update=True,
            intermediate_state_indices=idx,
        ).float()

    out = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        q=q,
        k=k,
        v=v,
        b=b,
        initial_state_source=state.clone(),
        initial_state_indices=idx,
        cu_seqlens=cu,
        use_qk_l2norm_in_kernel=True,
        disable_state_update=True,
    ).float()
    ref_rounded, ref_fp32 = reference(beta_rounded), reference(beta_fp32)
    rounding_effect = (ref_rounded - ref_fp32).abs().max().item()
    error = (out.view_as(ref_rounded) - ref_rounded).abs().max().item()
    assert rounding_effect > 0
    assert error < 0.25 * rounding_effect, (error, rounding_effect)
