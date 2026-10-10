"""Regression tests: decode kernels must keep the sigmoid(beta) gate in fp32.

Both kernels covered here computed
``tl.sigmoid(b_val).to(b.dtype.element_ty).to(tl.float32)``: the fp32 sigmoid
was rounded to the input dtype (bf16 in serving) and immediately cast back,
discarding the low mantissa bits with no bf16 arithmetic in between. The
rounded gate multiplies the value vector in the delta-rule update of the
persistent SSM state, which is read and written on every decode step, so the
per-token rounding error accumulates with context length (issue #38975). The
same-origin vllm copy of the packed kernel had the identical line and fixed it
in vllm-project/vllm#53877.

Degenerate config (B=H=HV=K=V=1, zero initial state, unit q=k=v): the
delta-rule correction vanishes on a zero state and the outer product v (x) k
is 1, so the stored state after one decode step equals sigmoid(b) itself.
Each test reads the gate straight out of the state pool and compares it with
a torch fp32 reference (rtol/atol=1e-6, the vllm regression-test tolerance).
With bf16 gating input the pre-fix state was 0.62109375 vs fp32
0.622459352016449 (rel err 0.2194%, exactly one bf16 rounding); keeping beta
in fp32 makes it bit-exact.

The ReplaySSM test runs at L=1, where ``write_pos == L-1`` makes every step a
flush, so the checkpoint state (not just the ring) is updated immediately and
reduces to the same one-step delta-rule update. That kernel tiles its
reconstruction around ``tl.dot`` and requires K, V >= 16, so its unit vectors
are one-hot (q = 0, k = e_0, v = e_0) and the expected state is all zeros with
sigmoid(b) in the single [0, 0] element.
"""

import unittest

import torch

# Register for CUDA/AMD CI, kernel-kind suites (see sibling kernel tests).
# Module-level calls: the CI collector parses them statically via AST.
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=20, stage="jit-kernel-unit", runner_config="amd")


def _one_step_inputs(device, dtype):
    """One token, B=H=HV=K=V=1, unit q=k=v, b=0.5 (sigmoid far from bf16 grid)."""
    mixed_qkv = torch.ones((1, 3), device=device, dtype=dtype)
    a = torch.zeros((1, 1), device=device, dtype=dtype)
    b = torch.full((1, 1), 0.5, device=device, dtype=dtype)
    params = torch.zeros((1,), device=device, dtype=dtype)
    return mixed_qkv, a, b, params


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestDecodeKeepsBetaInFp32(CustomTestCase):
    def test_gdn_packed_decode(self):
        from sglang.kernels.ops.attention.fla.fused_recurrent import (
            fused_recurrent_gated_delta_rule_packed_decode as packed_decode,
        )

        device, dtype = "cuda", torch.bfloat16
        mixed_qkv, a, b, params = _one_step_inputs(device, dtype)
        # 2-slot pool: state[1] stays zero and isolates the written slot.
        state = torch.zeros((2, 1, 1, 1), device=device, dtype=torch.float32)
        out = torch.empty((1, 1, 1, 1), device=device, dtype=dtype)
        packed_decode(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=params,
            dt_bias=params,
            scale=1.0,
            initial_state=state,
            out=out,
            ssm_state_indices=torch.zeros((1,), device=device, dtype=torch.int32),
        )
        torch.testing.assert_close(
            state[0].reshape(-1),
            torch.sigmoid(b.float()).reshape(-1),
            atol=1e-6,
            rtol=1e-6,
            msg="packed decode must keep sigmoid(beta) in fp32",
        )

    def test_linear_replayssm_decode(self):
        from sglang.kernels.ops.attention.fla.fused_recurrent_linear_replayssm import (
            fused_recurrent_linear_replayssm_decode as replayssm_decode,
        )

        device, dtype = "cuda", torch.bfloat16
        # Same degenerate recurrence, sized for this kernel's tl.dot tiling
        # (K, V >= 16): q = 0, k = e_0, v = e_0 puts sigmoid(b) alone in the
        # state at [v=0, k=0]; every other element stays exactly zero.
        K = V = 16
        mixed_qkv = torch.zeros((1, 2 * K + V), device=device, dtype=dtype)
        mixed_qkv[0, K] = 1.0  # k = e_0
        mixed_qkv[0, 2 * K] = 1.0  # v = e_0
        a = torch.zeros((1, 1), device=device, dtype=dtype)
        b = torch.full((1, 1), 0.5, device=device, dtype=dtype)
        params = torch.zeros((1,), device=device, dtype=dtype)
        state = torch.zeros((1, 1, V, K), device=device, dtype=torch.float32)
        out = torch.empty((1, 1, 1, V), device=device, dtype=dtype)
        L = 1  # write_pos == L-1: this step is a flush
        replayssm_decode(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=params,
            dt_bias=params,
            scale=1.0,
            initial_state=state,
            d_cache=torch.zeros((1, 1, L, V), device=device, dtype=dtype),
            k_cache=torch.zeros((1, 1, L, K), device=device, dtype=dtype),
            g_cache=torch.zeros((1, 1, L), device=device, dtype=torch.float32),
            out=out,
            ssm_state_indices=torch.zeros((1,), device=device, dtype=torch.int32),
            write_pos=torch.zeros((1,), device=device, dtype=torch.int32),
            nk=1,  # BKT = BK/nk must stay >= 16 at BK=16
        )
        expected = torch.zeros((1, V, K), device=device, dtype=torch.float32)
        expected[0, 0, 0] = torch.sigmoid(b.float()).item()
        torch.testing.assert_close(
            state[0],
            expected,
            atol=1e-6,
            rtol=1e-6,
            msg="replayssm decode must keep sigmoid(beta) in fp32",
        )


if __name__ == "__main__":
    unittest.main()
