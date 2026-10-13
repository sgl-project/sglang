import unittest

import torch

from sglang.kernels.ops.attention.fla.kda import chunk_kda, kda_gate_chunk_cumsum
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestKDAGateBetaCumsum(unittest.TestCase):
    @torch.inference_mode()
    def test_gate_cumsum_beta_matches_separate_sigmoid(self):
        torch.manual_seed(42)
        for cu_seqlens, chunks, batch, tokens in (
            (None, None, 2, 65),
            (
                torch.tensor([0, 0, 1, 65, 130], device="cuda", dtype=torch.int32),
                torch.tensor(
                    [[1, 0], [2, 0], [3, 0], [3, 1]], device="cuda", dtype=torch.int32
                ),
                1,
                130,
            ),
        ):
            heads, dim = 3, 128
            gate = torch.randn(
                batch, tokens, heads, dim, device="cuda", dtype=torch.bfloat16
            )
            a_log = torch.randn(heads, device="cuda")
            bias = torch.randn(heads * dim, device="cuda")
            packed = torch.randn(
                batch, tokens, 4 * heads + 7, device="cuda", dtype=torch.bfloat16
            )
            beta = packed[..., 2 : 2 + heads]
            for lower_bound in (None, -5.0):
                with self.subTest(
                    varlen=cu_seqlens is not None, lower_bound=lower_bound
                ):
                    kwargs = dict(
                        A_log=a_log,
                        chunk_size=64,
                        dt_bias=bias,
                        cu_seqlens=cu_seqlens,
                        chunk_indices=chunks,
                        lower_bound=lower_bound,
                    )
                    expected_gate = kda_gate_chunk_cumsum(gate, **kwargs)
                    actual_gate, actual_beta = kda_gate_chunk_cumsum(
                        gate, beta=beta, **kwargs
                    )
                    torch.testing.assert_close(
                        actual_gate, expected_gate, atol=1e-4, rtol=1e-6
                    )
                    torch.testing.assert_close(
                        actual_beta, beta.float().sigmoid(), atol=1.2e-7, rtol=1e-6
                    )
                    self.assertEqual(actual_beta.dtype, torch.float32)

    @torch.inference_mode()
    def test_chunk_raw_beta_matches_activated_beta_and_final_state(self):
        torch.manual_seed(17)
        tokens, heads, dim = 68, 2, 64
        shape = (1, tokens, heads, dim)
        q, k, v, gate = [
            torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(4)
        ]
        packed = torch.randn(
            1,
            tokens,
            3 * heads * dim + heads + 2 * dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        layouts = (
            torch.randn(1, tokens, heads, device="cuda", dtype=torch.bfloat16),
            torch.randn(
                1, heads, tokens, device="cuda", dtype=torch.bfloat16
            ).transpose(1, 2),
            packed[..., 3 * heads * dim : 3 * heads * dim + heads],
        )
        a_log = torch.zeros(heads, device="cuda")
        bias = torch.randn(heads * dim, device="cuda")
        cu_seqlens = torch.tensor([0, 3, tokens], device="cuda", dtype=torch.int32)
        state = torch.randn(2, heads, dim, dim, device="cuda") * 0.01
        indices = torch.arange(2, device="cuda", dtype=torch.int32)
        for beta in layouts:
            for fused_gate in (False, True):
                with self.subTest(stride=beta.stride(), fused_gate=fused_gate):
                    kwargs = dict(
                        q=q,
                        k=k,
                        scale=dim**-0.5,
                        use_qk_l2norm_in_kernel=True,
                        initial_state_indices=indices,
                        cu_seqlens=cu_seqlens,
                        A_log=a_log if fused_gate else None,
                        dt_bias=bias if fused_gate else None,
                        lower_bound=-5.0 if fused_gate else None,
                    )
                    g = (
                        gate
                        if fused_gate
                        else -torch.nn.functional.softplus(gate.float())
                    )
                    expected_state, actual_state = state.clone(), state.clone()
                    expected = chunk_kda(
                        v=v.clone(),
                        g=g.clone(),
                        beta=beta.float().sigmoid(),
                        initial_state=expected_state,
                        **kwargs,
                    )
                    actual = chunk_kda(
                        v=v.clone(),
                        g=g.clone(),
                        beta=beta,
                        beta_is_raw=True,
                        initial_state=actual_state,
                        **kwargs,
                    )
                    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
                    torch.testing.assert_close(
                        actual_state, expected_state, atol=2e-4, rtol=2e-3
                    )


if __name__ == "__main__":
    unittest.main()
