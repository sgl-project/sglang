import unittest

import torch

from sglang.kernels.ops.attention.fla.kda import chunk_kda
from sglang.kernels.ops.attention.linear.kda_ptx_prefill import (
    chunk_kda_fwd as ptx_chunk_kda_fwd,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=180, stage="base-c", runner_config="4-gpu-gb300")


class TestKdaPtxPrefill(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        if torch.cuda.get_device_capability() != (10, 3):
            raise unittest.SkipTest("PTX KDA prefill requires GB300 (SM103)")

    @torch.inference_mode()
    def test_matches_fla_reference_with_fused_transforms(self):
        generator = torch.Generator(device="cuda").manual_seed(1)
        batch_size, seq_len, num_heads, head_dim = 1, 128, 2, 128
        shape = (batch_size, seq_len, num_heads, head_dim)

        q = torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16)
        v = (
            0.1
            * torch.randn(
                shape, generator=generator, device="cuda", dtype=torch.float32
            )
        ).to(torch.bfloat16)
        raw_gate = torch.randn(
            shape, generator=generator, device="cuda", dtype=torch.bfloat16
        )
        beta_logits = torch.randn(
            shape[:-1], generator=generator, device="cuda", dtype=torch.bfloat16
        )
        a_log = torch.randn(
            (num_heads,), generator=generator, device="cuda", dtype=torch.float32
        )
        dt_bias = torch.randn(
            (num_heads * head_dim,),
            generator=generator,
            device="cuda",
            dtype=torch.float32,
        )
        reference_state = torch.zeros(
            (batch_size, num_heads, head_dim, head_dim),
            device="cuda",
            dtype=torch.float32,
        )
        ptx_state = reference_state.transpose(-1, -2).contiguous()

        actual_result = ptx_chunk_kda_fwd(
            q=q,
            k=k,
            v=v,
            g=raw_gate,
            beta=beta_logits,
            scale=head_dim**-0.5,
            initial_state=ptx_state,
            output_final_state=True,
            safe_gate=True,
            lower_bound=-5.0,
            use_gate_in_kernel=True,
            A_log=a_log,
            dt_bias=dt_bias,
            use_qk_l2norm_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
        )
        actual_output, actual_state = actual_result[:2]
        expected_output = chunk_kda(
            q=q,
            k=k,
            v=v,
            g=raw_gate,
            beta=torch.sigmoid(beta_logits.float()).to(torch.bfloat16),
            scale=head_dim**-0.5,
            initial_state=reference_state,
            initial_state_indices=torch.arange(
                batch_size, device="cuda", dtype=torch.int32
            ),
            use_qk_l2norm_in_kernel=True,
            A_log=a_log,
            dt_bias=dt_bias,
            lower_bound=-5.0,
        )

        torch.testing.assert_close(
            actual_output.float(), expected_output.float(), rtol=2e-2, atol=3e-2
        )
        torch.testing.assert_close(
            actual_state.transpose(-1, -2),
            reference_state,
            rtol=2e-2,
            atol=3e-2,
        )


if __name__ == "__main__":
    unittest.main()
