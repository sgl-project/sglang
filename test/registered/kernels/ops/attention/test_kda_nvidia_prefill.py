import unittest

import torch
import torch.nn.functional as F

from sglang.kernels.ops.attention.fla.kda import chunk_kda
from sglang.kernels.ops.attention.linear.kda_nvidia_prefill import (
    chunk_kda_fwd as nvidia_chunk_kda_fwd,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=180, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


class TestKdaNvidiaPrefill(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        if torch.cuda.get_device_capability()[0] != 10:
            raise unittest.SkipTest("NVIDIA KDA prefill requires datacenter Blackwell")

    @torch.inference_mode()
    def test_matches_fla_reference(self):
        generator = torch.Generator(device="cuda").manual_seed(0)
        batch_size, seq_len, num_heads, head_dim = 1, 256, 2, 128
        shape = (batch_size, seq_len, num_heads, head_dim)

        q = F.normalize(
            torch.randn(shape, generator=generator, device="cuda", dtype=torch.float32),
            dim=-1,
        ).to(torch.bfloat16)
        k = F.normalize(
            torch.randn(shape, generator=generator, device="cuda", dtype=torch.float32),
            dim=-1,
        ).to(torch.bfloat16)
        v = (
            0.1
            * torch.randn(
                shape, generator=generator, device="cuda", dtype=torch.float32
            )
        ).to(torch.bfloat16)
        raw_gate = torch.randn(
            shape, generator=generator, device="cuda", dtype=torch.bfloat16
        )
        beta = torch.sigmoid(
            torch.randn(
                shape[:-1],
                generator=generator,
                device="cuda",
                dtype=torch.float32,
            )
        ).to(torch.bfloat16)
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
        nvidia_state = reference_state.transpose(-1, -2).contiguous()

        actual_result = nvidia_chunk_kda_fwd(
            q=q,
            k=k,
            v=v,
            g=raw_gate,
            beta=beta,
            scale=head_dim**-0.5,
            initial_state=nvidia_state,
            output_final_state=True,
            safe_gate=True,
            lower_bound=-5.0,
            use_gate_in_kernel=True,
            A_log=a_log,
            dt_bias=dt_bias,
        )
        actual_output, actual_state = actual_result[:2]
        expected_output = chunk_kda(
            q=q,
            k=k,
            v=v,
            g=raw_gate,
            beta=beta,
            scale=head_dim**-0.5,
            initial_state=reference_state,
            initial_state_indices=torch.arange(
                batch_size, device="cuda", dtype=torch.int32
            ),
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
