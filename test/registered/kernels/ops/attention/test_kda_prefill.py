import unittest

import torch
import torch.nn.functional as F

from sglang.kernels.ops.attention.fla.kda import chunk_kda
from sglang.kernels.ops.attention.linear.kda_nvidia_prefill import (
    chunk_kda_fwd as nvidia_chunk_kda_fwd,
)
from sglang.kernels.ops.attention.linear.kda_ptx_prefill import (
    chunk_kda_fwd as ptx_chunk_kda_fwd,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=180, stage="base-b-kernel-unit", runner_config="4-gpu-b200")
register_cuda_ci(est_time=26, stage="base-c", runner_config="4-gpu-gb300")


def _inputs(seed, seq_len=128):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    batch_size, num_heads, head_dim = 1, 2, 128
    shape = (batch_size, seq_len, num_heads, head_dim)
    q = torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16)
    v = (
        0.1
        * torch.randn(
            shape,
            generator=generator,
            device="cuda",
            dtype=torch.float32,
        )
    ).to(torch.bfloat16)
    gate = torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16)
    beta_logits = torch.randn(
        shape[:-1],
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    a_log = torch.randn(
        num_heads, generator=generator, device="cuda", dtype=torch.float32
    )
    dt_bias = torch.randn(
        num_heads * head_dim,
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )
    state = torch.zeros(
        batch_size,
        num_heads,
        head_dim,
        head_dim,
        device="cuda",
        dtype=torch.float32,
    )
    return q, k, v, gate, beta_logits, a_log, dt_bias, state


def _reference(q, k, v, gate, beta, a_log, dt_bias, state, fused_qk_norm):
    return chunk_kda(
        q=q,
        k=k,
        v=v,
        g=gate,
        beta=beta,
        scale=q.shape[-1] ** -0.5,
        initial_state=state,
        initial_state_indices=torch.arange(
            q.shape[0], device="cuda", dtype=torch.int32
        ),
        use_qk_l2norm_in_kernel=fused_qk_norm,
        A_log=a_log,
        dt_bias=dt_bias,
        lower_bound=-5.0,
    )


class TestKdaPrefill(CustomTestCase):
    @torch.inference_mode()
    def test_nvidia_prefill(self):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
            self.skipTest("NVIDIA KDA prefill requires datacenter Blackwell")
        q, k, v, gate, beta_logits, a_log, dt_bias, state = _inputs(0)
        q = F.normalize(q.float(), dim=-1).to(torch.bfloat16)
        k = F.normalize(k.float(), dim=-1).to(torch.bfloat16)
        beta = torch.sigmoid(beta_logits.float()).to(torch.bfloat16)
        actual, actual_state = nvidia_chunk_kda_fwd(
            q=q,
            k=k,
            v=v,
            g=gate,
            beta=beta,
            scale=q.shape[-1] ** -0.5,
            initial_state=state.transpose(-1, -2).contiguous(),
            output_final_state=True,
            safe_gate=True,
            lower_bound=-5.0,
            use_gate_in_kernel=True,
            A_log=a_log,
            dt_bias=dt_bias,
        )[:2]
        expected = _reference(
            q, k, v, gate, beta, a_log, dt_bias, state, fused_qk_norm=False
        )
        torch.testing.assert_close(
            actual.float(), expected.float(), rtol=2e-2, atol=3e-2
        )
        torch.testing.assert_close(
            actual_state.transpose(-1, -2),
            state,
            rtol=2e-2,
            atol=3e-2,
        )

    @torch.inference_mode()
    def test_ptx_prefill(self):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (
            10,
            3,
        ):
            self.skipTest("PTX KDA prefill requires GB300")
        q, k, v, gate, beta_logits, a_log, dt_bias, state = _inputs(1)
        actual, actual_state = ptx_chunk_kda_fwd(
            q=q,
            k=k,
            v=v,
            g=gate,
            beta=beta_logits,
            scale=q.shape[-1] ** -0.5,
            initial_state=state.transpose(-1, -2).contiguous(),
            output_final_state=True,
            safe_gate=True,
            lower_bound=-5.0,
            use_gate_in_kernel=True,
            A_log=a_log,
            dt_bias=dt_bias,
            use_qk_l2norm_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
        )[:2]
        expected = _reference(
            q,
            k,
            v,
            gate,
            torch.sigmoid(beta_logits.float()).to(torch.bfloat16),
            a_log,
            dt_bias,
            state,
            fused_qk_norm=True,
        )
        torch.testing.assert_close(
            actual.float(), expected.float(), rtol=2e-2, atol=3e-2
        )
        torch.testing.assert_close(
            actual_state.transpose(-1, -2),
            state,
            rtol=2e-2,
            atol=3e-2,
        )


if __name__ == "__main__":
    unittest.main()
