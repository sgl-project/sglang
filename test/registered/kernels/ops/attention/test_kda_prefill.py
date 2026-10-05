import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.kernels.ops.attention.fla.kda import chunk_kda
from sglang.kernels.ops.attention.linear.kda_nvidia_prefill import (
    chunk_kda_fwd as nvidia_chunk_kda_fwd,
)
from sglang.kernels.ops.attention.linear.kda_ptx_prefill import (
    SM_ARCHS,
)
from sglang.kernels.ops.attention.linear.kda_ptx_prefill import (
    chunk_kda_fwd as ptx_chunk_kda_fwd,
)
from sglang.srt.layers.attention.linear.kernels.kda_ptx import PtxKDAKernel
from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=300, stage="base-b-kernel-unit", runner_config="4-gpu-b200")
register_cuda_ci(est_time=223, stage="base-c", runner_config="4-gpu-gb300")


def _ptx_supported():
    return torch.cuda.is_available() and torch.cuda.get_device_capability() in SM_ARCHS


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
    def test_ptx_padded_raw_beta(self):
        """Raw beta must match Triton, including final state after neutral padding."""
        if not _ptx_supported():
            self.skipTest("PTX KDA prefill requires SM100 or SM103")
        q, k, v, gate, beta, a_log, dt_bias, state = _inputs(2, seq_len=1025)
        state.fill_(0.1)
        actual_state = state.clone()
        inputs = dict(
            q=q,
            k=k,
            v=v,
            g=gate,
            beta=beta,
            cache_indices=torch.zeros(1, device="cuda", dtype=torch.int32),
            query_start_loc=torch.tensor([0, 1025], device="cuda", dtype=torch.int32),
            A_log=a_log,
            dt_bias=dt_bias,
            lower_bound=-5.0,
            beta_is_raw=True,
            extend_seq_lens_cpu=[1025],
        )
        kernel = PtxKDAKernel()
        with patch.object(
            kernel._triton,
            "extend",
            side_effect=AssertionError("PTX unexpectedly fell back to Triton"),
        ):
            actual = kernel.extend(**inputs, ssm_states=actual_state)
        # Triton may mutate inputs, so run the reference last.
        expected = TritonKDAKernel().extend(**inputs, ssm_states=state)
        torch.testing.assert_close(
            actual.float(), expected.float(), rtol=2e-2, atol=3e-2
        )
        torch.testing.assert_close(actual_state, state, rtol=2e-2, atol=3e-2)

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
        if not _ptx_supported():
            self.skipTest("PTX KDA prefill requires SM100 or SM103")
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

    @torch.inference_mode()
    def test_ptx_without_lower_bound_matches_triton(self):
        """Without a gate lower bound (Kimi-Linear) the kernel's per-chunk decay
        overflows to NaN; prefill must still return Triton's result."""
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (
            10,
            3,
        ):
            self.skipTest("PTX KDA prefill requires GB300")

        def run(kernel):
            # Fresh inputs per call: Triton may mutate them.
            q, k, v, gate, beta, _, dt_bias, state = _inputs(3, seq_len=256)
            out = kernel.extend(
                q=q,
                k=k,
                v=v,
                g=gate,
                beta=beta,
                ssm_states=state,
                cache_indices=torch.zeros(1, device="cuda", dtype=torch.int32),
                query_start_loc=torch.tensor(
                    [0, 256], device="cuda", dtype=torch.int32
                ),
                # exp(A_log) ~ 200, the top of Kimi-Linear-48B's first KDA layer.
                A_log=torch.full((q.shape[2],), 5.3, device="cuda"),
                dt_bias=dt_bias,
                lower_bound=None,
                beta_is_raw=True,
                extend_seq_lens_cpu=[256],
            )
            return out, state

        actual, actual_state = run(PtxKDAKernel())
        expected, expected_state = run(TritonKDAKernel())
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(
            actual.float(), expected.float(), rtol=2e-2, atol=3e-2
        )
        torch.testing.assert_close(actual_state, expected_state, rtol=2e-2, atol=3e-2)

    @torch.inference_mode()
    def test_ptx_workspace_memory_is_bounded(self):
        """Each new (tokens, sequences) shape gets a kernel workspace, and serving
        sees a new shape almost every batch: the device memory they hold must stay
        bounded instead of growing with every shape seen."""
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (
            10,
            3,
        ):
            self.skipTest("PTX KDA prefill requires GB300")
        q, k, v, gate, beta, a_log, dt_bias, _ = _inputs(4, seq_len=2048)
        num_heads, chunks = q.shape[2], 2048 // 64
        retained = []
        prev = torch.cuda.memory_allocated()
        for num_seqs in range(1, 17):
            base, extra = divmod(chunks, num_seqs)
            lens = [64 * (base + (i < extra)) for i in range(num_seqs)]
            cu_cpu = torch.tensor([0, *lens], dtype=torch.int32).cumsum(0).int()
            out, final_state = ptx_chunk_kda_fwd(
                q=q,
                k=k,
                v=v,
                g=gate,
                beta=beta,
                scale=q.shape[-1] ** -0.5,
                initial_state=torch.zeros(num_seqs, num_heads, 128, 128, device="cuda"),
                output_final_state=True,
                cu_seqlens=cu_cpu.cuda(),
                cu_seqlens_cpu=cu_cpu,
                safe_gate=True,
                lower_bound=-5.0,
                use_gate_in_kernel=True,
                A_log=a_log,
                dt_bias=dt_bias,
                use_qk_l2norm_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
            )[:2]
            del out, final_state
            torch.cuda.synchronize()
            now = torch.cuda.memory_allocated()
            retained.append(now - prev)
            prev = now
        # An unbounded cache keeps one more workspace per shape, 12 over the last
        # 12 shapes; a bounded one holds a few whatever it held before the test.
        self.assertLess(sum(retained[4:]), 4 * max(retained))


if __name__ == "__main__":
    unittest.main()
