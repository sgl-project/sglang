"""XPU GDN / KDA recurrent kernels: target-verify parity with Triton on
branching draft trees, the intermediate-state buffer pitch, and the 4-D KDA
gate stride."""

import unittest

import torch

from sglang.srt.hardware_backend.xpu.kernels.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.layers.attention.linear.kernels.gdn_xpu import XpuGDNKernel
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=30, suite="extra-test-1-gpu-xpu")

# parent[i] is the in-request index of draft node i's parent; node 0 is the
# root (parent unused). Nodes 1 and 2 are siblings, as are 3/4 and 5/6.
TREES = {
    "branching": [-1, 0, 0, 1, 1, 2, 2, 3],
    "wide_fan_out": [-1, 0, 0, 0, 0, 1, 1, 2],
    "chain": [-1, 0, 1, 2, 3, 4, 5, 6],
}


@unittest.skipUnless(
    hasattr(torch, "xpu") and torch.xpu.is_available(), "requires an XPU device"
)
class TestXpuGdnTreeVerify(CustomTestCase):
    def _run(self, kernel, parents, seed=0, buffer_steps=None):
        torch.manual_seed(seed)
        dev = "xpu"
        num_req, steps = len(parents), len(parents[0])
        H, HV, K, V = 2, 4, 128, 128
        tokens = num_req * steps
        q = torch.randn(1, tokens, H, K, device=dev, dtype=torch.bfloat16)
        k = torch.randn(1, tokens, H, K, device=dev, dtype=torch.bfloat16)
        v = torch.randn(1, tokens, HV, V, device=dev, dtype=torch.bfloat16)
        a = torch.randn(tokens, HV, device=dev, dtype=torch.bfloat16)
        b = torch.randn(tokens, HV, device=dev, dtype=torch.bfloat16)
        A_log = torch.randn(HV, device=dev, dtype=torch.float32)
        dt_bias = torch.randn(HV, device=dev, dtype=torch.bfloat16)
        ssm_states = torch.randn(num_req + 1, HV, V, K, device=dev) * 0.1
        buffer = torch.zeros(num_req + 1, buffer_steps or steps, HV, V, K, device=dev)
        indices = torch.arange(num_req, device=dev, dtype=torch.int32)
        out = kernel.target_verify(
            A_log,
            dt_bias,
            q,
            k,
            v,
            a.view(1, tokens, HV) if isinstance(kernel, TritonGDNKernel) else a,
            b.view(1, tokens, HV) if isinstance(kernel, TritonGDNKernel) else b,
            ssm_states=ssm_states,
            cache_indices=indices,
            query_start_loc=torch.arange(
                0, tokens + 1, steps, device=dev, dtype=torch.int32
            ),
            intermediate_states_buffer=buffer,
            intermediate_state_indices=indices,
            cache_steps=steps,
            retrieve_parent_token=torch.tensor(parents, device=dev, dtype=torch.int32),
        )
        return out, buffer, ssm_states

    def test_branching_trees_match_triton(self):
        for name, parent in TREES.items():
            with self.subTest(tree=name):
                parents = [parent, parent[:1] + [0] * (len(parent) - 1)]
                ref_out, ref_buf, ref_ssm = self._run(TritonGDNKernel(), parents)
                out, buf, ssm = self._run(XpuGDNKernel(), parents)
                torch.testing.assert_close(
                    out.float(), ref_out.float(), atol=2e-2, rtol=2e-2
                )
                torch.testing.assert_close(buf, ref_buf, atol=2e-2, rtol=2e-2)
                # Verify must not commit state; only the commit step does.
                torch.testing.assert_close(ssm, ref_ssm, atol=0, rtol=0)

    def test_triton_verify_uses_allocated_buffer_pitch(self):
        """Adaptive spec verifies fewer drafts than the buffer was sized for; each
        request's states must still land at its allocated per-request pitch."""
        parent = TREES["branching"]
        parents = [parent, parent]
        steps = len(parent)
        ref_out, ref_buf, _ = self._run(TritonGDNKernel(), parents)
        out, buf, _ = self._run(TritonGDNKernel(), parents, buffer_steps=steps + 3)
        torch.testing.assert_close(out, ref_out, atol=0, rtol=0)
        torch.testing.assert_close(buf[:, :steps], ref_buf, atol=0, rtol=0)
        self.assertEqual(torch.count_nonzero(buf[:, steps:]).item(), 0)


@unittest.skipUnless(
    hasattr(torch, "xpu") and torch.xpu.is_available(), "requires an XPU device"
)
class TestXpuRecurrentWrapperKdaGate(CustomTestCase):
    def _run(self, gate_layout):
        torch.manual_seed(0)
        dev = "xpu"
        B, T, HV, K = 2, 4, 4, 128
        q = torch.randn(B, T, HV, K, device=dev, dtype=torch.bfloat16)
        k = torch.randn(B, T, HV, K, device=dev, dtype=torch.bfloat16)
        v = torch.randn(B, T, HV, K, device=dev, dtype=torch.bfloat16)
        a = torch.randn(B, T, HV, K, device=dev, dtype=torch.bfloat16)
        b = torch.randn(B, T, HV, device=dev, dtype=torch.bfloat16)
        state = torch.randn(B, HV, K, K, device=dev) * 0.1
        out = fused_sigmoid_gating_delta_rule_update(
            A_log=torch.randn(HV, device=dev, dtype=torch.float32),
            a=a if gate_layout == "4d" else a.view(B, T, HV * K),
            dt_bias=torch.randn(HV * K, device=dev, dtype=torch.float32),
            softplus_beta=1.0,
            softplus_threshold=20.0,
            q=q,
            k=k,
            v=v,
            b=b,
            initial_state_source=state,
            initial_state_indices=torch.arange(B, device=dev, dtype=torch.int32),
            use_qk_l2norm_in_kernel=True,
            is_kda=True,
        )
        return out, state

    def test_4d_gate_advances_by_token_stride(self):
        """A [B, T, H, K] KDA gate must step by its token stride, not its head
        stride; it reads the same values as the [B, T, H*K] layout."""
        ref_out, ref_state = self._run("3d")
        out, state = self._run("4d")
        torch.testing.assert_close(out, ref_out, atol=0, rtol=0)
        torch.testing.assert_close(state, ref_state, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
