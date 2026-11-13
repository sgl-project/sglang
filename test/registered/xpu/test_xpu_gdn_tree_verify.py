"""Numerical parity of the XPU GDN target-verify kernel with Triton on a
branching draft tree (siblings and multi-level parents), including the
per-node intermediate recurrent states the commit step reads back."""

import unittest

import torch

from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.layers.attention.linear.kernels.gdn_xpu import XpuGDNKernel
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=30, suite="stage-b-test-1-gpu-xpu")

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
    def _run(self, kernel, parents, seed=0):
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
        buffer = torch.zeros(num_req + 1, steps, HV, V, K, device=dev)
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


if __name__ == "__main__":
    unittest.main()
