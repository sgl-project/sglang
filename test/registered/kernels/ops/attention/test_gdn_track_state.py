"""Preserve the FP32 GDN checkpoint at an interior prefill boundary."""

import unittest

import torch

from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=25, stage="base-b-kernel-unit", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestGdnTrackState(CustomTestCase):
    @torch.inference_mode()
    def test_interior_snapshot_matches_fp32_prefix_final_state(self):
        for dim in (64, 128):
            with self.subTest(dim=dim):
                self._check_snapshot(dim)

    def _check_snapshot(self, dim):
        torch.manual_seed(7)
        heads, total = 2, 164
        q, k, v = [
            torch.randn(1, total, heads, dim, device="cuda", dtype=torch.bfloat16) * 0.1
            for _ in range(3)
        ]
        g = -torch.rand(1, total, heads, device="cuda") * 0.1
        beta = torch.rand(1, total, heads, device="cuda", dtype=torch.bfloat16)
        initial = torch.randn(3, heads, dim, dim, device="cuda") * 0.01
        indices = torch.tensor([2, 0], device="cuda", dtype=torch.int32)
        starts = torch.tensor([0, 100, 164], device="cuda", dtype=torch.int32)
        snapshot = torch.full_like(initial[:2], float("nan"))
        kernel = TritonGDNKernel()
        self.assertTrue(kernel.supports_track_state_snapshot)

        def run(state, **kwargs):
            return kernel.extend(
                q,
                k,
                v,
                g,
                beta,
                ssm_states=state,
                cache_indices=indices,
                query_start_loc=starts,
                **kwargs,
            )

        plain_state, tracked_state = initial.clone(), initial.clone()
        plain_output, _, plain_h = run(plain_state)
        output, _, h = run(
            tracked_state,
            track_state=snapshot,
            track_chunk_idx=torch.tensor([1, -1], device="cuda", dtype=torch.int32),
        )
        torch.testing.assert_close(output, plain_output, rtol=0, atol=0)
        torch.testing.assert_close(tracked_state, plain_state, rtol=0, atol=0)
        torch.testing.assert_close(h, plain_h, rtol=0, atol=0)
        self.assertTrue(torch.isnan(snapshot[1]).all())
        self.assertTrue(torch.isfinite(snapshot[0]).all())

        # Main's mapped output must coexist with the replay checkpoint snapshot.
        mapped = torch.full(
            (1, total * 2, heads, dim), 7, device=q.device, dtype=q.dtype
        )
        mapped_state = initial.clone()
        mapped_snapshot = torch.full_like(snapshot, float("nan"))
        mapped_output, _, _ = run(
            mapped_state,
            out=mapped,
            output_indices=torch.arange(0, total * 2, 2, device=q.device),
            track_state=mapped_snapshot,
            track_chunk_idx=torch.tensor([1, -1], device=q.device, dtype=torch.int32),
        )
        torch.testing.assert_close(mapped_output[:, ::2], plain_output, rtol=0, atol=0)
        self.assertTrue((mapped_output[:, 1::2] == 7).all())
        torch.testing.assert_close(mapped_state, plain_state, rtol=0, atol=0)
        torch.testing.assert_close(mapped_snapshot[0], snapshot[0], rtol=0, atol=0)
        self.assertTrue(torch.isnan(mapped_snapshot[1]).all())

        reference = initial.clone()
        chunk_gated_delta_rule(
            q[:, :64].contiguous(),
            k[:, :64].contiguous(),
            v[:, :64].contiguous(),
            g[:, :64].contiguous(),
            beta[:, :64].contiguous(),
            initial_state=reference,
            initial_state_indices=indices[:1],
            use_qk_l2norm_in_kernel=True,
        )
        torch.testing.assert_close(snapshot[0], reference[2], rtol=0, atol=0)
        self.assertFalse(torch.equal(h[0, 1].float(), reference[2]))
        torch.testing.assert_close(h[0, 1], snapshot[0].to(h.dtype), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
