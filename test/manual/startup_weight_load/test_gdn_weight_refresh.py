"""Refresh FlashInfer GDN parameters after graph capture."""

import unittest

import torch

from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
    FlashInferGDNKernel,
)
from sglang.test.test_utils import CustomTestCase


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestGDNWeightRefresh(CustomTestCase):
    @torch.no_grad()
    def test_decode_replay_uses_refreshed_parameters(self):
        torch.manual_seed(42)
        kernel = FlashInferGDNKernel()
        q = torch.randn(1, 1, 2, 128, dtype=torch.bfloat16, device="cuda")
        k = torch.randn_like(q)
        v = torch.randn(1, 1, 8, 128, dtype=torch.bfloat16, device="cuda")
        a = torch.randn(1, 1, 8, dtype=torch.bfloat16, device="cuda")
        b = torch.randn_like(a)
        # Both parameters require separate aligned storage, even on SM90.
        A_log = torch.full((9,), -2.0, device="cuda")[1:]
        dt_bias = torch.full((9,), -3.0, dtype=torch.bfloat16, device="cuda")[1:]
        initial_state = torch.randn(1, 8, 128, 128, device="cuda")
        state = torch.empty_like(initial_state)
        cache_indices = torch.zeros(1, dtype=torch.int32, device="cuda")
        query_start_loc = torch.tensor([0, 1], dtype=torch.int32, device="cuda")

        def run(active_kernel):
            state.copy_(initial_state)
            return active_kernel.decode(
                q,
                k,
                v,
                a,
                b,
                A_log=A_log,
                dt_bias=dt_bias,
                ssm_states=state,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
            )

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                run(kernel)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run(kernel)
        graph.replay()
        sentinel_output = captured.clone()
        cache_before = {
            key: (prepared, prepared.data_ptr())
            for key, (_, prepared) in kernel._aligned_parameter_cache.items()
        }

        A_log.fill_(1)
        dt_bias.fill_(1)
        graph.replay()
        torch.testing.assert_close(captured, sentinel_output, rtol=0, atol=0)

        kernel.on_after_weight_load()
        graph.replay()
        actual_output = captured.clone()
        actual_state = state.clone()
        expected_output = run(FlashInferGDNKernel())
        torch.testing.assert_close(actual_output, expected_output, rtol=0, atol=0)
        torch.testing.assert_close(actual_state, state, rtol=0, atol=0)
        self.assertFalse(torch.equal(actual_output, sentinel_output))
        for key, (prepared, pointer) in cache_before.items():
            self.assertIs(kernel._aligned_parameter_cache[key][1], prepared)
            self.assertEqual(prepared.data_ptr(), pointer)


if __name__ == "__main__":
    unittest.main()
