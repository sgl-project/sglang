"""Narrow chain verification must commit each request's own GDN snapshots."""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestGDNVerifyNarrowSnapshots(CustomTestCase):
    def test_pool_width_matches_exact_width_in_eager_and_graph(self):
        if torch.cuda.get_device_capability()[0] != 10:
            self.skipTest("BF16 FlashInfer MTP requires SM100/SM103")
        from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
            FlashInferGDNKernel,
        )

        kernel = FlashInferGDNKernel()
        generator = torch.Generator(device="cuda").manual_seed(721)

        def rand(*shape, dtype=torch.bfloat16):
            return torch.randn(shape, device="cuda", dtype=dtype, generator=generator)

        for bs in (1, 2, 8):
            for width in (1, 2, 4, 8):
                for captured in (False, True):
                    with self.subTest(bs=bs, width=width, captured=captured):
                        heads, value_heads, dim = 4, 8, 128
                        state = rand(bs + 2, value_heads, dim, dim)
                        initial_state = state.clone()
                        args = dict(
                            A_log=rand(value_heads, dtype=torch.float32),
                            dt_bias=rand(value_heads),
                            q=rand(1, bs * width, heads, dim),
                            k=rand(1, bs * width, heads, dim),
                            v=rand(1, bs * width, value_heads, dim),
                            a=rand(bs * width, value_heads),
                            b=rand(bs * width, value_heads),
                            ssm_states=state,
                            cache_indices=torch.arange(
                                1, bs + 1, device="cuda", dtype=torch.int32
                            ),
                            query_start_loc=torch.arange(
                                bs + 1, device="cuda", dtype=torch.int32
                            )
                            * width,
                            intermediate_state_indices=torch.arange(
                                bs, device="cuda", dtype=torch.int32
                            ),
                            retrieve_parent_token=None,
                        )
                        exact = torch.full(
                            (bs, width, value_heads, dim, dim),
                            float("nan"),
                            device="cuda",
                            dtype=torch.bfloat16,
                        )
                        pool = torch.full(
                            (bs, 8, value_heads, dim, dim),
                            float("nan"),
                            device="cuda",
                            dtype=torch.bfloat16,
                        )
                        reference = kernel.target_verify(
                            **args, intermediate_states_buffer=exact, cache_steps=width
                        ).clone()

                        def run():
                            return kernel.target_verify(
                                **args, intermediate_states_buffer=pool, cache_steps=8
                            )

                        actual = run()
                        if captured:
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):
                                actual = run()
                            pool.fill_(float("nan"))
                            graph.replay()
                        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                        torch.testing.assert_close(
                            pool[:, :width], exact, rtol=0, atol=0
                        )
                        self.assertTrue(torch.isnan(pool[:, width:]).all().item())
                        torch.testing.assert_close(state, initial_state, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
