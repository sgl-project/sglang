"""Compiled Mamba2 decode must update the per-layer state pool in place.

MambaPool hands each layer a select view of the all-layer pool. When the
decode state updates are traced as Triton kernels, Inductor clones that
per-layer pool (sized by max_running_requests) around every call. The
custom ops declare the mutation, so a decode step allocates only activations.
"""

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")

import unittest
from types import SimpleNamespace

import torch

from sglang.test.test_utils import CustomTestCase

NUM_LAYERS, POOL, BATCH = 2, 64, 2
NHEADS, HEAD_DIM, DSTATE, DIM, WIDTH = 32, 64, 128, 4096, 4
LAYER = 1


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestMamba2DecodeStateUpdateCompile(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.kernels.ops.mamba.triton_ops.ssu_dispatch import (
            initialize_mamba_selective_state_update_backend,
        )

        initialize_mamba_selective_state_update_backend(
            SimpleNamespace(
                mamba_backend="triton",
                enable_mamba_cache_stochastic_rounding=False,
                mamba_cache_philox_rounds=0,
            )
        )

    def _inputs(self):
        torch.manual_seed(0)
        dev, dtype = "cuda", torch.float16
        return [
            torch.randn(
                NUM_LAYERS, POOL, NHEADS, HEAD_DIM, DSTATE, device=dev, dtype=dtype
            ),
            torch.randn(NUM_LAYERS, POOL, DIM, WIDTH - 1, device=dev, dtype=dtype),
            torch.randn(BATCH, DIM, device=dev, dtype=dtype),
            torch.randn(DIM, WIDTH, device=dev, dtype=dtype),
            torch.randn(DIM, device=dev, dtype=dtype),
            torch.randn(BATCH, NHEADS, HEAD_DIM, device=dev, dtype=dtype),
            torch.randn(BATCH, NHEADS, HEAD_DIM, device=dev, dtype=dtype),
            -torch.rand(NHEADS, HEAD_DIM, DSTATE, device=dev, dtype=torch.float32),
            torch.randn(BATCH, 1, DSTATE, device=dev, dtype=dtype),
            torch.randn(BATCH, 1, DSTATE, device=dev, dtype=dtype),
            torch.randn(NHEADS, HEAD_DIM, device=dev, dtype=dtype),
            torch.randn(NHEADS, HEAD_DIM, device=dev, dtype=dtype),
            torch.tensor([3, 5], device=dev, dtype=torch.int32),
        ]

    @staticmethod
    def _step(ssm_pool, conv_pool, xc, weight, bias, x, dt, A, B, C, D, dt_bias, idx):
        from sglang.srt.layers.attention.mamba.mamba import (
            mamba2_decode_conv_state_update,
            mamba2_decode_ssm_state_update,
        )

        conv_out = mamba2_decode_conv_state_update(
            xc, conv_pool[LAYER], weight, bias, "silu", idx
        )
        out = torch.empty_like(x)
        mamba2_decode_ssm_state_update(
            ssm_pool[LAYER], x, dt, A, B, C, D, dt_bias, idx, out
        )
        return conv_out, out

    def test_compiled_matches_eager_without_copying_the_pool(self):
        eager_args = self._inputs()
        compiled_args = self._inputs()
        compiled = torch.compile(self._step, fullgraph=True)

        want = self._step(*eager_args)
        got = compiled(*[a.clone() for a in compiled_args])
        for w, g in zip(want, got):
            torch.testing.assert_close(g, w, rtol=0, atol=0)

        compiled(*compiled_args)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        compiled(*compiled_args)
        torch.cuda.synchronize()
        step_bytes = torch.cuda.max_memory_allocated() - before

        layer_pool_bytes = compiled_args[0][LAYER].nbytes
        self.assertLess(step_bytes, layer_pool_bytes // 8)

        # Both steps advanced the same two pool slots; the rest are untouched.
        reference = self._inputs()
        self._step(*reference)
        self._step(*reference)
        for pool in (0, 1):
            torch.testing.assert_close(
                compiled_args[pool], reference[pool], rtol=0, atol=0
            )


if __name__ == "__main__":
    unittest.main()
