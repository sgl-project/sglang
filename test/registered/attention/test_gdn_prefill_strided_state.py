"""GDN prefill kernels on an envelope-strided state pool.

GDNAttnBackend runs prefill in place on a pool whose slots are pitched wider than
their state (unified memory) when the conv and extend kernels follow that pitch.
Each case runs a kernel on such a pool and on a packed pool holding the same
values: outputs and states must match, and the bytes between slots stay untouched.
"""

import unittest

import torch

from sglang.srt.layers.attention.mamba.causal_conv1d import causal_conv1d_fn
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-large")

# Arbitrary; any gap breaks the packing.
PAD = 7


def envelope(pool: torch.Tensor):
    """`pool`'s values with every slot PAD elements apart, NaN in between."""
    slots, per_slot = pool.shape[0], pool[0].numel()
    base = torch.full(
        (slots, per_slot + PAD), float("nan"), dtype=pool.dtype, device=pool.device
    )
    view = base[:, :per_slot].view(pool.shape)
    view.copy_(pool)
    return view, base


@unittest.skipUnless(torch.cuda.is_available(), "Test requires CUDA")
class TestGDNPrefillStridedStatePool(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.device = "cuda"
        # Three sequences of unequal length on non-adjacent slots of six.
        self.lens = [5, 17, 64]
        self.cache_indices = torch.tensor(
            [4, 0, 2], dtype=torch.int32, device=self.device
        )
        self.query_start_loc = torch.tensor(
            [0, 5, 22, 86], dtype=torch.int32, device=self.device
        )

    def assert_same_pool(self, packed, strided, base):
        torch.testing.assert_close(strided, packed)
        self.assertTrue(torch.isnan(base[:, packed[0].numel() :]).all())

    def gdn_inputs(self, qk_heads, v_heads, head_dim=128):
        tokens, dtype = sum(self.lens), torch.bfloat16
        q = torch.randn(1, tokens, qk_heads, head_dim, dtype=dtype, device=self.device)
        k = torch.randn(1, tokens, qk_heads, head_dim, dtype=dtype, device=self.device)
        v = torch.randn(1, tokens, v_heads, head_dim, dtype=dtype, device=self.device)
        g = torch.nn.functional.logsigmoid(
            torch.randn(1, tokens, v_heads, dtype=dtype, device=self.device)
        )
        beta = torch.sigmoid(
            torch.randn(1, tokens, v_heads, dtype=dtype, device=self.device)
        )
        packed = (
            torch.randn(
                6, v_heads, head_dim, head_dim, dtype=torch.float32, device=self.device
            )
            * 0.1
        )
        return (q, k, v, g, beta), packed

    def check_extend(self, kernel, qk_heads, v_heads):
        args, packed = self.gdn_inputs(qk_heads, v_heads)
        strided, base = envelope(packed)
        outputs = [
            kernel.extend(
                *args,
                ssm_states=pool,
                cache_indices=self.cache_indices,
                query_start_loc=self.query_start_loc,
            )[0]
            for pool in (packed, strided)
        ]
        torch.testing.assert_close(outputs[1], outputs[0])
        self.assert_same_pool(packed, strided, base)

    def test_flashinfer_extend(self):
        try:
            from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
                FlashInferGDNKernel,
            )

            kernel = FlashInferGDNKernel()
        except (ImportError, RuntimeError) as exc:
            self.skipTest(f"FlashInfer GDN prefill is unavailable: {exc}")
        self.check_extend(kernel, qk_heads=4, v_heads=8)

    def test_triton_conv(self):
        dim, width, dtype = 256, 4, torch.bfloat16
        x = torch.randn(sum(self.lens), dim, dtype=dtype, device=self.device)
        weight = torch.randn(dim, width, dtype=dtype, device=self.device)
        bias = torch.randn(dim, dtype=dtype, device=self.device)
        has_initial_state = torch.tensor([True, False, True], device=self.device)
        packed = torch.randn(6, dim, width - 1, dtype=dtype, device=self.device)
        strided, base = envelope(packed)
        # GDN prefill hands the conv a transposed input and host lengths, which
        # routes it to the Triton kernel.
        outputs = [
            causal_conv1d_fn(
                x.transpose(0, 1),
                weight,
                bias,
                activation="silu",
                conv_states=pool,
                has_initial_state=has_initial_state,
                cache_indices=self.cache_indices,
                query_start_loc=self.query_start_loc,
                seq_lens_cpu=self.lens,
            )
            for pool in (packed, strided)
        ]
        torch.testing.assert_close(outputs[1], outputs[0])
        self.assert_same_pool(packed, strided, base)


if __name__ == "__main__":
    unittest.main()
