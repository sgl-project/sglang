"""Shape and graph contracts for opt-in CUDA shared-KV full-attention verify."""

import unittest
from functools import partial

import torch

from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd
from sglang.kernels.ops.attention.verify_mla import verify_shared_kv_fwd
from sglang.srt.layers.attention.triton_backend import _can_use_cuda_verify
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


@unittest.skipIf(not torch.cuda.is_available(), "GPU required")
class TestCudaSharedKVVerify(CustomTestCase):
    def test_cancelling_prefix_values(self):
        if torch.version.hip or torch.cuda.get_device_capability() not in (
            (8, 9),
            (9, 0),
        ):
            self.skipTest("CUDA tile validated on Ada and Hopper")
        # Zero Q/K make attention uniform. All V entries are exactly BF16
        # representable; the prefix sums to 7 * 0.25 = 1.75 in every channel.
        # Rounding positive/negative split averages before merging loses that
        # residual, even though the final BF16 result can represent it well.
        q = torch.zeros(3, 16, 512, dtype=torch.bfloat16, device="cuda")
        k = torch.zeros(3, 1, 512, dtype=q.dtype, device=q.device)
        v = torch.zeros_like(k)
        kb = torch.zeros(32, 1, 512, dtype=q.dtype, device=q.device)
        vb = torch.full_like(kb, 32)
        vb[:7] = 32.25
        vb[16:] = -32
        qo = torch.tensor([0, 3], dtype=torch.int32, device=q.device)
        kp = torch.tensor([0, 32], dtype=torch.int32, device=q.device)
        ids = torch.arange(32, dtype=torch.int32, device=q.device)
        expected = (1.75 / torch.arange(33, 36, device=q.device).double())[
            :, None, None
        ].expand_as(q)
        for kernel in (extend_attention_fwd, verify_shared_kv_fwd):
            out = torch.empty_like(q)
            kernel(
                q,
                k,
                v,
                out,
                kb,
                vb,
                qo,
                kp,
                ids,
                None,
                True,
                None,
                3,
                1.0,
                1.0,
                sm_scale=512**-0.5,
            )
            torch.testing.assert_close(out.double(), expected, atol=2e-2, rtol=1e-2)

    def test_shape_contract(self):
        # The contract is device independent; pass a validated arch explicitly.
        eligible = partial(_can_use_cuda_verify, arch=(9, 0))
        q = torch.empty(6, 16, 512, dtype=torch.bfloat16, device="cuda")
        k = torch.empty(6, 1, 512, dtype=torch.bfloat16, device="cuda")
        pool = torch.empty(64, 1, 512, dtype=torch.bfloat16, device="cuda")
        self.assertTrue(eligible(q, k, k, pool, pool, 3))
        for width in (0, 5, None):
            self.assertFalse(eligible(q, k, k, pool, pool, width))
        self.assertFalse(eligible(q, None, None, pool, pool, 3))
        self.assertFalse(eligible(q.float(), k, k, pool, pool, 3))
        self.assertFalse(eligible(q, k, k, pool.float(), pool, 3))
        self.assertFalse(eligible(q[:, :8], k, k, pool, pool, 3))
        self.assertFalse(eligible(q, k[:3], k, pool, pool, 3))
        strided_pool = torch.empty(64, 1, 1024, dtype=torch.bfloat16, device="cuda")[
            :, :, ::2
        ]
        self.assertFalse(eligible(q, k, k, strided_pool, strided_pool, 3))
        # QKV projection views can have a larger token stride; the kernel
        # receives that stride explicitly and only requires dense head elements.
        projection = torch.empty(6, 18, 512, dtype=torch.bfloat16, device="cuda")
        self.assertTrue(
            eligible(
                projection[:, :16],
                projection[:, 16:17],
                projection[:, 17:],
                pool,
                pool,
                3,
            )
        )

    def test_head512_graph_replay(self):
        if torch.version.hip or torch.cuda.get_device_capability() not in (
            (8, 9),
            (9, 0),
        ):
            self.skipTest("CUDA tile validated on Ada and Hopper")
        generator = torch.Generator(device="cuda").manual_seed(20261013)
        device = "cuda"
        lengths = [0, 33, 4097]
        pool_size = 16 * ((sum(lengths) + 15) // 16)

        def rand(*shape):
            return torch.randn(
                *shape, dtype=torch.bfloat16, device=device, generator=generator
            )

        for width in (1, 2, 3, 4):
            with self.subTest(width=width):
                # Real QKV projection views have non-contiguous token strides.
                projection = rand(3 * width, 36, 512)
                q, k, v = (
                    projection[:, :32:2],
                    projection[:, 32:33],
                    projection[:, 34:35],
                )
                kb, vb = rand(pool_size, 1, 512), rand(pool_size, 1, 512)
                pages = torch.randperm(
                    pool_size // 16, device=device, generator=generator
                )
                ids = (pages[:, None] * 16 + torch.arange(16, device=device)).flatten()
                ptr = torch.tensor([0, 0, 33, 4130], dtype=torch.int32, device=device)
                qo = torch.arange(4, device=device, dtype=torch.int32) * width
                actual, generic = torch.empty_like(q), torch.empty_like(q)

                def invoke(fn, out):
                    return fn(
                        q,
                        k,
                        v,
                        out,
                        kb,
                        vb,
                        qo,
                        ptr,
                        ids,
                        None,
                        True,
                        None,
                        width,
                        1.0,
                        1.0,
                        sm_scale=512**-0.5,
                    )

                def reference(current_lengths):
                    # FP64 arithmetic avoids depending on either Triton kernel
                    # or the global TF32 matmul setting for the numerical oracle.
                    result = torch.empty_like(q, dtype=torch.float64)
                    start = 0
                    for batch, length in enumerate(current_lengths):
                        rows = slice(batch * width, (batch + 1) * width)
                        physical = ids[start : start + length]
                        keys = torch.cat((kb[physical, 0], k[rows, 0])).double()
                        values = torch.cat((vb[physical, 0], v[rows, 0])).double()
                        scores = q[rows].double().transpose(0, 1) @ keys.T
                        scores *= 512**-0.5
                        causal = torch.arange(length + width, device=device)[
                            None, :
                        ] <= (length + torch.arange(width, device=device)[:, None])
                        scores.masked_fill_(~causal[None], -float("inf"))
                        result[rows] = (scores.softmax(-1) @ values).transpose(0, 1)
                        start += length
                    return result

                expected = reference(lengths)
                self.assertTrue(invoke(verify_shared_kv_fwd, actual))
                invoke(extend_attention_fwd, generic)
                for output in (actual, generic):
                    torch.testing.assert_close(
                        output.double(), expected, atol=2e-2, rtol=1e-2
                    )
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        invoke(verify_shared_kv_fwd, actual)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    invoke(verify_shared_kv_fwd, actual)
                # Same graph, changed request lengths, values and physical pages.
                # This includes empty -> nonempty and nonempty -> empty requests.
                q.mul_(0.7)
                ids.copy_(ids.flip(0))
                ptr.copy_(
                    torch.tensor([0, 1, 1, 4097], device=device, dtype=torch.int32)
                )
                actual.fill_(float("nan"))
                graph.replay()
                torch.testing.assert_close(
                    actual.double(), reference([1, 0, 4096]), atol=2e-2, rtol=1e-2
                )


if __name__ == "__main__":
    unittest.main()
