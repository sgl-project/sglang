import unittest

import torch
import torch.nn.functional as F
from tqdm import tqdm

from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.moe.fused_moe_triton.triton_kernels_moe import (
    triton_kernel_moe_forward,
)
from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
from sglang.srt.layers.moe.topk import TopK
from sglang.test.test_utils import CustomTestCase


class TestFusedMOE(CustomTestCase):
    NUM_EXPERTS = [8, 64]
    TOP_KS = [2, 4]

    @staticmethod
    def create_random_cuda_tensor(shape, dtype, mean=0, std=0.01):
        """Create a random CUDA tensor

        Args:
            shape: Tensor shape
            dtype: Data type
            mean: Mean value
            std: Standard deviation

        Returns:
            torch.Tensor: Randomly initialized CUDA tensor
        """
        return torch.empty(shape, dtype=dtype, device="cuda").normal_(mean, std)

    def get_tolerance(self, dtype):
        """Get tolerance values for different data types

        Args:
            dtype: Data type

        Returns:
            tuple: (relative tolerance, absolute tolerance)
        """
        if dtype == torch.float32:
            return 1e-5, 1e-5
        elif dtype in [torch.float16, torch.bfloat16]:
            return 1e-5, 1e-5
        else:
            return 1e-2, 1e-2  # Default values for other types

    def torch_naive_moe(
        self,
        a,
        w1,
        w2,
        score,
        topk,
    ):
        B, D = a.shape
        a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
        out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)
        score = torch.softmax(score, dim=-1, dtype=torch.float32)
        topk_weight, topk_ids = torch.topk(score, topk)
        topk_weight = topk_weight.view(-1)
        topk_ids = topk_ids.view(-1)

        if w1.dtype == torch.float8_e4m3fn:
            w1_compute = w1.to(a.dtype)
            w2_compute = w2.to(a.dtype)
        else:
            w1_compute = w1
            w2_compute = w2

        for i in range(w1_compute.shape[0]):
            mask = topk_ids == i
            if mask.sum():
                out[mask] = SiluAndMul()(
                    a[mask] @ w1_compute[i].transpose(0, 1)
                ) @ w2_compute[i].transpose(0, 1)

        return (
            out.view(B, -1, w2.shape[1]) * topk_weight.view(B, -1, 1).to(out.dtype)
        ).sum(dim=1)

    def _test_case(self, m, n, k, e, topk, dtype):
        rtol, atol = self.get_tolerance(dtype)

        a = self.create_random_cuda_tensor((m, k), dtype)
        w1 = self.create_random_cuda_tensor((e, 2 * n, k), dtype)
        w2 = self.create_random_cuda_tensor((e, k, n), dtype)
        w1_tri = w1.clone()
        w2_tri = w2.clone()
        w1_tri = w1_tri.transpose(-2, -1).contiguous()
        w2_tri = w2_tri.transpose(-2, -1).contiguous()
        score = self.create_random_cuda_tensor((m, e), dtype)

        topk_op = TopK(
            top_k=topk,
            renormalize=False,
            use_grouped_topk=False,
        )
        topk_op.use_triton_kernels = True
        triton_topk_output = topk_op.forward_cuda(
            hidden_states=a,
            router_logits=score,
        )

        moe_runner_config = MoeRunnerConfig(
            inplace=False,
        )
        triton_output = triton_kernel_moe_forward(
            a, w1_tri, w2_tri, triton_topk_output, moe_runner_config
        )
        torch_output = self.torch_naive_moe(a, w1, w2, score, topk)
        torch.testing.assert_close(triton_output, torch_output, rtol=rtol, atol=atol)

    def test_various_configurations(self):
        m_values = [1, 32, 64, 256]
        n_values = [128, 1024]
        k_values = [128, 512, 1024]
        dtypes = [torch.bfloat16]

        # Calculate total number of tests
        total_tests = (
            len(m_values)
            * len(n_values)
            * len(k_values)
            * len(self.NUM_EXPERTS)
            * len(self.TOP_KS)
            * len(dtypes)
        )

        # Create progress bar
        with tqdm(total=total_tests, desc="Running MoE tests") as pbar:
            for m in m_values:
                for n in n_values:
                    for k in k_values:
                        for e in self.NUM_EXPERTS:
                            for topk in self.TOP_KS:
                                for dtype in dtypes:
                                    with self.subTest(
                                        m=m,
                                        n=n,
                                        k=k,
                                        e=e,
                                        topk=topk,
                                        dtype=dtype,
                                    ):
                                        self._test_case(
                                            m,
                                            n,
                                            k,
                                            e,
                                            topk,
                                            dtype,
                                        )
                                        torch.cuda.empty_cache()
                                    pbar.update(1)


if __name__ == "__main__":
    unittest.main()
