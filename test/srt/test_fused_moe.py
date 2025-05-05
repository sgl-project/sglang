import unittest

import torch
import torch.nn.functional as F
from tqdm import tqdm

from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_moe
from sglang.srt.utils import get_device
from sglang.test.test_utils import CustomTestCase


class TestFusedMOE(CustomTestCase):
    NUM_EXPERTS = [8, 64]
    TOP_KS = [2, 6]

    @classmethod
    def setUpClass(cls):
        cls.device = get_device()

    @staticmethod
    def create_random_cuda_tensor(shape, dtype, mean=0, std=0.01, device="cuda"):
        """Create a random CUDA tensor

        Args:
            shape: Tensor shape
            dtype: Data type
            mean: Mean value
            std: Standard deviation

        Returns:
            torch.Tensor: Randomly initialized CUDA tensor
        """
        return torch.empty(shape, dtype=dtype, device=device).normal_(mean, std)

    def get_tolerance(self, dtype):
        """Get tolerance values for different data types

        Args:
            dtype: Data type

        Returns:
            tuple: (relative tolerance, absolute tolerance)
        """
        if dtype == torch.float32:
            return 1e-3, 1e-5
        elif dtype in [torch.float16, torch.bfloat16]:
            return 1e-1, 1e-2
        else:
            return 1e-2, 1e-2  # Default values for other types

    def torch_naive_moe(
        self,
        a,
        w1,
        w2,
        score,
        topk,
        w1_scale=None,
        w2_scale=None,
        a1_scale=None,
        a2_scale=None,
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

            if w1_scale is not None:
                w1_compute = (w1_compute * w1_scale.view(-1, 1, 1)).to(a.dtype)
            if w2_scale is not None:
                w2_compute = (w2_compute * w2_scale.view(-1, 1, 1)).to(a.dtype)
            if a1_scale is not None:
                a = (a * a1_scale).to(a.dtype)
            if a2_scale is not None:
                a = (a * a2_scale).to(a.dtype)
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

    def _test_case(self, m, n, k, e, topk, dtype, use_fp8_w8a8=False):
        rtol, atol = self.get_tolerance(dtype)

        if use_fp8_w8a8:
            # AssertionError: fp8e4nv data type is not supported on CUDA arch < 89
            if torch.cuda.is_available():
                capability = torch.cuda.get_device_capability()
                if not (capability[0] >= 9 or capability == (8, 9)):
                    return

            a = self.create_random_cuda_tensor((m, k), dtype, device=self.device)
            w1 = self.create_random_cuda_tensor(
                (e, 2 * n, k), dtype, device=self.device
            )
            w2 = self.create_random_cuda_tensor((e, k, n), dtype, device=self.device)
            w1 = w1.to(torch.float8_e4m3fn)
            w2 = w2.to(torch.float8_e4m3fn)
            score = self.create_random_cuda_tensor((m, e), dtype, device=self.device)

            w1_scale = self.create_random_cuda_tensor(
                e, torch.float32, device=self.device
            )
            w2_scale = self.create_random_cuda_tensor(
                e, torch.float32, device=self.device
            )
            a1_scale = self.create_random_cuda_tensor(
                1, torch.float32, device=self.device
            )
            a2_scale = self.create_random_cuda_tensor(
                1, torch.float32, device=self.device
            )

            sglang_output = fused_moe(
                a,
                w1,
                w2,
                score,
                topk,
                renormalize=False,
                use_fp8_w8a8=True,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                a1_scale=a1_scale,
                a2_scale=a2_scale,
            )

            torch_output = self.torch_naive_moe(
                a, w1, w2, score, topk, w1_scale, w2_scale, a1_scale, a2_scale
            )
            torch.testing.assert_close(
                sglang_output, torch_output, rtol=rtol, atol=atol
            )

        else:
            a = self.create_random_cuda_tensor((m, k), dtype, device=self.device)
            w1 = self.create_random_cuda_tensor(
                (e, 2 * n, k), dtype, device=self.device
            )
            w2 = self.create_random_cuda_tensor((e, k, n), dtype, device=self.device)
            score = self.create_random_cuda_tensor((m, e), dtype, device=self.device)

            triton_output = fused_moe(a, w1, w2, score, topk, renormalize=False)
            torch_output = self.torch_naive_moe(a, w1, w2, score, topk)
            torch.testing.assert_close(
                triton_output, torch_output, rtol=rtol, atol=atol
            )

    def test_various_configurations(self):
        m_values = [1, 33, 64, 222]
        n_values = [128, 1024]
        k_values = [128, 511, 1024]
        dtypes = [torch.float16, torch.bfloat16]
        fp8_modes = [False, True]

        # Calculate total number of tests
        total_tests = (
            len(m_values)
            * len(n_values)
            * len(k_values)
            * len(self.NUM_EXPERTS)
            * len(self.TOP_KS)
            * len(dtypes)
            * len(fp8_modes)
        )

        # Create progress bar
        with tqdm(total=total_tests, desc="Running MoE tests") as pbar:
            for m in m_values:
                for n in n_values:
                    for k in k_values:
                        for e in self.NUM_EXPERTS:
                            for topk in self.TOP_KS:
                                for dtype in dtypes:
                                    for use_fp8_w8a8 in fp8_modes:
                                        with self.subTest(
                                            m=m,
                                            n=n,
                                            k=k,
                                            e=e,
                                            topk=topk,
                                            dtype=dtype,
                                            fp8=use_fp8_w8a8,
                                        ):
                                            self._test_case(
                                                m,
                                                n,
                                                k,
                                                e,
                                                topk,
                                                dtype,
                                                use_fp8_w8a8=use_fp8_w8a8,
                                            )
                                            torch.cuda.empty_cache()
                                        pbar.update(1)


if __name__ == "__main__":
    unittest.main()
