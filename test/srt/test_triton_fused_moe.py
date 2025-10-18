import unittest

import torch
import torch.nn.functional as F
from tqdm import tqdm

from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton_kernels import TritonKernelsQuantInfo
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import TopK, TopKOutputFormat
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
        topk_op.topk_config.output_format = TopKOutputFormat.TRITON_KERNEL
        triton_topk_output = topk_op.forward_cuda(
            hidden_states=a,
            router_logits=score,
        )

        quant_info = TritonKernelsQuantInfo(w13_weight=w1_tri, w2_weight=w2_tri)

        fused_config = MoeRunnerConfig(inplace=False)
        runner_fused = MoeRunner(MoeRunnerBackend.TRITON_KERNEL, fused_config)
        fused_output = runner_fused.run(
            StandardDispatchOutput(hidden_states=a, topk_output=triton_topk_output),
            quant_info,
        ).hidden_states

        torch_output = self.torch_naive_moe(a, w1, w2, score, topk)
        torch.testing.assert_close(fused_output, torch_output, rtol=rtol, atol=atol)

        fused_scaled_config = MoeRunnerConfig(
            inplace=False,
            routed_scaling_factor=0.5,
        )
        runner_fused_scaled = MoeRunner(
            MoeRunnerBackend.TRITON_KERNEL, fused_scaled_config
        )
        fused_scaled_output = runner_fused_scaled.run(
            StandardDispatchOutput(hidden_states=a, topk_output=triton_topk_output),
            quant_info,
        ).hidden_states
        torch.testing.assert_close(
            fused_scaled_output, 0.5 * fused_output, rtol=rtol, atol=atol
        )

        # TODO: Triton-kernel kernels currently emit combined activations even when
        # no_combine is requested. Re-enable this check once the kernel supports
        # per-expert outputs.
        # no_combine_config = MoeRunnerConfig(inplace=False, no_combine=True)
        # runner_no_combine = MoeRunner(
        #     MoeRunnerBackend.TRITON_KERNEL, no_combine_config
        # )
        # no_combine_output = runner_no_combine.run(
        #     StandardDispatchOutput(hidden_states=a, topk_output=triton_topk_output),
        #     quant_info,
        # ).hidden_states
        # self.assertEqual(
        #     no_combine_output.shape,
        #     (a.shape[0], topk, w2.shape[1]),
        # )

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
