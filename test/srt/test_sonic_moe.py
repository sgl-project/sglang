import unittest

import torch
from tqdm import tqdm

from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig

# Replaced fused_moe with sonic_moe for this test.
from sglang.srt.layers.moe.sonic_moe import sonic_moe
from sglang.srt.layers.moe.topk import TopKConfig, select_experts
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.utils import is_hip
from sglang.test.test_utils import CustomTestCase

_is_hip = is_hip()


class TestSonicMOE(CustomTestCase):
    NUM_EXPERTS = [64]
    TOP_KS = [2, 8]

    @staticmethod
    def create_random_cuda_tensor(shape, dtype, mean=0, std=0.01):
        """Create a random CUDA tensor.

        Args:
            shape: Tensor shape.
            dtype: Data type.
            mean: Mean value.
            std: Standard deviation.

        Returns:
            torch.Tensor: Randomly initialized CUDA tensor.
        """
        return torch.empty(shape, dtype=dtype, device="cuda").normal_(mean, std)

    def get_tolerance(self, dtype):
        """Get tolerance values for different data types.

        Args:
            dtype: Data type.

        Returns:
            tuple: (relative tolerance, absolute tolerance)
        """
        if dtype == torch.float32:
            return 1e-3, 1e-5
        elif dtype in [torch.float16, torch.bfloat16]:
            return 1e-1, 1e-2
        else:
            return 1e-2, 1e-2  # Default values for other types

    def torch_naive_moe(self, a, w1, w2, score, topk):
        """Naive MoE implementation in PyTorch (reference).

        Args:
            a: Hidden states, shape (M, K).
            w1: Up projection weights, shape (E, 2*N, K).
            w2: Down projection weights, shape (E, K, N).
            score: Router logits, shape (M, E).
            topk: Top-k experts per token.

        Returns:
            torch.Tensor: Output tensor, shape (M, N).
        """
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

        B, D = a.shape
        a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
        out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)

        score = torch.softmax(score, dim=-1, dtype=torch.float32)
        topk_weight, topk_ids = torch.topk(score, topk)
        topk_weight = topk_weight.view(-1)
        topk_ids = topk_ids.view(-1)

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
        """Single test case comparing Sonic MoE vs torch naive reference."""
        rtol, atol = self.get_tolerance(dtype)

        # Sonic MoE requires CUDA.
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for Sonic MoE test.")

        # Sonic MoE is CUDA-only; skip on HIP/ROCm.
        if _is_hip:
            self.skipTest("Skip on HIP/ROCm: Sonic MoE is CUDA-only.")

        # Sonic MoE is expected to run on Hopper (compute capability 9.x).
        capability = torch.cuda.get_device_capability()
        if capability[0] < 9:
            self.skipTest(f"Skip non-Hopper GPU: require CC>=9.x, got {capability}.")

        a = self.create_random_cuda_tensor((m, k), dtype)
        w1 = self.create_random_cuda_tensor((e, 2 * n, k), dtype)
        w2 = self.create_random_cuda_tensor((e, k, n), dtype)
        score = self.create_random_cuda_tensor((m, e), dtype)

        topk_output = select_experts(
            hidden_states=a,
            router_logits=score,
            topk_config=TopKConfig(top_k=topk, renormalize=False),
        )

        # Skip if Sonic MoE is not available/supported for this configuration.
        non_fused_no_combine_config = MoeRunnerConfig(
            inplace=False, no_combine=True, top_k=topk
        )

        # if not is_sonic_moe_supported(a, w1, w2, non_fused_no_combine_config):
        #     self.skipTest("Sonic MoE is not installed or not supported on this setup.")

        sonic_output = sonic_moe(a, w1, w2, topk_output)
        torch_output = self.torch_naive_moe(a, w1, w2, score, topk)

        torch.testing.assert_close(sonic_output, torch_output, rtol=rtol, atol=atol)

    def test_various_configurations(self):
        m_values = [1, 33, 64, 222]
        n_values = [128, 1024]
        k_values = [128, 511, 1024]
        dtypes = [torch.float16, torch.bfloat16]

        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

        # Calculate total number of tests.
        total_tests = (
            len(m_values)
            * len(n_values)
            * len(k_values)
            * len(self.NUM_EXPERTS)
            * len(self.TOP_KS)
            * len(dtypes)
        )

        # Create progress bar.
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
                                        self._test_case(m, n, k, e, topk, dtype)
                                        torch.cuda.empty_cache()
                                    pbar.update(1)


if __name__ == "__main__":
    unittest.main()
