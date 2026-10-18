import unittest
from unittest.mock import patch

import torch
from tqdm import tqdm

from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton_kernels import TritonKernelsQuantInfo
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import TopK, TopKOutputFormat
from sglang.srt.runtime_context import get_context
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=12, stage="base-b", runner_config="1-gpu-large")


class TestFusedMOE(CustomTestCase):
    NUM_EXPERTS = [8, 64]
    TOP_KS = [2, 4]

    class RecordingCapturer:
        def __init__(self):
            self.layer_id = None
            self.topk_indices = None

        def capture(self, layer_id, topk_indices):
            self.layer_id = layer_id
            self.topk_indices = topk_indices.clone()

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
        return_per_expert: bool = False,
    ):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

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

        weighted = out.view(B, -1, w2.shape[1]) * topk_weight.view(B, -1, 1).to(
            out.dtype
        )

        if return_per_expert:
            return weighted

        return weighted.sum(dim=1)

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
            layer_id=0,
            renormalize=False,
            use_grouped_topk=False,
        )
        topk_op.topk_config.output_format = TopKOutputFormat.TRITON_KERNEL
        triton_topk_output = topk_op.forward_cuda(
            hidden_states=a,
            router_logits=score,
        )

        quant_info = TritonKernelsQuantInfo(w13_weight=w1_tri, w2_weight=w2_tri)

        dispatch_output = StandardDispatchOutput(
            hidden_states=a, hidden_states_scale=None, topk_output=triton_topk_output
        )

        torch_per_expert = self.torch_naive_moe(
            a, w1, w2, score, topk, return_per_expert=True
        )
        torch_combined = torch_per_expert.sum(dim=1)

        def run_runner(config):
            runner = MoeRunner(MoeRunnerBackend.TRITON_KERNELS, config)
            result = runner.run(dispatch_output, quant_info)
            return result.hidden_states

        # Combined output (no_combine=False)
        non_fused_config = MoeRunnerConfig(inplace=False)
        non_fused_output = run_runner(non_fused_config)
        torch.testing.assert_close(
            non_fused_output, torch_combined, rtol=rtol, atol=atol
        )

        # Per-expert output (no_combine=True)
        non_fused_no_combine_config = MoeRunnerConfig(
            inplace=False, no_combine=True, top_k=topk
        )
        non_fused_no_combine_output = run_runner(non_fused_no_combine_config)
        torch.testing.assert_close(
            non_fused_no_combine_output, torch_per_expert, rtol=rtol, atol=atol
        )

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

    def test_bypassed_topk_refuses_capture_but_preserves_draft_opt_out(self):
        """BYPASSED routing must fail instead of returning an untouched cache."""
        hidden_states = self.create_random_cuda_tensor((2, 8), torch.bfloat16)
        router_logits = self.create_random_cuda_tensor((2, 4), torch.bfloat16)

        with get_context().override_server_args(enable_return_routed_experts=True):
            target_topk = TopK(
                top_k=2,
                layer_id=0,
                output_format=TopKOutputFormat.BYPASSED,
            )
            with self.assertRaisesRegex(RuntimeError, "BYPASSED top-k"):
                target_topk.forward_cuda(hidden_states, router_logits)

            draft_topk = TopK(
                top_k=2,
                layer_id=0,
                output_format=TopKOutputFormat.BYPASSED,
                allow_routed_experts_capture=False,
            )
            output = draft_topk.forward_cuda(hidden_states, router_logits)
            self.assertEqual(output.format, TopKOutputFormat.BYPASSED)

    def test_triton_capture_covers_both_renormalization_orders(self):
        """Triton must publish its selected experts for both softmax orders."""
        hidden_states = self.create_random_cuda_tensor((8, 16), torch.bfloat16)
        router_logits = self.create_random_cuda_tensor((8, 8), torch.bfloat16)

        for renormalize in (False, True):
            with self.subTest(renormalize=renormalize):
                capturer = self.RecordingCapturer()
                topk_op = TopK(
                    top_k=2,
                    layer_id=3,
                    renormalize=renormalize,
                    output_format=TopKOutputFormat.TRITON_KERNEL,
                )
                with patch(
                    "sglang.srt.layers.moe.topk.get_global_experts_capturer",
                    return_value=capturer,
                ):
                    topk_op.forward_cuda(hidden_states, router_logits)

                captured_ids = capturer.topk_indices.to(torch.int64)
                captured_scores = router_logits.gather(dim=-1, index=captured_ids)
                cutoff = torch.topk(router_logits, 2, dim=-1).values[:, -1:]
                self.assertEqual(capturer.layer_id, 3)
                self.assertEqual(captured_ids.shape, (8, 2))
                self.assertTrue(torch.all(captured_scores >= cutoff).item())
                self.assertTrue(
                    torch.all(captured_ids[:, 0] != captured_ids[:, 1]).item(),
                    "each token must route to top_k distinct experts",
                )

        draft_capturer = self.RecordingCapturer()
        draft_topk = TopK(
            top_k=2,
            layer_id=3,
            output_format=TopKOutputFormat.TRITON_KERNEL,
            allow_routed_experts_capture=False,
        )
        with patch(
            "sglang.srt.layers.moe.topk.get_global_experts_capturer",
            return_value=draft_capturer,
        ):
            draft_topk.forward_cuda(hidden_states, router_logits)
        self.assertIsNone(draft_capturer.topk_indices)


if __name__ == "__main__":
    unittest.main()
