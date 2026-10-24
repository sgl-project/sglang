"""Compare caller-owned and per-call TRT-LLM MoE scratch on Blackwell."""

import unittest

import torch

from sglang.srt.layers.moe.trtllm_moe_workspace import get_trtllm_moe_workspace
from sglang.srt.runtime_context import get_resources, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


class TestTrtllmMoeWorkspaceParity(CustomTestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    @torch.inference_mode()
    def test_small_first_and_cuda_graph(self):
        import flashinfer.fused_moe as fi
        from flashinfer.fused_moe.core import ActivationType

        if torch.cuda.get_device_capability()[0] != 10:
            self.skipTest("requires Blackwell")
        if not hasattr(fi, "trtllm_fused_moe_workspace_size"):
            self.skipTest("requires FlashInfer caller-owned MoE workspace API")
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy", chunked_prefill_size=32), role="test")
        torch.manual_seed(42)
        for activation in (ActivationType.Swiglu, ActivationType.Relu2):
            with self.subTest(activation=activation):
                self._check_activation(fi, activation)

    def _check_activation(self, fi, activation):
        from flashinfer.fused_moe.core import ActivationType

        fc1 = 512 if activation == ActivationType.Swiglu else 256
        # Random values already in the kernel's BlockMajorK layout.
        w1 = torch.randn(8, 2, fc1, 128, device="cuda", dtype=torch.bfloat16) / 10
        w2 = torch.randn(8, 2, 256, 128, device="cuda", dtype=torch.bfloat16) / 10
        x = torch.randn(32, 256, device="cuda", dtype=torch.bfloat16) / 10
        ids = torch.tensor([0, 1], device="cuda", dtype=torch.int32).repeat(32, 1)
        weights = torch.full((32, 2), 0.5, device="cuda")
        buffers = []

        def checked_accuracy(**kwargs):
            buffers.append(kwargs.get("workspace_buffer"))
            actual = fi.trtllm_bf16_routed_moe(**kwargs)
            reference = fi.trtllm_bf16_routed_moe(
                **{**kwargs, "workspace_buffer": None}
            )
            torch.testing.assert_close(actual, reference)
            return actual

        def kwargs(rows):
            return {
                "hidden_states": x[:rows],
                "topk_ids": (ids[:rows], weights[:rows]),
                "gemm1_weights": w1,
                "gemm2_weights": w2,
                "num_experts": 8,
                "local_num_experts": 8,
                "local_expert_offset": 0,
                "top_k": 2,
                "n_group": None,
                "topk_group": None,
                "intermediate_size": 256,
                "activation_type": activation.value,
                "tune_max_num_tokens": 32,
            }

        for rows in (1, 32, 16):
            checked_accuracy(
                **kwargs(rows),
                workspace_buffer=get_trtllm_moe_workspace("bf16", **kwargs(rows)),
            )
        self.assertIsNotNone(buffers[0])
        # All workspace buffers are the same due to the same tune_max_num_tokens
        self.assertTrue(all(buffer is buffers[0] for buffer in buffers))
        # Warm the FlashInfer kernels on the eager stream, then discard only
        # cached sizes to exercise sizing and allocation during capture.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        workspace = get_resources().buffers["trtllm_moe_workspace"]
        workspace.sizes.clear()
        before = len(workspace.buffers)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            capture_workspace = get_trtllm_moe_workspace("bf16", **kwargs(32))
            captured = fi.trtllm_bf16_routed_moe(
                **kwargs(32),
                workspace_buffer=capture_workspace,
            )
        self.assertIsNotNone(capture_workspace)
        self.assertEqual(len(workspace.sizes), 1)
        self.assertNotEqual(capture_workspace.data_ptr(), buffers[0].data_ptr())
        x.mul_(0.5)
        graph.replay()
        expected = fi.trtllm_bf16_routed_moe(**kwargs(32))
        torch.testing.assert_close(captured, expected)
        self.assertEqual(len(workspace.buffers), before + 1)


if __name__ == "__main__":
    unittest.main()
