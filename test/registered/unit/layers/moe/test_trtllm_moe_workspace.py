"""Workspace lifetime, capture safety, and compatibility regressions."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.moe.trtllm_moe_workspace import (
    TrtllmMoeWorkspace,
    _make_workspace,
    get_trtllm_moe_workspace,
)
from sglang.srt.runtime_context import get_resources, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestTrtllmMoeWorkspace(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        self.sizer = Mock(return_value=1024)
        self.fi = SimpleNamespace(trtllm_fused_moe_workspace_size=self.sizer)
        for patcher in (
            patch.dict(
                "sys.modules", {"flashinfer": SimpleNamespace(fused_moe=self.fi)}
            ),
            patch(
                "torch.cuda.current_stream", return_value=SimpleNamespace(cuda_stream=1)
            ),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.kwargs = {
            "hidden_states": torch.empty(1, 256),
            "intermediate_size": 256,
            "num_experts": 8,
            "local_num_experts": 4,
            "top_k": 2,
            "tune_max_num_tokens": 32,
        }

    def test_small_first_reuse_and_stream_isolation(self):
        workspace = TrtllmMoeWorkspace(32)
        first = workspace.get("bf16", self.kwargs)
        self.assertEqual(first.numel(), 1024)
        self.assertEqual(first.dtype, torch.uint8)
        self.kwargs["hidden_states"] = torch.empty(32, 256)
        self.assertIs(workspace.get("bf16", self.kwargs), first)
        self.assertEqual(self.sizer.call_args.args[0], 32)
        self.assertEqual(self.sizer.call_count, 1)
        with patch(
            "torch.cuda.current_stream", return_value=SimpleNamespace(cuda_stream=2)
        ):
            second = workspace.get("bf16", self.kwargs)
        # Must use different workspace buffers for different streams
        self.assertNotEqual(first.data_ptr(), second.data_ptr())

    def test_fallbacks_do_not_allocate(self):
        workspace = TrtllmMoeWorkspace(32)
        for override in (
            {"do_finalize": False},
            {"hidden_states": torch.empty(33, 256)},
            {"hidden_states": torch.empty(0, 256)},
            {"tune_max_num_tokens": 64},
        ):
            self.assertIsNone(workspace.get("bf16", {**self.kwargs, **override}))
        self.assertFalse(workspace.buffers)
        self.assertFalse(workspace.sizes)
        del self.fi.trtllm_fused_moe_workspace_size
        self.assertIsNone(workspace.get("bf16", self.kwargs))
        self.fi.trtllm_fused_moe_workspace_size = self.sizer
        self.sizer.side_effect = ValueError("unsupported")
        self.assertIsNone(workspace.get("bf16", self.kwargs))
        self.assertIsNone(workspace.get("bf16", self.kwargs))
        self.assertEqual(self.sizer.call_count, 1)
        self.assertFalse(workspace.buffers)

    def test_sizing_options_and_packed_hidden_dimension(self):
        workspace = TrtllmMoeWorkspace(32)
        options = {
            "activation_type": 3,
            "per_token_scale": torch.ones(1),
            "valid_hidden_size": 384,
            "valid_intermediate_size": 192,
        }
        workspace.get("nvfp4", {**self.kwargs, **options})
        self.assertEqual(self.sizer.call_args.args[1], 512)
        self.assertTrue(self.sizer.call_args.kwargs["use_per_token_scale"])
        self.assertEqual(self.sizer.call_args.kwargs["valid_hidden_size"], 384)
        self.assertEqual(self.sizer.call_args.kwargs["activation_type"], 3)
        workspace.get("mxfp4_mxfp8", self.kwargs)
        self.assertEqual(self.sizer.call_args.args[1], 256)
        self.assertEqual(len(workspace.sizes), 2)

    def test_runtime_sharing_reset_and_legacy_call(self):
        self.assertIsNone(get_trtllm_moe_workspace("bf16", **self.kwargs))
        publish(ServerArgs(model_path="dummy", chunked_prefill_size=32), role="test")
        first = get_trtllm_moe_workspace("bf16", **self.kwargs)
        self.assertIsNotNone(first)
        self.assertIs(get_trtllm_moe_workspace("bf16", **self.kwargs), first)
        del self.fi.trtllm_fused_moe_workspace_size
        self.assertIsNone(get_trtllm_moe_workspace("bf16", **self.kwargs))
        reset_context()
        self.assertNotIn("trtllm_moe_workspace", get_resources().buffers)

    def test_sizing_includes_dp_and_tuning_bucket(self):
        publish(
            ServerArgs(
                model_path="dummy", chunked_prefill_size=33, tp_size=2, attn_dp_size=2
            ),
            role="test",
        )
        self.assertEqual(_make_workspace().max_rows, 128)


if __name__ == "__main__":
    unittest.main()
