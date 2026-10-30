"""The persistent FlashInfer MXFP4 CUTLASS MoE workspace: one buffer per device
and stream, sized by FlashInfer for the largest layer, reused by every eager call."""

import types
import unittest
from unittest import mock

import torch

from sglang.srt.layers.moe.moe_runner import flashinfer_cutlass as fc
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Layer(torch.nn.Module):
    moe_tp_size = 2
    moe_tp_rank = 1
    moe_ep_size = 2
    moe_ep_rank = 0


def fake_x(tokens, device=torch.device("cuda", 0)):
    return types.SimpleNamespace(
        shape=(tokens, 4096), device=device, dtype=torch.float8_e4m3fn
    )


class TestMxfp4MoeWorkspace(CustomTestCase):
    def setUp(self):
        fc._MXFP4_MOE_WORKSPACES.clear()
        fc._MXFP4_MOE_WORKSPACE_REJECTED = False
        fc._mxfp4_moe_workspace_bytes.cache_clear()
        self.addCleanup(fc._MXFP4_MOE_WORKSPACES.clear)
        self.addCleanup(fc._mxfp4_moe_workspace_bytes.cache_clear)
        self.stream = types.SimpleNamespace(cuda_stream=7)
        self.queries, self.allocations, self.flushes = [], [], []

        def size(*args, **kwargs):
            self.queries.append((args, kwargs))
            # Grows with the token bound and with top-k.
            return 1000 + args[0] + args[4]

        def empty(nbytes, **kwargs):
            buffer = types.SimpleNamespace(nbytes=nbytes)
            self.allocations.append(buffer)
            return buffer

        flashinfer_moe = types.SimpleNamespace(cutlass_fused_moe_workspace_size=size)
        for target, value in (
            ("sys.modules", None),
            ("torch.cuda.is_current_stream_capturing", lambda: False),
            ("torch.cuda.current_stream", lambda device: self.stream),
            ("torch.cuda.empty_cache", lambda: self.flushes.append(1)),
            (f"{fc.__name__}._cutlass_moe_tune_max_num_tokens", lambda n: 8192),
        ):
            if target == "sys.modules":
                patcher = mock.patch.dict(
                    "sys.modules", {"flashinfer.fused_moe": flashinfer_moe}
                )
            else:
                patcher = mock.patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.empty = mock.patch.object(fc.torch, "empty", empty)
        self.empty.start()
        self.addCleanup(self.empty.stop)

    def workspace(self, tokens, top_k=6, **kwargs):
        return fc._mxfp4_moe_workspace(
            x=fake_x(tokens, **kwargs),
            fc1_expert_weights=types.SimpleNamespace(dtype=torch.int64),
            w2_weight=types.SimpleNamespace(shape=(64, 4096, 576)),
            top_k=top_k,
            runner_config=MoeRunnerConfig(layer=_Layer()),
            activation_type="swiglu",
            use_mxfp8_act_scaling=True,
            use_wfp4afp8_humming=False,
            use_fused_finalize=True,
        )

    def test_one_buffer_sized_for_the_prefill_bound_is_reused(self):
        first = self.workspace(8192)
        for tokens in (1, 512, 8192):
            self.assertIs(self.workspace(tokens), first)
        self.assertEqual(len(self.queries), 1)
        args, kwargs = self.queries[0]
        # max tokens, hidden, logical intermediate (two FP4 a byte), global experts, top-k
        self.assertEqual(args, (8192, 4096, 1152, 128, 6))
        self.assertEqual(kwargs["weight_dtype"], torch.int64)
        self.assertEqual((kwargs["tp_size"], kwargs["tp_rank"]), (2, 1))
        self.assertEqual((kwargs["ep_size"], kwargs["ep_rank"]), (2, 0))
        self.assertTrue(kwargs["use_mxfp8_act_scaling"])
        self.assertFalse(kwargs["use_w4_group_scaling"])
        self.assertEqual(first.nbytes, 1000 + 8192 + 6)
        # The allocator's unused blocks are returned once, before the buffer is taken.
        self.assertEqual(len(self.flushes), 1)

    def test_a_larger_call_replaces_the_buffer(self):
        first = self.workspace(8192)
        second = self.workspace(16384)
        self.assertIsNot(second, first)
        self.assertEqual(second.nbytes, 1000 + 16384 + 6)
        self.assertIs(self.workspace(100), second)

    def test_each_stream_has_its_own_buffer(self):
        first = self.workspace(8192)
        self.stream = types.SimpleNamespace(cuda_stream=9)
        self.assertIsNot(self.workspace(8192), first)
        self.assertEqual(len(self.allocations), 2)

    def test_no_buffer_under_capture_on_cpu_or_after_a_rejection(self):
        with mock.patch("torch.cuda.is_current_stream_capturing", lambda: True):
            self.assertIsNone(self.workspace(8192))
        self.assertIsNone(self.workspace(8192, device=torch.device("cpu")))
        fc._MXFP4_MOE_WORKSPACE_REJECTED = True
        self.assertIsNone(self.workspace(8192))
        self.assertEqual(self.allocations, [])

    def test_layers_on_a_stream_share_the_largest_buffer(self):
        small = self.workspace(8192, top_k=6)
        large = self.workspace(8192, top_k=8)
        self.assertIsNot(large, small)
        self.assertEqual(large.nbytes, 1000 + 8192 + 8)
        self.assertIs(self.workspace(8192, top_k=6), large)
        # Each layer's size is queried once.
        self.assertEqual(len(self.queries), 2)


class TestMxfp4MoeWorkspaceFallback(CustomTestCase):
    def setUp(self):
        fc._MXFP4_MOE_WORKSPACES.clear()
        fc._MXFP4_MOE_WORKSPACE_REJECTED = False
        self.addCleanup(fc._MXFP4_MOE_WORKSPACES.clear)
        self.addCleanup(setattr, fc, "_MXFP4_MOE_WORKSPACE_REJECTED", False)

    def test_a_rejected_size_falls_back_to_per_call_scratch(self):
        fc._MXFP4_MOE_WORKSPACES[(0, 7)] = (10, "held")
        calls = []

        def run(ws):
            calls.append(ws)
            if ws is not None:
                raise RuntimeError("workspace_buffer too small: need 20 bytes, got 10")

        fc._run_with_mxfp4_moe_workspace(run, "held")
        self.assertEqual(calls, ["held", None])
        self.assertTrue(fc._MXFP4_MOE_WORKSPACE_REJECTED)
        self.assertEqual(fc._MXFP4_MOE_WORKSPACES, {})

    def test_other_errors_propagate(self):
        def fail(ws):
            raise RuntimeError("illegal memory access")

        with self.assertRaises(RuntimeError):
            fc._run_with_mxfp4_moe_workspace(fail, "held")
        self.assertFalse(fc._MXFP4_MOE_WORKSPACE_REJECTED)

        def too_small(ws):
            raise RuntimeError("workspace_buffer too small")

        # Without a workspace there is nothing to fall back from.
        with self.assertRaises(RuntimeError):
            fc._run_with_mxfp4_moe_workspace(too_small, None)
        self.assertFalse(fc._MXFP4_MOE_WORKSPACE_REJECTED)


if __name__ == "__main__":
    unittest.main()
