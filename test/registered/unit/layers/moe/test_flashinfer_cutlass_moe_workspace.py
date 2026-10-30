"""The persistent FlashInfer MXFP4 CUTLASS MoE workspace: one buffer per device
and stream, sized by FlashInfer for the largest layer, reused by every eager call."""

import contextlib
import types
import unittest
import weakref
from unittest import mock

import torch

from sglang.srt.layers.moe.moe_runner import flashinfer_cutlass as fc
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Layer(torch.nn.Module):
    moe_tp_size = 2
    moe_tp_rank = 1
    moe_ep_size = 2
    moe_ep_rank = 0


class _Buffer:
    def __init__(self, nbytes):
        self.nbytes = nbytes


def fake_x(tokens, device=torch.device("cuda", 0)):
    return types.SimpleNamespace(
        shape=(tokens, 4096), device=device, dtype=torch.float8_e4m3fn
    )


def _reset():
    fc._MXFP4_MOE_WORKSPACES.clear()
    fc._MXFP4_MOE_WORKSPACE_OFF = False
    fc._mxfp4_moe_workspace_bytes.cache_clear()


class TestMxfp4MoeWorkspace(CustomTestCase):
    def setUp(self):
        _reset()
        self.addCleanup(_reset)
        self.stream = types.SimpleNamespace(cuda_stream=7)
        self.free = 1 << 40
        self.queries, self.allocations, self.flushes = [], [], []
        # Whether every earlier buffer was already released at each allocation.
        self.released_before = []
        self.refs = []

        def size(*args, **kwargs):
            self.queries.append((args, kwargs))
            # Grows with the token bound and with top-k.
            return 1000 + args[0] + args[4]

        def empty(nbytes, **kwargs):
            self.released_before.append(all(r() is None for r in self.refs))
            buffer = _Buffer(nbytes)
            self.allocations.append(nbytes)
            self.refs.append(weakref.ref(buffer))
            return buffer

        flashinfer_moe = types.SimpleNamespace(cutlass_fused_moe_workspace_size=size)
        for target, value in (
            ("torch.cuda.is_current_stream_capturing", lambda: False),
            ("torch.cuda.current_stream", lambda *args: self.stream),
            ("torch.cuda.device", lambda device: contextlib.nullcontext()),
            ("torch.cuda.mem_get_info", lambda *args: (self.free, 1 << 40)),
            ("torch.cuda.empty_cache", lambda: self.flushes.append(1)),
            ("torch.compiler.is_compiling", lambda: False),
            (f"{fc.__name__}._cutlass_moe_tune_max_num_tokens", lambda n: 8192),
        ):
            patcher = mock.patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        for patcher in (
            mock.patch.dict("sys.modules", {"flashinfer.fused_moe": flashinfer_moe}),
            mock.patch.object(fc.torch, "empty", empty),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def workspace(self, tokens, top_k=6, **kwargs):
        return fc._mxfp4_moe_workspace(
            x=fake_x(tokens, **kwargs),
            fc1_expert_weights=types.SimpleNamespace(dtype=torch.int64),
            w2_weight=types.SimpleNamespace(shape=(64, 4096, 576)),
            top_k=top_k,
            runner_config=MoeRunnerConfig(layer=_Layer()),
            output_dtype=torch.bfloat16,
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
        self.assertEqual(kwargs["x_dtype"], torch.float8_e4m3fn)
        self.assertEqual(kwargs["weight_dtype"], torch.int64)
        self.assertEqual(kwargs["output_dtype"], torch.bfloat16)
        self.assertEqual(kwargs["activation_type"], "swiglu")
        self.assertEqual((kwargs["tp_size"], kwargs["tp_rank"]), (2, 1))
        self.assertEqual((kwargs["ep_size"], kwargs["ep_rank"]), (2, 0))
        self.assertTrue(kwargs["use_mxfp8_act_scaling"])
        self.assertFalse(kwargs["use_w4_group_scaling"])
        self.assertTrue(kwargs["use_fused_finalize"])
        self.assertFalse(kwargs["use_wfp4afp8_humming"])
        self.assertEqual(kwargs["device"], torch.device("cuda", 0))
        self.assertEqual(first.nbytes, 1000 + 8192 + 6)
        # The driver had room, so the allocator's cache was left alone.
        self.assertEqual(self.flushes, [])

    def test_the_cache_is_emptied_only_when_the_driver_lacks_room(self):
        self.free = 100
        self.workspace(8192)
        self.assertEqual(len(self.flushes), 1)

    def test_a_larger_call_replaces_the_buffer_after_releasing_it(self):
        self.workspace(8192)
        second = self.workspace(16384)
        self.assertEqual(second.nbytes, 1000 + 16384 + 6)
        self.assertEqual(self.released_before, [True, True])
        self.assertIs(self.workspace(100), second)

    def test_each_stream_has_its_own_buffer(self):
        first = self.workspace(8192)
        self.stream = types.SimpleNamespace(cuda_stream=9)
        self.assertIsNot(self.workspace(8192), first)
        self.assertEqual(len(self.allocations), 2)

    def test_layers_on_a_stream_share_the_largest_buffer(self):
        self.workspace(8192, top_k=6)
        large = self.workspace(8192, top_k=8)
        self.assertEqual(large.nbytes, 1000 + 8192 + 8)
        self.assertIs(self.workspace(8192, top_k=6), large)
        # Each layer's size is queried once.
        self.assertEqual(len(self.queries), 2)

    def test_no_buffer_under_capture_compile_on_cpu_or_when_off(self):
        with mock.patch("torch.cuda.is_current_stream_capturing", lambda: True):
            self.assertIsNone(self.workspace(8192))
        with mock.patch("torch.compiler.is_compiling", lambda: True):
            self.assertIsNone(self.workspace(8192))
        self.assertIsNone(self.workspace(8192, device=torch.device("cpu")))
        fc._MXFP4_MOE_WORKSPACE_OFF = True
        self.assertIsNone(self.workspace(8192))
        self.assertEqual(self.allocations, [])

    def test_an_allocation_failure_turns_workspaces_off(self):
        def oom(nbytes, **kwargs):
            raise torch.OutOfMemoryError("out of memory")

        with mock.patch.object(fc.torch, "empty", oom):
            self.assertIsNone(self.workspace(8192))
        self.assertTrue(fc._MXFP4_MOE_WORKSPACE_OFF)
        self.assertIsNone(self.workspace(8192))
        self.assertEqual(fc._MXFP4_MOE_WORKSPACES, {})


class TestMxfp4MoeWorkspaceFallback(CustomTestCase):
    def setUp(self):
        _reset()
        self.addCleanup(_reset)

    def test_a_rejected_size_reruns_with_per_call_scratch(self):
        held = _Buffer(10)
        fc._MXFP4_MOE_WORKSPACES[(0, 7)] = (10, held)
        ref = weakref.ref(held)
        del held
        calls = []

        def run(ws):
            calls.append(ws is not None)
            if ws is not None:
                raise RuntimeError("workspace_buffer too small: need 20 bytes, got 10")
            # The rejected buffer is released before the rerun.
            self.assertIsNone(ref())

        fc._run_with_mxfp4_moe_workspace(
            run, lambda: fc._MXFP4_MOE_WORKSPACES[(0, 7)][1]
        )
        self.assertEqual(calls, [True, False])
        self.assertTrue(fc._MXFP4_MOE_WORKSPACE_OFF)
        self.assertEqual(fc._MXFP4_MOE_WORKSPACES, {})

    def test_other_errors_propagate(self):
        def fail(ws):
            raise RuntimeError("illegal memory access")

        with self.assertRaises(RuntimeError):
            fc._run_with_mxfp4_moe_workspace(fail, lambda: _Buffer(10))
        self.assertFalse(fc._MXFP4_MOE_WORKSPACE_OFF)

        def too_small(ws):
            raise RuntimeError("workspace_buffer too small")

        # Without a workspace there is nothing to fall back from.
        with self.assertRaises(RuntimeError):
            fc._run_with_mxfp4_moe_workspace(too_small, lambda: None)
        self.assertFalse(fc._MXFP4_MOE_WORKSPACE_OFF)


class TestMxfp4MoeCallPassesTheWorkspace(CustomTestCase):
    def test_the_runner_hands_the_workspace_to_flashinfer(self):
        experts, hidden, intermediate, tokens, top_k = 4, 128, 128, 16, 2
        seen = {}
        sentinel = object()

        def fused_moe(**kwargs):
            seen.update(kwargs)
            return [kwargs["output"]]

        activation = types.SimpleNamespace(Swiglu="swiglu", SwigluStep="swiglu_step")
        dispatch = StandardDispatchOutput(
            hidden_states=torch.empty(tokens, hidden, dtype=torch.bfloat16),
            hidden_states_scale=None,
            topk_output=StandardTopKOutput(
                topk_weights=torch.full((tokens, top_k), 1.0 / top_k),
                topk_ids=torch.zeros(tokens, top_k, dtype=torch.int32),
                router_logits=torch.empty(tokens, experts),
            ),
        )
        quant_info = fc.FlashInferCutlassMxfp4MoeQuantInfo(
            w13_weight=torch.empty(
                experts, 2 * intermediate, hidden // 2, dtype=torch.uint8
            ),
            w2_weight=torch.empty(
                experts, hidden, intermediate // 2, dtype=torch.uint8
            ),
            w13_weight_scale=torch.empty(
                experts, 2 * intermediate, hidden // 32, dtype=torch.uint8
            ),
            w2_weight_scale=torch.empty(
                experts, hidden, intermediate // 32, dtype=torch.uint8
            ),
        )
        layer = _Layer()
        layer.moe_tp_size, layer.moe_tp_rank = 1, 0
        layer.moe_ep_size, layer.moe_ep_rank = 1, 0
        with (
            get_parallel().override(tp_group=None),
            mock.patch.object(fc, "_cutlass_moe_tune_max_num_tokens", lambda n: n),
            mock.patch.object(
                fc,
                "_flashinfer_cutlass_fused_moe",
                return_value=(fused_moe, activation),
            ),
            mock.patch.object(
                fc, "_mxfp4_moe_workspace", return_value=sentinel
            ) as workspace,
        ):
            fc._fused_experts_flashinfer_mxfp4_cutlass(
                dispatch_output=dispatch,
                quant_info=quant_info,
                runner_config=MoeRunnerConfig(layer=layer),
            )
        self.assertIs(seen["workspace_buffer"], sentinel)
        self.assertEqual(workspace.call_args.kwargs["top_k"], top_k)
        self.assertEqual(workspace.call_args.kwargs["output_dtype"], torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
