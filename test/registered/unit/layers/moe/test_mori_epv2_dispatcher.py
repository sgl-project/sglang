"""CPU-only tests for how the MORI EPv2 dispatcher drives its transport.

The MORI op is a fake that logs its calls, so these pin which calls each mode
makes and when, not what the kernels compute.
"""

import os
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.token_dispatcher import moriep
from sglang.srt.layers.moe.token_dispatcher.moriep import MoriEPDispatcher
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import DeepEPMode
from sglang.srt.runtime_context import get_flags, reset_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_HIDDEN = 8
_TOPK = 2


def _fake_mori_modules():
    """Just enough of MORI for the EPv2 impls' import probe."""
    mori = ModuleType("mori")
    cco = ModuleType("mori.cco")
    cco.Communicator = object
    ops = ModuleType("mori.ops")
    v2 = ModuleType("mori.ops.dispatch_combine_v2")
    v2.EpDispatchCombineOp = object
    return {
        "mori": mori,
        "mori.cco": cco,
        "mori.ops": ops,
        "mori.ops.dispatch_combine_v2": v2,
    }


class _FakeOp:
    def __init__(self, log):
        self.log = log
        self.cfg = SimpleNamespace(
            effective_max_recv=64, is_internode=False, is_scatter=False
        )
        self.backend_name = "hip"
        self.dispatched = None
        self.combined = None

    def _dispatch_result(self, hidden, weights, indices):
        self.dispatched = (
            hidden.clone(),
            weights.clone(),
            None,
            indices.clone(),
            torch.tensor([hidden.shape[0]]),
            "routing",
        )
        return self.dispatched

    def dispatch(self, hidden, weights, scales, indices, **kwargs):
        self.log.append("dispatch")
        return self._dispatch_result(hidden, weights, indices)

    def dispatch_send(self, hidden, weights, scales, indices, return_routing):
        self.log.append("dispatch_send")
        return self._dispatch_result(hidden, weights, indices)

    def dispatch_recv(self):
        self.log.append("dispatch_recv")

    def combine(self, hidden, weights, indices, routing):
        self.log.append("combine")
        self.combined = hidden.clone()
        return self.combined, None

    def combine_send(self, hidden, weights, indices, routing):
        self.log.append("combine_send")
        self.combined = hidden.clone()
        return self.combined, None

    def combine_recv(self):
        self.log.append("combine_recv")


class _MoriEPv2TestCase(CustomTestCase):
    def setUp(self):
        reset_context()
        self.log = []
        self._patches = [
            patch.dict(sys.modules, _fake_mori_modules()),
            patch.dict(os.environ),
            patch.object(moriep, "get_is_extend_in_batch", return_value=False),
            patch.object(
                moriep, "_should_record_expert_distribution", return_value=False
            ),
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
        ]
        for item in self._patches:
            item.start()
        os.environ.pop("MORI_ENABLE_SDMA", None)
        envs.SGLANG_MORI_EP_V2.set(True)

    def tearDown(self):
        for item in reversed(self._patches):
            item.stop()
        reset_context()

    def _make_dispatcher(self, deepep_mode, async_finish=False):
        dispatcher = MoriEPDispatcher(
            group=None,
            router_topk=_TOPK,
            hidden_size=_HIDDEN,
            params_dtype=torch.bfloat16,
            deepep_mode=deepep_mode,
            async_finish=async_finish,
        )
        dispatcher._get_impl()._mori_op = _FakeOp(self.log)
        return dispatcher

    @staticmethod
    def _inputs(num_tokens=3):
        hidden = torch.randn(num_tokens, _HIDDEN, dtype=torch.bfloat16)
        ids = torch.randint(0, 4, (num_tokens, _TOPK), dtype=torch.int32)
        weights = torch.rand(num_tokens, _TOPK)
        return hidden, StandardTopKOutput(weights, ids, None)


class TestMoriEPv2Transport(_MoriEPv2TestCase):
    def test_low_latency_sends_in_a_and_receives_in_b(self):
        """TBO overlaps the sibling batch with the transfer only if the send half
        runs in *_a and the receive half in *_b; both halves in *_b still give the
        right result but lose the overlap."""
        dispatcher = self._make_dispatcher(DeepEPMode.LOW_LATENCY)
        op = dispatcher._get_impl().mori_op
        hidden, topk_output = self._inputs()

        dispatcher.dispatch_a(hidden, topk_output)
        self.log.append("sibling")
        dispatched = dispatcher.dispatch_b()
        self.assertEqual(self.log, ["dispatch_send", "sibling", "dispatch_recv"])
        # dispatch_b hands out the send half's buffers, which the receive fills.
        self.assertIs(dispatched.hidden_states, op.dispatched[0])
        self.assertIs(dispatched.topk_ids, op.dispatched[3])

        del self.log[:]
        dispatcher.combine_a(
            (dispatched.hidden_states, dispatched.topk_ids, dispatched.topk_weights)
        )
        self.log.append("sibling")
        combined = dispatcher.combine_b()
        self.assertEqual(self.log, ["combine_send", "sibling", "combine_recv"])
        self.assertIs(combined, op.combined)

    def test_recv_bound_skips_async_ll_layouts(self):
        """mori_recv_bound is proven only for the intranode receive layout;
        trimming an AsyncLL receive view could drop received rows."""
        for deepep_mode, sdma, verified in (
            (DeepEPMode.NORMAL, False, True),
            (DeepEPMode.NORMAL, True, False),
            (DeepEPMode.LOW_LATENCY, False, False),
        ):
            with self.subTest(deepep_mode=deepep_mode, sdma=sdma):
                os.environ["MORI_ENABLE_SDMA"] = "1" if sdma else "0"
                impl = self._make_dispatcher(deepep_mode)._get_impl()
                self.assertEqual(impl._is_recv_layout_verified(), verified)


class TestMoriEPv2TboPerCall(_MoriEPv2TestCase):
    """--enable-two-batch-overlap must only change the calls that belong to a TBO
    micro-batch; decode, verify and unsplit prefill forwards of the same server
    keep the ordinary transfer settings."""

    def _make_tbo_dispatcher(self):
        stream = object()
        with (
            get_flags().moe.override(tbo_enabled=True),
            patch.object(
                moriep.CommStreamPool, "get_stream_from_pool", return_value=stream
            ),
        ):
            dispatcher = self._make_dispatcher(DeepEPMode.NORMAL, async_finish=True)
        self.assertIs(dispatcher._get_impl()._comm_stream, stream)
        return dispatcher

    def test_tbo_child_marked_only_while_its_stage_runs(self):
        """A TBO child's stage sees itself marked; once the stage returns the mark
        is gone, so later forwards are ordinary even where no DP metadata is
        republished (no DP attention)."""
        from sglang.srt.batch_overlap.operations import _StageExecutor
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            get_forward_context,
            has_forward_context,
        )

        seen = []

        def probe(state, **inputs):
            seen.append(get_forward_context().is_micro_batch)
            return inputs

        child = SimpleNamespace(
            global_dp_buffer_len=None,
            tbo_padded_len=None,
            global_num_tokens_cpu=None,
            dp_padding_mode=SimpleNamespace(is_max_len=lambda: False),
            moe_num_token_non_padded=None,
        )
        # Like execute_overlapped_operations, hand the child its own context.
        executor = _StageExecutor(
            "a",
            [[SimpleNamespace(debug_name="probe", fn=probe)]],
            inputs={"forward_batch": child},
            child_ctx=ForwardContext(attn_backend=None),
            micro_batch=True,
        )
        executor.next()
        self.assertEqual(seen, [True])
        self.assertFalse(has_forward_context())

        # An unsplit forward (execute_operations) runs no micro-batch.
        _StageExecutor(
            "primary",
            [[SimpleNamespace(debug_name="probe", fn=probe)]],
            inputs={"forward_batch": child},
            child_ctx=ForwardContext(attn_backend=None),
        ).next()
        self.assertEqual(seen[-1], False)

    def test_comm_stream_only_for_micro_batches(self):
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )

        dispatcher = self._make_tbo_dispatcher()
        with forward_context(ForwardContext(attn_backend=None, is_micro_batch=True)):
            self.assertTrue(dispatcher._get_impl()._use_comm_stream())

        # An ordinary forward runs the transfer on the compute stream: no event,
        # no comm-stream hop.
        impl = dispatcher._get_impl()
        self.assertFalse(impl._use_comm_stream())
        self.assertIsNone(impl._capture_event_if_async())
        hidden, topk_output = self._inputs()
        dispatched = dispatcher.dispatch(hidden, topk_output)
        dispatcher.combine(
            (dispatched.hidden_states, dispatched.topk_ids, dispatched.topk_weights)
        )
        self.assertEqual(self.log, ["dispatch", "combine"])


if __name__ == "__main__":
    unittest.main()
