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
from sglang.srt.runtime_context import reset_context
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

    def test_normal_sdma_receives_right_after_sending(self):
        """EPv2 reaches SDMA only through the AsyncLL kernels, so normal + SDMA must
        use them (the native calls do not exist on an AsyncLL op) and, with
        nothing to overlap, receive as soon as it sends."""
        os.environ["MORI_ENABLE_SDMA"] = "1"
        dispatcher = self._make_dispatcher(DeepEPMode.NORMAL)
        hidden, topk_output = self._inputs()

        dispatcher.dispatch_a(hidden, topk_output)
        self.log.append("sibling")
        dispatched = dispatcher.dispatch_b()
        dispatcher.combine_a(
            (dispatched.hidden_states, dispatched.topk_ids, dispatched.topk_weights)
        )
        self.log.append("sibling")
        dispatcher.combine_b()
        self.assertEqual(
            self.log,
            [
                "dispatch_send",
                "dispatch_recv",
                "sibling",
                "combine_send",
                "combine_recv",
                "sibling",
            ],
        )

    def test_async_ll_replaces_fp4_dispatch(self):
        """AsyncLL has no fp4 dispatch: FP4-weight MoE falls back to MXFP8 when
        aiter can quantize it, else to bf16; other weights fall back to bf16, and
        so does gfx1250, whose a8w4 MoE takes no pre-quantized activation."""
        fp4, bf16 = torch.float4_e2m1fn_x2, torch.bfloat16
        for weight_dtype, mxfp8_ok, gfx1250, expected in (
            (fp4, True, False, moriep.DispatchDtype.mxfp8),
            (fp4, False, False, moriep.DispatchDtype.bf16),
            (bf16, True, False, moriep.DispatchDtype.bf16),
            (fp4, True, True, moriep.DispatchDtype.bf16),
        ):
            with self.subTest(
                weight_dtype=weight_dtype, mxfp8_ok=mxfp8_ok, gfx1250=gfx1250
            ):
                impl = self._make_dispatcher(DeepEPMode.LOW_LATENCY)._get_impl()
                impl.quant_config = {"weight_dtype": weight_dtype}
                impl.dispatch_dtype = moriep.DispatchDtype.fp4
                with (
                    patch.object(
                        moriep, "_aiter_supports_mxfp8_dispatch", return_value=mxfp8_ok
                    ),
                    patch.object(moriep, "is_gfx1250_supported", return_value=gfx1250),
                ):
                    impl._restrict_transport_dtypes()
                self.assertEqual(impl.dispatch_dtype, expected)


if __name__ == "__main__":
    unittest.main()
