"""CPU-only tests for the DeepEP v2 BF16 / FP8 wire format."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.moe.token_dispatcher import deepep_v2
from sglang.srt.layers.moe.utils import (
    DeepEPv2Fp8ScaleFormat,
    DispatcherOutputDtype,
    get_deepep_v2_dispatcher_output_dtype,
)
from sglang.srt.runtime_context import get_context, get_exec, reset_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

HIDDEN = 256
TOPK = 4
NUM_EXPERTS = 8
NUM_LOCAL_EXPERTS = 2
NUM_MAX_TOKENS = 16


class _FakeGroup:
    pass


class _FakeHandle:
    def __init__(self, num_recv):
        self.psum_num_recv_tokens_per_scaleup_rank = torch.tensor([num_recv])
        self.psum_num_recv_tokens_per_expert = torch.tensor(
            [num_recv] * NUM_LOCAL_EXPERTS, dtype=torch.int32
        )


class _FakeBuffer:
    """Echoes the dispatch input back so the wire format stays observable."""

    last = None

    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self.num_bytes = 1 << 20
        type(self).last = self

    def dispatch(self, x, **kwargs):
        self.dispatch_x = x
        num_recv = (x[0] if isinstance(x, tuple) else x).shape[0]
        topk_idx = kwargs["topk_idx"]
        topk_weights = kwargs["topk_weights"]
        event = SimpleNamespace(event=None, current_stream_wait=lambda: None)
        return x, topk_idx, topk_weights, _FakeHandle(num_recv), event


def _fake_quant(hidden_states, block_size, **kwargs):
    return (
        torch.zeros_like(hidden_states, dtype=torch.float8_e4m3fn),
        torch.zeros(
            (hidden_states.shape[0], hidden_states.shape[1] // block_size),
            dtype=torch.float32,
        ),
    )


class _DeepEPv2WireDtypeBase(CustomTestCase):
    def setUp(self):
        reset_context()
        # `_get_allow_hybrid_mode` reads the published `exec.moe` bag.
        self._published = get_context().override_server_args(model_path="dummy")
        self._published.install()
        _FakeBuffer.last = None
        self._patches = [
            patch.object(deepep_v2, "use_deepep_v2", True),
            patch.object(deepep_v2, "ElasticBuffer", _FakeBuffer, create=True),
            patch.object(deepep_v2, "sglang_per_token_group_quant_fp8", _fake_quant),
            patch.object(deepep_v2.dist, "get_world_size", return_value=4),
            patch.object(deepep_v2.dist, "get_rank", return_value=0),
            patch.object(
                deepep_v2,
                "get_deepep_v2_fp8_scale_format",
                lambda: DeepEPv2Fp8ScaleFormat(tma_aligned=False, ue8m0=False),
            ),
        ]
        for item in self._patches:
            item.start()

    def tearDown(self):
        self._published.restore()
        reset_context()
        for item in reversed(self._patches):
            item.stop()

    def _dispatch(self, use_fp8_dispatch, num_tokens=8, is_extend_in_batch=True):
        dispatcher = deepep_v2.DeepEPv2Dispatcher(
            group=_FakeGroup(),
            router_topk=TOPK,
            num_experts=NUM_EXPERTS,
            num_local_experts=NUM_LOCAL_EXPERTS,
            hidden_size=HIDDEN,
            params_dtype=torch.bfloat16,
            use_fp8_dispatch=use_fp8_dispatch,
        )
        dispatcher._impl.num_max_dispatch_tokens_per_rank = NUM_MAX_TOKENS
        hidden_states = torch.randn((num_tokens, HIDDEN), dtype=torch.bfloat16)
        topk_output = SimpleNamespace(
            topk_ids=torch.zeros((num_tokens, TOPK), dtype=torch.int32),
            topk_weights=torch.ones((num_tokens, TOPK), dtype=torch.float32),
        )
        with patch.object(
            deepep_v2, "get_is_extend_in_batch", lambda: is_extend_in_batch
        ):
            return hidden_states, dispatcher._impl.dispatch(hidden_states, topk_output)


class TestDeepEPv2WireDtype(_DeepEPv2WireDtypeBase):
    def test_bf16_dispatch_sends_unquantized_activations(self):
        hidden_states, out = self._dispatch(use_fp8_dispatch=False)
        self.assertIs(_FakeBuffer.last.dispatch_x, hidden_states)
        self.assertIsNone(out.hidden_states_scale)
        self.assertEqual(out.hidden_states.dtype, torch.bfloat16)
        self.assertFalse(out.hidden_states_scale_tma_aligned)

    def test_fp8_dispatch_still_sends_activations_and_scales(self):
        _, out = self._dispatch(use_fp8_dispatch=True)
        self.assertIsInstance(_FakeBuffer.last.dispatch_x, tuple)
        self.assertIsNotNone(out.hidden_states_scale)
        self.assertEqual(out.hidden_states.dtype, torch.float8_e4m3fn)

    def test_bf16_masked_decode_dispatch_stays_unquantized(self):
        hidden_states, out = self._dispatch(
            use_fp8_dispatch=False, is_extend_in_batch=False
        )
        self.assertIs(_FakeBuffer.last.dispatch_x, hidden_states)
        self.assertTrue(out.use_masked_gemm)
        self.assertIsNone(out.hidden_states_scale)

    def test_wire_dtype_selects_the_elastic_buffer_layout(self):
        for use_fp8_dispatch in (True, False):
            with self.subTest(use_fp8_dispatch=use_fp8_dispatch):
                _FakeBuffer.last = None
                deepep_v2.DeepEPv2Buffer.destroy()
                self._dispatch(use_fp8_dispatch=use_fp8_dispatch)
                self.assertEqual(
                    _FakeBuffer.last.kwargs["use_fp8_dispatch"], use_fp8_dispatch
                )


class TestDeepEPv2DispatcherOutputDtype(CustomTestCase):
    def setUp(self):
        reset_context()
        self._published = get_context().override_server_args(model_path="dummy")
        self._published.install()

    def tearDown(self):
        self._published.restore()
        reset_context()

    def _flag(self, value):
        return get_exec().moe.override(deepep_dispatcher_output_dtype=value)

    def test_auto_follows_the_checkpoint(self):
        with self._flag("auto"):
            self.assertIs(
                get_deepep_v2_dispatcher_output_dtype(True), DispatcherOutputDtype.FP8
            )
            self.assertIs(
                get_deepep_v2_dispatcher_output_dtype(False), DispatcherOutputDtype.BF16
            )

    def test_explicit_flag_matching_the_checkpoint_is_accepted(self):
        with self._flag("bf16"):
            self.assertIs(
                get_deepep_v2_dispatcher_output_dtype(False), DispatcherOutputDtype.BF16
            )
        with self._flag("fp8"):
            self.assertIs(
                get_deepep_v2_dispatcher_output_dtype(True), DispatcherOutputDtype.FP8
            )

    def test_explicit_flag_contradicting_the_checkpoint_is_rejected(self):
        with self._flag("bf16"):
            with self.assertRaisesRegex(ValueError, "contradicts this checkpoint"):
                get_deepep_v2_dispatcher_output_dtype(True)
        with self._flag("fp8"):
            with self.assertRaisesRegex(ValueError, "contradicts this checkpoint"):
                get_deepep_v2_dispatcher_output_dtype(False)


if __name__ == "__main__":
    unittest.main()
