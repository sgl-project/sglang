"""takes_fp8_dcp_verify_q must reach the backend that runs target verify through every wrapper.

A wrapper that falls back to the base default, or asks the wrong child,
leaves the model gathering bf16 for cprr or fp8 for a Gluon consumer that needs the window.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.hybrid_attn_backend import HybridAttnBackend
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.layers.attention.tbo_backend import TboAttnBackend
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.models.deepseek_common.attention_forward_methods import forward_mla_rocm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _child(takes: bool):
    return SimpleNamespace(takes_fp8_dcp_verify_q=lambda: takes)


def _wrapper(cls, **attrs):
    backend = cls.__new__(cls)
    backend.__dict__.update(attrs)
    return backend


class TestTakesFp8DcpVerifyQ(CustomTestCase):
    def test_base_backend_declines(self):
        self.assertFalse(AttentionBackend.takes_fp8_dcp_verify_q(object()))

    def test_hybrid_asks_the_target_verify_child(self):
        for spec_attn_is_decode in (True, False):
            with self.subTest(spec_attn_is_decode=spec_attn_is_decode):
                backend = _wrapper(
                    HybridAttnBackend,
                    spec_attn_is_decode=spec_attn_is_decode,
                    decode_backend=_child(spec_attn_is_decode),
                    prefill_backend=_child(not spec_attn_is_decode),
                )
                self.assertTrue(backend.takes_fp8_dcp_verify_q())

    def test_hybrid_linear_asks_the_full_attention_child(self):
        for takes in (True, False):
            with self.subTest(takes=takes):
                backend = _wrapper(
                    HybridLinearAttnBackend, full_attn_backend=_child(takes)
                )
                self.assertEqual(backend.takes_fp8_dcp_verify_q(), takes)

    def test_tbo_asks_the_primary(self):
        for takes in (True, False):
            with self.subTest(takes=takes):
                backend = _wrapper(TboAttnBackend, primary=_child(takes))
                self.assertEqual(backend.takes_fp8_dcp_verify_q(), takes)


class TestDcpVerifyQFp8Routing(CustomTestCase):
    """The model gathers an fp8 query only where core can consume it unscaled."""

    def _routes(
        self, *, aiter=True, mode=ForwardMode.TARGET_VERIFY, scaling=None, takes=True
    ):
        with (
            patch.object(forward_mla_rocm, "_use_aiter", aiter),
            patch.object(
                forward_mla_rocm, "get_attn_backend", return_value=_child(takes)
            ),
        ):
            return forward_mla_rocm._gathers_dcp_verify_q_fp8(
                forward_batch=SimpleNamespace(forward_mode=mode),
                llama_4_scaling=scaling,
            )

    def test_target_verify_on_a_taking_backend_gathers_fp8(self):
        self.assertTrue(self._routes())

    def test_each_other_condition_keeps_the_bf16_gather(self):
        for name, kwargs in (
            ("not aiter", dict(aiter=False)),
            ("decode", dict(mode=ForwardMode.DECODE)),
            ("llama 4 scaling", dict(scaling=object())),
            ("backend declines", dict(takes=False)),
        ):
            with self.subTest(case=name):
                self.assertFalse(self._routes(**kwargs))


if __name__ == "__main__":
    unittest.main()
