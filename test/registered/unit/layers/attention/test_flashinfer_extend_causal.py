"""Causal flag and sliding window of FlashInfer extend attention per attention type."""

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_EXTEND = SimpleNamespace(forward_mode=ForwardMode.EXTEND)
_DLLM_EXTEND = SimpleNamespace(forward_mode=ForwardMode.DLLM_EXTEND)


def _backend(*, allow_bidirectional: bool, multi_item_enabled: bool = False):
    backend = FlashInferAttnBackend.__new__(FlashInferAttnBackend)
    backend.allow_bidirectional_attention_in_extend = allow_bidirectional
    backend.forward_metadata = SimpleNamespace(
        multi_item_params=(
            SimpleNamespace(is_enabled=lambda: True) if multi_item_enabled else None
        )
    )
    return backend


def _gate(
    *,
    graphs_disabled: bool = True,
    breakable_prefill: bool = False,
    chunked_prefill_size: int = -1,
    cp_enabled: bool = False,
) -> bool:
    with mock.patch.multiple(
        "sglang.srt.layers.attention.flashinfer_backend",
        cuda_graph_fully_disabled=lambda: graphs_disabled,
        check_cuda_graph_backend=lambda phase, backend: breakable_prefill,
        get_schedule=lambda: SimpleNamespace(chunked_prefill_size=chunked_prefill_size),
        is_cp_enabled=lambda: cp_enabled,
    ):
        return FlashInferAttnBackend._bidirectional_extend_allowed()


def _layer(attn_type, *, is_cross_attention=False, sliding_window_size=-1):
    return SimpleNamespace(
        attn_type=attn_type,
        is_cross_attention=is_cross_attention,
        sliding_window_size=sliding_window_size,
    )


class TestFlashInferExtendCausal(CustomTestCase):
    def test_bidirectional_layer_is_not_causal(self):
        """FlashInfer prefill ran DECODER_BIDIRECTIONAL layers causally, so the
        same checkpoint attended differently than on the Triton and
        FlashAttention backends."""
        backend = _backend(allow_bidirectional=True)
        layer = _layer(AttentionType.DECODER_BIDIRECTIONAL, sliding_window_size=512)

        self.assertFalse(backend._extend_is_causal(layer, _EXTEND))
        # FlashInfer has no right window, so the layer attends the whole sequence.
        self.assertEqual(backend._extend_window_left(layer, causal=False), -1)

    def test_bidirectional_layer_stays_causal_when_gate_is_off(self):
        backend = _backend(allow_bidirectional=False)
        layer = _layer(AttentionType.DECODER_BIDIRECTIONAL, sliding_window_size=512)

        self.assertTrue(backend._extend_is_causal(layer, _EXTEND))
        self.assertEqual(backend._extend_window_left(layer, causal=True), 512)
        # A DLLM extend is one complete canvas, so it is bidirectional regardless.
        self.assertFalse(backend._extend_is_causal(layer, _DLLM_EXTEND))

    def test_bidirectional_layer_stays_causal_on_a_window_plan(self):
        """A paged wrapper planned with a window reads only each query's causal
        window, so running it without the window would skip keys."""
        backend = _backend(allow_bidirectional=True)
        layer = _layer(AttentionType.DECODER_BIDIRECTIONAL, sliding_window_size=512)

        for forward_batch in (_EXTEND, _DLLM_EXTEND):
            self.assertTrue(
                backend._extend_is_causal(layer, forward_batch, window_planned=True)
            )
        self.assertEqual(backend._extend_window_left(layer, causal=True), 512)

    def test_gate_needs_the_whole_prompt_in_one_forward(self):
        self.assertTrue(_gate())
        self.assertTrue(_gate(graphs_disabled=False, breakable_prefill=True))
        self.assertFalse(_gate(graphs_disabled=False))
        self.assertFalse(_gate(chunked_prefill_size=4096))
        # A context-parallel shard reads keys only up to its own end.
        self.assertFalse(_gate(cp_enabled=True))

    def test_other_attention_types_are_unchanged(self):
        backend = _backend(allow_bidirectional=True)

        self.assertTrue(
            backend._extend_is_causal(_layer(AttentionType.DECODER), _EXTEND)
        )
        self.assertFalse(
            backend._extend_is_causal(_layer(AttentionType.ENCODER_ONLY), _EXTEND)
        )
        self.assertFalse(
            backend._extend_is_causal(
                _layer(AttentionType.DECODER, is_cross_attention=True), _EXTEND
            )
        )

    def test_multi_item_scoring_disables_sliding_window(self):
        backend = _backend(allow_bidirectional=False, multi_item_enabled=True)
        layer = _layer(AttentionType.DECODER, sliding_window_size=512)

        self.assertEqual(backend._extend_window_left(layer, causal=True), -1)


if __name__ == "__main__":
    unittest.main()
