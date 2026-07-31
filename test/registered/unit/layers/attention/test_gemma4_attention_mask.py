"""Regression tests for compact Gemma 4 image-attention metadata."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.attention.triton_backend import (  # noqa: E402
    TritonAttnBackend,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from sglang.srt.models.gemma4_mm import (  # noqa: E402
    Gemma4ForConditionalGeneration,
)

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


class _FakeTritonBackend:
    def __init__(self):
        self.forward_metadata = SimpleNamespace(
            custom_mask=None,
            mask_indptr=None,
            image_span_indptr=None,
            image_span_begin=None,
            image_span_end=None,
        )


class _FakeImage:
    def __init__(self, offsets):
        self.offsets = offsets

    @staticmethod
    def is_image():
        return True


class TestGemma4AttentionMask(CustomTestCase):
    def test_image_mask_uses_compact_spans_on_sliding_attention(self):
        backend = _FakeTritonBackend()
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            batch_size=1,
            extend_seq_lens_cpu=[8],
            extend_prefix_lens_cpu=[0],
            mm_inputs=[
                SimpleNamespace(mm_items=[_FakeImage(offsets=((1, 3), (5, 6)))])
            ],
        )

        with (
            patch(
                "sglang.srt.models.gemma4_mm.TritonAttnBackend",
                _FakeTritonBackend,
            ),
            patch(
                "sglang.srt.models.gemma4_mm.get_attn_backend",
                return_value=backend,
            ),
        ):
            Gemma4ForConditionalGeneration.prepare_attn_masks(
                None,
                forward_batch,
                input_ids=torch.arange(8),
                mask_dtype=torch.bool,
            )

        metadata = backend.forward_metadata
        torch.testing.assert_close(
            metadata.image_span_indptr, torch.tensor([0, 2], dtype=torch.int64)
        )
        torch.testing.assert_close(
            metadata.image_span_begin, torch.tensor([1, 5], dtype=torch.int64)
        )
        torch.testing.assert_close(
            metadata.image_span_end, torch.tensor([4, 7], dtype=torch.int64)
        )

        full_spans = TritonAttnBackend._get_image_spans_for_layer(
            backend, SimpleNamespace(sliding_window_size=-1)
        )
        self.assertTrue(all(value is None for value in full_spans.values()))

        sliding_spans = TritonAttnBackend._get_image_spans_for_layer(
            backend, SimpleNamespace(sliding_window_size=1024)
        )
        self.assertIs(sliding_spans["image_span_indptr"], metadata.image_span_indptr)
        self.assertIs(sliding_spans["image_span_begin"], metadata.image_span_begin)
        self.assertIs(sliding_spans["image_span_end"], metadata.image_span_end)

        dense_elements = 8 * 8
        compact_elements = sum(tensor.numel() for tensor in sliding_spans.values())
        self.assertLess(compact_elements, dense_elements)


if __name__ == "__main__":
    unittest.main()
