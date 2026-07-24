"""Regression tests for Gemma 4 full/sliding attention mask selection."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.managers.schedule_batch import ForwardMode
from sglang.srt.models.gemma4_mm import (
    Gemma4ForConditionalGeneration,
)

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


class _FakeTritonBackend:
    def __init__(self):
        self.forward_metadata = SimpleNamespace(
            custom_mask=None,
            mask_indptr=None,
            sliding_custom_mask=None,
            sliding_mask_indptr=None,
        )


class _FakeImage:
    offsets = ((1, 3),)

    @staticmethod
    def is_image():
        return True


class TestGemma4AttentionMask(CustomTestCase):
    def test_image_mask_applies_only_to_sliding_attention(self):
        backend = _FakeTritonBackend()
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            batch_size=1,
            extend_seq_lens=[6],
            extend_prefix_lens=[0],
            mm_inputs=[SimpleNamespace(mm_items=[_FakeImage()])],
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
                input_ids=torch.arange(6),
                mask_dtype=torch.bool,
            )

        full_mask, full_mask_indptr = TritonAttnBackend._get_custom_mask_for_layer(
            backend,
            SimpleNamespace(sliding_window_size=-1),
        )
        self.assertIsNone(full_mask)
        self.assertIsNone(full_mask_indptr)

        sliding_window_size = 2
        sliding_mask, sliding_mask_indptr = (
            TritonAttnBackend._get_custom_mask_for_layer(
                backend,
                SimpleNamespace(sliding_window_size=sliding_window_size),
            )
        )
        self.assertIs(sliding_mask, backend.forward_metadata.sliding_custom_mask)
        self.assertIs(
            sliding_mask_indptr,
            backend.forward_metadata.sliding_mask_indptr,
        )

        causal_mask = torch.ones(6, 6, dtype=torch.bool).tril()
        same_image_block = torch.zeros(6, 6, dtype=torch.bool)
        same_image_block[1:4, 1:4] = True
        positions = torch.arange(6)
        sliding_window_mask = positions[:, None] <= (
            positions[None, :] + sliding_window_size
        )
        expected_sliding_mask = (causal_mask | same_image_block) & sliding_window_mask
        actual_sliding_mask = sliding_mask.view(6, 6) & sliding_window_mask
        torch.testing.assert_close(actual_sliding_mask, expected_sliding_mask)

    def test_sliding_attention_falls_back_to_generic_custom_mask(self):
        generic_mask = torch.tensor([True, False])
        generic_mask_indptr = torch.tensor([0, 2], dtype=torch.int64)
        backend = SimpleNamespace(
            forward_metadata=SimpleNamespace(
                custom_mask=generic_mask,
                mask_indptr=generic_mask_indptr,
                sliding_custom_mask=None,
                sliding_mask_indptr=None,
            )
        )

        selected_mask, selected_mask_indptr = (
            TritonAttnBackend._get_custom_mask_for_layer(
                backend,
                SimpleNamespace(sliding_window_size=2),
            )
        )
        self.assertIs(selected_mask, generic_mask)
        self.assertIs(selected_mask_indptr, generic_mask_indptr)


if __name__ == "__main__":
    unittest.main()
