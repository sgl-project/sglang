"""CPU coverage for Gemma-4's image-mask producer on the FA4 metadata path.

`prepare_attn_masks` must hand `image_token_ranges` to the prefill backend
both when it is fa4 and when it is fa3 upgraded to FA4 for the 512-wide
global layers; any other non-triton backend falls back to causal attention
without touching the metadata.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.models import gemma4_mm
from sglang.srt.models.gemma4_mm import Gemma4ForConditionalGeneration
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _image(*offsets):
    return MultimodalInputs(
        mm_items=[MultimodalDataItem(modality=Modality.IMAGE, offsets=list(offsets))]
    )


def _forward_batch(extend_lens, prefix_lens, mm_inputs):
    return SimpleNamespace(
        extend_seq_lens_cpu=list(extend_lens),
        extend_prefix_lens_cpu=list(prefix_lens),
        mm_inputs=list(mm_inputs),
    )


def _prepare(backend, forward_batch):
    with patch.object(gemma4_mm, "get_attn_backend", return_value=backend):
        Gemma4ForConditionalGeneration.prepare_attn_masks(
            None, forward_batch, torch.zeros(1, dtype=torch.int64), torch.bfloat16
        )


class TestGemma4ImageTokenRanges(CustomTestCase):
    def _backend(self, **flags):
        return SimpleNamespace(forward_metadata=SimpleNamespace(), **flags)

    def test_fa3_upgraded_to_fa4_emits_image_token_ranges(self):
        # Two requests: a fresh one with an image at [2, 5], and a chunked
        # one whose image at absolute [5, 7] sits inside the extend window.
        backend = self._backend(fa3_upgrades_to_fa4=True)
        _prepare(
            backend,
            _forward_batch(
                extend_lens=(8, 5),
                prefix_lens=(0, 4),
                mm_inputs=(_image((2, 5)), _image((5, 7))),
            ),
        )

        ranges = backend.forward_metadata.image_token_ranges
        self.assertEqual(ranges.dtype, torch.int32)
        expected = torch.full((13, 2), -1, dtype=torch.int32)
        expected[2:6] = torch.tensor([2, 5], dtype=torch.int32)
        expected[8 + 1 : 8 + 4] = torch.tensor([5, 7], dtype=torch.int32)
        self.assertTrue(torch.equal(ranges, expected), (ranges, expected))

    def test_fa4_backend_takes_the_same_path(self):
        backend = self._backend(fa_impl_ver=4)
        _prepare(backend, _forward_batch((4,), (0,), (_image((1, 2)),)))
        expected = torch.tensor([[-1, -1], [1, 2], [1, 2], [-1, -1]], dtype=torch.int32)
        self.assertTrue(
            torch.equal(backend.forward_metadata.image_token_ranges, expected)
        )

    def test_hybrid_wrapper_is_resolved_through_prefill_backend(self):
        prefill = self._backend(fa3_upgrades_to_fa4=True)
        wrapper = SimpleNamespace(
            prefill_backend=prefill, forward_metadata=SimpleNamespace()
        )
        _prepare(wrapper, _forward_batch((3,), (0,), (_image((0, 1)),)))
        self.assertTrue(hasattr(prefill.forward_metadata, "image_token_ranges"))
        self.assertFalse(hasattr(wrapper.forward_metadata, "image_token_ranges"))

    def test_split_image_and_non_image_items_stay_causal(self):
        backend = self._backend(fa3_upgrades_to_fa4=True)
        audio = MultimodalInputs(
            mm_items=[MultimodalDataItem(modality=Modality.AUDIO, offsets=[(0, 2)])]
        )
        _prepare(
            backend,
            _forward_batch(
                extend_lens=(4, 4),
                prefix_lens=(2, 0),
                # Image [1, 3] straddles the chunk start at prefix 2.
                mm_inputs=(_image((1, 3)), audio),
            ),
        )
        ranges = backend.forward_metadata.image_token_ranges
        self.assertTrue(torch.all(ranges == -1), ranges)

    def test_plain_fa3_without_upgrade_leaves_metadata_alone(self):
        backend = self._backend(fa_impl_ver=3, fa3_upgrades_to_fa4=False)
        _prepare(backend, _forward_batch((4,), (0,), (_image((1, 2)),)))
        self.assertFalse(hasattr(backend.forward_metadata, "image_token_ranges"))


if __name__ == "__main__":
    unittest.main()
