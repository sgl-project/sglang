"""Unit tests for MetadataBuffers.get_spec_only_aux_indices().

These tests guard the invariant that ``get_spec_only_aux_indices()``
returns the exact positions occupied by ``output_topk_p``,
``output_topk_index`` and ``output_hidden_states`` within the list
returned by :meth:`MetadataBuffers.get_buf_infos`, for every combination
of the optional aux slots (``enable_sampling_mask``, ``dsa_topk_indices``).

If a future change adds or reorders aux slots without updating
``get_spec_only_aux_indices()``, the send-side skip in
``mooncake/conn.py::send_aux`` would target the wrong slots and re-open
the PP + MTP garbled-output race documented in the accompanying fix.
"""

from __future__ import annotations

import contextlib
import itertools
import unittest
from unittest.mock import patch

import torch

from sglang.srt.disaggregation.utils import MetadataBuffers


# The DSA optional slot is enabled by setting output_dsa_topk_indices_dim>0.
DSA_DIMS = (0, 8)
SAMPLING_MASK_VALUES = (0, 32)  # 0 = disabled; >0 = enabled


class MetadataBuffersSpecOnlyIndicesTest(unittest.TestCase):
    def _make_buffers(
        self, *, sampling_mask_tokens: int, dsa_dim: int
    ) -> MetadataBuffers:
        # MetadataBuffers touches torch.cuda.use_mem_pool via nullcontext when
        # custom_mem_pool is None -- fine on CPU. Force CPU-side tensors by
        # leaving custom_mem_pool=None and letting is_npu()/nvlink checks fall
        # through to device="cpu".
        return MetadataBuffers(
            size=2,
            hidden_size=16,
            hidden_states_dtype=torch.float32,
            max_sampling_mask_tokens=sampling_mask_tokens,
            output_dsa_topk_indices_dim=dsa_dim,
        )

    def test_indices_match_actual_buf_positions(self):
        for sampling_mask_tokens, dsa_dim in itertools.product(
            SAMPLING_MASK_VALUES, DSA_DIMS
        ):
            with self.subTest(
                sampling_mask_tokens=sampling_mask_tokens, dsa_dim=dsa_dim
            ):
                buffers = self._make_buffers(
                    sampling_mask_tokens=sampling_mask_tokens, dsa_dim=dsa_dim
                )
                ptrs, _, _ = buffers.get_buf_infos()
                indices = buffers.get_spec_only_aux_indices()

                # Exactly three spec-only slots.
                self.assertEqual(len(indices), 3)
                # Contiguous.
                self.assertEqual(indices, list(range(indices[0], indices[0] + 3)))
                # All within bounds.
                self.assertTrue(all(0 <= i < len(ptrs) for i in indices))
                # And they must alias the three spec-only tensors exactly.
                self.assertEqual(ptrs[indices[0]], buffers.output_topk_p.data_ptr())
                self.assertEqual(ptrs[indices[1]], buffers.output_topk_index.data_ptr())
                self.assertEqual(
                    ptrs[indices[2]], buffers.output_hidden_states.data_ptr()
                )

    def test_indices_shift_with_sampling_mask(self):
        # Baseline (no sampling_mask, no dsa): spec-only trio starts at 6.
        buffers_off = self._make_buffers(sampling_mask_tokens=0, dsa_dim=0)
        self.assertEqual(buffers_off.get_spec_only_aux_indices(), [6, 7, 8])

        # Sampling mask on shifts spec-only trio by +3 (mask_len / mask_idx /
        # sampling_logprobs are inserted before spec-only).
        buffers_on = self._make_buffers(sampling_mask_tokens=32, dsa_dim=0)
        self.assertEqual(buffers_on.get_spec_only_aux_indices(), [9, 10, 11])

    def test_dsa_does_not_shift_spec_only(self):
        # DSA slot is appended after spec-only, so it must not affect the trio.
        buffers_no_dsa = self._make_buffers(sampling_mask_tokens=0, dsa_dim=0)
        buffers_dsa = self._make_buffers(sampling_mask_tokens=0, dsa_dim=8)
        self.assertEqual(
            buffers_no_dsa.get_spec_only_aux_indices(),
            buffers_dsa.get_spec_only_aux_indices(),
        )


if __name__ == "__main__":
    unittest.main()
