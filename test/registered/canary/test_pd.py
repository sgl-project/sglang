"""Pin the "PD is unaware of canary" invariant.

After the stateless redesign:

- The K/V shadow tensors ride the regular KV transfer via the layer-shaped
  ``get_contiguous_buf_infos`` patch — PD's transport sees them as two
  extra layers. No ``MetadataBuffers`` extension, no canary-aware PD field.
- The decode side has no canary host state to transport. The first decode
  forward reads (req_id, K_req) from the sglang ``ForwardBatch`` itself
  and the splitmix64 chain restarts from ``CanaryConfig.seed`` until the
  decode side's writes start advancing it.

These tests pin the contract so future PD changes can't silently re-add
canary-aware fields.
"""

from __future__ import annotations

import unittest

import torch

from sglang.srt.disaggregation.utils import MetadataBuffers
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="base-b-test-1-gpu-small")


class TestMetadataBuffersHasNoCanaryFields(unittest.TestCase):
    def test_metadata_buffers_constructs_without_canary_kwarg(self) -> None:
        mb = MetadataBuffers(size=8, hidden_size=16, hidden_states_dtype=torch.float32)
        self.assertFalse(hasattr(mb, "canary_k_req"))
        self.assertFalse(hasattr(mb, "canary_prev_hash_tail"))

    def test_get_buf_infos_returns_only_original_entries(self) -> None:
        mb = MetadataBuffers(size=4, hidden_size=16, hidden_states_dtype=torch.float32)
        ptrs, lens, item_lens = mb.get_buf_infos()
        self.assertEqual(len(ptrs), 10)
        self.assertEqual(len(lens), 10)
        self.assertEqual(len(item_lens), 10)

    def test_get_buf_returns_only_original_entries(self) -> None:
        mb = MetadataBuffers(size=4, hidden_size=16, hidden_states_dtype=torch.float32)
        outputs = mb.get_buf(2)
        self.assertEqual(len(outputs), 10)


if __name__ == "__main__":
    unittest.main()
