"""Staging slot ids stay aligned once a draft KV pool is registered.

The staging gather writes every k_buffer and then every v_buffer, while
kv_data_ptrs (and therefore kv_layer_ids) is ordered
[K target, V target, K draft, V draft]. Labelling slots with kv_layer_ids
silently pairs a layer's KV with another layer's staging slot as soon as a
draft pool exists.
"""

import unittest

from sglang.srt.disaggregation.utils import build_staging_slot_metadata
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Pool(MHATokenToKVPool):
    def __init__(self, tag, layer_ids):
        self.k_buffer = [f"{tag}K{i}" for i in layer_ids]
        self.v_buffer = [f"{tag}V{i}" for i in layer_ids]


def _kv_layer_ids(target_ids, draft_ids):
    """kv_data_ptrs order: K target, V target, K draft, V draft."""
    return list(target_ids) + list(target_ids) + list(draft_ids) + list(draft_ids)


class TestStagingDraftKvSlots(CustomTestCase):
    def test_undescribable_draft_still_yields_target_buffers(self):
        # Returning nothing here left the caller skipping set_kv_buffer_tensors
        # entirely, and staging then came up with no buffers at all.
        class _NoBuffers:
            pass

        k_buffers, v_buffers, slot_ids = build_staging_slot_metadata(
            kv_layer_ids=_kv_layer_ids([87], [92]),
            num_draft_entries=2,
            kv_pool=_Pool("t", [87]),
            draft_kv_pool=_NoBuffers(),
        )
        self.assertEqual(k_buffers, ["tK87"])
        self.assertEqual(v_buffers, ["tV87"])
        self.assertEqual(slot_ids, [])

    def test_pool_without_contiguous_tensors_is_declined(self):
        # MLA pools have no k_buffer/v_buffer to stage; the caller relies on None
        # to skip the registration rather than register empty lists.
        class _NoBuffers:
            pass

        self.assertIsNone(
            build_staging_slot_metadata(
                kv_layer_ids=[],
                num_draft_entries=0,
                kv_pool=_NoBuffers(),
                draft_kv_pool=None,
            )
        )


if __name__ == "__main__":
    unittest.main()
