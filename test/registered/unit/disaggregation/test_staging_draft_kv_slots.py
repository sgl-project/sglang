"""Staging slot ids stay aligned once a draft KV pool is registered.

The staging gather writes every k_buffer and then every v_buffer, while
kv_data_ptrs (and therefore kv_layer_ids) is ordered
[K target, V target, K draft, V draft]. Labelling slots with kv_layer_ids
silently pairs a layer's KV with another layer's staging slot as soon as a
draft pool exists.
"""

import unittest

from sglang.srt.disaggregation.utils import (
    build_staging_slot_metadata,
    build_transfer_entry_pairs,
)
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, MHATokenToKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=9, suite="base-a-test-cpu")


class _Pool(MHATokenToKVPool):
    def __init__(self, tag, layer_ids):
        self.k_buffer = [f"{tag}K{i}" for i in layer_ids]
        self.v_buffer = [f"{tag}V{i}" for i in layer_ids]


class _Wrapper(HybridLinearKVPool):
    def __init__(self, inner):
        self.full_kv_pool = inner


def _kv_layer_ids(target_ids, draft_ids):
    """kv_data_ptrs order: K target, V target, K draft, V draft."""
    return list(target_ids) + list(target_ids) + list(draft_ids) + list(draft_ids)


class TestStagingDraftKvSlots(CustomTestCase):
    def test_draft_slots_follow_gather_order(self):
        target, draft = [87, 91], [92]
        k_buffers, v_buffers, slot_ids = build_staging_slot_metadata(
            kv_layer_ids=_kv_layer_ids(target, draft),
            num_draft_entries=2,
            kv_pool=_Pool("t", target),
            draft_kv_pool=_Pool("d", draft),
        )
        self.assertEqual(k_buffers, ["tK87", "tK91", "dK92"])
        self.assertEqual(v_buffers, ["tV87", "tV91", "dV92"])
        self.assertEqual(slot_ids, [87, 91, 92, 87, 91, 92])
        self.assertNotEqual(slot_ids, _kv_layer_ids(target, draft))

    def test_without_draft_matches_kv_layer_ids(self):
        # The two orders coincide with no draft pool, so every deployment that
        # predates draft KV must keep its exact slot labelling.
        target = [3, 7]
        _, _, slot_ids = build_staging_slot_metadata(
            kv_layer_ids=_kv_layer_ids(target, []),
            num_draft_entries=0,
            kv_pool=_Pool("t", target),
            draft_kv_pool=None,
        )
        self.assertEqual(slot_ids, _kv_layer_ids(target, []))

    def test_pp_stage_pairs_against_full_decode(self):
        # A prefill stage holds a slice of the layers while decode holds them
        # all, so the ids -- not the positions -- have to drive the pairing.
        src = build_staging_slot_metadata(
            kv_layer_ids=_kv_layer_ids([87, 91], [92]),
            num_draft_entries=2,
            kv_pool=_Pool("t", [87, 91]),
            draft_kv_pool=_Pool("d", [92]),
        )[2]
        decode_target = [3, 7, 11, 87, 91]
        dst = build_staging_slot_metadata(
            kv_layer_ids=_kv_layer_ids(decode_target, [92]),
            num_draft_entries=2,
            kv_pool=_Pool("t", decode_target),
            draft_kv_pool=_Pool("d", [92]),
        )[2]
        pairs = build_transfer_entry_pairs(src, dst, len(src), len(dst))
        self.assertEqual(len(pairs), len(src))
        for i, j in pairs:
            self.assertEqual(src[i], dst[j])
        self.assertEqual(len({j for _, j in pairs}), len(pairs))

    def test_hybrid_wrapper_pools_are_unwrapped(self):
        # A hybrid draft pool that is left wrapped looks exactly like a draft
        # pool with no buffers, which drops draft KV out of staging.
        target, draft = [87, 91], [92]
        k_buffers, _, slot_ids = build_staging_slot_metadata(
            kv_layer_ids=_kv_layer_ids(target, draft),
            num_draft_entries=2,
            kv_pool=_Wrapper(_Pool("t", target)),
            draft_kv_pool=_Wrapper(_Pool("d", draft)),
        )
        self.assertEqual(k_buffers, ["tK87", "tK91", "dK92"])
        self.assertEqual(slot_ids, [87, 91, 92, 87, 91, 92])

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
