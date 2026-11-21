"""CPU-only unit tests for recurrent-state checkpoint KV events.

With ``publish_recurrent_checkpoints`` on, the MAMBA component of the unified
radix cache publishes where a cached recurrent state sits, in addition to the
attention ``BlockStored`` / ``BlockRemoved`` events: a tagged ``BlockStored``
(``kv_cache_spec_kind="mamba"``) at the hash of the node's last page when a
state is cached, and a tagged ``BlockRemoved`` when the state is evicted while
the attention pages stay. Both tree-core backends are covered.
"""

import unittest
from unittest.mock import patch

from sglang.srt.disaggregation.kv_events import (
    RECURRENT_CHECKPOINT_GROUP_IDX,
    RECURRENT_CHECKPOINT_SPEC_KIND,
    BlockRemoved,
    BlockStored,
    StorageMedium,
)
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kits.unified_radix_cache.config import CacheConfig
from sglang.test.kits.unified_radix_cache.fixtures import build_fixture, match_prefix
from sglang.test.kits.unified_radix_cache.mixins import UnifiedRadixCacheFixtures
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

PAGE = 4
PREFIX = list(range(1, 2 * PAGE + 1))
EXTENDED = PREFIX + list(range(101, 101 + 2 * PAGE))


def _is_checkpoint(event) -> bool:
    return event.kv_cache_spec_kind == RECURRENT_CHECKPOINT_SPEC_KIND


def _split(events):
    """(attention events, checkpoint events), each in emission order."""
    attention = [e for e in events if not _is_checkpoint(e)]
    checkpoints = [e for e in events if _is_checkpoint(e)]
    return attention, checkpoints


class _MambaCheckpointEventsBase(UnifiedRadixCacheFixtures):
    tree_core_backend: str

    def _build(self, *, publish: bool, dynamo_format: bool = False):
        # Paged states need the extra-buffer strategy; the checkpoint grid is
        # one page so a state can sit at every node end.
        self.cfg = CacheConfig(
            page_size=PAGE,
            components=(ComponentType.FULL, ComponentType.MAMBA),
            enable_mamba_extra_buffer=True,
            kv_size=256,
            max_context_len=256,
            mamba_cache_size=8,
        )
        self._rid = 0
        with patch(
            "sglang.test.kits.unified_radix_cache.fixtures.get_device",
            return_value="cpu",
        ):
            cache, allocator, rtp = build_fixture(
                self.cfg,
                enable_kv_cache_events=True,
                dynamo_kv_event_format=dynamo_format,
                publish_recurrent_checkpoints=publish,
                tree_core_backend=self.tree_core_backend,
                mamba_cache_chunk_size=PAGE,
            )
        cache.take_events()
        return cache, allocator, rtp

    def _insert_state(self, cache, allocator, rtp, tokens, **key_fields):
        """Cache ``tokens`` with a recurrent state at their end; return the
        (attention, checkpoint) events this emitted."""
        self._insert(cache, allocator, rtp, tokens, **key_fields)
        return _split(cache.take_events())

    def _assert_checkpoint_shape(self, event, attention_events, parent_hash):
        self.assertIsInstance(event, BlockStored)
        self.assertEqual(event.group_idx, RECURRENT_CHECKPOINT_GROUP_IDX)
        self.assertEqual(event.kv_cache_spec_kind, RECURRENT_CHECKPOINT_SPEC_KIND)
        self.assertEqual(event.medium, StorageMedium.GPU)
        self.assertEqual(event.token_ids, [])
        self.assertEqual(event.block_size, PAGE)
        self.assertEqual(event.parent_block_hash, parent_hash)
        # The state is keyed by the last page the attention store announced.
        attention_hashes = [h for e in attention_events for h in e.block_hashes]
        self.assertEqual(event.block_hashes, attention_hashes[-1:])

    def test_state_store_tombstone_and_leaf_delete(self):
        cache, allocator, rtp = self._build(publish=True)

        attention, checkpoints = self._insert_state(cache, allocator, rtp, PREFIX)
        self.assertEqual(len(checkpoints), 1)
        self.assertTrue(all(e.group_idx is None for e in attention))
        self._assert_checkpoint_shape(checkpoints[0], attention, parent_hash=None)
        prefix_hash = checkpoints[0].block_hashes[0]

        attention, checkpoints = self._insert_state(cache, allocator, rtp, EXTENDED)
        self.assertEqual(len(checkpoints), 1)
        self._assert_checkpoint_shape(
            checkpoints[0], attention, parent_hash=prefix_hash
        )
        extended_hash = checkpoints[0].block_hashes[0]
        self.assertEqual(match_prefix(cache, EXTENDED).device_prefix_len, len(EXTENDED))

        # Tombstone: the LRU state (the prefix node, an internal node) goes,
        # its attention pages stay.
        cache.evict(EvictParams(mamba_num=1))
        attention, checkpoints = _split(cache.take_events())
        self.assertEqual(attention, [])
        self.assertEqual(
            checkpoints,
            [
                BlockRemoved(
                    block_hashes=[prefix_hash],
                    medium=StorageMedium.GPU,
                    group_idx=RECURRENT_CHECKPOINT_GROUP_IDX,
                    kv_cache_spec_kind=RECURRENT_CHECKPOINT_SPEC_KIND,
                )
            ],
        )
        # The engine itself can no longer resume at the prefix end.
        self.assertLess(match_prefix(cache, PREFIX).device_prefix_len, len(PREFIX))
        self.assertEqual(match_prefix(cache, EXTENDED).device_prefix_len, len(EXTENDED))

        # Deleting the leaf removes its attention pages and its state.
        cache.evict(EvictParams(num_tokens=len(EXTENDED) - len(PREFIX)))
        attention, checkpoints = _split(cache.take_events())
        self.assertTrue(attention)
        self.assertTrue(all(isinstance(e, BlockRemoved) for e in attention))
        self.assertEqual(
            [h for e in checkpoints for h in e.block_hashes], [extended_hash]
        )
        self.assertTrue(all(isinstance(e, BlockRemoved) for e in checkpoints))

    def test_flag_off_publishes_attention_only(self):
        cache_on, allocator_on, rtp_on = self._build(publish=True)
        cache_off, allocator_off, rtp_off = self._build(publish=False)
        for tokens in (PREFIX, EXTENDED):
            attention_on, checkpoints_on = self._insert_state(
                cache_on, allocator_on, rtp_on, tokens
            )
            attention_off, checkpoints_off = self._insert_state(
                cache_off, allocator_off, rtp_off, tokens
            )
            self.assertEqual(len(checkpoints_on), 1)
            self.assertEqual(checkpoints_off, [])
            self.assertEqual(attention_off, attention_on)
        cache_off.evict(EvictParams(mamba_num=1))
        self.assertEqual(cache_off.take_events(), [])

    def test_namespaced_checkpoint_matches_attention_hash(self):
        cache, allocator, rtp = self._build(publish=True, dynamo_format=True)
        attention, checkpoints = self._insert_state(
            cache, allocator, rtp, PREFIX, cache_salt="tenant-a"
        )
        self.assertEqual(len(checkpoints), 1)
        self._assert_checkpoint_shape(checkpoints[0], attention, parent_hash=None)
        self.assertEqual(checkpoints[0].cache_salt, "tenant-a")
        prefix_hash = checkpoints[0].block_hashes[0]

        attention, checkpoints = self._insert_state(
            cache, allocator, rtp, EXTENDED, cache_salt="tenant-a"
        )
        self.assertEqual(len(checkpoints), 1)
        self._assert_checkpoint_shape(
            checkpoints[0], attention, parent_hash=prefix_hash
        )

        # The prefix node is internal now: its state is tombstoned under the
        # same namespaced hash, its attention pages stay.
        cache.evict(EvictParams(mamba_num=1))
        attention, removed = _split(cache.take_events())
        self.assertEqual(attention, [])
        self.assertEqual([h for e in removed for h in e.block_hashes], [prefix_hash])


class TestMambaCheckpointKVEventsPython(_MambaCheckpointEventsBase, CustomTestCase):
    tree_core_backend = "python"


class TestMambaCheckpointKVEventsRust(_MambaCheckpointEventsBase, CustomTestCase):
    tree_core_backend = "rust"


if __name__ == "__main__":
    unittest.main()
