"""Child-slot bookkeeping of MambaRadixCache._split_node.

Regression for an eviction-time crash on Ling-3.0-Flash (hybrid KDA + radix cache,
page_size > 1): `AssertionError: parent does not have child key, ()` raised from
`_delete_tombstone_leaf` during `alloc_req_slots -> evict -> evict_mamba`.

A node is filed in `parent.children` under a slot computed at registration, but
removed with a slot recomputed from `node.key`. `_split_node` used to register the
new intermediate node under the *pre-split* key while giving it the *post-split*
(truncated) key, so the two agree only when `split_len >= page_size`. Below that the
node becomes unremovable; at `split_len == 0` its key is empty and `child_key()`
renders it as `()` -- the exact crash signature.
"""

import unittest
from array import array

from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache, TreeNode
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PAGE_SIZE = 64


def _bare_cache(page_size: int = PAGE_SIZE) -> MambaRadixCache:
    """A MambaRadixCache with only what the split/delete bookkeeping touches.

    __init__ builds GPU pools, so bypass it: these paths read `page_size` and the
    tree, nothing else.
    """
    cache = MambaRadixCache.__new__(MambaRadixCache)
    cache.page_size = page_size
    cache.disable = False
    cache.full_evictable_size_ = 0
    cache.mamba_evictable_size_ = 0
    cache.root_node = TreeNode()
    cache.root_node.key = RadixKey(array("q"), None)
    cache.root_node.value = None
    return cache


def _key(start: int, n: int) -> RadixKey:
    return RadixKey(array("q", range(start, start + n)), None)


class TestMambaRadixChildKey(unittest.TestCase):
    def _split_and_get(self, split_len: int):
        """Split a child at `split_len`; return (cache, new_node)."""
        cache = _bare_cache()
        child = TreeNode()
        child.key = _key(1000, 3 * PAGE_SIZE)
        child.parent = cache.root_node
        child.full_lock_ref = 0
        cache.root_node.children[child.key.child_key(PAGE_SIZE)] = child

        # _split_node touches the LRU lists and hash helpers; stub them out so the
        # test isolates the children-slot bookkeeping.
        cache.full_lru_list = _NoopLru()
        cache.mamba_lru_list = _NoopLru()
        child.value = _FakeValue(3 * PAGE_SIZE)
        child.hash_value = []
        child.event_hash_value = []

        new_node = cache._split_node(child.key, child, split_len)
        return cache, new_node

    def test_registered_slot_matches_node_key_when_page_aligned(self):
        """Healthy path: a page-aligned split stays consistent."""
        for split_len in (PAGE_SIZE, 2 * PAGE_SIZE):
            with self.subTest(split_len=split_len):
                cache, new_node = self._split_and_get(split_len)
                slot = next(
                    k for k, v in cache.root_node.children.items() if v is new_node
                )
                self.assertEqual(slot, new_node.key.child_key(PAGE_SIZE))

    def test_sub_page_split_stays_removable(self):
        """The bug: a sub-page split filed the node under a slot its key never derives.

        Pre-fix the registered slot came from the pre-split key, so this lookup --
        the one `_delete_tombstone_leaf` performs -- missed and the node could never
        be removed.
        """
        cache, new_node = self._split_and_get(PAGE_SIZE // 2)
        slot = next(k for k, v in cache.root_node.children.items() if v is new_node)
        self.assertEqual(
            slot,
            new_node.key.child_key(PAGE_SIZE),
            "node is filed under a slot its own key does not derive; "
            "_delete_tombstone_leaf would raise 'parent does not have child key'",
        )
        # And the removal itself must find it.
        popped = cache.root_node.children.pop(new_node.key.child_key(PAGE_SIZE), None)
        self.assertIs(popped, new_node)

    def test_zero_split_is_rejected(self):
        """split_len == 0 builds an empty-key node, whose slot renders as `()`.

        `child_key()` returns `()` for an empty key at page_size > 1 rather than
        raising, so without this guard the corruption is silent until eviction.
        """
        with self.assertRaises(AssertionError):
            self._split_and_get(0)

    def test_sanity_check_detects_a_drifted_slot(self):
        """The tree-wide invariant catches a mis-filed node directly."""
        cache = _bare_cache()
        child = TreeNode()
        child.key = _key(2000, 2 * PAGE_SIZE)
        child.parent = cache.root_node
        cache.root_node.children[("wrong", "slot")] = child

        with self.assertRaises(AssertionError):
            cache._sanity_check_child_keys()


class _NoopLru:
    def remove_node(self, node):
        pass

    def insert_mru(self, node):
        pass

    def reset_node_mru(self, node):
        pass


class _FakeValue:
    """Stands in for the kv-index tensor: only slicing and len are exercised."""

    def __init__(self, n: int, offset: int = 0):
        self.n = n
        self.offset = offset

    def __len__(self):
        return self.n

    def __getitem__(self, s):
        start = s.start or 0
        stop = self.n if s.stop is None else min(s.stop, self.n)
        return _FakeValue(max(0, stop - start), self.offset + start)

    def clone(self):
        return _FakeValue(self.n, self.offset)


if __name__ == "__main__":
    unittest.main()
