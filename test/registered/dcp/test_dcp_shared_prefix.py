"""CPU unit test for reading prefix sharing off the radix match.

[Test Category] Correctness
[Test Target] layers/dcp/layout.py::dcp_shared_prefix

The extend gather sends each request's prefix separately, so a batch on one
cached document sends it once per request. Deduplicating it needs to know what
is shared, and the cheap answer is the radix tree: a request's prefix is its
root-to-``last_node`` path, so sharing is the paths' common ancestors -- no
device read, no index comparison.

Three claims, and the first is the one that decides the design:

    1. ``walked`` reproduces each request's TREE-OWNED prefix, which is what a
       served run checks against the scheduler's ``cache_protected_len``;
    2. ``union_rows`` counts a shared node once and an unshared one every time,
       i.e. the rows a deduplicated gather would send for the shared part;
    3. a chunked request's partial page is outside both -- it lives in
       ``prefix_indices`` but not in the tree, so a dedup must still send it
       per request, and comparing the walk against ``len(prefix_indices)``
       instead would call every chunked request a failure;
    4. it reads BOTH node shapes. The default cache is UnifiedRadixCache, whose
       UnifiedTreeNode keeps the Full KV under ``component_data`` and has no
       ``.value`` at all -- a walk that reads only ``.value`` reports every
       served batch as empty.

Usage:
    python -m pytest test_dcp_shared_prefix.py -v
    python test_dcp_shared_prefix.py
"""

import unittest

from sglang.srt.layers.dcp.layout import dcp_shared_prefix
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class FakeNode:
    """RadixCache's TreeNode shape: a plain ``value``, only its length read."""

    def __init__(self, rows, parent=None):
        self.value = [0] * rows
        self.parent = parent


class FakeComponentData:
    def __init__(self, value):
        self.value = value


class FakeUnifiedNode:
    """UnifiedTreeNode's shape: the Full KV sits in ``component_data`` and there
    is no ``.value``. BASE_COMPONENT_TYPE is ComponentType.FULL == 0."""

    def __init__(self, rows, parent=None):
        self.component_data = [FakeComponentData(None if rows is None else [0] * rows)]
        self.parent = parent


def chain(parent, *lengths, cls=FakeNode):
    """Extend a path by one node per length, returning the last."""
    for rows in lengths:
        parent = cls(rows, parent)
    return parent


ROOT = None  # the real tree's root has value None, which stops the walk


class TestDcpSharedPrefix(unittest.TestCase):
    def test_one_request_walks_its_own_prefix(self):
        node = chain(ROOT, 128, 128, 64)
        got = dcp_shared_prefix([node], [320])
        self.assertEqual(got.walked, [320])
        self.assertEqual(got.union_rows, 320)

    def test_a_shared_document_is_counted_once(self):
        # Three requests, one cached prefix, different tails -- the serving
        # target. The gather sends 3x; dedup sends 1x plus the tails.
        base = chain(ROOT, 512, 512)
        reqs = [chain(base, 64), chain(base, 32), chain(base, 16)]
        got = dcp_shared_prefix(reqs, [1088, 1056, 1040])
        self.assertEqual(got.walked, [1088, 1056, 1040])
        self.assertEqual(got.union_rows, 1024 + 64 + 32 + 16)

    def test_disjoint_prefixes_share_nothing(self):
        reqs = [chain(ROOT, 256), chain(ROOT, 256)]
        got = dcp_shared_prefix(reqs, [256, 256])
        self.assertEqual(got.walked, [256, 256])
        self.assertEqual(got.union_rows, 512)
        # Nothing to save, so the lever is worth nothing on this batch.
        self.assertEqual(sum(got.walked), got.union_rows)

    def test_one_request_extending_another(self):
        # A continued conversation: the shorter prefix is a strict ancestor,
        # so the union is just the longer path.
        short = chain(ROOT, 512)
        long = chain(short, 256)
        got = dcp_shared_prefix([short, long], [512, 768])
        self.assertEqual(got.walked, [512, 768])
        self.assertEqual(got.union_rows, 768)

    def test_the_same_node_object_twice(self):
        # Two requests that matched identically share the node itself, not
        # merely an equal length; identity is what the walk dedups on.
        node = chain(ROOT, 300)
        got = dcp_shared_prefix([node, node], [300, 300])
        self.assertEqual(got.walked, [300, 300])
        self.assertEqual(got.union_rows, 300)

    def test_a_request_with_no_match(self):
        # Decode-only or a cold request: no node, no prefix, and it must not
        # raise -- the probe runs on every extend forward.
        got = dcp_shared_prefix([None, chain(ROOT, 128)], [0, 128])
        self.assertEqual(got.walked, [0, 128])
        self.assertEqual(got.union_rows, 128)

    def test_a_chunked_request_keeps_a_private_partial_page(self):
        # checkpoint leaves a partial page in prefix_indices that is
        # not in the tree. The walk must match cache_protected_len, NOT
        # len(prefix_indices) -- under page_size > 1 that is every chunked
        # request, which is most of a 1M prefill.
        base = chain(ROOT, 512)
        reqs = [chain(base, 256), chain(base, 256)]
        got = dcp_shared_prefix(reqs, [768, 768])
        self.assertEqual(got.walked, got.protected)
        self.assertEqual(got.union_rows, 512 + 256 + 256)
        # The gather still sends the two private pages on top of the union.
        prefix_lens = [768 + 37, 768 + 91]
        private = sum(prefix_lens) - sum(got.protected)
        self.assertEqual(private, 37 + 91)

    def test_the_unified_node_shape_the_live_cache_builds(self):
        # UnifiedRadixCache is the fall-through in mem_cache/registry.py and
        # nothing passes cache_class, so this is the shape a served run has.
        # Reading .value here yields nothing, which is a silent zero rather
        # than an error -- the probe would print saved=0% on every batch.
        base = chain(ROOT, 512, 512, cls=FakeUnifiedNode)
        reqs = [
            chain(base, 64, cls=FakeUnifiedNode),
            chain(base, 32, cls=FakeUnifiedNode),
        ]
        got = dcp_shared_prefix(reqs, [1088, 1056])
        self.assertEqual(got.walked, [1088, 1056])
        self.assertEqual(got.union_rows, 1024 + 64 + 32)

    def test_a_unified_root_holds_an_empty_value(self):
        # UnifiedTreeNode's root carries value=[], not None, so the walk cannot
        # use "value is None" as its stop condition; it stops at parent None.
        root = FakeUnifiedNode(0)
        got = dcp_shared_prefix([chain(root, 256, cls=FakeUnifiedNode)], [256])
        self.assertEqual(got.walked, [256])
        self.assertEqual(got.union_rows, 256)

    def test_a_value_less_node_is_stepped_over_not_stopped_at(self):
        # Under HiCache the match skips an evicted-but-backuped node and keeps
        # walking, so device_indices has a hole. Stopping there instead would
        # under-count every ancestor above it.
        base = chain(ROOT, 256, cls=FakeUnifiedNode)
        evicted = FakeUnifiedNode(None, base)
        leaf = chain(evicted, 64, cls=FakeUnifiedNode)
        got = dcp_shared_prefix([leaf], [320])
        self.assertEqual(got.walked, [320])
        self.assertEqual(got.union_rows, 320)

    def test_an_empty_batch(self):
        got = dcp_shared_prefix([], [])
        self.assertEqual(got.walked, [])
        self.assertEqual(got.union_rows, 0)


if __name__ == "__main__":
    unittest.main()
