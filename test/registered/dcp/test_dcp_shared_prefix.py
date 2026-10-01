"""CPU unit test for reading prefix sharing off the radix match.

[Test Category] Correctness
[Test Target] layers/dcp/layout.py::dcp_shared_prefix

The extend gather sends each request's prefix separately, so a batch on one
cached document sends it once per request. Deduplicating it needs to know what
is shared, and the cheap answer is the radix tree: a request's prefix is its
root-to-``last_node`` path, so sharing is the paths' common ancestors -- no
device read, no index comparison.

Two claims, and the second is the one that decides the design:

    1. ``walked`` reproduces each request's prefix length, which is how a
       served run can check the walk against the scheduler's own lengths;
    2. ``union_rows`` counts a shared node once and an unshared one every time,
       i.e. it is the row count a deduplicated gather would send.

Usage:
    python -m pytest test_dcp_shared_prefix.py -v
    python test_dcp_shared_prefix.py
"""

import unittest

from sglang.srt.layers.dcp.layout import dcp_shared_prefix
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class FakeNode:
    """The two attributes the walk reads. The real TreeNode carries a device
    tensor in ``value``; only its length is read."""

    def __init__(self, rows, parent=None):
        self.value = [0] * rows
        self.parent = parent


def chain(parent, *lengths):
    """Extend a path by one node per length, returning the last."""
    for rows in lengths:
        parent = FakeNode(rows, parent)
    return parent


ROOT = None  # the real tree's root has value None, which stops the walk


class TestDcpSharedPrefix(unittest.TestCase):
    def test_one_request_walks_its_own_prefix(self):
        node = chain(ROOT, 128, 128, 64)
        got = dcp_shared_prefix([node])
        self.assertEqual(got.walked, [320])
        self.assertEqual(got.union_rows, 320)

    def test_a_shared_document_is_counted_once(self):
        # Three requests, one cached prefix, different tails -- the serving
        # target. The gather sends 3x; dedup sends 1x plus the tails.
        base = chain(ROOT, 512, 512)
        reqs = [chain(base, 64), chain(base, 32), chain(base, 16)]
        got = dcp_shared_prefix(reqs)
        self.assertEqual(got.walked, [1088, 1056, 1040])
        self.assertEqual(got.union_rows, 1024 + 64 + 32 + 16)

    def test_disjoint_prefixes_share_nothing(self):
        reqs = [chain(ROOT, 256), chain(ROOT, 256)]
        got = dcp_shared_prefix(reqs)
        self.assertEqual(got.walked, [256, 256])
        self.assertEqual(got.union_rows, 512)
        # Nothing to save, so the lever is worth nothing on this batch.
        self.assertEqual(sum(got.walked), got.union_rows)

    def test_one_request_extending_another(self):
        # A continued conversation: the shorter prefix is a strict ancestor,
        # so the union is just the longer path.
        short = chain(ROOT, 512)
        long = chain(short, 256)
        got = dcp_shared_prefix([short, long])
        self.assertEqual(got.walked, [512, 768])
        self.assertEqual(got.union_rows, 768)

    def test_the_same_node_object_twice(self):
        # Two requests that matched identically share the node itself, not
        # merely an equal length; identity is what the walk dedups on.
        node = chain(ROOT, 300)
        got = dcp_shared_prefix([node, node])
        self.assertEqual(got.walked, [300, 300])
        self.assertEqual(got.union_rows, 300)

    def test_a_request_with_no_match(self):
        # Decode-only or a cold request: no node, no prefix, and it must not
        # raise -- the probe runs on every extend forward.
        got = dcp_shared_prefix([None, chain(ROOT, 128)])
        self.assertEqual(got.walked, [0, 128])
        self.assertEqual(got.union_rows, 128)

    def test_an_empty_batch(self):
        got = dcp_shared_prefix([])
        self.assertEqual(got.walked, [])
        self.assertEqual(got.union_rows, 0)


if __name__ == "__main__":
    unittest.main()
