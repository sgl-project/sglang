"""Unit tests for FX-graph utility functions in srt/compilation/fx_utils.py."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import operator
import unittest

import torch.fx as fx

from sglang.srt.compilation.fx_utils import (
    find_getitem,
    find_getitem_maybe,
    find_specified_fn,
    find_specified_fn_maybe,
    get_only_user,
    is_func,
)
from sglang.test.test_utils import CustomTestCase

# ---------------------------------------------------------------------------
# Small helpers to build minimal FX graphs
# ---------------------------------------------------------------------------


def _graph_with_call(target, *extra_targets):
    """Return (graph, placeholder, call_node) for a call_function node."""
    g = fx.Graph()
    a = g.placeholder("a")
    node = g.call_function(target, (a, a))
    g.output(node)
    return g, a, node


def _graph_with_getitem_user(parent_target, idx):
    """Return (graph, placeholder, parent, getitem_node)."""
    g = fx.Graph()
    a = g.placeholder("a")
    parent = g.call_function(parent_target, (a, a))
    item = g.call_function(operator.getitem, (parent, idx))
    g.output(item)
    return g, a, parent, item


# ---------------------------------------------------------------------------
# is_func
# ---------------------------------------------------------------------------


class TestIsFunc(CustomTestCase):
    def test_returns_true_for_matching_call_function_node(self):
        _, _, node = _graph_with_call(operator.add)
        self.assertTrue(is_func(node, operator.add))

    def test_returns_false_for_wrong_target(self):
        _, _, node = _graph_with_call(operator.add)
        self.assertFalse(is_func(node, operator.mul))

    def test_returns_false_for_placeholder_op(self):
        _, placeholder, _ = _graph_with_call(operator.add)
        self.assertFalse(is_func(placeholder, operator.add))

    def test_returns_false_for_output_node(self):
        g = fx.Graph()
        a = g.placeholder("a")
        out = g.output(a)
        self.assertFalse(is_func(out, operator.add))

    def test_getitem_node_matches_operator_getitem(self):
        _, _, _, item = _graph_with_getitem_user(operator.add, 0)
        self.assertTrue(is_func(item, operator.getitem))


# ---------------------------------------------------------------------------
# find_specified_fn_maybe / find_specified_fn
# ---------------------------------------------------------------------------


class TestFindSpecifiedFn(CustomTestCase):
    def _make_nodes(self):
        g = fx.Graph()
        a = g.placeholder("a")
        add_node = g.call_function(operator.add, (a, a))
        g.output(add_node)
        return list(g.nodes), add_node

    def test_find_maybe_returns_node_when_present(self):
        nodes, add_node = self._make_nodes()
        result = find_specified_fn_maybe(nodes, operator.add)
        self.assertIs(result, add_node)

    def test_find_maybe_returns_none_when_absent(self):
        nodes, _ = self._make_nodes()
        result = find_specified_fn_maybe(nodes, operator.mul)
        self.assertIsNone(result)

    def test_find_maybe_returns_first_match(self):
        g = fx.Graph()
        a = g.placeholder("a")
        n1 = g.call_function(operator.add, (a, a))
        n2 = g.call_function(operator.add, (a, a))
        g.output((n1, n2))
        nodes = list(g.nodes)
        result = find_specified_fn_maybe(nodes, operator.add)
        self.assertIs(result, n1)

    def test_find_raises_when_absent(self):
        nodes, _ = self._make_nodes()
        with self.assertRaises(AssertionError):
            find_specified_fn(nodes, operator.mul)

    def test_find_returns_node_when_present(self):
        nodes, add_node = self._make_nodes()
        result = find_specified_fn(nodes, operator.add)
        self.assertIs(result, add_node)


# ---------------------------------------------------------------------------
# find_getitem_maybe / find_getitem
# ---------------------------------------------------------------------------


class TestFindGetitem(CustomTestCase):
    def test_find_getitem_maybe_returns_node_for_matching_index(self):
        _, _, parent, item0 = _graph_with_getitem_user(operator.add, 0)
        result = find_getitem_maybe(parent, 0)
        self.assertIs(result, item0)

    def test_find_getitem_maybe_returns_none_when_no_user(self):
        g = fx.Graph()
        a = g.placeholder("a")
        parent = g.call_function(operator.add, (a, a))
        g.output(parent)
        result = find_getitem_maybe(parent, 0)
        self.assertIsNone(result)

    def test_find_getitem_maybe_returns_none_for_wrong_index(self):
        _, _, parent, _ = _graph_with_getitem_user(operator.add, 0)
        result = find_getitem_maybe(parent, 5)
        self.assertIsNone(result)

    def test_find_getitem_returns_node_for_matching_index(self):
        _, _, parent, item0 = _graph_with_getitem_user(operator.add, 0)
        result = find_getitem(parent, 0)
        self.assertIs(result, item0)

    def test_find_getitem_raises_when_index_absent(self):
        _, _, parent, _ = _graph_with_getitem_user(operator.add, 0)
        with self.assertRaises(AssertionError):
            find_getitem(parent, 99)

    def test_find_getitem_with_multiple_users_returns_correct_one(self):
        g = fx.Graph()
        a = g.placeholder("a")
        parent = g.call_function(operator.add, (a, a))
        item0 = g.call_function(operator.getitem, (parent, 0))
        item2 = g.call_function(operator.getitem, (parent, 2))
        g.output((item0, item2))

        self.assertIs(find_getitem(parent, 0), item0)
        self.assertIs(find_getitem(parent, 2), item2)


# ---------------------------------------------------------------------------
# get_only_user
# ---------------------------------------------------------------------------


class TestGetOnlyUser(CustomTestCase):
    def test_returns_the_single_user(self):
        g = fx.Graph()
        a = g.placeholder("a")
        user = g.call_function(operator.neg, (a,))
        g.output(user)
        self.assertIs(get_only_user(a), user)

    def test_raises_with_zero_users(self):
        g = fx.Graph()
        a = g.placeholder("a")
        g.output(None)
        with self.assertRaises(AssertionError):
            get_only_user(a)

    def test_raises_with_two_users(self):
        g = fx.Graph()
        a = g.placeholder("a")
        u1 = g.call_function(operator.neg, (a,))
        u2 = g.call_function(operator.neg, (a,))
        g.output((u1, u2))
        # a has two distinct user nodes
        self.assertEqual(len(a.users), 2)
        with self.assertRaises(AssertionError):
            get_only_user(a)


if __name__ == "__main__":
    unittest.main()
