"""Unit tests for FixFunctionalizationPass helpers in srt/compilation/fix_functionalization.py."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import operator
import unittest

import torch.fx as fx
from torch._higher_order_ops.auto_functionalize import auto_functionalized

from sglang.srt.compilation.fix_functionalization import FixFunctionalizationPass
from sglang.test.test_utils import CustomTestCase


def _make_pass() -> FixFunctionalizationPass:
    """Return a FixFunctionalizationPass with nodes_to_remove initialised."""
    p = FixFunctionalizationPass()
    p.nodes_to_remove = []
    return p


# ---------------------------------------------------------------------------
# getitem_users
# ---------------------------------------------------------------------------


class TestGetitemUsers(CustomTestCase):
    def test_returns_empty_dict_when_no_getitem_users(self):
        g = fx.Graph()
        a = g.placeholder("a")
        parent = g.call_function(operator.add, (a, a))
        # output uses parent directly, no getitem
        g.output(parent)
        fxp = _make_pass()
        result = fxp.getitem_users(parent)
        self.assertEqual(result, {})

    def test_returns_correct_mapping_for_single_getitem_user(self):
        g = fx.Graph()
        a = g.placeholder("a")
        parent = g.call_function(operator.add, (a, a))
        item0 = g.call_function(operator.getitem, (parent, 0))
        g.output(item0)
        fxp = _make_pass()
        result = fxp.getitem_users(parent)
        self.assertEqual(len(result), 1)
        self.assertIs(result[0], item0)

    def test_returns_correct_mapping_for_multiple_getitem_users(self):
        g = fx.Graph()
        a = g.placeholder("a")
        parent = g.call_function(operator.add, (a, a))
        item0 = g.call_function(operator.getitem, (parent, 0))
        item2 = g.call_function(operator.getitem, (parent, 2))
        g.output((item0, item2))
        fxp = _make_pass()
        result = fxp.getitem_users(parent)
        self.assertEqual(len(result), 2)
        self.assertIs(result[0], item0)
        self.assertIs(result[2], item2)

    def test_non_getitem_user_excluded(self):
        g = fx.Graph()
        a = g.placeholder("a")
        parent = g.call_function(operator.add, (a, a))
        # A user that is NOT operator.getitem
        other_user = g.call_function(operator.neg, (parent,))
        g.output(other_user)
        fxp = _make_pass()
        result = fxp.getitem_users(parent)
        self.assertEqual(result, {})


# ---------------------------------------------------------------------------
# insert_defunctionalized
# ---------------------------------------------------------------------------


class TestInsertDefunctionalized(CustomTestCase):
    def _make_auto_fn_graph(self):
        """Graph: placeholder x, auto_functionalized(operator.add, self=x, other=x), output."""
        g = fx.Graph()
        x = g.placeholder("x")
        # args[0] is the function insert_defunctionalized will call
        node = g.call_function(
            auto_functionalized,
            args=(operator.add,),
            kwargs={"self": x, "other": x},
        )
        g.output(node)
        return g, x, node

    def test_inserts_one_extra_node(self):
        g, x, af_node = self._make_auto_fn_graph()
        count_before = sum(1 for _ in g.nodes)
        fxp = _make_pass()
        fxp.insert_defunctionalized(g, af_node)
        count_after = sum(1 for _ in g.nodes)
        self.assertEqual(count_after, count_before + 1)

    def test_new_node_inserted_immediately_before_auto_fn_node(self):
        g, x, af_node = self._make_auto_fn_graph()
        fxp = _make_pass()
        fxp.insert_defunctionalized(g, af_node)
        nodes = list(g.nodes)
        idx = nodes.index(af_node)
        self.assertGreater(idx, 0)
        new_node = nodes[idx - 1]
        # The new node's target is args[0] of the auto_functionalized node
        self.assertIs(new_node.target, operator.add)

    def test_new_node_uses_kwargs_from_original(self):
        g, x, af_node = self._make_auto_fn_graph()
        fxp = _make_pass()
        fxp.insert_defunctionalized(g, af_node)
        nodes = list(g.nodes)
        idx = nodes.index(af_node)
        new_node = nodes[idx - 1]
        # kwargs should be propagated from af_node.kwargs
        self.assertIn("self", new_node.kwargs)
        self.assertIn("other", new_node.kwargs)
        self.assertIs(new_node.kwargs["self"], x)

    def test_raises_if_node_is_not_auto_functionalized(self):
        g = fx.Graph()
        a = g.placeholder("a")
        plain_node = g.call_function(operator.add, (a, a))
        g.output(plain_node)
        fxp = _make_pass()
        with self.assertRaises(AssertionError):
            fxp.insert_defunctionalized(g, plain_node)


# ---------------------------------------------------------------------------
# replace_users_with_mutated_args
# ---------------------------------------------------------------------------


class TestReplaceUsersWithMutatedArgs(CustomTestCase):
    def _make_graph_with_getitem(self):
        """placeholder a, placeholder b, add(a,b), getitem(add, 0), output(getitem)."""
        g = fx.Graph()
        a = g.placeholder("a")
        b = g.placeholder("b")
        parent = g.call_function(operator.add, (a, b))
        item0 = g.call_function(operator.getitem, (parent, 0))
        g.output(item0)
        out_node = next(n for n in g.nodes if n.op == "output")
        return g, a, b, parent, item0, out_node

    def test_getitem_user_staged_for_removal(self):
        _, a, b, parent, item0, _ = self._make_graph_with_getitem()
        fxp = _make_pass()
        fxp.replace_users_with_mutated_args(parent, {0: b})
        self.assertIn(item0, fxp.nodes_to_remove)

    def test_uses_of_getitem_replaced_with_arg(self):
        _, a, b, parent, item0, out_node = self._make_graph_with_getitem()
        # Before replacement: output uses item0
        self.assertIs(out_node.args[0], item0)
        fxp = _make_pass()
        fxp.replace_users_with_mutated_args(parent, {0: b})
        # After replacement: output uses b
        self.assertIs(out_node.args[0], b)

    def test_no_getitem_users_means_nothing_staged(self):
        g = fx.Graph()
        a = g.placeholder("a")
        parent = g.call_function(operator.add, (a, a))
        g.output(parent)
        fxp = _make_pass()
        fxp.replace_users_with_mutated_args(parent, {0: a})
        self.assertEqual(fxp.nodes_to_remove, [])

    def test_multiple_getitem_users_replaced_independently(self):
        g = fx.Graph()
        a = g.placeholder("a")
        b = g.placeholder("b")
        c = g.placeholder("c")
        parent = g.call_function(operator.add, (a, b))
        item0 = g.call_function(operator.getitem, (parent, 0))
        item1 = g.call_function(operator.getitem, (parent, 1))
        g.output((item0, item1))
        fxp = _make_pass()
        fxp.replace_users_with_mutated_args(parent, {0: b, 1: c})
        self.assertIn(item0, fxp.nodes_to_remove)
        self.assertIn(item1, fxp.nodes_to_remove)


if __name__ == "__main__":
    unittest.main()
