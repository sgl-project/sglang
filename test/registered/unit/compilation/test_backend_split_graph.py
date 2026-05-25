"""Unit tests for PCG FX split graph manipulation."""

import unittest

import torch
import torch.fx as fx

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=1, stage="base-b", runner_config="1-gpu-small")

if not torch.cuda.is_available():
    raise unittest.SkipTest("backend.split_graph currently imports CUDA-only PCG code")

from sglang.srt.compilation.backend import split_graph


def _make_graph(nodes):
    graph = fx.Graph()
    env = {}
    for name, op, target, args in nodes:
        if op == "placeholder":
            env[name] = graph.placeholder(name)
        elif op == "call_function":
            env[name] = graph.call_function(target, tuple(env[arg] for arg in args))
        elif op == "output":
            graph.output(env[args[0]])
        else:
            raise AssertionError(f"Unsupported node op: {op}")
    return fx.GraphModule({}, graph)


def _call_targets(graph_module):
    return [
        node.target
        for node in graph_module.graph.nodes
        if node.op == "call_function"
    ]


def _split_summary(items):
    return [(item.graph_id, item.is_splitting_graph) for item in items]


class TestSplitGraphPolicy(CustomTestCase):
    def test_two_adjacent_split_ops_share_one_eager_submodule(self):
        gm = _make_graph(
            [
                ("x", "placeholder", None, ()),
                ("neg", "call_function", torch.neg, ("x",)),
                ("relu", "call_function", torch.relu, ("neg",)),
                ("sigmoid", "call_function", torch.sigmoid, ("relu",)),
                ("out", "output", None, ("sigmoid",)),
            ]
        )

        _, items = split_graph(gm, [str(torch.neg), str(torch.relu)])

        self.assertEqual(_split_summary(items), [(1, True), (2, False)])
        self.assertEqual(_call_targets(items[0].graph), [torch.neg, torch.relu])
        self.assertEqual(_call_targets(items[1].graph), [torch.sigmoid])

    def test_three_adjacent_split_ops_share_one_eager_submodule(self):
        gm = _make_graph(
            [
                ("x", "placeholder", None, ()),
                ("neg", "call_function", torch.neg, ("x",)),
                ("relu", "call_function", torch.relu, ("neg",)),
                ("sigmoid", "call_function", torch.sigmoid, ("relu",)),
                ("abs", "call_function", torch.abs, ("sigmoid",)),
                ("out", "output", None, ("abs",)),
            ]
        )

        _, items = split_graph(
            gm,
            [str(torch.neg), str(torch.relu), str(torch.sigmoid), str(torch.abs)],
        )

        self.assertEqual(_split_summary(items), [(1, True)])
        self.assertEqual(
            _call_targets(items[0].graph),
            [torch.neg, torch.relu, torch.sigmoid, torch.abs],
        )

    def test_non_split_fx_node_breaks_split_op_adjacency(self):
        gm = _make_graph(
            [
                ("x", "placeholder", None, ()),
                ("neg", "call_function", torch.neg, ("x",)),
                ("sigmoid", "call_function", torch.sigmoid, ("neg",)),
                ("relu", "call_function", torch.relu, ("sigmoid",)),
                ("out", "output", None, ("relu",)),
            ]
        )

        _, items = split_graph(gm, [str(torch.neg), str(torch.relu)])

        self.assertEqual(_split_summary(items), [(1, True), (2, False), (3, True)])
        self.assertEqual(_call_targets(items[0].graph), [torch.neg])
        self.assertEqual(_call_targets(items[1].graph), [torch.sigmoid])
        self.assertEqual(_call_targets(items[2].graph), [torch.relu])

    def test_leading_split_op_keeps_following_compiled_submodule_separate(self):
        gm = _make_graph(
            [
                ("x", "placeholder", None, ()),
                ("neg", "call_function", torch.neg, ("x",)),
                ("sigmoid", "call_function", torch.sigmoid, ("neg",)),
                ("out", "output", None, ("sigmoid",)),
            ]
        )

        _, items = split_graph(gm, [str(torch.neg)])

        self.assertEqual(_split_summary(items), [(1, True), (2, False)])
        self.assertEqual(_call_targets(items[0].graph), [torch.neg])
        self.assertEqual(_call_targets(items[1].graph), [torch.sigmoid])

    def test_trailing_split_op_keeps_previous_compiled_submodule_separate(self):
        gm = _make_graph(
            [
                ("x", "placeholder", None, ()),
                ("sigmoid", "call_function", torch.sigmoid, ("x",)),
                ("relu", "call_function", torch.relu, ("sigmoid",)),
                ("out", "output", None, ("relu",)),
            ]
        )

        _, items = split_graph(gm, [str(torch.relu)])

        self.assertEqual(_split_summary(items), [(0, False), (1, True)])
        self.assertEqual(_call_targets(items[0].graph), [torch.sigmoid])
        self.assertEqual(_call_targets(items[1].graph), [torch.relu])

    def test_placeholder_resets_split_op_adjacency(self):
        gm = _make_graph(
            [
                ("x", "placeholder", None, ()),
                ("neg", "call_function", torch.neg, ("x",)),
                ("y", "placeholder", None, ()),
                ("relu", "call_function", torch.relu, ("y",)),
                ("add", "call_function", torch.add, ("neg", "relu")),
                ("out", "output", None, ("add",)),
            ]
        )

        _, items = split_graph(gm, [str(torch.neg), str(torch.relu)])

        self.assertEqual(_split_summary(items), [(1, True), (3, True), (4, False)])
        self.assertEqual(_call_targets(items[0].graph), [torch.neg])
        self.assertEqual(_call_targets(items[1].graph), [torch.relu])
        self.assertEqual(_call_targets(items[2].graph), [torch.add])


if __name__ == "__main__":
    unittest.main()
