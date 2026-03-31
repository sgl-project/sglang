# Adapted from https://github.com/vllm-project/vllm/blob/v0.10.0/vllm/compilation/fix_functionalization.py

import logging
import operator
from collections import defaultdict
from collections.abc import Iterable
from typing import Optional, Union

import torch
from torch._higher_order_ops.auto_functionalize import auto_functionalized

from sglang.srt.compilation.fx_utils import is_func
from sglang.srt.compilation.inductor_pass import SGLangInductorPass

logger = logging.getLogger(__name__)


def _get_op(namespace: str, op_name: str) -> Optional[torch.ops.OpOverload]:
    """Return ``torch.ops.<namespace>.<op_name>.default``, or None."""
    ns = getattr(torch.ops, namespace, None)
    if ns is None:
        return None
    op = getattr(ns, op_name, None)
    if op is None:
        return None
    return getattr(op, "default", None)


class FixFunctionalizationPass(SGLangInductorPass):
    """
    This pass defunctionalizes certain nodes to avoid redundant tensor copies.
    After this pass, DCE (dead-code elimination) should never be run,
    as de-functionalized nodes may appear as dead code.

    To add new ops, add entries to ``_build_op_table``.
    """

    def _build_op_table(self):
        """Build a dispatch table mapping op targets to handler functions."""
        table = {}

        # Activation ops — 'out' kwarg conflicts with inductor lowering.
        for name in ("silu_and_mul", "gelu_tanh_and_mul", "gelu_and_mul"):
            op = _get_op("sgl_kernel", name)
            if op is not None:
                table[op] = lambda g, n: self.defunctionalize(
                    g, n, {1: "out"}, args=("out", "input")
                )

        # Normalization ops
        for name in ("rmsnorm", "gemma_rmsnorm"):
            op = _get_op("sgl_kernel", name)
            if op is not None:
                table[op] = lambda g, n: self.defunctionalize(g, n, {1: "output"})

        for name in ("fused_add_rmsnorm", "gemma_fused_add_rmsnorm"):
            op = _get_op("sgl_kernel", name)
            if op is not None:
                table[op] = lambda g, n: self.defunctionalize(
                    g, n, {1: "input", 2: "residual"}
                )

        # Rotary embedding ops
        op = _get_op("sgl_kernel", "rotary_embedding")
        if op is not None:
            table[op] = lambda g, n: self.defunctionalize(g, n, {1: "query", 2: "key"})

        op = _get_op("sgl_kernel", "apply_rope_pos_ids_cos_sin_cache")
        if op is not None:
            table[op] = self._handle_apply_rope_pos_ids

        logger.debug("FixFunctionalizationPass: %d ops registered", len(table))
        return table

    def _handle_apply_rope_pos_ids(self, graph, node):
        """Handle optional mutated args (k_buffer, v_buffer)."""
        mutated_args = {1: "q_rope", 2: "k_rope"}
        getitem_map = self.getitem_users(node)
        if 3 in getitem_map:
            mutated_args[3] = "k_buffer"
        if 4 in getitem_map:
            mutated_args[4] = "v_buffer"
        self.defunctionalize(graph, node, mutated_args)

    def __call__(self, graph: torch.fx.Graph):
        self.begin()
        self.dump_graph(graph, "before_fix_functionalization")

        op_table = self._build_op_table()

        self.nodes_to_remove: list[torch.fx.Node] = []
        count = 0
        for node in graph.nodes:
            if not is_func(node, auto_functionalized):
                continue

            handler = op_table.get(node.args[0])
            if handler is not None:
                handler(graph, node)
                count += 1

        self.dump_graph(graph, "before_fix_functionalization_cleanup")

        # Remove the nodes all at once
        count_removed = len(self.nodes_to_remove)
        for node in self.nodes_to_remove:
            graph.erase_node(node)

        logger.debug(
            "De-functionalized %s nodes, removed %s nodes", count, count_removed
        )
        self.dump_graph(graph, "after_fix_functionalization")
        self.end_and_log()

    def _remove(self, node_or_nodes: Union[torch.fx.Node, Iterable[torch.fx.Node]]):
        """
        Stage a node (or nodes) for removal at the end of the pass.
        """
        if isinstance(node_or_nodes, torch.fx.Node):
            self.nodes_to_remove.append(node_or_nodes)
        else:
            self.nodes_to_remove.extend(node_or_nodes)

    def defunctionalize(
        self,
        graph: torch.fx.Graph,
        node: torch.fx.Node,
        mutated_args: dict[int, Union[torch.fx.Node, str]],
        args: Optional[tuple[Union[torch.fx.Node, str], ...]] = None,
    ):
        """
        De-functionalize a node by replacing it with a call to the original.
        It also replaces the getitem users with the mutated arguments.
        See replace_users_with_mutated_args and insert_defunctionalized.
        """
        self.replace_users_with_mutated_args(node, mutated_args)
        self.insert_defunctionalized(graph, node, args=args)
        self._remove(node)

    def replace_users_with_mutated_args(
        self, node: torch.fx.Node, mutated_args: dict[int, Union[torch.fx.Node, str]]
    ):
        """
        Replace all getitem users of the auto-functionalized node with the
        mutated arguments.
        :param node: The auto-functionalized node
        :param mutated_args: The mutated arguments, indexed by getitem index.
        If the value of an arg is a string, `node.kwargs[arg]` is used.
        """
        for idx, users in self.getitem_users(node).items():
            if idx not in mutated_args:
                continue
            arg = mutated_args[idx]
            arg = node.kwargs[arg] if isinstance(arg, str) else arg
            for user in users:
                user.replace_all_uses_with(arg)
                self._remove(user)

    def getitem_users(self, node: torch.fx.Node) -> dict[int, list[torch.fx.Node]]:
        """
        Returns the operator.getitem users of the auto-functionalized node,
        indexed by the index they are getting. Multiple getitem nodes for the
        same index are collected into a list.
        """
        users: dict[int, list[torch.fx.Node]] = defaultdict(list)
        for user in node.users:
            if is_func(user, operator.getitem):
                users[user.args[1]].append(user)
        return users

    def insert_defunctionalized(
        self,
        graph: torch.fx.Graph,
        node: torch.fx.Node,
        args: Optional[tuple[Union[torch.fx.Node, str], ...]] = None,
    ):
        """
        Insert a new defunctionalized node into the graph before node.
        If one of the kwargs is 'out', provide args directly,
        as node.kwargs cannot be used.
        See https://github.com/pytorch/pytorch/blob/a00faf440888ffb724bad413f329a49e2b6388e7/torch/_inductor/lowering.py#L351

        :param graph: Graph to insert the defunctionalized node into
        :param node: The auto-functionalized node to defunctionalize
        :param args: If we cannot use kwargs, specify args directly.
        If an arg is a string, `node.kwargs[arg]` is used.
        """  # noqa: E501
        assert is_func(
            node, auto_functionalized
        ), f"node must be auto-functionalized, is {node} instead"

        # Create a new call to the original function
        with graph.inserting_before(node):
            function = node.args[0]
            if args is None:
                graph.call_function(function, kwargs=node.kwargs)
            else:
                # Args passed as strings refer to items in node.kwargs
                args = tuple(
                    node.kwargs[arg] if isinstance(arg, str) else arg for arg in args
                )
                graph.call_function(function, args=args)
