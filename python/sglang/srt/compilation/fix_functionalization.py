# Adapted from https://github.com/vllm-project/vllm/blob/v0.10.0/vllm/compilation/fix_functionalization.py

import logging
import operator
from collections.abc import Iterable
from typing import Optional, Union

import torch
from torch._higher_order_ops.auto_functionalize import auto_functionalized

from sglang.srt.compilation.fx_utils import is_func
from sglang.srt.compilation.inductor_pass import SGLangInductorPass

logger = logging.getLogger(__name__)


class FixFunctionalizationPass(SGLangInductorPass):
    """
    This pass defunctionalizes certain nodes to avoid redundant tensor copies.
    After this pass, DCE (dead-code elimination) should never be run,
    as de-functionalized nodes may appear as dead code.

    To add new nodes to defunctionalize, add to the if-elif chain in __call__.
    """

    def __call__(self, graph: torch.fx.Graph):
        self.begin()
        self.dump_graph(graph, "before_fix_functionalization")
        
        print("\n" + "=" * 100)
        print("🔍 BEFORE FIX FUNCTIONALIZATION - FX Graph:")
        print("=" * 100)
        print(graph)
        print("\n📊 Graph Statistics:")
        print(f"  Total nodes: {len(list(graph.nodes))}")
        auto_func_nodes = [n for n in graph.nodes if is_func(n, auto_functionalized)]
        print(f"  Auto-functionalized nodes: {len(auto_func_nodes)}")
        if auto_func_nodes:
            print(f"  Auto-functionalized node details:")
            for i, node in enumerate(auto_func_nodes[:5], 1):  # 只显示前5个
                print(f"    {i}. {node.name}: {node.target}")
                if len(node.args) > 0:
                    print(f"       First arg: {node.args[0]}")
        print("=" * 100 + "\n")

        self.nodes_to_remove: list[torch.fx.Node] = []
        count = 0
        for node in graph.nodes:
            if not is_func(node, auto_functionalized):
                continue  # Avoid deep if-elif nesting
            
            # Handle sgl_kernel custom ops
            # Debug: print the op we're processing
            if len(node.args) > 0:
                op_func = node.args[0]
                logger.info(f"Processing auto_functionalized node: {op_func}")
            
            # IMPORTANT: Check more specific patterns first!
            # fused_add_rmsnorm must be checked before rmsnorm
            if self._is_sgl_kernel_op(node, "fused_add_rmsnorm"):
                logger.info("Calling _defunctionalize_fused_add_rmsnorm")
                self._defunctionalize_fused_add_rmsnorm(graph, node)
                count += 1
            elif self._is_sgl_kernel_op(node, "rmsnorm"):
                # This will match plain rmsnorm but not fused_add_rmsnorm (already handled above)
                logger.info("Calling _defunctionalize_rmsnorm")
                self._defunctionalize_rmsnorm(graph, node)
                count += 1
            elif self._is_sgl_kernel_op(node, "apply_rope") or self._is_sgl_kernel_op(node, "rope"):
                # Matches: apply_rope_with_cos_sin_cache_inplace, apply_rope_pos_ids_cos_sin_cache, etc.
                logger.info("Calling _defunctionalize_rope")
                self._defunctionalize_rope(graph, node)
                count += 1

        self.dump_graph(graph, "before_fix_functionalization_cleanup")

        # Remove the nodes all at once
        count_removed = len(self.nodes_to_remove)
        for node in self.nodes_to_remove:
            graph.erase_node(node)

        logger.debug(
            "De-functionalized %s nodes, removed %s nodes", count, count_removed
        )
        
        print("\n" + "=" * 100)
        print("✅ AFTER FIX FUNCTIONALIZATION - FX Graph:")
        print("=" * 100)
        print(graph)
        print("\n📊 Processing Summary:")
        print(f"  De-functionalized nodes: {count}")
        print(f"  Removed nodes: {count_removed}")
        auto_func_nodes_after = [n for n in graph.nodes if is_func(n, auto_functionalized)]
        print(f"  Remaining auto-functionalized nodes: {len(auto_func_nodes_after)}")
        
        sgl_kernel_nodes = []
        for node in graph.nodes:
            if hasattr(node, 'target') and 'sgl_kernel' in str(node.target):
                sgl_kernel_nodes.append(node)
        print(f"  Total sgl_kernel ops in graph: {len(sgl_kernel_nodes)}")
        if sgl_kernel_nodes:
            print(f"  sgl_kernel ops found:")
            for i, node in enumerate(sgl_kernel_nodes[:10], 1):
                print(f"    {i}. {node.name}: {node.target}")
        print("=" * 100 + "\n")
        
        self.dump_graph(graph, "after_fix_functionalization")
        self.end_and_log()

    def _is_sgl_kernel_op(self, node: torch.fx.Node, op_name: str) -> bool:
        """
        Check if the node is a sgl_kernel custom op with the given name.
        """
        if len(node.args) == 0:
            return False
        first_arg = node.args[0]
        if not hasattr(first_arg, "__name__"):
            return False
        # Check if it's torch.ops.sgl_kernel.{op_name}
        func_name = first_arg.__name__
        return (
            "torch.ops.sgl_kernel" in str(first_arg) 
            or op_name in func_name
        )

    def _defunctionalize_rmsnorm(self, graph: torch.fx.Graph, node: torch.fx.Node):
        """
        Defunctionalize rmsnorm op.
        rmsnorm(output, input, weight, eps, enable_pdl)
        auto_functionalized returns (output, input) where:
        - output: the result tensor (mutated)
        - input: the input tensor (unchanged)
        """
        # Debug: print what getitem indices are actually being used
        getitem_users = self.getitem_users(node)
        logger.info(f"RMSNorm node: {node}")
        logger.info(f"RMSNorm node.args[0]: {node.args[0] if len(node.args) > 0 else 'N/A'}")
        logger.info(f"RMSNorm node kwargs: {list(node.kwargs.keys())}")
        logger.info(f"RMSNorm getitem_users indices: {list(getitem_users.keys())}")
        
        # Map getitem indices to kwargs
        # Build mutated_args dynamically based on actual usage
        mutated_args = {}
        for idx in getitem_users.keys():
            if idx == 0:
                mutated_args[0] = "output"
            elif idx == 1:
                mutated_args[1] = "input"
            else:
                # Unexpected index, log warning
                logger.warning(f"Unexpected getitem index {idx} for rmsnorm node")
                logger.warning(f"Node: {node}")
                logger.warning(f"Node kwargs: {node.kwargs}")
                # For unexpected indices, we can't safely defunctionalize
                # Just keep the getitem as-is and don't add to mutated_args
                continue
        
        self.replace_users_with_mutated_args(node, mutated_args)
        self.insert_defunctionalized(graph, node)
        self._remove(node)

    def _defunctionalize_fused_add_rmsnorm(
        self, graph: torch.fx.Graph, node: torch.fx.Node
    ):
        """
        Defunctionalize fused_add_rmsnorm op.
        fused_add_rmsnorm(input, residual, weight, eps, enable_pdl)
        This op mutates input and residual in-place.
        auto_functionalized returns (?, input_mutated, residual_mutated) - indices 1, 2
        """
        # Debug: print what getitem indices are actually being used
        getitem_users = self.getitem_users(node)
        logger.info(f"fused_add_rmsnorm node kwargs: {list(node.kwargs.keys())}")
        logger.info(f"fused_add_rmsnorm getitem_users indices: {list(getitem_users.keys())}")
        
        # Build mutated_args dynamically based on actual usage
        mutated_args = {}
        for idx in getitem_users.keys():
            if idx == 0:
                # Index 0 might be unused or original input
                mutated_args[0] = "input"
            elif idx == 1:
                # Index 1 is mutated input
                mutated_args[1] = "input"
            elif idx == 2:
                # Index 2 is mutated residual
                mutated_args[2] = "residual"
            else:
                logger.warning(f"Unexpected getitem index {idx} for fused_add_rmsnorm")
                continue
        
        self.replace_users_with_mutated_args(node, mutated_args)
        self.insert_defunctionalized(graph, node)
        self._remove(node)

    def _defunctionalize_rope(self, graph: torch.fx.Graph, node: torch.fx.Node):
        """
        Defunctionalize apply_rope ops.
        Different rope ops have different signatures:
        - apply_rope_with_cos_sin_cache_inplace: mutates query and key
        - apply_rope_pos_ids_cos_sin_cache: has q, k, q_rope, k_rope args
        
        auto_functionalized returns the mutated tensors.
        """
        # Inspect the actual kwargs to determine which args are mutated
        getitem_users = self.getitem_users(node)
        
        if len(getitem_users) > 0:
            # Build mutated_args based on the kwargs
            mutated_args = {}
            
            # Common patterns for rope ops:
            # - apply_rope_pos_ids_cos_sin_cache: has q, k, q_rope, k_rope args
            #   Returns (?, q_rope_out, k_rope_out, ...) - indices 1, 2 are the mutated tensors
            # - apply_rope_with_cos_sin_cache_inplace: mutates query and key
            # Map getitem indices to the appropriate kwargs
            for idx in getitem_users.keys():
                # Try to find the corresponding kwarg based on index and available kwargs
                if idx == 0:
                    # Index 0 could be q, query, or first output
                    if "q" in node.kwargs:
                        mutated_args[0] = "q"
                    elif "query" in node.kwargs:
                        mutated_args[0] = "query"
                    elif "q_rope" in node.kwargs:
                        mutated_args[0] = "q_rope"
                elif idx == 1:
                    # Index 1 could be k, key, q_rope, or second output
                    if "q_rope" in node.kwargs:
                        mutated_args[1] = "q_rope"
                    elif "k" in node.kwargs:
                        mutated_args[1] = "k"
                    elif "key" in node.kwargs:
                        mutated_args[1] = "key"
                elif idx == 2:
                    # Index 2 is typically k_rope
                    if "k_rope" in node.kwargs:
                        mutated_args[2] = "k_rope"
                    elif "k" in node.kwargs:
                        mutated_args[2] = "k"
                else:
                    # For higher indices, log and skip
                    logger.warning(f"Unknown rope arg at index {idx}, node kwargs: {node.kwargs.keys()}")
                    continue
            
            if mutated_args:
                self.replace_users_with_mutated_args(node, mutated_args)
        
        self.insert_defunctionalized(graph, node)
        self._remove(node)

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
        for idx, user in self.getitem_users(node).items():
            # Only process indices that are defined in mutated_args
            if idx not in mutated_args:
                logger.warning(f"Getitem index {idx} not in mutated_args, skipping user {user}")
                continue
            
            arg = mutated_args[idx]
            arg = node.kwargs[arg] if isinstance(arg, str) else arg
            user.replace_all_uses_with(arg)
            self._remove(user)

    def getitem_users(self, node: torch.fx.Node) -> dict[int, torch.fx.Node]:
        """
        Returns the operator.getitem users of the auto-functionalized node,
        indexed by the index they are getting.
        """
        users = {}
        for user in node.users:
            if is_func(user, operator.getitem):
                idx = user.args[1]
                users[idx] = user
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
