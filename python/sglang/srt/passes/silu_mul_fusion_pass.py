import logging
import operator
import torch
import torch.fx as fx
from torch.fx.node import Node
import sgl_kernel

from .base_pass import BaseFXPass

logger = logging.getLogger(__name__)


class SiLUMulFusionPass(BaseFXPass):
    def __init__(self):
        super().__init__()
    
    def is_applicable(self, graph: fx.Graph) -> bool:
        for node in graph.nodes:
            if self._is_silu_node(node):
                return True
        return False
    
    def apply(self, graph: fx.Graph) -> bool:
        modified = False
        
        for node in list(graph.nodes):
            if self._is_silu_node(node):
                for user in list(node.users):
                    if self._is_mul_node(user) and self._can_fuse_silu_mul(node, user):
                        self._create_fused_silu_mul(graph, node, user)
                        modified = True
                        logger.debug(f"Fused SiLU * x: {node.name} -> {user.name}")
        return modified
    
    def _is_silu_node(self, node: Node) -> bool:
        if node.op == 'call_function':
            return node.target in [torch.nn.functional.silu, torch.ops.aten.silu.default]
        elif node.op == 'call_module':
            return 'silu' in str(node.target).lower()
        return False
    
    def _is_mul_node(self, node: Node) -> bool:
        if node.op == 'call_function':
            mul_targets = [
                torch.mul,
                operator.mul,
            ]
            return node.target in mul_targets
        elif node.op == 'call_method':
            return 'mul' in str(node.target).lower()
        return False
    
    def _can_fuse_silu_mul(self, silu_node: Node, mul_node: Node) -> bool:
        return silu_node in mul_node.args
    
    def _create_fused_silu_mul(self, graph: fx.Graph, silu_node: Node, mul_node: Node):
        silu_input = silu_node.args[0]
        
        other_arg = None
        for arg in mul_node.args:
            if arg != silu_node:
                other_arg = arg
                break
        
        if (hasattr(silu_input, 'op') and hasattr(other_arg, 'op') and
            silu_input.op == 'call_function' and other_arg.op == 'call_function' and
            silu_input.target == operator.getitem and other_arg.target == operator.getitem and
            silu_input.args[0] == other_arg.args[0]):
            
            base_tensor = silu_input.args[0]
            
            with graph.inserting_after(mul_node):
                fused_node = graph.call_function(
                    self._get_fused_silu_mul_op(),
                    args=(base_tensor,),
                    kwargs={}
                )
        
        mul_node.replace_all_uses_with(fused_node)
        
        graph.erase_node(mul_node)
        if len(silu_node.users) == 0:
            graph.erase_node(silu_node)
    
    def _get_fused_silu_mul_op(self):
        return sgl_kernel.silu_and_mul
