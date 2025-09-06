# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
SGLang FX Pass Manager - torch.fx based kernel fusion pass manager
"""

import logging
import operator
from typing import Optional

import sgl_kernel
import torch
import torch.fx as fx
import torch.nn as nn
from torch.fx.node import Node

logger = logging.getLogger(__name__)


class SiLUMulFusionPass:
    def __init__(self):
        self.name = "SiLUMulFusionPass"

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
        if node.op == "call_function":
            return node.target in [
                torch.nn.functional.silu,
                torch.ops.aten.silu.default,
            ]
        elif node.op == "call_module":
            return "silu" in str(node.target).lower()
        return False

    def _is_mul_node(self, node: Node) -> bool:
        if node.op == "call_function":
            mul_targets = [
                torch.mul,
                torch.ops.aten.mul.Tensor,
                torch.ops.aten.mul.default,
                operator.mul,
            ]
            return node.target in mul_targets
        elif node.op == "call_method":
            return "mul" in str(node.target).lower()
        return False

    def _can_fuse_silu_mul(self, silu_node: Node, mul_node: Node) -> bool:
        return silu_node in mul_node.args

    def _create_fused_silu_mul(self, graph: fx.Graph, silu_node: Node, mul_node: Node):
        silu_input = silu_node.args[0]  # This should be x[..., :dim]

        # Find the mul's other operand (should be x[..., dim:])
        other_arg = None
        for arg in mul_node.args:
            if arg != silu_node:
                other_arg = arg
                break

        if (
            hasattr(silu_input, "op")
            and hasattr(other_arg, "op")
            and silu_input.op == "call_function"
            and other_arg.op == "call_function"
            and silu_input.target == operator.getitem
            and other_arg.target == operator.getitem
            and len(silu_input.args) > 0
            and len(other_arg.args) > 0
            and silu_input.args[0] == other_arg.args[0]
        ):

            base_tensor = silu_input.args[0]

            with graph.inserting_after(mul_node):
                fused_node = graph.call_function(
                    self._get_fused_silu_mul_op(), args=(base_tensor,), kwargs={}
                )

        mul_node.replace_all_uses_with(fused_node)

        graph.erase_node(mul_node)
        if len(silu_node.users) == 0:
            graph.erase_node(silu_node)

    def _get_fused_silu_mul_op(self):
        return sgl_kernel.silu_and_mul


class SGLangFXPassManager:
    """SGLang FX Pass Manager for SiLU and Mul fusion"""

    def __init__(self):
        self.passes = [SiLUMulFusionPass()]

    def apply_passes(self, model: nn.Module) -> nn.Module:
        logger.info("Starting SGLang FX Pass optimization...")

        try:
            tracer = fx.Tracer()
            graph = tracer.trace(model)
            fx_model = fx.GraphModule(model, graph)

            total_modifications = 0
            for pass_instance in self.passes:
                if pass_instance.is_applicable(graph):
                    logger.info(f"Applying {pass_instance.name}...")
                    modified = pass_instance.apply(graph)
                    if modified:
                        total_modifications += 1
                        logger.info(f"{pass_instance.name} applied successfully")
                else:
                    logger.debug(f"Skipping {pass_instance.name} (not applicable)")

            fx_model.recompile()

            logger.info(
                f"FX Pass optimization completed. Applied {total_modifications} passes."
            )
            return fx_model

        except Exception as e:
            logger.warning(f"FX Pass optimization failed: {e}")
            return model


def apply_sglang_fx_optimization(model: nn.Module) -> nn.Module:
    pass_manager = SGLangFXPassManager()
    return pass_manager.apply_passes(model)
