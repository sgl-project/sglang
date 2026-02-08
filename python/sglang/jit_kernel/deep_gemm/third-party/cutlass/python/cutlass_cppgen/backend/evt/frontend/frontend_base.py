#################################################################################################
#
# Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#################################################################################################

"""
Base class for Python EVT Frontend
"""

from typing import Union

from cutlass_library import DataType
from cutlass_cppgen.backend.evt.ir import (
    ComputeNode,
    DAGIR,
    LayoutNode,
    LoadNode,
    StoreNode,
)
from cutlass_cppgen.backend.evt.passes import (
    EVTGraphDrawer,
    EVTPassManager,
    GetSmemSize,
    PassDAG2Tree,
    PassGetArgumentType,
    PassGetImpl,
    PassFixElementD,
    PassLayoutManipulateElimination,
    PassPreprocessRed,
    PassShapeTypePropagation,
)
from cutlass_cppgen.backend.evt.passes.util import cc_map
from cutlass_cppgen.backend.utils import device_cc
from cutlass_cppgen.epilogue.evt_ops import permute, reshape
from cutlass_cppgen.utils.datatypes import library_type


class EVTFrontendBase:
    layout_fns = {
        "permute": permute,
        "reshape": reshape
    }

    def __init__(self, cc, element_compute=DataType.f32, additional_passes=[], **kwargs) -> None:
        self.cc = cc
        self.element_compute = library_type(element_compute)
        self.dag_ir = DAGIR(self.cc, self.element_compute)
        self.compute_cnt = 0
        self.layout_cnt = 0
        self.imm_cnt = 0

        self.pass_manager = EVTPassManager(
            self.dag_ir,
            [
                PassPreprocessRed,
                PassGetArgumentType,
                PassShapeTypePropagation,
                PassLayoutManipulateElimination,
                PassGetImpl,
                PassDAG2Tree,
                PassFixElementD
            ] + additional_passes)

        if self.cc == 80:
            self._epilogue_stages = 1
        else:
            self._epilogue_stages = None

    @property
    def epilogue_stages(self):
        return self._epilogue_stages

    @epilogue_stages.setter
    def epilogue_stages(self, stages):
        self._epilogue_stages = stages


    def parse(self, *args, **kwargs):
        raise NotImplementedError(f"The 'parse' function must be overloaded in frontend class")

    def trace(self, *args, **kwargs):
        # Parse the input
        self.parse(*args, **kwargs)

        # Verify the DAG IR to ensure that "D" is the output node with out_degree = 0
        if (self.cc >= 90):
            if (self.dag_ir.out_degree("D") != 0):
                raise RuntimeError(
                    f"On SM90 or higher, D is expected to be a output node with 0 users to "
                    f"enable smem reuse between C and D, but got {self.dag_ir.out_degree('D')}")

        # Run the passes
        self.pass_manager()
        # Set the epilogue type
        self.epilogue_thread_type = self.dag_ir.epilogue_thread_type
        if cc_map[self.cc] in [90, 100]:
            self.arg_c_type = self.dag_ir.arg_c_type
            self.arg_d_type = self.dag_ir.arg_d_type
        self.reduction_names = self.dag_ir.reduction_names

    #
    # Helper functions for DAG IR manipulation
    #

    def add_node(self, node):
        self.dag_ir.add_node(node)

    def add_edge(self, src, tgt, weight=0):
        self.dag_ir.add_edge(src, tgt, weight=weight)

    def set_tensor(self, node_name, example):
        """
        Add an example tensor to node {node_name} in the DAG IR
        """
        meta = self.dag_ir.get_node_meta(node_name)
        meta.tensor = {"tensor": example}

    def set_store_tensor(self, node_name, example):
        """
        Add an example tensor to node {node_name} in the DAG IR
        """
        meta = self.dag_ir.get_node_meta(node_name)
        meta.store_tensor = {"tensor": example}

    def mark_output(self, node_name):
        """
        Mark a store node as output
        """
        meta = self.dag_ir.get_node_meta(node_name)
        if not isinstance(meta, StoreNode):
            raise ValueError(
                f"Only StoreNodes can be marked as output. "
                f"Got {type(meta).__name__}: {node_name}")
        meta.is_output = True

    # Add node with specific type

    def add_load_node(self, name, example):
        """
        Add a Load node to DAG IR
        :param name: name of the loaded variable
        :type name: str
        :param example: example input
        :type example: np.ndarray|torch.Tensor|cupy.ndarray|float
        """
        if name is None:
            raise ValueError(f"Name is not provided.")
        if example is None:
            raise ValueError(f"Example input for {name} is not provided.")
        load_node = LoadNode(name)
        load_node.tensor = {"tensor": example}
        # Special logics for accumulator
        if name == "accum":
            if load_node.tensor.rank == 2:
                new_shape = tuple([1, ] + list(load_node.tensor.shape))
                load_node.tensor.broadcast(new_shape)
            elif load_node.tensor.rank < 2 or load_node.tensor.rank > 3:
                raise ValueError(f"Expect example inputs for 'accum' be a rank-2 or rank-3 tensor. Got {load_node.tensor.shape}.")
        self.add_node(load_node)

    def add_imm(self, value: Union[float,int]):
        """
        Add an immediate scalar value to DAG IR
        :param value: the value of the immediate scalar
        :type value: float
        """
        try:
            value = float(value)
        except:
            raise ValueError(f"{type(value).__name__} cannot be converted to float.")

        name = f"imm_{value}_k{self.imm_cnt}".replace('.', '_')
        self.imm_cnt += 1
        load_node = LoadNode(name)
        load_node.tensor = {"tensor": value, "is_constant": True}
        self.add_node(load_node)
        return name

    def add_compute_node(self, op, name=None):
        """
        Add a compute node.
        :param op: the computation op
        :param name: the node name (optional)
        :type name: str
        :return: the name of the compute node
        """
        if name is None:
            name = f"compute_{self.compute_cnt}"
            self.compute_cnt += 1
        compute_node = ComputeNode(
            name=name, fn=op,
            element_output=self.element_compute,
            element_compute=self.element_compute)
        self.add_node(compute_node)
        return compute_node.name

    def add_layout_node(self, op, kwargs, name=None):
        """
        Add a layout node.
        :param op: the layout op
        :type op: evt_ops
        :param name: the node name (optional)
        :type name: str
        :return: the name of the layout node
        """
        if name is None:
            name = f"layout_{self.layout_cnt}"
            self.layout_cnt += 1
        layout_node = LayoutNode(name=name, fn=op, kwargs=kwargs)
        self.add_node(layout_node)
        return layout_node.name

    def add_store_node(self, name):
        store_node = StoreNode(name)
        self.add_node(store_node)

    #
    # Visualization The DAG IR
    #

    def visualize(self, name="dag_ir"):
        """
        Visualize the dag ir with svg file
        :param name: the name of the graph
        """
        drawer = EVTGraphDrawer(self.dag_ir, name)
        try:
            for name, graph in drawer.get_dot_graph():
                graph.write_svg(f"./{name}.svg")
        except:
            raise RuntimeError(
                "'dot' is not found in path. GraphDrawer is disabled. "
                "Please install it with 'sudo apt-get install graphviz'."
            )

    #
    # Get shared memory size
    #

    def get_smem_size(self, tile_description):
        """
        Get the shared memory size of the epilogue
        """
        smem_size = GetSmemSize(self.dag_ir)(tile_description)
        return smem_size
