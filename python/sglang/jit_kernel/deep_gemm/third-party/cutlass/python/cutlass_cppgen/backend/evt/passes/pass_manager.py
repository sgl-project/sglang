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
Pass manager for DAG IR.
"""

from typing import Any

import networkx as nx

from cutlass_cppgen.backend.evt.ir import DAGIR
from cutlass_cppgen.backend.evt.passes.util import cc_map


class EVTPassBase:
    """
    Base class for EVT Passes
    """
    dependencies = []
    def __init__(self, dag_ir: DAGIR) -> None:
        self.dag_ir = dag_ir
        self.cc = self.dag_ir.cc

    def requires(self) -> None:
        """
        This function will be called before the pass is run.
        """
        pass

    def call(self) -> None:
        """
        The pass that is run through the self.dag_ir
        """
        raise NotImplementedError(
            f"__call__ is not overwritten in Pass {self.__class__.__name__}")

    def ensures(self) -> None:
        """
        This function will be called after the pass is run.
        """
        pass

    def __call__(self) -> Any:
        self.requires()
        self.call()
        self.ensures()

    def cc_specific_method(self, func):
        """
        This enables defining function that behaves differently under different cc
        The simplest example of using this function is the following

        .. highlight:: python
        .. code-block:: python

        class ExamplePass(EVTPassBase):

            def call(sekf):
                # This automatically select the smXX_func based on current cc
                self.cc_specific_method(self.func)()

            # Interface func, can be empty
            def func(self):
                pass

            # Sm90 specific func
            def sm90_func(self):
                // sm90 specific method
                return

            # Sm80 specific func
            def sm80_func(self):
                // sm80 specific method
                return
        """
        func_name = f"sm{cc_map[self.cc]}_{func.__name__}"
        if hasattr(self, func_name):
            return getattr(self, func_name)
        else:
            raise NotImplementedError(f"func {func.__name__} is not overwritten for Sm{self.cc}")


class EVTPassManager(nx.DiGraph):
    """
    Topological-based Pass Manager.
    Each registered pass has a list of dependencies. The pass manager organizes
    the passes as a DAG and launch the compiler passes under topological order.
    """
    def __init__(self, dag_ir: DAGIR, pass_list):
        super().__init__()
        self.dag_ir = dag_ir
        for pass_cls in pass_list:
            self.add_pass(pass_cls)

        self.sorted_passes = self.schedule()

    def get_callable(self, pass_name):
        """
        Return the callable of the pass
        """
        return self.nodes[pass_name]["callable"]

    def add_pass(self, pass_cls):
        """
        Add a pass to the pass manager
        :param pass_cls: the class of pass
        :type pass_cls: derived class of EVTPassBase
        """
        name = pass_cls.__name__
        pass_callable = pass_cls(self.dag_ir)
        self.add_node(name, callable=pass_callable)

    def schedule(self):
        """
        Schedule the added passes under topological order
        """
        # Add edges
        for pass_name in self.nodes:
            callable = self.get_callable(pass_name)
            for dependency_cls in callable.dependencies:
                self.add_edge(
                    dependency_cls.__name__,
                    type(callable).__name__)

        # Topological sort
        return list(nx.topological_sort(self))

    def __call__(self) -> Any:
        """
        Launch the registered passes
        """
        for pass_name in self.sorted_passes:
            callable = self.get_callable(pass_name)
            callable()
