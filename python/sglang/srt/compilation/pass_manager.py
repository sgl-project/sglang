# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/v0.10.0/vllm/compilation/pass_manager.py

from torch import fx as fx

from sglang.srt.compilation.inductor_pass import (
    CustomGraphPass,
    InductorPass,
    SGLangInductorPass,
    get_pass_context,
)


class PostGradPassManager(CustomGraphPass):
    """Run post-grad passes in insertion order for the current runtime shape.

    The ordered pass UUIDs identify the manager in Inductor's code cache.
    """

    def __init__(self):
        self.passes: list[SGLangInductorPass] = []

    def __call__(self, graph: fx.Graph):
        shape = get_pass_context().runtime_shape
        for pass_ in self.passes:
            if pass_.is_applicable_for_shape(shape):
                pass_(graph)

    def add(self, pass_: InductorPass):
        assert isinstance(pass_, InductorPass)
        self.passes.append(pass_)

    def uuid(self):
        """
        The PostGradPassManager is set as a custom pass in the Inductor and
        affects compilation caching. Its uuid depends on the UUIDs of all
        dependent passes. See InductorPass for more info.
        """
        return InductorPass.hash_dict({"passes": [p.uuid() for p in self.passes]})
