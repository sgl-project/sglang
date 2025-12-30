# Adapted from https://github.com/vllm-project/vllm/blob/v0.10.0/vllm/compilation/pass_manager.py

import logging

from torch import fx as fx

from sglang.srt.compilation.collective_fusion import AllReduceFusionPass
from sglang.srt.compilation.fix_functionalization import FixFunctionalizationPass
from sglang.srt.compilation.inductor_pass import (
    CustomGraphPass,
    InductorPass,
    get_pass_context,
)
from sglang.srt.compilation.sglang_config import SGLangConfig, set_current_sglang_config
from sglang.srt.compilation.sglang_inductor_pass import SGLangInductorPass

logger = logging.getLogger(__name__)


class PostGradPassManager(CustomGraphPass):
    """
    The pass manager for post-grad passes.
    It handles configuration, adding custom passes, and running passes.
    It supports uuid for the Inductor code cache. That includes torch<2.6
    support using pickling (in .inductor_pass.CustomGraphPass).

    The order of the post-grad post-passes is:
    1. passes (constructor parameter)
    2. default passes (NoopEliminationPass, FusionPass)
    3. config["post_grad_custom_post_pass"] (if it exists)
    4. fix_functionalization
    This way, all passes operate on a functionalized graph.
    """

    def __init__(self):
        self.passes: list[SGLangInductorPass] = []

    def __call__(self, graph: fx.Graph):
        shape = get_pass_context().runtime_shape
        for pass_ in self.passes:
            if pass_.is_applicable_for_shape(shape):
                pass_(graph)

        # always run fix_functionalization last
        self.fix_functionalization(graph)

    def configure(self, config: SGLangConfig):
        self.pass_config = config.compilation_config.pass_config

        with set_current_sglang_config(config, check_compile=False):
            if self.pass_config.enable_fi_allreduce_fusion:
                self.passes += [AllReduceFusionPass(config)]
            self.fix_functionalization = FixFunctionalizationPass(config)

    def add(self, pass_: InductorPass):
        assert isinstance(pass_, InductorPass)
        self.passes.append(pass_)

    def uuid(self):
        """
        The PostGradPassManager is set as a custom pass in the Inductor and
        affects compilation caching. Its uuid depends on the UUIDs of all
        dependent passes and the pass config. See InductorPass for more info.
        """
        pass_manager_uuid = "fshdakhsa"
        state = {"pass_config": pass_manager_uuid, "passes": []}
        for pass_ in self.passes:
            state["passes"].append(pass_.uuid())
        state["passes"].append(self.fix_functionalization.uuid())
        return InductorPass.hash_dict(state)
