# Adapted from https://github.com/vllm-project/vllm/blob/v0.10.0/vllm/compilation/pass_manager.py

import logging

from torch import fx as fx

from sglang.srt.compilation.fusion.passes.fused_activation import FusedActivationPass
from sglang.srt.compilation.fusion.passes.rmsnorm_quant import RMSNormQuantPass

# from sglang.srt.compilation.fix_functionalization import FixFunctionalizationPass
from sglang.srt.compilation.inductor_pass import (
    CustomGraphPass,
    InductorPass,
    SGLangInductorPass,
)
from sglang.srt.compilation.pass_config import PassConfig

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

    def __init__(self, pass_config: PassConfig):
        self.pass_config = pass_config
        self.passes: list[SGLangInductorPass] = []

    def __call__(self, graph: fx.Graph):
        logger.debug("Running custom inductor passes.")

        # TODO pass context is not set when running the pass manager
        # directly, i.e during torch compile in cuda graph runner
        # shape = get_pass_context().runtime_shape
        for pass_ in self.passes:
            # if pass_.is_applicable_for_shape(shape):
            pass_(graph)

        # TODO: not required if using auto_functionalized_v2
        # always run fix_functionalization last
        # self.fix_functionalization(graph)

    def configure(self):
        # self.fix_functionalization = FixFunctionalizationPass()
        if self.pass_config.enable_fusion:
            if not self.pass_config.disable_rmsnorm_quant_pass:
                self.passes.append(RMSNormQuantPass(self.pass_config))

            if not self.pass_config.disable_fused_activation_pass:
                self.passes.append(FusedActivationPass(self.pass_config))

        logger.debug(
            f"Passes Configured: {list(map(lambda x: x.pass_name, self.passes))}"
        )

    def add(self, pass_: InductorPass):
        assert isinstance(pass_, InductorPass)
        self.passes.append(pass_)

    def uuid(self):
        """
        The PostGradPassManager is set as a custom pass in the Inductor and
        affects compilation caching. Its uuid depends on the UUIDs of all
        dependent passes and the pass config. See InductorPass for more info.
        """
        state = {"pass_config": self.pass_config.uuid(), "passes": []}
        for pass_ in self.passes:
            state["passes"].append(pass_.uuid())
        # state["passes"].append(self.fix_functionalization.uuid())
        return InductorPass.hash_dict(state)
