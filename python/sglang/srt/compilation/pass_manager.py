# Adapted from https://github.com/vllm-project/vllm/blob/v0.10.0/vllm/compilation/pass_manager.py

import logging
from typing import Optional

from torch import fx as fx

from sglang.srt.compilation.fix_functionalization import FixFunctionalizationPass
from sglang.srt.compilation.inductor_pass import (
    CustomGraphPass,
    InductorPass,
    SGLangInductorPass,
    get_pass_context,
)
from sglang.srt.compilation.replace_scaled_mm import ReplaceScaledMMWithCutlassPass
from sglang.srt.layers.quantization.fp8_utils import get_fp8_gemm_runner_backend

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

        if self.replace_scaled_mm is not None:
            self.replace_scaled_mm(graph)

        # always run fix_functionalization last
        self.fix_functionalization(graph)

    def configure(
        self,
    ):
        self.pass_config = dict()
        # Gate the CUTLASS replacement on --fp8-gemm-backend cutlass. When the
        # user hasn't opted in, leave aten::_scaled_mm alone (cuBLASLt nvjet
        # path, status quo). The pass is a runtime no-op on graphs without
        # eligible _scaled_mm nodes, but we still skip registration so the
        # inductor cache key for non-cutlass backends is unchanged.
        self.replace_scaled_mm: Optional[ReplaceScaledMMWithCutlassPass] = None
        if get_fp8_gemm_runner_backend().is_cutlass():
            self.replace_scaled_mm = ReplaceScaledMMWithCutlassPass()
            logger.info(
                "fp8-gemm-backend=cutlass: registering "
                "ReplaceScaledMMWithCutlassPass to rewrite eligible "
                "aten::_scaled_mm nodes to sgl_kernel.fp8_scaled_mm."
            )
        self.fix_functionalization = FixFunctionalizationPass()

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
        if self.replace_scaled_mm is not None:
            state["passes"].append(self.replace_scaled_mm.uuid())
        state["passes"].append(self.fix_functionalization.uuid())
        return InductorPass.hash_dict(state)
