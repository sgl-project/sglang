# This design is borrowed from https://blog.vllm.ai/2025/08/20/torch-compile.html

import logging

from torch import fx

from sglang.srt.configs.compilation_config import CompilationConfig
from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_bool_env_var,
    is_cpu,
    is_cuda,
    is_flashinfer_available,
    is_hip,
    is_npu,
    is_xpu,
    supports_custom_op,
)

from .sglang_inductor_pass import SglangInductorPass

_is_cuda = is_cuda()

if _is_cuda:
    from .collective_fusion import AllReduceFusionPass

from .inductor_pass import CustomGraphPass, InductorPass, get_pass_context

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

    def __init__(
        self,
        compilation_config: CompilationConfig,
        model_config: ModelConfig,
        device_config: DeviceConfig,
    ):
        self.passes: list[InductorPass] = []
        self.configure(compilation_config, model_config, device_config)

    def __call__(self, graph: fx.Graph):
        SglangInductorPass.dump_prefix = 0  # reset dump index

        shape = get_pass_context().runtime_shape
        for pass_ in self.passes:
            if pass_.is_applicable_for_shape(shape):
                pass_(graph)
                SglangInductorPass.dump_prefix += 1

        # post-cleanup goes before fix_functionalization
        # because it requires a functional graph
        self.post_cleanup(graph)
        SglangInductorPass.dump_prefix += 1

        # always run fix_functionalization last
        SglangInductorPass.dump_prefix = None  # Cleanup index

    # TODO: wrap three configs into a SglangConfig
    def configure(
        self,
        compilation_config: CompilationConfig,
        model_config: ModelConfig,
        device_config: DeviceConfig,
    ):
        self.pass_config = compilation_config.pass_config

        if self.pass_config.enable_fi_allreduce_fusion:
            self.passes += [
                AllReduceFusionPass(compilation_config, model_config, device_config)
            ]

        # TODO: add more pass for fusion and tp.

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
        state["passes"].append(self.fix_functionalization.uuid())
        return InductorPass.hash_dict(state)
