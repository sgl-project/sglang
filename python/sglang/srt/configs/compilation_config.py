import logging
from dataclasses import asdict, dataclass, field
from typing import Optional

import torch

from sglang.compilation.inductor_pass import InductorPass
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class CompilationLevel:
    # constants for the levels of the compilation process
    NO_COMPILATION = 0
    DYNAMO_AS_IS = 1
    DYNAMO_ONCE = 2
    PIECEWISE = 3


class PassConfig:
    """Configuration for custom Inductor passes.

    This is separate from general `CompilationConfig` so that inductor passes
    don't all have access to full configuration - that would create a cycle as
    the `PassManager` is set as a property of config."""

    """Whether to enable the custom fusion (RMSNorm/SiluMul+quant) pass."""
    enable_fusion: bool = True

    """Whether to enable flashinfer allreduce fusion."""
    enable_fi_allreduce_fusion: bool = True

    """Max number of tokens to used in flashinfer allreduce fusion."""
    fi_allreduce_fusion_max_token_num: int = 16384

    def uuid(self):
        """
        Produces a hash unique to the pass configuration.
        Any new fields that affect compilation should be added to the hash.
        Any future fields that don't affect compilation should be excluded.
        """
        return InductorPass.hash_dict(asdict(self))


@dataclass
class CompilationConfig:
    """Configuration for compilation. It has three parts:

    - Top-level Compilation control:
        -[''] TODO
    - CudaGraph capture:
        -[''] TODO
    - Inductor compilation:
        -[''] TODO
    Why we have different sizes for cudagraph and inductor:
    - cudagraph: a cudagraph captured for a specific size can only be used
        for the same size. We need to capture all the sizes we want to use.
    - inductor: a graph compiled by inductor for a general shape can be used
        for different sizes. Inductor can also compile for specific sizes,
        where it can have more information to optimize the graph with fully
        static shapes. However, we find the general shape compilation is
        sufficient for most cases. It might be beneficial to compile for
        certain small batchsizes, where inductor is good at optimizing.
    """

    # Top-level Compilation control
    level: Optional[int] = None

    custom_ops: list[str] = field(default_factory=list)

    # Inductor capture
    use_inductor: bool = True

    inductor_compile_config: dict = field(default_factory=dict)

    inductor_passes: dict[str, str] = field(default_factory=dict)

    pass_config: PassConfig = field(default_factory=PassConfig)

    @staticmethod
    def from_server_args(server_args: ServerArgs):
        # TODO(yuan-luo): Complete the Config.
        return CompilationConfig(
            level=CompilationLevel.DYNAMO_AS_IS,
        )
