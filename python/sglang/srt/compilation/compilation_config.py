# Adapted from https://github.com/vllm-project/vllm/blob/v0.10.0/vllm/config/compilation.py

import logging
from dataclasses import asdict, dataclass, field
from typing import List, Optional

from sglang.srt.compilation.inductor_pass import InductorPass

logger = logging.getLogger(__name__)


class CompilationMode:
    """The compilation approach used for torch.compile-based compilation of the
    model."""

    NONE = 0
    """No torch.compile compilation is applied, model runs in fully eager pytorch mode.
    The model runs as-is."""
    STOCK_TORCH_COMPILE = 1
    """The standard `torch.compile` compilation pipeline."""
    DYNAMO_TRACE_ONCE = 2
    """Single Dynamo trace through the model, avoiding recompilation."""
    SGLANG_COMPILE = 3
    """Custom SGLang Inductor-based backend with caching, piecewise compilation,
    shape specialization, and custom passes."""


@dataclass
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

    capture_sizes: Optional[List[int]] = None

    compiler: str = "eager"

    enable_debug_mode: bool = False

    traced_files: set[str] = field(default_factory=set, init=False)

    split_ops: list[str] = field(default_factory=list, init=False)

    # Top-level Compilation control
    level: Optional[int] = None

    mode: CompilationMode | None = None

    # The backend for compilation. It needs to be a string:
    # (empty string): use the default backend ("inductor" on CUDA-alike
    # platforms).
    backend: str = ""

    custom_ops: list[str] = field(default_factory=list)

    # Inductor capture
    use_inductor: bool = True

    splitting_ops: list[str] | None = None

    use_inductor_graph_partition: bool = False

    inductor_compile_config: dict = field(default_factory=dict)

    inductor_passes: dict[str, str] = field(default_factory=dict)

    pass_config: PassConfig = field(default_factory=PassConfig)

    # time taken for compilation
    compilation_time: float = field(default=0.0, init=False)

    def __post_init__(self):
        self.split_ops = [
            "sglang.unified_attention_with_output",
            "sglang.gdn_with_output",
            "sglang.inplace_all_reduce",
        ]

    def add_split_op(self, op: str):
        self.split_ops.append(op)

    def add_traced_file(self, file_path: str):
        self.traced_files.add(file_path)

    def get_traced_files(self):
        return self.traced_files

    def get_capture_sizes(self):
        return self.capture_sizes

    def get_enable_debug_mode(self):
        return self.enable_debug_mode
