# The pinned sgl-nvidia-mega imports this path; drop once it uses
# sglang.kernels.plugin_kernels.nvidia.spec.
import sys

from sglang.kernels.plugin_kernels.nvidia import spec

sys.modules[__name__] = spec
