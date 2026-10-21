# The pinned sgl-nvidia-mega imports this path; drop once it imports
# sglang.kernels.plugin_kernels.nvidia.
import sys

from sglang.kernels.plugin_kernels.nvidia import guards

sys.modules[__name__] = guards
