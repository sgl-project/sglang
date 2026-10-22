# The pinned sgl-nvidia-mega imports this path; drop once it uses
# sglang.srt.kernel_plugins.nvidia.binding.
import sys

from sglang.srt.kernel_plugins.nvidia import binding

sys.modules[__name__] = binding
