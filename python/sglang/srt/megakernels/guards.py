# The pinned sgl-nvidia-mega imports this path; drop once it uses
# sglang.srt.kernel_plugins.nvidia.integration.
import sys

from sglang.srt.kernel_plugins.nvidia import integration

sys.modules[__name__] = integration
