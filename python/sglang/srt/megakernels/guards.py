# The pinned sgl-nvidia-mega imports this path; drop once it imports
# sglang.kernels.low_latency_kernels.megakernels.
import sys

from sglang.kernels.low_latency_kernels.megakernels import guards

sys.modules[__name__] = guards
