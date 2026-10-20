# The pinned sgl-nvidia-mega imports this path; drop once it imports
# sglang.kernels.low_latency_kernels.megakernels.
import sys

from sglang.kernels.low_latency_kernels.megakernels import binding

sys.modules[__name__] = binding
