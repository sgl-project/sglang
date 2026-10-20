# The pinned sgl-amd-mono imports this path; drop once it imports
# sglang.kernels.low_latency_kernels.monokernels.
import sys

from sglang.kernels.low_latency_kernels.monokernels import spec

sys.modules[__name__] = spec
