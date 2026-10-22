# The pinned sgl-amd-mono imports this path; drop once it uses
# sglang.kernels.plugin_kernels.amd.spec.
import sys

from sglang.kernels.plugin_kernels.amd import spec

sys.modules[__name__] = spec
