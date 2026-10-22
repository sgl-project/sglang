# The pinned sgl-amd-mono imports this path; drop once it uses
# sglang.srt.kernel_plugins.amd.integration.
import sys

from sglang.srt.kernel_plugins.amd import integration

sys.modules[__name__] = integration
