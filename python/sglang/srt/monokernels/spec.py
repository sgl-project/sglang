"""The monokernel contracts at their former import path.

The MSA plugin pinned by MSA_COMMIT in docker/Dockerfile.cu134 imports them from
here; drop this module once the pinned plugin imports
``sglang.kernels.plugin_kernels.amd.spec``.
"""

from sglang.kernels.plugin_kernels.amd.spec import *  # noqa: F401,F403
