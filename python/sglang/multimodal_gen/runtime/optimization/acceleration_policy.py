# SPDX-License-Identifier: Apache-2.0
"""Public facade for diffusion acceleration policy helpers.

Callers import this module as the stable policy surface. Implementation is split
by concern under `runtime.optimization` so attention, full DiT compile, and
kernel-wise CustomOp compile policies can evolve independently.
"""

from sglang.multimodal_gen.runtime.optimization.attention_policy import (
    attention_allows_cudnn_sdp,
    attention_autotune_config,
)
from sglang.multimodal_gen.runtime.optimization.kernel_policy import (
    DEFAULT_KERNEL_COMPILE_OPS,
    KERNEL_COMPILE_ITERS_ENV,
    KERNEL_COMPILE_LIVE_MISS_ENV,
    KERNEL_COMPILE_MIN_SPEEDUP_ENV,
    KERNEL_COMPILE_MODE_ENV,
    KERNEL_COMPILE_OPS_ENV,
    KERNEL_COMPILE_OP_GROUPS,
    KERNEL_COMPILE_POLICY_ENV,
    KERNEL_COMPILE_WARMUP_ENV,
    KernelCompileAutotuneConfig,
    custom_op_kernel_compile_policy,
    kernel_compile_autotune_config,
    kernel_compile_autotune_suppressed,
    kernel_compile_kwargs,
    should_torch_compile_custom_op,
    suppress_kernel_compile_autotune,
)
from sglang.multimodal_gen.runtime.optimization.runtime_config import (
    configure_acceleration_policy,
)
from sglang.multimodal_gen.runtime.optimization.torch_compile_policy import (
    TORCH_COMPILE_MODE_ENV,
    TorchCompileAutotuneConfig,
    torch_compile_autotune_config,
    torch_compile_kwargs,
)
