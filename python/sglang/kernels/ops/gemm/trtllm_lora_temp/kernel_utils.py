from sglang.kernels.jit.utils import is_arch_support_pdl
from sglang.kernels.ops.gemm.kernel_utils import (
    _resolve_token_positions as _resolve_token_positions,
)


def get_pdl_launch_metadata() -> tuple[bool, dict]:
    """Return (ENABLE_PDL constexpr value, extra launch kwargs) for LoRA kernels.

    ``launch_pdl`` is NVIDIA-only Triton launch metadata; the HIP backend
    rejects unknown kwargs, so it is only included when PDL is supported.
    """
    enable_pdl = is_arch_support_pdl()
    return enable_pdl, ({"launch_pdl": True} if enable_pdl else {})
