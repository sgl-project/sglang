"""Kernel types only the SAIL vendor DeepGEMM build serves.

``layers/deep_gemm_wrapper/compile_utils.py`` owns the warmup machinery and the
enum for the kernel types every DeepGEMM build serves. The nopad, channel-wise,
int8, fp4, int4 and gather-read fused types exist in the vendor build only, so
they are declared here; the shared tables key on ``.name`` and accept either
enum.

Declared as a plain :class:`enum.Enum` rather than an ``IntEnum`` on purpose.
``Enum`` hashes a member by its name, which is exactly the identity the shared
tables use, while ``IntEnum`` hashes by value and would collide with a different
type that happens to carry the same number. The flip side of hashing by name is
that a member name here must stay distinct from every member name in
``DeepGemmKernelType``.
"""

from enum import Enum, auto


class SailDeepGemmKernelType(Enum):
    # Grouped GEMMs whose caller hands over the per-expert row counts (nopad) or
    # asks for per-channel weight scales.
    GROUPED_GEMM_NT_F8F8BF16_NOPAD = auto()
    GROUPED_GEMM_NT_F8F8BF16_MASKED_CHANNEL = auto()
    GROUPED_GEMM_NT_F8F8BF16_CONTIG_CHANNEL = auto()
    GROUPED_GEMM_NT_F8F8BF16_NOPAD_CHANNEL = auto()

    # The dense fp8 GEMM with one scale per output channel. Declared here
    # because only the vendor build accepts that scale layout, so the shared
    # entrypoint names it inside its vendor branch.
    GEMM_NT_F8F8BF16_CHANNEL = auto()

    GROUPED_GEMM_NT_I8I8BF16_MASKED = auto()
    GROUPED_GEMM_NT_I8I8BF16_CONTIG = auto()
    GROUPED_GEMM_NT_I8I8BF16_NOPAD = auto()
    GEMM_NT_I8I8BF16 = auto()

    GROUPED_GEMM_NT_BF16_NOPAD = auto()

    GROUPED_GEMM_NT_F4F4BF16_MASKED = auto()
    GROUPED_GEMM_NT_F4F4BF16_NOPAD = auto()
    GEMM_NT_F4F4BF16 = auto()
    GROUPED_GEMM_NT_F4F4BF16_MASKED_BIAS = auto()
    GROUPED_GEMM_NT_F4F4BF16_NOPAD_BIAS = auto()
    GEMM_NT_F4F4BF16_BIAS = auto()

    GROUPED_GEMM_NT_BF16I4BF16_MASKED = auto()
    GROUPED_GEMM_NT_BF16I4BF16_NOPAD = auto()

    # GEMM1 reads the routing indices directly, skipping the physical scatter.
    GROUPED_GEMM_NT_F8F8BF16_FUSED = auto()
    GROUPED_GEMM_NT_F8F8BF16_FUSED_CHANNEL = auto()
    GROUPED_GEMM_NT_I8I8BF16_FUSED = auto()
    GROUPED_GEMM_NT_BF16_FUSED = auto()
    GROUPED_GEMM_NT_F4F4BF16_FUSED = auto()

    # The two int4 weight encodings: the MMA one keeps the packed 4-bit K, the
    # other expands it, and the VALU one pairs fp4 weights with a bf16 scale.
    GROUPED_GEMM_NT_BF16I4BF16_MMA_MASKED = auto()
    GROUPED_GEMM_NT_BF16I4BF16_MMA_NOPAD = auto()
    GROUPED_GEMM_NT_BF16I4BF16_MMA_FUSED = auto()
    GROUPED_GEMM_NT_BF16I4BF16_FUSED = auto()
    GROUPED_GEMM_NT_BF16I4BF16_VALU_FUSED = auto()
