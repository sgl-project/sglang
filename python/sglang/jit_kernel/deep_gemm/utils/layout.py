try:
    from .._C import (
        get_tma_aligned_size,
        get_mn_major_tma_aligned_tensor,
        get_mn_major_tma_aligned_packed_ue8m0_tensor,
        get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor
    )
except ImportError:
    # Expected behavior for CUDA runtime version before 12.1
    pass

# Valid for all CUDA versions
from .._C import get_mk_alignment_for_contiguous_layout

# Some alias
get_m_alignment_for_contiguous_layout = get_mk_alignment_for_contiguous_layout
get_k_alignment_for_contiguous_layout = get_mk_alignment_for_contiguous_layout
