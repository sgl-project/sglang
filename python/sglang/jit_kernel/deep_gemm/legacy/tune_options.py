from triton import Config
from .._C import get_mk_alignment_for_contiguous_layout


def get_config_smem_size(config: Config, elem_bytes: int = 2):
    # NOTES: FP8 kernels will not use Triton, so by default we assume BF16 kernels
    return (config.kwargs['BLOCK_SIZE_M'] + config.kwargs['BLOCK_SIZE_N']) * config.kwargs['BLOCK_SIZE_K'] * elem_bytes * config.num_stages


_gemm_configs = [
    Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
    Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
    Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
    Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
    Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
    Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
    Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
    Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
    Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
]

# NOTES: we only consider A100 shared memory sizes here, as legacy kernels are only used for Ampere
_gemm_configs = list(filter(lambda x: get_config_smem_size(x) <= 166912, _gemm_configs))
_gemm_configs = list(filter(lambda x: x.kwargs['BLOCK_SIZE_M'] <= get_mk_alignment_for_contiguous_layout(), _gemm_configs))
_gemm_configs = list(filter(lambda x: x.kwargs['BLOCK_SIZE_K'] <= get_mk_alignment_for_contiguous_layout(), _gemm_configs))

get_m_grouped_gemm_configs = lambda: list(filter(lambda x: x.kwargs['BLOCK_SIZE_M'] <= get_mk_alignment_for_contiguous_layout(), _gemm_configs))
get_k_grouped_gemm_configs = lambda: list(filter(lambda x: x.kwargs['BLOCK_SIZE_K'] <= get_mk_alignment_for_contiguous_layout(), _gemm_configs))
