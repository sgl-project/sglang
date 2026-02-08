import os
import subprocess
import sys
import torch

# Set some default environment provided at setup
try:
    # noinspection PyUnresolvedReferences
    from .envs import persistent_envs
    for key, value in persistent_envs.items():
        if key not in os.environ:
            os.environ[key] = value
except ImportError:
    pass

# Configs
try:
    from . import _C
except ImportError:
    from torch.utils.cpp_extension import CUDA_HOME, load

    _root = os.path.dirname(os.path.abspath(__file__))
    _cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or CUDA_HOME
    if _cuda_home is None:
        raise

    _include_dirs = [
        os.path.join(_root, "csrc"),
        os.path.join(_root, "include"),
        os.path.join(_root, "third-party", "cutlass", "include"),
        os.path.join(_root, "third-party", "fmt", "include"),
        os.path.join(str(_cuda_home), "include"),
        os.path.join(str(_cuda_home), "include", "cccl"),
    ]

    _lib_dirs = [
        os.path.join(str(_cuda_home), "lib64"),
        os.path.join(str(_cuda_home), "lib"),
    ]
    _lib_dir_flags = [f"-L{p}" for p in _lib_dirs if os.path.isdir(p)]

    _extra_cflags = [
        "-std=c++17",
        "-O3",
        "-fPIC",
        "-Wno-psabi",
        "-Wno-deprecated-declarations",
        f"-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}",
    ]
    if int(os.environ.get("DG_JIT_USE_RUNTIME_API", "0")) == 1:
        _extra_cflags.append("-DDG_JIT_USE_RUNTIME_API")

    _extra_ldflags = _lib_dir_flags + ["-lcudart", "-lnvrtc"]

    _loaded = load(
        name="sglang_deep_gemm_C",
        sources=[os.path.join(_root, "csrc", "python_api.cpp")],
        extra_cflags=_extra_cflags,
        extra_cuda_cflags=[],
        extra_ldflags=_extra_ldflags,
        extra_include_paths=_include_dirs,
        with_cuda=True,
        verbose=False,
    )
    sys.modules[__name__ + "._C"] = _loaded
    _C = _loaded
from ._C import (
    set_num_sms,
    get_num_sms,
    set_tc_util,
    get_tc_util,
)

# cuBLASLt Kernels
from ._C import (
    cublaslt_gemm_nt, cublaslt_gemm_nn,
    cublaslt_gemm_tn, cublaslt_gemm_tt,
)

try:
    # DeepGEMM Kernels
    from ._C import (
        # FP8 FP4 GEMMs
        fp8_fp4_gemm_nt, fp8_fp4_gemm_nn,
        fp8_fp4_gemm_tn, fp8_fp4_gemm_tt,
        m_grouped_fp8_fp4_gemm_nt_contiguous,
        m_grouped_fp8_fp4_gemm_nn_contiguous,
        m_grouped_fp8_fp4_gemm_nt_masked,
        # FP8 GEMMs
        fp8_gemm_nt, fp8_gemm_nn,
        fp8_gemm_tn, fp8_gemm_tt,
        fp8_gemm_nt_skip_head_mid,
        m_grouped_fp8_gemm_nt_contiguous,
        m_grouped_fp8_gemm_nn_contiguous,
        m_grouped_fp8_gemm_nt_masked,
        k_grouped_fp8_gemm_nt_contiguous,
        k_grouped_fp8_gemm_tn_contiguous,
        # BF16 GEMMs
        bf16_gemm_nt, bf16_gemm_nn,
        bf16_gemm_tn, bf16_gemm_tt,
        m_grouped_bf16_gemm_nt_contiguous,
        m_grouped_bf16_gemm_nn_contiguous,
        m_grouped_bf16_gemm_nt_masked,
        k_grouped_bf16_gemm_tn_contiguous,
        # Einsum kernels
        einsum,
        fp8_einsum,
        # Attention kernels
        fp8_mqa_logits,
        get_paged_mqa_logits_metadata,
        fp8_paged_mqa_logits,
        # Hyperconnection kernels
        tf32_hc_prenorm_gemm,
        # Layout kernels
        transform_sf_into_required_layout,
        get_mk_alignment_for_contiguous_layout
    )

    # Some alias for legacy supports
    # TODO: remove these later
    fp8_m_grouped_gemm_nt_masked = m_grouped_fp8_gemm_nt_masked
    bf16_m_grouped_gemm_nt_masked = m_grouped_bf16_gemm_nt_masked
except ImportError:
    # Expected behavior for CUDA runtime version before 12.1
    pass

# Some utils
from . import testing
from . import utils
from .utils import *

# Legacy Triton kernels for A100
try:
    from . import legacy
except Exception as e:
    print(f'Failed to load legacy DeepGEMM A100 Triton kernels: {e}')

# Initialize CPP modules
def _find_cuda_home() -> str:
    # TODO: reuse PyTorch API later
    # For some PyTorch versions, the original `_find_cuda_home` will initialize CUDA, which is incompatible with process forks
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home is None:
        # noinspection PyBroadException
        try:
            with open(os.devnull, 'w') as devnull:
                nvcc = subprocess.check_output(['which', 'nvcc'], stderr=devnull).decode().rstrip('\r\n')
                cuda_home = os.path.dirname(os.path.dirname(nvcc))
        except Exception:
            cuda_home = '/usr/local/cuda'
            if not os.path.exists(cuda_home):
                cuda_home = None
    assert cuda_home is not None
    return cuda_home


_C.init(
    os.path.dirname(os.path.abspath(__file__)), # Library root directory path
    _find_cuda_home()                           # CUDA home
)

__version__ = '2.3.0'
