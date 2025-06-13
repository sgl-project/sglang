from contextlib import contextmanager
from enum import IntEnum, auto


class DeepGemmKernelType(IntEnum):
    GROUPED_GEMM_NT_F8F8BF16_MASKED = auto()
    GROUPED_GEMM_NT_F8F8BF16_CONTIG = auto()
    GEMM_NT_F8F8BF16 = auto()


@contextmanager
def _log_jit_build(M: int, N: int, K: int, kernel_type: DeepGemmKernelType):
    if _IN_PRECOMPILE_STAGE:
        yield
        return

    from deep_gemm.jit.runtime import RuntimeCache

    origin_func = RuntimeCache.get

    def __patched_func(self, *args, **kwargs):
        ret = origin_func(self, *args, **kwargs)
        if ret is None:
            kernel_helper = _KERNEL_HELPER_DICT[kernel_type]
            _compile_warning_2()
            logger.warning(
                f"DeepGEMM JIT Compiling for <{kernel_helper.name}> M={M}, N={N}, K={K}. Please wait."
            )
        return ret

    RuntimeCache.get = __patched_func
    yield
    RuntimeCache.get = origin_func
