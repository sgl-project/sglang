from cuda import cuda


class SharedMemoryAllocator:
    def malloc(self):
        return TODO

    def get_mem_handle(self):
        return TODO

    def open_mem_handle(self):
        return TODO


# copied from TensorRT-LLM
def _check_cu_result(cu_func_ret):
    # TODO optimize code
    if isinstance(cu_func_ret, tuple):
        cu_result, *others = cu_func_ret
        if cu_result != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(cu_result)
        if len(others) == 1:
            return others[0]
        elif len(others) > 1:
            return tuple(others)
        else:
            return None
    else:
        if cu_func_ret != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(cu_func_ret)
        return None
