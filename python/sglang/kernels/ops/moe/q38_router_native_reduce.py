"""Frozen native SplitK+quant for M4/M8; existing K2 for native F.linear M16."""
from functools import cache
import hashlib
from pathlib import Path

@cache
def verify_installed_source():
    from flashinfer.gemm.kernels import dense_bf16_gemm_sm100_splitk as installed
    expected = "f5edd4c11d95822c836a363dfa1dd19f7843b4e354e091c24c0acac12f52d9d2"
    actual = hashlib.sha256(Path(installed.__file__).read_bytes()).hexdigest()
    assert actual == expected, ("installed FlashInfer SplitK identity", actual, expected)
    return actual

def router_gemm_quant(x, weight, global_scale, *, ksplit):
    if x.shape[0] in (4, 8):
        assert ksplit == 4
        verify_installed_source()
        from .q38_router_native_kernel.fused import router_gemm_quant as native_fused
        return native_fused(x, weight, global_scale)
    assert x.shape[0] == 16 and ksplit == 2
    from .q38_router_front import router_gemm_quant as k2
    return k2(x, weight, global_scale, ksplit=2)
