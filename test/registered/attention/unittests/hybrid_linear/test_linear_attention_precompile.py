"""The FlashInfer bridge uses the existing AOT cache without changing other keys."""

import sys

import pytest
from sglang.kernels.jit.cute_aot_cache import JITCache
from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
    _FlashInferGDNCache,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_flashinfer_bridge_reuses_aot_cache():
    kernel = type("Kernel", (), {"__module__": "flashinfer.gdn_kernels.test"})
    ffi_option = type("EnableTVMFFI", (), {})
    key = (((kernel, object), ("head_dim", 128)), ((ffi_option, True),))
    aot = JITCache()
    first = _FlashInferGDNCache({}, aot)
    assert first.get(key) is None
    compiled = object()
    first[key] = compiled
    assert aot[key] is compiled
    second = _FlashInferGDNCache({}, aot)
    assert second.get(key) is compiled
    assert second[key] is compiled
    other_key = ("non-GDN", ())
    second[other_key] = compiled
    assert second.get(other_key) is compiled
    assert other_key not in aot


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
