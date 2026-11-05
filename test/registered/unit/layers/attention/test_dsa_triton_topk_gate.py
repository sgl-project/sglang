"""The opt-in selector must not require Gluon when its route is disabled."""

import builtins
import sys

import pytest

from sglang.srt.layers.attention.dsa import dsa_topk_backend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    "is_hip,enabled", [(False, True), (True, False), (False, False)]
)
def test_disabled_route_does_not_import_optional_kernel(monkeypatch, is_hip, enabled):
    monkeypatch.setattr(dsa_topk_backend, "_is_hip", is_hip)
    monkeypatch.setenv("SGLANG_DSA_TOPK_TRITON", "1" if enabled else "0")
    original_import = builtins.__import__

    def checked_import(name, *args, **kwargs):
        if name == "sglang.kernels.ops.attention.dsa.triton_topk_transform":
            raise AssertionError("Disabled top-k route imported the optional kernel")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked_import)
    # An inactive route must not inspect tensor metadata either.
    assert dsa_topk_backend._use_triton_topk(None, 2048) is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
