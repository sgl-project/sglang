from types import SimpleNamespace

from sglang._flashinfer_nsys_patch import (
    _PATCHED_SPEC_NAME,
    _make_no_timeout_spec,
)


def test_make_no_timeout_spec_preserves_existing_flags():
    original = SimpleNamespace(
        name="mnnvl_moe_alltoall",
        extra_cuda_cflags=["-DENABLE_BF16"],
    )

    patched = _make_no_timeout_spec(lambda: original)

    assert patched.name == _PATCHED_SPEC_NAME
    assert patched.extra_cuda_cflags == ["-DENABLE_BF16", "-DDISABLE_TIMEOUT=1"]


def test_make_no_timeout_spec_does_not_duplicate_flag():
    original = SimpleNamespace(
        name="mnnvl_moe_alltoall",
        extra_cuda_cflags=["-DDISABLE_TIMEOUT=1"],
    )

    patched = _make_no_timeout_spec(lambda: original)

    assert patched.extra_cuda_cflags == ["-DDISABLE_TIMEOUT=1"]
