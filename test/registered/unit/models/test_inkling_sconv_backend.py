from types import SimpleNamespace

from sglang.srt.layers.attention.linear.inkling_sconv_backend import (
    InklingShortConvHybridAttnBackend,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_forward_metadata_delegates_read_and_write():
    initial_metadata = object()
    full_attn_backend = SimpleNamespace(forward_metadata=initial_metadata)
    backend = object.__new__(InklingShortConvHybridAttnBackend)
    backend.full_attn_backend = full_attn_backend

    assert backend.forward_metadata is initial_metadata

    new_metadata = object()
    backend.forward_metadata = new_metadata

    assert full_attn_backend.forward_metadata is new_metadata
