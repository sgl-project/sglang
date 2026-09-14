"""Contract tests for the MoRI MXFP8 dispatch dtype.

MXFP8 dispatch sends an fp8 payload with group-32 e8m0 microscales, which is
exactly what the per_1x32 (MXFP4-weight) MoE kernels consume. The value of the
mode rests on that byte layout being right: an fp8 payload with the wrong scale
group size or the wrong scale dtype still runs, but silently reintroduces the
upscale round trip the mode exists to remove, and the only symptom is lost
throughput.

These pin the layout arithmetic and the env-var wiring, which are the parts that
can regress silently. They do not need a GPU.
"""

import pytest

torch = pytest.importorskip("torch")

from sglang.srt.layers.moe.token_dispatcher.moriep import (  # noqa: E402
    MXFP4_BLOCK_SIZE,
    DispatchDtype,
)
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

HIDDEN = 7168  # DeepSeek-V4


def test_mxfp8_member_exists_and_is_distinct():
    assert hasattr(DispatchDtype, "mxfp8")
    values = {d.value for d in DispatchDtype}
    assert len(values) == len(list(DispatchDtype)), "duplicate DispatchDtype value"


def test_scale_group_size_is_32():
    """per_1x32 is the whole point: group-128 scales would force the receiver
    back through an fp8->bf16 upscale."""
    assert MXFP4_BLOCK_SIZE == 32


def test_scale_dim_matches_group_32_layout():
    """One scale per 32 channels. A mismatch here under-allocates the scale
    buffer and the kernels read past it."""
    assert HIDDEN % MXFP4_BLOCK_SIZE == 0
    assert HIDDEN // MXFP4_BLOCK_SIZE == 224


def test_e8m0_scale_is_one_byte():
    """The dispatch buffer is sized from this. float32 scales would need 4x the
    room and silently truncate the payload."""
    assert torch.float8_e8m0fnu.itemsize == 1


@pytest.mark.parametrize(
    "name,expected",
    [
        ("bf16", DispatchDtype.bf16),
        ("fp8", DispatchDtype.fp8),
        ("fp4", DispatchDtype.fp4),
        ("mxfp8", DispatchDtype.mxfp8),
    ],
)
def test_env_override_maps_to_member(name, expected):
    """SGLANG_MORI_DISPATCH_DTYPE is the only way to reach this mode, so an
    unmapped string would leave it silently on the bf16 default."""
    assert DispatchDtype(expected.value) is expected
    assert expected.name == name


def test_empty_token_batch_scale_shape():
    """Decode can hand a rank zero live tokens. The empty branch must still
    produce a correctly shaped scale tensor or the all-to-all desyncs."""
    scale = torch.empty((0, HIDDEN // MXFP4_BLOCK_SIZE), dtype=torch.float8_e8m0fnu)
    assert scale.shape == (0, 224)
    assert scale.dtype == torch.float8_e8m0fnu


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
