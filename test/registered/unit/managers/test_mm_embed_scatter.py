import weakref

import pytest
import torch

from sglang.srt.managers import mm_utils
from sglang.srt.managers.mm_utils import _scatter_mm_embedding
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu-intel")

NUM_TOKENS = 64


def _make_mask(pattern: str) -> torch.Tensor:
    mask = torch.zeros(NUM_TOKENS, dtype=torch.bool)
    if pattern == "interleaved":
        mask[::3] = True
    elif pattern == "blocks":
        mask[5:20] = True
        mask[40:41] = True
    elif pattern == "all_true":
        mask[:] = True
    return mask.unsqueeze(-1)


@pytest.mark.parametrize("width", [8, 24])
@pytest.mark.parametrize("src_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "mask_pattern", ["interleaved", "blocks", "all_true", "all_false"]
)
def test_scatter_matches_masked_scatter_bitwise(width, src_dtype, mask_pattern):
    """The row-index mm embedding merge must stay bitwise identical to
    masked_scatter_ semantics, whose internal transients it avoids."""
    torch.manual_seed(0)
    mask = _make_mask(mask_pattern)
    dest = torch.randn(NUM_TOKENS, width).to(torch.bfloat16)
    src = torch.randn(int(mask.sum()), width, dtype=src_dtype)

    expected = dest.clone()
    expected.masked_scatter_(mask.expand_as(expected), src.to(expected.dtype))

    actual = dest.clone()
    _scatter_mm_embedding(dest=actual, mask=mask, src=[src])
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("staging_bytes", [None, 40])
@pytest.mark.parametrize("mask_pattern", ["blocks", "all_false"])
def test_scatter_of_row_pieces_matches_single_source(
    mask_pattern, staging_bytes, monkeypatch
):
    """Scattering row pieces, including empty ones, must equal scattering the
    concatenated source, whether pieces are staged together or copied alone."""
    if staging_bytes is not None:
        # 40 B is 2.5 bf16 rows of width 8: mixes staged runs with lone pieces.
        monkeypatch.setattr(mm_utils, "_SCATTER_STAGING_BYTES", staging_bytes)
    torch.manual_seed(0)
    mask = _make_mask(mask_pattern)
    src = torch.randn(int(mask.sum()), 8)
    pieces = list(torch.tensor_split(src, [0, 3, 3, 7]))

    expected = torch.randn(NUM_TOKENS, 8).to(torch.bfloat16)
    actual = expected.clone()
    _scatter_mm_embedding(dest=expected, mask=mask, src=[src])
    _scatter_mm_embedding(dest=actual, mask=mask, src=pieces)
    assert torch.equal(actual, expected)


def test_scatter_frees_each_staging_copy_before_the_next(monkeypatch):
    """At most one staging copy may be alive, so the transient stays within
    _SCATTER_STAGING_BYTES rather than twice that."""
    # 32 B is 2 bf16 rows of width 8, so 1-row pieces stage in pairs.
    monkeypatch.setattr(mm_utils, "_SCATTER_STAGING_BYTES", 32)
    staged = []
    cast_run = mm_utils._cast_run

    def tracking_cast_run(run, dest):
        assert all(ref() is None for ref in staged), "previous staging copy alive"
        copy = cast_run(run=run, dest=dest)
        if len(run) > 1:
            staged.append(weakref.ref(copy))
        return copy

    monkeypatch.setattr(mm_utils, "_cast_run", tracking_cast_run)
    mask = _make_mask("interleaved")
    src = torch.randn(int(mask.sum()), 8).to(torch.bfloat16)

    expected = torch.randn(NUM_TOKENS, 8).to(torch.bfloat16)
    actual = expected.clone()
    _scatter_mm_embedding(dest=expected, mask=mask, src=[src])
    _scatter_mm_embedding(dest=actual, mask=mask, src=list(src.split(1)))
    assert len(staged) > 1
    assert torch.equal(actual, expected)


def test_scatter_row_count_mismatch_fails_loud():
    """A mask/src row-count mismatch must raise, not silently corrupt rows."""
    dest = torch.zeros(8, 4)
    src_short_mask = _make_mask("all_false")[:8]
    src_short_mask[1] = True
    with pytest.raises((RuntimeError, IndexError)):
        _scatter_mm_embedding(
            dest=dest, mask=src_short_mask, src=[torch.ones(2, 4), torch.ones(1, 4)]
        )
    mask_heavy = src_short_mask.clone()
    mask_heavy[2:6] = True
    with pytest.raises((RuntimeError, IndexError)):
        _scatter_mm_embedding(dest=dest, mask=mask_heavy, src=[torch.ones(1, 4)])


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
