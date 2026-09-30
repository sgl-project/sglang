import pytest

from sglang.srt.mem_cache.unified_cache.swa_retention import retained_swa_ranges
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_retained_swa_ranges_coalesces_checkpoint_and_prompt_windows():
    assert retained_swa_ranges(
        28672,
        33792,
        prompt_boundary=33792,
        window=4096,
        interval=32768,
        page_size=256,
    ) == [(28672, 33792)]


def test_retained_swa_ranges_can_exclude_prompt_boundary():
    assert retained_swa_ranges(
        28672,
        33792,
        prompt_boundary=33792,
        window=4096,
        interval=32768,
        page_size=256,
        include_prompt_boundary=False,
    ) == [(28672, 32768)]


def test_retained_swa_ranges_requires_page_alignment():
    with pytest.raises(ValueError, match="page aligned"):
        retained_swa_ranges(
            1,
            256,
            prompt_boundary=256,
            window=256,
            interval=256,
            page_size=256,
        )
