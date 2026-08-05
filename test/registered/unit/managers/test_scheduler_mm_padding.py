from array import array
from types import SimpleNamespace

import pytest

from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_mm_padding_keeps_streaming_fill_ids_coherent() -> None:
    audio_token = 200038
    audio_pad = 1 << 50
    prefix = [10, 11, 12]
    request = SimpleNamespace(
        origin_input_ids=array("q", prefix + [audio_token]),
        full_untruncated_fill_ids=array("q", prefix + [audio_token]),
        session=SimpleNamespace(streaming=True),
    )
    received = SimpleNamespace(input_ids=array("q", [audio_token]))
    mm_inputs = SimpleNamespace(padded_input_ids=[audio_pad])

    assert Scheduler._try_apply_padded_mm_input_ids(received, request, mm_inputs)

    expected = prefix + [audio_pad]
    assert list(request.origin_input_ids) == expected
    assert list(request.full_untruncated_fill_ids) == expected


def test_mm_padding_invalidates_incoherent_streaming_fill_ids() -> None:
    request = SimpleNamespace(
        origin_input_ids=array("q", [10, 200038]),
        full_untruncated_fill_ids=array("q", [10]),
        session=SimpleNamespace(streaming=True),
    )
    received = SimpleNamespace(input_ids=array("q", [200038]))
    mm_inputs = SimpleNamespace(padded_input_ids=[1 << 50])

    assert Scheduler._try_apply_padded_mm_input_ids(received, request, mm_inputs)

    assert request.full_untruncated_fill_ids == array("q")


def test_mm_padding_keeps_legacy_non_streaming_fill_ids_untouched() -> None:
    request = SimpleNamespace(
        origin_input_ids=array("q", [10, 200038]),
        full_untruncated_fill_ids=array("q", [10, 200038]),
        session=None,
    )
    received = SimpleNamespace(input_ids=array("q", [200038]))
    mm_inputs = SimpleNamespace(padded_input_ids=[1 << 50])

    assert Scheduler._try_apply_padded_mm_input_ids(received, request, mm_inputs)

    assert list(request.origin_input_ids) == [10, 1 << 50]
    assert list(request.full_untruncated_fill_ids) == [10, 200038]


def test_model_mm_padding_expands_only_carried_streaming_suffix() -> None:
    prefix = [10, 11]
    request = SimpleNamespace(
        origin_input_ids=array("q", prefix + [200038]),
        full_untruncated_fill_ids=array("q", prefix + [200038]),
    )

    Scheduler._set_padded_mm_input_ids(
        request,
        prefix + [1 << 50, 1 << 50],
        prefix_len=len(prefix),
    )

    expected = prefix + [1 << 50, 1 << 50]
    assert list(request.origin_input_ids) == expected
    assert list(request.full_untruncated_fill_ids) == expected


def test_mm_padding_with_zero_prefix_replaces_full_prompt() -> None:
    request = SimpleNamespace(
        origin_input_ids=array("q", [200038]),
        full_untruncated_fill_ids=array("q", [200038]),
    )

    Scheduler._set_padded_mm_input_ids(request, [1 << 50, 1 << 50], prefix_len=0)

    assert list(request.origin_input_ids) == [1 << 50, 1 << 50]
    assert list(request.full_untruncated_fill_ids) == [1 << 50, 1 << 50]


def test_mm_padding_preserves_empty_fill_cache() -> None:
    request = SimpleNamespace(
        origin_input_ids=array("q", [10, 200038]),
        full_untruncated_fill_ids=array("q"),
    )

    Scheduler._set_padded_mm_input_ids(request, [10, 1 << 50], prefix_len=1)

    assert request.full_untruncated_fill_ids == array("q")


def test_mm_padding_rejects_malformed_prefix_length() -> None:
    request = SimpleNamespace(
        origin_input_ids=array("q", [10, 11, 200038]),
        full_untruncated_fill_ids=array("q", [10, 11, 200038]),
    )

    with pytest.raises(ValueError, match="Invalid multimodal padding prefix"):
        Scheduler._set_padded_mm_input_ids(request, [10], prefix_len=2)


def test_mm_padding_rejects_fallback_that_rewrites_prefix() -> None:
    request = SimpleNamespace(
        origin_input_ids=array("q", [10, 200038]),
        full_untruncated_fill_ids=array("q", [10, 200038]),
    )

    with pytest.raises(ValueError, match="changed tokens before"):
        Scheduler._set_padded_mm_input_ids(request, [99, 1 << 50], prefix_len=1)
