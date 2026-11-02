"""Validate the serving override for the newly measured PA partition count."""

import pytest

from sglang.srt.layers.attention.aiter_utils import get_flydsl_mimo_num_partitions


@pytest.mark.parametrize("value", [8, 16, 24, 32, 64])
def test_supported_partition_override(monkeypatch, value):
    monkeypatch.setenv("SGLANG_FLYDSL_PA_NUM_PARTITIONS", str(value))
    assert get_flydsl_mimo_num_partitions() == value


@pytest.mark.parametrize("value", ["0", "1", "-1", "48", "65", "16.0", "invalid"])
def test_invalid_partition_override_is_rejected(monkeypatch, value):
    monkeypatch.setenv("SGLANG_FLYDSL_PA_NUM_PARTITIONS", value)
    with pytest.raises(ValueError):
        get_flydsl_mimo_num_partitions()
