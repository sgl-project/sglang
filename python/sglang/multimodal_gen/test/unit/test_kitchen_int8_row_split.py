# SPDX-License-Identifier: Apache-2.0
"""Unsplit default and explicit row limits for the Kitchen INT8 backend."""

import importlib.util

import pytest

from sglang.multimodal_gen.runtime.layers.quantization import (
    convrot_int8_comfy_kitchen as kitchen_int8,
)


@pytest.mark.parametrize("limit", [None, "0", "4096", "8192"])
@pytest.mark.parametrize("min_output", [None, "8192", "16384"])
def test_row_split_environment_policy(monkeypatch, limit, min_output):
    for name, value in (
        ("SGLANG_KITCHEN_INT8_MAX_ROWS", limit),
        ("SGLANG_KITCHEN_INT8_MIN_SPLIT_N", min_output),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    # Execute the import-time policy without replacing classes used by other tests.
    spec = importlib.util.spec_from_file_location(
        "_kitchen_int8_policy_test", kitchen_int8.__file__
    )
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    max_rows = int(limit or "0")
    min_width = int(min_output or "8192")
    for rows in (4095, 4096, 4097, 8192, 8193, 32700):
        for width in (5376, 8191, 8192, 16128, 16384):
            expected = (
                max_rows
                if max_rows > 0 and rows > max_rows and width >= min_width
                else None
            )
            assert policy._row_split(rows, width) == expected
