import types

import pytest

try:
    from sglang.srt.managers.utils import validate_input_length
except ImportError:
    pytest.skip("SGLang dependencies unavailable", allow_module_level=True)


def _make_req(length: int):
    return types.SimpleNamespace(origin_input_ids=list(range(length)))


def test_auto_truncate_reserves_generation_tokens():
    req = _make_req(120)

    err = validate_input_length(req, max_req_input_len=100, allow_auto_truncate=True, reserved_tokens=20)

    assert err is None
    assert len(req.origin_input_ids) == 80


def test_error_when_auto_truncate_disabled():
    req = _make_req(120)

    err = validate_input_length(req, max_req_input_len=100, allow_auto_truncate=False, reserved_tokens=20)

    assert err is not None
    assert len(req.origin_input_ids) == 120
