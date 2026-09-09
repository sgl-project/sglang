import sys
from types import SimpleNamespace

import pytest

from sglang.srt.sampling.watermark import (
    build_watermark_batch_config,
    normalize_watermark_request,
    resolve_watermark_request,
)
from sglang.srt.utils.request_logger import _transform_data_for_logging
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


_DEFAULT_KEY = "0123456789abcdef"
_REQUEST_KEY = "fedcba9876543210"
_REQUEST_FORMS = {
    "omitted": None,
    "disabled": normalize_watermark_request({"enabled": False}),
    "enabled": normalize_watermark_request({"enabled": True}),
    "key": normalize_watermark_request({"key": _REQUEST_KEY}),
}


@pytest.mark.parametrize(
    (
        "server_enabled",
        "default_enabled",
        "enforce_all",
        "request_form",
        "expected_key",
        "expected_enabled",
    ),
    [
        pytest.param(False, False, False, "omitted", None, False, id="off-omitted"),
        pytest.param(False, False, False, "disabled", None, False, id="off-opt-out"),
        pytest.param(False, False, False, "enabled", ValueError, None, id="off-opt-in"),
        pytest.param(False, False, False, "key", ValueError, None, id="off-key"),
        pytest.param(True, False, False, "omitted", None, False, id="opt-in-omitted"),
        pytest.param(True, False, False, "disabled", None, False, id="opt-in-opt-out"),
        pytest.param(
            True, False, False, "enabled", _DEFAULT_KEY, True, id="opt-in-enabled"
        ),
        pytest.param(True, False, False, "key", _REQUEST_KEY, True, id="opt-in-key"),
        pytest.param(
            True, True, False, "omitted", _DEFAULT_KEY, True, id="default-on-omitted"
        ),
        pytest.param(
            True, True, False, "disabled", None, False, id="default-on-opt-out"
        ),
        pytest.param(
            True, True, False, "enabled", _DEFAULT_KEY, True, id="default-on-enabled"
        ),
        pytest.param(True, True, False, "key", _REQUEST_KEY, True, id="default-on-key"),
        pytest.param(
            True, False, True, "omitted", _DEFAULT_KEY, True, id="enforce-omitted"
        ),
        pytest.param(
            True, False, True, "disabled", ValueError, None, id="enforce-opt-out"
        ),
        pytest.param(
            True, False, True, "enabled", _DEFAULT_KEY, True, id="enforce-enabled"
        ),
        pytest.param(True, False, True, "key", _REQUEST_KEY, True, id="enforce-key"),
    ],
)
def test_request_enablement_matrix(
    server_enabled,
    default_enabled,
    enforce_all,
    request_form,
    expected_key,
    expected_enabled,
):
    def resolve():
        return resolve_watermark_request(
            _REQUEST_FORMS[request_form],
            server_enabled=server_enabled,
            default_key=_DEFAULT_KEY,
            default_context_window=4,
            default_enabled=default_enabled,
            enforce_all=enforce_all,
        )

    if expected_key is ValueError:
        with pytest.raises(ValueError):
            resolve()
        return

    key, context_window, enabled = resolve()
    assert key == expected_key
    assert context_window == 4
    assert enabled is expected_enabled


def test_per_request_config_resolution_and_redaction():
    secret = _REQUEST_KEY
    requests = [
        SimpleNamespace(
            sampling_params=SimpleNamespace(
                watermark=normalize_watermark_request(
                    {"key": secret, "context_window": 2}
                )
            )
        ),
        SimpleNamespace(sampling_params=SimpleNamespace(watermark=None)),
        SimpleNamespace(
            sampling_params=SimpleNamespace(
                watermark=normalize_watermark_request({"enabled": False})
            )
        ),
        SimpleNamespace(
            sampling_params=SimpleNamespace(
                watermark=normalize_watermark_request({"enabled": True})
            )
        ),
    ]

    keys, context_windows, enabled = build_watermark_batch_config(
        requests,
        default_key=_DEFAULT_KEY,
        default_context_window=4,
        default_enabled=False,
        enforce_all=False,
        device="cpu",
    )

    assert keys.tolist() == [
        0xFEDCBA9876543210 - (1 << 64),
        0,
        0,
        0x0123456789ABCDEF,
    ]
    assert context_windows.tolist() == [2, 4, 4, 4]
    assert enabled.tolist() == [True, False, False, True]

    keys, context_windows, enabled = build_watermark_batch_config(
        requests[:3],
        default_key=_DEFAULT_KEY,
        default_context_window=4,
        default_enabled=True,
        enforce_all=False,
        device="cpu",
    )
    assert keys.tolist() == [
        0xFEDCBA9876543210 - (1 << 64),
        0x0123456789ABCDEF,
        0,
    ]
    assert context_windows.tolist() == [2, 4, 4]
    assert enabled.tolist() == [True, True, False]
    assert secret not in repr(requests[0].sampling_params.watermark)
    logged = _transform_data_for_logging(
        {
            "sampling_params": {"watermark": {"key": secret, "context_window": 2}},
            "watermark_key": secret,
            "internal_states": [{"watermark_key": secret}],
        }
    )
    assert logged["sampling_params"]["watermark"]["key"] == "<redacted>"
    assert logged["watermark_key"] == "<redacted>"
    assert logged["internal_states"][0]["watermark_key"] == "<redacted>"

    with pytest.raises(ValueError, match="unknown fields"):
        normalize_watermark_request({"key": secret, "provider": "textseal"})
    with pytest.raises(ValueError, match="enabled must be a boolean"):
        normalize_watermark_request({"enabled": 1})
    with pytest.raises(ValueError, match="must set enabled or key"):
        resolve_watermark_request(
            normalize_watermark_request({"context_window": 2}),
            server_enabled=True,
            default_key=_DEFAULT_KEY,
            default_context_window=4,
            default_enabled=True,
            enforce_all=False,
        )
    with pytest.raises(ValueError, match="cannot be combined"):
        resolve_watermark_request(
            normalize_watermark_request({"enabled": False, "key": secret}),
            server_enabled=True,
            default_key=_DEFAULT_KEY,
            default_context_window=4,
            default_enabled=False,
            enforce_all=False,
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
