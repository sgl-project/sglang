import sys
from types import SimpleNamespace

import pytest

from sglang.srt.sampling.watermark import (
    build_watermark_batch_config,
    normalize_watermark_request,
)
from sglang.srt.utils.request_logger import _transform_data_for_logging
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def test_per_request_config_resolution_and_redaction():
    secret = "fedcba9876543210"
    requests = [
        SimpleNamespace(
            sampling_params=SimpleNamespace(
                watermark=normalize_watermark_request(
                    {"key": secret, "context_window": 2}
                )
            )
        ),
        SimpleNamespace(sampling_params=SimpleNamespace(watermark=None)),
    ]

    keys, context_windows, enabled = build_watermark_batch_config(
        requests,
        default_key="0123456789abcdef",
        default_context_window=4,
        device="cpu",
    )

    assert keys.tolist() == [0xFEDCBA9876543210 - (1 << 64), 0x0123456789ABCDEF]
    assert context_windows.tolist() == [2, 4]
    assert enabled.tolist() == [True, True]

    keys, context_windows, enabled = build_watermark_batch_config(
        requests,
        default_key=None,
        default_context_window=4,
        device="cpu",
    )
    assert keys.tolist() == [0xFEDCBA9876543210 - (1 << 64), 0]
    assert context_windows.tolist() == [2, 4]
    assert enabled.tolist() == [True, False]
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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
