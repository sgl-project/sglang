"""CPU-only tests for compressed-tensors configuration helpers."""

from unittest.mock import patch

import pytest

from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_unsupported_capability_error_is_readable():
    config = CompressedTensorsConfig.__new__(CompressedTensorsConfig)

    with patch("torch.cuda.get_device_capability", return_value=(7, 5)):
        with pytest.raises(RuntimeError) as exc_info:
            config._check_scheme_supported(min_capability=80)

    assert exc_info.value.args == (
        "Quantization scheme is not supported for the current GPU. "
        "Min capability: 80. Current capability: 75.",
    )
