import logging
from unittest.mock import Mock, patch

import torch

from sglang.srt.utils.bounded_telemetry import (
    BoundedTelemetryLogger,
    format_telemetry_fields,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_format_telemetry_fields_uses_tensor_metadata_only():
    tensor = torch.empty((2, 3), dtype=torch.bfloat16)

    assert format_telemetry_fields(
        rows=2, total_recv=tensor, enabled=True, missing=None
    ) == (
        "rows=2 total_recv=tensor(shape=[2,3],dtype=torch.bfloat16,device=cpu) "
        "enabled=true missing=none"
    )


def test_bounded_telemetry_logs_each_key_once_and_honors_limit():
    logger = Mock(spec=logging.Logger)
    telemetry = BoundedTelemetryLogger(
        logger, "[TEST_TELEMETRY]", enabled=True, max_events=2
    )

    with patch("sglang.srt.utils.bounded_telemetry.is_rank_zero", return_value=True):
        assert telemetry.log("dispatch-0", "dispatch", rows=4)
        assert not telemetry.log("dispatch-0", "dispatch", rows=8)
        assert telemetry.log("combine-0", "combine", rows=4)
        assert not telemetry.log("dispatch-1", "dispatch", rows=2)

    assert logger.info.call_count == 2
    logger.info.assert_any_call("%s %s", "[TEST_TELEMETRY]", "event=dispatch rows=4")


def test_bounded_telemetry_is_disabled_off_rank():
    logger = Mock(spec=logging.Logger)
    telemetry = BoundedTelemetryLogger(logger, "[TEST_TELEMETRY]", enabled=True)

    with patch("sglang.srt.utils.bounded_telemetry.is_rank_zero", return_value=False):
        assert not telemetry.log("dispatch", "dispatch", rows=4)

    logger.info.assert_not_called()


def test_bounded_telemetry_can_log_off_rank_when_rank_zero_only_is_false():
    logger = Mock(spec=logging.Logger)
    telemetry = BoundedTelemetryLogger(
        logger,
        "[TEST_TELEMETRY]",
        enabled=True,
        rank_zero_only=False,
    )

    with patch("sglang.srt.utils.bounded_telemetry.is_rank_zero", return_value=False):
        assert telemetry.log("dispatch", "dispatch", rows=4)

    logger.info.assert_called_once_with(
        "%s %s", "[TEST_TELEMETRY]", "event=dispatch rows=4"
    )
