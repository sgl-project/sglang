import pytest

from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    FINISH_LENGTH,
    FINISH_MATCHED_TOKEN,
)
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    _spec_worker_finish_flags,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    "reason,expected",
    [
        (FINISH_MATCHED_TOKEN(matched=2), (True, True)),
        (FINISH_LENGTH(length=512), (False, True)),
        (FINISH_ABORT("cancelled"), (False, False)),
    ],
)
def test_spec_worker_finish_flags(reason, expected):
    assert _spec_worker_finish_flags(reason) == expected
