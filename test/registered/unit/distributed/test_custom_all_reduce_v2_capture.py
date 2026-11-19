from unittest import mock

import pytest
import torch

from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@pytest.fixture(autouse=True)
def _not_capturing():
    # capture() checks the stream on exit; no GPU is needed for that here.
    with mock.patch.object(
        torch.cuda, "is_current_stream_capturing", return_value=False
    ):
        yield


def _bare_communicator(*, disabled):
    # Skip __init__: it needs a process group and a symmetric workspace.
    comm = CustomAllReduceV2.__new__(CustomAllReduceV2)
    comm.disabled = disabled
    comm._in_capture = False
    comm._pending_graph_inputs = []
    # close() (run from __del__) reads these
    comm._hier = comm._ring = comm._hier_ring = None
    return comm


def test_capture_builds_communicator_before_capturing():
    comm = _bare_communicator(disabled=False)
    calls = []
    with mock.patch.object(
        CustomAllReduceV2,
        "obj",
        new_callable=mock.PropertyMock,
        side_effect=lambda: calls.append(comm._in_capture),
    ):
        with comm.capture():
            pass
    # Built once, before any capture, even though nothing inside used it.
    assert calls == [False]


def test_disabled_capture_does_not_build_communicator():
    comm = _bare_communicator(disabled=True)
    with mock.patch.object(
        CustomAllReduceV2, "obj", new_callable=mock.PropertyMock
    ) as obj:
        with comm.capture():
            pass
    obj.assert_not_called()
