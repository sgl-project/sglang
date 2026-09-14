from collections import defaultdict, deque
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import torch

from sglang.srt.distributed.parallel_state import (
    GroupCoordinator,
    TensorMetadata,
)
from sglang.srt.managers.scheduler_pp_mixin import (
    SchedulerPPMixin,
    _allocate_distinct_cuda_stream,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class FakeWork:
    def __init__(self):
        self.wait_count = 0

    def wait(self):
        self.wait_count += 1


class FakeStream:
    def __init__(self, stream_id):
        self.cuda_stream = stream_id


class FakeEvent:
    def __init__(self):
        self.recorded_stream = None

    def record(self, stream):
        self.recorded_stream = stream


def test_irecv_tensor_dict_defers_payload_wait():
    coordinator = object.__new__(GroupCoordinator)
    coordinator.world_size = 2
    coordinator.rank_in_group = 1
    coordinator.ranks = [0, 1]
    coordinator.device_group = "device-group"
    coordinator.cpu_group = "cpu-group"
    coordinator.recv_object = Mock(
        return_value=[
            (
                "hidden_states",
                TensorMetadata(torch.device("cpu"), torch.float32, (2, 3)),
            ),
            ("kind", "proxy"),
        ]
    )
    work = FakeWork()

    with (
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.irecv", return_value=work) as irecv,
    ):
        tensors, works, postprocess = coordinator.irecv_tensor_dict()

    assert tensors["hidden_states"].shape == (2, 3)
    assert tensors["kind"] == "proxy"
    assert len(works) == 1
    assert works[0].work is work
    assert works[0].payload is tensors["hidden_states"]
    assert postprocess == []
    assert work.wait_count == 0
    irecv.assert_called_once_with(tensors["hidden_states"], src=0, group="cpu-group")


def test_graph_proxy_send_records_forward_reuse_fence():
    work = FakeWork()
    comm_stream = FakeStream(4)
    scheduler = SimpleNamespace(
        pp_comm_overlap=True,
        pp_comm_stream_ctx=nullcontext(),
        pp_comm_stream=comm_stream,
        pp_send_done_event=None,
        device_module=SimpleNamespace(Event=FakeEvent),
    )
    works = [SimpleNamespace(work=work)]

    SchedulerPPMixin._pp_commit_comm_work(scheduler, works, fence_next_forward=True)

    assert work.wait_count == 1
    assert works == []
    assert scheduler.pp_send_done_event.recorded_stream is comm_stream


def test_forward_waits_for_graph_send_and_proxy_receive():
    schedule_stream = FakeStream(1)
    send_done_event = object()
    recv_event = object()
    forward_stream = Mock()
    scheduler = SimpleNamespace(
        schedule_stream=schedule_stream,
        forward_stream=forward_stream,
        pp_send_done_event=send_done_event,
        pp_proxy_recv_event=recv_event,
    )

    SchedulerPPMixin._pp_wait_forward_dependencies(scheduler)

    forward_stream.wait_stream.assert_called_once_with(schedule_stream)
    assert forward_stream.wait_event.call_args_list == [
        call(send_done_event),
        call(recv_event),
    ]
    assert scheduler.pp_send_done_event is None
    assert scheduler.pp_proxy_recv_event is None


def test_inbox_restores_original_receive_event():
    recv_event = object()
    tensor_dict = {"__msg_type__": "output", "value": torch.arange(2)}
    scheduler = SimpleNamespace(
        _pp_tensor_dict_inbox=defaultdict(
            deque, {"output": deque([(tensor_dict, recv_event)])}
        ),
        _pp_last_recv_event=None,
    )

    received = SchedulerPPMixin._pp_recv_typed_dict(scheduler, "output")

    assert received is tensor_dict
    assert scheduler._pp_last_recv_event is recv_event


def test_pp_comm_stream_avoids_schedule_forward_and_copy_streams():
    candidates = iter([FakeStream(1), FakeStream(2), FakeStream(3), FakeStream(4)])
    device_module = SimpleNamespace(Stream=lambda priority=0: next(candidates))

    stream = _allocate_distinct_cuda_stream(
        device_module, (FakeStream(1), FakeStream(2), FakeStream(3))
    )

    assert stream.cuda_stream == 4


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
