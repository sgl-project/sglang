"""Routing invariants for DP serving.

Generation requests go to exactly one replica, control ops reach every replica,
and a realtime session always lands on the same replica it started on.
"""

import itertools

from sglang.multimodal_gen.runtime.entrypoints.utils import SetLoraReq, ShutdownReq
from sglang.multimodal_gen.runtime.pipelines_core import Req
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.scheduler_client import (
    _CONTROL_REQ_TYPES,
    _merge_fanout_results,
    _select_replica,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


def _req(**kwargs) -> Req:
    return Req(prompt="a pear", **kwargs)


def test_round_robin_covers_every_replica():
    counter = itertools.count()
    picks = [_select_replica([_req()], 3, counter) for _ in range(6)]
    assert picks == [0, 1, 2, 0, 1, 2]


def test_dp1_always_selects_replica_zero():
    counter = itertools.count()
    assert {_select_replica([_req()], 1, counter) for _ in range(5)} == {0}


def test_session_requests_stick_to_one_replica():
    counter = itertools.count()
    session_req = _req(realtime_session_id="session-abc")
    picks = {_select_replica([session_req], 4, counter) for _ in range(8)}
    assert len(picks) == 1
    # and the counter was never consumed by session traffic
    assert _select_replica([_req()], 4, counter) == 0


def test_control_reqs_are_recognized():
    assert isinstance(SetLoraReq(lora_nickname="x", lora_path="y"), _CONTROL_REQ_TYPES)
    assert isinstance(ShutdownReq(), _CONTROL_REQ_TYPES)
    assert not isinstance([_req()], _CONTROL_REQ_TYPES)
    assert not isinstance(_req(), _CONTROL_REQ_TYPES)


def test_fanout_merge_surfaces_the_failing_replica():
    ok = OutputBatch(output=None)
    bad = OutputBatch(error="replica 1 exploded")
    assert _merge_fanout_results([ok, bad]) is bad
    assert _merge_fanout_results([ok, OutputBatch(output=None)]) is ok


def test_scheduler_endpoints_one_per_replica():
    args = ServerArgs.__new__(ServerArgs)
    args.host = "localhost"
    args.dp_size = 3
    args.scheduler_port = 6000
    args.scheduler_ports = None
    assert ServerArgs.scheduler_endpoints.fget(args) == [
        "tcp://127.0.0.1:6000",
        "tcp://127.0.0.1:6001",
        "tcp://127.0.0.1:6002",
    ]
    # settled ports need not be consecutive; the endpoint list follows them
    args.scheduler_ports = [6000, 7005, 7100]
    assert ServerArgs.scheduler_endpoint_for(args, 2) == "tcp://127.0.0.1:7100"
