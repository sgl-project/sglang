"""A bind that moves to another port must not leave its clients on the old one.

Two servers starting at once can settle on the same scheduler port, and the one
that loses the bind must either fail or carry its clients across to the port it
really got.
"""

import pytest
import zmq

from sglang.multimodal_gen.runtime.launch_server import _republish_scheduler_endpoints
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.common import (
    PORT_DRIFT_STRIDE,
    get_zmq_socket,
    is_port_available,
)


@pytest.fixture
def context():
    context = zmq.Context(io_threads=1)
    yield context
    context.destroy(linger=0)


@pytest.fixture
def port():
    """A port whose drift target is free too, so the stride is what is asserted on."""
    for candidate in range(5600, 5700):
        if is_port_available(candidate) and is_port_available(
            candidate + PORT_DRIFT_STRIDE
        ):
            return candidate
    pytest.skip("no free port pair available")


def _bind(context, port, **kwargs):
    # Two attempts: enough to drift once, few enough to keep the retry sleep cheap.
    return get_zmq_socket(
        context,
        zmq.ROUTER,
        f"tcp://127.0.0.1:{port}",
        bind=True,
        max_bind_retries=2,
        **kwargs,
    )


def test_taken_port_raises_by_default(context, port):
    _holder, endpoint = _bind(context, port)
    assert endpoint == f"tcp://127.0.0.1:{port}"

    with pytest.raises(zmq.ZMQError):
        _bind(context, port)


def test_taken_port_drifts_when_allowed(context, port):
    _holder, _ = _bind(context, port)

    _drifted, endpoint = _bind(context, port, allow_port_drift=True)

    assert endpoint == f"tcp://127.0.0.1:{port + PORT_DRIFT_STRIDE}"


def _server_args(num_gpus, dp_size, scheduler_ports):
    # __new__ skips __post_init__, which would resolve a model; see test_dp_routing.
    server_args = ServerArgs.__new__(ServerArgs)
    server_args.host = "localhost"
    server_args.num_gpus = num_gpus
    server_args.dp_size = dp_size
    server_args.scheduler_ports = scheduler_ports
    return server_args


def test_republish_records_the_bound_port():
    server_args = _server_args(num_gpus=2, dp_size=2, scheduler_ports=[5600, 5601])

    _republish_scheduler_endpoints(
        server_args=server_args,
        scheduler_infos=[
            {"status": "ready", "scheduler_endpoint": "tcp://127.0.0.1:5642"},
            {"status": "ready", "scheduler_endpoint": "tcp://127.0.0.1:5601"},
        ],
        rank_offset=0,
    )

    assert server_args.scheduler_ports == [5642, 5601]
    assert server_args.scheduler_endpoints == [
        "tcp://127.0.0.1:5642",
        "tcp://127.0.0.1:5601",
    ]


def test_republish_skips_ranks_that_do_not_bind():
    server_args = _server_args(num_gpus=2, dp_size=1, scheduler_ports=[5600])

    # Only a replica's first rank binds the ingress; the rest report None.
    _republish_scheduler_endpoints(
        server_args=server_args,
        scheduler_infos=[
            {"status": "ready", "scheduler_endpoint": "tcp://127.0.0.1:5642"},
            {"status": "ready", "scheduler_endpoint": None},
        ],
        rank_offset=0,
    )

    assert server_args.scheduler_ports == [5642]
