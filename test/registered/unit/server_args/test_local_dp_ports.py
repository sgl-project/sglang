import tempfile
from contextlib import ExitStack
from unittest.mock import patch

import pytest
import zmq

from sglang.srt.environ import envs
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    "local_dp_ipc,rust_server,overrides",
    [
        (False, False, {}),
        (True, True, {}),
        (True, False, {"elastic_ep_backend": "mooncake"}),
        (True, False, {"dist_init_addr": "127.0.0.1:25000"}),
        (True, False, {"nnodes": 2, "dist_init_addr": "127.0.0.1:25000"}),
    ],
)
def test_independent_dp_peers_keep_tcp_endpoints(local_dp_ipc, rust_server, overrides):
    args = ServerArgs(model_path="dummy", tp_size=2, attn_dp_size=2, **overrides)
    with (
        envs.SGLANG_RUST_SERVER.override(rust_server),
        patch("sglang.srt.server_args.wait_port_available"),
    ):
        plan = PortArgs.init_new(args, local_dp_ipc=local_dp_ipc)
        rank = plan.for_dp_worker(args, 1, None if rust_server else [31001, 31002])
    assert plan.tokenizer_ipc_name.startswith("tcp://")
    assert rank.tokenizer_ipc_name == plan.tokenizer_ipc_name
    assert rank.scheduler_input_ipc_name.startswith("tcp://127.0.0.1:")
    assert rank.instance_id == plan.instance_id
    assert rank.nccl_port == plan.nccl_port


def test_adjacent_attention_dp_servers():
    with (
        tempfile.TemporaryDirectory() as tmp,
        patch.object(tempfile, "tempdir", tmp),
        envs.SGLANG_RUST_SERVER.override(False),
        zmq.Context() as context,
        ExitStack() as sockets,
    ):
        # Both parents finish their port checks before either starts its children.
        args = [
            ServerArgs(model_path="dummy", port=p, tp_size=2, attn_dp_size=2)
            for p in (5000, 5001)
        ]
        plans = [PortArgs.init_new(a, local_dp_ipc=True) for a in args]
        for plan in plans:
            for endpoint in (
                plan.tokenizer_ipc_name,
                plan.detokenizer_ipc_name,
                plan.scheduler_input_ipc_name,
                plan.metrics_ipc_name,
            ):
                sockets.enter_context(context.socket(zmq.PULL)).bind(endpoint)

        for server_args, plan in zip(args, plans):
            senders = [
                sockets.enter_context(context.socket(zmq.PUSH)) for _ in range(2)
            ]
            worker_ports = [s.bind_to_random_port("tcp://127.0.0.1") for s in senders]
            for rank, sender in enumerate(senders):
                worker = plan.for_dp_worker(server_args, rank, worker_ports)
                assert worker.tokenizer_ipc_name == plan.tokenizer_ipc_name
                assert worker.detokenizer_ipc_name == plan.detokenizer_ipc_name
                assert worker.metrics_ipc_name == plan.metrics_ipc_name
                assert worker.nccl_port == plan.nccl_port
                assert worker.instance_id == plan.instance_id

                receiver = sockets.enter_context(context.socket(zmq.PULL))
                receiver.connect(worker.scheduler_input_ipc_name)
                assert sender.poll(5000, zmq.POLLOUT)
                sender.send_string(f"{server_args.port}/{rank}")
                assert receiver.poll(5000)
                assert receiver.recv_string() == f"{server_args.port}/{rank}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
