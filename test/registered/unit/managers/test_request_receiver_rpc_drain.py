"""The scheduler's rpc channel must survive rust-server mode.

Rank 0 has two input channels: the tokenizer channel (which the embedded Rust
server replaces with an in-process ring) and the rpc channel (a zmq DEALER pair
with the offline ``Engine``, which Rust has no equivalent for). A regression
made ``_pull_raw_reqs`` return as soon as it had drained the ring, so the rpc
socket was created and wired up but never read, and the matching reply branch in
``process_input_requests`` was shadowed by the rust branch.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import zmq

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.distributed.parallel_state_wrapper import ParallelState  # noqa: E402
from sglang.srt.managers.io_struct import (  # noqa: E402
    RpcReqInput,
    RpcReqOutput,
    msgpack_encode,
)
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.managers.scheduler_components.request_receiver import (  # noqa: E402
    SchedulerRequestReceiver,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeSocket:
    """A zmq socket holding a fixed queue, then empty.

    Yields each queued object once and then raises ``zmq.Again``, which is what a
    real socket does under ``zmq.NOBLOCK`` with nothing queued. Both receive
    methods are implemented so the test holds under either IPC encoding
    (``sock_recv`` picks between them on ``SGLANG_USE_PICKLE_IPC``).

    A fake rather than a real socket pair on purpose: a freshly connected
    ``inproc://`` / ``ipc://`` pair races an immediate non-blocking ``recv``, so
    a real pair would make this test flaky.
    """

    def __init__(self, objs=()):
        self.objs = list(objs)

    def _pop(self):
        if not self.objs:
            raise zmq.Again()
        return self.objs.pop(0)

    def recv_pyobj(self, flags=0):
        return self._pop()

    def recv(self, flags=0):
        return msgpack_encode(self._pop())


class _FakeRing:
    """Stand-in for ``RustServer`` as the receiver's ingress source.

    ``RustServer.drain`` returns already-decoded request objects, so what they
    are is the ring's business, not the receiver's — hence plain sentinels.
    """

    def __init__(self, objs=()):
        self.objs = list(objs)
        self.drain_calls = []

    def drain(self, max_recv):
        self.drain_calls.append(max_recv)
        drained, self.objs = self.objs, []
        return drained


def _make_receiver(*, recv_from_tokenizer, recv_from_rpc) -> SchedulerRequestReceiver:
    group = SimpleNamespace(rank=0, ranks=[0], cpu_group=object())
    return SchedulerRequestReceiver(
        recv_from_tokenizer=recv_from_tokenizer,
        recv_from_rpc=recv_from_rpc,
        recv_skipper=None,
        input_blocker=None,
        mm_receiver=None,
        # All-defaults: pp_rank / attn_tp_rank / attn_cp_rank are 0, which is the
        # rank-zero branch of _pull_raw_reqs — the only one that owns sockets.
        ps=ParallelState.trivial(),
        tp_group=group,
        tp_cpu_group=group,
        attn_tp_group=group,
        attn_tp_cpu_group=group,
        attn_cp_group=group,
        attn_cp_cpu_group=group,
        world_group=group,
        server_args=SimpleNamespace(
            enable_dp_attention=False,
            enable_dp_attention_local_control_broadcast=False,
        ),
        model_config=SimpleNamespace(is_multimodal=False),
        max_recv_per_poll=-1,
        stream_output=lambda *args, **kwargs: None,
        get_last_batch=lambda: None,
    )


class TestRequestReceiverRpcDrain(unittest.TestCase):
    def test_rust_mode_drains_rpc_socket(self):
        """Rust mode must read the rpc socket, not just the ingress ring.

        Draining only the ring leaves rpc requests queued forever, and
        ``collective_rpc`` blocks on a reply that is never sent.
        """
        ring_req = object()
        rpc_req = RpcReqInput(method="save_remote_model", parameters={"url": "s3://x"})
        receiver = _make_receiver(
            recv_from_tokenizer=_FakeRing([ring_req]),
            recv_from_rpc=_FakeSocket([rpc_req]),
        )

        with patch(
            "sglang.srt.managers.scheduler_components.request_receiver."
            "envs.SGLANG_RUST_SERVER.get",
            return_value=True,
        ):
            recv_reqs = receiver._pull_raw_reqs()

        self.assertEqual(len(recv_reqs), 2)
        self.assertIs(recv_reqs[0], ring_req)
        self.assertEqual(recv_reqs[1], rpc_req)

    def test_rust_mode_does_not_read_the_zmq_tokenizer_socket(self):
        """The ring replaces the tokenizer socket; both must not be drained.

        Guards the branch that makes the ring an *alternative* to the tokenizer
        socket rather than an addition — reading both would double-admit work in
        any deployment that still has a producer on the zmq side.
        """
        ring = _FakeRing([object()])
        receiver = _make_receiver(recv_from_tokenizer=ring, recv_from_rpc=_FakeSocket())

        with patch(
            "sglang.srt.managers.scheduler_components.request_receiver."
            "envs.SGLANG_RUST_SERVER.get",
            return_value=True,
        ):
            receiver._pull_raw_reqs()

        # -1 (unlimited) is forwarded verbatim; RustServer.drain owns the fallback.
        self.assertEqual(ring.drain_calls, [-1])

    def test_zmq_mode_still_drains_both_sockets(self):
        """The non-rust path keeps draining tokenizer *and* rpc into one list.

        The rust branch sits directly above these two loops, so restructuring it
        can silently drop either one.
        """
        tokenizer_req = RpcReqInput(method="tokenizer_channel_sentinel")
        rpc_req = RpcReqInput(
            method="save_sharded_model", parameters={"path": "/tmp/x"}
        )
        receiver = _make_receiver(
            recv_from_tokenizer=_FakeSocket([tokenizer_req]),
            recv_from_rpc=_FakeSocket([rpc_req]),
        )

        with patch(
            "sglang.srt.managers.scheduler_components.request_receiver."
            "envs.SGLANG_RUST_SERVER.get",
            return_value=False,
        ):
            recv_reqs = receiver._pull_raw_reqs()

        self.assertEqual(recv_reqs, [tokenizer_req, rpc_req])


class TestRpcReplyRouting(unittest.TestCase):
    def _make_scheduler(self, *, rust_server) -> Scheduler:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.session_controller = MagicMock()
        scheduler.return_health_check_ipcs = []
        scheduler.rust_server = rust_server
        scheduler.ipc_channels = SimpleNamespace(
            recv_from_rpc=object(),
            send_to_tokenizer=MagicMock(),
        )
        scheduler.flush_wrapper = MagicMock()
        scheduler.external_corpus_manager = None
        return scheduler

    def test_rust_mode_replies_to_rpc_over_zmq(self):
        """An RpcReqOutput must go back down the DEALER pair, not the egress ring.

        The Rust egress routes by a rust-minted rid, which an rpc request that
        arrived over zmq does not carry — so pushing the reply there asserts and
        takes the scheduler down.
        """
        rpc_req = RpcReqInput(method="save_remote_model", parameters={"url": "s3://x"})
        rpc_out = RpcReqOutput(success=True, message="")
        rust_server = MagicMock()
        scheduler = self._make_scheduler(rust_server=rust_server)
        scheduler._request_dispatcher = MagicMock(return_value=rpc_out)

        with patch("sglang.srt.managers.scheduler.sock_send") as sock_send:
            scheduler.process_input_requests([rpc_req])

        sock_send.assert_called_once_with(scheduler.ipc_channels.recv_from_rpc, rpc_out)
        rust_server.push_control_output.assert_not_called()

    def test_rust_mode_still_routes_other_control_output_to_the_ring(self):
        """Only RpcReqOutput is special-cased; the rust branch keeps the rest.

        Without this, hoisting the RpcReqOutput check could just as easily have
        been written to swallow every control response.
        """
        recv_req = SimpleNamespace(rid="abc", http_worker_ipc=None)
        output = SimpleNamespace()
        rust_server = MagicMock()
        scheduler = self._make_scheduler(rust_server=rust_server)
        scheduler._request_dispatcher = MagicMock(return_value=output)

        with patch("sglang.srt.managers.scheduler.sock_send") as sock_send:
            scheduler.process_input_requests([recv_req])

        rust_server.push_control_output.assert_called_once_with(recv_req, output)
        sock_send.assert_not_called()


if __name__ == "__main__":
    unittest.main()
