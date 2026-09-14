"""Unit tests for MultiDetokenizerRouter. No server, no model loading."""

import contextlib
import unittest
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers import multi_tokenizer_mixin as mtm
from sglang.srt.managers.io_struct import (
    BatchTokenIDOutput,
    ConfigureLoggingReq,
    FreezeGCReq,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _StopLoop(Exception):
    """Breaks out of the router's infinite event loop."""


class _RecordingSocketMapping:
    def __init__(self):
        self.sent = []

    def send_output(self, ipc_name, output, is_tokenizer=False):
        self.sent.append((ipc_name, output))

    def clear_all_sockets(self):
        pass


def _make_router(num_workers: int) -> mtm.MultiDetokenizerRouter:
    ipc_names = [f"ipc://detok-{i}" for i in range(num_workers)]
    with (
        patch.object(mtm, "zmq", MagicMock()),
        patch.object(mtm, "get_zmq_socket", return_value=MagicMock()),
        patch.object(mtm, "SocketMapping", _RecordingSocketMapping),
    ):
        return mtm.MultiDetokenizerRouter(ipc_names, MagicMock())


def _drive(router: mtm.MultiDetokenizerRouter, recv_objs):
    """Feed recv_objs through the router's event loop and return what it sent."""
    with patch.object(mtm, "sock_recv", side_effect=list(recv_objs) + [_StopLoop()]):
        with contextlib.suppress(_StopLoop):
            router.event_loop()
    return router.socket_mapping.sent


def _batch_token_id_output(rids, http_worker_ipcs):
    """A BatchTokenIDOutput shaped like what the scheduler streams out."""
    n = len(rids)
    return BatchTokenIDOutput(
        rids=list(rids),
        http_worker_ipcs=http_worker_ipcs,
        finished_reasons=[None] * n,
        decoded_texts=[""] * n,
        decode_ids=[[1]] * n,
        read_offsets=[0] * n,
        output_ids=[[1]] * n,
        skip_special_tokens=[True] * n,
        spaces_between_special_tokens=[True] * n,
        no_stop_trim=[False] * n,
        prompt_tokens=[1] * n,
        reasoning_tokens=[0] * n,
        completion_tokens=[1] * n,
        cached_tokens=[0] * n,
        input_token_logprobs_val=[[]] * n,
        input_token_logprobs_idx=[[]] * n,
        output_token_logprobs_val=[[]] * n,
        output_token_logprobs_idx=[[]] * n,
        input_top_logprobs_val=[[]] * n,
        input_top_logprobs_idx=[[]] * n,
        output_top_logprobs_val=[[]] * n,
        output_top_logprobs_idx=[[]] * n,
        input_token_ids_logprobs_val=[[]] * n,
        input_token_ids_logprobs_idx=[[]] * n,
        output_token_ids_logprobs_val=[[]] * n,
        output_token_ids_logprobs_idx=[[]] * n,
        output_token_entropy_val=[None] * n,
        output_token_sampling_mask=None,
        output_token_sampling_logprobs=None,
        output_hidden_states=[None] * n,
        routed_experts=None,
        indexer_topk=None,
        placeholder_tokens_idx=None,
        placeholder_tokens_val=None,
    )


class TestMultiDetokenizerRouter(CustomTestCase):
    def test_routes_batch_when_http_worker_ipcs_are_unset(self):
        """--detokenizer-worker-num > 1 must work with a single tokenizer worker.

        The scheduler leaves http_worker_ipcs as a list of None unless
        --tokenizer-worker-num > 1, so the router cannot key on it.
        """
        router = _make_router(4)
        rids = [f"rid-{i}" for i in range(64)]

        sent = _drive(router, [_batch_token_id_output(rids, [None] * len(rids))])

        self.assertEqual(len(sent), len(rids))
        self.assertEqual(len({ipc for ipc, _ in sent}), 4)

    def test_spreads_requests_over_every_worker(self):
        """A single HTTP worker must still feed every detokenizer worker.

        Keying on http_worker_ipc made the reachable worker count equal to the
        tokenizer worker count, so extra detokenizers stayed idle forever.
        """
        router = _make_router(4)
        rids = [f"rid-{i}" for i in range(400)]
        ipcs = ["ipc://tok-0"] * len(rids)

        sent = _drive(router, [_batch_token_id_output(rids, ipcs)])

        per_worker = {ipc: 0 for ipc in router.ipc_name_list}
        for ipc, _ in sent:
            per_worker[ipc] += 1
        # crc32 over distinct rids spreads far more evenly than the ~1 worker
        # a per-HTTP-worker key can reach, so require a real share each.
        for ipc, count in per_worker.items():
            self.assertGreater(count, 0, f"{ipc} received no requests")
            self.assertLess(count, len(rids) // 2, f"{ipc} is a hot spot")

    def test_same_rid_always_lands_on_the_same_worker(self):
        """Incremental detokenization keeps per-rid state on one worker.

        DetokenizerManager.decode_status is keyed by rid, so every chunk of a
        request must reach the same worker no matter what else changes about
        the batch it arrives in.
        """
        router = _make_router(4)
        rids = [f"rid-{i}" for i in range(32)]

        # Three streaming chunks for the same rids, mixed with a rid-only batch.
        sent = _drive(
            router,
            [
                _batch_token_id_output(rids, [None] * len(rids)),
                _batch_token_id_output(rids, ["ipc://tok-0"] * len(rids)),
                _batch_token_id_output(rids, ["ipc://tok-1"] * len(rids)),
            ],
        )

        rid_to_ipcs = {}
        for ipc, obj in sent:
            rid_to_ipcs.setdefault(obj.rids[0], set()).add(ipc)
        self.assertEqual(len(rid_to_ipcs), len(rids))
        for rid, ipcs in rid_to_ipcs.items():
            self.assertEqual(len(ipcs), 1, f"{rid} was split across {ipcs}")

    def test_forwards_per_item_http_worker_ipc(self):
        """Splitting a batch must keep each item's reply address.

        The detokenizer answers the HTTP worker named by http_worker_ipcs, so
        losing it when the router splits a batch would strand the response.
        """
        router = _make_router(4)
        rids = ["rid-a", "rid-b", "rid-c"]
        ipcs = ["ipc://tok-0", "ipc://tok-1", "ipc://tok-0"]

        sent = _drive(router, [_batch_token_id_output(rids, ipcs)])

        forwarded = {obj.rids[0]: obj.http_worker_ipcs for _, obj in sent}
        self.assertEqual(
            forwarded,
            {
                "rid-a": ["ipc://tok-0"],
                "rid-b": ["ipc://tok-1"],
                "rid-c": ["ipc://tok-0"],
            },
        )

    def test_control_messages_reach_every_worker(self):
        """Control messages carry no rid, so they must be broadcast.

        ConfigureLoggingReq used to be routed like a request, which crashed the
        router with a single tokenizer worker and otherwise reconfigured only
        one of the N detokenizers.
        """
        for req in (FreezeGCReq(), ConfigureLoggingReq(log_level="debug")):
            with self.subTest(req=type(req).__name__):
                router = _make_router(4)
                sent = _drive(router, [req])
                self.assertEqual({ipc for ipc, _ in sent}, set(router.ipc_name_list))

    def test_idle_batch_reaches_every_worker(self):
        """An idle batch carries no rid, so per-rid routing must not drop it.

        Routing a batch item by item is driven by rids, and an idle batch has
        none. Without the empty-batch broadcast it would reach no worker at all.
        """
        router = _make_router(4)

        sent = _drive(router, [_batch_token_id_output([], [])])

        self.assertEqual({ipc for ipc, _ in sent}, set(router.ipc_name_list))


if __name__ == "__main__":
    unittest.main()
