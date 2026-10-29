"""Embedding outputs: arrays from the scheduler (pickle IPC), lists and floats in responses, and
the direct scheduler -> tokenizer-worker send (one message per worker) with several workers."""

import pickle
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers import io_struct
from sglang.srt.managers.io_struct import msgpack_decode, msgpack_encode
from sglang.srt.managers.scheduler_components import (
    batch_result_processor,
    output_streamer,
)
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.scheduler_components.output_streamer import (
    SchedulerOutputStreamer,
)
from sglang.srt.managers.tokenizer_manager import _embedding_as_list
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FinishedReq:
    def __init__(self, rid, embedding, http_worker_ipc, pooled_hidden_state=None):
        self.rid = rid
        self.http_worker_ipc = http_worker_ipc
        self.finished_reason = SimpleNamespace(
            to_json=lambda: {"type": "stop", "rid": rid}
        )
        self.embedding = embedding
        self.origin_input_ids = [1] * (len(rid) + 2)
        self.cached_tokens = len(rid)
        self.cached_tokens_device = len(rid) + 1
        self.cached_tokens_host = len(rid) - 1
        self.cached_tokens_storage = 0
        self.time_stats = f"stats-{rid}"
        self.retraction_count = len(rid) - 1
        self.pooled_hidden_state = pooled_hidden_state

    def finished(self):
        return True


def _streamer(socket, nnodes=1):
    return SchedulerOutputStreamer(
        send_to_detokenizer=Mock(socket=socket),
        tree_cache=None,
        server_args=SimpleNamespace(nnodes=nnodes),
        is_generation=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
        disaggregation_mode=DisaggregationMode.NULL,
        enable_hicache_storage=lambda: False,
    )


def _convert(embeddings, pickle_ipc=True):
    with patch.object(batch_result_processor, "_USE_PICKLE_IPC", pickle_ipc):
        return SchedulerBatchResultProcessor._convert_embeddings(
            None, result=SimpleNamespace(embeddings=embeddings)
        )


class TestEmbeddingConversion(CustomTestCase):
    def test_rows_become_float32_arrays_and_responses_keep_the_values(self):
        emb = torch.randn(3, 8, dtype=torch.bfloat16)
        out = _convert(emb)
        self.assertEqual([row.dtype for row in out], [np.float32] * 3)
        self.assertEqual([_embedding_as_list(row) for row in out], emb.tolist())

    def test_full_precision_outputs_keep_their_dtype(self):
        emb = torch.randn(2, 4, dtype=torch.float64)
        self.assertEqual(
            [_embedding_as_list(row) for row in _convert(emb)], emb.tolist()
        )

    def test_one_value_per_request_becomes_python_floats(self):
        # 1-D pooled outputs (e.g. reward scores) give a numpy scalar per request.
        scores = torch.tensor([0.25, -1.5])
        converted = [_embedding_as_list(x) for x in _convert(scores)]
        self.assertEqual(converted, [0.25, -1.5])
        self.assertTrue(all(type(x) is float for x in converted))

    def test_lists_pass_through(self):
        self.assertEqual(_embedding_as_list([0.5, 1.0]), [0.5, 1.0])

    def test_pickle_ipc_round_trips_the_arrays(self):
        emb = torch.randn(2, 4, dtype=torch.bfloat16)
        output = output_streamer._embedding_output(_fields(["a", "b"], _convert(emb)))
        decoded = pickle.loads(pickle.dumps(output))
        self.assertEqual(
            [_embedding_as_list(e) for e in decoded.embeddings], emb.tolist()
        )

    def test_msgpack_ipc_keeps_lists_and_round_trips(self):
        emb = torch.randn(2, 4)
        out = _convert(emb, pickle_ipc=False)
        self.assertEqual(out, emb.tolist())
        with patch.object(io_struct, "_USE_PICKLE_IPC", False):
            output = output_streamer._embedding_output(_fields(["a", "b"], out))
            decoded = msgpack_decode(msgpack_encode(output))
        self.assertEqual(decoded.embeddings, emb.tolist())


def _fields(rids, embeddings, ipcs=None):
    n = len(rids)
    return dict(
        rids=rids,
        http_worker_ipcs=ipcs or ["ipc://w0"] * n,
        time_stats=[None] * n,
        finished_reasons=[{"type": "stop"}] * n,
        embeddings=embeddings,
        prompt_tokens=[3] * n,
        cached_tokens=[0] * n,
        cached_tokens_details=[None] * n,
        retraction_counts=[0] * n,
    )


class TestEmbeddingOutputRouting(CustomTestCase):
    def _stream(self, streamer, reqs):
        """Run _stream_output_embedding; return (worker ipc, output) for each direct send."""
        sent = []
        with (
            patch.object(
                SchedulerOutputStreamer, "_worker_socket", lambda self, ipc: ipc
            ),
            patch.object(
                output_streamer,
                "sock_send",
                lambda socket, output: sent.append((socket, output)),
            ),
        ):
            streamer._stream_output_embedding(reqs)
        return sent

    def test_several_workers_get_one_message_each(self):
        streamer = _streamer(socket=object())
        reqs = [
            _FinishedReq("a", np.full(4, 1.0, np.float32), "ipc://w0"),
            _FinishedReq("b", np.full(4, 2.0, np.float32), "ipc://w1"),
            _FinishedReq("c", np.full(4, 3.0, np.float32), "ipc://w0"),
        ]

        sent = self._stream(streamer, reqs)

        streamer.send_to_detokenizer.send_output.assert_not_called()
        by_worker = {
            ipc: (out.rids, [e.tolist() for e in out.embeddings], out.http_worker_ipcs)
            for ipc, out in sent
        }
        self.assertEqual(len(sent), 2)
        self.assertEqual(
            by_worker,
            {
                "ipc://w0": (["a", "c"], [[1.0] * 4, [3.0] * 4], ["ipc://w0"] * 2),
                "ipc://w1": (["b"], [[2.0] * 4], ["ipc://w1"]),
            },
        )
        # Every per-request field follows its request into the worker's message.
        for _, out in sent:
            self.assertEqual(
                out.finished_reasons, [{"type": "stop", "rid": r} for r in out.rids]
            )
            self.assertEqual(out.prompt_tokens, [len(r) + 2 for r in out.rids])
            self.assertEqual(out.cached_tokens, [len(r) for r in out.rids])
            self.assertEqual(out.retraction_counts, [len(r) - 1 for r in out.rids])
            self.assertEqual(out.time_stats, [f"stats-{r}" for r in out.rids])
            self.assertEqual(
                out.cached_tokens_details,
                [{"device": len(r) + 1, "host": len(r) - 1} for r in out.rids],
            )
            self.assertIsNone(out.pooled_hidden_states)

    def _assert_detokenizer_path(self, streamer, reqs):
        self.assertEqual(self._stream(streamer, reqs), [])
        (output,), _ = streamer.send_to_detokenizer.send_output.call_args
        self.assertEqual(output.rids, [req.rid for req in reqs])
        return output

    def test_single_tokenizer_keeps_the_detokenizer_path(self):
        reqs = [_FinishedReq("a", np.ones(4, np.float32), None)]
        self._assert_detokenizer_path(_streamer(socket=object()), reqs)

    def test_several_nodes_keep_the_detokenizer_path(self):
        # Tokenizer-worker sockets are local to the head node.
        reqs = [_FinishedReq("a", np.ones(4, np.float32), "ipc://w0")]
        self._assert_detokenizer_path(_streamer(socket=object(), nnodes=2), reqs)

    def test_pooled_hidden_states_keep_the_detokenizer_path(self):
        reqs = [
            _FinishedReq("a", np.ones(4, np.float32), "ipc://w0", torch.ones(2)),
            _FinishedReq("b", np.ones(4, np.float32), "ipc://w1", torch.ones(2)),
        ]
        output = self._assert_detokenizer_path(_streamer(socket=object()), reqs)
        self.assertIsNotNone(output.pooled_hidden_states)

    def test_close_releases_the_worker_sockets(self):
        streamer = _streamer(socket=object())
        socket = Mock()
        streamer._embedding_worker_sockets = {"ipc://w0": socket}
        streamer.close_embedding_worker_sockets()
        socket.close.assert_called_once_with(linger=0)
        self.assertIsNone(streamer._embedding_worker_sockets)

    def test_non_output_ranks_send_nothing_directly(self):
        streamer = _streamer(socket=None)
        reqs = [_FinishedReq("a", np.ones(4, np.float32), "ipc://w0")]

        self.assertEqual(self._stream(streamer, reqs), [])


if __name__ == "__main__":
    unittest.main()
