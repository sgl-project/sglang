import multiprocessing as mp
import os
import pickle
import tempfile
import time
import traceback
import unittest
from array import array
from contextlib import nullcontext
from functools import partial
from types import SimpleNamespace
from unittest import mock
from unittest.mock import DEFAULT, MagicMock, patch

import numpy as np
import torch
import zmq
from prometheus_client import (
    CollectorRegistry,
    Histogram,
    generate_latest,
    multiprocess,
)

from sglang.srt.sampling.sampling_mask import SamplingMaskChunk
from sglang.srt.utils.weight_versions import WeightVersionSpan
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.beam_search.types import BeamSearchSequence
from sglang.srt.entrypoints import http_server
from sglang.srt.environ import envs
from sglang.srt.managers import io_struct, ipc_metrics, multi_tokenizer_mixin
from sglang.srt.managers.detokenizer_manager import DetokenizerManager
from sglang.srt.managers.incremental_detokenize import DetokenizeHeader
from sglang.srt.managers.io_struct import (
    AbortReq,
    BatchEmbeddingOutput,
    BatchStrOutput,
    BatchTokenIDOutput,
    BeamSearchOutput,
    ConfigureLoggingReq,
    RequestOutputStats,
)
from sglang.srt.managers.multi_tokenizer_mixin import (
    MultiDetokenizerRouter,
    MultiHttpWorkerDetokenizerMixin,
    SocketMapping,
    _cached_output_fields,
    _handle_output_by_index,
    _handle_output_by_indices,
    _send_output_to_tokenizer_workers,
)
from sglang.srt.runtime_context import get_context

register_cpu_ci(est_time=35, suite="base-a-test-cpu")


def _request_stats(prompt_tokens, cached_tokens, cached_tokens_details=None):
    return RequestOutputStats(
        prompt_tokens=prompt_tokens,
        cached_tokens=cached_tokens,
        cached_tokens_details=cached_tokens_details,
        retraction_count=0,
        image_tokens=0,
        audio_tokens=0,
        video_tokens=0,
        time_stats=None,
    )


_REQUEST_STATS_1 = _request_stats(20, 4, {"device": 1, "host": 3})


def _make_batch_str_output() -> BatchStrOutput:
    return BatchStrOutput(
        rids=["rid-0", "rid-1"],
        spec_verify_ct=[0, 0],
        spec_num_correct_drafts=[0, 0],
        spec_correct_drafts_histogram=[[], []],
        finished_reasons=[None, {"type": "length"}],
        output_strs=["first", "second"],
        # Rows [1] and [2, 3].
        output_ids_flat=array("q", [1, 2, 3]),
        output_ids_offsets=[0, 1, 3],
        completion_tokens=[1, 2],
        reasoning_tokens=[0, 0],
        # Only rid-1 has new request stats in this step.
        request_stats=[None, _REQUEST_STATS_1],
        input_token_logprobs_val=[[], []],
        input_token_logprobs_idx=[[], []],
        output_token_logprobs_val=[[], []],
        output_token_logprobs_idx=[[], []],
        input_top_logprobs_val=[[], []],
        input_top_logprobs_idx=[[], []],
        output_top_logprobs_val=[[], []],
        output_top_logprobs_idx=[[], []],
        input_token_ids_logprobs_val=[[], []],
        input_token_ids_logprobs_idx=[[], []],
        output_token_ids_logprobs_val=[[], []],
        output_token_ids_logprobs_idx=[[], []],
        output_token_entropy_val=[0.0, 0.0],
        output_token_sampling_mask=[
            SamplingMaskChunk(
                lengths=np.array([2], np.int32),
                token_ids=np.array([7, 8], np.int32),
                logprobs=np.array([-0.5, -1.0], np.float32),
            ),
            SamplingMaskChunk(
                lengths=np.array([1], np.int32),
                token_ids=np.array([9], np.int32),
                logprobs=np.array([0.0], np.float32),
            ),
        ],
        output_hidden_states=[None, None],
        routed_experts=[None, None],
        indexer_topk=[None, None],
        placeholder_tokens_idx=[None, None],
        placeholder_tokens_val=[None, None],
        weight_versions=[
            [
                WeightVersionSpan(version="v1", start=0, end=3),
                WeightVersionSpan(version="v2", start=3, end=5),
            ],
            [WeightVersionSpan(version="v2", start=0, end=2)],
        ],
        input_top_logprobs_val_flat=[np.array([[1.0]], np.float32), None],
        input_top_logprobs_idx_flat=[np.array([[1]], np.int32), None],
        input_top_logprobs_flat_null_prefix=[0, None],
        beam_search_output=[
            BeamSearchOutput(
                sequences=[
                    BeamSearchSequence(tokens=[11], cum_logprob=-1.0),
                ]
            ),
            None,
        ],
        token_steps=[[1], [2]],
        prefill_weight_versions=[
            [WeightVersionSpan(version="v0", start=0, end=7)],
            None,
        ],
    )


def _make_batch_token_id_output() -> BatchTokenIDOutput:
    return BatchTokenIDOutput(
        rids=["token-rid-0", "token-rid-1"],
        # OutputStreamer leaves the spec_* lists empty without speculative decoding.
        spec_verify_ct=[],
        spec_num_correct_drafts=[],
        spec_correct_drafts_histogram=[],
        spec_num_block_accept_tokens=[],
        spec_num_cap_tokens=[],
        spec_cap_lens_histogram=[],
        finished_reasons=[None, None],
        # Rows [1] and [].
        output_ids_flat=array("q", [1]),
        output_ids_offsets=[0, 1, 1],
        reasoning_tokens=[0, 0],
        completion_tokens=[1, 2],
        input_token_logprobs_val=[[], []],
        input_token_logprobs_idx=[[], []],
        output_token_logprobs_val=[[], []],
        output_token_logprobs_idx=[[], []],
        input_top_logprobs_val=[[], []],
        input_top_logprobs_idx=[[], []],
        output_top_logprobs_val=[[], []],
        output_top_logprobs_idx=[[], []],
        input_token_ids_logprobs_val=[[], []],
        input_token_ids_logprobs_idx=[[], []],
        output_token_ids_logprobs_val=[[], []],
        output_token_ids_logprobs_idx=[[], []],
        output_token_entropy_val=[None, None],
        output_token_sampling_mask=[None, None],
        output_hidden_states=[None, None],
        routed_experts=[None, None],
        indexer_topk=[None, None],
        placeholder_tokens_idx=[None, None],
        placeholder_tokens_val=[None, None],
    )


def _make_batch_embedding_output() -> BatchEmbeddingOutput:
    return BatchEmbeddingOutput(
        rids=["embedding-rid-0", "embedding-rid-1"],
        finished_reasons=[None, None],
        embeddings=[[0.1, 0.2], [0.3, 0.4]],
        request_stats=[_request_stats(10, 3), _request_stats(20, 4)],
        placeholder_tokens_idx=[None, None],
        placeholder_tokens_val=[None, None],
    )


def _make_router(targets_by_rid=None) -> MultiDetokenizerRouter:
    router = MultiDetokenizerRouter.__new__(MultiDetokenizerRouter)
    router.ipc_name_list = ["detok-a", "detok-b"]
    router.num_workers = 2
    router._pick = (targets_by_rid or {}).__getitem__
    return router


class TestMultiTokenizerMixin(CustomTestCase):
    def test_each_tokenizer_worker_freezes_locally(self):
        threshold = [700, 10, 1000]
        server_args = mock.Mock(gc_threshold=threshold)
        timeline = mock.Mock()
        execution = mock.Mock(moe=mock.Mock(is_ep_scale_joiner=False))
        with (
            get_context().override_server_args(
                checkpoint_engine_wait_weights_before_ready=False,
                delete_ckpt_after_loading=False,
                skip_server_warmup=False,
                debug_tensor_dump_input_file=None,
            ),
            mock.patch.object(http_server, "get_exec", return_value=execution),
            mock.patch.object(http_server, "freeze_gc", side_effect=timeline.worker),
            mock.patch.object(
                http_server,
                "_freeze_gc_after_server_warmup",
                side_effect=timeline.routed,
            ),
        ):
            for worker_id in range(4):
                timeline.reset_mock()
                kwargs = http_server._multi_tokenizer_warmup_kwargs(
                    server_args, worker_id
                )
                http_server._wait_and_warmup(
                    **kwargs,
                    execute_warmup_func=lambda _: timeline.warmup() or True,
                )
                self.assertEqual(
                    timeline.mock_calls,
                    [
                        mock.call.warmup(),
                        mock.call.worker(f"Tokenizer worker {worker_id}", threshold),
                        mock.call.routed(server_args),
                    ],
                )

    def test_router_cpu_metric_started_only_when_enabled(self):
        for enabled in (False, True):
            with self.subTest(enable_metrics=enabled):
                with (
                    patch.multiple(
                        multi_tokenizer_mixin,
                        kill_itself_when_parent_died=DEFAULT,
                        configure_logger=DEFAULT,
                        MultiDetokenizerRouter=DEFAULT,
                        start_cpu_monitor_thread=DEFAULT,
                    ) as mocks,
                    patch("setproctitle.setproctitle"),
                    patch("psutil.Process"),
                ):
                    multi_tokenizer_mixin.run_multi_detokenizer_router_process(
                        [], None, None, enable_metrics=enabled
                    )

                self.assertEqual(
                    mocks["start_cpu_monitor_thread"].call_count, 1 if enabled else 0
                )
                mocks["MultiDetokenizerRouter"].assert_called_once_with(
                    [], None, enable_metrics=enabled
                )

    def test_batch_str_output_keeps_request_stats_per_request(self):
        output = _make_batch_str_output()

        single_output = _handle_output_by_index(output, 1)

        self.assertEqual(single_output.rids, ["rid-1"])
        self.assertEqual(single_output.request_stats, [_REQUEST_STATS_1])
        self.assertEqual(_handle_output_by_index(output, 0).request_stats, [None])

    def test_batch_str_output_keeps_weight_versions_nested_per_request(self):
        """Per-request segment lists stay one level nested after the split."""
        output = _make_batch_str_output()

        self.assertEqual(
            _handle_output_by_index(output, 0).weight_versions,
            [
                [
                    WeightVersionSpan(version="v1", start=0, end=3),
                    WeightVersionSpan(version="v2", start=3, end=5),
                ]
            ],
        )
        self.assertEqual(
            _handle_output_by_index(output, 1).weight_versions,
            [[WeightVersionSpan(version="v2", start=0, end=2)]],
        )

    def test_batch_str_output_keeps_sampling_distribution_aligned(self):
        output = _make_batch_str_output()

        single_output = _handle_output_by_index(output, 0)

        (chunk,) = single_output.output_token_sampling_mask
        self.assertEqual(
            chunk.to_lists(support_logprobs=True), ([[7, 8]], [[-0.5, -1.0]])
        )

    def test_batch_str_output_without_weight_versions_stays_none(self):
        """An output from an older server without the field splits into None."""
        output = _make_batch_str_output()
        output.weight_versions = None

        self.assertIsNone(_handle_output_by_index(output, 0).weight_versions)

    def test_grouped_split_preserves_latest_fields_and_order(self):
        output = _make_batch_str_output()

        grouped = _handle_output_by_indices(output, [1, 0])

        self.assertEqual(grouped.rids, ["rid-1", "rid-0"])
        self.assertEqual(grouped.output_strs, ["second", "first"])
        self.assertEqual(grouped.request_stats, [_REQUEST_STATS_1, None])
        self.assertEqual(
            grouped.weight_versions,
            [
                [WeightVersionSpan(version="v2", start=0, end=2)],
                [
                    WeightVersionSpan(version="v1", start=0, end=3),
                    WeightVersionSpan(version="v2", start=3, end=5),
                ],
            ],
        )
        self.assertEqual(grouped.input_top_logprobs_flat_null_prefix, [None, 0])
        self.assertIsNone(grouped.beam_search_output[0])
        self.assertEqual(
            grouped.beam_search_output[1].sequences[0].tokens,
            [11],
        )
        np.testing.assert_array_equal(
            grouped.input_top_logprobs_val_flat[1], np.array([[1.0]], np.float32)
        )

    def test_split_slices_each_rows_output_ids(self):
        """Rows of different lengths, an empty one included, keep their ids."""
        for output, rows in (
            (_make_batch_str_output(), [[1], [2, 3]]),
            (_make_batch_token_id_output(), [[1], []]),
        ):
            with self.subTest(type=type(output).__name__):
                for i, ids in enumerate(rows):
                    one = _handle_output_by_index(output, i)
                    self.assertEqual(one.output_ids_flat, array("q", ids))
                    self.assertEqual(one.output_ids_offsets, [0, len(ids)])
                grouped = _handle_output_by_indices(output, [1, 0])
                self.assertEqual(grouped.output_ids_flat, array("q", rows[1] + rows[0]))
                self.assertEqual(
                    grouped.output_ids_offsets,
                    [0, len(rows[1]), len(rows[1]) + len(rows[0])],
                )

    def test_grouped_split_caches_struct_fields_by_output_type(self):
        self.addCleanup(_cached_output_fields.cache_clear)
        _cached_output_fields.cache_clear()

        str_output = _make_batch_str_output()
        token_output = _make_batch_token_id_output()
        _handle_output_by_indices(str_output, [0, 1])
        _handle_output_by_indices(str_output, [1, 0])
        _handle_output_by_indices(token_output, [0, 1])
        _handle_output_by_indices(token_output, [1, 0])

        cache_info = _cached_output_fields.cache_info()
        self.assertEqual(cache_info.misses, 2)
        self.assertEqual(cache_info.hits, 2)
        self.assertEqual(cache_info.currsize, 2)

    def test_router_groups_by_target_and_preserves_owner_ipcs(self):
        output = _make_batch_str_output()
        output.http_worker_ipcs = ["worker-a", "worker-a"]
        router = _make_router({"rid-0": "detok-a", "rid-1": "detok-a"})
        sent = []
        router._send = lambda target, obj: sent.append((target, obj))

        router._send_batch(output)

        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0][0], "detok-a")
        self.assertEqual(sent[0][1].rids, ["rid-0", "rid-1"])
        self.assertEqual(sent[0][1].http_worker_ipcs, ["worker-a", "worker-a"])
        self.assertEqual(output.rids, ["rid-0", "rid-1"])
        self.assertEqual(output.http_worker_ipcs, ["worker-a", "worker-a"])

    def test_router_keeps_first_target_order_and_does_not_cross_messages(self):
        output = _make_batch_str_output()
        output.http_worker_ipcs = ["worker-b", "worker-a"]
        router = _make_router({"rid-0": "detok-b", "rid-1": "detok-a"})
        sent = []
        router._send = lambda target, obj: sent.append((target, obj))

        router._send_batch(output)

        self.assertEqual(
            [(target, obj.rids, obj.http_worker_ipcs) for target, obj in sent],
            [
                ("detok-b", ["rid-0"], ["worker-b"]),
                ("detok-a", ["rid-1"], ["worker-a"]),
            ],
        )

    def test_router_groups_token_output_without_speculative_decoding(self):
        output = _make_batch_token_id_output()
        router = _make_router({"token-rid-0": "detok-a", "token-rid-1": "detok-a"})
        sent = []
        router._send = lambda target, obj: sent.append((target, obj))

        router._send_batch(output)

        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0][0], "detok-a")
        self.assertEqual(sent[0][1].rids, ["token-rid-0", "token-rid-1"])
        self.assertEqual(sent[0][1].output_ids_flat, array("q", [1]))
        self.assertEqual(sent[0][1].output_ids_offsets, [0, 1, 1])
        self.assertIsNone(sent[0][1].spec_verify_ct)

    def test_router_split_keeps_each_finished_rows_route(self):
        output = _make_batch_token_id_output()
        output.finished_reasons = [None, {"type": "length"}]
        output.http_worker_ipcs = [None, "worker-b"]
        header = DetokenizeHeader(
            prompt_tail_ids=array("q", [1, 2]),
            skip_special_tokens=True,
            spaces_between_special_tokens=True,
            no_stop_trim=False,
            http_worker_ipc="worker-a",
        )
        output.detokenize_headers = [header, None]
        router = _make_router({"token-rid-0": "detok-a", "token-rid-1": "detok-b"})
        sent = []
        router._send = lambda target, obj: sent.append((target, obj))

        router._send_batch(output)

        self.assertEqual(
            [
                (target, obj.detokenize_headers, obj.http_worker_ipcs)
                for target, obj in sent
            ],
            [("detok-a", [header], [None]), ("detok-b", [None], ["worker-b"])],
        )

    def test_embedding_split_preserves_per_request_fields(self):
        output = _make_batch_embedding_output()
        output.pooled_hidden_states = [[1.0], [2.0]]

        split = _handle_output_by_index(output, 1)

        self.assertEqual(split.rids, ["embedding-rid-1"])
        self.assertEqual(split.embeddings, [[0.3, 0.4]])
        self.assertEqual(split.request_stats, [_request_stats(20, 4)])
        self.assertEqual(split.pooled_hidden_states, [[2.0]])

    def test_embedding_split_takes_a_row_of_stacked_pooled_hidden_states(self):
        output = _make_batch_embedding_output()
        output.pooled_hidden_states = [torch.tensor([[1.0, 1.5], [2.0, 2.5]])]

        for index, row in enumerate(([1.0, 1.5], [2.0, 2.5])):
            split = _handle_output_by_index(output, index)
            self.assertEqual(len(split.pooled_hidden_states), 1)
            self.assertEqual(split.pooled_hidden_states[0].tolist(), row)

    def test_embedding_split_keeps_absent_optional_fields_none(self):
        split = _handle_output_by_index(_make_batch_embedding_output(), 0)

        self.assertEqual(split.rids, ["embedding-rid-0"])
        self.assertIsNone(split.pooled_hidden_states)

    def test_router_sends_embedding_output_per_request(self):
        output = _make_batch_embedding_output()
        output.http_worker_ipcs = ["worker-a", "worker-a"]
        router = _make_router(
            {"embedding-rid-0": "detok-a", "embedding-rid-1": "detok-a"}
        )
        sent = []
        router._send = lambda target, obj: sent.append((target, obj))

        router._send_batch(output)

        self.assertEqual(
            [(target, obj.rids) for target, obj in sent],
            [("detok-a", ["embedding-rid-0"]), ("detok-a", ["embedding-rid-1"])],
        )

    def test_batch_str_output_keeps_prefill_weight_versions_nested_per_request(self):
        """Prefill spans split per request the same way the sampling spans do."""
        output = _make_batch_str_output()

        self.assertEqual(
            _handle_output_by_index(output, 0).prefill_weight_versions,
            [[WeightVersionSpan(version="v0", start=0, end=7)]],
        )
        self.assertEqual(
            _handle_output_by_index(output, 1).prefill_weight_versions, [None]
        )

    def test_batch_str_output_without_prefill_weight_versions_stays_none(self):
        """An output produced without the flag splits into None."""
        output = _make_batch_str_output()
        output.prefill_weight_versions = None

        self.assertIsNone(_handle_output_by_index(output, 0).prefill_weight_versions)

    def test_router_keys_on_rid_not_tokenizer_worker_ipc(self):
        output = _make_batch_str_output()
        output.http_worker_ipcs = ["worker-a", None]
        router = _make_router()
        router._pick = lambda key: f"detok-for-{key}"
        sent = []
        router._send = lambda target, obj: sent.append(
            (target, obj.rids, obj.http_worker_ipcs)
        )

        router._send_batch(output)

        self.assertEqual(
            sent,
            [
                ("detok-for-rid-0", ["rid-0"], ["worker-a"]),
                ("detok-for-rid-1", ["rid-1"], [None]),
            ],
        )

    def test_router_broadcasts_logging_config_and_routes_single_req_by_rid(self):
        router = _make_router()
        router.recv_from_scheduler = object()
        router._pick = lambda key: f"detok-for-{key}"
        sent = []
        router._send = lambda target, obj: sent.append((target, type(obj).__name__))

        with (
            patch(
                "sglang.srt.managers.multi_tokenizer_mixin.sock_recv",
                side_effect=[
                    ConfigureLoggingReq(log_level="debug"),
                    AbortReq(rid="rid-0", http_worker_ipc="worker-a"),
                    EOFError("end of input"),
                ],
            ),
            self.assertRaisesRegex(EOFError, "end of input"),
        ):
            router.event_loop()

        self.assertEqual(
            sent,
            [
                ("detok-a", "ConfigureLoggingReq"),
                ("detok-b", "ConfigureLoggingReq"),
                ("detok-for-rid-0", "AbortReq"),
            ],
        )

    def test_return_path_batching_preserves_owner_order(self):
        output = _handle_output_by_indices(_make_batch_str_output(), [0, 1, 0])
        output.rids = ["rid-0", "rid-1", "rid-2"]
        output.output_strs = ["first", "second", "third"]
        ipcs = ["worker-b", "worker-a", "worker-b"]
        output.http_worker_ipcs = list(ipcs)
        original = pickle.dumps(output)
        sent = []
        mapping = SimpleNamespace(
            send_output=lambda target, obj, *, is_tokenizer: sent.append(
                (target, obj.rids, obj.output_strs, obj.http_worker_ipcs, is_tokenizer)
            )
        )

        expected_grouped = [
            (
                "worker-b",
                ["rid-0", "rid-2"],
                ["first", "third"],
                ["worker-b", "worker-b"],
                True,
            ),
            ("worker-a", ["rid-1"], ["second"], ["worker-a"], True),
        ]
        expected_individual = [
            ("worker-b", ["rid-0"], ["first"], None, True),
            ("worker-a", ["rid-1"], ["second"], None, True),
            ("worker-b", ["rid-2"], ["third"], None, True),
        ]
        for batching in (True, False):
            sent.clear()
            with (
                self.subTest(batching=batching),
                envs.SGLANG_BATCH_TOKENIZER_OUTPUTS.override(batching),
            ):
                _send_output_to_tokenizer_workers(output, ipcs, mapping)
            self.assertEqual(
                sent, expected_grouped if batching else expected_individual
            )
            self.assertEqual(pickle.dumps(output), original)

    def test_return_path_falls_back_before_sending_any_group(self):
        output = _make_batch_embedding_output()
        output.rids.append("embedding-rid-2")
        output.embeddings.append([0.5, 0.6])
        output.request_stats.append(_request_stats(30, 5))
        output.finished_reasons.append(None)
        sent = []
        mapping = SimpleNamespace(
            send_output=lambda target, obj, *, is_tokenizer: sent.append((target, obj))
        )

        _send_output_to_tokenizer_workers(
            output, ["worker-b", "worker-a", "worker-a"], mapping
        )

        self.assertEqual(
            [(target, obj.rids) for target, obj in sent],
            [
                ("worker-b", ["embedding-rid-0"]),
                ("worker-a", ["embedding-rid-1"]),
                ("worker-a", ["embedding-rid-2"]),
            ],
        )

    def test_event_loop_sends_one_message_per_owner(self):
        output = _make_batch_str_output()
        output.http_worker_ipcs = ["worker-a", "worker-a"]
        worker = MultiHttpWorkerDetokenizerMixin()
        worker.recv_from_scheduler = object()
        worker.soft_watchdog = MagicMock()
        worker._request_dispatcher = lambda obj: obj
        sent = []
        mapping = SimpleNamespace(
            send_output=lambda target, obj, *, is_tokenizer: sent.append(
                (target, obj.rids)
            )
        )

        with (
            patch(
                "sglang.srt.managers.multi_tokenizer_mixin.SocketMapping",
                return_value=mapping,
            ),
            patch(
                "sglang.srt.managers.multi_tokenizer_mixin.sock_recv",
                side_effect=[output, EOFError("end of input")],
            ),
            self.assertRaisesRegex(EOFError, "end of input"),
        ):
            worker.multi_http_worker_event_loop()

        self.assertEqual(sent, [("worker-a", ["rid-0", "rid-1"])])


def _delivery_worker(
    endpoint, report, release, tokenizer_workers, pickle_ipc, domain_override
):
    """Spawned worker: exercise the production receive loop and metric export."""
    context = zmq.Context()
    try:
        socket = context.socket(zmq.PULL)
        socket.setsockopt(zmq.RCVTIMEO, 15000)
        socket.bind(endpoint)
        worker = DetokenizerManager.__new__(DetokenizerManager)
        worker.recv_from_scheduler = socket
        worker.soft_watchdog = SimpleNamespace(disable=nullcontext, feed=lambda: None)
        received = []

        def dispatch(obj):
            received.append(obj.rid)
            if len(received) == 2:
                raise EOFError("done")

        worker._request_dispatcher = dispatch
        with (
            get_context().override_server_args(
                enable_metrics=True,
                detokenizer_worker_num=2,
                tokenizer_worker_num=tokenizer_workers,
            ),
            patch.object(io_struct, "_USE_PICKLE_IPC", pickle_ipc),
            (
                patch.object(
                    multi_tokenizer_mixin,
                    "read_clock_domain",
                    return_value=domain_override,
                )
                if domain_override is not None
                else nullcontext()
            ),
        ):
            worker._init_delivery_metrics()
            report.send(("ready", worker.delivery_clock_domain))
            if not release.wait(30):
                raise TimeoutError("Parent did not release the receiver")
            try:
                if tokenizer_workers == 1:
                    worker.event_loop()
                else:
                    worker.multi_http_worker_event_loop()
            except EOFError:
                pass
            report.send(("done", received))
            worker.maybe_clear_socket_mapping()
            if hasattr(worker, "socket_mapping"):
                worker.socket_mapping._zmq_context.destroy(linger=0)
    except BaseException:
        report.send(("error", traceback.format_exc()))
        raise
    finally:
        context.destroy(linger=0)
        report.close()


class TestDetokenizerDelivery(CustomTestCase):
    """Guard message accounting, both receive loops, and disabled wire paths."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="sgl-delivery-")
        self.addCleanup(self.tmp.cleanup)
        self.context = zmq.Context()
        self.addCleanup(self.context.destroy, linger=0)
        self.endpoints = [
            "ipc://" + self.tmp.name + "/worker-" + str(i) for i in range(2)
        ]
        self.receivers = []
        for endpoint in self.endpoints:
            socket = self.context.socket(zmq.PULL)
            socket.setsockopt(zmq.RCVTIMEO, 5000)
            socket.bind(endpoint)
            self.receivers.append(socket)

    def _router(self, enabled=True, workers=2, domain=b"a" * 24):
        with patch.object(
            multi_tokenizer_mixin, "read_clock_domain", return_value=domain
        ):
            router = MultiDetokenizerRouter(
                self.endpoints[:workers],
                SimpleNamespace(
                    detokenizer_ipc_name="ipc://" + self.tmp.name + "/router"
                ),
                enable_metrics=enabled,
            )
        self.addCleanup(router.recv_from_scheduler.context.destroy, linger=0)
        self.addCleanup(router.socket_mapping._zmq_context.destroy, linger=0)
        return router

    def _worker(self, registry):
        worker = DetokenizerManager.__new__(DetokenizerManager)
        worker.tokenizer = None
        worker.recv_from_scheduler = self.receivers[0]
        with (
            patch("prometheus_client.Histogram", partial(Histogram, registry=registry)),
            patch("sglang.srt.managers.detokenizer_manager.start_cpu_monitor_thread"),
            patch(
                "sglang.srt.managers.detokenizer_manager.Watchdog.create",
                return_value=SimpleNamespace(disable=nullcontext, feed=lambda: None),
            ),
        ):
            worker.init_running_status(None)
        return worker

    def test_fanout_timestamps_each_actual_message(self):
        """Grouping/fallback/broadcast must not reuse a batch-wide start time."""
        router = self._router()
        router._pick = {
            "rid-0": self.endpoints[0],
            "rid-1": self.endpoints[0],
            "embedding-rid-0": self.endpoints[0],
            "embedding-rid-1": self.endpoints[0],
            "single": self.endpoints[1],
        }.__getitem__
        empty = _make_batch_str_output()
        empty.rids = []
        messages = [
            _make_batch_str_output(),
            _make_batch_embedding_output(),
            ConfigureLoggingReq(log_level="debug"),
            empty,
            AbortReq(rid="single"),
        ]
        # One grouped send, two fallback sends, two broadcasts of two, one single.
        for pickle_ipc in (False, True):
            with (
                self.subTest(pickle_ipc=pickle_ipc),
                patch.object(io_struct, "_USE_PICKLE_IPC", pickle_ipc),
                patch.object(
                    multi_tokenizer_mixin,
                    "sock_recv",
                    side_effect=messages + [EOFError()],
                ),
                patch.object(
                    ipc_metrics.time, "monotonic_ns", side_effect=range(100, 108)
                ),
                self.assertRaises(EOFError),
            ):
                router.event_loop()
            stamps, rows = [], []
            for receiver, count in zip(self.receivers, (5, 3)):
                for _ in range(count):
                    frames = receiver.recv_multipart()
                    self.assertEqual(len(frames), 2)
                    self.assertEqual(frames[1][8:], b"\x01" + b"a" * 24)
                    stamps.append(int.from_bytes(frames[1][:8], "big"))
                    obj = (
                        pickle.loads(frames[0])
                        if pickle_ipc
                        else io_struct.msgpack_decode(frames[0])
                    )
                    rows.append(getattr(obj, "rids", None))
            self.assertEqual(sorted(stamps), list(range(100, 108)))
            self.assertEqual(
                rows[:3], [["rid-0", "rid-1"], ["embedding-rid-0"], ["embedding-rid-1"]]
            )

    def test_both_loops_observe_after_decode_before_dispatch(self):
        for multi_http in (False, True):
            for pickle_ipc in (False, True):
                with self.subTest(multi_http=multi_http, pickle_ipc=pickle_ipc):
                    registry = CollectorRegistry()
                    with (
                        get_context().override_server_args(
                            enable_metrics=True, detokenizer_worker_num=2
                        ),
                        patch.object(
                            multi_tokenizer_mixin,
                            "read_clock_domain",
                            return_value=b"a" * 24,
                        ),
                    ):
                        worker = self._worker(registry)
                    mapping = SocketMapping(clock_domain=b"a" * 24)
                    self.addCleanup(mapping._zmq_context.destroy, linger=0)
                    clock = [1_000_000_000]
                    message = AbortReq(rid="timed")
                    decoder_owner = io_struct.pickle if pickle_ipc else io_struct
                    decoder_name = "loads" if pickle_ipc else "msgpack_decode"
                    decode = getattr(decoder_owner, decoder_name)

                    def slow_decode(*args, **kwargs):
                        clock[0] += 250_000_000
                        return decode(*args, **kwargs)

                    dispatched = []

                    def dispatch(obj):
                        dispatched.append(obj)
                        self.assertEqual(
                            registry.get_sample_value(
                                "sglang:detokenizer_delivery_seconds_count"
                            ),
                            1,
                        )
                        raise EOFError()

                    worker._request_dispatcher = dispatch
                    with (
                        patch.object(io_struct, "_USE_PICKLE_IPC", pickle_ipc),
                        patch.object(
                            ipc_metrics.time,
                            "monotonic_ns",
                            side_effect=lambda: clock[0],
                        ),
                        patch.object(
                            multi_tokenizer_mixin,
                            "monotonic_ns",
                            side_effect=lambda: clock[0],
                        ),
                        patch.object(
                            decoder_owner, decoder_name, side_effect=slow_decode
                        ),
                    ):
                        mapping.send_output(self.endpoints[0], message)
                        with self.assertRaises(EOFError):
                            if multi_http:
                                worker.multi_http_worker_event_loop()
                            else:
                                worker.event_loop()
                    self.assertEqual(dispatched, [message])
                    self.assertEqual(
                        registry.get_sample_value(
                            "sglang:detokenizer_delivery_seconds_sum"
                        ),
                        0.25,
                    )
                    worker.maybe_clear_socket_mapping()
                    if hasattr(worker, "socket_mapping"):
                        worker.socket_mapping._zmq_context.destroy(linger=0)

    def test_disabled_and_single_worker_keep_legacy_framing(self):
        for enabled, workers in ((False, 2), (True, 1)):
            for multi_http in (False, True):
                with self.subTest(
                    enabled=enabled, workers=workers, multi_http=multi_http
                ):
                    # Fail on any new clock read, including domain detection.
                    with (
                        get_context().override_server_args(
                            enable_metrics=enabled, detokenizer_worker_num=workers
                        ),
                        patch.object(
                            multi_tokenizer_mixin,
                            "read_clock_domain",
                            side_effect=AssertionError("clock domain read"),
                        ),
                        patch.object(
                            ipc_metrics.time,
                            "monotonic_ns",
                            side_effect=AssertionError("clock read"),
                        ),
                        patch.object(
                            multi_tokenizer_mixin,
                            "monotonic_ns",
                            side_effect=AssertionError("clock read"),
                        ),
                    ):
                        registry = CollectorRegistry()
                        worker = self._worker(registry)
                        self.assertIsNone(worker.delivery_seconds)
                        self.assertEqual(generate_latest(registry), b"")
                        router = MultiDetokenizerRouter(
                            self.endpoints[:workers],
                            SimpleNamespace(detokenizer_ipc_name="inproc://unused"),
                            enable_metrics=enabled,
                        )
                        try:
                            message = AbortReq(rid="untimed")
                            router._send(self.endpoints[0], message)
                            frames = self.receivers[0].recv_multipart()
                            self.assertEqual(len(frames), 1)
                            # Replay the unchanged payload through each receive loop.
                            router._send(self.endpoints[0], message)
                            worker.recv_from_scheduler = self.receivers[0]
                            worker.soft_watchdog = SimpleNamespace(
                                disable=nullcontext, feed=lambda: None
                            )
                            dispatched = []

                            def dispatch(obj):
                                dispatched.append(obj)
                                raise EOFError()

                            worker._request_dispatcher = dispatch
                            with self.assertRaises(EOFError):
                                if multi_http:
                                    worker.multi_http_worker_event_loop()
                                else:
                                    worker.event_loop()
                            self.assertEqual(dispatched, [message])
                        finally:
                            router.recv_from_scheduler.context.destroy(linger=0)
                            router.socket_mapping._zmq_context.destroy(linger=0)
                            if hasattr(worker, "socket_mapping"):
                                worker.socket_mapping._zmq_context.destroy(linger=0)

    def test_unverified_and_negative_durations_are_not_observed(self):
        registry = CollectorRegistry()
        with (
            get_context().override_server_args(
                enable_metrics=True, detokenizer_worker_num=2
            ),
            patch.object(
                multi_tokenizer_mixin, "read_clock_domain", return_value=b"a" * 24
            ),
        ):
            worker = self._worker(registry)
        mapping = SocketMapping(clock_domain=b"b" * 24)
        self.addCleanup(mapping._zmq_context.destroy, linger=0)
        message = AbortReq(rid="unobserved")
        for domain in (None, b"b" * 24, b"a" * 24):
            with self.subTest(domain=domain):
                mapping.clock_domain = domain
                with patch.object(ipc_metrics.time, "monotonic_ns", return_value=100):
                    mapping.send_output(self.endpoints[0], message)
                with patch.object(
                    multi_tokenizer_mixin, "monotonic_ns", return_value=99
                ):
                    self.assertEqual(worker._recv_with_delivery_metrics(), message)
        self.assertEqual(
            registry.get_sample_value("sglang:detokenizer_delivery_seconds_count"), 0
        )
        self.assertEqual(
            registry.get_sample_value("sglang:detokenizer_delivery_seconds_sum"), 0
        )

    def test_spawned_workers_export_real_delivery_wait(self):
        # Always test the native identity, including the unobserved fallback on
        # kernels without /proc/self/ns/time. A controlled identity also exercises
        # timestamped transport there; neither the clocks nor ZMQ are mocked.
        for domain_override in (None, b"a" * 24):
            with self.subTest(controlled_identity=domain_override is not None):
                self._check_spawned_delivery(domain_override)

    def _check_spawned_delivery(self, domain_override):
        domain = domain_override or ipc_metrics.read_clock_domain()
        context = mp.get_context("spawn")
        with tempfile.TemporaryDirectory(prefix="sgl-delivery-mp-") as tmp:
            endpoints = ["ipc://" + tmp + "/worker-" + str(i) for i in range(2)]
            metrics_dir = tmp + "/metrics"
            os.mkdir(metrics_dir)
            release = context.Event()
            children = []
            with (
                patch.object(
                    multi_tokenizer_mixin,
                    "read_clock_domain",
                    return_value=domain_override,
                )
                if domain_override is not None
                else nullcontext()
            ):
                router = MultiDetokenizerRouter(
                    endpoints,
                    SimpleNamespace(detokenizer_ipc_name="ipc://" + tmp + "/router"),
                    enable_metrics=True,
                )
            try:
                for index in range(2):
                    report, child_report = context.Pipe(duplex=False)
                    process = context.Process(
                        target=_delivery_worker,
                        args=(
                            endpoints[index],
                            child_report,
                            release,
                            index + 1,
                            io_struct._USE_PICKLE_IPC,
                            domain_override,
                        ),
                    )
                    with patch.dict(
                        os.environ, {"PROMETHEUS_MULTIPROC_DIR": metrics_dir}
                    ):
                        process.start()
                    child_report.close()
                    children.append((process, report))
                for process, report in children:
                    self.assertTrue(
                        report.poll(60), f"worker {process.pid} did not become ready"
                    )
                    self.assertEqual(report.recv(), ("ready", domain))
                for endpoint in endpoints:
                    router._send(endpoint, AbortReq(rid="queued"))
                    router._send(endpoint, AbortReq(rid="last"))
                # Hold both receivers behind an explicit gate after sends complete.
                # This pins a lower bound on real queued delivery, without relying
                # on a race between producer and consumer scheduling.
                start = time.monotonic()
                release.wait(0.05)
                held_seconds = time.monotonic() - start
                release.set()
                for process, report in children:
                    self.assertTrue(
                        report.poll(30), f"worker {process.pid} did not finish"
                    )
                    self.assertEqual(report.recv(), ("done", ["queued", "last"]))
                    process.join(10)
                    self.assertEqual(process.exitcode, 0)
                registry = CollectorRegistry()
                multiprocess.MultiProcessCollector(registry, path=metrics_dir)
                self.assertEqual(
                    registry.get_sample_value(
                        "sglang:detokenizer_delivery_seconds_count"
                    ),
                    4 if domain is not None else 0,
                )
                total = registry.get_sample_value(
                    "sglang:detokenizer_delivery_seconds_sum"
                )
                if domain is not None:
                    self.assertGreaterEqual(total, 4 * held_seconds)
                else:
                    self.assertEqual(total, 0)
                self.assertIn(
                    b"sglang:detokenizer_delivery_seconds_bucket",
                    generate_latest(registry),
                )
            finally:
                release.set()
                router.recv_from_scheduler.context.destroy(linger=0)
                router.socket_mapping._zmq_context.destroy(linger=0)
                for process, report in children:
                    process.join(5)
                    if process.is_alive():
                        process.terminate()
                        process.join(5)
                    if process.is_alive():
                        process.kill()
                        process.join(5)
                    report.close()
                    process.close()


if __name__ == "__main__":
    unittest.main()
