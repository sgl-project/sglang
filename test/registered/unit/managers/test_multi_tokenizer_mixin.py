import pickle
import unittest
from types import SimpleNamespace
from unittest import mock
from unittest.mock import DEFAULT, MagicMock, patch

import numpy as np
import torch

from sglang.srt.sampling.sampling_mask import SamplingMaskChunk
from sglang.srt.utils.weight_versions import WeightVersionSpan
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.beam_search.types import BeamSearchSequence
from sglang.srt.entrypoints import http_server
from sglang.srt.environ import envs
from sglang.srt.managers import multi_tokenizer_mixin
from sglang.srt.managers.io_struct import (
    AbortReq,
    BatchEmbeddingOutput,
    BatchStrOutput,
    BatchTokenIDOutput,
    BeamSearchOutput,
    ConfigureLoggingReq,
    unwrap_from_pickle,
    wrap_as_pickle,
)
from sglang.srt.managers.multi_tokenizer_mixin import (
    MultiDetokenizerRouter,
    MultiHttpWorkerDetokenizerMixin,
    _cached_output_fields,
    _handle_output_by_index,
    _handle_output_by_indices,
    _send_output_to_tokenizer_workers,
)
from sglang.srt.runtime_context import get_context

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _make_batch_str_output() -> BatchStrOutput:
    return BatchStrOutput(
        rids=["rid-0", "rid-1"],
        spec_verify_ct=[0, 0],
        spec_num_correct_drafts=[0, 0],
        spec_correct_drafts_histogram=[[], []],
        finished_reasons=[None, {"type": "length"}],
        output_strs=["first", "second"],
        output_ids=[[1], [2]],
        prompt_tokens=[10, 20],
        completion_tokens=[1, 2],
        reasoning_tokens=[0, 0],
        cached_tokens=[3, 4],
        cached_tokens_details=[
            {"device": 3, "host": 0},
            {"device": 1, "host": 3},
        ],
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
        retraction_counts=[0, 0],
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
        decoded_texts=["first", "second"],
        decode_ids=[[1], [2]],
        read_offsets=[0, 1],
        output_ids=[[1], [2]],
        skip_special_tokens=[True, True],
        spaces_between_special_tokens=[True, True],
        no_stop_trim=[False, False],
        prompt_tokens=[10, 20],
        reasoning_tokens=[0, 0],
        completion_tokens=[1, 2],
        cached_tokens=[3, 4],
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
        prompt_tokens=[10, 20],
        cached_tokens=[3, 4],
        placeholder_tokens_idx=[None, None],
        placeholder_tokens_val=[None, None],
    )


def _make_router() -> MultiDetokenizerRouter:
    router = MultiDetokenizerRouter.__new__(MultiDetokenizerRouter)
    router.ipc_name_list = ["detok-a", "detok-b"]
    router.num_workers = 2
    router._pick = lambda key: {
        "worker-a": "detok-a",
        "worker-b": "detok-b",
    }[key]
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

    def test_batch_str_output_preserves_cached_tokens_details(self):
        output = _make_batch_str_output()

        single_output = _handle_output_by_index(output, 1)

        self.assertEqual(single_output.rids, ["rid-1"])
        self.assertEqual(single_output.cached_tokens, [4])
        self.assertEqual(
            single_output.cached_tokens_details,
            [{"device": 1, "host": 3}],
        )

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
        self.assertEqual(
            grouped.cached_tokens_details,
            [{"device": 1, "host": 3}, {"device": 3, "host": 0}],
        )
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
        router = _make_router()
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
        router = _make_router()
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
        output.http_worker_ipcs = ["worker-a", "worker-a"]
        router = _make_router()
        sent = []
        router._send = lambda target, obj: sent.append((target, obj))

        router._send_batch(output)

        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0][0], "detok-a")
        self.assertEqual(sent[0][1].rids, ["token-rid-0", "token-rid-1"])
        self.assertEqual(sent[0][1].decoded_texts, ["first", "second"])
        self.assertIsNone(sent[0][1].spec_verify_ct)

    def test_embedding_split_preserves_per_request_fields(self):
        output = _make_batch_embedding_output()
        output.retraction_counts = [0, 7]
        output.cached_tokens_details = [{"radix": 0}, {"radix": 2}]
        output.time_stats = wrap_as_pickle(["stats-0", "stats-1"])
        output.pooled_hidden_states = [[1.0], [2.0]]

        split = _handle_output_by_index(output, 1)

        self.assertEqual(split.rids, ["embedding-rid-1"])
        self.assertEqual(split.embeddings, [[0.3, 0.4]])
        self.assertEqual(split.retraction_counts, [7])
        self.assertEqual(split.cached_tokens_details, [{"radix": 2}])
        self.assertEqual(unwrap_from_pickle(split.time_stats), ["stats-1"])
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
        self.assertIsNone(split.retraction_counts)
        self.assertIsNone(split.time_stats)
        self.assertIsNone(split.pooled_hidden_states)

    def test_router_sends_embedding_output_per_request(self):
        output = _make_batch_embedding_output()
        output.http_worker_ipcs = ["worker-a", "worker-a"]
        router = _make_router()
        sent = []
        router._send = lambda target, obj: sent.append((target, obj))

        router._send_batch(output)

        self.assertEqual(
            [(target, obj.rids) for target, obj in sent],
            [("detok-a", ["embedding-rid-0"]), ("detok-a", ["embedding-rid-1"])],
        )

    def test_router_keys_on_rid_without_tokenizer_worker_ipcs(self):
        output = _make_batch_str_output()
        output.http_worker_ipcs = [None, None]
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
                ("detok-for-rid-0", ["rid-0"], [None]),
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
                    AbortReq(rid="rid-0"),
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
        output.prompt_tokens.append(30)
        output.cached_tokens.append(5)
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


if __name__ == "__main__":
    unittest.main()
