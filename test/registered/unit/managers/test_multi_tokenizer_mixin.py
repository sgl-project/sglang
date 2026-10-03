import copy
import pickle
import unittest
from array import array
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from sglang.srt.sampling.sampling_mask import SamplingMaskChunk
from sglang.srt.utils.weight_versions import WeightVersionSpan
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers import io_struct, multi_tokenizer_mixin
from sglang.srt.managers.io_struct import (
    BatchEmbeddingOutput,
    BatchStrOutput,
    BatchTokenIDOutput,
)
from sglang.srt.managers.multi_tokenizer_mixin import (
    TokenizerWorker,
    _handle_output_by_index,
    get_tokenizer_worker_class,
)
from sglang.srt.observability.req_time_stats import SchedulerReqTimeStats

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class CustomTokenizerWorker(TokenizerWorker):
    pass


class NotAWorker:
    pass


class DefaultServerArgs:
    def get_tokenizer_worker_class(self):
        return TokenizerWorker


class CustomServerArgs:
    def get_tokenizer_worker_class(self):
        return CustomTokenizerWorker


class InvalidServerArgs:
    def get_tokenizer_worker_class(self):
        return NotAWorker


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
    )


class TestMultiTokenizerMixin(CustomTestCase):
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

    def test_get_tokenizer_worker_class_uses_default(self):
        self.assertIs(get_tokenizer_worker_class(DefaultServerArgs()), TokenizerWorker)

    def test_get_tokenizer_worker_class_resolves_custom_class(self):
        self.assertIs(
            get_tokenizer_worker_class(CustomServerArgs()),
            CustomTokenizerWorker,
        )

    def test_get_tokenizer_worker_class_rejects_non_worker(self):
        with self.assertRaisesRegex(TypeError, "TokenizerWorker"):
            get_tokenizer_worker_class(InvalidServerArgs())


def _make_fanout_output(kind, count=32):
    """Populate each request field and include both wrapped metadata fields."""
    template = _make_batch_str_output()
    fields = {}
    for name in kind.__struct_fields__:
        value = getattr(template, name, None)
        fields[name] = (
            [copy.deepcopy(value[i % len(value)]) for i in range(count)]
            if isinstance(value, list)
            else value
        )
    stats = []
    for i in range(count):
        stat = SchedulerReqTimeStats(enable_metrics=True)
        stat.wait_queue_entry_time = 10 + i / 1000
        stat.prefill_finished_time = 14 + i / 1000
        stats.append(stat)
    fields.update(
        rids=[f"rid-{i}" for i in range(count)],
        http_worker_ipcs=[f"worker-{i % 4}" for i in range(count)],
        time_stats=io_struct.wrap_as_pickle(stats),
    )
    if kind is BatchEmbeddingOutput:
        fields["embeddings"] = [[float(i), 0.5] for i in range(count)]
    else:
        fields["output_ids"] = [array("q", [i, 7]) for i in range(count)]
        fields["customized_info"] = io_struct.wrap_as_pickle(
            {
                "metric": [float(i) for i in range(count)],
                "opaque": [complex(i, 1) for i in range(count)],
            }
        )
        if kind is BatchTokenIDOutput:
            fields.update(
                decoded_texts=[f"text-{i}" for i in range(count)],
                decode_ids=[array("q", [i, 7]) for i in range(count)],
                read_offsets=[0] * count,
                skip_special_tokens=[True] * count,
                spaces_between_special_tokens=[True] * count,
                no_stop_trim=[False] * count,
            )
    return kind(**fields)


def _encode_output(output):
    """Compare all fields through the selected native wire representation."""
    if io_struct._USE_PICKLE_IPC:
        return pickle.dumps(output, protocol=pickle.HIGHEST_PROTOCOL)
    return io_struct.msgpack_encode(output)


class TestFanoutMetadata(CustomTestCase):
    def _run_fanout(self, output, stage, send):
        if stage == "router":
            router = multi_tokenizer_mixin.MultiDetokenizerRouter.__new__(
                multi_tokenizer_mixin.MultiDetokenizerRouter
            )
            router.recv_from_scheduler = object()
            router.ipc_name_list = ["detok-a", "detok-b"]
            router.num_workers = 2
            router._send = send
            with (
                patch.object(
                    multi_tokenizer_mixin,
                    "sock_recv",
                    side_effect=[output, StopIteration],
                ),
                self.assertRaises(StopIteration),
            ):
                router.event_loop()
        else:
            manager = SimpleNamespace(
                recv_from_scheduler=object(),
                soft_watchdog=SimpleNamespace(disable=nullcontext, feed=lambda: None),
                _request_dispatcher=lambda _message: output,
            )
            mapping = SimpleNamespace(
                send_output=lambda ipc, row, is_tokenizer: send(ipc, row)
            )
            with (
                patch.object(
                    multi_tokenizer_mixin,
                    "sock_recv",
                    side_effect=[output, StopIteration],
                ),
                patch.object(
                    multi_tokenizer_mixin, "SocketMapping", return_value=mapping
                ),
                self.assertRaises(StopIteration),
            ):
                multi_tokenizer_mixin.MultiHttpWorkerDetokenizerMixin.multi_http_worker_event_loop(
                    manager
                )

    def test_fanout_preserves_wire_fields_and_bounds_metadata_decoding(self):
        for stage in ("router", "http"):
            for kind in (BatchTokenIDOutput, BatchStrOutput, BatchEmbeddingOutput):
                with self.subTest(stage=stage, kind=kind.__name__):
                    batch = _make_fanout_output(kind)
                    before = _encode_output(batch)
                    expected = []
                    for i in range(len(batch.rids)):
                        row = _handle_output_by_index(batch, i)
                        if stage == "router":
                            row.http_worker_ipcs = [batch.http_worker_ipcs[i]]
                        expected.append(_encode_output(row))
                    sent = []
                    with patch.object(
                        multi_tokenizer_mixin,
                        "unwrap_from_pickle",
                        wraps=io_struct.unwrap_from_pickle,
                    ) as unwrap:
                        self._run_fanout(
                            batch, stage, lambda target, row: sent.append((target, row))
                        )
                    decoded_wrappers = sum(
                        isinstance(call.args[0], io_struct.PickleWrapper)
                        for call in unwrap.call_args_list
                    )
                    self.assertEqual(
                        decoded_wrappers,
                        0
                        if io_struct._USE_PICKLE_IPC or kind is BatchEmbeddingOutput
                        else 4,
                    )
                    self.assertEqual([_encode_output(row) for _, row in sent], expected)
                    self.assertEqual(_encode_output(batch), before)
                    if stage == "http":
                        self.assertEqual(
                            [target for target, _ in sent], batch.http_worker_ipcs
                        )
                    if not io_struct._USE_PICKLE_IPC:
                        for _, row in sent:
                            decoded = io_struct.msgpack_decode(_encode_output(row))
                            self.assertEqual(
                                _encode_output(decoded), _encode_output(row)
                            )

    def test_first_send_precedes_metadata_reuse(self):
        for stage in ("router", "http"):
            with self.subTest(stage=stage):
                batch = _make_fanout_output(BatchStrOutput)
                sent = []
                with patch.object(
                    multi_tokenizer_mixin,
                    "_materialize_fanout_metadata",
                    wraps=multi_tokenizer_mixin._materialize_fanout_metadata,
                ) as prepare:

                    def send(target, row):
                        if not sent:
                            self.assertEqual(prepare.call_count, 0)
                        sent.append(row.rids[0])

                    self._run_fanout(batch, stage, send)
                self.assertEqual(sent, batch.rids)

    def test_metadata_reuse_leaves_no_op_cases_untouched(self):
        for kind in (BatchTokenIDOutput, BatchStrOutput, BatchEmbeddingOutput):
            for count in (0, 1):
                with self.subTest(kind=kind.__name__, count=count):
                    batch = _make_fanout_output(kind, count)
                    self.assertIs(
                        multi_tokenizer_mixin._materialize_fanout_metadata(batch), batch
                    )
        batch = _make_fanout_output(BatchStrOutput)
        batch.time_stats = io_struct.unwrap_from_pickle(batch.time_stats)
        batch.customized_info = io_struct.unwrap_from_pickle(batch.customized_info)
        self.assertIs(multi_tokenizer_mixin._materialize_fanout_metadata(batch), batch)
        request = io_struct.AbortReq(rid="abort", http_worker_ipc="worker-0")
        self.assertIs(
            multi_tokenizer_mixin._materialize_fanout_metadata(request), request
        )

    def test_partial_materialization_preserves_original_fields(self):
        batch = _make_fanout_output(BatchStrOutput)
        batch.customized_info = io_struct.unwrap_from_pickle(batch.customized_info)
        time_stats = batch.time_stats
        customized_info = batch.customized_info
        prepared = multi_tokenizer_mixin._materialize_fanout_metadata(batch)
        self.assertIs(batch.time_stats, time_stats)
        self.assertIs(batch.customized_info, customized_info)
        for i in range(len(batch.rids)):
            self.assertEqual(
                _encode_output(_handle_output_by_index(prepared, i)),
                _encode_output(_handle_output_by_index(batch, i)),
            )

    def test_embedding_ignored_metadata_stays_ignored(self):
        batch = _make_fanout_output(BatchEmbeddingOutput)
        batch.time_stats = io_struct.PickleWrapper(data=b"invalid-unused-metadata")
        for stage in ("router", "http"):
            with self.subTest(stage=stage):
                sent = []
                self._run_fanout(batch, stage, lambda target, row: sent.append(row))
                self.assertEqual(len(sent), len(batch.rids))
                self.assertTrue(all(row.time_stats is None for row in sent))

    def test_router_late_invalid_route_keeps_delivered_prefix(self):
        batch = _make_fanout_output(BatchTokenIDOutput)
        batch.http_worker_ipcs[5] = 42
        sent = []
        with self.assertRaises(AttributeError):
            self._run_fanout(batch, "router", lambda target, row: sent.extend(row.rids))
        self.assertEqual(sent, batch.rids[:5])

    def test_send_failure_is_not_retried(self):
        for stage in ("router", "http"):
            with self.subTest(stage=stage):
                batch = _make_fanout_output(BatchStrOutput)
                sent = []

                def send(target, row):
                    sent.append(row.rids[0])
                    if len(sent) == 3:
                        raise RuntimeError("send failed")

                with self.assertRaisesRegex(RuntimeError, "send failed"):
                    self._run_fanout(batch, stage, send)
                self.assertEqual(sent, batch.rids[:3])


if __name__ == "__main__":
    unittest.main()
