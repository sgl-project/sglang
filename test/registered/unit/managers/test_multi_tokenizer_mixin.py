import unittest
from array import array
from unittest.mock import Mock

import numpy as np
import zmq

from sglang.srt.sampling.sampling_mask import SamplingMaskChunk
from sglang.srt.utils.weight_versions import WeightVersionSpan
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.beam_search.types import BeamSearchSequence
from sglang.srt.managers.io_struct import (
    BatchEmbeddingOutput,
    BatchStrOutput,
    BatchTokenIDOutput,
    BeamSearchOutput,
    sock_recv,
    unwrap_from_pickle,
    wrap_as_pickle,
)
from sglang.srt.managers.multi_tokenizer_mixin import (
    MultiDetokenizerRouter,
    MultiHttpWorkerDetokenizerMixin,
    SocketMapping,
    TokenizerWorker,
    _handle_output_by_index,
    _handle_output_by_indices,
    get_tokenizer_worker_class,
)

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


def _make_detokenizer() -> MultiHttpWorkerDetokenizerMixin:
    manager = MultiHttpWorkerDetokenizerMixin()
    manager.socket_mapping = Mock()
    return manager


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

    def test_detokenizer_groups_by_destination_and_preserves_owner_ipcs(self):
        output = _make_batch_str_output()
        manager = _make_detokenizer()

        manager._send_batch_output(
            output, ["worker-a", "worker-a"], enable_batching=True
        )

        manager.socket_mapping.send_output.assert_called_once()
        ipc_name, grouped = manager.socket_mapping.send_output.call_args.args
        self.assertEqual(ipc_name, "worker-a")
        self.assertEqual(grouped.rids, ["rid-0", "rid-1"])
        self.assertEqual(grouped.http_worker_ipcs, ["worker-a", "worker-a"])
        self.assertTrue(
            manager.socket_mapping.send_output.call_args.kwargs["is_tokenizer"]
        )
        self.assertEqual(output.rids, ["rid-0", "rid-1"])

    def test_detokenizer_keeps_first_destination_order(self):
        output = _make_batch_str_output()
        manager = _make_detokenizer()

        manager._send_batch_output(
            output, ["worker-b", "worker-a"], enable_batching=True
        )

        self.assertEqual(
            [
                (call.args[0], call.args[1].rids, call.args[1].http_worker_ipcs)
                for call in manager.socket_mapping.send_output.call_args_list
            ],
            [
                ("worker-b", ["rid-0"], ["worker-b"]),
                ("worker-a", ["rid-1"], ["worker-a"]),
            ],
        )

    def test_detokenizer_groups_token_output_without_speculative_decoding(self):
        output = _make_batch_token_id_output()
        manager = _make_detokenizer()

        manager._send_batch_output(
            output, ["worker-a", "worker-a"], enable_batching=True
        )

        manager.socket_mapping.send_output.assert_called_once()
        grouped = manager.socket_mapping.send_output.call_args.args[1]
        self.assertEqual(grouped.rids, ["token-rid-0", "token-rid-1"])
        self.assertEqual(grouped.decoded_texts, ["first", "second"])
        self.assertIsNone(grouped.spec_verify_ct)

    def test_detokenizer_grouped_outputs_round_trip(self):
        output = _make_batch_str_output()
        for name in output.__struct_fields__:
            value = getattr(output, name)
            if isinstance(value, list):
                setattr(output, name, value * 2)
        output.rids = ["rid-0", "rid-1", "rid-2", "rid-3"]
        output.output_ids = [array("q", [i]) for i in range(4)]
        output.customized_info = wrap_as_pickle({"score": [0, 1, 2, 3]})
        output.time_stats = wrap_as_pickle([0, 1, 2, 3])
        ipc_names = ["inproc://worker-b", "inproc://worker-a"] * 2
        output.http_worker_ipcs = ipc_names.copy()

        manager = MultiHttpWorkerDetokenizerMixin()
        manager.socket_mapping = SocketMapping()
        context = manager.socket_mapping._zmq_context
        self.addCleanup(context.destroy, linger=0)
        self.addCleanup(manager.socket_mapping.clear_all_sockets)
        context.setsockopt(zmq.LINGER, 0)
        context.setsockopt(zmq.SNDTIMEO, 1000)
        receivers = {}
        for ipc_name in dict.fromkeys(ipc_names):
            receiver = context.socket(zmq.PULL)
            self.addCleanup(receiver.close, linger=0)
            receiver.setsockopt(zmq.RCVTIMEO, 1000)
            receiver.bind(ipc_name)
            receivers[ipc_name] = receiver

        manager._send_batch_output(output, ipc_names, enable_batching=True)

        for ipc_name, indices in (
            ("inproc://worker-b", [0, 2]),
            ("inproc://worker-a", [1, 3]),
        ):
            received = sock_recv(receivers[ipc_name])
            self.assertEqual(received.rids, [output.rids[i] for i in indices])
            self.assertEqual(received.http_worker_ipcs, [ipc_name] * 2)
            self.assertEqual(
                received.output_ids, [output.output_ids[i] for i in indices]
            )
            self.assertEqual(
                received.output_strs, [output.output_strs[i] for i in indices]
            )
            self.assertEqual(
                received.finished_reasons, [output.finished_reasons[i] for i in indices]
            )
            self.assertEqual(
                received.weight_versions, [output.weight_versions[i] for i in indices]
            )
            self.assertEqual(
                unwrap_from_pickle(received.customized_info), {"score": indices}
            )
            self.assertEqual(unwrap_from_pickle(received.time_stats), indices)
            for offset, index in enumerate(indices):
                np.testing.assert_equal(
                    received.input_top_logprobs_val_flat[offset],
                    output.input_top_logprobs_val_flat[index],
                )
                self.assertEqual(
                    received.output_token_sampling_mask[offset].to_lists(True),
                    output.output_token_sampling_mask[index].to_lists(True),
                )
            with self.assertRaises(zmq.Again):
                sock_recv(receivers[ipc_name], flags=zmq.NOBLOCK)
        self.assertEqual(output.http_worker_ipcs, ipc_names)

    def test_detokenizer_falls_back_before_any_grouped_send(self):
        output = BatchEmbeddingOutput(
            rids=["rid-0", "rid-1", "rid-2"],
            finished_reasons=[None, None, None],
            embeddings=[[0.0], [1.0], [2.0]],
            prompt_tokens=[10, 20, 30],
            cached_tokens=[1, 2, 3],
            placeholder_tokens_idx=[None, None, None],
            placeholder_tokens_val=[None, None, None],
        )
        manager = _make_detokenizer()

        manager._send_batch_output(
            output,
            ["worker-a", "worker-b", "worker-b"],
            enable_batching=True,
        )

        self.assertEqual(
            [
                (call.args[0], call.args[1].rids)
                for call in manager.socket_mapping.send_output.call_args_list
            ],
            [
                ("worker-a", ["rid-0"]),
                ("worker-b", ["rid-1"]),
                ("worker-b", ["rid-2"]),
            ],
        )

    def test_detokenizer_batching_disabled_uses_per_request_sends(self):
        output = _make_batch_str_output()
        manager = _make_detokenizer()

        manager._send_batch_output(
            output, ["worker-a", "worker-a"], enable_batching=False
        )

        self.assertEqual(manager.socket_mapping.send_output.call_count, 2)
        self.assertEqual(
            [
                call.args[1].rids
                for call in manager.socket_mapping.send_output.call_args_list
            ],
            [["rid-0"], ["rid-1"]],
        )

    def test_detokenizer_send_error_propagates_without_resending(self):
        output = _make_batch_str_output()
        manager = _make_detokenizer()
        for successful_sends in (0, 1):
            with self.subTest(successful_sends=successful_sends):
                manager.socket_mapping.reset_mock()
                manager.socket_mapping.send_output.side_effect = [
                    *([None] * successful_sends),
                    RuntimeError("send failed"),
                ]

                with self.assertRaisesRegex(RuntimeError, "send failed"):
                    manager._send_batch_output(
                        output, ["worker-a", "worker-b"], enable_batching=True
                    )

                self.assertEqual(
                    [
                        (call.args[0], call.args[1].rids)
                        for call in manager.socket_mapping.send_output.call_args_list
                    ],
                    [
                        ("worker-a", ["rid-0"]),
                        ("worker-b", ["rid-1"]),
                    ][: successful_sends + 1],
                )

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


if __name__ == "__main__":
    unittest.main()
