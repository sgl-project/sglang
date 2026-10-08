import pickle
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import msgspec
import numpy as np

from sglang.srt.sampling.sampling_mask import SamplingMaskChunk
from sglang.srt.utils.weight_versions import WeightVersionSpan
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.beam_search.types import BeamSearchSequence
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import (
    BatchEmbeddingOutput,
    BatchStrOutput,
    BatchTokenIDOutput,
    BeamSearchOutput,
    msgpack_decode,
    msgpack_encode,
)
from sglang.srt.managers.multi_tokenizer_mixin import (
    MultiDetokenizerRouter,
    MultiHttpWorkerDetokenizerMixin,
    TokenizerWorker,
    _handle_output_by_index,
    _handle_output_by_indices,
    _send_output_to_tokenizer_workers,
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
        output_ids=[array("i", [1]), array("i", [2])],
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
        output_ids=[array("i", [1]), array("i", [2])],
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

    def test_return_path_preserves_noncontiguous_owner_order_and_wire_fields(self):
        output = _make_batch_str_output()
        for field_name in msgspec.structs.fields(BatchStrOutput):
            values = object.__getattribute__(output, field_name.name)
            if isinstance(values, list):
                values.append(values[0])
        output.rids = ["rid-0", "rid-1", "rid-2"]
        output.output_strs[2] = "third"
        output.output_ids[2] = array("i", [3])
        output.finished_reasons[2] = {"type": "length"}
        output.weight_versions[2] = [WeightVersionSpan(version="v3", start=0, end=1)]
        output.token_steps[2] = [3]
        ipcs = ["worker-b", "worker-a", "worker-b"]
        output.http_worker_ipcs = ipcs[:]
        original = pickle.dumps(output)
        sent = []
        mapping = SimpleNamespace(
            send_output=lambda target, obj, *, is_tokenizer: sent.append(
                (target, obj, is_tokenizer)
            )
        )

        for _ in range(2):
            _send_output_to_tokenizer_workers(
                output=output, ipcs=ipcs, socket_mapping=mapping, group_send=True
            )

        self.assertEqual(
            [(target, obj.rids, tokenizer) for target, obj, tokenizer in sent],
            [
                ("worker-b", ["rid-0", "rid-2"], True),
                ("worker-a", ["rid-1"], True),
            ]
            * 2,
        )
        self.assertEqual(pickle.dumps(output), original)
        for encode, decode in (
            (pickle.dumps, pickle.loads),
            (msgpack_encode, msgpack_decode),
        ):
            with self.subTest(codec=encode.__name__):
                group = decode(encode(sent[0][1]))
                self.assertEqual(group.rids, ["rid-0", "rid-2"])
                self.assertEqual(group.http_worker_ipcs, ["worker-b", "worker-b"])
                self.assertEqual(group.output_strs, ["first", "third"])
                self.assertEqual(group.output_ids, [array("i", [1]), array("i", [3])])
                self.assertEqual(group.finished_reasons, [None, {"type": "length"}])
                self.assertEqual(
                    group.weight_versions[1],
                    [WeightVersionSpan(version="v3", start=0, end=1)],
                )
                self.assertEqual(group.token_steps, [[1], [3]])
                self.assertEqual(group.beam_search_output[1].sequences[0].tokens, [11])
                self.assertEqual(
                    group.output_token_sampling_mask[1].to_lists(support_logprobs=True),
                    ([[7, 8]], [[-0.5, -1.0]]),
                )

    def test_return_path_unique_owners_preserve_per_request_payloads(self):
        output = _make_batch_str_output()
        output.http_worker_ipcs = ["worker-b", "worker-a"]
        results = []
        for enabled in (False, True):
            sent = []
            mapping = SimpleNamespace(
                send_output=lambda target, obj, *, is_tokenizer: sent.append(
                    (target, pickle.dumps(obj), is_tokenizer)
                )
            )
            _send_output_to_tokenizer_workers(
                output=output,
                ipcs=output.http_worker_ipcs,
                socket_mapping=mapping,
                group_send=enabled,
            )
            self.assertEqual(
                [target for target, _, _ in sent], ["worker-b", "worker-a"]
            )
            self.assertEqual(
                [pickle.loads(payload).rids for _, payload, _ in sent],
                [["rid-0"], ["rid-1"]],
            )
            results.append(sent)
        self.assertEqual(results[0], results[1])

    def test_return_path_groups_token_output_and_falls_back_for_embedding(self):
        # Token output is covered by the shared splitter, so it is grouped;
        # embedding output is not, so the whole batch falls back to per-request.
        output = _make_batch_token_id_output()
        sent = []
        mapping = SimpleNamespace(
            send_output=lambda target, obj, *, is_tokenizer: sent.append(obj)
        )
        _send_output_to_tokenizer_workers(
            output=output,
            ipcs=["worker-a", "worker-a"],
            socket_mapping=mapping,
            group_send=True,
        )
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0].rids, output.rids)
        self.assertEqual(sent[0].http_worker_ipcs, ["worker-a", "worker-a"])
        self.assertEqual(sent[0].output_ids, [array("i", [1]), array("i", [2])])
        self.assertEqual(sent[0].decode_ids, [[1], [2]])
        self.assertEqual(sent[0].read_offsets, [0, 1])

        output = _make_batch_embedding_output()
        sent = []
        _send_output_to_tokenizer_workers(
            output=output,
            ipcs=["worker-a", "worker-a"],
            socket_mapping=mapping,
            group_send=True,
        )
        self.assertEqual(
            [obj.rids for obj in sent],
            [["embedding-rid-0"], ["embedding-rid-1"]],
        )

    def test_return_path_falls_back_before_sending_any_group(self):
        output = _make_batch_embedding_output()
        output.rids.append("embedding-rid-2")
        output.embeddings.append([0.5, 0.6])
        output.prompt_tokens.append(30)
        output.cached_tokens.append(5)
        output.finished_reasons.append(None)
        output.retraction_counts = [0, 0, 0]
        sent = []
        mapping = SimpleNamespace(
            send_output=lambda target, obj, *, is_tokenizer: sent.append((target, obj))
        )

        _send_output_to_tokenizer_workers(
            output=output,
            ipcs=["worker-b", "worker-a", "worker-a"],
            socket_mapping=mapping,
            group_send=True,
        )

        self.assertEqual(
            [(target, obj.rids) for target, obj in sent],
            [
                ("worker-b", ["embedding-rid-0"]),
                ("worker-a", ["embedding-rid-1"]),
                ("worker-a", ["embedding-rid-2"]),
            ],
        )

    def test_return_path_send_error_does_not_resend_or_fall_back(self):
        sent = []

        def send_then_fail(target, obj, *, is_tokenizer):
            sent.append((target, obj.rids))
            if target == "worker-a":
                raise RuntimeError("send failed")

        with self.assertRaisesRegex(RuntimeError, "send failed"):
            _send_output_to_tokenizer_workers(
                output=_handle_output_by_indices(_make_batch_str_output(), [0, 1, 0]),
                ipcs=["worker-b", "worker-b", "worker-a"],
                socket_mapping=SimpleNamespace(send_output=send_then_fail),
                group_send=True,
            )
        self.assertEqual(
            sent, [("worker-b", ["rid-0", "rid-1"]), ("worker-a", ["rid-0"])]
        )

    def test_return_path_event_loop_honors_switch_and_default_off(self):
        self.assertFalse(envs.SGLANG_OPT_DETOK_TOKENIZER_GROUP_SEND.default)
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                output = _make_batch_str_output()
                output.http_worker_ipcs = ["worker-a", "worker-a"]
                worker = MultiHttpWorkerDetokenizerMixin()
                worker.recv_from_scheduler = object()
                worker.soft_watchdog = MagicMock()
                worker._request_dispatcher = lambda obj: obj
                sent = []
                mapping = SimpleNamespace(
                    send_output=lambda target, obj, *, is_tokenizer: sent.append(obj)
                )
                with (
                    envs.SGLANG_OPT_DETOK_TOKENIZER_GROUP_SEND.override(enabled),
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
                self.assertEqual(
                    [obj.rids for obj in sent],
                    [["rid-0", "rid-1"]] if enabled else [["rid-0"], ["rid-1"]],
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
