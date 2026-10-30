"""Cache admission accounting must survive IPC, routing, and native metadata.

These checks use the real producer/consumer methods without loading a model or
allocating GPU memory. Unknown accounting stays absent, rather than becoming a
billable-looking zero, and stream chunks carry cumulative snapshots.
"""

import asyncio
import copy
import pickle
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
from pydantic import TypeAdapter

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode  # noqa: E402
from sglang.srt.managers.detokenizer_manager import DetokenizerManager  # noqa: E402
from sglang.srt.managers.io_struct import BatchStrOutput  # noqa: E402
from sglang.srt.managers.io_struct import (
    BatchTokenIDOutput,
    GenerateReqInput,
    RequestOutputStats,
    TokenizedGenerateReqInput,
    msgpack_decode,
    msgpack_encode,
)
from sglang.srt.managers.multi_tokenizer_mixin import (  # noqa: E402
    _handle_output_by_index,
)
from sglang.srt.managers.scheduler_components.output_streamer import (  # noqa: E402
    _GenerationStreamAccumulator,
)
from sglang.srt.managers.tokenizer_manager import ReqState  # noqa: E402
from sglang.srt.managers.tokenizer_manager import (
    TokenizerManager,
)
from sglang.srt.sampling.sampling_params import SamplingParams  # noqa: E402
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _batch(cls, values):
    count = len(values)
    fields = {
        f.name: None
        for f in msgspec.structs.fields(cls)
        if f.default is msgspec.NODEFAULT and f.default_factory is msgspec.NODEFAULT
    }
    fields.update(
        rids=[f"cache-{i}" for i in range(count)],
        http_worker_ipcs=[None] * count,
        finished_reasons=[None] * count,
        output_ids=[array("i", [1])] * count,
        reasoning_tokens=[0] * count,
        completion_tokens=[1] * count,
        cache_write_tokens=values,
        request_stats=[
            RequestOutputStats(
                prompt_tokens=20,
                cached_tokens=4,
                cached_tokens_details={"device": 2, "host": 2},
                retraction_count=0,
                image_tokens=0,
                audio_tokens=0,
                video_tokens=0,
                time_stats=None,
            )
            for _ in values
        ],
    )
    if cls is BatchTokenIDOutput:
        fields.update(
            decode_ids=[array("i", [1])] * count,
            read_offsets=[0] * count,
            skip_special_tokens=[True] * count,
            spaces_between_special_tokens=[True] * count,
            no_stop_trim=[False] * count,
        )
    else:
        fields["output_strs"] = ["hello"] * count
    return cls(**fields)


def _detokenize(payload):
    manager = DetokenizerManager.__new__(DetokenizerManager)
    manager._decode_batch_token_id_output = lambda obj: ["hello"] * len(obj.rids)
    return manager.handle_batch_token_id_out(payload)


class TestCacheWriteTransport(CustomTestCase):
    def test_unknown_zero_and_positive_survive_real_transport(self):
        token_output = _batch(BatchTokenIDOutput, [None, 0, 16])
        token_output = msgpack_decode(msgpack_encode(token_output))
        str_output = _detokenize(token_output)
        for payload in (token_output, str_output):
            with self.subTest(payload=type(payload).__name__):
                received = msgpack_decode(msgpack_encode(payload))
                self.assertEqual(received.cache_write_tokens, [None, 0, 16])
                for index, value in enumerate([None, 0, 16]):
                    routed = _handle_output_by_index(received, index)
                    routed = msgpack_decode(msgpack_encode(routed))
                    self.assertEqual(routed.rids, [f"cache-{index}"])
                    self.assertEqual(routed.cache_write_tokens, [value])

    def test_old_ipc_without_tail_field_decodes_as_unknown(self):
        for cls in (BatchTokenIDOutput, BatchStrOutput):
            with self.subTest(payload=cls.__name__):
                self.assertEqual(cls.__struct_fields__[-1], "cache_write_tokens")
                wire = msgspec.msgpack.decode(msgpack_encode(_batch(cls, [16])))
                wire.pop()  # old array-like IPC has no newly appended field
                decoded = msgpack_decode(msgspec.msgpack.encode(wire))
                self.assertIsNone(decoded.cache_write_tokens)
                self.assertIsNone(
                    _handle_output_by_index(decoded, 0).cache_write_tokens
                )

    def test_scheduler_reports_cumulative_snapshots(self):
        req = SimpleNamespace(
            rid="cache-stream",
            http_worker_ipc=None,
            finished=lambda: False,
            beam_group=None,
            stream=True,
            sampling_params=SimpleNamespace(
                stream_interval=1,
                skip_special_tokens=True,
                spaces_between_special_tokens=True,
                no_stop_trim=False,
            ),
            output_ids=[1],
            output_ids_through_stop=[1],
            send_token_offset=0,
            send_output_token_logprobs_offset=0,
            send_decode_id_offset=0,
            origin_input_ids=[9] * 20,
            finished_reason=None,
            decoded_text="",
            reasoning_tokens=0,
            cached_tokens=4,
            cache_write_tokens=8,
            get_mm_token_counts=lambda: (0, 0, 0),
            take_request_stats_due=lambda: True,
            mm_image_tokens=0,
            mm_audio_tokens=0,
            mm_video_tokens=0,
            multimodal_inputs=None,
            retraction_count=0,
            time_stats=None,
            customized_info=None,
            check_match_stop_str_prefix=lambda: False,
            init_incremental_detokenize=lambda: ([1], 0),
        )
        snapshots = []
        for value in (8, 16, 16):
            req.cache_write_tokens = value
            req.output_ids.append(2)
            req.output_ids_through_stop = req.output_ids
            acc = _GenerationStreamAccumulator(
                return_logprob=False,
                return_hidden_states=False,
                return_routed_experts=False,
                return_indexer_topk=False,
                spec_algorithm=SpeculativeAlgorithm.NONE,
                disaggregation_mode=DisaggregationMode.NULL,
                default_stream_interval=1,
                default_force_stream_interval=1,
                build_request_stats=lambda req: RequestOutputStats(
                    prompt_tokens=len(req.origin_input_ids),
                    cached_tokens=req.cached_tokens,
                    cached_tokens_details={"device": 2, "host": 2},
                    retraction_count=req.retraction_count,
                    image_tokens=0,
                    audio_tokens=0,
                    video_tokens=0,
                    time_stats=None,
                ),
                is_request_level_customized_info_key=lambda key: False,
                current_weight_version=None,
            )
            acc.accept(req=req)
            snapshots.append(acc.to_payload(dp_rank=0, is_idle_batch=False))
        self.assertEqual([x.cache_write_tokens for x in snapshots], [[8], [16], [16]])
        self.assertEqual(
            [x.request_stats[0].cached_tokens for x in snapshots], [4, 4, 4]
        )


class TestCacheWriteNativeMetadata(unittest.IsolatedAsyncioTestCase, CustomTestCase):
    async def test_native_metadata_omits_unknown_and_keeps_measured_zero(self):
        for cls in (BatchTokenIDOutput, BatchStrOutput):
            for value in (None, 0, 16):
                with self.subTest(payload=cls.__name__, value=value):
                    manager = TokenizerManager.__new__(TokenizerManager)
                    manager.enable_metrics = False
                    manager.incremental_streaming_output = True
                    manager.config_value = lambda key: "test"
                    manager.dump_requests_folder = ""
                    manager.crash_dump_folder = ""
                    state = ReqState(
                        out_list=[],
                        finished=False,
                        event=asyncio.Event(),
                        obj=SimpleNamespace(stream=True, return_logprob=False),
                        time_stats=SimpleNamespace(first_token_time=1.0),
                    )
                    manager.rid_to_state = {"cache-0": state}
                    with patch(
                        "sglang.srt.managers.tokenizer_manager.get_serving",
                        return_value=SimpleNamespace(batch_notify_size=1),
                    ):
                        await manager._handle_batch_output(_batch(cls, [value]))
                    meta = state.out_list[-1]["meta_info"]
                    if value is None:
                        self.assertNotIn("cache_write_tokens", meta)
                    else:
                        self.assertEqual(meta["cache_write_tokens"], value)

    async def test_cached_stats_stay_physical_while_metadata_uses_public_window(self):
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.enable_metrics = False
        manager.incremental_streaming_output = True
        manager.config_value = lambda key: "test"
        manager.dump_requests_folder = ""
        manager.crash_dump_folder = ""
        obj = GenerateReqInput(input_ids=list(range(20)), stream=True)
        obj._cache_write_reporting_suffix_tokens = 3
        state = ReqState(
            out_list=[],
            finished=False,
            event=asyncio.Event(),
            obj=obj,
            time_stats=SimpleNamespace(first_token_time=1.0),
        )
        manager.rid_to_state = {"cache-0": state}
        stats = RequestOutputStats(
            prompt_tokens=20,
            cached_tokens=19,
            cached_tokens_details={"device": 19, "host": 0},
            retraction_count=0,
            image_tokens=0,
            audio_tokens=0,
            video_tokens=0,
            time_stats=None,
        )
        for index, value in enumerate((0, 1, None)):
            payload = _batch(BatchStrOutput, [value])
            payload.request_stats = [stats] if index != 1 else None
            with patch(
                "sglang.srt.managers.tokenizer_manager.get_serving",
                return_value=SimpleNamespace(batch_notify_size=1),
            ):
                await manager._handle_batch_output(payload)
            meta = state.out_list[-1]["meta_info"]
            self.assertEqual(meta["prompt_tokens"], 20)
            self.assertEqual(meta["cached_tokens"], 17)
            self.assertIsNone(meta["cached_tokens_details"])
            if value is None:
                self.assertNotIn("cache_write_tokens", meta)
            else:
                self.assertEqual(meta["cache_write_tokens"], value)
            self.assertEqual(state.request_stats.cached_tokens, 19)
            self.assertEqual(
                state.request_stats.cached_tokens_details, {"device": 19, "host": 0}
            )


class TestCacheWriteReportingWindowTransport(CustomTestCase):
    def test_native_json_cannot_select_reporting_window(self):
        adapter = TypeAdapter(GenerateReqInput)
        name = "_cache_write_reporting_suffix_tokens"
        native = adapter.validate_python({"input_ids": [1, 2, 3], name: 3})
        self.assertEqual(native._cache_write_reporting_suffix_tokens, 0)
        self.assertNotIn(name, adapter.json_schema()["properties"])
        with self.assertRaises(TypeError):
            GenerateReqInput(input_ids=[1, 2, 3], **{name: 3})

    def test_batch_and_parallel_copies_keep_only_server_selected_policy(self):
        batch = GenerateReqInput(
            input_ids=[[1, 2, 3, 4], [5, 6, 7, 8]], sampling_params={"n": 2}
        )
        batch._cache_write_reporting_suffix_tokens = 3
        batch.normalize_batch_and_arguments()
        for i in range(len(batch.input_ids)):
            item = batch[i]
            self.assertIs(batch[i], item)
            for copied in (item, copy.copy(item), pickle.loads(pickle.dumps(item))):
                self.assertEqual(copied._cache_write_reporting_suffix_tokens, 3)
        self.assertEqual(GenerateReqInput()._cache_write_reporting_suffix_tokens, 0)

    def test_policy_crosses_tokenizer_and_append_only_ipc(self):
        obj = GenerateReqInput(
            input_ids=[1, 2, 3, 4],
            sampling_params={"temperature": 0, "max_new_tokens": 1},
        )
        obj._cache_write_reporting_suffix_tokens = 3
        obj.normalize_batch_and_arguments()
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.preferred_sampling_params = None
        manager.sampling_params_class = SamplingParams
        manager.tokenized_generate_req_type = TokenizedGenerateReqInput
        manager.tokenizer = None
        manager.model_config = SimpleNamespace(vocab_size=100)
        manager.rid_to_state = {
            obj.rid: SimpleNamespace(
                time_stats=SimpleNamespace(set_tokenize_finish_time=str)
            )
        }
        with patch(
            "sglang.srt.managers.tokenizer_manager.get_disagg",
            return_value=SimpleNamespace(disaggregation_transfer_backend="none"),
        ):
            tokenized = manager._create_tokenized_object(obj, None, obj.input_ids)
        self.assertEqual(tokenized.cache_write_reporting_suffix_tokens, 3)
        tokenized.wrap_pickle_fields()
        encoded = msgpack_encode(tokenized)
        self.assertEqual(msgpack_decode(encoded).cache_write_reporting_suffix_tokens, 3)
        self.assertEqual(
            pickle.loads(pickle.dumps(tokenized)).cache_write_reporting_suffix_tokens, 3
        )
        self.assertEqual(
            TokenizedGenerateReqInput.__struct_fields__[-1],
            "cache_write_reporting_suffix_tokens",
        )
        legacy = msgspec.msgpack.decode(encoded)
        legacy.pop()
        self.assertEqual(
            msgpack_decode(
                msgspec.msgpack.encode(legacy)
            ).cache_write_reporting_suffix_tokens,
            0,
        )

    def test_responses_constructs_requests_with_the_same_reporting_policy(self):
        from sglang.srt.entrypoints.openai.protocol import ResponsesRequest
        from sglang.srt.entrypoints.openai.serving_responses import (
            OpenAIServingResponses,
        )

        async def run(encoding, stream, uses_response_template):
            server = OpenAIServingResponses.__new__(OpenAIServingResponses)
            server.chat_encoding_spec = encoding
            server.tokenizer_manager = SimpleNamespace(
                tokenizer=SimpleNamespace(decode=lambda *a, **k: "rendered prefix")
            )
            server.enable_response_store = False
            server._translates_tools = False
            server.use_harmony = False
            server.tool_server = None
            server.tool_call_parser = None
            server.default_sampling_params = {}
            server._validate_model = lambda model: None
            server.extract_routed_dp_rank_from_header = lambda raw, rank: rank
            processed = SimpleNamespace(
                stop=None,
                tool_call_constraint=None,
                reasoning_end_token_ids=None,
                skip_special_tokens=True,
                image_data=None,
                video_data=None,
                audio_data=None,
                modalities=None,
                require_reasoning=False,
                uses_response_template=uses_response_template,
                template_tool_call_grammar=False,
                message_start_tool_call_constraint=None,
            )

            async def make_request(*args, **kwargs):
                return [], [list(range(512))], [list(range(512))], processed

            captured = []

            def generate(request_id, prompt, adapted, *args, **kwargs):
                captured.append(adapted)

                async def empty():
                    if False:
                        yield None

                return empty()

            async def full(*args, **kwargs):
                return "full"

            server._make_request = make_request
            server._generate_with_builtin_tools = generate
            server.responses_stream_generator_non_harmony = lambda *a, **k: "stream"
            server.responses_full_generator = full
            request = ResponsesRequest(
                model="test", input="hello", stream=stream, store=False
            )
            result = await server.create_responses(request)
            self.assertEqual(result, "stream" if stream else "full")
            self.assertEqual(len(captured), 1)
            self.assertEqual(len(captured[0].input_ids), 512)
            self.assertEqual(
                captured[0]._cache_write_reporting_suffix_tokens,
                3 if encoding == "kimi_k3" else 0,
            )
            if uses_response_template:
                self.assertEqual(request._response_parser_prefix, "rendered prefix")

        for encoding in ("kimi_k3", None):
            for stream in (False, True):
                for uses_response_template in (False, True):
                    with self.subTest(
                        encoding=encoding,
                        stream=stream,
                        uses_response_template=uses_response_template,
                    ):
                        asyncio.run(run(encoding, stream, uses_response_template))

    def test_responses_tool_retry_keeps_the_server_reporting_policy(self):
        from sglang.srt.entrypoints.openai.serving_responses import (
            OpenAIServingResponses,
        )

        async def run(encoding):
            server = OpenAIServingResponses.__new__(OpenAIServingResponses)
            server.chat_encoding_spec = encoding
            server.is_disaggregated = False
            captured = []

            async def generate(adapted, raw):
                captured.append(adapted)
                yield {"text": "tool turn"}

            async def call_tool():
                return {"text": "tool result"}

            server.tokenizer_manager = SimpleNamespace(
                generate_request=generate,
                model_config=SimpleNamespace(context_len=4096),
                num_reserved_tokens=1,
            )
            context = SimpleNamespace(
                append_output=lambda output: None,
                need_builtin_tool_call=lambda: len(captured) == 1,
                call_tool=call_tool,
                render_for_completion=lambda: list(range(512)),
            )
            initial = GenerateReqInput(input_ids=list(range(512)))
            server._set_cache_reporting_window(initial)
            async for _ in server._generate_with_builtin_tools(
                "test", initial.input_ids, initial, {"max_new_tokens": 1}, context
            ):
                pass
            self.assertEqual(len(captured), 2)
            self.assertEqual(
                [item._cache_write_reporting_suffix_tokens for item in captured],
                [3 if encoding == "kimi_k3" else 0] * 2,
            )

        for encoding in ("kimi_k3", None):
            with self.subTest(encoding=encoding):
                asyncio.run(run(encoding))


if __name__ == "__main__":
    unittest.main()
