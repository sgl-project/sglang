"""Cache admission accounting must survive IPC, routing, and native metadata.

These checks use the real producer/consumer methods without loading a model or
allocating GPU memory. Unknown accounting stays absent, rather than becoming a
billable-looking zero, and stream chunks carry cumulative snapshots.
"""

import asyncio
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import patch

import msgspec

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode  # noqa: E402
from sglang.srt.managers.detokenizer_manager import DetokenizerManager  # noqa: E402
from sglang.srt.managers.io_struct import BatchStrOutput  # noqa: E402
from sglang.srt.managers.io_struct import (
    BatchTokenIDOutput,
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
        prompt_tokens=[20] * count,
        reasoning_tokens=[0] * count,
        completion_tokens=[1] * count,
        cached_tokens=[4] * count,
        cache_write_tokens=values,
        retraction_counts=[0] * count,
    )
    if cls is BatchTokenIDOutput:
        fields.update(
            decoded_texts=[""] * count,
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
        for value in (8, 16):
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
                get_cached_tokens_details=lambda req: None,
                is_request_level_customized_info_key=lambda key: False,
                current_weight_version=None,
            )
            acc.accept(req=req)
            snapshots.append(acc.to_payload(dp_rank=0, is_idle_batch=False))
        self.assertEqual([x.cache_write_tokens for x in snapshots], [[8], [16]])


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


if __name__ == "__main__":
    unittest.main()
