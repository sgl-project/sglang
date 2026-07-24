import copy
import pickle
import unittest
from array import array
from unittest.mock import patch

import msgspec

import sglang.srt.observability.req_time_stats as req_time_stats_module
import sglang.srt.observability.trace as trace_module
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers import io_struct
from sglang.srt.managers.io_struct import (
    BatchEmbeddingOutput,
    BatchTokenIDOutput,
    BatchTokenizedEmbeddingReqInput,
    BatchTokenizedGenerateReqInput,
    GenerateReqInput,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    sock_recv,
    sock_send,
)
from sglang.srt.observability.req_time_stats import (
    APIServerReqTimeStats,
    DPControllerReqTimeStats,
    SchedulerReqTimeStats,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import (
    register_amd_ci,
    register_cpu_ci,
    register_cuda_ci,
)
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
)

register_cuda_ci(est_time=8, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=8, suite="stage-b-test-1-gpu-small-amd")
register_cpu_ci(est_time=8, suite="base-c-test-cpu")


class _LoopbackSocket:
    def send(self, data, flags=0):
        self.data = data

    def recv(self, flags=0):
        return self.data

    def send_pyobj(self, obj, flags=0, protocol=None):
        self.data = pickle.dumps(obj, protocol=protocol)

    def recv_pyobj(self, flags=0):
        return pickle.loads(self.data)


def _transport_round_trip(obj):
    socket = _LoopbackSocket()
    sock_send(socket, obj)
    return sock_recv(socket)


def _make_batch_token_id_output(time_stats):
    return BatchTokenIDOutput(
        rids=["rid"],
        http_worker_ipcs=[None],
        finished_reasons=[None],
        decoded_texts=[""],
        decode_ids=[array("i", [1])],
        read_offsets=[0],
        output_ids=[array("i", [1])],
        skip_special_tokens=[True],
        spaces_between_special_tokens=[True],
        no_stop_trim=[False],
        prompt_tokens=[1],
        reasoning_tokens=[0],
        completion_tokens=[1],
        cached_tokens=[0],
        input_token_logprobs_val=None,
        input_token_logprobs_idx=None,
        output_token_logprobs_val=None,
        output_token_logprobs_idx=None,
        input_top_logprobs_val=None,
        input_top_logprobs_idx=None,
        output_top_logprobs_val=None,
        output_top_logprobs_idx=None,
        input_token_ids_logprobs_val=None,
        input_token_ids_logprobs_idx=None,
        output_token_ids_logprobs_val=None,
        output_token_ids_logprobs_idx=None,
        output_token_entropy_val=None,
        output_token_sampling_mask=None,
        output_token_sampling_logprobs=None,
        output_hidden_states=None,
        routed_experts=None,
        indexer_topk=None,
        placeholder_tokens_idx=None,
        placeholder_tokens_val=None,
        time_stats=[time_stats],
    )


class TestReqTimeStatsTransport(CustomTestCase):
    def test_tokenized_request_time_stats_round_trip(self):
        for use_pickle in (False, True):
            requests = [
                (
                    TokenizedEmbeddingReqInput(
                        input_text="hello",
                        input_ids=array("i", [1, 2, 3]),
                        mm_inputs=None,
                        token_type_ids=None,
                        sampling_params=SamplingParams(),
                        time_stats=APIServerReqTimeStats(
                            disagg_mode=DisaggregationMode.PREFILL
                        ).to_ipc(),
                    ),
                    BatchTokenizedEmbeddingReqInput,
                    APIServerReqTimeStats,
                ),
                (
                    TokenizedGenerateReqInput(
                        input_text="hello",
                        input_ids=array("i", [1, 2, 3]),
                        input_embeds=None,
                        mm_inputs=None,
                        token_type_ids=None,
                        sampling_params=SamplingParams(),
                        return_logprob=False,
                        logprob_start_len=0,
                        top_logprobs_num=0,
                        token_ids_logprob=None,
                        stream=False,
                        time_stats=DPControllerReqTimeStats(
                            disagg_mode=DisaggregationMode.PREFILL
                        ).to_ipc(),
                    ),
                    BatchTokenizedGenerateReqInput,
                    DPControllerReqTimeStats,
                ),
            ]

            for req, batch_type, expected_type in requests:
                for batched in (False, True):
                    with self.subTest(
                        use_pickle=use_pickle,
                        request_type=type(req).__name__,
                        batched=batched,
                    ):
                        with patch.object(io_struct, "_USE_PICKLE_IPC", use_pickle):
                            req.wrap_pickle_fields()
                            payload = batch_type(batch=[req]) if batched else req
                            decoded = _transport_round_trip(payload)
                            decoded_req = decoded.batch[0] if batched else decoded
                            decoded_req.unwrap_pickle_fields()

                        self.assertIsInstance(decoded_req.time_stats, expected_type)
                        self.assertEqual(
                            decoded_req.time_stats.disagg_mode,
                            DisaggregationMode.PREFILL,
                        )
                        self.assertFalse(
                            decoded_req.time_stats.trace_ctx.tracing_enable
                        )

    def test_tracing_time_stats_with_128_bit_id_round_trip(self):
        trace_id = (1 << 127) + 1
        span_id = (1 << 63) + 1
        trace_state = {
            "tracing_enable": True,
            "last_span_context": {
                "trace_id": f"{trace_id:032x}",
                "span_id": f"{span_id:016x}",
            },
        }

        class FakeTraceContext:
            tracing_enable = True

            def __getstate__(self):
                return trace_state

        time_stats = APIServerReqTimeStats()
        time_stats.trace_ctx = FakeTraceContext()
        snapshot = time_stats.to_ipc()

        request = TokenizedEmbeddingReqInput(
            input_text="hello",
            input_ids=array("i", [1, 2, 3]),
            mm_inputs=None,
            token_type_ids=None,
            sampling_params=SamplingParams(),
            time_stats=snapshot,
        )

        for use_pickle in (False, True):
            with self.subTest(use_pickle=use_pickle):
                with patch.object(io_struct, "_USE_PICKLE_IPC", use_pickle):
                    decoded = _transport_round_trip(request)

                encoded_span_context = decoded.time_stats.trace_ctx_state[
                    "last_span_context"
                ]
                self.assertEqual(
                    encoded_span_context,
                    trace_state["last_span_context"],
                )

    def test_legacy_trace_state_relay_is_msgpack_safe(self):
        trace_id = (1 << 127) + 1
        span_id = (1 << 63) + 1
        legacy_trace_state = {
            "tracing_enable": True,
            "last_span_context": {
                "trace_id": trace_id,
                "span_id": span_id,
            },
        }

        with patch.object(trace_module, "opentelemetry_initialized", False):
            constructed = APIServerReqTimeStats(trace_ctx_state=legacy_trace_state)
            restored = APIServerReqTimeStats()
            restored.__setstate__({"trace_ctx_state": legacy_trace_state})

        for source, time_stats in (("constructor", constructed), ("state", restored)):
            with self.subTest(source=source):
                encoded = msgspec.msgpack.encode(time_stats.to_ipc())
                decoded = msgspec.msgpack.decode(encoded)
                span_context = decoded["trace_ctx_state"]["last_span_context"]
                self.assertEqual(span_context["trace_id"], f"{trace_id:032x}")
                self.assertEqual(span_context["span_id"], f"{span_id:016x}")

        self.assertEqual(legacy_trace_state["last_span_context"]["trace_id"], trace_id)
        self.assertEqual(legacy_trace_state["last_span_context"]["span_id"], span_id)

    def test_time_stats_init_converts_clock_for_all_request_types(self):
        with patch.object(
            req_time_stats_module, "global_diff_realtime_monotonic", 90.0
        ):
            api_time_stats = APIServerReqTimeStats(
                created_time=10.0,
                diff_realtime_monotonic=100.0,
            )
            dp_time_stats = DPControllerReqTimeStats(
                dpc_dispatch_time=10.0,
                diff_realtime_monotonic=100.0,
            )
            empty_time_stats = APIServerReqTimeStats(
                created_time=0.0,
                diff_realtime_monotonic=100.0,
            )

        self.assertEqual(api_time_stats.created_time, 20.0)
        self.assertEqual(dp_time_stats.dpc_dispatch_time, 20.0)
        self.assertEqual(empty_time_stats.created_time, 0.0)
        self.assertEqual(api_time_stats.diff_realtime_monotonic, 90.0)
        self.assertEqual(dp_time_stats.diff_realtime_monotonic, 90.0)
        self.assertEqual(empty_time_stats.diff_realtime_monotonic, 90.0)

    def test_time_stats_setstate_converts_clock_without_mutating_state(self):
        state = {
            "disagg_mode": DisaggregationMode.DECODE.value,
            "created_time": 10.0,
            "trace_ctx": {"tracing_enable": False},
            "diff_realtime_monotonic": 100.0,
        }
        with patch.object(
            req_time_stats_module, "global_diff_realtime_monotonic", 90.0
        ):
            time_stats = APIServerReqTimeStats()
            time_stats.__setstate__(state)

        self.assertEqual(time_stats.disagg_mode, DisaggregationMode.DECODE)
        self.assertEqual(time_stats.created_time, 20.0)
        self.assertEqual(time_stats.diff_realtime_monotonic, 90.0)
        self.assertEqual(state["disagg_mode"], DisaggregationMode.DECODE.value)
        self.assertEqual(state["created_time"], 10.0)
        self.assertIn("trace_ctx", state)
        self.assertEqual(state["diff_realtime_monotonic"], 100.0)

    def test_scheduler_time_stats_round_trip_converts_clock(self):
        with patch.object(
            req_time_stats_module, "global_diff_realtime_monotonic", 100.0
        ):
            time_stats = SchedulerReqTimeStats(
                enable_metrics=True,
                wait_queue_entry_time=10.0,
                forward_entry_time=12.0,
                prefill_finished_time=14.0,
            )
            output = BatchEmbeddingOutput(
                rids=["rid"],
                http_worker_ipcs=[None],
                finished_reasons=[None],
                embeddings=[1.0],
                prompt_tokens=[3],
                cached_tokens=[1],
                placeholder_tokens_idx=None,
                placeholder_tokens_val=None,
                retraction_counts=[0],
                time_stats=[time_stats.to_ipc()],
            )
            sockets = {}
            for use_pickle in (False, True):
                with patch.object(io_struct, "_USE_PICKLE_IPC", use_pickle):
                    socket = _LoopbackSocket()
                    sock_send(socket, output)
                    sockets[use_pickle] = socket

        for use_pickle, socket in sockets.items():
            with self.subTest(use_pickle=use_pickle):
                with patch.object(
                    req_time_stats_module, "global_diff_realtime_monotonic", 90.0
                ), patch.object(io_struct, "_USE_PICKLE_IPC", use_pickle):
                    decoded = sock_recv(socket)

                decoded_time_stats = decoded.time_stats[0]
                self.assertIsInstance(decoded_time_stats, SchedulerReqTimeStats)
                self.assertEqual(decoded_time_stats.wait_queue_entry_time, 20.0)
                self.assertEqual(decoded_time_stats.forward_entry_time, 22.0)
                self.assertEqual(decoded_time_stats.prefill_finished_time, 24.0)
                self.assertEqual(decoded_time_stats.diff_realtime_monotonic, 90.0)

    def test_batch_token_id_output_time_stats_round_trip(self):
        for use_pickle in (False, True):
            with self.subTest(use_pickle=use_pickle):
                output = _make_batch_token_id_output(
                    SchedulerReqTimeStats(
                        enable_metrics=True, wait_queue_entry_time=1.0
                    ).to_ipc()
                )
                with patch.object(io_struct, "_USE_PICKLE_IPC", use_pickle):
                    decoded = _transport_round_trip(output)

                self.assertIsInstance(decoded, BatchTokenIDOutput)
                self.assertIsInstance(decoded.time_stats[0], SchedulerReqTimeStats)
                self.assertEqual(decoded.time_stats[0].wait_queue_entry_time, 1.0)

    def test_scheduler_time_stats_omit_defaults(self):
        snapshot = SchedulerReqTimeStats(
            enable_metrics=True,
            wait_queue_entry_time=10.0,
            forward_entry_time=12.0,
            prefill_finished_time=14.0,
        ).to_ipc()

        msgpack_state = msgspec.msgpack.decode(msgspec.msgpack.encode(snapshot))
        self.assertEqual(
            set(msgpack_state),
            {
                "type",
                "wait_queue_entry_time",
                "forward_entry_time",
                "prefill_finished_time",
                "diff_realtime_monotonic",
            },
        )

        _, restore_args = snapshot.__reduce_ex__(pickle.HIGHEST_PROTOCOL)
        self.assertEqual(
            set(restore_args[1]),
            {
                "wait_queue_entry_time",
                "forward_entry_time",
                "prefill_finished_time",
                "diff_realtime_monotonic",
            },
        )

    def test_scheduler_time_stats_disabled_metrics_snapshot_is_empty(self):
        time_stats = SchedulerReqTimeStats(
            wait_queue_entry_time=10.0,
            forward_entry_time=12.0,
            prefill_finished_time=14.0,
        )

        snapshot = time_stats.to_ipc()

        self.assertFalse(snapshot.enable_metrics)
        self.assertEqual(snapshot.wait_queue_entry_time, 0.0)
        self.assertEqual(snapshot.forward_entry_time, 0.0)
        self.assertEqual(snapshot.prefill_finished_time, 0.0)
        self.assertEqual(
            msgspec.msgpack.decode(msgspec.msgpack.encode(snapshot)),
            {"type": "SchedulerReqTimeStats"},
        )
        pickle_snapshot = pickle.loads(pickle.dumps(snapshot))
        self.assertFalse(pickle_snapshot.enable_metrics)
        self.assertEqual(pickle_snapshot.wait_queue_entry_time, 0.0)
        self.assertEqual(pickle_snapshot.forward_entry_time, 0.0)
        self.assertEqual(pickle_snapshot.prefill_finished_time, 0.0)
        self.assertEqual(pickle_snapshot.diff_realtime_monotonic, 0.0)

    def test_new_from_obj_copies_shared_and_runtime_state(self):
        metrics_collector = object()
        with patch.object(
            req_time_stats_module, "global_diff_realtime_monotonic", 100.0
        ):
            api_time_stats = APIServerReqTimeStats(
                enable_metrics=True,
                disagg_mode=DisaggregationMode.DECODE,
                diff_realtime_monotonic=100.0,
                created_time=11.0,
                api_server_dispatch_time=12.0,
            )
        api_time_stats.metrics_collector = metrics_collector

        with patch.object(
            req_time_stats_module, "global_diff_realtime_monotonic", 90.0
        ), patch.object(req_time_stats_module, "calibrate_time_diff"):
            dp_time_stats = DPControllerReqTimeStats.new_from_obj(api_time_stats)
            dp_time_stats.dpc_dispatch_time = 13.0
            dp_time_stats_copy = DPControllerReqTimeStats.new_from_obj(dp_time_stats)
            scheduler_time_stats = SchedulerReqTimeStats.new_from_obj(dp_time_stats)

        self.assertTrue(scheduler_time_stats.enable_metrics)
        self.assertEqual(scheduler_time_stats.disagg_mode, DisaggregationMode.DECODE)
        self.assertEqual(scheduler_time_stats.diff_realtime_monotonic, 90.0)
        self.assertIs(scheduler_time_stats.metrics_collector, metrics_collector)
        self.assertIs(scheduler_time_stats.trace_ctx, api_time_stats.trace_ctx)
        self.assertEqual(dp_time_stats.created_time, 21.0)
        self.assertEqual(dp_time_stats.api_server_dispatch_time, 22.0)
        self.assertEqual(dp_time_stats_copy.dpc_dispatch_time, 13.0)
        self.assertEqual(scheduler_time_stats.created_time, 21.0)
        self.assertEqual(scheduler_time_stats.api_server_dispatch_time, 22.0)
        self.assertEqual(scheduler_time_stats.dpc_dispatch_time, 13.0)


class TestGenerateReqInputNormalization(CustomTestCase):
    """Test the normalization of GenerateReqInput for batch processing and different input formats."""

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST

    def setUp(self):
        # Common setup for all tests
        self.base_req = GenerateReqInput(
            text=["Hello", "World"],
            sampling_params=[{}, {}],
            rid=["id1", "id2"],
        )

    def test_single_image_to_list_of_lists(self):
        """Test that a single image is converted to a list of single-image lists."""
        req = copy.deepcopy(self.base_req)
        req.image_data = "single_image.jpg"  # A single image (non-list)

        req.normalize_batch_and_arguments()

        # Should be converted to [[image], [image]]
        self.assertEqual(len(req.image_data), 2)
        self.assertEqual(len(req.image_data[0]), 1)
        self.assertEqual(len(req.image_data[1]), 1)
        self.assertEqual(req.image_data[0][0], "single_image.jpg")
        self.assertEqual(req.image_data[1][0], "single_image.jpg")

        # Check modalities
        self.assertEqual(req.modalities, ["image", "image"])

    def test_list_of_images_to_list_of_lists(self):
        """Test that a list of images is converted to a list of single-image lists."""
        req = copy.deepcopy(self.base_req)
        req.image_data = ["image1.jpg", "image2.jpg"]  # List of images

        req.normalize_batch_and_arguments()

        # Should be converted to [[image1], [image2]]
        self.assertEqual(len(req.image_data), 2)
        self.assertEqual(len(req.image_data[0]), 1)
        self.assertEqual(len(req.image_data[1]), 1)
        self.assertEqual(req.image_data[0][0], "image1.jpg")
        self.assertEqual(req.image_data[1][0], "image2.jpg")

        # Check modalities
        self.assertEqual(req.modalities, ["image", "image"])

    def test_list_of_lists_with_different_modalities(self):
        """Test handling of list of lists of images with different modalities."""
        req = copy.deepcopy(self.base_req)
        req.image_data = [
            ["image1.jpg"],  # Single image (image modality)
            ["image2.jpg", "image3.jpg"],  # Multiple images (multi-images modality)
        ]

        req.normalize_batch_and_arguments()

        # Structure should remain the same
        self.assertEqual(len(req.image_data), 2)
        self.assertEqual(len(req.image_data[0]), 1)
        self.assertEqual(len(req.image_data[1]), 2)

        # Check modalities
        self.assertEqual(req.modalities, ["image", "multi-images"])

    def test_list_of_lists_with_none_values(self):
        """Test handling of list of lists with None values."""
        req = copy.deepcopy(self.base_req)
        req.image_data = [
            [None],  # None value
            ["image.jpg"],  # Single image
        ]

        req.normalize_batch_and_arguments()

        # Structure should remain the same
        self.assertEqual(len(req.image_data), 2)
        self.assertEqual(len(req.image_data[0]), 1)
        self.assertEqual(len(req.image_data[1]), 1)

        # Check modalities
        self.assertEqual(req.modalities, [None, "image"])

    def test_expanding_parallel_sample_correlation(self):
        """Test that when expanding with parallel samples, prompts, images and modalities are properly correlated."""
        req = copy.deepcopy(self.base_req)
        req.text = ["Prompt 1", "Prompt 2"]
        req.image_data = [
            ["image1.jpg"],
            ["image2.jpg", "image3.jpg"],
        ]
        req.sampling_params = {"n": 3}  # All prompts get 3 samples

        # Define expected values before normalization
        expected_text = req.text * 3
        expected_images = req.image_data * 3
        expected_modalities = ["image", "multi-images"] * 3

        req.normalize_batch_and_arguments()

        # Should be expanded to 6 items (2 original * 3 parallel)
        self.assertEqual(len(req.image_data), 6)

        # Check that images are properly expanded
        self.assertEqual(req.image_data, expected_images)

        # Check modalities
        self.assertEqual(req.modalities, expected_modalities)

        # Ensure that text items are properly duplicated too
        self.assertEqual(req.text, expected_text)

    def test_specific_parallel_n_per_sample(self):
        """Test parallel expansion when different samples have different n values."""
        req = copy.deepcopy(self.base_req)
        req.text = ["Prompt 1", "Prompt 2"]
        req.image_data = [
            ["image1.jpg"],
            ["image2.jpg", "image3.jpg"],
        ]
        req.sampling_params = [
            {"n": 2},
            {"n": 2},
        ]  # First prompt gets 2 samples, second prompt gets 2 samples

        expected_images = req.image_data * 2
        expected_modalities = ["image", "multi-images"] * 2
        expected_text = req.text * 2

        req.normalize_batch_and_arguments()

        # Should be expanded to 4 items (2 original * 2 parallel)
        self.assertEqual(len(req.image_data), 4)

        # Check that the first 2 are copies for the first prompt
        self.assertEqual(req.image_data, expected_images)

        # Check modalities
        self.assertEqual(req.modalities, expected_modalities)

        # Check text expansion
        self.assertEqual(req.text, expected_text)

    def test_mixed_none_and_images_with_parallel_samples(self):
        """Test that when some batch items have images and others None, parallel expansion works correctly."""
        req = copy.deepcopy(self.base_req)
        req.text = ["Prompt 1", "Prompt 2", "Prompt 3"]
        req.rid = ["id1", "id2", "id3"]
        req.image_data = [
            ["image1.jpg"],
            None,
            ["image3_1.jpg", "image3_2.jpg"],
        ]
        req.sampling_params = {"n": 2}  # All prompts get 2 samples

        expected_images = req.image_data * 2
        expected_modalities = ["image", None, "multi-images"] * 2
        expected_text = req.text * 2

        req.normalize_batch_and_arguments()

        # Should be expanded to 6 items (3 original * 2 parallel)
        self.assertEqual(len(req.image_data), 6)

        # Check image data
        self.assertEqual(req.image_data, expected_images)

        # Check modalities
        self.assertEqual(req.modalities, expected_modalities)

        # Check text expansion
        self.assertEqual(req.text, expected_text)

    def test_correlation_with_sampling_params(self):
        """Test that sampling parameters are correctly correlated with prompts during expansion."""
        req = copy.deepcopy(self.base_req)
        req.text = ["Prompt 1", "Prompt 2"]
        req.image_data = [
            ["image1.jpg"],
            ["image2.jpg"],
        ]
        req.sampling_params = [
            {"temperature": 0.7, "n": 2},
            {"temperature": 0.9, "n": 2},
        ]

        req.normalize_batch_and_arguments()

        # Check sampling params expansion
        self.assertEqual(len(req.sampling_params), 4)
        self.assertEqual(req.sampling_params[0]["temperature"], 0.7)
        self.assertEqual(req.sampling_params[1]["temperature"], 0.9)
        self.assertEqual(req.sampling_params[2]["temperature"], 0.7)
        self.assertEqual(req.sampling_params[3]["temperature"], 0.9)

        # Should be expanded to 4 items (2 original * 2 parallel)
        self.assertEqual(len(req.image_data), 4)

        # Check correlation with images
        self.assertEqual(req.image_data[0], ["image1.jpg"])
        self.assertEqual(req.image_data[1], ["image2.jpg"])
        self.assertEqual(req.image_data[2], ["image1.jpg"])
        self.assertEqual(req.image_data[3], ["image2.jpg"])

    def test_single_example_with_image(self):
        """Test handling of single example with image."""
        req = GenerateReqInput(
            text="Hello",
            image_data="single_image.jpg",
        )

        req.normalize_batch_and_arguments()

        # For single examples, image_data doesn't get processed into lists
        self.assertEqual(req.image_data, "single_image.jpg")
        self.assertIsNone(req.modalities)  # Modalities isn't set for single examples

    def test_single_to_batch_with_parallel_sampling(self):
        """Test single example converted to batch with parallel sampling."""
        req = GenerateReqInput(
            text="Hello",
            image_data="single_image.jpg",
            sampling_params={"n": 3},  # parallel_sample_num = 3
        )

        # Define expected values before normalization
        expected_text = ["Hello"] * 3

        req.normalize_batch_and_arguments()

        # Should be converted to batch with text=["Hello"]
        self.assertEqual(req.text, expected_text)

        # Image should be automatically wrapped to list of lists with length 1*3=3
        self.assertEqual(len(req.image_data), 3)
        self.assertEqual(req.image_data[0][0], "single_image.jpg")
        self.assertEqual(req.image_data[1][0], "single_image.jpg")
        self.assertEqual(req.image_data[2][0], "single_image.jpg")

        # Modalities should be set for all 3 examples
        self.assertEqual(req.modalities, ["image", "image", "image"])

    def test_audio_data_handling(self):
        """Test handling of audio_data."""
        req = copy.deepcopy(self.base_req)
        req.audio_data = "audio.mp3"  # Single audio

        req.normalize_batch_and_arguments()

        # Should be converted to ["audio.mp3", "audio.mp3"]
        self.assertEqual(len(req.audio_data), 2)
        self.assertEqual(req.audio_data[0], "audio.mp3")
        self.assertEqual(req.audio_data[1], "audio.mp3")

        # Test with list
        req = copy.deepcopy(self.base_req)
        req.audio_data = ["audio1.mp3", "audio2.mp3"]

        req.normalize_batch_and_arguments()

        # Should remain the same
        self.assertEqual(len(req.audio_data), 2)
        self.assertEqual(req.audio_data[0], "audio1.mp3")
        self.assertEqual(req.audio_data[1], "audio2.mp3")

    def test_input_ids_normalization(self):
        """Test normalization of input_ids instead of text."""
        # Test single input_ids
        req = GenerateReqInput(input_ids=[1, 2, 3])
        req.normalize_batch_and_arguments()
        self.assertTrue(req.is_single)
        self.assertEqual(req.batch_size, 1)

        # Test batch input_ids
        req = GenerateReqInput(input_ids=[[1, 2, 3], [4, 5, 6]])
        req.normalize_batch_and_arguments()
        self.assertFalse(req.is_single)
        self.assertEqual(req.batch_size, 2)

        # Test with parallel sampling
        req = GenerateReqInput(
            input_ids=[[1, 2, 3], [4, 5, 6]], sampling_params={"n": 2}
        )
        req.normalize_batch_and_arguments()
        self.assertEqual(len(req.input_ids), 4)  # 2 original * 2 parallel

    def test_input_embeds_normalization(self):
        """Test normalization of input_embeds."""
        # Test single input_embeds
        req = GenerateReqInput(input_embeds=[[0.1, 0.2], [0.3, 0.4]])
        req.normalize_batch_and_arguments()
        self.assertTrue(req.is_single)
        self.assertEqual(req.batch_size, 1)

        # Test batch input_embeds
        req = GenerateReqInput(input_embeds=[[[0.1, 0.2]], [[0.3, 0.4]]])
        req.normalize_batch_and_arguments()
        self.assertFalse(req.is_single)
        self.assertEqual(req.batch_size, 2)

    def test_input_embeds_with_parallel_sampling(self):
        """Test input_embeds normalization with parallel sampling (n > 1)."""
        # Test single input_embeds with parallel sampling
        req = GenerateReqInput(
            input_embeds=[[0.1, 0.2]],  # single embedding vector
            sampling_params={"n": 2},
        )
        req.normalize_batch_and_arguments()

        # Should be converted from single to batch and then expanded
        self.assertFalse(req.is_single)
        self.assertEqual(len(req.input_embeds), 2)
        # Both should be the same input_embeds
        self.assertEqual(req.input_embeds[0], [[0.1, 0.2]])
        self.assertEqual(req.input_embeds[1], [[0.1, 0.2]])

        # Test batch input_embeds with parallel sampling
        req = GenerateReqInput(
            input_embeds=[[[0.1, 0.2]], [[0.3, 0.4]]], sampling_params={"n": 3}
        )
        req.normalize_batch_and_arguments()

        # Should be expanded
        self.assertFalse(req.is_single)
        self.assertEqual(len(req.input_embeds), 6)

        # Check that the expansion is correct
        expected_embeds = [[[0.1, 0.2]], [[0.3, 0.4]]] * 3
        self.assertEqual(req.input_embeds, expected_embeds)

        # Test with different n values per sample (should raise error)
        req = GenerateReqInput(
            input_embeds=[[[0.1, 0.2]], [[0.3, 0.4]]],
            sampling_params=[{"n": 2}, {"n": 3}],
        )
        with self.assertRaises(ValueError):
            req.normalize_batch_and_arguments()

    def test_lora_path_normalization(self):
        """Test normalization of lora_path."""
        # Test single lora_path with batch input
        req = GenerateReqInput(text=["Hello", "World"], lora_path="path/to/lora")

        # Define expected lora_paths before normalization
        expected_lora_paths = ["path/to/lora", "path/to/lora"]

        req.normalize_batch_and_arguments()
        self.assertEqual(req.lora_path, expected_lora_paths)

        # Test list of lora_paths
        req = GenerateReqInput(text=["Hello", "World"], lora_path=["path1", "path2"])

        # Define expected lora_paths before normalization
        expected_lora_paths = ["path1", "path2"]

        req.normalize_batch_and_arguments()
        self.assertEqual(req.lora_path, expected_lora_paths)

        # Test with parallel sampling
        req = GenerateReqInput(
            text=["Hello", "World"],
            lora_path=["path1", "path2"],
            sampling_params={"n": 2},
        )

        # Define expected lora_paths before normalization
        expected_lora_paths = ["path1", "path2"] * 2

        req.normalize_batch_and_arguments()
        self.assertEqual(req.lora_path, expected_lora_paths)

    def test_extra_key_normalization(self):
        """Test normalization of extra_key."""
        # Per-request list
        req = GenerateReqInput(
            text=["Hello", "World"],
            extra_key=["tenant-A", "tenant-B"],
            sampling_params=[{}, {}],
        )
        req.normalize_batch_and_arguments()
        self.assertEqual(req.extra_key, ["tenant-A", "tenant-B"])
        self.assertEqual(req[0].extra_key, "tenant-A")
        self.assertEqual(req[1].extra_key, "tenant-B")

        # Scalar broadcast
        req = GenerateReqInput(
            text=["Hello", "World"],
            extra_key="shared",
            sampling_params=[{}, {}],
        )
        req.normalize_batch_and_arguments()
        self.assertEqual(req.extra_key, ["shared", "shared"])

        # None stays None
        req = GenerateReqInput(text=["Hello", "World"], sampling_params=[{}, {}])
        req.normalize_batch_and_arguments()
        self.assertIsNone(req.extra_key)
        self.assertIsNone(req[0].extra_key)

        # Parallel sampling expansion
        req = GenerateReqInput(
            text=["Hello", "World"],
            extra_key=["tenant-A", "tenant-B"],
            sampling_params={"n": 2},
        )
        req.normalize_batch_and_arguments()
        self.assertEqual(req.extra_key, ["tenant-A", "tenant-B"] * 2)

        # Wrong-length list
        req = GenerateReqInput(
            text=["Hello", "World"],
            extra_key=["only-one"],
            sampling_params=[{}, {}],
        )
        with self.assertRaisesRegex(ValueError, "batch size"):
            req.normalize_batch_and_arguments()

        # Non-batched scalar unchanged
        req = GenerateReqInput(text="Hello", extra_key="solo")
        req.normalize_batch_and_arguments()
        self.assertEqual(req.extra_key, "solo")

    def test_logprob_parameters_normalization(self):
        """Test normalization of logprob-related parameters."""
        # Test single example
        req = GenerateReqInput(
            text="Hello",
            return_logprob=True,
            logprob_start_len=10,
            top_logprobs_num=5,
            token_ids_logprob=[7, 8, 9],
        )
        req.normalize_batch_and_arguments()
        self.assertEqual(req.return_logprob, True)
        self.assertEqual(req.logprob_start_len, 10)
        self.assertEqual(req.top_logprobs_num, 5)
        self.assertEqual(req.token_ids_logprob, [7, 8, 9])

        # Test batch with scalar values
        req = GenerateReqInput(
            text=["Hello", "World"],
            return_logprob=True,
            logprob_start_len=10,
            top_logprobs_num=5,
            token_ids_logprob=[7, 8, 9],
        )
        req.normalize_batch_and_arguments()
        self.assertEqual(req.return_logprob, [True, True])
        self.assertEqual(req.logprob_start_len, [10, 10])
        self.assertEqual(req.top_logprobs_num, [5, 5])
        self.assertEqual(req.token_ids_logprob, [[7, 8, 9], [7, 8, 9]])

        # Test batch with list values
        req = GenerateReqInput(
            text=["Hello", "World"],
            return_logprob=[True, False],
            logprob_start_len=[10, 5],
            top_logprobs_num=[5, 3],
            token_ids_logprob=[[7, 8, 9], [4, 5, 6]],
            return_hidden_states=[False, False, True],
        )
        req.normalize_batch_and_arguments()
        self.assertEqual(req.return_logprob, [True, False])
        self.assertEqual(req.logprob_start_len, [10, 5])
        self.assertEqual(req.top_logprobs_num, [5, 3])
        self.assertEqual(req.token_ids_logprob, [[7, 8, 9], [4, 5, 6]])
        self.assertEqual(req.return_hidden_states, [False, False, True])

    def test_custom_logit_processor_normalization(self):
        """Test normalization of custom_logit_processor."""
        # Test single processor
        req = GenerateReqInput(
            text=["Hello", "World"], custom_logit_processor="serialized_processor"
        )
        req.normalize_batch_and_arguments()
        self.assertEqual(
            req.custom_logit_processor, ["serialized_processor", "serialized_processor"]
        )

        # Test list of processors
        req = GenerateReqInput(
            text=["Hello", "World"], custom_logit_processor=["processor1", "processor2"]
        )
        req.normalize_batch_and_arguments()
        self.assertEqual(req.custom_logit_processor, ["processor1", "processor2"])

    def test_session_params_handling(self):
        """Test handling of session_params."""
        # Test with dict
        req = GenerateReqInput(
            text=["Hello", "World"], session_params={"id": "session1", "offset": 10}
        )
        req.normalize_batch_and_arguments()
        self.assertEqual(req.session_params, {"id": "session1", "offset": 10})

        # Test with list of dicts
        req = GenerateReqInput(
            text=["Hello", "World"],
            session_params=[{"id": "session1"}, {"id": "session2"}],
        )
        req.normalize_batch_and_arguments()
        self.assertEqual(req.session_params, [{"id": "session1"}, {"id": "session2"}])

    def test_session_id_handling(self):
        req = GenerateReqInput(
            text=["Hello", "World"],
            session_id="session1",
            sampling_params={"n": 2},
        )
        req.normalize_batch_and_arguments()
        self.assertEqual(req.session_id, "session1")
        self.assertIsNone(req.session_params)
        self.assertEqual(req[2].session_id, "session1")

        with self.assertRaisesRegex(ValueError, "cannot both be set"):
            GenerateReqInput(
                text="Hello",
                session_id="explicit",
                session_params={"id": "legacy"},
            ).normalize_batch_and_arguments()

    def test_getitem_method(self):
        """Test the __getitem__ method."""
        req = GenerateReqInput(
            text=["Hello", "World"],
            image_data=[["img1.jpg"], ["img2.jpg"]],
            audio_data=["audio1.mp3", "audio2.mp3"],
            sampling_params=[{"temp": 0.7}, {"temp": 0.8}],
            rid=["id1", "id2"],
            return_logprob=[True, False],
            logprob_start_len=[10, 5],
            top_logprobs_num=[5, 3],
            token_ids_logprob=[[7, 8, 9], [4, 5, 6]],
            stream=True,
            log_metrics=True,
            modalities=["image", "image"],
            lora_path=["path1", "path2"],
            custom_logit_processor=["processor1", "processor2"],
            return_hidden_states=True,
        )
        req.normalize_batch_and_arguments()

        # Get the first item
        item0 = req[0]
        self.assertEqual(item0.text, "Hello")
        self.assertEqual(item0.image_data, ["img1.jpg"])
        self.assertEqual(item0.audio_data, "audio1.mp3")
        self.assertEqual(item0.sampling_params, {"temp": 0.7})
        self.assertEqual(item0.rid, "id1")
        self.assertEqual(item0.return_logprob, True)
        self.assertEqual(item0.logprob_start_len, 10)
        self.assertEqual(item0.top_logprobs_num, 5)
        self.assertEqual(item0.token_ids_logprob, [7, 8, 9])
        self.assertEqual(item0.stream, True)
        self.assertEqual(item0.log_metrics, True)
        self.assertEqual(item0.modalities, "image")
        self.assertEqual(item0.lora_path, "path1")
        self.assertEqual(item0.custom_logit_processor, "processor1")
        self.assertEqual(item0.return_hidden_states, True)

    def test_getitem_preserves_return_prompt_token_ids(self):
        """Batch subrequests must keep the prompt-token-id return flag."""
        req = GenerateReqInput(
            input_ids=[[1, 2, 3], [4, 5, 6]],
            sampling_params=[{}, {}],
            rid=["id1", "id2"],
            return_prompt_token_ids=True,
        )
        req.normalize_batch_and_arguments()

        self.assertTrue(req[0].return_prompt_token_ids)
        self.assertTrue(req[1].return_prompt_token_ids)

    def test_regenerate_rid(self):
        """Test the regenerate_rid method."""
        req = GenerateReqInput(text="Hello")
        req.normalize_batch_and_arguments()

        original_rid = req.rid
        new_rid = req.regenerate_rid()

        self.assertNotEqual(original_rid, new_rid)
        self.assertEqual(req.rid, new_rid)

    def test_error_cases(self):
        """Test various error cases."""
        # Test when neither text, input_ids, nor input_embeds is provided
        with self.assertRaises(ValueError):
            req = GenerateReqInput()
            req.normalize_batch_and_arguments()

        # Test when all of text, input_ids, and input_embeds are provided
        with self.assertRaises(ValueError):
            req = GenerateReqInput(
                text="Hello", input_ids=[1, 2, 3], input_embeds=[[0.1, 0.2]]
            )
            req.normalize_batch_and_arguments()


if __name__ == "__main__":
    unittest.main()
