"""Model packages use one extension for server arguments and worker startup."""

import json
import os
import pickle
import threading
import unittest
from array import array
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch, sentinel

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import flatten_output_ids  # noqa: E402
from sglang.srt.managers.schedule_batch import Req  # noqa: E402
from sglang.srt.rust_server import server as server_module  # noqa: E402
from sglang.srt.rust_server.server import RustServer  # noqa: E402

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

# The Rust frontend decodes these headers and renders their `meta_info` keys
# (rust/sglang-server/src/api_server/core/frame.rs); they must match Python.
RUST_FRAME_METADATA_FIXTURE = os.path.join(
    os.path.dirname(__file__),
    *[os.pardir] * 6,
    "rust/sglang-server/testdata/frame_metadata.json",
)
# Its native `/generate` frames report the scheduler's customized_info
# (rust/sglang-server/src/api_server/http/native_api.rs).
RUST_CUSTOMIZED_INFO_FIXTURE = os.path.join(
    os.path.dirname(RUST_FRAME_METADATA_FIXTURE), "customized_info.json"
)


# A payload's optional columns, unset.
NO_EXTRAS = dict.fromkeys(
    [
        f"{family}_{part}"
        for family in (
            "output_token_logprobs",
            "input_token_logprobs",
            "output_top_logprobs",
            "input_top_logprobs",
            "output_token_ids_logprobs",
            "input_token_ids_logprobs",
        )
        for part in ("val", "idx")
    ]
    + ["output_hidden_states", "customized_info"]
)


class _StreamedReq:
    """A request as `SchedulerOutputStreamer` reads it, in either egress mode:
    retracted, waiting on the cache and verifying drafts as the arguments say."""

    def __init__(
        self, rid, output_ids, *, finished, retractions, time_stats, spec_counters
    ):
        self.rid = rid
        self.http_worker_ipc = None
        self.beam_group = None
        self._finished = finished
        self.finished_reason = (
            SimpleNamespace(to_json=lambda: {"type": "length", "length": 3})
            if finished
            else None
        )
        self.finished_output = False
        self.finished_len = None
        # Streamed, so the scheduler sends no decode latency.
        self.stream = True
        self.sampling_params = SimpleNamespace(
            stream_interval=None,
            skip_special_tokens=True,
            spaces_between_special_tokens=True,
            no_stop_trim=False,
        )
        self.output_ids = self.output_ids_through_stop = array("q", output_ids)
        self.send_token_offset = 0
        self.send_output_token_logprobs_offset = 0
        self.send_decode_id_offset = 0
        self.decoded_text = ""
        self.origin_input_ids = [1, 2]
        self.reasoning_tokens = 0
        self.cached_tokens = 0
        self.cached_tokens_device = self.cached_tokens_host = 0
        self.cached_tokens_storage = 0
        self.mm_image_tokens = self.mm_audio_tokens = self.mm_video_tokens = 0
        self.multimodal_inputs = None
        self.retraction_count = retractions
        self.request_stats_retraction_count = None
        self.detokenize_header_sent = False
        self.origin_input_ids_unpadded = array("q")
        self.prefill_weight_versions = None
        self.weight_version_events = []
        self.time_stats = time_stats
        (
            self.spec_verify_ct,
            self.spec_num_correct_drafts,
            self.spec_num_cap_tokens,
            self.spec_num_block_accept_tokens,
            self.spec_correct_drafts_histogram,
            self.spec_cap_lens_histogram,
        ) = spec_counters
        self.customized_info = None
        self.return_hidden_states = False
        self.return_routed_experts = False
        self.return_indexer_topk = False
        self.return_sampling_mask = False

    def finished(self):
        return self._finished

    take_request_stats_due = Req.take_request_stats_due
    take_detokenize_header_due = Req.take_detokenize_header_due

    def get_mm_token_counts(self):
        return self.mm_image_tokens, self.mm_audio_tokens, self.mm_video_tokens

    def init_incremental_detokenize(self):
        return self.output_ids_through_stop, 0

    def check_match_stop_str_prefix(self):
        return False


def build_rust_frame_metadata_fixture() -> dict:
    """The headers `push_generation` and `push_abort` send for the weight-version,
    media-count, decode-latency, cached-token-split and request-stats columns,
    and the `meta_info` keys Python's TokenizerManager derives from the same
    values, with "launch" as the server's weight version, plus the
    `sglang:cached_tokens_total` increments by `cache_source` its metrics
    collector records for each request of the cache-tier batches.
    Regenerate the fixture with ``json.dump(build_rust_frame_metadata_fixture(),
    f, indent=2)`` after changing either side."""
    from sglang.srt.disaggregation.utils import DisaggregationMode
    from sglang.srt.managers.io_struct import AbortReq
    from sglang.srt.managers.scheduler_components.output_streamer import (
        SchedulerOutputStreamer,
    )
    from sglang.srt.managers.tokenizer_manager import TokenizerManager
    from sglang.srt.observability.metrics_collector import (
        TokenizerMetricsCollector,
    )
    from sglang.srt.observability.req_time_stats import SchedulerReqTimeStats
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
    from sglang.srt.utils.weight_versions import (
        WeightVersionSpan,
        add_weight_versions_to_meta_info,
    )

    def meta_info(
        spans, completion_tokens, image=0, audio=0, video=0, *, cached=None, abort=False
    ):
        # TokenizerManager._handle_batch_output / _handle_abort_req; only the
        # former reports the cached-token details.
        meta = {"weight_version": "launch"}
        if spans is not None:
            add_weight_versions_to_meta_info(meta, spans, completion_tokens)
        if not abort:
            meta["cached_tokens_details"] = cached
        for key, count in (
            ("image_tokens", image),
            ("audio_tokens", audio),
            ("video_tokens", video),
        ):
            if count:
                meta[key] = count
        return meta

    def push_generation(
        multimodal,
        decode_latencies,
        cached_tokens_split=None,
        storage_backend=None,
        request_stats=None,
        **columns,
    ):
        server = RustServer(Mock(), http_port=0)
        server._multimodal_enabled = multimodal
        payload = SimpleNamespace(**{**NO_EXTRAS, **columns})
        server.push_generation(
            payload,
            decode_latencies,
            cached_tokens_split,
            storage_backend,
            request_stats,
        )
        header, data = server.server.push_decode_result_batch.call_args.args
        return {
            "header": header.hex(),
            "data": b"".join(data).hex(),
            "decode_latencies": {
                columns["rids"][row]: latency
                for row, latency in decode_latencies.items()
            },
        }

    swapped = [
        WeightVersionSpan(version="old", start=0, end=2),
        WeightVersionSpan(version="new", start=2, end=3),
    ]
    batch = push_generation(
        True,
        {0: 0.25},
        rids=["done", "running"],
        finished_reasons=[{"type": "length", "length": 3}, None],
        prompt_tokens=[5, 6],
        output_ids_flat=array("q", [1, 2, 3, 4]),
        output_ids_offsets=[0, 3, 4],
        cached_tokens=[0, 0],
        reasoning_tokens=[0, 0],
        weight_versions=[swapped, None],
        image_tokens=[8, 0],
        audio_tokens=[0, 2],
        video_tokens=[0, 0],
    )
    batch["meta_info"] = {
        "done": meta_info(swapped, 3, image=8),
        "running": meta_info(None, 1, audio=2),
    }
    # A text model's frame with output logprobs: the spans and the decode
    # latency follow the extras columns rather than their padding.
    single = [WeightVersionSpan(version="v1", start=0, end=2)]
    logprob_batch = push_generation(
        False,
        {0: 0.5},
        rids=["logprobs", "plain"],
        finished_reasons=[{"type": "length", "length": 2}, None],
        prompt_tokens=[3, 4],
        output_ids_flat=array("q", [7, 8, 9]),
        output_ids_offsets=[0, 2, 3],
        cached_tokens=[0, 0],
        reasoning_tokens=[0, 0],
        weight_versions=[single, None],
        image_tokens=[0, 0],
        audio_tokens=[0, 0],
        video_tokens=[0, 0],
        output_token_logprobs_val=[[-0.5, -0.25], []],
        output_token_logprobs_idx=[[7, 8], []],
    )
    logprob_batch["meta_info"] = {
        "logprobs": meta_info(single, 2),
        "plain": meta_info(None, 1),
    }

    class HiCacheFile:
        """Stands in for a storage backend, which the details name by type."""

    def cache_batch(storage_attached, rows):
        """A frame from a cache with a host tier and, when `storage_attached`,
        a storage tier, whose `rows` map rid -> (cached_tokens, device, host,
        storage)."""
        with patch(
            "sglang.srt.managers.scheduler_components.output_streamer.get_observability",
            return_value=SimpleNamespace(
                enable_metrics=False, enable_request_time_stats_logging=False
            ),
        ):
            streamer = SchedulerOutputStreamer(
                send_to_detokenizer=None,
                tree_cache=SimpleNamespace(
                    cache_controller=SimpleNamespace(
                        storage_backend=HiCacheFile() if storage_attached else None
                    )
                ),
                server_args=SimpleNamespace(enable_lmcache=False),
                is_generation=True,
                spec_algorithm=SpeculativeAlgorithm.NONE,
                disaggregation_mode=DisaggregationMode.NULL,
                skip_tokenizer_init=False,
                enable_hicache_storage=lambda: storage_attached,
            )
        reqs = {
            rid: SimpleNamespace(
                cached_tokens=cached,
                cached_tokens_device=device,
                cached_tokens_host=host,
                cached_tokens_storage=storage,
            )
            for rid, (cached, device, host, storage) in rows.items()
        }
        rids = list(reqs)
        frame = push_generation(
            False,
            {},
            rids=rids,
            finished_reasons=[{"type": "length", "length": 1}] * len(rids),
            prompt_tokens=[16] * len(rids),
            output_ids_flat=array("q", [1] * len(rids)),
            output_ids_offsets=list(range(len(rids) + 1)),
            cached_tokens=[req.cached_tokens for req in reqs.values()],
            reasoning_tokens=[0] * len(rids),
            weight_versions=None,
            cached_tokens_split=[
                (
                    req.cached_tokens_device,
                    req.cached_tokens_host,
                    req.cached_tokens_storage,
                )
                for req in reqs.values()
            ],
            storage_backend=(
                streamer._get_storage_backend_type() if storage_attached else None
            ),
        )
        details = {
            rid: streamer.get_cached_tokens_details(req) for rid, req in reqs.items()
        }
        frame["meta_info"] = {
            rid: meta_info(None, 1, cached=cached) for rid, cached in details.items()
        }
        frame["cached_tokens_total"] = {
            rid: cached_tokens_total(reqs[rid].cached_tokens, cached)
            for rid, cached in details.items()
        }
        return frame

    def cached_tokens_total(cached_tokens, details):
        # TokenizerMetricsCollector.observe_one_finished_request's increments.
        collector = Mock()
        counts = {}
        collector.cached_tokens_total.labels.side_effect = lambda **labels: Mock(
            inc=lambda value: counts.__setitem__(labels["cache_source"], value)
        )
        TokenizerMetricsCollector.observe_one_finished_request(
            collector, {}, 16, 1, cached_tokens, 0.0, False, details
        )
        return counts

    host_tier = cache_batch(
        False,
        {
            "miss": (0, 0, 0, 0),
            "device": (4, 4, 0, 0),
            "host": (6, 2, 4, 0),
            # A PD decode server's storage hit, reported by prefill.
            "storage": (5, 1, 1, 3),
            # A hit the scheduler did not split, and one it split short.
            "unsplit": (3, 0, 0, 0),
            "short": (5, 4, 0, 0),
        },
    )
    storage_tier = cache_batch(
        True,
        {
            "miss": (0, 0, 0, 0),
            "device": (4, 4, 0, 0),
            "storage": (8, 2, 2, 4),
            "unsplit": (2, 0, 0, 0),
        },
    )

    def stats_batch():
        """A speculative scheduler's frame with metrics on, in cap-accept
        ragged verify mode, for a finished request and a running one: the
        header the Rust egress sends, and the finished request's `meta_info`
        keys Python's TokenizerManager derives from its own copy of the batch
        output (`_handle_batch_output`: `num_retractions`, the scheduler time
        stats, `_calculate_spec_decoding_metrics`). The running request
        reports none in Rust mode."""

        def reqs():
            return [
                _StreamedReq(
                    "done",
                    [1, 2, 3],
                    finished=True,
                    retractions=2,
                    time_stats=SchedulerReqTimeStats(
                        enable_metrics=True,
                        forward_entry_time=100.25,
                        prefill_finished_time=100.75,
                        queue_duration_s=0.125,
                        storage_prefetch_wait_s=0.5,
                    ),
                    spec_counters=(3, 4, 5, 7, [0, 1, 2], [0, 3]),
                ),
                _StreamedReq(
                    "running",
                    [4],
                    finished=False,
                    retractions=0,
                    time_stats=SchedulerReqTimeStats(enable_metrics=True),
                    spec_counters=(1, 1, 0, 0, [0, 1], []),
                ),
            ]

        def streamer(rust_server, send_to_detokenizer):
            return SchedulerOutputStreamer(
                send_to_detokenizer=send_to_detokenizer,
                tree_cache=None,
                server_args=SimpleNamespace(
                    enable_hierarchical_cache=False, enable_lmcache=False
                ),
                is_generation=True,
                spec_algorithm=SpeculativeAlgorithm.EAGLE,
                disaggregation_mode=DisaggregationMode.NULL,
                skip_tokenizer_init=False,
                enable_hicache_storage=lambda: False,
                rust_server=rust_server,
            )

        streamer_module = "sglang.srt.managers.scheduler_components.output_streamer"
        spec = SimpleNamespace(
            speculative_algorithm="EAGLE", speculative_num_draft_tokens=4
        )
        with (
            patch(
                f"{streamer_module}.get_serving",
                return_value=SimpleNamespace(
                    stream_interval=1, weight_version="launch"
                ),
            ),
            patch(
                f"{streamer_module}.get_observability",
                return_value=SimpleNamespace(
                    enable_request_time_stats_logging=False, enable_metrics=True
                ),
            ),
            patch(
                f"{streamer_module}.get_parallel",
                return_value=SimpleNamespace(dp_rank=0, attn_tp_rank=0),
            ),
            patch(f"{streamer_module}.get_spec", return_value=spec),
            patch("sglang.srt.managers.tokenizer_manager.get_spec", return_value=spec),
            patch(
                "sglang.srt.speculative.spec_decoding_metrics.ragged_verify_cap_accept",
                return_value=True,
            ),
            patch(
                "sglang.srt.observability.req_time_stats.global_diff_realtime_monotonic",
                1_760_000_000.0,
            ),
        ):
            rust_server = RustServer(Mock(), http_port=0)
            streamer(rust_server, None)._stream_output_generation(reqs(), False)
            header, data = rust_server.server.push_decode_result_batch.call_args.args

            # Python mode: the batch output crosses to TokenizerManager's
            # process.
            detokenizer = Mock()
            streamer(None, detokenizer)._stream_output_generation(reqs(), False)
            recv_obj = pickle.loads(
                pickle.dumps(detokenizer.send_output.call_args.args[0])
            )
            meta = {"num_retractions": recv_obj.request_stats[0].retraction_count}
            meta.update(recv_obj.request_stats[0].time_stats.to_meta_info())
            TokenizerManager._calculate_spec_decoding_metrics(None, meta, recv_obj, 0)
        return {
            "header": header.hex(),
            "data": b"".join(data).hex(),
            "decode_latencies": {},
            "meta_info": {
                "done": meta_info(
                    [WeightVersionSpan(version="launch", start=0, end=3)], 3
                ),
                "running": meta_info(None, 1),
            },
            "request_stats": {"done": meta},
        }

    # The scheduler's spans run past the three tokens the client received.
    aborted = [
        WeightVersionSpan(version="old", start=0, end=2),
        WeightVersionSpan(version="new", start=2, end=9),
    ]
    server = RustServer(Mock(), http_port=0)
    req = SimpleNamespace(origin_input_ids=[1, 2], cached_tokens=0, reasoning_tokens=0)
    server.push_abort(AbortReq(rid="aborted", weight_versions=aborted), req)
    abort_header, _ = server.server.push_decode_result_batch.call_args.args
    return {
        "batch": batch,
        "logprob_batch": logprob_batch,
        "host_tier": host_tier,
        "storage_tier": storage_tier,
        "request_stats": stats_batch(),
        "abort": {
            "header": abort_header.hex(),
            "completion_tokens": 3,
            "meta_info": meta_info(aborted, 3, abort=True),
        },
    }


def generation_payload(rows: dict, **columns) -> SimpleNamespace:
    """A `push_generation` payload for `rows` (rid -> new output ids), every
    optional column unset unless `columns` sets it."""
    rids = list(rows)
    output_ids_flat, output_ids_offsets = flatten_output_ids(rows.values())
    return SimpleNamespace(
        **{
            **NO_EXTRAS,
            "rids": rids,
            "finished_reasons": [None] * len(rids),
            "prompt_tokens": [1] * len(rids),
            "output_ids_flat": output_ids_flat,
            "output_ids_offsets": output_ids_offsets,
            "cached_tokens": [0] * len(rids),
            "reasoning_tokens": [0] * len(rids),
            "weight_versions": None,
            **columns,
        }
    )


def build_rust_customized_info_fixture() -> dict:
    """The frames `push_generation` and `push_abort` send for three requests
    whose batches carry customized_info (`score` is request-level: one value,
    on one output), with what Python's TokenizerManager reports on each
    request's frames by streaming mode and what the Rust frontend shows a
    generate hook for them (`sglang.test.rust_customized_info`). A cumulative
    frame re-reports every key surfaced so far, a scheduler abort's included.
    Python slices an incremental frame at the request's output-token offset,
    which loses values once an output lacked a key (`late`'s prefill); the
    Rust frontend reports each frame's own values, and `python_incremental`
    records Python's where they differ.
    Regenerate the fixture with ``json.dump(build_rust_customized_info_fixture(),
    f, indent=2)`` after changing either side."""
    from sglang.srt.managers.io_struct import AbortReq
    from sglang.test.rust_customized_info import (
        MODES,
        SchedulerOutput,
        hook_inputs,
        python_incremental,
        reported_customized_info,
    )

    # As the scheduler's streamer builds them: per key, each request's values
    # for its new output tokens, a request without the key padded with None
    # (with nothing for a request-level key).
    batches = [
        # A prefill: no request has customized_info yet.
        ({"late": [10]}, None, {}),
        (
            {"streamed": [1], "aborted": [2]},
            {
                "probe": [[None], [0.5]],
                "aux": [["a"], [None]],
                "score": [[0.25], []],
            },
            {},
        ),
        # With output logprobs, after whose columns the customized_info one
        # comes.
        (
            {"streamed": [3, 4], "aborted": [5], "late": [11]},
            {
                "probe": [[1.5, None], [-1.0], [7]],
                "aux": [[None, None], ["b"], [None]],
                "score": [[], [], [9.5]],
            },
            {
                "output_token_logprobs_val": [[-0.5, -0.25], [-1.0], [-2.0]],
                "output_token_logprobs_idx": [[3, 4], [5], [11]],
            },
        ),
        ("aborted", None, {}),
        # Without customized_info: no key on `late`'s frame.
        ({"late": [12]}, None, {}),
        # Finishes both.
        (
            {"streamed": [6], "late": [13]},
            {"probe": [[2], [8]], "aux": [[None], [None]]},
            {"finished_reasons": [{"type": "length", "length": 4}] * 2},
        ),
    ]

    frames = []
    outputs = {rid: [] for rid in ("streamed", "aborted", "late")}
    for rows, customized_info, columns in batches:
        server = RustServer(Mock(), http_port=0)
        if isinstance(rows, str):
            # `_handle_abort_req` reports no customized_info.
            abort = AbortReq(rid=rows)
            req = SimpleNamespace(
                origin_input_ids=[1], cached_tokens=0, reasoning_tokens=0
            )
            server.push_abort(abort, req)
            outputs[rows].append(
                SchedulerOutput(
                    output_ids=[],
                    finish_reason={
                        "type": "abort",
                        "message": "Abort in waiting queue",
                    },
                )
            )
        else:
            payload = generation_payload(
                rows, customized_info=customized_info, **columns
            )
            server.push_generation(payload, {})
            for i, rid in enumerate(rows):
                outputs[rid].append(
                    SchedulerOutput(
                        output_ids=rows[rid],
                        customized_info=(
                            None
                            if customized_info is None
                            else {
                                key: column[i]
                                for key, column in customized_info.items()
                            }
                        ),
                        finish_reason=payload.finished_reasons[i],
                    )
                )
        header, data = server.server.push_decode_result_batch.call_args.args
        frames.append({"header": header.hex(), "data": b"".join(data).hex()})
    meta_info = {
        rid: {mode: reported_customized_info(outs, mode) for mode in MODES}
        for rid, outs in outputs.items()
    }
    python_frames = {rid: python_incremental(outs) for rid, outs in outputs.items()}
    return {
        "frames": frames,
        "meta_info": meta_info,
        "hook_inputs": {
            rid: {mode: hook_inputs(outs, mode) for mode in MODES}
            for rid, outs in outputs.items()
        },
        "python_incremental": {
            rid: python_frames[rid]
            for rid in outputs
            if python_frames[rid] != meta_info[rid]["incremental"]
        },
    }


@contextmanager
def launch_context(server_cls, cores=(None, None)):
    """Patch what `launch` reads so it runs as DP rank 1 of a two-rank node,
    with `cores` as the `(launch_cores, server_cores)` partition (unpinned by
    default); yields the `_build_server_args` mock."""
    with (
        patch.object(
            server_module,
            "get_exec",
            return_value=SimpleNamespace(moe=SimpleNamespace(is_ep_scale_joiner=False)),
        ),
        patch.object(
            server_module,
            "get_parallel",
            return_value=SimpleNamespace(
                nnodes=1,
                pp_size=1,
                dp_size=1,
                attn_dp_enabled=True,
                num_dp_ranks=2,
                attn_dp_rank=1,
                tp_size=2,
                tp_rank=1,
                attn_tp_size=1,
                attn_cp_size=1,
            ),
        ),
        patch.object(server_cls, "_partition_cores", return_value=cores),
        patch.object(
            server_module,
            "get_mm",
            return_value=SimpleNamespace(mm_processor_worker_num=8),
        ),
        patch.object(
            server_module,
            "get_serving",
            return_value=SimpleNamespace(host="::", port=30000),
        ),
        patch.object(
            server_module, "_build_server_args", return_value=sentinel.args
        ) as build_args,
    ):
        yield build_args


class TestRustServerExtension(CustomTestCase):
    def test_launch_uses_the_model_extension_and_instance_worker_state(self):
        extension = ModuleType("model_server")
        extension.Server = Mock()

        class ModelServer(RustServer):
            @classmethod
            def _load_extension(cls):
                return extension

            @classmethod
            def _server_kwargs(cls, scheduler):
                return {"model_setting": sentinel.setting}

            @classmethod
            def _resolve_multimodal(cls, scheduler):
                return None

            def _start_multimodal(self, scheduler):
                self.server.start_mm_workers(sentinel.spec, 8)

        scheduler = SimpleNamespace(
            model_config=SimpleNamespace(is_multimodal=True),
        )
        with launch_context(ModelServer) as build_args:
            instance = ModelServer.launch(scheduler)

        build_args.assert_called_once_with(
            scheduler,
            extension=extension,
            health_generate_headers={},
            default_max_new_tokens=128,
        )
        extension.Server.assert_called_once_with(
            sentinel.args,
            cores=None,
            port_offset=1,
            model_setting=sentinel.setting,
        )
        instance.server.start_mm_workers.assert_called_once_with(sentinel.spec, 8)
        self.assertIsInstance(instance, ModelServer)
        self.assertEqual(instance.http_port, 30001)
        self.assertTrue(instance._multimodal_enabled)

    def test_gather_server_endpoints_lists_every_listener_by_dp_rank(self):
        """Every rank joins the collective; only listener ranks contribute an
        entry, an unspecified bind host is replaced by a reachable one, and
        the table comes back sorted by DP rank."""
        gathered = []

        def all_gather_object(obj):
            gathered.append(obj)
            return [
                {"dp_rank": 1, "endpoint": "http://10.0.0.2:30000"},
                None,
                obj,
                None,
            ]

        with (
            patch.object(
                server_module,
                "get_exec",
                return_value=SimpleNamespace(
                    moe=SimpleNamespace(is_ep_scale_joiner=False)
                ),
            ),
            patch.object(
                server_module,
                "get_parallel",
                return_value=SimpleNamespace(
                    nnodes=1,
                    pp_size=1,
                    tp_size=2,
                    tp_rank=0,
                    attn_tp_size=1,
                    attn_cp_size=1,
                    attn_dp_size=2,
                    attn_dp_rank=0,
                    dp_size=2,
                    attn_dp_enabled=True,
                    dp_rank=0,
                    world_group=SimpleNamespace(all_gather_object=all_gather_object),
                ),
            ),
            patch.object(
                server_module,
                "get_serving",
                return_value=SimpleNamespace(host="0.0.0.0", port=30000),
            ),
            patch.object(server_module, "get_local_ip_auto", return_value="10.0.0.1"),
        ):
            table = RustServer.gather_server_endpoints(hosts_listener=True)
            others_only = RustServer.gather_server_endpoints(hosts_listener=False)

        self.assertEqual(
            gathered[0], {"dp_rank": 0, "endpoint": "http://10.0.0.1:30000"}
        )
        self.assertIsNone(gathered[1], "a non-listener rank contributes nothing")
        self.assertEqual(
            others_only, [{"dp_rank": 1, "endpoint": "http://10.0.0.2:30000"}]
        )
        self.assertEqual(
            table,
            [
                {"dp_rank": 0, "endpoint": "http://10.0.0.1:30000"},
                {"dp_rank": 1, "endpoint": "http://10.0.0.2:30000"},
            ],
        )

    def test_gather_server_endpoints_warns_on_an_unreachable_loopback_host(self):
        """A multi-node launch whose listener is loopback warns, naming the
        host and rendezvous address, unless the nodes rendezvous on loopback
        too (several nodes emulated on one machine), and still gathers."""
        all_gather_object = Mock(return_value=[None, None])
        for dist_init_addr, warns in (
            ("10.0.0.1:5000", True),
            ("127.0.0.1:5000", False),
            ("[::1]:5000", False),
            (None, True),
        ):
            parallel = SimpleNamespace(
                nnodes=2,
                dist_init_addr=dist_init_addr,
                world_group=SimpleNamespace(all_gather_object=all_gather_object),
            )
            with (
                patch.object(server_module, "get_parallel", return_value=parallel),
                patch.object(
                    server_module,
                    "get_serving",
                    return_value=SimpleNamespace(host="127.0.0.1", port=30000),
                ),
                patch.object(server_module.logger, "warning") as warning,
            ):
                RustServer.gather_server_endpoints(hosts_listener=False)
            self.assertEqual(warning.called, warns, dist_init_addr)
            if warns:
                self.assertIn("127.0.0.1", warning.call_args.args[1])
                self.assertEqual(warning.call_args.args[3], dist_init_addr)
        self.assertEqual(all_gather_object.call_count, 4)

    def test_scheduler_aborts_reach_the_rust_egress(self):
        """An abort the scheduler sends outside the output stream becomes a
        one-request terminal batch in Rust mode, keeping the request's counts
        and marked as outside the stream; without the Rust server it still
        goes to the tokenizer socket, with the request's detokenizer state released."""
        import msgspec

        from sglang.srt.managers.io_struct import AbortReq
        from sglang.srt.managers.scheduler import Scheduler

        req = SimpleNamespace(
            origin_input_ids=[1, 2, 3], cached_tokens=2, reasoning_tokens=1
        )
        abort = AbortReq(rid="r#1", abort_message="gone")
        server = RustServer(Mock(), http_port=30000)
        scheduler = SimpleNamespace(
            rust_server=server, ipc_channels=Mock(), output_streamer=Mock()
        )
        Scheduler.send_abort_output(scheduler, abort, req)
        header, cols = server.server.push_decode_result_batch.call_args.args
        self.assertEqual(
            msgspec.msgpack.decode(header),
            [["r#1"], [{"type": "abort", "message": "gone"}], [3], [0], [2], [1]]
            + [[]] * 17
            + [[True]],
        )
        self.assertEqual(cols, [b""])
        scheduler.ipc_channels.send_to_tokenizer.send_output.assert_not_called()
        scheduler.output_streamer.release_detokenize_state.assert_not_called()

        scheduler.rust_server = None
        Scheduler.send_abort_output(scheduler, abort, req)
        scheduler.ipc_channels.send_to_tokenizer.send_output.assert_called_once_with(
            abort, req
        )
        scheduler.output_streamer.release_detokenize_state.assert_called_once_with(req)

    def test_mixed_logprob_batch_frame(self):
        """In a batch where only some rows carry each logprob family, every row
        reads back its own cells from the frame (`for_each_chunk`'s column
        order), and a family no row carries leaves its header columns empty
        instead of a zero per request."""
        import msgspec

        no_cells = [[] for _ in range(4)]
        payload = SimpleNamespace(
            rids=["a", "b", "c", "d"],
            finished_reasons=[None] * 4,
            prompt_tokens=[3, 3, 3, 3],
            output_ids=[
                array("q", [7, 8]),
                array("q", [9]),
                array("q", [5]),
                array("q"),
            ],
            cached_tokens=[0] * 4,
            reasoning_tokens=[0] * 4,
            weight_versions=None,
            image_tokens=None,
            audio_tokens=None,
            video_tokens=None,
            output_token_logprobs_val=[[-0.5, -0.25], [], [-1.5], []],
            output_token_logprobs_idx=[[7, 8], [], [5], []],
            input_token_logprobs_val=[[], [None, -2.0, -3.0], [], []],
            input_token_logprobs_idx=[[], [1, 2, 3], [], []],
            output_top_logprobs_val=[[[-0.5, -1.0], [-0.25]], [], [], []],
            output_top_logprobs_idx=[[[7, 4], [8]], [], [], []],
            input_top_logprobs_val=[[], [None, [-2.0]], [], []],
            input_top_logprobs_idx=[[], [None, [2]], [], []],
            output_token_ids_logprobs_val=[[], [], [[-4.0, -6.0]], []],
            output_token_ids_logprobs_idx=[[], [], [[11, 12]], []],
            input_token_ids_logprobs_val=no_cells,
            input_token_ids_logprobs_idx=no_cells,
            output_hidden_states=[None, None, None, [[0.5, 1.5]]],
            customized_info=None,
        )
        server = RustServer(Mock(), http_port=0)
        server.push_generation(payload, {})
        header, data_cols = server.server.push_decode_result_batch.call_args.args
        h = msgspec.msgpack.decode(header)
        data = b"".join(data_cols)
        offset = 0

        def take(typecode, count):
            nonlocal offset
            out = array(typecode)
            out.frombytes(data[offset : offset + count * out.itemsize])
            offset += count * out.itemsize
            return [None if x != x else x for x in out]  # NaN -> None

        def by_row(counts, items):
            starts = [sum(counts[:r]) for r in range(len(counts))]
            return [items[s : s + c] for s, c in zip(starts, counts)]

        def flat(col):
            vals, idxs = take("f", sum(h[col])), take("i", sum(h[col]))
            return [
                list(zip(*p)) for p in zip(by_row(h[col], vals), by_row(h[col], idxs))
            ]

        def ragged(col):
            reqlens, poslens = h[col], h[col + 1]
            vals, idxs = take("f", sum(poslens)), take("i", sum(poslens))
            positions = [
                list(zip(*p)) or None
                for p in zip(by_row(poslens, vals), by_row(poslens, idxs))
            ]
            return by_row(reqlens, positions)

        self.assertEqual(by_row(h[3], take("q", sum(h[3]))), [[7, 8], [9], [5], []])
        self.assertEqual(flat(6), [[(-0.5, 7), (-0.25, 8)], [], [(-1.5, 5)], []])
        self.assertEqual(flat(7), [[], [(None, 1), (-2.0, 2), (-3.0, 3)], [], []])
        self.assertEqual(
            ragged(8), [[[(-0.5, 7), (-1.0, 4)], [(-0.25, 8)]], [], [], []]
        )
        self.assertEqual(ragged(10), [[], [None, [(-2.0, 2)]], [], []])
        self.assertEqual(ragged(12), [[], [], [[(-4.0, 11), (-6.0, 12)]], []])
        # input_token_ids_logprobs: no row carries it.
        self.assertEqual((h[14], h[15]), ([], []))
        self.assertEqual((h[16], h[17]), ([0, 0, 0, 1], [2]))
        self.assertEqual(take("f", 2), [0.5, 1.5])
        self.assertEqual(offset, len(data))

    def test_customized_info_goes_out_as_python_responses_render_it(self):
        """The egress sends customized_info as the JSON Python's responses
        render: numpy values as lists (float32 at its own precision), NaN as
        null, non-string keys as strings; a value orjson has no form for (bytes,
        which fail a Python response) as its `str`. Even a batch orjson cannot
        encode goes out, without its customized_info."""
        import msgspec
        import numpy as np

        server = RustServer(Mock(), http_port=0)
        payload = generation_payload(
            {"r": [7, 8]},
            customized_info={
                "vec": [[np.float32(0.1), np.array([1, 2])]],
                "odd": [[{1: float("nan")}, b"x"]],
            },
        )
        server.push_generation(payload, {})
        header, _ = server.server.push_decode_result_batch.call_args.args
        self.assertEqual(
            msgspec.msgpack.decode(header)[-1],
            b'{"vec":[[0.1,[1,2]]],"odd":[[{"1":null},"b\'x\'"]]}',
        )

        payload.customized_info = {"big": [[1 << 70, None]]}
        server.push_generation(payload, {})
        header, _ = server.server.push_decode_result_batch.call_args.args
        self.assertEqual(len(msgspec.msgpack.decode(header)), 6)

    def test_frame_metadata_fixture_matches_python(self):
        with open(RUST_FRAME_METADATA_FIXTURE) as f:
            self.assertEqual(json.load(f), build_rust_frame_metadata_fixture())
        with open(RUST_CUSTOMIZED_INFO_FIXTURE) as f:
            self.assertEqual(json.load(f), build_rust_customized_info_fixture())

    def test_launch_resolves_the_mm_pipeline_before_the_server_starts(self):
        """The listener answers once the server is constructed, so a model
        without a Rust MM pipeline must be refused before that; a supported
        one gets workers for the spec resolved up front."""
        extension = ModuleType("server")
        extension.Server = Mock()

        class BuiltinServer(RustServer):
            @classmethod
            def _load_extension(cls):
                return extension

        scheduler = SimpleNamespace(
            server_args=sentinel.server_args,
            processor=sentinel.processor,
            model_config=SimpleNamespace(
                is_multimodal=True,
                hf_config=SimpleNamespace(model_type="unsupported_vlm"),
            ),
        )
        with (
            launch_context(BuiltinServer),
            patch.object(server_module, "RustMmProcessor") as mm_processor,
            patch.object(
                BuiltinServer, "_build_mm_spec", return_value=sentinel.worker_spec
            ),
        ):
            mm_processor.return_value.resolve_spec.return_value = None
            with self.assertRaisesRegex(
                RuntimeError, "no Rust MM pipeline for model_type='unsupported_vlm'"
            ):
                BuiltinServer.launch(scheduler)
            extension.Server.assert_not_called()

            mm_processor.return_value.resolve_spec.return_value = sentinel.spec
            instance = BuiltinServer.launch(scheduler)

        extension.Server.assert_called_once()
        instance.server.start_mm_workers.assert_called_once_with(
            sentinel.worker_spec, 8
        )
        self.assertIs(instance.mm_spec, sentinel.spec)
        self.assertTrue(instance._multimodal_enabled)

    @unittest.skipUnless(
        hasattr(os, "sched_getaffinity") and len(os.sched_getaffinity(0)) >= 2,
        "needs two CPUs",
    )
    def test_launch_confines_mm_threads_but_leaves_the_scheduler_unpinned(self):
        """MM threads are created on the server cores, and the scheduler
        thread gets its full mask back: pinned to the few reserved cores, it
        waits out whatever else runs there, and every TP rank waits on it."""
        full_mask = os.sched_getaffinity(0)
        launch_core, *server_cores = sorted(full_mask)
        extension = ModuleType("model_server")
        extension.Server = Mock()
        masks = {}

        class ModelServer(RustServer):
            @classmethod
            def _load_extension(cls):
                return extension

            @classmethod
            def _resolve_multimodal(cls, scheduler):
                return None

            def _start_multimodal(self, scheduler):
                masks["mm"] = os.sched_getaffinity(0)

        def launch():
            # Affinity is per thread: launch on a thread of its own, as the
            # scheduler's launch thread, so the test runner keeps its mask.
            scheduler = SimpleNamespace(
                model_config=SimpleNamespace(is_multimodal=True)
            )
            with launch_context(ModelServer, cores=([launch_core], server_cores)):
                ModelServer.launch(scheduler)
            masks["scheduler"] = os.sched_getaffinity(0)

        thread = threading.Thread(target=launch)
        thread.start()
        thread.join()

        self.assertEqual(masks["mm"], set(server_cores))
        self.assertEqual(masks["scheduler"], full_mask)


if __name__ == "__main__":
    unittest.main()
