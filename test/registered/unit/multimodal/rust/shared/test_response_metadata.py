"""Check native response fixtures against the actual Python formatter."""

import base64
import json
import unittest
from array import array
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def speculative_fixtures():
    maybe_stub_sgl_kernel()
    from sglang.srt.managers import tokenizer_manager

    cases = [
        (
            "static",
            4,
            False,
            {
                "completion_tokens": [9, 6, 0],
                "spec_verify_ct": [2, 3, 0],
                "spec_num_correct_drafts": [5, 0, 8],
                "spec_num_cap_tokens": [8, 0, 0],
                "spec_num_block_accept_tokens": [7, 2, 0],
                "spec_correct_drafts_histogram": [[0, 1, 1], [], [1]],
                "spec_cap_lens_histogram": [[1, 1], [], []],
            },
        ),
        (
            "ragged_cap_accept",
            8,
            True,
            {
                "completion_tokens": [13, 4],
                "spec_verify_ct": [2, 4],
                "spec_num_correct_drafts": [11, 0],
                "spec_num_cap_tokens": [14, 4],
                "spec_num_block_accept_tokens": [12, 0],
                "spec_correct_drafts_histogram": [[0, 0, 2], [4]],
                "spec_cap_lens_histogram": [[0, 0, 1, 1], [4]],
            },
        ),
        (
            "no_proposals",
            1,
            True,
            {
                "completion_tokens": [2],
                "spec_verify_ct": [2],
                "spec_num_correct_drafts": [0],
                "spec_num_cap_tokens": [2],
                "spec_num_block_accept_tokens": [0],
                "spec_correct_drafts_histogram": [[2]],
                "spec_cap_lens_histogram": [[2]],
            },
        ),
        (
            "unreported_correct_counts",
            4,
            False,
            {
                "completion_tokens": [4],
                "spec_verify_ct": [1],
                "spec_num_correct_drafts": [],
                "spec_correct_drafts_histogram": [[1]],
            },
        ),
        (
            "partial_optional_columns",
            3,
            True,
            {
                "completion_tokens": [3, 1, 2],
                "spec_verify_ct": [1, 1, 1],
                "spec_num_correct_drafts": [2, 0],
                "spec_num_cap_tokens": [0],
                "spec_num_block_accept_tokens": [0],
                "spec_correct_drafts_histogram": [[0, 0, 1]],
            },
        ),
    ]
    fixtures = []
    for name, draft_tokens, cap_accept, columns in cases:
        for field in (
            "spec_num_cap_tokens",
            "spec_num_block_accept_tokens",
            "spec_correct_drafts_histogram",
            "spec_cap_lens_histogram",
        ):
            columns.setdefault(field, [])
        expected = []
        with (
            patch.object(
                tokenizer_manager,
                "get_spec",
                return_value=SimpleNamespace(speculative_num_draft_tokens=draft_tokens),
            ),
            patch.object(
                tokenizer_manager, "_ragged_verify_cap_accept", return_value=cap_accept
            ),
        ):
            for i in range(len(columns["completion_tokens"])):
                meta_info = {}
                tokenizer_manager.TokenizerManager._calculate_spec_decoding_metrics(
                    None, meta_info, SimpleNamespace(**columns), i
                )
                expected.append(meta_info)
        completion_tokens = columns.pop("completion_tokens")
        fixtures.append(
            {
                "name": name,
                "columns": {
                    **columns,
                    "generation_tokens": completion_tokens,
                    "reasoning_tokens": [0] * len(completion_tokens),
                    "cached_tokens": [0] * len(completion_tokens),
                    "spec_num_draft_tokens": draft_tokens,
                    "spec_ragged_verify_cap_accept": cap_accept,
                },
                "expected": expected,
            }
        )
    return fixtures


def timing_fixtures():
    import msgspec

    from sglang.srt.disaggregation.utils import DisaggregationMode
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler_components.output_streamer import (
        SchedulerOutputStreamer,
    )
    from sglang.srt.observability import req_time_stats
    from sglang.srt.runtime_context import get_context
    from sglang.srt.rust_server.server import RustServer
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    epoch = 1_700_000_000
    scheduler = []
    api = []
    with patch.object(
        req_time_stats, "convert_time_to_realtime", lambda ts: epoch + ts
    ):
        for enabled in (False, True):
            with get_context().override_server_args(enable_metrics=enabled):
                native = Mock()
                streamer = SchedulerOutputStreamer(
                    send_to_detokenizer=Mock(),
                    tree_cache=SimpleNamespace(),
                    ps=SimpleNamespace(dp_rank=0),
                    server_args=get_context().server_args,
                    is_generation=True,
                    spec_algorithm=SpeculativeAlgorithm.NONE,
                    disaggregation_mode=DisaggregationMode.NULL,
                    enable_hicache_storage=lambda: False,
                    rust_server=RustServer(native, http_port=30000),
                )
                requests = []
                expected = []
                for i, (wait, forward, prefill) in enumerate(
                    ((0, 0, 0), (10, 14, 15), (20, 22.5, 0))
                ):
                    req = Req(
                        str(i), "", array("q", [1, 2]), SamplingParams(), stream=True
                    )
                    req.output_ids.append(3)
                    req.time_stats = req_time_stats.SchedulerReqTimeStats(
                        disagg_mode=DisaggregationMode.NULL,
                        wait_queue_entry_time=wait,
                        forward_entry_time=forward,
                        prefill_finished_time=prefill,
                    )
                    expected.append(
                        req.time_stats.convert_to_output_meta_info() if enabled else {}
                    )
                    requests.append(req)
                streamer.stream_output(requests, return_logprob=False)
                header, data = native.push_decode_result_batch.call_args.args
                scheduler.append(
                    {
                        "enabled": enabled,
                        "header": msgspec.msgpack.decode(header),
                        "data_b64": base64.b64encode(b"".join(data)).decode(),
                        "expected": expected,
                    }
                )

        for first, finished, sent, tokens in (
            (11, 14, 0, 9),
            (11, 14, 11.25, 9),
            (11, 11, 0, 1),
            (11, 14, 0, 0),
        ):
            times = {
                "created_time": 10,
                "first_token_time": first,
                "finished_time": finished,
                "api_server_dispatch_finish_time": 10.5,
                "response_sent_to_client_time": sent,
            }
            times = {key: float(value) for key, value in times.items()}
            stats = req_time_stats.APIServerReqTimeStats(
                disagg_mode=DisaggregationMode.NULL, **times
            )
            expected = stats.convert_to_output_meta_info(completion_tokens=tokens)
            expected["e2e_latency"] = stats.get_e2e_latency()
            api.append({"times": times, "tokens": tokens, "expected": expected})
    return {"epoch": epoch, "scheduler": scheduler, "api": api}


class TestResponseMetadata(unittest.TestCase):
    def test_timing_fixtures_match_python_and_scheduler_transport(self):
        fixture = (
            Path(__file__).resolve().parents[6]
            / "rust/sglang-server/testdata/timing_stats_python.json"
        )
        self.assertEqual(timing_fixtures(), json.loads(fixture.read_text()))

    def test_speculative_statistics_fixture_matches_python(self):
        fixture = (
            Path(__file__).resolve().parents[6]
            / "rust/sglang-server/testdata/speculative_stats_python.json"
        )
        self.assertEqual(speculative_fixtures(), json.loads(fixture.read_text()))


if __name__ == "__main__":
    unittest.main()
