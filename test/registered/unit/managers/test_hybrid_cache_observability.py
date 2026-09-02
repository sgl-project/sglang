import unittest
from types import SimpleNamespace

from sglang.srt.entrypoints.openai.utils import cached_tokens_details_from_dict
from sglang.srt.managers.scheduler_components.output_streamer import (
    SchedulerOutputStreamer,
    get_hybrid_cache_details,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestHybridCacheObservability(unittest.TestCase):
    def test_reports_raw_joint_trimmed_and_compute_tokens(self):
        req = SimpleNamespace(
            origin_input_ids=list(range(21)),
            cached_tokens=8,
            cached_tokens_device=8,
            cached_tokens_host=0,
            cached_tokens_storage=0,
            hybrid_cache_full_attention_cached_tokens=16,
            hybrid_cache_recurrent_state_cached_tokens=8,
        )

        details = get_hybrid_cache_details(req)

        self.assertEqual(details["full_attention_cached_tokens"], 16)
        self.assertEqual(details["recurrent_state_cached_tokens"], 8)
        self.assertEqual(details["usable_cached_tokens"], 8)
        self.assertEqual(details["trimmed_full_attention_tokens"], 8)
        self.assertEqual(details["full_attention_recomputed_tokens"], 13)
        self.assertEqual(details["recurrent_recomputed_tokens"], 13)
        self.assertEqual(details["recurrent_replayed_tokens"], 0)

    def test_native_and_openai_details_preserve_hybrid_fields(self):
        req = SimpleNamespace(
            origin_input_ids=list(range(9)),
            cached_tokens=0,
            cached_tokens_device=0,
            cached_tokens_host=0,
            cached_tokens_storage=0,
            hybrid_cache_full_attention_cached_tokens=0,
            hybrid_cache_recurrent_state_cached_tokens=0,
        )
        streamer = SimpleNamespace(enable_hicache_storage=lambda: False)

        details = SchedulerOutputStreamer.get_cached_tokens_details(streamer, req)
        wire_details = cached_tokens_details_from_dict(details).model_dump(
            exclude_none=True
        )

        self.assertEqual(wire_details, details)
        self.assertEqual(wire_details["usable_cached_tokens"], 0)
        self.assertEqual(wire_details["full_attention_recomputed_tokens"], 9)

    def test_non_hybrid_request_keeps_existing_shape(self):
        req = SimpleNamespace(
            origin_input_ids=list(range(9)),
            cached_tokens=4,
            cached_tokens_device=4,
            cached_tokens_host=0,
            cached_tokens_storage=0,
        )
        streamer = SimpleNamespace(enable_hicache_storage=lambda: False)

        details = SchedulerOutputStreamer.get_cached_tokens_details(streamer, req)

        self.assertEqual(details, {"device": 4, "host": 0})


if __name__ == "__main__":
    unittest.main()
