import unittest


class PDFlipTraceReplayTest(unittest.TestCase):
    def test_build_trace_creates_mixed_200_request_workload(self):
        from scripts.playground.disaggregation.pd_flip_trace_replay import build_trace

        trace = build_trace(
            num_requests=200,
            interval_seconds=1.0,
            model="deepseek_v3.1_terminus",
            seed=7,
        )

        self.assertEqual(len(trace), 200)
        self.assertEqual([r["arrival_offset_s"] for r in trace[:4]], [0.0, 1.0, 2.0, 3.0])
        self.assertEqual(trace[-1]["arrival_offset_s"], 199.0)

        kinds = {record["prompt_kind"] for record in trace}
        self.assertEqual(kinds, {"short", "medium", "long"})

        first = trace[0]
        self.assertEqual(first["request_id"], "trace-0000")
        self.assertGreater(first["ttft_slo_s"], 0)
        self.assertGreater(first["tpot_slo_s"], 0)
        body = first["body"]
        self.assertEqual(body["model"], "deepseek_v3.1_terminus")
        self.assertTrue(body["stream"])
        self.assertIn("messages", body)
        self.assertEqual(
            body["custom_params"]["pd_flip_slo"]["ttft_seconds"],
            first["ttft_slo_s"],
        )
        self.assertEqual(
            body["custom_params"]["pd_flip_slo"]["tpot_seconds"],
            first["tpot_slo_s"],
        )

    def test_build_trace_can_generate_non_streaming_requests(self):
        from scripts.playground.disaggregation.pd_flip_trace_replay import build_trace

        trace = build_trace(
            num_requests=3,
            interval_seconds=0.5,
            model="deepseek_v3.1_terminus",
            seed=7,
            stream=False,
        )

        self.assertFalse(trace[0]["stream"])
        self.assertFalse(trace[0]["body"]["stream"])

    def test_build_trace_can_generate_40_request_char_count_mix(self):
        from scripts.playground.disaggregation.pd_flip_trace_replay import build_trace

        trace = build_trace(
            num_requests=40,
            interval_seconds=0.5,
            model="deepseek_v3.1_terminus",
            seed=7,
            short_chars=1000,
            long_chars=10000,
            short_count=20,
            long_count=20,
        )

        self.assertEqual(len(trace), 40)
        self.assertEqual(trace[-1]["arrival_offset_s"], 19.5)
        kinds = [record["prompt_kind"] for record in trace]
        self.assertEqual(kinds.count("short"), 20)
        self.assertEqual(kinds.count("long"), 20)
        self.assertNotIn("medium", kinds)
        for record in trace:
            target = 1000 if record["prompt_kind"] == "short" else 10000
            self.assertLessEqual(abs(record["prompt_chars"] - target), target * 0.02)
            content = record["body"]["messages"][0]["content"]
            self.assertEqual(len(content), record["prompt_chars"])

    def test_extract_non_stream_text_handles_chat_message(self):
        from scripts.playground.disaggregation.pd_flip_trace_replay import (
            _extract_non_stream_text,
        )

        choice = {
            "message": {
                "reasoning_content": "think ",
                "content": "answer",
            }
        }

        self.assertEqual(_extract_non_stream_text(choice), "think answer")

    def test_compute_metrics_reports_ttft_tpot_and_slo_attainment(self):
        from scripts.playground.disaggregation.pd_flip_trace_replay import compute_metrics

        record = {
            "request_id": "trace-0001",
            "arrival_offset_s": 1.0,
            "ttft_slo_s": 0.50,
            "tpot_slo_s": 0.20,
        }

        metrics = compute_metrics(
            record,
            scheduled_monotonic=10.0,
            start_monotonic=10.0,
            first_token_monotonic=10.4,
            token_monotonic_times=[10.4, 10.55, 10.90],
            end_monotonic=11.0,
            status="completed",
            error=None,
        )

        self.assertAlmostEqual(metrics["ttft_s"], 0.4)
        self.assertAlmostEqual(metrics["avg_tpot_s"], 0.25)
        self.assertAlmostEqual(metrics["p95_tpot_s"], 0.35)
        self.assertEqual(metrics["good_tpot_intervals"], 1)
        self.assertEqual(metrics["total_tpot_intervals"], 2)
        self.assertTrue(metrics["ttft_met"])
        self.assertFalse(metrics["tpot_avg_met"])
        self.assertFalse(metrics["all_met"])


if __name__ == "__main__":
    unittest.main()
