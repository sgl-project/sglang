import unittest

from sglang.test.test_utils import (
    DEFAULT_SMALL_VLM_MODEL_NAME_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    run_bench_serving_video,
    write_github_step_summary,
)


class TestBenchServingVideo(CustomTestCase):
    # Video Throughput Tests
    def test_video_throughput(self):
        res = run_bench_serving_video(
            model=DEFAULT_SMALL_VLM_MODEL_NAME_FOR_TEST,
            other_server_args=[],
            num_prompts=20,
            video_seconds=5,
            unique_video=False,
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_video_throughput\n"
                f"Video throughput: {res['throughput_vid_sec']:.2f} vid_sec/s\n"
            )
        self.assertLess(res["throughput_vid_sec"], 90)
        print("Test video throughput:", res["throughput_vid_sec"])

    def test_video_throughput_cache_hit(self):
        res = run_bench_serving_video(
            model=DEFAULT_SMALL_VLM_MODEL_NAME_FOR_TEST,
            other_server_args=[],
            num_prompts=20,
            video_seconds=5,
            unique_video=True,
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_video_throughput_cache_hit\n"
                f"Video throughput: {res['throughput_vid_sec']:.2f} vid_sec/s\n"
            )
        self.assertLess(res["throughput_vid_sec"], 80)
        print("Test video throughput cache hit:", res["throughput_vid_sec"])

    # Token Throughput Tests
    def test_token_throughput(self):
        res = run_bench_serving_video(
            model=DEFAULT_SMALL_VLM_MODEL_NAME_FOR_TEST,
            other_server_args=[],
            num_prompts=20,
            video_seconds=5,
            unique_video=False,
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_token_throughput\n"
                f"Token throughput: {res['output_throughput']:.2f} tok/s\n"
            )
        self.assertLess(res["output_throughput"], 300)
        print("Test token throughput:", res["output_throughput"])

    def test_token_throughput_cache_hit(self):
        res = run_bench_serving_video(
            model=DEFAULT_SMALL_VLM_MODEL_NAME_FOR_TEST,
            other_server_args=[],
            num_prompts=20,
            video_seconds=5,
            unique_video=True,
            request_rate=float("inf"),
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_token_throughput_cache_hit\n"
                f"Token throughput: {res['output_throughput']:.2f} tok/s\n"
            )
        self.assertLess(res["output_throughput"], 275)
        print("Test token throughput cache hit:", res["output_throughput"])

    # Latency Tests
    def test_latency(self):
        res = run_bench_serving_video(
            model=DEFAULT_SMALL_VLM_MODEL_NAME_FOR_TEST,
            other_server_args=[],
            num_prompts=20,
            video_seconds=5,
            unique_video=False,
            request_rate=1,
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_latency\n"
                f"Mean latency: {res['mean_latency_ms']:.2f} ms\n"
            )
        self.assertLess(res["mean_latency_ms"], 300)
        print("Test latency:", res["mean_latency_ms"])


    def test_latency_cache_hit(self):
        res = run_bench_serving_video(
            model=DEFAULT_SMALL_VLM_MODEL_NAME_FOR_TEST,
            other_server_args=[],
            num_prompts=20,
            video_seconds=5,
            unique_video=True,
            request_rate=1,
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_latency_cache_hit\n"
                f"Mean latency: {res['mean_latency_ms']:.2f} ms\n"
            )
        self.assertLess(res["mean_latency_ms"], 250)
        print("Test latency cache hit:", res["mean_latency_ms"])


if __name__ == "__main__":
    unittest.main()
