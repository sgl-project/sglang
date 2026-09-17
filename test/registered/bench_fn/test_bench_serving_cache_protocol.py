import importlib.util
import unittest
from pathlib import Path

_CACHE_PROTOCOL_PATH = (
    Path(__file__).resolve().parents[3]
    / "python"
    / "sglang"
    / "benchmark"
    / "cache_protocol.py"
)
_spec = importlib.util.spec_from_file_location(
    "sglang_benchmark_cache_protocol", _CACHE_PROTOCOL_PATH
)
_cache_protocol = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_cache_protocol)
resolve_benchmark_cache_protocol = _cache_protocol.resolve_benchmark_cache_protocol

try:
    from sglang.test.ci.ci_register import register_cpu_ci

    register_cpu_ci(est_time=1, suite="base-a-test-cpu")
    register_cpu_ci(est_time=1, suite="stage-b-test-cpu-intel")
except ImportError:
    pass


class TestBenchmarkCacheProtocolRecord(unittest.TestCase):
    def test_result_row_stamps_flush_decision_and_provenance(self):
        user_default = resolve_benchmark_cache_protocol("sglang", False, in_ci=False)
        ci = resolve_benchmark_cache_protocol("sglang", False, in_ci=True)
        cli = resolve_benchmark_cache_protocol("sglang", True, in_ci=False)
        cli_and_ci = resolve_benchmark_cache_protocol("sglang", True, in_ci=True)
        vllm_in_ci = resolve_benchmark_cache_protocol("vllm", False, in_ci=True)

        self.assertEqual(
            user_default,
            {
                "flushed_cache": False,
                "cache_flush_reason": "none",
                "cache_state": "uncontrolled",
            },
        )
        self.assertEqual(
            ci,
            {
                "flushed_cache": True,
                "cache_flush_reason": "ci",
                "cache_state": "cold-flushed",
            },
        )
        self.assertEqual(
            cli,
            {
                "flushed_cache": True,
                "cache_flush_reason": "cli",
                "cache_state": "cold-flushed",
            },
        )
        self.assertEqual(
            cli_and_ci,
            {
                "flushed_cache": True,
                "cache_flush_reason": "cli_and_ci",
                "cache_state": "cold-flushed",
            },
        )
        self.assertEqual(
            vllm_in_ci,
            {
                "flushed_cache": False,
                "cache_flush_reason": "none",
                "cache_state": "uncontrolled",
            },
        )
        self.assertEqual(
            resolve_benchmark_cache_protocol(
                "sglang", False, ci_env="1"
            )["flushed_cache"],
            True,
        )
        self.assertEqual(
            resolve_benchmark_cache_protocol(
                "sglang", False, ci_env="true"
            )["cache_flush_reason"],
            "ci",
        )
        self.assertEqual(
            resolve_benchmark_cache_protocol(
                "sglang", False, ci_env=None
            )["flushed_cache"],
            False,
        )

    def test_four_issue_states_are_distinguishable_in_the_record(self):
        prompt_tokens = 512
        user_protocol = resolve_benchmark_cache_protocol("sglang", False, in_ci=False)
        ci_protocol = resolve_benchmark_cache_protocol("sglang", False, in_ci=True)

        def row(protocol, cached_tokens):
            return {
                **protocol,
                "total_prompt_tokens": prompt_tokens,
                "total_cached_tokens": cached_tokens,
            }

        fresh = row(user_protocol, 21)
        rerun = row(user_protocol, 512)
        ci = row(ci_protocol, 0)
        post_flush = row(user_protocol, 21)

        def identity(r):
            return (
                r["flushed_cache"],
                r["cache_flush_reason"],
                r["cache_state"],
                r["total_cached_tokens"],
            )

        self.assertNotEqual(identity(fresh), identity(rerun))
        self.assertNotEqual(identity(fresh), identity(ci))
        self.assertNotEqual(identity(rerun), identity(ci))
        self.assertNotEqual(identity(post_flush), identity(rerun))
        self.assertNotEqual(identity(post_flush), identity(ci))
        self.assertEqual(identity(fresh), identity(post_flush))
        self.assertIn("flushed_cache", fresh)
        self.assertIn("cache_flush_reason", fresh)
        self.assertIn("cache_state", fresh)


if __name__ == "__main__":
    unittest.main()
