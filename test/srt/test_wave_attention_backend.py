"""
Usage:
python3 -m unittest test_wave_attention_backend.TestWaveAttnBackend.test_mmlu
"""

import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    is_in_ci,
    popen_launch_server,
    run_bench_one_batch,
)


def is_wave_backend_available():
    """Check if wave backend dependencies are available."""
    try:
        import wave_lang.kernel.lang.global_symbols

        return True
    except ImportError:
        return False


class TestWaveAttnBackend(unittest.TestCase):
    def setUp(self):
        """Skip tests if wave backend is not available."""
        if not is_wave_backend_available():
            self.skipTest(
                "wave_lang dependency not available - wave backend not supported"
            )

    def test_latency(self):
        try:
            _, output_throughput, _ = run_bench_one_batch(
                DEFAULT_MODEL_NAME_FOR_TEST,
                [
                    "--attention-backend",
                    "wave",
                    "--enable-torch-compile",
                ],
            )

            if is_in_ci():
                self.assertGreater(output_throughput, 153)
        except Exception as e:
            if "wave_lang" in str(e) or "No module named 'wave_lang'" in str(e):
                self.skipTest(f"wave_lang dependency not available: {e}")
            else:
                raise

    def test_mmlu(self):
        """Test MMLU evaluation with wave attention backend."""
        model = DEFAULT_MODEL_NAME_FOR_TEST
        base_url = DEFAULT_URL_FOR_TEST

        try:
            process = popen_launch_server(
                model,
                base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=["--attention-backend", "wave"],
            )

            try:
                args = SimpleNamespace(
                    base_url=base_url,
                    model=model,
                    eval_name="mmlu",
                    num_examples=64,
                    num_threads=32,
                )

                metrics = run_eval(args)
                self.assertGreaterEqual(metrics["score"], 0.65)
            finally:
                kill_process_tree(process.pid)
        except Exception as e:
            if "wave_lang" in str(e) or "No module named 'wave_lang'" in str(e):
                self.skipTest(f"wave_lang dependency not available: {e}")
            else:
                raise


if __name__ == "__main__":
    unittest.main()
