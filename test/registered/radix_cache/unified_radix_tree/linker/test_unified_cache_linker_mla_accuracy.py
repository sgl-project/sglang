"""Task accuracy of the direct linker on a plain MLA pool.

One server scores the same evaluation twice: first with an empty store, then
after flushing the device tier. The second round therefore restores every
prompt's KV from Mooncake instead of recomputing it, and must reproduce the
same answers. A restore path that mixes up or truncates pages shows up as
different generations here long before a reviewer would notice.

`--enable-deterministic-inference` keeps greedy decoding reproducible across
the two rounds, so "same generations" is a real assertion rather than a
tolerance. `SGLANG_LINKER_MLA_ACCURACY_DATASET` switches the benchmark;
gsm8k over 100 examples is the statistically stronger variant (57.00% cold and
57.00% restored measured on DeepSeek-V2-Lite-Chat), aime25 is the default
because it is the harder, coarser signal.
"""

import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kl_test_utils import _flush_cache
from sglang.test.mooncake_utils import MooncakeTestServices
from sglang.test.sgl_eval_utils import run_sgl_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    find_available_port,
    popen_launch_server,
    terminate_and_kill_process_tree,
    unified_radix_tree_server_env,
)

register_cuda_ci(est_time=600, stage="extra-b", runner_config="1-gpu-large")

_MODEL_PATH = os.environ.get(
    "SGLANG_LINKER_MLA_ACCURACY_MODEL", "deepseek-ai/DeepSeek-V2-Lite-Chat"
)
_DATASET = os.environ.get("SGLANG_LINKER_MLA_ACCURACY_DATASET", "aime25")
_NUM_EXAMPLES = int(os.environ.get("SGLANG_LINKER_MLA_ACCURACY_EXAMPLES", "12"))
_MAX_TOKENS = int(os.environ.get("SGLANG_LINKER_MLA_ACCURACY_MAX_TOKENS", "8192"))
# Measured score minus a margin; 0 for the datasets whose score is too coarse
# over this many examples to carry the assertion on its own.
BASELINE_SCORE = float(os.environ.get("SGLANG_LINKER_MLA_ACCURACY_BASELINE", "0.0"))


class TestPlainMLALinkerAccuracy(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = _MODEL_PATH
        cls.base_url = f"http://127.0.0.1:{find_available_port(30000)}"
        cls.mooncake = MooncakeTestServices()
        cls.addClassCleanup(cls.mooncake.stop)
        cls.mooncake.start()
        cls.out_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "linker_mla_accuracy"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--attention-backend",
                "triton",
                "--page-size",
                "64",
                # Greedy decoding must be reproducible across the two rounds.
                "--enable-deterministic-inference",
                "--mem-fraction-static",
                "0.85",
                "--enable-cache-report",
                "--enable-unified-cache-external-linker",
            ],
            env=unified_radix_tree_server_env("python", **cls.mooncake.server_env()),
        )
        cls.addClassCleanup(
            terminate_and_kill_process_tree, cls.process, wait_timeout=60
        )

    def _score(self) -> tuple[dict, list[str]]:
        metrics = run_sgl_eval(
            SimpleNamespace(
                eval_name=_DATASET,
                base_url=f"{self.base_url}/v1",
                model=self.model,
                num_examples=_NUM_EXAMPLES,
                num_threads=1,
                # One pass: a repeat would multiply the run by the spec's default.
                repeat=1,
                max_tokens=_MAX_TOKENS,
                temperature=0.0,
                sgl_eval_out_dir=str(self.out_dir),
            )
        )
        predictions = sorted(
            (Path(metrics["sgl_eval_metrics_path"]).parent).glob("*.jsonl")
        )
        generations = [
            json.loads(line)["generation"]
            for line in predictions[0].read_text().splitlines()
        ]
        return metrics, generations

    def test_store_restore_preserves_task_accuracy(self):
        cold_metrics, cold_generations = self._score()

        _flush_cache(self.base_url)

        restored_metrics, restored_generations = self._score()

        for name, metrics in (("cold", cold_metrics), ("restored", restored_metrics)):
            self.assertGreaterEqual(
                metrics["score"], BASELINE_SCORE, f"{name} score regressed"
            )
        self.assertEqual(restored_metrics["score"], cold_metrics["score"])
        self.assertEqual(
            restored_metrics["truncated_rate"], cold_metrics["truncated_rate"]
        )
        self.assertEqual(len(restored_generations), len(cold_generations))
        for index, (restored, cold) in enumerate(
            zip(restored_generations, cold_generations)
        ):
            self.assertEqual(
                restored, cold, f"restored generation differs at example {index}"
            )
        print(
            f"{_DATASET}: cold={cold_metrics['score']:.4f} "
            f"restored={restored_metrics['score']:.4f} "
            f"examples={_NUM_EXAMPLES} "
            f"latency cold={cold_metrics['latency']:.1f}s "
            f"restored={restored_metrics['latency']:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    unittest.main()