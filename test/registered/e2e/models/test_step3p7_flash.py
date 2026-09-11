import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval_once
from sglang.test.simple_eval_common import make_report
from sglang.test.simple_eval_mmmu_vlm import MMMUVLMEval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    terminate_and_kill_process_tree,
    write_github_step_summary,
)

# Two cold server launches and a serialized MMMU baseline cost more than
# the Step3.5 GSM8K test. This is an estimate pending the first GPU CI run.
register_cuda_ci(est_time=1800, stage="extra-b", runner_config="8-gpu-h200")


class TestStep3p7Flash(CustomTestCase):
    model = "stepfun-ai/Step-3.7-Flash"
    base_url = DEFAULT_URL_FOR_TEST
    # Reuse the Step3.5 Flash E2E model-loading and TP configuration.
    other_args = [
        "--tp",
        "8",
        "--trust-remote-code",
        "--attention-backend",
        "fa3",
        "--mem-fraction-static",
        "0.75",
        "--chunked-prefill-size",
        "8192",
        "--disable-radix-cache",
        "--model-loader-extra-config",
        '{"enable_multithread_load": true, "num_threads": 64}',
    ]

    def test_mmmu_serial_vs_concurrent(self):
        # Reuse the nightly MMMU dataset selection, prompts, answer parser,
        # and scorer. Sharing this object guarantees identical samples.
        evaluator = MMMUVLMEval(num_examples=100, num_threads=64)
        self.assertEqual(len(evaluator.samples), 100)
        args = SimpleNamespace(model=self.model, max_tokens=1024, temperature=0)
        report_dir = Path(tempfile.mkdtemp(prefix="step3p7_mmmu_"))
        print(f"Step3.7 MMMU reports: {report_dir}")
        scores = {}

        for mode, max_running_requests in [("serial", 1), ("concurrent", 8)]:
            # Restart for each mode: a warm vision embedding cache could hide
            # the changed encoder path. The client workload is unchanged;
            # only the scheduler's maximum request concurrency differs.
            process = popen_launch_server(
                self.model,
                self.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 3,
                other_args=self.other_args
                + ["--max-running-requests", str(max_running_requests)],
            )
            try:
                result, latency, _ = run_eval_once(
                    args, self.base_url + "/v1", evaluator
                )
                scores[mode] = result.score
                (report_dir / f"{mode}.html").write_text(make_report(result))
                (report_dir / f"{mode}.json").write_text(
                    json.dumps(
                        {
                            "score": result.score,
                            "latency": latency,
                            "sample_ids": [s["id"] for s in evaluator.samples],
                            "answers": [c[-1]["content"] for c in result.convos],
                        },
                        indent=2,
                    )
                )
                self.assertEqual(len(result.convos), len(evaluator.samples))
                self.assertTrue(
                    all(c[-1]["content"].strip() for c in result.convos),
                    f"{mode}: empty responses; see {report_dir}",
                )
                # Reject a vacuous comparison where both runs score zero.
                self.assertGreater(result.score, 0, f"{mode}: {report_dir}")
            finally:
                terminate_and_kill_process_tree(process, wait_timeout=60)

        summary = (
            f"Step3.7 MMMU (100 samples): serial={scores['serial']:.4f}, "
            f"concurrent={scores['concurrent']:.4f}. Reports: {report_dir}"
        )
        print(summary)
        if is_in_ci():
            write_github_step_summary(summary + "\n")
        # No measured Step3.7 absolute threshold exists yet. Require no score
        # regression relative to the same-checkpoint serialized control;
        # inspect the paired reports if batch-dependent numerics change it.
        self.assertGreaterEqual(scores["concurrent"], scores["serial"], summary)


if __name__ == "__main__":
    unittest.main()
