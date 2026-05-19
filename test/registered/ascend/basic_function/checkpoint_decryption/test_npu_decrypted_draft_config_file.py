import os
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_8B_EAGLE3_WEIGHTS_PATH,
    QWEN3_8B_WEIGHTS_PATH,
    run_command,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(
    est_time=400,
    suite="nightly-2-npu-a3",
    nightly=True,
)


class TestDraftConfigFile(CustomTestCase):
    """Testcase: Verify set --decrypted-config-file, --decrypted-draft-config-file parameter,
    will use the specified config.json and the GSM8K dataset is no less than 0.95.

    [Test Category] Parameter
    [Test Target] --decrypted-config-file, --decrypted-draft-config-file
    """

    @classmethod
    def setUpClass(cls):
        # Modify the config.json under the weight path
        run_command(
            f"mv {os.path.join(QWEN3_8B_WEIGHTS_PATH, 'config.json')} {os.path.join(QWEN3_8B_WEIGHTS_PATH, '_config.json')}"
        )
        run_command(
            f"mv {os.path.join(QWEN3_8B_EAGLE3_WEIGHTS_PATH, 'config.json')} {os.path.join(QWEN3_8B_EAGLE3_WEIGHTS_PATH, '_config.json')}"
        )
        try:
            cls.process = popen_launch_server(
                QWEN3_8B_WEIGHTS_PATH,
                DEFAULT_URL_FOR_TEST,
                DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--trust-remote-code",
                    "--attention-backend",
                    "ascend",
                    "--disable-radix-cache",
                    "--chunked-prefill-size",
                    "-1",
                    "--max-prefill-tokens",
                    "1024",
                    "--speculative-algorithm",
                    "EAGLE3",
                    "--speculative-draft-model-path",
                    QWEN3_8B_EAGLE3_WEIGHTS_PATH,
                    "--speculative-num-steps",
                    "3",
                    "--speculative-eagle-topk",
                    "1",
                    "--speculative-num-draft-tokens",
                    "4",
                    "--tp-size",
                    "2",
                    "--mem-fraction-static",
                    "0.68",
                    "--disable-cuda-graph",
                    "--dtype",
                    "bfloat16",
                    "--decrypted-config-file",
                    "/__w/sglang/sglang/test/registered/ascend/basic_function/checkpoint_decryption/Qwen3-8B/config.json",
                    "--decrypted-draft-config-file",
                    "/__w/sglang/sglang/test/registered/ascend/basic_function/checkpoint_decryption/Qwen3-8B_eagle3/config.json",
                ],
                env={
                    "SGLANG_ENABLE_SPEC_V2": "1",
                    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
                },
            )

        except Exception as e:
            # Service failed to start, restoring original file name
            run_command(
                f"mv {os.path.join(QWEN3_8B_WEIGHTS_PATH, '_config.json')} {os.path.join(QWEN3_8B_WEIGHTS_PATH, 'config.json')}"
            )
            run_command(
                f"mv {os.path.join(QWEN3_8B_EAGLE3_WEIGHTS_PATH, '_config.json')} {os.path.join(QWEN3_8B_EAGLE3_WEIGHTS_PATH, 'config.json')}"
            )
            if cls.process:
                kill_process_tree(cls.process.pid)
            raise RuntimeError(f"Failed to launch server: {e}") from e

    @classmethod
    def tearDownClass(cls):
        run_command(
            f"mv {os.path.join(QWEN3_8B_WEIGHTS_PATH, '_config.json')} {os.path.join(QWEN3_8B_WEIGHTS_PATH, 'config.json')}"
        )
        run_command(
            f"mv {os.path.join(QWEN3_8B_EAGLE3_WEIGHTS_PATH, '_config.json')} {os.path.join(QWEN3_8B_EAGLE3_WEIGHTS_PATH, 'config.json')}"
        )
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            max_new_tokens=512,
            base_url=DEFAULT_URL_FOR_TEST,
            model=QWEN3_8B_WEIGHTS_PATH,
            eval_name="gsm8k",
            api="completion",
            num_examples=200,
            num_threads=128,
            num_shots=5,
        )
        metrics = run_eval(args)
        self.assertGreater(metrics["score"], 0.95)


if __name__ == "__main__":
    unittest.main()
