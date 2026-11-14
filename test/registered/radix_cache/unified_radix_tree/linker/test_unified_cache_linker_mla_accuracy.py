"""Accuracy of the direct linker on a plain MLA pool, over long prompts.

The question this answers is narrow: with `--enable-unified-cache-external-linker`
enabled, a prefix restored from Mooncake must decode exactly like the same prefix
computed on the device.

Prompt length decides whether that is even testable. The store is page-granular,
so a 226-token prompt only ever restores its three full pages (~8% of the
request) and a corrupted page has little chance of changing the answer. These
samples are ~3k tokens, so >95% of the prompt comes back from the store and the
comparison covers the whole read path: one wrong page changes the logprobs.
"""

import os
import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kl_test_utils import (
    _extract_output_logprobs,
    _flush_cache,
    _generate,
    get_input_ids,
)
from sglang.test.mooncake_utils import MooncakeTestServices
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    find_available_port,
    popen_launch_server,
    terminate_and_kill_process_tree,
    unified_radix_tree_server_env,
)

register_cuda_ci(
    est_time=900,
    stage="extra-b",
    # Direct GPU transfers require an RDMA-capable runner, even for TP=1.
    runner_config="4-gpu-h100",
)

_MODEL_PATH = os.environ.get(
    "SGLANG_LINKER_MLA_ACCURACY_MODEL", "deepseek-ai/DeepSeek-V2-Lite-Chat"
)

PAGE_SIZE = 64
PROMPT_TOKENS = 3000
NUM_SAMPLES = 4
MAX_NEW_TOKENS = 32
# Whole pages only, minus the last partial one: 3000 tokens restore 2944.
MIN_RESTORE_RATIO = 0.9


class _MLALinkerCase(CustomTestCase):
    """One linker-enabled MLA server with long prompts loaded."""

    @classmethod
    def setUpClass(cls):
        cls.model = _MODEL_PATH
        cls.base_url = f"http://127.0.0.1:{find_available_port(30000)}"
        cls.mooncake = MooncakeTestServices()
        cls.addClassCleanup(cls.mooncake.stop)
        cls.mooncake.start()
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--attention-backend",
                "triton",
                "--page-size",
                str(PAGE_SIZE),
                # Greedy decoding must be reproducible across the two rounds.
                "--enable-deterministic-inference",
                "--mem-fraction-static",
                "0.85",
                "--max-total-tokens",
                "16384",
                "--enable-cache-report",
                "--enable-unified-cache-external-linker",
            ],
            env=unified_radix_tree_server_env("python", **cls.mooncake.server_env()),
        )
        cls.addClassCleanup(
            terminate_and_kill_process_tree, cls.process, wait_timeout=60
        )
        cls.input_ids = [
            ids[:PROMPT_TOKENS]
            for ids in get_input_ids(
                tokenizer_path=cls.model,
                max_prompt_tokens=PROMPT_TOKENS,
                num_samples=NUM_SAMPLES,
                trust_remote_code=True,
            )
        ]

    def _run(self):
        return [
            _generate(self.base_url, [ids], MAX_NEW_TOKENS, return_logprob=True)[0]
            for ids in self.input_ids
        ]


class TestPlainMLALinkerRestoreAccuracy(_MLALinkerCase):
    def test_restored_prefix_decodes_like_a_computed_one(self):
        cold_results = self._run()

        _flush_cache(self.base_url)

        restored_results = self._run()

        restores = []
        for index, (cold, restored) in enumerate(zip(cold_results, restored_results)):
            meta = restored["meta_info"]
            details = meta.get("cached_tokens_details") or {}
            prompt_tokens = int(meta["prompt_tokens"])
            store_tokens = int(details.get("host", 0))
            restores.append(f"{store_tokens}/{prompt_tokens}")

            # Without this the decode comparison below could pass on a run whose
            # lookup missed entirely.
            self.assertGreaterEqual(
                store_tokens,
                MIN_RESTORE_RATIO * prompt_tokens,
                f"sample {index}: only {store_tokens} of {prompt_tokens} prompt "
                "tokens came from the store",
            )
            self.assertEqual(
                int(details.get("device", 0)), 0, f"sample {index}: device hit"
            )
            self.assertEqual(
                restored["output_ids"],
                cold["output_ids"],
                f"sample {index}: restored decode diverged",
            )
            self.assertEqual(
                _extract_output_logprobs(restored),
                _extract_output_logprobs(cold),
                f"sample {index}: restored logprobs diverged",
            )

        print(
            f"store-restored prompt tokens per sample: {', '.join(restores)}",
            flush=True,
        )


@unittest.skipUnless(
    os.environ.get("SGLANG_LINKER_MLA_RUN_AIME"),
    "opt-in: scores the shared server with sgl-eval",
)
class TestPlainMLALinkerAimeAccuracy(_MLALinkerCase):
    """Task-level companion: an AIME score must not move either.

    Kept opt-in because sgl-eval is a heavier dependency and, with unique short
    problems, it restores ~8% of each prompt — a coarse signal next to the long
    prompt test above.
    """

    def test_store_restore_preserves_task_accuracy(self):
        from types import SimpleNamespace

        from sglang.test.sgl_eval_utils import run_sgl_eval

        out_dir = os.path.join(os.environ.get("TMPDIR", "/tmp"), "linker_mla_aime")

        def score():
            return run_sgl_eval(
                SimpleNamespace(
                    eval_name="aime25",
                    base_url=f"{self.base_url}/v1",
                    model=self.model,
                    num_examples=12,
                    num_threads=1,
                    repeat=1,
                    max_tokens=8192,
                    temperature=0.0,
                    sgl_eval_out_dir=out_dir,
                )
            )

        cold = score()
        _flush_cache(self.base_url)
        restored = score()

        self.assertEqual(restored["score"], cold["score"])
        self.assertEqual(restored["truncated_rate"], cold["truncated_rate"])


if __name__ == "__main__":
    unittest.main()
