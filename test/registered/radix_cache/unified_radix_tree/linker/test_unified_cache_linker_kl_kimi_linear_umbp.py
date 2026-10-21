"""KL for the UMBP direct linker on Kimi Linear (MLA + KDA, as Kimi K3) at TP4:
every rank restores the one shared MLA KV copy and its own KDA state."""

import json
import os
import subprocess
import tempfile
import time
import unittest

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.kits.unified_radix_cache_kit import _random_suffixes
from sglang.test.kl_multiturn_utils import (
    _extract_output_logprobs,
    _flush_cache,
    _generate,
    _replay_and_compare_kl,
)
from sglang.test.kl_test_utils import get_input_ids
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    find_available_port,
    popen_launch_server,
    terminate_and_kill_process_tree,
    unified_radix_tree_server_env,
)

register_amd_ci(est_time=600, suite="nightly-amd-8-gpu-mi35x", nightly=True)

_MODEL_PATH = os.environ.get(
    "KIMI_LINEAR_MODEL_PATH", "moonshotai/Kimi-Linear-48B-A3B-Instruct"
)

# Not bit-exact on ROCm: on MI355X a correct restore measured 0.9e-3 to 1.8e-3,
# one that left the KDA state zeroed about 5e-2.
KL_DIV_THRESHOLD = 1e-2
PAGE_SIZE = 64
TRACK_INTERVAL = 512
# Short, so turn 2 is scored right after the restore boundary; further out the
# MLA layers mask a wrong KDA state.
TAIL_TOKENS = 16


@unittest.skipUnless(is_hip(), "The UMBP standalone server needs ROCm.")
class TestKimiLinearUMBPLinkerKLTP4(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        # Importing the package publishes UMBP_STANDALONE_BIN.
        import mori.umbp  # noqa: F401

        cls.model = _MODEL_PATH
        cls.base_url = f"http://127.0.0.1:{find_available_port(30000)}"
        # Keep the UDS path short: AF_UNIX paths are limited to 108 bytes.
        socket_dir = tempfile.TemporaryDirectory(prefix="umbp-sa-", dir="/tmp")
        cls.addClassCleanup(socket_dir.cleanup)
        socket_path = os.path.join(socket_dir.name, "sa.sock")
        cls.umbp = subprocess.Popen(
            [os.environ["UMBP_STANDALONE_BIN"], f"unix://{socket_path}"],
            env={
                **os.environ,
                "UMBP_DRAM_CAPACITY": str(16 << 30),
                "UMBP_SSD_ENABLED": "0",
            },
        )
        cls.addClassCleanup(terminate_and_kill_process_tree, cls.umbp, wait_timeout=30)
        deadline = time.monotonic() + 300
        while not os.path.exists(socket_path):
            if cls.umbp.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("The UMBP standalone server did not start.")
            time.sleep(1)

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp-size",
                "4",
                "--attention-backend",
                "aiter",
                "--page-size",
                str(PAGE_SIZE),
                "--mamba-radix-cache-strategy",
                "extra_buffer",
                "--mamba-track-interval",
                str(TRACK_INTERVAL),
                "--mem-fraction-static",
                "0.7",
                "--max-running-requests",
                "1",
                "--enable-cache-report",
                "--enable-unified-cache-external-linker",
                "--unified-cache-external-linker-backend",
                "mori",
                "--hicache-storage-backend-extra-config",
                json.dumps({"standalone_address": f"unix://{socket_path}"}),
            ],
            env=unified_radix_tree_server_env("python", SGLANG_USE_AITER="1"),
        )
        cls.addClassCleanup(
            terminate_and_kill_process_tree, cls.process, wait_timeout=60
        )
        cls.input_ids = get_input_ids(
            tokenizer_path=cls.model, num_samples=4, trust_remote_code=True
        )

    def test_load_back_after_flush(self):
        """Every second-turn prefix comes back from UMBP alone."""
        first_turn = [list(ids[:1024]) for ids in self.input_ids]
        histories = [
            ids + _generate(self.base_url, [ids], TAIL_TOKENS)[0]["output_ids"]
            for ids in first_turn
        ]

        _flush_cache(self.base_url)
        sequences, logprobs = [], []
        for history, suffix in zip(
            histories, _random_suffixes(len(histories), TAIL_TOKENS, seed=100)
        ):
            ids = history + suffix
            (result,) = _generate(self.base_url, [ids], 512, return_logprob=True)
            expected = (len(history) - 1) // TRACK_INTERVAL * TRACK_INTERVAL
            cached_tokens = result["meta_info"]["cached_tokens"]
            details = result["meta_info"].get("cached_tokens_details") or {}
            self.assertGreaterEqual(cached_tokens, expected)
            self.assertEqual(int(details.get("host", 0)), cached_tokens)
            sequences.append(ids + result["output_ids"])
            logprobs.append(_extract_output_logprobs(result))

        _replay_and_compare_kl(
            self.base_url,
            self.model,
            KL_DIV_THRESHOLD,
            sequences,
            logprobs,
            label="umbp_linker_load_back",
            sampling_temperature=0,
        )


if __name__ == "__main__":
    unittest.main()
