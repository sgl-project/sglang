"""DeepSeek-V4.1-Flash external-linker round trip through Mooncake on GB300."""

import json
import time
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.mooncake_utils import MooncakeTestServices
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    try_cached_model,
)

register_cuda_ci(est_time=1800, stage="extra-b", runner_config="4-gpu-gb300")

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
PAGE_SIZE = 256
SECRET = "purple giraffe"
# Long enough for several full pages ahead of the question.
FILLER = (
    "The river bends east past the old mill, where the willows lean over the "
    "water and the road climbs toward the quiet village on the hill. "
) * 120
PROMPT = (
    f"Remember this: the secret code word is '{SECRET}'.\n\n{FILLER}\n"
    "Question: What is the secret code word?\nAnswer: The secret code word is '"
)


class DSV41FlashLinkerRoundTripMixin:
    """The first request offloads its full pages to the store; after a device
    flush the second request must restore them and still recall the secret."""

    extra_args = []

    @classmethod
    def setUpClass(cls):
        cls.model = try_cached_model(MODEL)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.mooncake = MooncakeTestServices()
        cls.mooncake.start()
        cls.addClassCleanup(cls.mooncake.stop)
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=3600,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "4",
                "--ep-size",
                "4",
                "--speculative-algorithm",
                "DSPARK",
                "--speculative-dspark-block-size",
                "5",
                "--page-size",
                str(PAGE_SIZE),
                "--chunked-prefill-size",
                "4096",
                "--mem-fraction-static",
                "0.8",
                "--max-running-requests",
                "4",
                "--cuda-graph-max-bs-decode",
                "4",
                "--enable-cache-report",
                "--enable-unified-cache-external-linker",
                "--hicache-storage-backend-extra-config",
                json.dumps({"enable_group_semantics": True}),
                *cls.extra_args,
            ],
            env=cls.mooncake.server_env(),
        )
        cls.addClassCleanup(kill_process_tree, cls.process.pid)

    def _generate(self):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": PROMPT,
                "sampling_params": {"temperature": 0, "max_new_tokens": 16},
            },
            timeout=600,
        )
        response.raise_for_status()
        return response.json()

    def test_restore_after_device_flush(self):
        first = self._generate()
        self.assertEqual(first["meta_info"]["cached_tokens"], 0)
        self.assertIn(SECRET, first["text"])

        # Offloads land below the request path; let them finish before flushing.
        time.sleep(15)
        requests.post(self.base_url + "/flush_cache", timeout=90).raise_for_status()

        second = self._generate()
        meta = second["meta_info"]
        self.assertGreaterEqual(
            meta["cached_tokens"], meta["prompt_tokens"] - PAGE_SIZE, meta
        )
        self.assertGreater(int(meta["cached_tokens_details"]["host"]), 0, meta)
        self.assertIn(SECRET, second["text"])


class TestDSV41FlashLinkerPagedSWA(DSV41FlashLinkerRoundTripMixin, CustomTestCase):
    """Target and packed DSpark draft SWA pages ride along with the shared KV."""


class TestDSV41FlashLinkerEncoderReplay(DSV41FlashLinkerRoundTripMixin, CustomTestCase):
    """The per-request window is rebuilt on restore instead of stored."""

    extra_args = [
        "--enable-encoder-swa-bounded-replay",
        "--cuda-graph-backend-prefill",
        "disabled",
    ]


if __name__ == "__main__":
    unittest.main()
