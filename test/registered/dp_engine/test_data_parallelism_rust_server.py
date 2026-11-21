"""Weight updates sent at once to both DP listeners of a dp=2 Rust server
(SGLANG_RUST_SERVER=1) must both finish, and the server must still serve
generations afterwards. Each listener used to hold its own update lock while
waiting for the other's, so neither update returned and admission stayed
closed (sgl-project/sglang#51333)."""

import unittest
from concurrent.futures import ThreadPoolExecutor

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    ADMIN_LAUNCH_ARGS_FOR_TEST,
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    admin_post,
    is_rust_server_built,
    popen_launch_server,
)

register_cuda_ci(est_time=120, stage="base-b", runner_config="2-gpu-large")

DP_SIZE = 2

# Arbitrary and generous: reloading the test model takes seconds, while a
# deadlocked update never returns.
UPDATE_TIMEOUT_S = 300


@unittest.skipUnless(is_rust_server_built(), "embedded rust server extension not built")
class TestDataParallelismRustServer(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        host, port = cls.base_url.rsplit(":", 1)
        # Plain DP: DP rank k's listener binds --port + k.
        cls.listener_urls = [f"{host}:{int(port) + rank}" for rank in range(DP_SIZE)]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--dp", DP_SIZE, *ADMIN_LAUNCH_ARGS_FOR_TEST],
            env={"SGLANG_RUST_SERVER": "1"},
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_concurrent_updates_on_every_listener_finish(self):
        def update(url):
            return admin_post(
                url + "/update_weights_from_disk",
                json={"model_path": self.model},
                timeout=UPDATE_TIMEOUT_S,
            )

        with ThreadPoolExecutor(max_workers=DP_SIZE) as pool:
            responses = list(pool.map(update, self.listener_urls))
        for url, response in zip(self.listener_urls, responses):
            self.assertEqual(response.status_code, 200, f"{url}: {response.text}")
            body = response.json()
            self.assertTrue(body["success"], f"{url}: {body}")
            self.assertEqual(
                len(body["num_paused_requests"]),
                DP_SIZE,
                f"{url}: one entry per DP rank: {body}",
            )

        for url in self.listener_urls:
            response = requests.post(
                url + "/generate",
                json={
                    "text": "The capital of France is",
                    "sampling_params": {"temperature": 0, "max_new_tokens": 8},
                },
                timeout=UPDATE_TIMEOUT_S,
            )
            self.assertEqual(response.status_code, 200, f"{url}: {response.text}")
            self.assertTrue(response.json()["text"], f"{url}: {response.text}")


if __name__ == "__main__":
    unittest.main()
