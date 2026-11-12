"""Offline Engine generation through the Rust frontend and Python scheduler."""

import asyncio
import os
import unittest
from unittest.mock import patch

from sglang.srt.entrypoints.engine import Engine
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_rust_server_built,
)

register_cuda_ci(est_time=300, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(is_rust_server_built(), "embedded rust server extension not built")
class TestRustOfflineEngine(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls._rust_server_env = patch.dict(os.environ, {"SGLANG_RUST_SERVER": "1"})
        cls._rust_server_env.start()
        host, port = DEFAULT_URL_FOR_TEST.removeprefix("http://").rsplit(":", 1)
        cls.engine = Engine(
            model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            host=host,
            port=int(port),
        )

    @classmethod
    def tearDownClass(cls):
        try:
            engine = getattr(cls, "engine", None)
            if engine is not None:
                engine.shutdown()
        finally:
            cls._rust_server_env.stop()

    def test_unary_batch_async_and_streaming_generation(self):
        unary = self.engine.generate(
            prompt="The capital of France is",
            sampling_params={"temperature": 0.0, "max_new_tokens": 3},
        )
        self.assertTrue(unary["output_ids"])
        self.assertIn("meta_info", unary)

        batch = self.engine.generate(
            prompt=["The capital of France is", "The capital of Italy is"],
            sampling_params={"temperature": 0.0, "max_new_tokens": 2},
        )
        self.assertEqual(len(batch), 2)
        self.assertTrue(all(item["output_ids"] for item in batch))

        async_result = asyncio.run(
            self.engine.async_generate(
                prompt="The capital of Spain is",
                sampling_params={"temperature": 0.0, "max_new_tokens": 2},
            )
        )
        self.assertTrue(async_result["output_ids"])

        chunks = list(
            self.engine.generate(
                prompt="Rust offline streaming",
                sampling_params={"temperature": 0.0, "max_new_tokens": 3},
                stream=True,
            )
        )
        self.assertGreaterEqual(len(chunks), 1)
        self.assertTrue(chunks[-1]["meta_info"].get("finish_reason"))


if __name__ == "__main__":
    unittest.main()
