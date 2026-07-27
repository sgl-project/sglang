"""
Streaming session tests for NPU.

Tests:
  - KV cache inheritance
  - Concurrent logprob leak detection
  - Abort recovery
  - Long session stability
  - EAGLE3 speculative decoding
"""

import os
import time
import unittest

import requests

from sglang.srt.environ import envs
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_8B_EAGLE3_WEIGHTS_PATH,
    QWEN3_8B_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.kits.streaming_session_kit import StreamingSessionKitMixin
from sglang.test.server_fixtures.streaming_session_fixture import (
    StreamingSessionServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)


class NPUStreamingSessionServerBase(StreamingSessionServerBase):
    npu_env = {
        **os.environ,
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "HCCL_EXEC_TIMEOUT": "200",
    }

    @classmethod
    def setUpClass(cls):
        import contextlib

        with contextlib.ExitStack() as stack:
            stack.enter_context(
                envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(1)
            )
            stack.enter_context(envs.SGLANG_CHECK_KV_PAGE_INVARIANTS.override(True))
            for name, val in cls.env_overrides:
                stack.enter_context(getattr(envs, name).override(val))
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=["--enable-streaming-session"] + list(cls.extra_args),
                env=cls.npu_env,
            )
        cls.tokenizer = get_tokenizer(cls.model)


class TestNPUStreamingSession(NPUStreamingSessionServerBase, StreamingSessionKitMixin):
    model = QWEN3_8B_WEIGHTS_PATH
    extra_args = [
        "--trust-remote-code",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
        "--enable-streaming-session",
        "--mem-fraction-static",
        "0.7",
        "--page-size",
        "4",
    ]
    kv_inherit_offsets = (0,)

    def test_first_mid_abort_recovery(self) -> None:
        """NPU override: Qwen3-8B context ~40K, max_new_tokens reduced to 30000."""
        requests.post(self.base_url + "/flush_cache")

        resp = requests.post(
            self.base_url + "/open_session",
            json={"capacity_of_str_len": 50000, "streaming": True},
        )
        self.assertEqual(resp.status_code, 200)
        session_id = resp.json()

        try:
            ids_1 = self.tokenizer.encode("Tell me a very long story about a wizard.")

            import threading

            result = [None]

            def do_generate():
                r = requests.post(
                    self.base_url + "/generate",
                    json={
                        "input_ids": ids_1,
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 30000,
                        },
                        "session_params": {"id": session_id, "rid": None},
                    },
                    timeout=60,
                )
                result[0] = r

            t = threading.Thread(target=do_generate)
            t.start()
            time.sleep(0.5)
            abort_resp = requests.post(
                self.base_url + "/abort_request",
                json={"rid": "", "abort_all": True},
                timeout=10,
            )
            self.assertEqual(abort_resp.status_code, 200, abort_resp.text)
            t.join(timeout=30)

            self.assertIsNotNone(result[0], "Turn 1 should have returned")
            data_1 = result[0].json()
            self.assertEqual(
                data_1["meta_info"]["finish_reason"]["type"],
                "abort",
                "Turn 1 should be aborted, not finished normally",
            )

            ids_2 = self.tokenizer.encode("Tell me a short joke.")
            for attempt in range(20):
                resp_2 = requests.post(
                    self.base_url + "/generate",
                    json={
                        "input_ids": ids_2,
                        "sampling_params": {"temperature": 0, "max_new_tokens": 8},
                        "session_params": {"id": session_id, "rid": None},
                    },
                    timeout=30,
                )
                if resp_2.status_code == 200:
                    break
                time.sleep(0.5)
            self.assertEqual(resp_2.status_code, 200, resp_2.text)
            data_2 = resp_2.json()
            self.assertEqual(
                data_2["meta_info"]["prompt_tokens"],
                len(ids_2),
                "prompt_tokens must equal turn 2 input only (no inherited context)",
            )
        finally:
            requests.post(
                self.base_url + "/close_session",
                json={"session_id": session_id},
                timeout=10,
            )

        health = requests.get(self.base_url + "/health", timeout=10)
        self.assertEqual(health.status_code, 200)


class TestNPUStreamingSessionLargePage(TestNPUStreamingSession):
    extra_args = [
        "--trust-remote-code",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
        "--enable-streaming-session",
        "--mem-fraction-static",
        "0.7",
        "--page-size",
        "256",
    ]


class TestNPUStreamingSessionEagle3(TestNPUStreamingSession):
    model = QWEN3_8B_WEIGHTS_PATH
    extra_args = [
        "--trust-remote-code",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
        "--enable-streaming-session",
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
        "--mem-fraction-static",
        "0.7",
        "--page-size",
        "4",
    ]
    kv_inherit_offsets = (-1,)


class TestNPUStreamingSessionEagle3LargePage(TestNPUStreamingSession):
    model = QWEN3_8B_WEIGHTS_PATH
    extra_args = [
        "--trust-remote-code",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
        "--enable-streaming-session",
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
        "--mem-fraction-static",
        "0.7",
        "--page-size",
        "256",
    ]
    kv_inherit_offsets = (-1,)


class TestNPUStreamingSessionRetract(TestNPUStreamingSession):
    extra_args = [
        "--trust-remote-code",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
        "--enable-streaming-session",
        "--mem-fraction-static",
        "0.7",
        "--page-size",
        "4",
    ]
    env_overrides = [("SGLANG_TEST_RETRACT", True)]


class TestNPUStreamingSessionEagle3Retract(TestNPUStreamingSession):
    model = QWEN3_8B_WEIGHTS_PATH
    extra_args = [
        "--trust-remote-code",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
        "--enable-streaming-session",
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
        "--mem-fraction-static",
        "0.7",
        "--page-size",
        "4",
    ]
    env_overrides = [("SGLANG_TEST_RETRACT", True)]
    kv_inherit_offsets = (-1,)


class TestNPUStreamingSessionEagle3RetractLargePage(TestNPUStreamingSession):
    model = QWEN3_8B_WEIGHTS_PATH
    extra_args = [
        "--trust-remote-code",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
        "--enable-streaming-session",
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
        "--mem-fraction-static",
        "0.7",
        "--page-size",
        "256",
    ]
    env_overrides = [("SGLANG_TEST_RETRACT", True)]
    kv_inherit_offsets = (-1,)


__all__ = [
    "TestNPUStreamingSession",
    "TestNPUStreamingSessionLargePage",
    "TestNPUStreamingSessionEagle3",
    "TestNPUStreamingSessionEagle3LargePage",
    "TestNPUStreamingSessionRetract",
    "TestNPUStreamingSessionEagle3Retract",
    "TestNPUStreamingSessionEagle3RetractLargePage",
]


if __name__ == "__main__":
    unittest.main()
