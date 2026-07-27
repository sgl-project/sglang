import json
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=600, suite="full-1-npu-a3", nightly=True)


class TestStreamResponseIncludeUsage(CustomTestCase):
    """Test --stream-response-default-include-usage parameter: forces
    usage (token consumption statistics) into every streaming response
    regardless of whether the client requested it.

    Business scenario: Multi-tenant SaaS platform — different client
    SDK versions may or may not request usage.  The server-side flag
    ensures every stream ends with a usage chunk for billing, rate
    limiting, or cost analytics.

    This parameter ONLY affects the /v1/chat/completions endpoint for
    streaming requests.  /v1/completions is not covered here (same
    ``should_include_usage`` function but different output assembly).


    [Test Category] Parameter
    [Test Target] --stream-response-default-include-usage
    """

    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST
    NPU_FIXTURE_ARGS = [
        "--attention-backend",
        "ascend",
    ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @classmethod
    def _launch_server(cls, extra_args):
        all_args = cls.NPU_FIXTURE_ARGS + extra_args
        return popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=all_args,
        )

    @staticmethod
    def _stream_chat(send_stream_options: bool, client_include_usage: bool = False):
        """Send a streaming /v1/chat/completions request.

        Returns (chunks, has_usage_chunk).

        ``send_stream_options`` controls whether the request body
        includes a ``stream_options`` key.  When True,
        ``client_include_usage`` selects the value of
        ``stream_options.include_usage``.
        """
        body = {
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
            "max_tokens": 32,
        }
        if send_stream_options:
            body["stream_options"] = {"include_usage": client_include_usage}

        resp = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/v1/chat/completions",
            json=body,
        )
        resp.raise_for_status()

        chunks = []
        has_usage = False
        for line in resp.iter_lines():
            if not line:
                continue
            line_str = line.decode("utf-8") if isinstance(line, bytes) else line
            if line_str.startswith("data:") and not line_str.startswith("data: [DONE]"):
                chunk = json.loads(line_str[6:])
                chunks.append(chunk)
                if chunk.get("usage") is not None:
                    has_usage = True
        return chunks, has_usage

    def _assert_usage_chunk_valid(self, chunks):
        """Assert usage chunk is valid: only in last chunk, contains
        non-zero token counts, and last chunk has empty choices.
        Also validates that preceding chunks contain normal content."""
        self.assertGreater(
            len(chunks), 1, "Expected at least 2 chunks: content + usage"
        )
        # Non-final chunks must be normal content frames (no usage, has delta)
        has_content_text = False
        for chunk in chunks[:-1]:
            self.assertIsNone(
                chunk.get("usage"),
                "Usage should only appear in the final chunk",
            )
            choices = chunk.get("choices")
            self.assertIsNotNone(choices, "Non-last chunk must have choices")
            self.assertEqual(
                len(choices), 1, "Expected exactly one choice per content chunk"
            )
            delta = choices[0].get("delta")
            self.assertIsNotNone(delta, "Content chunk must have delta")
            if delta.get("content"):
                has_content_text = True

        self.assertTrue(
            has_content_text, "Expected at least one chunk with non-empty content"
        )

        last_chunk = chunks[-1]
        usage = last_chunk.get("usage")
        self.assertIsNotNone(usage, "Last chunk must contain usage")
        self.assertIn("completion_tokens", usage)
        self.assertIn("prompt_tokens", usage)
        self.assertIn("total_tokens", usage)
        self.assertGreater(
            usage["prompt_tokens"], 0, "Usage prompt_tokens should be > 0"
        )
        self.assertGreater(
            usage["completion_tokens"], 0, "Usage completion_tokens should be > 0"
        )
        self.assertGreater(usage["total_tokens"], 0, "Usage total_tokens should be > 0")
        self.assertEqual(
            usage["prompt_tokens"] + usage["completion_tokens"],
            usage["total_tokens"],
            "total_tokens must equal prompt_tokens + completion_tokens",
        )
        self.assertEqual(
            last_chunk.get("choices"), [], "Final usage chunk must have empty choices"
        )

    def _assert_content_chunks_valid(self, chunks):
        """Assert all chunks are normal content frames with no usage."""
        self.assertGreater(len(chunks), 0, "Expected at least one content chunk")
        has_content_text = False
        for chunk in chunks:
            self.assertIsNone(
                chunk.get("usage"),
                "Expected no usage in content-only response",
            )
            choices = chunk.get("choices")
            self.assertIsNotNone(choices, "Content chunk must have choices")
            self.assertEqual(len(choices), 1, "Expected exactly one choice per chunk")
            delta = choices[0].get("delta")
            self.assertIsNotNone(delta, "Content chunk must have delta")
            if delta.get("content"):
                has_content_text = True

        self.assertTrue(
            has_content_text, "Expected at least one chunk with non-empty content"
        )

    # ==================================================================
    # server: --stream-response-default-include-usage (True)
    # ==================================================================

    def test_server_default_true(self):
        """Server with --stream-response-default-include-usage.
        Covers TC-IU-01 and TC-IU-02."""
        process = self._launch_server(["--stream-response-default-include-usage"])
        try:
            health = requests.get(f"{self.base_url}/health_generate")
            self.assertEqual(health.status_code, 200)

            # TC-IU-01: no stream_options → usage forced by server default
            #   Branch: utils.py:88-91 B1
            chunks, has_usage = self._stream_chat(
                send_stream_options=False,
            )
            self.assertGreater(len(chunks), 0, "Expected at least one data chunk")
            self.assertTrue(
                has_usage,
                "Expected usage chunk when server default is True "
                "and stream_options is absent",
            )
            self._assert_usage_chunk_valid(chunks)

            # TC-IU-02: client include_usage=False → server overrides
            #   Branch: utils.py:84-85 A2 (False or True == True)
            chunks, has_usage = self._stream_chat(
                send_stream_options=True,
                client_include_usage=False,
            )
            self.assertGreater(len(chunks), 0, "Expected at least one data chunk")
            self.assertTrue(
                has_usage,
                "Server default=True must override " "client include_usage=False",
            )
            self._assert_usage_chunk_valid(chunks)
        finally:
            kill_process_tree(process.pid)

    # ==================================================================
    # server: no --stream-response-default-include-usage (default False)
    # ==================================================================

    def test_server_default_false(self):
        """Server without --stream-response-default-include-usage.
        Covers TC-IU-03, TC-IU-04, and TC-IU-05."""
        process = self._launch_server([])
        try:
            health = requests.get(f"{self.base_url}/health_generate")
            self.assertEqual(health.status_code, 200)

            # TC-IU-03: client include_usage=True → opted in
            #   Branch: utils.py:84-85 A1 (True or False == True)
            chunks, has_usage = self._stream_chat(
                send_stream_options=True,
                client_include_usage=True,
            )
            self.assertGreater(len(chunks), 0, "Expected at least one data chunk")
            self.assertTrue(has_usage, "Expected usage chunk when client opts in")
            self._assert_usage_chunk_valid(chunks)

            # TC-IU-04: client include_usage=False → no usage
            #   Branch: utils.py:84-85 A3 (False or False == False)
            chunks, has_usage = self._stream_chat(
                send_stream_options=True,
                client_include_usage=False,
            )
            self.assertFalse(
                has_usage,
                "Expected NO usage chunk when both " "server and client disable it",
            )
            self._assert_content_chunks_valid(chunks)

            # TC-IU-05: no stream_options at all → default, no usage
            #   Branch: utils.py:88-91 B2
            chunks, has_usage = self._stream_chat(
                send_stream_options=False,
            )
            self.assertFalse(has_usage, "Expected NO usage chunk in default mode")
            self._assert_content_chunks_valid(chunks)
        finally:
            kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
