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


class TestIncrementalStreamingOutput(CustomTestCase):
    """Test --incremental-streaming-output parameter: controls whether
    streaming chunks carry incremental deltas or full snapshots.

    Business scenarios:
      B1 (enabled): Frontend typewriter effect — each chunk is the new
                    content since the last push.  Client appends directly.
      B2 (disabled, default): Log relay / audit proxy — each chunk is
                    the full accumulated output so far.

    [Test Category] Parameter
    [Test Target] --incremental-streaming-output
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
        return popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.NPU_FIXTURE_ARGS + extra_args,
        )

    @staticmethod
    def _streaming_generate_chunks(prompt="Hello", max_tokens=32):
        """Return list of parsed JSON dicts from a streaming /generate request."""
        resp = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": prompt,
                "sampling_params": {"temperature": 0, "max_new_tokens": max_tokens},
                "stream": True,
            },
        )
        resp.raise_for_status()
        chunks = []
        for line in resp.iter_lines():
            if not line:
                continue
            line_str = line.decode("utf-8") if isinstance(line, bytes) else line
            if line_str.startswith("data:"):
                payload = line_str[5:].lstrip()
                if payload == "[DONE]":
                    continue
                chunks.append(json.loads(payload))
        return chunks

    # ------------------------------------------------------------------
    # TC-ISO-01: incremental=True, streaming
    #   Branches: tokenizer_manager.py:1947T, 1452, serving_chat.py:357T
    #   Each chunk: delta text, delta output_ids, sliced meta_info
    # ------------------------------------------------------------------

    def test_incremental_streaming_delta_chunks(self):
        """incremental=True + stream → each SSE chunk carries only the
        tokens generated since the previous chunk (delta semantics)."""
        process = self._launch_server(["--incremental-streaming-output"])
        try:
            health = requests.get(f"{self.base_url}/health_generate")
            self.assertEqual(health.status_code, 200)

            # Rich prompt to produce more/chunkier output for validation.
            prompt = "Count from 1 to 10, each on a new line."
            max_tokens = 64

            chunks = self._streaming_generate_chunks(prompt, max_tokens=max_tokens)
            self.assertGreater(len(chunks), 1, "Expected multiple streaming chunks")

            # Extract delta content and output_ids from each chunk
            content_parts = []
            token_id_parts = []
            for chunk in chunks:
                text = chunk.get("text", "")
                ids = chunk.get("output_ids")
                if text:
                    content_parts.append(text)
                if ids:
                    token_id_parts.append(ids)

            self.assertGreater(
                len(content_parts), 0, "Expected at least one content delta"
            )

            # 1. Text delta validation: each part is short, not full text
            full_text = "".join(content_parts)
            self.assertGreater(len(full_text), 0, "Full text must be non-empty")
            for i, part in enumerate(content_parts[:-1]):
                self.assertLess(
                    len(part),
                    len(full_text),
                    f"Chunk {i}: expected delta snippet, got: {part!r}",
                )

            # 2. output_ids delta validation: each chunk carries new tokens
            self.assertGreater(
                len(token_id_parts), 0, "Expected output_ids in streaming chunks"
            )
            all_token_ids = []
            for ids in token_id_parts:
                self.assertGreater(
                    len(ids), 0, f"Each delta chunk must have at least 1 output token"
                )
                all_token_ids.extend(ids)

            # Sum of delta token counts ≤ max_tokens (no overflow)
            self.assertLessEqual(
                len(all_token_ids),
                max_tokens,
                f"Total output_ids ({len(all_token_ids)}) exceeds max_tokens={max_tokens}",
            )

            # Each individual chunk's token count is short relative to total
            total_count = len(all_token_ids)
            for i, ids in enumerate(token_id_parts[:-1]):
                self.assertLess(
                    len(ids),
                    total_count,
                    f"Chunk {i} output_ids ({len(ids)}) == total ({total_count}); "
                    f"expected delta, not full sequence",
                )
        finally:
            kill_process_tree(process.pid)

    # ------------------------------------------------------------------
    # TC-ISO-02: incremental=False (default), streaming
    #   Each chunk: full accumulated text (intermediate: text=None → resolved)
    # ------------------------------------------------------------------

    def test_default_streaming_full_snapshot(self):
        """Default (non-incremental) + stream → intermediate chunks carry
        full accumulated text, resolved from deferred None via
        state.get_text()."""
        process = self._launch_server([])  # default — no incremental flag
        try:
            health = requests.get(f"{self.base_url}/health_generate")
            self.assertEqual(health.status_code, 200)

            # Use /generate endpoint for output_ids visibility.
            prompt = "Count from 1 to 10, each on a new line."
            max_tokens = 64

            chunks = self._streaming_generate_chunks(prompt, max_tokens=max_tokens)
            self.assertGreater(len(chunks), 1, "Expected multiple streaming chunks")

            # Extract content and output_ids from each chunk
            content_parts = []
            token_id_parts = []
            for chunk in chunks:
                text = chunk.get("text", "")
                ids = chunk.get("output_ids")
                if text:
                    content_parts.append(text)
                if ids:
                    token_id_parts.append(ids)

            self.assertGreater(
                len(content_parts), 1, "Expected at least 2 content chunks"
            )
            self.assertGreater(
                len(token_id_parts), 1, "Expected at least 2 token_id chunks"
            )

            # 1. Full-snapshot text validation: each chunk's content should
            #    be a prefix of the next (non-decreasing accumulation).
            for i in range(len(content_parts) - 1):
                self.assertTrue(
                    content_parts[i + 1].startswith(content_parts[i]),
                    f"Chunk {i + 1} must start with chunk {i} content "
                    f"(full-snapshot semantics). "
                    f"Chunk {i}: {content_parts[i]!r}, "
                    f"Chunk {i + 1}: {content_parts[i + 1]!r}",
                )

            # 2. output_ids full-snapshot: each chunk's output_ids list
            #    should be a prefix of the next (accumulating token sequence).
            for i in range(len(token_id_parts) - 1):
                prev = token_id_parts[i]
                curr = token_id_parts[i + 1]
                self.assertEqual(
                    curr[: len(prev)],
                    prev,
                    f"Chunk {i + 1} output_ids must start with chunk {i} ids "
                    f"(full-snapshot). len(prev)={len(prev)}, len(curr)={len(curr)}",
                )

            # 3. Total token count must not exceed max_tokens
            if token_id_parts:
                final_ids = token_id_parts[-1]
                self.assertLessEqual(
                    len(final_ids),
                    max_tokens,
                    f"Final output_ids ({len(final_ids)}) exceeds max_tokens={max_tokens}",
                )
        finally:
            kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
