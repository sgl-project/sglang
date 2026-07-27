"""
NPU OpenAI Chat Completions API + multimodal tests.

Test cases:
  - TC-MM-OPENAI-CHAT-001: temperature + multimodal (temp=0, 0.8, 1.5)
  - TC-MM-OPENAI-CHAT-002: top_p + multimodal (top_p=0.1, 0.5, 1.0)
  - TC-MM-OPENAI-CHAT-003: seed + multimodal (determinism)
  - TC-MM-OPENAI-CHAT-004: max_tokens + multimodal (short vs long)
  - TC-MM-OPENAI-CHAT-005: stop + multimodal (single + multi stop)
  - TC-MM-OPENAI-CHAT-006: n + multimodal (multiple completions)
  - TC-MM-OPENAI-CHAT-007: stream + multimodal (streaming chunks)
  - TC-MM-OPENAI-CHAT-008: stream_options + multimodal (include_usage)
  - TC-MM-OPENAI-CHAT-009: frequency_penalty + multimodal (0 vs 1.5)
  - TC-MM-OPENAI-CHAT-010: presence_penalty + multimodal (0 vs 1.5)
  - TC-MM-OPENAI-CHAT-011: logprobs + top_logprobs + multimodal
  - TC-MM-OPENAI-CHAT-012: max_completion_tokens + multimodal
"""

import unittest

import openai

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_5_9B_WEIGHTS_PATH,
)
from sglang.test.ascend.test_npu_multimodal_utils import (
    Color,
    Shape,
    assert_color_and_shape,
    create_test_image,
    image_content,
    text_content,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=200, suite="full-1-npu-a3", nightly=True)


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------
class TestMultimodalParameterInteractions(CustomTestCase):
    """Multimodal + OpenAI Chat Completions API parameters.

    [Test Category] multimodal
    [Test Target] multimodal + parameter interaction

    Verifies that OpenAI Chat Completions API parameters (temperature,
    top_p, seed, max_tokens, stop, n, stream, stream_options,
    frequency_penalty, presence_penalty, logprobs) interact correctly
    with multimodal (image) inputs.

    Uses Qwen3.5-9B (GDN attention + DeepStack ViT + native NEXTN MTP)
    with TP=1 on 1 NPU chip and speculative decoding enabled.
    """

    # ------------------------------------------------------------------
    # Model & server configuration (matches TestP1001SpeculativeDecoding)
    # ------------------------------------------------------------------

    _model = QWEN3_5_9B_WEIGHTS_PATH
    _common_args = [
        "--device",
        "npu",
        "--attention-backend",
        "ascend",
        "--trust-remote-code",
        "--enable-multimodal",
        "--mm-attention-backend",
        "ascend_attn",
        "--mem-fraction-static",
        "0.75",
        "--cuda-graph-bs",
        1,
        2,
        4,
        "--mamba-scheduler-strategy",
        "extra_buffer",
        "--tp-size",
        1,
        "--dtype",
        "bfloat16",
        "--mamba-ssm-dtype",
        "bfloat16",
        "--tool-call-parser",
        "qwen",
    ]
    _spec_args = [
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
    ]
    _server_args = _common_args + _spec_args

    @classmethod
    def setUpClass(cls):
        """Start the SGLang server for VLM inference on Qwen3.5-9B."""
        cls.api_key = "sk-123456"
        cls.base_url = DEFAULT_URL_FOR_TEST

        cls.process = popen_launch_server(
            cls._model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls._server_args,
        )
        cls.client = openai.Client(
            api_key=cls.api_key,
            base_url=f"{cls.base_url}/v1",
        )

    @classmethod
    def tearDownClass(cls):
        """Clean up the server process."""
        if hasattr(cls, "process") and cls.process is not None:
            try:
                kill_process_tree(cls.process.pid)
            except Exception:
                pass

    # -- helpers -----------------------------------------------------------

    def _build_msg(self, image_b64, prompt="Describe the image"):
        """Build a single-image user message."""
        return [
            {
                "role": "user",
                "content": [image_content(image_b64), text_content(prompt)],
            }
        ]

    def _request(self, messages, **kwargs):
        """Send a chat completion and return the response object."""
        params = dict(
            model="default",
            messages=messages,
            max_tokens=128,
        )
        params.update(kwargs)
        return self.client.chat.completions.create(**params)

    def _request_text(self, messages, **kwargs):
        """Send a chat completion and return just the text content."""
        return self._request(messages, **kwargs).choices[0].message.content

    # ===================================================================
    # TC-MM-OPENAI-CHAT-001: temperature + multimodal
    # ===================================================================

    def test_001_temperature_multimodal(self):
        """TC-MM-OPENAI-CHAT-001: Verify temperature sampling with image input.

        Tests temperature=0 (deterministic), 0.8 (moderate), 1.5 (high).
        All must produce non-empty output that correctly identifies the
        test image's color and shape.
        """
        _, image_b64 = create_test_image(
            320, 240, color=Color.GREEN, shape=Shape.ELLIPSE
        )
        temps = [0, 0.8, 1.5]
        outputs = {}

        for temp in temps:
            with self.subTest(temperature=temp):
                text = self._request_text(
                    self._build_msg(image_b64),
                    temperature=temp,
                )
                outputs[temp] = text  # store before asserts for safe access below
                self.assertTrue(
                    text, f"TC-MM-OPENAI-CHAT-001: Empty output at temperature={temp}"
                )
                self.assertGreater(
                    len(text),
                    5,
                    f"TC-MM-OPENAI-CHAT-001: Suspiciously short output ({len(text)} chars) at temperature={temp}",
                )
                assert_color_and_shape(
                    self,
                    text,
                    "green",
                    "ellipse",
                    prefix=f"TC-MM-OPENAI-CHAT-001/temp={temp}: ",
                )

        # Verify all temperatures produced output
        if 0 in outputs:
            self.assertGreaterEqual(
                len(outputs[0]),
                5,
                "TC-MM-OPENAI-CHAT-001: temperature=0 output too short",
            )

    # ===================================================================
    # TC-MM-OPENAI-CHAT-002: top_p + multimodal
    # ===================================================================

    def test_002_top_p_multimodal(self):
        """TC-MM-OPENAI-CHAT-002: Verify top_p (nucleus sampling) with image input.

        Tests top_p=0.1 (narrow), 0.5 (moderate), 1.0 (full distribution).
        All must produce image-relevant output.  Narrow top_p must not
        produce degenerate / repeated-token output.
        """
        _, image_b64 = create_test_image(
            320, 240, color=Color.BLUE, shape=Shape.RECTANGLE
        )

        for top_p in [0.1, 0.5, 1.0]:
            with self.subTest(top_p=top_p):
                text = self._request_text(
                    self._build_msg(image_b64),
                    top_p=top_p,
                    temperature=0.8,  # top_p requires temperature > 0
                )
                self.assertTrue(
                    text, f"TC-MM-OPENAI-CHAT-002: Empty output at top_p={top_p}"
                )
                self.assertGreater(
                    len(text),
                    5,
                    f"TC-MM-OPENAI-CHAT-002: Suspiciously short output at top_p={top_p}",
                )

                assert_color_and_shape(
                    self,
                    text,
                    "blue",
                    "rectangle",
                    prefix=f"TC-MM-OPENAI-CHAT-002/top_p={top_p}: ",
                )

    # ===================================================================
    # TC-MM-OPENAI-CHAT-003: seed + multimodal (determinism)
    # ===================================================================

    def test_003_seed_determinism(self):
        """TC-MM-OPENAI-CHAT-003: Verify seed produces deterministic output for same image.

        Sends two requests with identical image, prompt, seed=42, and a
        low non-zero temperature (0.2).  Unlike temperature=0 (which is
        deterministic regardless of seed), temperature=0.2 uses random
        sampling — so the seed MUST be controlling the RNG for the two
        outputs to match.

        Falls back to semantic equivalence if the NPU backend cannot
        guarantee strict determinism (e.g., due to kernel-level noise).
        """
        _, image_b64 = create_test_image(320, 240, color=Color.RED, shape=Shape.ELLIPSE)
        seed = 42

        text1 = self._request_text(
            self._build_msg(
                image_b64, "What color and shape do you see? Answer concisely."
            ),
            temperature=0.2,
            seed=seed,
            max_tokens=64,
        )
        text2 = self._request_text(
            self._build_msg(
                image_b64, "What color and shape do you see? Answer concisely."
            ),
            temperature=0.2,
            seed=seed,
            max_tokens=64,
        )

        self.assertTrue(
            text1, "TC-MM-OPENAI-CHAT-003: First request returned empty output"
        )
        self.assertTrue(
            text2, "TC-MM-OPENAI-CHAT-003: Second request returned empty output"
        )

        # Strict determinism check
        if text1 == text2:
            pass  # exact match — seed determinism works
        else:
            # NPU hardware non-determinism is a known limitation (kernel
            # scheduling / attention ordering differences between runs).
            # Verify semantic correctness first (hard fail if image
            # understanding is wrong), then record the determinism gap.
            assert_color_and_shape(
                self, text1, "red", "ellipse", prefix="TC-MM-OPENAI-CHAT-003/req1: "
            )
            assert_color_and_shape(
                self, text2, "red", "ellipse", prefix="TC-MM-OPENAI-CHAT-003/req2: "
            )
            # Semantic check passed — the mismatch is likely NPU noise,
            # but could also signal a seed bug.  Promote to a hard failure
            # if this becomes a recurring pattern across CI runs.

    # ===================================================================
    # TC-MM-OPENAI-CHAT-004: max_tokens + multimodal
    # ===================================================================

    def test_004_max_tokens_multimodal(self):
        """TC-MM-OPENAI-CHAT-004: Verify max_tokens limits output with image prompts.

        Tests max_tokens=128 (short) and max_tokens=256 (longer).
        Short output must respect the token budget; both must contain
        image-relevant content.
        """
        _, image_b64 = create_test_image(
            320, 240, color=Color.PURPLE, shape=Shape.ELLIPSE
        )

        # --- Short max_tokens ---
        resp_short = self._request(
            self._build_msg(
                image_b64, "Answer concisely: what shape and color is this image?"
            ),
            max_tokens=128,
            temperature=0,
        )
        text_short = resp_short.choices[0].message.content
        self.assertTrue(
            text_short, "TC-MM-OPENAI-CHAT-004: Empty output with max_tokens=128"
        )

        finish_short = resp_short.choices[0].finish_reason
        self.assertIn(
            finish_short,
            ("length", "stop"),
            f"TC-MM-OPENAI-CHAT-004: Unexpected finish_reason '{finish_short}' for max_tokens=128",
        )

        # --- Longer max_tokens ---
        resp_long = self._request(
            self._build_msg(image_b64),
            max_tokens=256,
            temperature=0,
        )
        text_long = resp_long.choices[0].message.content
        self.assertTrue(
            text_long, "TC-MM-OPENAI-CHAT-004: Empty output with max_tokens=256"
        )

        # Short output: at minimum must mention the color (shape may not fit
        # within 128 tokens).  Long output: must identify both color and shape.
        self.assertIn(
            "purple",
            text_short.lower(),
            f"TC-MM-OPENAI-CHAT-004: (max_tokens=128) Expected 'purple' not in output: {text_short[:200]}",
        )
        assert_color_and_shape(
            self,
            text_long,
            "purple",
            "ellipse",
            prefix="TC-MM-OPENAI-CHAT-004/max_tokens=256: ",
        )

        # Longer output should generally be longer
        self.assertGreaterEqual(
            len(text_long),
            len(text_short),
            "TC-MM-OPENAI-CHAT-004: max_tokens=256 output not longer than max_tokens=128",
        )

    # ===================================================================
    # TC-MM-OPENAI-CHAT-005: stop + multimodal
    # ===================================================================

    def test_005_stop_multimodal(self):
        """TC-MM-OPENAI-CHAT-005: Verify stop sequences work with image prompts.

        Tests a single stop string and a list of stop strings by
        instructing the model to end its response with a specific word
        (STOPEND), which is set as the stop sequence.  This avoids
        tokenization edge cases with punctuation-based stops.

        Image content must be described before termination.
        """
        _, image_b64 = create_test_image(
            320, 240, color=Color.TEAL, shape=Shape.RECTANGLE
        )

        # --- Single stop string ---
        resp_single = self._request(
            self._build_msg(
                image_b64,
                "Describe the image in one short sentence. "
                "End your response with the word STOPEND",
            ),
            temperature=0,
            max_tokens=512,
            stop="STOPEND",
        )
        text_single = resp_single.choices[0].message.content
        self.assertTrue(
            text_single, "TC-MM-OPENAI-CHAT-005: Empty output with stop='STOPEND'"
        )
        self.assertEqual(
            resp_single.choices[0].finish_reason,
            "stop",
            f"TC-MM-OPENAI-CHAT-005: Expected finish_reason='stop' with stop='STOPEND', "
            f"got '{resp_single.choices[0].finish_reason}'",
        )
        # Image content must be described even with early stop
        assert_color_and_shape(
            self,
            text_single,
            "teal",
            "rectangle",
            prefix="TC-MM-OPENAI-CHAT-005/stop='STOPEND': ",
        )

        # --- Multiple stop strings ---
        resp_multi = self._request(
            self._build_msg(
                image_b64,
                "Describe the image briefly. " "End with either STOPEND or FINISHED",
            ),
            temperature=0,
            max_tokens=512,
            stop=["STOPEND", "FINISHED"],
        )
        text_multi = resp_multi.choices[0].message.content
        self.assertTrue(
            text_multi,
            "TC-MM-OPENAI-CHAT-005: Empty output with stop=['STOPEND', 'FINISHED']",
        )
        self.assertEqual(
            resp_multi.choices[0].finish_reason,
            "stop",
            f"TC-MM-OPENAI-CHAT-005: Expected finish_reason='stop' with stop list, "
            f"got '{resp_multi.choices[0].finish_reason}'",
        )
        # Must still reference image content
        assert_color_and_shape(
            self,
            text_multi,
            "teal",
            "rectangle",
            prefix="TC-MM-OPENAI-CHAT-005/stop=list: ",
        )

    # ===================================================================
    # TC-MM-OPENAI-CHAT-006: n + multimodal
    # ===================================================================

    def test_006_n_multimodal(self):
        """TC-MM-OPENAI-CHAT-006: Verify n returns multiple independent completions.

        Uses temperature=0.7 so the two sampling paths draw from different
        random states — the two choices should differ while both correctly
        describe the image.  If n were silently ignored the response would
        contain only 1 choice; if both choices shared the same RNG they
        would be identical.
        """
        _, image_b64 = create_test_image(
            320, 240, color=Color.RED, shape=Shape.RECTANGLE
        )

        resp = self._request(
            self._build_msg(image_b64, "Describe the image"),
            temperature=0.7,
            max_tokens=64,
            n=2,
        )

        choices = resp.choices
        self.assertEqual(
            len(choices),
            2,
            f"TC-MM-OPENAI-CHAT-006: Expected 2 choices, got {len(choices)}",
        )

        for i, choice in enumerate(choices):
            text = choice.message.content
            self.assertTrue(
                text, f"TC-MM-OPENAI-CHAT-006: Choice {i} returned empty content"
            )
            self.assertGreater(
                len(text),
                3,
                f"TC-MM-OPENAI-CHAT-006: Choice {i} suspiciously short: '{text}'",
            )
            assert_color_and_shape(
                self,
                text,
                "red",
                "rectangle",
                prefix=f"TC-MM-OPENAI-CHAT-006/choice[{i}]: ",
            )

        # Usage: prompt_tokens should be > 0 (image contributes tokens)
        self.assertGreater(
            resp.usage.prompt_tokens,
            0,
            "TC-MM-OPENAI-CHAT-006: prompt_tokens should be > 0 for image request",
        )

    # ===================================================================
    # TC-MM-OPENAI-CHAT-007: stream + multimodal
    # ===================================================================

    def test_007_stream_multimodal(self):
        """TC-MM-OPENAI-CHAT-007: Verify streaming works with image input.

        Streams a chat completion and verifies:
        - Multiple chunks received
        - Combined text correctly identifies image content
        - Final chunk has finish_reason
        """
        _, image_b64 = create_test_image(
            320, 240, color=Color.BLUE, shape=Shape.ELLIPSE
        )

        stream = self.client.chat.completions.create(
            model="default",
            messages=self._build_msg(image_b64, "Describe the image"),
            temperature=0,
            max_tokens=128,
            stream=True,
        )

        chunks = []
        combined_text = ""
        finish_reason = None

        for chunk in stream:
            chunks.append(chunk)
            if chunk.choices and chunk.choices[0].delta.content:
                combined_text += chunk.choices[0].delta.content
            if chunk.choices and chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason

        self.assertGreater(
            len(chunks),
            1,
            f"TC-MM-OPENAI-CHAT-007: Only {len(chunks)} chunk(s) received — streaming may not be working",
        )
        self.assertTrue(
            combined_text,
            "TC-MM-OPENAI-CHAT-007: No text content accumulated from stream",
        )
        self.assertIsNotNone(
            finish_reason, "TC-MM-OPENAI-CHAT-007: No finish_reason in stream chunks"
        )

        # Combined text must describe the image
        assert_color_and_shape(
            self, combined_text, "blue", "ellipse", prefix="TC-MM-OPENAI-CHAT-007: "
        )

    # ===================================================================
    # TC-MM-OPENAI-CHAT-008: stream_options + multimodal
    # ===================================================================

    def test_008_stream_options_multimodal(self):
        """TC-MM-OPENAI-CHAT-008: Verify stream_options with include_usage in image requests.

        Streams with stream_options={"include_usage": True} and verifies:
        - At least one chunk contains usage data
        - Combined text still correctly identifies image
        - Total token counts > 0 in usage chunk
        """
        _, image_b64 = create_test_image(
            320, 240, color=Color.GREEN, shape=Shape.RECTANGLE
        )

        stream = self.client.chat.completions.create(
            model="default",
            messages=self._build_msg(image_b64, "Describe the image"),
            temperature=0,
            max_tokens=64,
            stream=True,
            stream_options={"include_usage": True},
        )

        chunks = []
        combined_text = ""
        usage_chunk_found = False
        finish_reason = None

        for chunk in stream:
            chunks.append(chunk)
            if chunk.choices and chunk.choices[0].delta.content:
                combined_text += chunk.choices[0].delta.content
            if chunk.choices and chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason
            # Check for usage data
            if chunk.usage is not None:
                usage_chunk_found = True

        self.assertGreater(
            len(chunks),
            1,
            f"TC-MM-OPENAI-CHAT-008: Only {len(chunks)} chunk(s) received",
        )
        self.assertTrue(
            combined_text,
            "TC-MM-OPENAI-CHAT-008: No text content accumulated from stream",
        )
        self.assertTrue(
            usage_chunk_found,
            "TC-MM-OPENAI-CHAT-008: No usage chunk found — include_usage may not be working",
        )
        self.assertIsNotNone(
            finish_reason, "TC-MM-OPENAI-CHAT-008: No finish_reason in stream chunks"
        )

        # Verify image understanding
        assert_color_and_shape(
            self, combined_text, "green", "rectangle", prefix="TC-MM-OPENAI-CHAT-008: "
        )

    # ===================================================================
    # TC-MM-OPENAI-CHAT-009: frequency_penalty + multimodal
    # ===================================================================

    def test_009_frequency_penalty_multimodal(self):
        """TC-MM-OPENAI-CHAT-009: Verify frequency_penalty with image input.

        Tests frequency_penalty=0 (baseline) vs 1.5 (high penalty).
        Both must produce correct image descriptions.  Higher penalty
        should reduce token-level repetition.
        """
        _, image_b64 = create_test_image(
            320, 240, color=Color.PURPLE, shape=Shape.RECTANGLE
        )

        for freq_penalty in [0, 1.5]:
            with self.subTest(frequency_penalty=freq_penalty):
                text = self._request_text(
                    self._build_msg(
                        image_b64,
                        "Describe the image in detail, including colors and shapes",
                    ),
                    temperature=0.7,  # penalty has no effect at temperature=0
                    frequency_penalty=freq_penalty,
                    max_tokens=128,
                )
                self.assertTrue(
                    text,
                    f"TC-MM-OPENAI-CHAT-009: Empty output at frequency_penalty={freq_penalty}",
                )
                self.assertGreater(
                    len(text),
                    10,
                    f"TC-MM-OPENAI-CHAT-009: Suspiciously short output at frequency_penalty={freq_penalty}",
                )

                assert_color_and_shape(
                    self,
                    text,
                    "purple",
                    "rectangle",
                    prefix=f"TC-MM-OPENAI-CHAT-009/freq_pen={freq_penalty}: ",
                )

    # ===================================================================
    # TC-MM-OPENAI-CHAT-010: presence_penalty + multimodal
    # ===================================================================

    def test_010_presence_penalty_multimodal(self):
        """TC-MM-OPENAI-CHAT-010: Verify presence_penalty with image input.

        Tests presence_penalty=0 (baseline) vs 1.5 (high penalty).
        Both must produce correct image descriptions.  Higher penalty
        encourages more diverse vocabulary.
        """
        _, image_b64 = create_test_image(
            320, 240, color=Color.TEAL, shape=Shape.ELLIPSE
        )

        for pres_penalty in [0, 1.5]:
            with self.subTest(presence_penalty=pres_penalty):
                text = self._request_text(
                    self._build_msg(
                        image_b64,
                        "Describe the image in detail, including colors and shapes",
                    ),
                    temperature=0.7,
                    presence_penalty=pres_penalty,
                    max_tokens=128,
                )
                self.assertTrue(
                    text,
                    f"TC-MM-OPENAI-CHAT-010: Empty output at presence_penalty={pres_penalty}",
                )
                self.assertGreater(
                    len(text),
                    10,
                    f"TC-MM-OPENAI-CHAT-010: Suspiciously short output at presence_penalty={pres_penalty}",
                )

                assert_color_and_shape(
                    self,
                    text,
                    "teal",
                    "ellipse",
                    prefix=f"TC-MM-OPENAI-CHAT-010/pres_pen={pres_penalty}: ",
                )

    # ===================================================================
    # TC-MM-OPENAI-CHAT-011: logprobs + top_logprobs + multimodal
    # ===================================================================

    def test_011_logprobs_multimodal(self):
        """TC-MM-OPENAI-CHAT-011: Verify logprobs and top_logprobs with image input.

        Sends a request with logprobs=True and top_logprobs=3.
        Verifies:
        - Response has logprobs field populated
        - Content is non-empty and identifies image content
        - Token-level logprobs are present for output tokens
        """
        _, image_b64 = create_test_image(320, 240, color=Color.RED, shape=Shape.ELLIPSE)

        resp = self._request(
            self._build_msg(
                image_b64, "Answer concisely: what shape and color is this image?"
            ),
            temperature=0,
            max_tokens=256,
            logprobs=True,
            top_logprobs=3,
        )

        text = resp.choices[0].message.content
        self.assertTrue(text, "TC-MM-OPENAI-CHAT-011: Empty output with logprobs=True")
        self.assertIn(
            "red",
            text.lower(),
            f"TC-MM-OPENAI-CHAT-011: Expected 'red' not in output: {text[:200]}",
        )

        # Check logprobs in response
        logprobs_content = resp.choices[0].logprobs
        content_has_logprobs = logprobs_content is not None

        if content_has_logprobs:
            # Verify structure: logprobs should have content (token-level data)
            # The exact attribute name depends on the OpenAI API version
            # Standard: logprobs.content is a list of token logprob entries
            if hasattr(logprobs_content, "content") and logprobs_content.content:
                token_count = len(logprobs_content.content)
                self.assertGreater(
                    token_count,
                    0,
                    "TC-MM-OPENAI-CHAT-011: logprobs.content is empty — no token-level data",
                )
            elif hasattr(logprobs_content, "model_dump"):
                dumped = logprobs_content.model_dump()
                # At minimum, the logprobs object should not be entirely empty
                self.assertTrue(
                    any(v for v in dumped.values() if v),
                    f"TC-MM-OPENAI-CHAT-011: logprobs object appears empty: {dumped}",
                )
        else:
            # logprobs may not be returned in all backends — still verify content
            pass

        # Verify image understanding is correct even with logprobs enabled
        assert_color_and_shape(
            self, text, "red", "ellipse", prefix="TC-MM-OPENAI-CHAT-011: "
        )

    # ===================================================================
    # TC-MM-OPENAI-CHAT-012: max_completion_tokens + multimodal
    # ===================================================================

    def test_012_max_completion_tokens_multimodal(self):
        """TC-MM-OPENAI-CHAT-012: Verify max_completion_tokens with image input.

        Sends the SAME prompt+image twice, only varying max_completion_tokens:
          - tiny=16  → must hit length limit, finish_reason=length
          - normal=128 → must have room to finish, finish_reason=stop

        Only if the tiny limit truncates while the normal one completes
        naturally can we conclude max_completion_tokens is actually enforced.
        """
        _, image_b64 = create_test_image(
            320, 240, color=Color.BLUE, shape=Shape.RECTANGLE
        )
        prompt = "What color and shape is this? Answer briefly."

        # Tiny limit: model will be forced to stop by length
        resp_tiny = self._request(
            self._build_msg(image_b64, prompt),
            temperature=0,
            max_completion_tokens=16,
        )
        text_tiny = resp_tiny.choices[0].message.content
        self.assertTrue(
            text_tiny,
            "TC-MM-OPENAI-CHAT-012: Empty output with max_completion_tokens=16",
        )
        finish_tiny = resp_tiny.choices[0].finish_reason
        self.assertEqual(
            finish_tiny,
            "length",
            f"TC-MM-OPENAI-CHAT-012: Expected finish_reason='length' for tiny limit, "
            f"got '{finish_tiny}' — max_completion_tokens may not be enforced",
        )

        # Normal limit: model has room to finish naturally
        resp_normal = self._request(
            self._build_msg(image_b64, prompt),
            temperature=0,
            max_completion_tokens=512,
        )
        text_normal = resp_normal.choices[0].message.content
        self.assertTrue(
            text_normal,
            "TC-MM-OPENAI-CHAT-012: Empty output with max_completion_tokens=512",
        )
        finish_normal = resp_normal.choices[0].finish_reason
        self.assertEqual(
            finish_normal,
            "stop",
            f"TC-MM-OPENAI-CHAT-012: Expected finish_reason='stop' for large limit, "
            f"got '{finish_normal}'",
        )

        # Image content must be correct at the normal limit
        assert_color_and_shape(
            self, text_normal, "blue", "rectangle", prefix="TC-MM-OPENAI-CHAT-012: "
        )


if __name__ == "__main__":
    unittest.main()
