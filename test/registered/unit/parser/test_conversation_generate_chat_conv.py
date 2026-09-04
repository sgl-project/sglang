"""Unit tests for srt/parser/conversation.py — generate_chat_conv function.

These tests cover the generate_chat_conv() function which converts ChatCompletionRequest
to Conversation objects. Tests focus on edge cases not covered by the main
test_conversation.py suite (which focuses on Conversation.get_prompt()).
"""

import unittest
from unittest.mock import MagicMock, patch

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionMessageContentImageURL,
    ChatCompletionMessageContentImagePart,
    ChatCompletionMessageContentTextPart,
    ChatCompletionMessageContentVideoURL,
    ChatCompletionMessageContentVideoPart,
    ChatCompletionMessageGenericParam,
    ChatCompletionMessageUserParam,
    ChatCompletionRequest,
)
from sglang.srt.parser.conversation import (
    Conversation,
    SeparatorStyle,
    generate_chat_conv,
    get_conv_template_by_model_path,
    register_conv_template,
    chat_templates,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=7, suite="base-a-test-cpu")
register_cpu_ci(est_time=7, suite="base-b-test-cpu")


class TestGenerateChatConvBasic(CustomTestCase):
    """Basic generate_chat_conv tests."""

    def _make_request(self, messages):
        return ChatCompletionRequest(messages=messages, model="test")

    def test_string_messages_raises(self):
        """Test that passing messages as a raw string raises ValueError."""
        request = self._make_request(
            [ChatCompletionMessageUserParam(role="user", content="Hi")]
        )
        # Manually override messages to be a string to trigger validation
        request.__dict__["messages"] = "not a list"
        with self.assertRaises(ValueError):
            generate_chat_conv(request, "chatml")

    def test_multi_turn_conversation(self):
        """Test multi-turn user/assistant conversation."""
        request = self._make_request(
            [
                ChatCompletionMessageUserParam(role="user", content="What is 2+2?"),
                ChatCompletionMessageGenericParam(role="assistant", content="4"),
                ChatCompletionMessageUserParam(role="user", content="And 3+3?"),
            ]
        )
        conv = generate_chat_conv(request, "chatml")
        # 3 explicit messages + 1 blank assistant placeholder
        self.assertEqual(len(conv.messages), 4)
        self.assertEqual(conv.messages[1][1], "4")
        self.assertIsNone(conv.messages[3][1])

    def test_assistant_message_as_list(self):
        """Test assistant message given as a single-element list of text parts."""
        request = self._make_request(
            [
                ChatCompletionMessageUserParam(role="user", content="Hi"),
                ChatCompletionMessageGenericParam(
                    role="assistant",
                    content=[
                        ChatCompletionMessageContentTextPart(type="text", text="Hello!")
                    ],
                ),
                ChatCompletionMessageUserParam(role="user", content="How are you?"),
            ]
        )
        conv = generate_chat_conv(request, "chatml")
        self.assertEqual(conv.messages[1][1], "Hello!")

    def test_assistant_invalid_list_raises(self):
        """Test that assistant message with non-text content raises ValueError."""
        request = self._make_request(
            [
                ChatCompletionMessageUserParam(role="user", content="Hi"),
                ChatCompletionMessageGenericParam(
                    role="assistant",
                    content=[
                        ChatCompletionMessageContentImagePart(
                            type="image_url",
                            image_url=ChatCompletionMessageContentImageURL(
                                url="http://example.com/img.jpg"
                            ),
                        )
                    ],
                ),
            ]
        )
        with self.assertRaises(ValueError):
            generate_chat_conv(request, "chatml")

    def test_user_message_with_image(self):
        """Test user message with image content part."""
        request = self._make_request(
            [
                ChatCompletionMessageUserParam(
                    role="user",
                    content=[
                        ChatCompletionMessageContentTextPart(
                            type="text", text="What's in this image?"
                        ),
                        ChatCompletionMessageContentImagePart(
                            type="image_url",
                            image_url=ChatCompletionMessageContentImageURL(
                                url="http://example.com/cat.jpg"
                            ),
                        ),
                    ],
                )
            ]
        )
        conv = generate_chat_conv(request, "chatml")
        self.assertEqual(len(conv.image_data), 1)
        self.assertEqual(conv.image_data[0].url, "http://example.com/cat.jpg")
        msg = conv.messages[0][1]
        self.assertIn("What's in this image?", msg)

    def test_user_message_with_video(self):
        """Test user message with video content part."""
        request = self._make_request(
            [
                ChatCompletionMessageUserParam(
                    role="user",
                    content=[
                        ChatCompletionMessageContentTextPart(
                            type="text", text="Describe this video"
                        ),
                        ChatCompletionMessageContentVideoPart(
                            type="video_url",
                            video_url=ChatCompletionMessageContentVideoURL(
                                url="http://example.com/vid.mp4"
                            ),
                        ),
                    ],
                )
            ]
        )
        conv = generate_chat_conv(request, "chatml")
        self.assertEqual(len(conv.video_data), 1)
        self.assertEqual(conv.video_data[0], "http://example.com/vid.mp4")

    def test_user_message_with_audio(self):
        """Test user message with audio content part."""
        request = self._make_request(
            [
                ChatCompletionMessageUserParam(
                    role="user",
                    content=[
                        ChatCompletionMessageContentTextPart(
                            type="text", text="Transcribe this"
                        ),
                        {
                            "type": "audio_url",
                            "audio_url": {"url": "http://example.com/audio.wav"},
                        },
                    ],
                )
            ]
        )
        conv = generate_chat_conv(request, "chatml")
        self.assertEqual(len(conv.audio_data), 1)
        self.assertEqual(conv.audio_data[0], "http://example.com/audio.wav")

    def test_user_message_image_at_prefix(self):
        """Test image_token_at_prefix=True puts image token before text."""
        tmp_name = "_test_prefix_img"
        register_conv_template(
            Conversation(
                name=tmp_name,
                roles=("<|im_start|>user", "<|im_start|>assistant"),
                messages=[],
                sep_style=SeparatorStyle.CHATML,
                sep="<|im_end|>",
                image_token_at_prefix=True,
            )
        )
        try:
            request = self._make_request(
                [
                    ChatCompletionMessageUserParam(
                        role="user",
                        content=[
                            ChatCompletionMessageContentTextPart(
                                type="text", text="Describe"
                            ),
                            ChatCompletionMessageContentImagePart(
                                type="image_url",
                                image_url=ChatCompletionMessageContentImageURL(
                                    url="http://example.com/img.jpg"
                                ),
                            ),
                        ],
                    )
                ]
            )
            conv = generate_chat_conv(request, tmp_name)
            msg = conv.messages[0][1]
            img_pos = msg.find("<image>")
            txt_pos = msg.find("Describe")
            self.assertGreater(txt_pos, img_pos)
        finally:
            del chat_templates[tmp_name]

    def test_deepseek_vl2_modality_supplement(self):
        """Test deepseek-vl2 modality supplement (add_token_as_needed path)."""
        request = self._make_request(
            [
                ChatCompletionMessageUserParam(
                    role="user",
                    content=[
                        ChatCompletionMessageContentTextPart(
                            type="text", text="Describe both"
                        ),
                        ChatCompletionMessageContentImagePart(
                            type="image_url",
                            image_url=ChatCompletionMessageContentImageURL(
                                url="http://example.com/img1.jpg"
                            ),
                        ),
                        ChatCompletionMessageContentImagePart(
                            type="image_url",
                            image_url=ChatCompletionMessageContentImageURL(
                                url="http://example.com/img2.jpg"
                            ),
                        ),
                    ],
                )
            ]
        )
        conv = generate_chat_conv(request, "deepseek-vl2")
        self.assertEqual(len(conv.image_data), 2)
        msg = conv.messages[0][1]
        self.assertIn("Describe both", msg)

    def test_unknown_role_raises(self):
        """Test that an unknown message role raises ValueError."""
        request = self._make_request(
            [ChatCompletionMessageUserParam(role="user", content="Hi")]
        )
        from types import SimpleNamespace
        request.__dict__["messages"] = [SimpleNamespace(role="alien", content="Hi")]
        with self.assertRaises(ValueError):
            generate_chat_conv(request, "chatml")

    def test_user_message_many_images_adds_newline(self):
        """Test that >16 images triggers newline before text content."""
        image_parts = [
            ChatCompletionMessageContentImagePart(
                type="image_url",
                image_url=ChatCompletionMessageContentImageURL(
                    url=f"http://example.com/img{i}.jpg"
                ),
            )
            for i in range(17)
        ]
        content = [
            ChatCompletionMessageContentTextPart(type="text", text="Describe all")
        ] + image_parts
        request = self._make_request(
            [ChatCompletionMessageUserParam(role="user", content=content)]
        )
        conv = generate_chat_conv(request, "chatml")
        self.assertEqual(len(conv.image_data), 17)
        self.assertIn("\nDescribe all", conv.messages[0][1])


class TestGenerateChatConvEdgeCases(CustomTestCase):
    """Edge cases for generate_chat_conv."""

    def _make_request(self, messages):
        return ChatCompletionRequest(messages=messages, model="test")

    def test_qwen2_vl_name_image_token_at_prefix_false(self):
        """Test qwen2-vl does NOT put image token at prefix."""
        request = self._make_request(
            [
                ChatCompletionMessageUserParam(
                    role="user",
                    content=[
                        ChatCompletionMessageContentTextPart(
                            type="text", text="Describe"
                        ),
                        ChatCompletionMessageContentImagePart(
                            type="image_url",
                            image_url=ChatCompletionMessageContentImageURL(
                                url="http://example.com/img.jpg"
                            ),
                        ),
                    ],
                )
            ]
        )
        conv = generate_chat_conv(request, "qwen2-vl")
        msg = conv.messages[0][1]
        # qwen2-vl should NOT have image_token at prefix (different from general case)
        self.assertNotIn("<| глаза", msg)

    def test_system_message_invalid_list_raises(self):
        """Test that system message with non-text content raises ValueError."""
        request = self._make_request(
            [
                ChatCompletionMessageGenericParam(
                    role="system",
                    content=[
                        ChatCompletionMessageContentImagePart(
                            type="image_url",
                            image_url=ChatCompletionMessageContentImageURL(
                                url="http://example.com/img.jpg"
                            ),
                        )
                    ],
                ),
                ChatCompletionMessageUserParam(role="user", content="Hi"),
            ]
        )
        with self.assertRaises(ValueError):
            generate_chat_conv(request, "chatml")

    def test_system_message_as_list_single_text(self):
        """Test system message given as a single-element list of text parts."""
        request = self._make_request(
            [
                ChatCompletionMessageGenericParam(
                    role="system",
                    content=[
                        ChatCompletionMessageContentTextPart(
                            type="text", text="System text"
                        )
                    ],
                ),
                ChatCompletionMessageUserParam(role="user", content="Hi"),
            ]
        )
        conv = generate_chat_conv(request, "chatml")
        self.assertEqual(conv.system_message, "System text")
        self.assertIn("Hi", conv.messages[0][1])

    def test_system_message_list_not_single_text_raises(self):
        """Test system message with multiple items raises ValueError."""
        request = self._make_request(
            [
                ChatCompletionMessageGenericParam(
                    role="system",
                    content=[
                        ChatCompletionMessageContentTextPart(
                            type="text", text="System text"
                        ),
                        ChatCompletionMessageContentImagePart(
                            type="image_url",
                            image_url=ChatCompletionMessageContentImageURL(
                                url="http://example.com/img.jpg"
                            ),
                        ),
                    ],
                ),
                ChatCompletionMessageUserParam(role="user", content="Hi"),
            ]
        )
        with self.assertRaises(ValueError):
            generate_chat_conv(request, "chatml")


class TestGenerateChatConvConvCopy(CustomTestCase):
    """Test that generate_chat_conv properly copies the template."""

    def _make_request(self, messages):
        return ChatCompletionRequest(messages=messages, model="test")

    def test_template_not_modified_in_place(self):
        """Test that generate_chat_conv does not modify the registered template."""
        original_messages = list(chat_templates["chatml"].messages)
        request = self._make_request(
            [ChatCompletionMessageUserParam(role="user", content="Hello")]
        )
        conv = generate_chat_conv(request, "chatml")
        conv.append_message("user", "Another message")
        # Original template should be unchanged
        self.assertEqual(list(chat_templates["chatml"].messages), original_messages)

    def test_conv_has_blank_assistant_placeholder(self):
        """Test that the returned conv ends with a blank assistant message."""
        request = self._make_request(
            [ChatCompletionMessageUserParam(role="user", content="Hello")]
        )
        conv = generate_chat_conv(request, "chatml")
        self.assertTrue(conv.messages[-1][1] is None)
        self.assertEqual(conv.messages[-1][0], conv.roles[1])


class TestGetConvTemplateByModelPathEdgeCases(CustomTestCase):
    """Edge case tests for get_conv_template_by_model_path."""

    def test_nonexistent_model_returns_none(self):
        """Test that an unknown model path returns None."""
        result = get_conv_template_by_model_path("totally-unknown-model-xyz-12345")
        self.assertIsNone(result)

    def test_partial_path_matches(self):
        """Test that partial model path matches the correct template."""
        result = get_conv_template_by_model_path("OpenGVLab/InternVL2-8B-A3")
        self.assertEqual(result, "internvl-2-5")


if __name__ == "__main__":
    unittest.main()