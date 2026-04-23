"""Unit tests for srt/parser/conversation.py"""

import json
import os
import tempfile
import unittest

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionMessageContentAudioPart,
    ChatCompletionMessageContentAudioURL,
    ChatCompletionMessageContentImagePart,
    ChatCompletionMessageContentImageURL,
    ChatCompletionMessageContentTextPart,
    ChatCompletionMessageContentVideoPart,
    ChatCompletionMessageContentVideoURL,
    ChatCompletionMessageGenericParam,
    ChatCompletionMessageUserParam,
    ChatCompletionRequest,
)
from sglang.srt.parser.conversation import (
    Conversation,
    SeparatorStyle,
    _get_full_multimodal_text_prompt,
    chat_template_exists,
    chat_templates,
    generate_chat_conv,
    generate_embedding_convs,
    get_conv_template_by_model_path,
    get_model_type,
    register_conv_template,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="stage-a-test-cpu")


class TestConversationGetPrompt(CustomTestCase):
    def test_add_colon_single(self):
        """Test prompt generation with ADD_COLON_SINGLE style."""
        conv = Conversation(
            name="test",
            system_message="System msg",
            roles=("User", "Assistant"),
            messages=[["User", "Hello"], ["Assistant", "Hi"], ["User", None]],
            sep_style=SeparatorStyle.ADD_COLON_SINGLE,
            sep="\n",
        )
        prompt = conv.get_prompt()
        self.assertIn("System msg\n", prompt)
        self.assertIn("User: Hello\n", prompt)
        self.assertIn("Assistant: Hi\n", prompt)
        self.assertTrue(prompt.endswith("User:"))

    def test_add_colon_two(self):
        """Test prompt generation with ADD_COLON_TWO style (alternating separators)."""
        conv = Conversation(
            name="test",
            system_message="Sys",
            roles=("User", "Assistant"),
            messages=[["User", "Q"], ["Assistant", "A"], ["User", None]],
            sep_style=SeparatorStyle.ADD_COLON_TWO,
            sep="<s1>",
            sep2="<s2>",
        )
        prompt = conv.get_prompt()
        self.assertIn("User: Q<s1>", prompt)
        self.assertIn("Assistant: A<s2>", prompt)
        self.assertTrue(prompt.endswith("User:"))

    def test_chatml(self):
        """Test prompt generation with CHATML style."""
        conv = Conversation(
            name="test",
            system_message="<|im_start|>system\nYou are helpful",
            roles=("<|im_start|>user", "<|im_start|>assistant"),
            messages=[
                ["<|im_start|>user", "Hello"],
                ["<|im_start|>assistant", None],
            ],
            sep_style=SeparatorStyle.CHATML,
            sep="<|im_end|>",
        )
        prompt = conv.get_prompt()
        self.assertIn("You are helpful<|im_end|>", prompt)
        self.assertIn("<|im_start|>user\nHello<|im_end|>", prompt)
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n"))

    def test_llama3(self):
        """Test prompt generation with LLAMA3 style."""
        conv = Conversation(
            name="test",
            system_message="<|start_header_id|>system<|end_header_id|>\n\nBe helpful<|eot_id|>",
            roles=("user", "assistant"),
            messages=[["user", "Hi"], ["assistant", None]],
            sep_style=SeparatorStyle.LLAMA3,
        )
        prompt = conv.get_prompt()
        self.assertIn("Be helpful<|eot_id|>", prompt)
        self.assertIn(
            "<|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>", prompt
        )
        self.assertTrue(
            prompt.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")
        )

    def test_no_colon_single(self):
        """Test prompt generation with NO_COLON_SINGLE style."""
        conv = Conversation(
            name="test",
            system_message="",
            roles=("[USER]", "[ASST]"),
            messages=[["[USER]", "Hello"], ["[ASST]", None]],
            sep_style=SeparatorStyle.NO_COLON_SINGLE,
            sep="\n",
        )
        prompt = conv.get_prompt()
        self.assertIn("[USER]Hello\n", prompt)
        self.assertTrue(prompt.endswith("[ASST]"))

    def test_none_message_in_prompt(self):
        """Test that None message produces role-only output (no content)."""
        conv = Conversation(
            name="test",
            system_message="",
            roles=("User", "Assistant"),
            messages=[["User", "Q"], ["Assistant", None]],
            sep_style=SeparatorStyle.ADD_COLON_SINGLE,
            sep="\n",
        )
        prompt = conv.get_prompt()
        self.assertTrue(prompt.endswith("Assistant:"))

    def test_empty_system_message(self):
        """Test that empty system message produces empty prefix for LLAMA3."""
        conv = Conversation(
            name="test",
            system_message="",
            roles=("User", "Assistant"),
            messages=[["User", "Hello"], ["Assistant", None]],
            sep_style=SeparatorStyle.LLAMA3,
        )
        prompt = conv.get_prompt()
        self.assertNotIn("system", prompt.lower())

    def test_add_colon_space_single(self):
        """Test prompt generation with ADD_COLON_SPACE_SINGLE style."""
        conv = Conversation(
            name="test",
            system_message="Sys",
            roles=("User", "Bot"),
            messages=[["User", "Hi"], ["Bot", None]],
            sep_style=SeparatorStyle.ADD_COLON_SPACE_SINGLE,
            sep="\n",
        )
        prompt = conv.get_prompt()
        self.assertIn("User: Hi\n", prompt)
        # None message should end with ": " (space after colon)
        self.assertTrue(prompt.endswith("Bot: "))

    def test_add_new_line_single(self):
        """Test prompt generation with ADD_NEW_LINE_SINGLE style."""
        conv = Conversation(
            name="test",
            system_message="Sys",
            roles=("User", "Bot"),
            messages=[["User", "Hi"], ["Bot", None]],
            sep_style=SeparatorStyle.ADD_NEW_LINE_SINGLE,
            sep="\n",
        )
        prompt = conv.get_prompt()
        self.assertIn("User\nHi\n", prompt)
        self.assertTrue(prompt.endswith("Bot\n"))

    def test_no_colon_two(self):
        """Test prompt generation with NO_COLON_TWO style (alternating separators)."""
        conv = Conversation(
            name="test",
            system_message="",
            roles=("[U]", "[A]"),
            messages=[["[U]", "Q"], ["[A]", "A"], ["[U]", None]],
            sep_style=SeparatorStyle.NO_COLON_TWO,
            sep="<s1>",
            sep2="<s2>",
        )
        prompt = conv.get_prompt()
        self.assertIn("[U]Q<s1>", prompt)
        self.assertIn("[A]A<s2>", prompt)
        self.assertTrue(prompt.endswith("[U]"))

    def test_llama2_with_system(self):
        """Test LLAMA2 with system message."""
        conv = Conversation(
            name="test",
            system_message="<<SYS>>\nBe helpful\n<</SYS>>\n\n",
            system_template="[INST] {system_message}",
            roles=("[INST]", "[/INST]"),
            messages=[["[INST]", "Hi"], ["[/INST]", None]],
            sep_style=SeparatorStyle.LLAMA2,
            sep=" ",
            sep2=" </s><s>",
        )
        prompt = conv.get_prompt()
        self.assertIn("Be helpful", prompt)
        self.assertIn("Hi ", prompt)

    def test_llama2_without_system(self):
        """Test LLAMA2 without system message falls back to '[INST] ' prefix."""
        conv = Conversation(
            name="test",
            system_message="",
            roles=("[INST]", "[/INST]"),
            messages=[["[INST]", "Hi"], ["[/INST]", None]],
            sep_style=SeparatorStyle.LLAMA2,
            sep=" ",
            sep2=" </s><s>",
        )
        prompt = conv.get_prompt()
        self.assertTrue(prompt.startswith("[INST] Hi"))

    def test_llama2_multi_turn(self):
        """Test LLAMA2 with multi-turn (i>0 uses tag+sep pattern)."""
        conv = Conversation(
            name="test",
            system_message="<<SYS>>\nSys\n<</SYS>>\n\n",
            system_template="[INST] {system_message}",
            roles=("[INST]", "[/INST]"),
            messages=[
                ["[INST]", "Q1"],
                ["[/INST]", "A1"],
                ["[INST]", "Q2"],
                ["[/INST]", None],
            ],
            sep_style=SeparatorStyle.LLAMA2,
            sep=" ",
            sep2=" </s><s>",
        )
        prompt = conv.get_prompt()
        # i=0: message + " " (no tag prefix)
        self.assertIn("Q1 ", prompt)
        # i=1: tag + " " + message + sep2
        self.assertIn("[/INST] A1 </s><s>", prompt)

    def test_llama4(self):
        """Test prompt generation with LLAMA4 style."""
        conv = Conversation(
            name="test",
            system_message="Be helpful",
            system_template="{system_message}",
            roles=("user", "assistant"),
            messages=[["user", "Hello"], ["assistant", None]],
            sep_style=SeparatorStyle.LLAMA4,
        )
        prompt = conv.get_prompt()
        self.assertIn("Be helpful", prompt)
        self.assertIn("<|header_start|>user<|header_end|>", prompt)
        self.assertIn("Hello<|eot|>", prompt)

    def test_llama4_empty_system(self):
        """Test LLAMA4 with empty system message omits system prefix."""
        conv = Conversation(
            name="test",
            system_message="",
            roles=("user", "assistant"),
            messages=[["user", "Hello"], ["assistant", None]],
            sep_style=SeparatorStyle.LLAMA4,
        )
        prompt = conv.get_prompt()
        self.assertTrue(prompt.startswith("<|header_start|>user"))

    def test_chatglm3(self):
        """Test prompt generation with CHATGLM3 style."""
        conv = Conversation(
            name="test",
            system_message="<|system|>\nBe helpful",
            roles=("<|user|>", "<|assistant|>"),
            messages=[["<|user|>", "Hi"], ["<|assistant|>", None]],
            sep_style=SeparatorStyle.CHATGLM3,
        )
        prompt = conv.get_prompt()
        self.assertIn("Be helpful", prompt)
        self.assertIn("<|user|>\nHi", prompt)
        self.assertTrue(prompt.endswith("<|assistant|>"))

    def test_deepseek_chat(self):
        """Test prompt generation with DEEPSEEK_CHAT style."""
        conv = Conversation(
            name="test",
            system_message="",
            roles=("User", "Assistant"),
            messages=[["User", "Q"], ["Assistant", "A"], ["User", None]],
            sep_style=SeparatorStyle.DEEPSEEK_CHAT,
            sep="\n\n",
            sep2="<end>",
        )
        prompt = conv.get_prompt()
        self.assertIn("User: Q\n\n", prompt)
        self.assertIn("Assistant: A<end>", prompt)
        self.assertTrue(prompt.endswith("User:"))

    def test_robin(self):
        """Test prompt generation with ROBIN style."""
        conv = Conversation(
            name="test",
            system_message="Sys",
            roles=("###Human", "###Assistant"),
            messages=[["###Human", "Hi"], ["###Assistant", None]],
            sep_style=SeparatorStyle.ROBIN,
            sep="\n",
        )
        prompt = conv.get_prompt()
        self.assertIn("###Human:\nHi\n", prompt)
        self.assertTrue(prompt.endswith("###Assistant:\n"))

    def test_falcon_chat(self):
        """Test prompt generation with FALCON_CHAT style."""
        conv = Conversation(
            name="test",
            system_message="System prompt.",
            roles=("User", "Falcon"),
            messages=[["User", "Hi"], ["Falcon", None]],
            sep_style=SeparatorStyle.FALCON_CHAT,
            sep="\n",
        )
        prompt = conv.get_prompt()
        self.assertIn("System prompt.\n", prompt)
        self.assertIn("User: Hi\n", prompt)
        self.assertTrue(prompt.endswith("Falcon:"))

    def test_metamath(self):
        """Test prompt generation with METAMATH style."""
        conv = Conversation(
            name="test",
            system_message="",
            roles=("Query", "Response"),
            messages=[["Query", "2+2?"], ["Response", None]],
            sep_style=SeparatorStyle.METAMATH,
            sep="\n",
            sep2="Let's think step by step.\n",
        )
        prompt = conv.get_prompt()
        self.assertIn("Query:\n2+2?\n", prompt)
        self.assertIn("Response: Let's think step by step.\n", prompt)

    def test_mpt(self):
        """Test prompt generation with MPT style."""
        conv = Conversation(
            name="test",
            system_message="<|system|>",
            roles=("<|user|>", "<|assistant|>"),
            messages=[["<|user|>", "Hi"], ["<|assistant|>", None]],
            sep_style=SeparatorStyle.MPT,
            sep="\n",
        )
        prompt = conv.get_prompt()
        self.assertIn("<|user|>Hi\n", prompt)
        self.assertTrue(prompt.endswith("<|assistant|>"))

    def test_chatintern(self):
        """Test prompt generation with CHATINTERN style."""
        conv = Conversation(
            name="test",
            system_message="",
            roles=("HUMAN", "BOT"),
            messages=[["HUMAN", "Hi"], ["BOT", "Hello"], ["HUMAN", None]],
            sep_style=SeparatorStyle.CHATINTERN,
            sep="\n",
            sep2="</s>",
        )
        prompt = conv.get_prompt()
        self.assertIn("<s>HUMAN:Hi\n", prompt)
        self.assertIn("BOT:Hello</s>", prompt)

    def test_dolly(self):
        """Test prompt generation with DOLLY style."""
        conv = Conversation(
            name="test",
            system_message="",
            roles=("Instruction", "Response"),
            messages=[["Instruction", "Q"], ["Response", "A"], ["Instruction", None]],
            sep_style=SeparatorStyle.DOLLY,
            sep="\n\n",
            sep2="</s>",
        )
        prompt = conv.get_prompt()
        self.assertIn("Instruction:\nQ\n\n", prompt)
        self.assertIn("Response:\nA</s>", prompt)
        self.assertTrue(prompt.endswith("Instruction:\n"))

    def test_phoenix(self):
        """Test prompt generation with PHOENIX style."""
        conv = Conversation(
            name="test",
            system_message="",
            roles=("Human", "Phoenix"),
            messages=[["Human", "Hi"], ["Phoenix", None]],
            sep_style=SeparatorStyle.PHOENIX,
        )
        prompt = conv.get_prompt()
        self.assertIn("Human: <s>Hi</s>", prompt)
        self.assertTrue(prompt.endswith("Phoenix: <s>"))

    def test_deepseek_vl2(self):
        """Test prompt generation with DeepSeekVL2 style."""
        conv = Conversation(
            name="test",
            system_message="Sys",
            roles=("User", "Assistant"),
            messages=[["User", "Q"], ["Assistant", None]],
            sep_style=SeparatorStyle.DeepSeekVL2,
            sep="\n",
            sep2="<end>",
        )
        prompt = conv.get_prompt()
        self.assertIn("Sys\n", prompt)
        self.assertIn("User: Q\n", prompt)
        self.assertTrue(prompt.endswith("Assistant:"))

    def test_deepseek_vl2_empty_system(self):
        """Test DeepSeekVL2 with empty system message omits system prefix."""
        conv = Conversation(
            name="test",
            system_message="",
            roles=("User", "Assistant"),
            messages=[["User", "Q"], ["Assistant", None]],
            sep_style=SeparatorStyle.DeepSeekVL2,
            sep="\n",
            sep2="<end>",
        )
        prompt = conv.get_prompt()
        self.assertTrue(prompt.startswith("User: Q"))

    def test_gemma3(self):
        """Test prompt generation with GEMMA3 style (first message special)."""
        conv = Conversation(
            name="test",
            system_message="",
            roles=("<start>", "<model>"),
            messages=[["<start>", "Hello"], ["<model>", "Hi"], ["<start>", None]],
            sep_style=SeparatorStyle.GEMMA3,
            sep="<end>",
        )
        prompt = conv.get_prompt()
        # First message: no role prefix, just message + sep
        self.assertTrue(prompt.startswith("Hello<end>"))
        # Subsequent: role + message + sep
        self.assertIn("<model>Hi<end>", prompt)

    def test_rwkv(self):
        """Test prompt generation with RWKV style (newline replacement)."""
        conv = Conversation(
            name="test",
            system_message="",
            roles=("Bob", "Alice"),
            messages=[["Bob", "Hello\n\nWorld"], ["Alice", None]],
            sep_style=SeparatorStyle.RWKV,
        )
        prompt = conv.get_prompt()
        # RWKV replaces \n\n with \n in message
        self.assertIn("Bob: Hello\nWorld\n\n", prompt)

    def test_qwen2_vl_embed(self):
        """Test prompt generation with QWEN2_VL_EMBED style."""
        conv = Conversation(
            name="test",
            system_message="Sys",
            roles=("user", "assistant"),
            messages=[["user", "Hi"], ["assistant", None]],
            sep_style=SeparatorStyle.QWEN2_VL_EMBED,
            sep="\n",
            stop_str="<|endoftext|>",
        )
        prompt = conv.get_prompt()
        self.assertIn("user\nHi\n", prompt)
        self.assertTrue(prompt.endswith("<|endoftext|>"))

    def test_chatglm(self):
        """Test prompt generation with CHATGLM style (round numbering)."""
        conv = Conversation(
            name="chatglm",
            system_message="",
            roles=("问", "答"),
            messages=[["问", "Hello"], ["答", "Hi"], ["问", None]],
            sep_style=SeparatorStyle.CHATGLM,
            sep="\n",
        )
        prompt = conv.get_prompt()
        self.assertIn("[Round 0]\n", prompt)
        self.assertIn("问：Hello\n", prompt)
        self.assertIn("答：Hi\n", prompt)
        self.assertTrue(prompt.endswith("问："))

    def test_chatglm2_round_offset(self):
        """Test CHATGLM style with chatglm2 name (round starts at 1 instead of 0)."""
        conv = Conversation(
            name="chatglm2",
            system_message="",
            roles=("问", "答"),
            messages=[["问", "Hello"], ["答", None]],
            sep_style=SeparatorStyle.CHATGLM,
            sep="\n",
        )
        prompt = conv.get_prompt()
        self.assertIn("[Round 1]\n", prompt)

    def test_chatglm_with_system(self):
        """Test CHATGLM with non-empty system message."""
        conv = Conversation(
            name="chatglm",
            system_message="You are helpful",
            roles=("问", "答"),
            messages=[["问", "Hi"], ["答", None]],
            sep_style=SeparatorStyle.CHATGLM,
            sep="\n",
        )
        prompt = conv.get_prompt()
        self.assertTrue(prompt.startswith("You are helpful\n"))

    def test_qwen2_audio(self):
        """Test QWEN2_AUDIO style with audio token counter replacement."""
        conv = Conversation(
            name="test",
            system_message="",
            roles=("user", "assistant"),
            messages=[
                ["user", "Listen: <audio>{idx}</audio> and <audio>{idx}</audio>"],
                ["assistant", None],
            ],
            sep_style=SeparatorStyle.QWEN2_AUDIO,
            sep="\n",
            audio_token="<audio>{idx}</audio>",
        )
        prompt = conv.get_prompt()
        # Audio tokens should be replaced with counter: idx=1, idx=2
        self.assertIn("<audio>1</audio>", prompt)
        self.assertIn("<audio>2</audio>", prompt)
        self.assertNotIn("{idx}", prompt)

    def test_paddle_ocr(self):
        """Test prompt generation with PADDLE_OCR style."""
        conv = Conversation(
            name="test",
            system_message="",
            roles=("USER", "ASSISTANT"),
            messages=[["USER", "Describe image"], ["ASSISTANT", None]],
            sep_style=SeparatorStyle.PADDLE_OCR,
            sep="<eos>",
        )
        prompt = conv.get_prompt()
        self.assertIn("USER: Describe image", prompt)
        self.assertTrue(prompt.endswith("ASSISTANT: "))

    def test_paddle_ocr_with_image_token(self):
        """Test PADDLE_OCR strips newline after image token for USER role."""
        conv = Conversation(
            name="test",
            system_message="",
            roles=("USER", "ASSISTANT"),
            messages=[
                ["USER", "<image>\nDescribe this"],
                ["ASSISTANT", "It shows a cat"],
            ],
            sep_style=SeparatorStyle.PADDLE_OCR,
            sep="<eos>",
            image_token="<image>",
        )
        prompt = conv.get_prompt()
        # image_token + "\n" should be replaced with just image_token
        self.assertIn("USER: <image>Describe this\n", prompt)
        self.assertIn("ASSISTANT: It shows a cat<eos>", prompt)

    def test_mpt_with_tuple_message(self):
        """Test MPT style extracts first element from tuple messages."""
        conv = Conversation(
            name="test",
            system_message="<|system|>",
            roles=("<|user|>", "<|assistant|>"),
            messages=[
                ["<|user|>", ("Hello", "extra1", "extra2")],
                ["<|assistant|>", None],
            ],
            sep_style=SeparatorStyle.MPT,
            sep="\n",
        )
        prompt = conv.get_prompt()
        self.assertIn("<|user|>Hello\n", prompt)
        self.assertNotIn("extra1", prompt)

    def test_invalid_sep_style_raises(self):
        """Test that an invalid SeparatorStyle raises ValueError."""
        conv = Conversation(
            name="test",
            system_message="",
            roles=("A", "B"),
            messages=[["A", "Hi"]],
            sep_style=999,
            sep="\n",
        )
        with self.assertRaises(ValueError):
            conv.get_prompt()


class TestConversationMethods(CustomTestCase):
    def _make_conv(self):
        return Conversation(
            name="test",
            roles=("User", "Assistant"),
            messages=[],
            sep_style=SeparatorStyle.ADD_COLON_SINGLE,
            sep="\n",
        )

    def test_append_message(self):
        """Test appending messages to conversation."""
        conv = self._make_conv()
        conv.append_message("User", "Hello")
        conv.append_message("Assistant", "Hi")
        self.assertEqual(len(conv.messages), 2)
        self.assertEqual(conv.messages[0], ["User", "Hello"])

    def test_set_system_message(self):
        """Test setting the system message."""
        conv = self._make_conv()
        conv.set_system_message("Be helpful")
        self.assertEqual(conv.system_message, "Be helpful")

    def test_update_last_message(self):
        """Test updating the last message in-place."""
        conv = self._make_conv()
        conv.append_message("User", "Q")
        conv.append_message("Assistant", None)
        conv.update_last_message("Answer")
        self.assertEqual(conv.messages[-1][1], "Answer")

    def test_to_openai_api_messages_with_system(self):
        """Test conversion to OpenAI format with system message."""
        conv = self._make_conv()
        conv.system_message = "Be helpful"
        conv.append_message("User", "Hello")
        conv.append_message("Assistant", "Hi")
        result = conv.to_openai_api_messages()
        self.assertEqual(result[0], {"role": "system", "content": "Be helpful"})
        self.assertEqual(result[1], {"role": "user", "content": "Hello"})
        self.assertEqual(result[2], {"role": "assistant", "content": "Hi"})

    def test_to_openai_api_messages_without_system(self):
        """Test conversion to OpenAI format without system message."""
        conv = self._make_conv()
        conv.append_message("User", "Hello")
        result = conv.to_openai_api_messages()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["role"], "user")

    def test_to_openai_api_messages_skips_none_assistant(self):
        """Test that None assistant message is omitted from OpenAI format."""
        conv = self._make_conv()
        conv.append_message("User", "Hello")
        conv.append_message("Assistant", None)
        result = conv.to_openai_api_messages()
        self.assertEqual(len(result), 1)  # only user message

    def test_to_gradio_chatbot(self):
        """Test conversion to Gradio chatbot format (user/assistant pairs)."""
        conv = self._make_conv()
        conv.append_message("User", "Q1")
        conv.append_message("Assistant", "A1")
        conv.append_message("User", "Q2")
        conv.append_message("Assistant", "A2")
        result = conv.to_gradio_chatbot()
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], ["Q1", "A1"])
        self.assertEqual(result[1], ["Q2", "A2"])

    def test_to_gradio_chatbot_pending_response(self):
        """Test Gradio format with pending assistant response (None)."""
        conv = self._make_conv()
        conv.append_message("User", "Q1")
        conv.append_message("Assistant", None)
        result = conv.to_gradio_chatbot()
        self.assertEqual(result, [["Q1", None]])

    def test_append_image(self):
        """Test appending image data to conversation."""
        conv = self._make_conv()
        conv.image_data = []
        conv.append_image("http://example.com/img.jpg", "auto")
        self.assertEqual(len(conv.image_data), 1)
        self.assertEqual(conv.image_data[0].url, "http://example.com/img.jpg")
        self.assertEqual(conv.image_data[0].detail, "auto")

    def test_append_video(self):
        """Test appending video data to conversation."""
        conv = self._make_conv()
        conv.video_data = []
        conv.append_video("http://example.com/vid.mp4")
        self.assertEqual(len(conv.video_data), 1)
        self.assertEqual(conv.video_data[0], "http://example.com/vid.mp4")

    def test_append_audio(self):
        """Test appending audio data to conversation."""
        conv = self._make_conv()
        conv.audio_data = []
        conv.append_audio("http://example.com/audio.wav")
        self.assertEqual(len(conv.audio_data), 1)
        self.assertEqual(conv.audio_data[0], "http://example.com/audio.wav")

    def test_copy_is_independent(self):
        """Test that copy() creates an independent conversation."""
        conv = self._make_conv()
        conv.append_message("User", "Hello")
        copied = conv.copy()
        copied.append_message("Assistant", "Hi")
        self.assertEqual(len(conv.messages), 1)
        self.assertEqual(len(copied.messages), 2)

    def test_dict_serialization(self):
        """Test dict() returns expected keys."""
        conv = self._make_conv()
        conv.append_message("User", "Hello")
        d = conv.dict()
        self.assertEqual(d["template_name"], "test")
        self.assertIn("messages", d)
        self.assertIn("roles", d)


class TestTemplateRegistry(CustomTestCase):
    def test_builtin_templates_exist(self):
        """Test that common built-in templates are registered."""
        self.assertTrue(chat_template_exists("chatml"))
        self.assertTrue(chat_template_exists("llama-2"))

    def test_unregistered_template_not_found(self):
        """Test that non-existent template returns False."""
        self.assertFalse(chat_template_exists("_nonexistent_template_xyz"))

    def test_register_and_lookup(self):
        """Test registering and looking up a custom template."""
        t = Conversation(
            name="_test_conv_template",
            roles=("A", "B"),
            messages=[],
            sep_style=SeparatorStyle.ADD_COLON_SINGLE,
            sep="\n",
        )
        register_conv_template(t)
        self.assertTrue(chat_template_exists("_test_conv_template"))
        # Cleanup
        del chat_templates["_test_conv_template"]

    def test_register_duplicate_raises(self):
        """Test that registering a duplicate name without override raises."""
        with self.assertRaises(AssertionError):
            register_conv_template(
                Conversation(
                    name="chatml",
                    roles=("A", "B"),
                    messages=[],
                    sep_style=SeparatorStyle.CHATML,
                    sep="",
                )
            )

    def test_get_conv_template_by_model_path_returns_none_for_unknown(self):
        """Test that unknown model path returns None."""
        result = get_conv_template_by_model_path("totally-unknown-model-xyz")
        self.assertIsNone(result)

    def test_get_conv_template_by_model_path_vicuna(self):
        """Test that vicuna model path is matched correctly."""
        result = get_conv_template_by_model_path("lmsys/vicuna-7b-v1.5")
        self.assertEqual(result, "vicuna_v1.1")

    def test_get_conv_template_by_model_path_internvl(self):
        """Test that internvl model path is matched correctly."""
        result = get_conv_template_by_model_path("OpenGVLab/InternVL2-8B")
        self.assertEqual(result, "internvl-2-5")

    def test_get_conv_template_by_model_path_deepseek_vl2(self):
        """Test that deepseek-vl2 model path is matched correctly."""
        result = get_conv_template_by_model_path("deepseek-ai/deepseek-vl2")
        self.assertEqual(result, "deepseek-vl2")

    def test_get_conv_template_by_model_path_whisper(self):
        """Test that whisper model path is matched correctly."""
        result = get_conv_template_by_model_path("openai/whisper-large-v3")
        self.assertEqual(result, "whisper")

    def test_get_conv_template_by_model_path_janus(self):
        """Test that janus model path is matched correctly."""
        result = get_conv_template_by_model_path("deepseek-ai/Janus-Pro-7B")
        self.assertEqual(result, "janus-pro")

    def test_get_conv_template_by_model_path_phi4_mm(self):
        """Test that phi-4-multimodal model path is matched correctly."""
        result = get_conv_template_by_model_path("microsoft/phi-4-multimodal")
        self.assertEqual(result, "phi-4-mm")

    def test_get_conv_template_by_model_path_llava_next(self):
        """Test that llava-next-video-34b model path returns chatml-llava."""
        result = get_conv_template_by_model_path("llava-hf/llava-next-video-34b")
        self.assertEqual(result, "chatml-llava")

    def test_get_conv_template_by_model_path_paddle_ocr(self):
        """Test that paddleocr model path is matched correctly."""
        result = get_conv_template_by_model_path("PaddleOCR/PaddleOCR-2.9")
        self.assertEqual(result, "paddle-ocr")

    def test_get_conv_template_by_model_path_deepseek_ocr(self):
        """Test that deepseek-ocr model path is matched correctly."""
        result = get_conv_template_by_model_path("deepseek-ai/deepseek-ocr-base")
        self.assertEqual(result, "deepseek-ocr")

    def test_get_conv_template_by_model_path_points(self):
        """Test that points model path is matched correctly."""
        result = get_conv_template_by_model_path("WePOINTS/points-v1.5")
        self.assertEqual(result, "points-v15-chat")

    def test_get_conv_template_by_model_path_minicpm_v(self):
        """Test that minicpm-v model path returns minicpmv."""
        result = get_conv_template_by_model_path("openbmb/MiniCPM-V-2_6")
        self.assertEqual(result, "minicpmv")

    def test_get_conv_template_by_model_path_minicpm_o(self):
        """Test that minicpm-o model path returns minicpmo."""
        result = get_conv_template_by_model_path("openbmb/MiniCPM-o-2_6")
        self.assertEqual(result, "minicpmo")


class TestGenerateEmbeddingConvs(CustomTestCase):
    def test_text_only(self):
        """Test generating embedding conversations with text only."""
        convs = generate_embedding_convs(
            texts=["Hello world"],
            images=[None],
            videos=[None],
            template_name="chatml",
        )
        self.assertEqual(len(convs), 1)
        self.assertEqual(len(convs[0].messages), 2)
        self.assertIn("Hello world", convs[0].messages[0][1])
        self.assertIsNone(convs[0].messages[1][1])  # assistant placeholder

    def test_with_image(self):
        """Test generating embedding conversations with image."""
        convs = generate_embedding_convs(
            texts=["Describe"],
            images=["http://example.com/img.jpg"],
            videos=[None],
            template_name="chatml",
        )
        self.assertEqual(len(convs), 1)
        msg = convs[0].messages[0][1]
        self.assertIn("<image>", msg)
        self.assertIn("Describe", msg)

    def test_with_video(self):
        """Test generating embedding conversations with video."""
        convs = generate_embedding_convs(
            texts=["Describe"],
            images=[None],
            videos=["http://example.com/vid.mp4"],
            template_name="chatml",
        )
        self.assertEqual(len(convs), 1)
        msg = convs[0].messages[0][1]
        self.assertIn("<video>", msg)
        self.assertIn("Describe", msg)

    def test_with_image_and_video(self):
        """Test embedding conv with both image and video."""
        convs = generate_embedding_convs(
            texts=["Desc"],
            images=["http://example.com/img.jpg"],
            videos=["http://example.com/vid.mp4"],
            template_name="chatml",
        )
        msg = convs[0].messages[0][1]
        self.assertIn("<image>", msg)
        self.assertIn("<video>", msg)

    def test_none_text(self):
        """Test embedding conv with None text (only media)."""
        convs = generate_embedding_convs(
            texts=[None],
            images=["http://example.com/img.jpg"],
            videos=[None],
            template_name="chatml",
        )
        msg = convs[0].messages[0][1]
        self.assertIn("<image>", msg)
        # None text should not produce "None" string
        self.assertNotIn("None", msg)

    def test_multiple_items(self):
        """Test generating multiple embedding conversations."""
        convs = generate_embedding_convs(
            texts=["text1", "text2"],
            images=[None, None],
            videos=[None, None],
            template_name="chatml",
        )
        self.assertEqual(len(convs), 2)


class TestGetFullMultimodalTextPrompt(CustomTestCase):
    def test_adds_missing_image_tokens(self):
        """Test adding missing image tokens to prompt."""
        result = _get_full_multimodal_text_prompt("<image>", 3, "Describe this.")
        self.assertEqual(result.count("<image>"), 3)
        self.assertIn("Describe this.", result)

    def test_preserves_existing_tokens(self):
        """Test that existing tokens in prompt are preserved."""
        result = _get_full_multimodal_text_prompt(
            "<image>", 2, "<image> What about this?"
        )
        self.assertEqual(result.count("<image>"), 2)

    def test_all_tokens_present_no_addition(self):
        """Test no addition when all tokens are already present."""
        result = _get_full_multimodal_text_prompt("<image>", 2, "<image> and <image>")
        self.assertEqual(result, "<image> and <image>")

    def test_more_tokens_than_data_raises(self):
        """Test that more placeholders than data items raises ValueError."""
        with self.assertRaises(ValueError):
            _get_full_multimodal_text_prompt("<image>", 1, "<image> <image>")

    def test_zero_count_with_no_tokens(self):
        """Test zero modality count with no tokens in prompt."""
        result = _get_full_multimodal_text_prompt("<image>", 0, "Just text")
        self.assertEqual(result, "Just text")

    def test_video_tokens(self):
        """Test adding missing video tokens."""
        result = _get_full_multimodal_text_prompt("<video>", 2, "Describe:")
        self.assertEqual(result.count("<video>"), 2)
        self.assertIn("Describe:", result)

    def test_tokens_joined_with_newline(self):
        """Test that missing tokens are joined with newlines before prompt."""
        result = _get_full_multimodal_text_prompt("<image>", 3, "text")
        # 3 images, 0 in prompt → 3 added, joined by \n, then \n before text
        lines = result.split("\n")
        self.assertEqual(lines[0], "<image>")
        self.assertEqual(lines[1], "<image>")
        self.assertEqual(lines[2], "<image>")
        self.assertEqual(lines[3], "text")


class TestGenerateChatConv(CustomTestCase):
    """Test generate_chat_conv with real Pydantic message objects."""

    def _make_request(self, messages):
        """Create a real ChatCompletionRequest with given messages."""
        return ChatCompletionRequest(messages=messages, model="test")

    def test_simple_user_message(self):
        """Test basic user string message."""
        request = self._make_request(
            [ChatCompletionMessageUserParam(role="user", content="Hello")]
        )
        conv = generate_chat_conv(request, "chatml")
        # user message + blank assistant placeholder
        self.assertEqual(len(conv.messages), 2)
        self.assertIn("Hello", conv.messages[0][1])
        self.assertIsNone(conv.messages[1][1])

    def test_system_then_user(self):
        """Test system message followed by user message."""
        request = self._make_request(
            [
                ChatCompletionMessageGenericParam(role="system", content="Be helpful"),
                ChatCompletionMessageUserParam(role="user", content="Hi"),
            ]
        )
        conv = generate_chat_conv(request, "chatml")
        self.assertEqual(conv.system_message, "Be helpful")
        self.assertIn("Hi", conv.messages[0][1])

    def test_system_message_as_list(self):
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

    def test_string_messages_raises(self):
        """Test that passing messages as a raw string raises ValueError."""
        request = self._make_request(
            [ChatCompletionMessageUserParam(role="user", content="Hi")]
        )
        # Manually override messages to be a string to trigger validation
        request.__dict__["messages"] = "not a list"
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
                        ChatCompletionMessageContentAudioPart(
                            type="audio_url",
                            audio_url=ChatCompletionMessageContentAudioURL(
                                url="http://example.com/audio.wav"
                            ),
                        ),
                    ],
                )
            ]
        )
        conv = generate_chat_conv(request, "chatml")
        self.assertEqual(len(conv.audio_data), 1)
        self.assertEqual(conv.audio_data[0], "http://example.com/audio.wav")

    def test_user_message_image_at_prefix(self):
        """Test image_token_at_prefix=True puts image token before text."""
        # Register a temporary template with image_token_at_prefix=True
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
            # Image token should be BEFORE "Describe"
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
        # deepseek-vl2 uses _get_full_multimodal_text_prompt to add image tokens
        self.assertIn("Describe both", msg)

    def test_unknown_role_raises(self):
        """Test that an unknown message role raises ValueError."""
        request = self._make_request(
            [ChatCompletionMessageUserParam(role="user", content="Hi")]
        )
        # Manually inject a message with unknown role
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
        # With >16 images, text content is prefixed with "\n"
        self.assertIn("\nDescribe all", conv.messages[0][1])


class TestGetModelType(CustomTestCase):
    def test_nonexistent_path_returns_none(self):
        """Test that a path without config.json returns None."""
        result = get_model_type("/nonexistent/path/abc123")
        self.assertIsNone(result)

    def test_valid_config_returns_model_type(self):
        """Test reading model_type from a real config.json file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"model_type": "llama", "hidden_size": 4096}
            with open(os.path.join(tmpdir, "config.json"), "w") as f:
                json.dump(config, f)
            result = get_model_type(tmpdir)
            self.assertEqual(result, "llama")

    def test_config_without_model_type_returns_none(self):
        """Test that config.json without model_type key returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"hidden_size": 4096}
            with open(os.path.join(tmpdir, "config.json"), "w") as f:
                json.dump(config, f)
            result = get_model_type(tmpdir)
            self.assertIsNone(result)

    def test_invalid_json_returns_none(self):
        """Test that malformed config.json returns None (JSONDecodeError)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "config.json"), "w") as f:
                f.write("not valid json{{{")
            result = get_model_type(tmpdir)
            self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
