import unittest

from sglang.lang.chat_template import ChatTemplate, ChatTemplateStyle, get_chat_template


class TestChatTemplateLlama2(unittest.TestCase):
    def test_llama2_default_system_prompt_no_duplicate_inst(self):
        template = ChatTemplate(
            name="llama-2-chat-with-default",
            default_system_prompt="You are a helpful assistant.",
            role_prefix_and_suffix={
                "system": ("<<SYS>>\n", "\n<</SYS>>\n\n"),
                "user": ("[INST] ", " [/INST]"),
                "assistant": ("", " </s><s>"),
            },
            style=ChatTemplateStyle.LLAMA2,
        )
        messages = [
            {"role": "system", "content": None},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi!"},
        ]
        prompt = template.get_prompt(messages)
        self.assertEqual(
            prompt,
            "[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\nHello! [/INST]Hi! </s><s>",
        )

    def test_llama2_non_system_first_message_preserves_inst_prefix(self):
        template = get_chat_template("llama-2-chat")
        prefix, suffix = template.get_prefix_and_suffix(
            "user", [{"role": "user", "content": "First turn"}]
        )
        self.assertEqual(prefix, "[INST] ")
        self.assertEqual(suffix, " [/INST]")

    def test_llama2_skipped_none_system_prompt_followed_by_system(self):
        template = get_chat_template("llama-2-chat")
        messages = [
            {"role": "system", "content": None},
            {"role": "system", "content": "Custom system"},
            {"role": "user", "content": "Hello!"},
        ]
        prompt = template.get_prompt(messages)
        self.assertEqual(
            prompt,
            "[INST] <<SYS>>\nCustom system\n<</SYS>>\n\nHello! [/INST]",
        )


if __name__ == "__main__":
    unittest.main()
