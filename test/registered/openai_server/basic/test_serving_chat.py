"""Tests for OpenAIServingChat that require GPU (ServerArgs device detection)."""

import unittest
from unittest.mock import Mock, patch

from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=8, suite="stage-b-test-1-gpu-small")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")


class TestDpskV32EncodingPath(unittest.TestCase):
    """Test DeepSeek V3.2 encoding path detection and application."""

    def test_dpsk_v32_encoding_path(self):
        from sglang.srt.managers.template_manager import TemplateManager
        from sglang.srt.server_args import PortArgs, ServerArgs

        server_args = ServerArgs(model_path="deepseek-ai/DeepSeek-V3.2")
        port_args = PortArgs.init_new(server_args)

        # Use mocks for TokenizerManager components to avoid full initialization
        with patch(
            "sglang.srt.managers.tokenizer_manager.TokenizerManager"
        ) as MockTokenizerManager:
            tokenizer_manager = MockTokenizerManager(server_args, port_args)
            tokenizer_manager.server_args = server_args
            tokenizer_manager.model_config = Mock()
            tokenizer_manager.model_config.get_default_sampling_params.return_value = (
                None
            )

            # Mock hf_config
            mock_hf_config = Mock()
            mock_hf_config.architectures = ["DeepseekV32ForCausalLM"]

            tokenizer_manager.model_config.hf_config = mock_hf_config

            # Case 1: No chat template in tokenizer -> should use dpsk encoding
            tokenizer_manager.tokenizer = Mock()
            tokenizer_manager.tokenizer.chat_template = None

            serving_chat = OpenAIServingChat(tokenizer_manager, TemplateManager())
            self.assertTrue(serving_chat.use_dpsk_v32_encoding)

            # Case 2: Chat template exists -> should NOT use dpsk encoding
            tokenizer_manager.tokenizer.chat_template = "some template"
            serving_chat = OpenAIServingChat(tokenizer_manager, TemplateManager())
            self.assertFalse(serving_chat.use_dpsk_v32_encoding)

            # Case 3: Not DeepSeek V3.2 architecture -> should NOT use dpsk encoding
            tokenizer_manager.tokenizer.chat_template = None
            mock_hf_config.architectures = ["LlamaForCausalLM"]
            serving_chat = OpenAIServingChat(tokenizer_manager, TemplateManager())
            self.assertFalse(serving_chat.use_dpsk_v32_encoding)


if __name__ == "__main__":
    unittest.main(verbosity=2)
