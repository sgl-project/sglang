import unittest
from unittest.mock import MagicMock, patch

from sglang.srt.server_args import ServerArgs


class TestServerArgsModelHandlersMixin(unittest.TestCase):
    def setUp(self):
        self.server_args = ServerArgs(model_path="dummy/model")

    def test_gemma_family_hybrid_swa_memory(self):
        hf_config = MagicMock()
        self.server_args.disable_hybrid_swa_memory = False

        self.server_args._handle_gemma_family(hf_config, "Gemma2ForCausalLM")

        self.assertTrue(self.server_args.disable_hybrid_swa_memory)

    def test_gemma4_family_attention_backend(self):
        hf_config = MagicMock()
        self.server_args.attention_backend = None
        self.server_args.prefill_attention_backend = None
        self.server_args.decode_attention_backend = None

        self.server_args._handle_gemma4_family(hf_config, "Gemma4ForConditionalGeneration")
        self.assertEqual(self.server_args.attention_backend, "triton")

    @patch("sglang.srt.server_args_model_handlers.is_sm100_supported", return_value=True)
    def test_llama4_family_sm100_attention_backend(self, mock_is_sm100_supported):
        hf_config = MagicMock()
        self.server_args.device = "cuda"
        self.server_args.attention_backend = None

        self.server_args._handle_llama4_family(hf_config, "Llama4")

        self.assertEqual(self.server_args.attention_backend, "trtllm_mha")

    @patch("sglang.srt.server_args_model_handlers.is_sm100_supported", return_value=False)
    @patch("sglang.srt.server_args_model_handlers.is_sm90_supported", return_value=True)
    def test_llama4_family_sm90_attention_backend(
        self, mock_is_sm90_supported, mock_is_sm100_supported
    ):
        hf_config = MagicMock()
        self.server_args.device = "cuda"
        self.server_args.attention_backend = None

        self.server_args._handle_llama4_family(hf_config, "Llama4")

        self.assertEqual(self.server_args.attention_backend, "fa3")

    def test_get_model_arch_handlers_mapping(self):
        handlers = self.server_args._get_model_arch_handlers()

        self.assertIn("Gemma2ForCausalLM", handlers)
        self.assertEqual(handlers["Gemma2ForCausalLM"], self.server_args._handle_gemma_family)

        self.assertIn("MistralLarge3ForCausalLM", handlers)
        self.assertEqual(
            handlers["MistralLarge3ForCausalLM"], self.server_args._handle_mistral_large_family
        )

    def test_get_model_arch_substring_handlers_mapping(self):
        substring_handlers = self.server_args._get_model_arch_substring_handlers()

        self.assertIn("Llama4", substring_handlers)
        self.assertEqual(substring_handlers["Llama4"], self.server_args._handle_llama4_family)
        
    def test_exaone_family_attention_backend(self):
        hf_config = MagicMock()
        hf_config.sliding_window_pattern = "some_pattern"
        self.server_args.attention_backend = "triton"
        self.server_args.disable_hybrid_swa_memory = False
        
        self.server_args._handle_exaone_family(hf_config, "Exaone4ForCausalLM")
        self.assertTrue(self.server_args.disable_hybrid_swa_memory)
        # Verify it doesn't assert since it's "triton"
        self.assertEqual(self.server_args.attention_backend, "triton")

    def test_exaone_family_attention_backend_invalid(self):
        hf_config = MagicMock()
        hf_config.sliding_window_pattern = "some_pattern"
        self.server_args.attention_backend = "invalid_backend"
        
        with self.assertRaises(AssertionError):
            self.server_args._handle_exaone_family(hf_config, "Exaone4ForCausalLM")

if __name__ == "__main__":
    unittest.main()
