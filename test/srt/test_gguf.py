import unittest
from pathlib import Path

from huggingface_hub import hf_hub_download

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.server_args import ServerArgs


class TestGGUF(unittest.TestCase):
    def test_load_model(self):
        model_path = hf_hub_download(
            "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
            filename="tinyllama-1.1b-chat-v1.0.Q2_K.gguf",
        )

        server_args = ServerArgs(
            model_path=model_path,
            random_seed=42,
            disable_radix_cache=True,
            load_format="auto",
        )
        self.assertEqual(server_args.load_format, "gguf")

        model_config = ModelConfig(
            server_args.model_path,
            trust_remote_code=server_args.trust_remote_code,
            context_length=server_args.context_length,
            model_override_args=server_args.json_model_override_args,
        )
        model_runner = ModelRunner(
            model_config=model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=0,
            tp_rank=0,
            tp_size=server_args.tp_size,
            nccl_port=8000,
            server_args=server_args,
        )
        self.assertEqual(model_runner.vllm_model_config.quantization, "gguf")

        tokenizer = get_tokenizer(
            server_args.tokenizer_path,
            tokenizer_mode=server_args.tokenizer_mode,
            trust_remote_code=server_args.trust_remote_code,
        )
        self.assertIsNotNone(tokenizer.vocab_file)
        self.assertEqual(Path(tokenizer.vocab_file).suffix, ".gguf")


if __name__ == "__main__":
    unittest.main()
