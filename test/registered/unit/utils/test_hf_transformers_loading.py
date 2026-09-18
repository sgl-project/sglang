"""Fast HF loader contracts for SGLang and downstream Transformers CI.

Run with the Transformers checkout under test installed:
    python -m pytest test/registered/unit/utils/test_hf_transformers_loading.py

Use real auto-loaders with local config/tokenizer/processor files. No model
weights, Hub downloads, server, GPU, or mocked Transformers APIs are needed.
"""

import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import CLIPImageProcessor, LlavaProcessor, PreTrainedTokenizerFast

from sglang.srt.utils.hf_transformers import (
    get_config,
    get_context_length,
    get_hf_text_config,
    get_processor,
    get_rope_config,
    get_tokenizer,
)
from sglang.srt.utils.patch_tokenizer import unpatch_tokenizer
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestHFTransformersLoading(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.model_path = directory.name

    def write_config(self, config):
        Path(self.model_path, "config.json").write_text(json.dumps(config))

    def text_config(self):
        return {
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 16,
            "max_position_embeddings": 512,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0,
        }

    def write_multimodal_config(self):
        self.write_config(
            {
                "model_type": "llava",
                "architectures": ["LlavaForConditionalGeneration"],
                "image_token_index": 3,
                "text_config": self.text_config(),
                "vision_config": {
                    "model_type": "clip_vision_model",
                    "hidden_size": 32,
                    "intermediate_size": 64,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 4,
                    "image_size": 28,
                    "patch_size": 14,
                },
            }
        )

    def make_tokenizer(self):
        vocab = ["<pad>", "<bos>", "<eos>", "<image>", "<unk>", "hello", "world"]
        backend = Tokenizer(
            WordLevel({token: i for i, token in enumerate(vocab)}, unk_token="<unk>")
        )
        backend.pre_tokenizer = WhitespaceSplit()
        return PreTrainedTokenizerFast(
            tokenizer_object=backend,
            pad_token="<pad>",
            bos_token="<bos>",
            eos_token="<eos>",
            unk_token="<unk>",
            additional_special_tokens=["<image>"],
            chat_template="{% for message in messages %}{{ message['content'] }}{{ eos_token }}{% endfor %}",
        )

    def test_text_config_loads_with_context_length_and_rope(self):
        self.write_config(self.text_config())

        config = get_config(
            self.model_path, trust_remote_code=False, local_files_only=True
        )
        text = get_hf_text_config(config)

        self.assertIs(text, config)
        self.assertEqual(config.architectures, ["LlamaForCausalLM"])
        self.assertEqual(text.num_key_value_heads, 2)
        self.assertEqual(get_context_length(text), 512)
        theta, rope = get_rope_config(text)
        self.assertEqual(theta, 10000.0)
        self.assertEqual(rope["rope_type"], "default")

    def test_nested_config_override_preserves_text_config(self):
        self.write_multimodal_config()

        config = get_config(
            self.model_path,
            trust_remote_code=False,
            local_files_only=True,
            model_override_args={"text_config": {"max_position_embeddings": 1024}},
        )
        text = get_hf_text_config(config)

        self.assertIs(text, config.text_config)
        self.assertEqual(config.architectures, ["LlavaForConditionalGeneration"])
        self.assertEqual(text.model_type, "llama")
        self.assertEqual(text.hidden_size, 32)
        self.assertEqual(text.eos_token_id, 2)
        self.assertEqual(get_context_length(text), 1024)
        self.assertEqual(config.vision_config.patch_size, 14)

    def test_tokenizer_loading_preserves_batch_special_tokens_and_chat(self):
        self.write_config(self.text_config())
        self.make_tokenizer().save_pretrained(self.model_path)

        tokenizer = get_tokenizer(self.model_path, local_files_only=True)
        self.addCleanup(unpatch_tokenizer, tokenizer)

        self.assertEqual(
            tokenizer.encode("hello world", add_special_tokens=False), [5, 6]
        )
        batch = tokenizer(
            ["hello world", "hello"], padding=True, add_special_tokens=False
        )
        self.assertEqual(batch["input_ids"], [[5, 6], [5, 0]])
        self.assertEqual(batch["attention_mask"], [[1, 1], [1, 0]])
        self.assertEqual(tokenizer.decode([5, 6]), "hello world")
        self.assertEqual(tokenizer.encode("<image>", add_special_tokens=False), [3])
        self.assertEqual(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": "hello"}],
                tokenize=True,
                return_dict=False,
            ),
            [5, 2],
        )

    def test_processor_loading_preserves_image_tokens_and_backend(self):
        self.write_multimodal_config()
        processor = LlavaProcessor(
            image_processor=CLIPImageProcessor(
                size={"shortest_edge": 28},
                crop_size={"height": 28, "width": 28},
            ),
            tokenizer=self.make_tokenizer(),
            patch_size=14,
            num_additional_image_tokens=1,
            vision_feature_select_strategy="default",
        )
        processor.save_pretrained(self.model_path)
        image = Image.new("RGB", (28, 28), color=(255, 0, 0))

        for backend in ("pil", "torchvision"):
            with self.subTest(backend=backend):
                loaded = get_processor(
                    self.model_path,
                    local_files_only=True,
                    image_processor_backend=backend,
                )
                self.addCleanup(unpatch_tokenizer, loaded.tokenizer)
                self.assertEqual(loaded.image_processor.backend, backend)
                batch = loaded(text="<image> hello", images=image, return_tensors="pt")

                self.assertEqual(batch["pixel_values"].shape, (1, 3, 28, 28))
                self.assertEqual(batch["pixel_values"].device.type, "cpu")
                self.assertTrue(torch.isfinite(batch["pixel_values"]).all())
                self.assertEqual(batch["input_ids"].tolist(), [[3, 3, 3, 3, 5]])
                self.assertEqual(batch["attention_mask"].tolist(), [[1, 1, 1, 1, 1]])


if __name__ == "__main__":
    unittest.main()
