"""End-to-end check of Nemotron-3.5-Embed support against a Hugging Face reference.

Builds a tiny random checkpoint with the same config shape as
nvidia/Nemotron-3.5-Embed-1B (bare ``Mistral3Model``: Pixtral vision tower,
bidirectional Ministral3 text model, ``pooling=avg``) and a PixtralProcessor
with the checkpoint's prompt layout, so the test does not download weights.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from PIL import Image
from tokenizers import AddedToken, Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import (
    Ministral3Config,
    Mistral3Config,
    Mistral3Model,
    PixtralImageProcessor,
    PixtralProcessor,
    PixtralVisionConfig,
    PreTrainedTokenizerFast,
)

from sglang import Engine
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=180, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=120, suite="stage-b-test-1-gpu-small-amd")

IMG, PAD, IMG_BREAK, IMG_END = 10, 11, 12, 13
# Same layout as the checkpoint's template: the system prefix, then every
# image, then the text.
CHAT_TEMPLATE = (
    "{%- for message in messages -%}"
    "{%- if message['role'] == 'system' -%}{{ message['content'] ~ ' ' }}"
    "{%- else -%}"
    "{%- set ns = namespace(out=false) -%}"
    "{%- for block in message['content'] if block['type'] == 'image' -%}"
    "{%- if ns.out %} {% endif -%}[IMG]{%- set ns.out = true -%}"
    "{%- endfor -%}"
    "{%- for block in message['content'] if block['type'] == 'text' -%}"
    "{%- if ns.out %} {% endif -%}{{ block['text'] }}{%- set ns.out = true -%}"
    "{%- endfor -%}"
    "{%- endif -%}"
    "{%- endfor -%}"
)
# longest_edge 56 resizes the 70x98 and 126x84 images; 28x42 stays as is.
IMAGES = {
    "small": Image.new("RGB", (28, 42), (200, 20, 40)),
    "tall": Image.new("RGB", (70, 98), (20, 160, 90)),
    "wide": Image.new("RGB", (126, 84), (60, 60, 220)),
}
# (prefix, image names, text)
INPUTS = [
    ("t3", [], "t4 t5 t6"),
    ("t7", ["tall"], ""),
    ("t7", ["wide"], "t8 t9"),
    ("t7", ["small", "tall"], "t5"),
]
LONGEST_EDGES = (56, 112)


def _messages(prefix, image_names, text, with_pixels=False):
    content = [
        {"type": "image", "image": IMAGES[name]} if with_pixels else {"type": "image"}
        for name in image_names
    ]
    if text:
        content.append({"type": "text", "text": text})
    return [
        {"role": "system", "content": prefix},
        {"role": "user", "content": content},
    ]


class TestNemotron35Embedding(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(42)
        text_config = Ministral3Config(
            vocab_size=128,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=128,
            max_position_embeddings=512,
            original_max_position_embeddings=512,
            initializer_range=0.1,
            architectures=["Ministral3Model"],
            is_causal=False,
            pooling="avg",
            tie_word_embeddings=True,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=PAD,
        )
        vision_config = PixtralVisionConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            head_dim=32,
            image_size=112,
            patch_size=14,
            # SGLang's Pixtral MLP is SiLU-gated, as in the real checkpoints.
            hidden_act="silu",
            initializer_range=0.1,
        )
        config = Mistral3Config(
            text_config=text_config,
            vision_config=vision_config,
            image_token_index=IMG,
            spatial_merge_size=2,
            vision_feature_layer=-1,
            multimodal_projector_bias=False,
            architectures=["Mistral3Model"],
            is_causal=False,
            pooling="avg",
            tie_word_embeddings=True,
        )
        config._attn_implementation = "sdpa"
        hf_model = Mistral3Model(config).eval()

        special = {"[IMG]": IMG, "<pad>": PAD, "[IMG_BREAK]": IMG_BREAK}
        special["[IMG_END]"] = IMG_END
        vocab = {f"t{i}": i for i in range(128) if i not in special.values()}
        vocab.update(special)
        backend = Tokenizer(WordLevel(vocab, unk_token="t0"))
        backend.pre_tokenizer = WhitespaceSplit()
        backend.add_special_tokens(
            [AddedToken(t, special=True) for t in ("[IMG]", "[IMG_BREAK]", "[IMG_END]")]
        )
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            unk_token="t0",
            pad_token="<pad>",
            bos_token="t1",
            eos_token="t2",
            # The real checkpoint pads to a multiple of 8 as well.
            pad_to_multiple_of=8,
        )

        cls.model_dir = tempfile.TemporaryDirectory()
        cls.references = {}
        for longest_edge in LONGEST_EDGES:
            processor = PixtralProcessor(
                image_processor=PixtralImageProcessor(
                    size={"longest_edge": longest_edge}, patch_size=14
                ),
                tokenizer=tokenizer,
                patch_size=14,
                spatial_merge_size=2,
                chat_template=CHAT_TEMPLATE,
            )
            cls.references[longest_edge] = []
            with torch.no_grad():
                for prefix, image_names, text in INPUTS:
                    inputs = processor.apply_chat_template(
                        _messages(prefix, image_names, text, with_pixels=True),
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                    )
                    hidden = hf_model(**inputs).last_hidden_state
                    cls.references[longest_edge].append(
                        F.normalize(hidden.float().mean(dim=1), dim=-1)[0]
                    )
        cls.prompts = [
            processor.apply_chat_template(_messages(*item), tokenize=False)
            for item in INPUTS
        ]
        hf_model.save_pretrained(cls.model_dir.name)
        # Saved with the last longest_edge; the engine overrides it per run.
        processor.save_pretrained(cls.model_dir.name)

    @classmethod
    def tearDownClass(cls):
        cls.model_dir.cleanup()

    def _check(self, *, dtype, engine_kwargs, assert_close):
        for longest_edge in LONGEST_EDGES:
            # No --is-embedding: the checkpoint config alone must select the model.
            engine = Engine(
                model_path=self.model_dir.name,
                dtype=dtype,
                mem_fraction_static=0.15,
                mm_process_config={"image": {"size": {"longest_edge": longest_edge}}},
                **engine_kwargs,
            )
            try:
                model_config = engine.tokenizer_manager.model_config
                self.assertEqual(
                    model_config.hf_config.architectures, ["EmbeddingNemotron35Model"]
                )
                self.assertFalse(model_config.is_generation)
                # Batches in different orders exercise the packed mean pooling.
                for order in ([0], [1], [3], [0, 1, 2, 3], [3, 2, 1, 0]):
                    with self.subTest(longest_edge=longest_edge, order=order):
                        results = engine.encode(
                            [self.prompts[i] for i in order],
                            image_data=[
                                [IMAGES[name] for name in INPUTS[i][1]] or None
                                for i in order
                            ],
                        )
                        for result, index in zip(results, order):
                            assert_close(
                                torch.tensor(result["embedding"]),
                                self.references[longest_edge][index],
                            )
            finally:
                engine.shutdown()

    def test_fp32_torch_native_matches_hf(self):
        def assert_close(actual, expected):
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)

        # Exact FP32 everywhere: the default ViT attention (triton) and cuDNN
        # convolutions use TF32 on Ampere and newer.
        env = {"SGLANG_FORCE_FUSED_OP_BACKEND": "torch", "NVIDIA_TF32_OVERRIDE": "0"}
        with patch.dict(os.environ, env):
            self._check(
                dtype="float32",
                engine_kwargs={
                    "attention_backend": "torch_native",
                    "mm_attention_backend": "sdpa",
                },
                assert_close=assert_close,
            )

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "requires a GPU with bf16 support",
    )
    def test_bf16_default_backend_matches_hf(self):
        def assert_close(actual, expected):
            cosine = F.cosine_similarity(actual, expected, dim=0).item()
            self.assertGreaterEqual(cosine, 0.999)

        self._check(dtype="bfloat16", engine_kwargs={}, assert_close=assert_close)


if __name__ == "__main__":
    unittest.main()
